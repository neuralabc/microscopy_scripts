import qupath.lib.images.servers.PixelCalibration
import ij.IJ
import ij.ImagePlus
import ij.process.FloatProcessor
import qupath.imagej.gui.IJExtension
import qupath.lib.objects.PathObjectTools
import qupath.lib.regions.RegionRequest


import static qupath.lib.gui.scripting.QPEx.*
import qupath.imagej.tools.IJTools
import java.lang.String
import java.io.File

// partially based on discussion here: https://forum.image.sc/t/scripted-densitymaps-and-exporting-to-image-file-with-imagej/78306/6
test_fname_output_only = false
auto_dMap_computation = false //use the automatic density map calculations (faster, but cannot yet control res)
manual_pixel_cell_count = true //use float processor to compute cell count image based on user specifications (see below)
save_centroids_csv = false // output centroid x,y coordinates to a csv file

// settings for manual pixel cell counting
//    double requestedPixelSizeMicrons = 10 
double requestedDownSampleValue = 29 // 29 for 10um // 145 for 50um (@ .343um/pix) // 41 // (for the same (in testing as densityMap radius=0))// how many pixels to project into a single pixel (squared, of course); actual size is this * pixelSize
// Set the downsample directly (without using the requestedPixelSize) if you want; 1.0 indicates the full resolution
// take the floor to ensure that there is no mismatch when we project into the lower res space
//    double downsample = (requestedPixelSizeMicrons / server.getPixelCalibration().getAveragedPixelSizeMicrons())
double downsample = requestedDownSampleValue
// ************************************************//

// SET OUTPUT
def out_dir = buildFilePath(PROJECT_BASE_DIR, 'results_cell_counts')
mkdirs(out_dir)

// limit processing to the full data, do not run on labels etc
if (getProjectEntry().getImageName().contains("20x")) { 
    //explicity set the image type so that we can use the opticaldensitysum channel for classification, tehse are defaults below
    QP.setImageType('BRIGHTFIELD_H_DAB');
    QP.setColorDeconvolutionStains('{"Name" : "H-DAB default", "Stain 1" : "Hematoxylin", "Values 1" : "0.65111 0.70119 0.29049", "Stain 2" : "DAB", "Values 2" : "0.26917 0.56824 0.77759", "Background" : " 255 255 255"}');

    def imageData = getCurrentImageData()
    def imageName = getProjectEntry().getImageName().replace('.ets','').replaceAll(' ','_').replace('.vsi','')
    double pixelSize = imageData.getServer().getPixelCalibration().getAveragedPixelSize()
     
    if (test_fname_output_only) {
        println("Only outputting the header for the name of the first output file:")
        println(imageName)
        System.exit(0)
    }

    // *************************** RUN THE CELL DETECTION ***************************
    selectAnnotations();
    def annotations = getAnnotationObjects()
    runPlugin('qupath.imagej.detect.cells.WatershedCellDetection', '{"detectionImageBrightfield": "Optical density sum",  "requestedPixelSizeMicrons": 0.0,  "backgroundRadiusMicrons": 0.0,  "medianRadiusMicrons": 0.0,  "sigmaMicrons": 1.5,  "minAreaMicrons": 10.0,  "maxAreaMicrons": 100.0,  "threshold": 0.23,  "maxBackground": 2.0,  "watershedPostProcess": true,  "excludeDAB": false,  "cellExpansionMicrons": 5.0,  "includeNuclei": true,  "smoothBoundaries": true,  "makeMeasurements": true}');


    if (auto_dMap_computation){
        // DENSITY MAPPING
        // use a predicate to determine which objects are included in the density mapping
        def predicate = PathObjectPredicates.filter(PathObjectFilter.DETECTIONS_ALL) //all your objects are belong to us
        builder = DensityMaps.builder(predicate)

        //set specific values that will determine the output map here, we are interested in simple SUM (centroid count) here, which is the default
        builder.buildClassifier(imageData) // to allow pixel size to be set according to input data
        builder.type(DensityMaps.DensityMapType.SUM) // this is the default, setting anyway

        //println builder.buildParameters().getPixelSize() // returns null if automagically calc'd
        // not sure if radius is in um units or pixels, default works as expected to give counts in downsampled voxels
        def dMap_radius = builder.buildParameters().getRadius() // initial radius is 0 (check code to see if radius==0 limits to per-pixel cell counts rather than radius-based)

        def autoDownsampleValue = builder.buildServer(imageData).getPreferredDownsamples()[0] //take only the first downsampled value here, there **should** be only one

        dMap_requestedPixelSizeMicrons = pixelSize * autoDownsampleValue  
        fileName_dMap = buildFilePath(out_dir, imageName + '_densityMap_'+ (int) dMap_radius + '_rad_' + (int) autoDownsampleValue + '_downsample_'+ dMap_requestedPixelSizeMicrons.round(3).toString().replace('.','p')+'um_pix.tif')
        if (new File(fileName_dMap).isFile){
            print "\n\tFile exists, skipping: " + fileName_dMap
        }
        else {
            writeDensityMapImage(imageData, builder, fileName_dMap) // actually perform the operation to compute the density's per pixel and then save to a file
            print "\n\tDensityMap output: \n\t " + fileName_dMap + " \n"
        }

        // // PIXEL SIZE CALCULATIONS FOR DOWNSAMPLING, if you wanted to set this manually (this code does not work as-is)
        // something likely wrong here, processing never finishes when this is applied.
        // PixelCalibration cal = imageData.getServer().getPixelCalibration()
        // print cal
        // print cal.createScaledInstance(scale_factor,scale_factor).getPixelHeight()
        // builder.pixelSize(cal.createScaledInstance(scale_factor,scale_factor))
        // // ***************************************************//

        //println builder.buildParameters().getPixelSize()

        //#way to extract pixel resolution in python to check that we are in the same space
        //#XResolution and YResolution store the pix value as two numbers (1st is divisor, 2nd is numerator; e.g., 'XResolution': ((70718, 1000000),) --> 1000000/70718 (in um))
        //from PIL import Image
        //from PIL.TiffTags import TAGS
        //
        //img = Image.open('test.tif')
        //meta_dict = {TAGS[key] : img.tag[key] for key in img.tag_v2}
    }

    if (manual_pixel_cell_count) {
        println "\n\tPerforming manual cell counting"
        
        // Get the current image
    //    def imageData = getCurrentImageData()
        def server = imageData.getServer()
        def request = RegionRequest.createInstance(server, downsample)
        def imp = IJTools.convertToImagePlus(server, request).getImage()

        requestedPixelSizeMicrons = pixelSize * requestedDownSampleValue  
        
        fileName_manual = buildFilePath(out_dir, imageName + '_cellCount_' + (int) requestedDownSampleValue + '_downsample_' + requestedPixelSizeMicrons.round(3).toString().replace('.','p')+'um_pix.tif')
        
        if (new File(fileName_manual).isFile){
            print "\n\tFile exists, skipping: " + fileName_manual
        }
        else {
            // Get the objects you want to count
            // Potentially you can add filters for specific objects, e.g. to get only those with a 'Positive' classification
            def detections = getDetectionObjects()
            //def detections = detections.findAll {it.getPathClass() == getPathClass('Positive')}
            def positiveDetections = detections.findAll {it.getPathClass() == getPathClass('Positive')}

            // Create a counts image in ImageJ, where each pixel corresponds to the number of centroids at that pixel
            int width = imp.getWidth()

            int height = imp.getHeight()

            def fp = new FloatProcessor(width, height)
            for (detection in detections) {
                // Get ROI for a detection; this method gets the nucleus if we have a cell object (and the only ROI for anything else)
                def roi = PathObjectTools.getROI(detection, true)
                int x = (int) ((roi.getCentroidX() / downsample))
                int y = (int) ((roi.getCentroidY() / downsample))
                
                // correct for potential (due to rounding) cell counts outside of width and height, subtract 1! (0-based of course)
                if (x==width) {
                x=width-1
                }
                if (y==height) {
                y=height-1
                }
                fp.setf(x, y, fp.getf(x,y) + 1 as float)
            }

            // Show the images
            //IJExtension.getImageJInstance()
            //ixmp//.show()
            imp2 = new ImagePlus(imp.getTitle() + "-counts", fp)
            imp2.show() // show in imageJ

            IJ.saveAsTiff(imp2, fileName_manual)
            print "\t Cell count output: \n\t " + fileName_manual + " \n"
            println " "
        }
    }

    if (save_centroids_csv) {
        fileName_csv = buildFilePath(out_dir, imageName.replace('.*','') + '_cellCentroidXY_' + requestedPixelSizeMicrons.round(3).toString().replace('.','p')+'um_pix.csv')
        
        if (new File(fileName_csv).isFile){
            print "\n\tFile exists, skipping: " + fileName_csv
        }
        else {
            println("\tSaving centroid locations to csv output:\n\t"+fileName_csv)
            File csvFile = new File(fileName_csv);
            FileWriter fileWriter = new FileWriter(csvFile);

            // adapted from: https://github.com/qupath/qupath/wiki/Scripting-examples
            // Set this to true to use a nucleus ROI, if available
            boolean useNucleusROI = true
            
            // Start building a String with a header, these **should** contain the x and y indeces in the downsampled data (after converting to int)
            sb = new StringBuilder("y_orig\tx_orig\ty_downsample_"+ ((int) requestedDownSampleValue).toString() +"\tx_downsampled_"+((int) requestedDownSampleValue).toString()+"\n") // this is intentionally y, then x, and assignments are x then y b/c data storage diffs (?)
                
            // Loop through detections
            for (detection in getDetectionObjects()) {
                def roi = detection.getROI()
                // Use a Groovy metaClass trick to check if we can get a nucleus ROI... if we need to
                // (could also use Java's instanceof qupath.lib.objects.PathCellObject)
                if (useNucleusROI && detection.metaClass.respondsTo(detection, "getNucleusROI") && detection.getNucleusROI() != null)
                    roi = detection.getNucleusROI()
                // ROI shouldn't be null... but still feel I should check...
                if (roi == null)
                    continue
                // Get centroid
                double cx = roi.getCentroidX()
                double cy = roi.getCentroidY()
                // Append to String
                sb.append(String.format("%.2f\t%.2f\t%.2f\t%.2f\n", cx, cy, cx/downsample, cy/downsample))
            }

            for (line in sb) {
                fileWriter.write(line.toString())
                fileWriter.close()
            }
        }
    }
}
