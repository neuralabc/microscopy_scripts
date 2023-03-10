// based on discussion here: https://forum.image.sc/t/scripted-densitymaps-and-exporting-to-image-file-with-imagej/78306/6
import qupath.lib.images.servers.PixelCalibration

// SET OUTPUT
def out_dir = buildFilePath(PROJECT_BASE_DIR, 'results_cell_counts')
mkdirs(out_dir)
// ************************************************//

def imageData = getCurrentImageData()
runPlugin('qupath.imagej.detect.cells.WatershedCellDetection', '{"detectionImageBrightfield": "Optical density sum",  "requestedPixelSizeMicrons": 0.0,  "backgroundRadiusMicrons": 0.0,  "medianRadiusMicrons": 0.0,  "sigmaMicrons": 1.5,  "minAreaMicrons": 10.0,  "maxAreaMicrons": 100.0,  "threshold": 0.23,  "maxBackground": 2.0,  "watershedPostProcess": true,  "excludeDAB": false,  "cellExpansionMicrons": 5.0,  "includeNuclei": true,  "smoothBoundaries": true,  "makeMeasurements": true}');

// // PIXEL SIZE CALCULATIONS FOR DOWNSAMPLING
// // this appears to slow processing down extremely (killed the process after >20 mins when running on small ROI)
// // define target resolution for ouptut pixels, in calibrated units (um if possible!)
// // note that using the builder.buildClassifier(imageData) approach does not allow you to set your pixel sizes directly, so comparability outside of QuPath can be an issue across multiple images  
// double requestedPixelSize = 10
// // determine the downsampling factor, once we know what the pixel sizes are for our x and y in the original image
// double pixelSize = imageData.getServer().getPixelCalibration().getAveragedPixelSize()
// double scale_factor = requestedPixelSize / pixelSize
// // ***************************************************//

// DENSITY MAPPING
// use a predicate to determine which objects are included in the density mapping
def predicate = PathObjectPredicates.filter(PathObjectFilter.DETECTIONS_ALL) //all your objects are belong to us
builder = DensityMaps.builder(predicate)

//set specific values that will determine the output map here, we are interested in simple SUM (centroid count) here, which is the default

builder.buildClassifier(imageData) // to allow pixel size to be set according to input data
//builder.radius(10) //this copies the value (essentially a setRadius), but does not go back to the params, which was only for setting up the builder
builder.type(DensityMaps.DensityMapType.SUM) // this is the default, setting anyway

//
println builder.buildParameters().getPixelSize() // returns null if automagically calc'd
println builder.buildParameters().getRadius() // initial radius is 0 (check code to see if radius==0 limits to per-pixel cell counts rather than radius-based)

fileName = buildFilePath(out_dir, 'dMap_r10.tif')

// // PIXEL SIZE CALCULATIONS FOR DOWNSAMPLING
// something likely wrong here, processing never finishes when this is applied.
// PixelCalibration cal = imageData.getServer().getPixelCalibration()
// print cal
// print cal.createScaledInstance(scale_factor,scale_factor).getPixelHeight()
// builder.pixelSize(cal.createScaledInstance(scale_factor,scale_factor))
// // ***************************************************//

println builder.buildParameters().getPixelSize()

writeDensityMapImage(imageData, builder, fileName)

//describe(cal)
//describe(builder)



