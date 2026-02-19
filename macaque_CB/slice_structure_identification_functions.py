import numpy as np
import nibabel as nb
from scipy.ndimage import binary_closing, binary_fill_holes, gaussian_filter
from skimage.morphology import disk  # or ball for 3D
from skimage.morphology import remove_small_objects

def make_filled_mask(cortex_mask, closing_radius=5):
    """
    cortex_mask: binary array where cortex voxels = 1
    """
    # close small breaks
    closed = binary_closing(cortex_mask, structure=disk(closing_radius))
    
    # fill interior
    filled = binary_fill_holes(closed)
    
    return filled

from scipy.ndimage import distance_transform_edt

def signed_distance(mask):
    """
    mask: filled binary mask (+ve inside object, -ve outside)
    """
    # distance inside object
    dist_inside = distance_transform_edt(mask)
    
    # distance outside object
    dist_outside = distance_transform_edt(~mask)
    
    # signed distance
    sdf = dist_inside - dist_outside
    
    return sdf

def compute_signed_distance_weight(img, img_smth_gauss=1, pctl_cut=90, closing_radius=5, smooth_weights_sigma=15, min_object_size=100, clip=10):
    """
    Docstring for compute_signed_distance_weight
    
    :param img: Description
    :param img_smth_gauss: Description
    :param pctl_cut: Description
    :param closing_radius: Description
    :param smooth_weights_sigma: Description
    :param min_object_size: Description
    :param clip: Clip the final weights to this value (both positive and negative), or None to no clip
                 ONLY applies when no smoothing of weights is applied (smooth_weights_sigma is None)
    """

    passed_nbimage = False
    if isinstance(img,str):
        passed_nbimage = True
        img = nb.load(img)
        affine = img.affine
        header = img.header
        d = img.get_fdata()
    elif isinstance(img,nb.Nifti1Image):
        passed_nbimage = True
        affine = img.affine
        header = img.header
        d = img.get_fdata()
    elif isinstance(img,np.ndarray):
        d = img
    else:
        raise ValueError("img must be a string path or a numpy array")  

    if img_smth_gauss is not None:
        img = gaussian_filter(d, sigma=img_smth_gauss)

    # Threshold to get binary mask
    pctl = np.percentile(d, pctl_cut)
    binary_mask = d > pctl
    
    binary_mask = remove_small_objects(binary_mask, min_size=min_object_size)

    # Fill mask to get smooth boundaries
    filled_mask = make_filled_mask(binary_mask, closing_radius=closing_radius)
    
    # Compute signed distance
    sdf = signed_distance(filled_mask)
    
    if smooth_weights_sigma is not None:
        # Convert to smooth weight (sigmoid)
        weights = 1.0 / (1.0 + np.exp(-sdf / smooth_weights_sigma))
    else:
        weights = sdf
        if clip is not None:
            weights = np.clip(weights, -clip, clip)
    
    #convert back to a nifti image if the input was a nifti image
    if passed_nbimage:
        weights = nb.Nifti1Image(weights, affine=affine, header=header)        
    
    return weights


def compute_signed_distance_weight_filled(img, img_smth_gauss=1, pctl_cut=90, closing_radius=5, 
                                          smooth_weights_sigma=15, min_object_size=100, clip=10,
                                          fill_method='convex_hull', cortical_detail_weight=0.3):
    """
    Compute signed distance weights with the entire brain interior filled.
    
    Unlike compute_signed_distance_weight which only captures the cortex/outer tissue,
    this version fills the entire brain region including interior structures.
    
    :param img: Input image (path string, nibabel image, or numpy array)
    :param img_smth_gauss: Sigma for Gaussian smoothing of input image (None to skip)
    :param pctl_cut: Percentile threshold for initial tissue detection
    :param closing_radius: Radius for morphological closing operation
    :param smooth_weights_sigma: Sigma for sigmoid smoothing of weights (None for raw SDF)
    :param min_object_size: Minimum size of objects to keep after thresholding
    :param clip: Clip the final weights to this value (both +/-), only when smooth_weights_sigma is None
    :param fill_method: Method to fill the brain interior:
        - 'convex_hull': Use convex hull of detected tissue (robust for brain slices)
        - 'flood_fill': Flood fill from image corners to detect background, then invert
    :param cortical_detail_weight: Weight (0-1) for preserving cortical ribbon detail.
        - 0.0: Pure filled brain SDF (smooth, no cortical detail)
        - 1.0: Pure cortical ribbon SDF (like original function, no interior fill)
        - 0.3-0.5: Recommended for balancing filled interior with cortical contrast
    
    :return: Signed distance weights (same type as input: nifti or array)
    """
    from skimage.morphology import convex_hull_image
    # from scipy.ndimage import label, binary_dilation
    
    passed_nbimage = False
    if isinstance(img, str):
        passed_nbimage = True
        img = nb.load(img)
        affine = img.affine
        header = img.header
        d = img.get_fdata()
    elif isinstance(img, nb.Nifti1Image):
        passed_nbimage = True
        affine = img.affine
        header = img.header
        d = img.get_fdata()
    elif isinstance(img, np.ndarray):
        d = img
    else:
        raise ValueError("img must be a string path, nibabel image, or numpy array")
    
    # Handle 3D images with singleton z dimension
    squeeze_z = False
    if len(d.shape) == 3 and d.shape[2] == 1:
        squeeze_z = True
        d = d[:, :, 0]
    
    if img_smth_gauss is not None and img_smth_gauss > 0:
        d_smooth = gaussian_filter(d, sigma=img_smth_gauss)
    else:
        d_smooth = d
    
    # Threshold to get binary mask of tissue
    pctl = np.percentile(d_smooth, pctl_cut)
    binary_mask = d_smooth > pctl
    
    # Remove small objects (noise)
    binary_mask = remove_small_objects(binary_mask, min_size=min_object_size)
    
    # Close small gaps in the tissue
    if closing_radius > 0:
        closed_mask = binary_closing(binary_mask, structure=disk(closing_radius))
    else:
        closed_mask = binary_mask
    
    # Fill the brain interior based on the chosen method
    if fill_method == 'convex_hull':
        # Convex hull captures the entire brain region
        filled_mask = convex_hull_image(closed_mask)
        
    elif fill_method == 'flood_fill':
        # Flood fill from corners to identify background
        from skimage.segmentation import flood_fill as sk_flood_fill
        
        # First try standard hole filling
        filled_mask = binary_fill_holes(closed_mask)
        
        # If that didn't work well, use flood fill from corners
        if filled_mask.sum() == closed_mask.sum():
            # Create a padded version to ensure corners are background
            padded = np.pad(closed_mask, pad_width=1, mode='constant', constant_values=0)
            # Flood fill from corner (0,0) to mark background
            background = sk_flood_fill(padded.astype(int), (0, 0), 2, connectivity=1)
            background = (background == 2)
            # Remove padding
            background = background[1:-1, 1:-1]
            # Brain is everything that's not background
            filled_mask = ~background
            
    else:
        raise ValueError(f"Unknown fill_method: {fill_method}. Use 'convex_hull' or 'flood_fill'")
    
    # Compute signed distance from filled mask (overall brain shape)
    sdf_filled = signed_distance(filled_mask)
    
    ## other approach to preserve more detail at the cortical ribbon
    # # Optionally blend with cortical ribbon detail using SPATIAL blending
    # if cortical_detail_weight > 0:
    #     # Compute SDF from the original cortical mask (with hole filling but not convex hull)
    #     cortical_mask = binary_fill_holes(closed_mask)
    #     sdf_cortical = signed_distance(cortical_mask)
        
    #     # Spatial blend: preserve cortical detail near the boundary, use filled SDF in deep interior
    #     # The idea: cortical SDF has sharp gradients at the ribbon boundary, but may have
    #     # very negative values in the interior (unfilled holes). We want to keep the sharp
    #     # boundary detail while "filling in" the interior.
        
    #     # Create a smooth transition weight based on distance from cortical boundary
    #     # Small |sdf_cortical| -> near boundary -> use more cortical detail
    #     # Large negative sdf_cortical -> deep interior hole -> use filled SDF
    #     transition_width = cortical_detail_weight * 20  # Controls transition zone (larger = wider blend)
        
    #     # Sigmoid transition: 1 near cortical boundary, 0 deep in interior
    #     # This preserves negative values outside the brain (where sdf_cortical and sdf_filled agree)
    #     # but fills in the interior holes
    #     blend_weight = 1.0 / (1.0 + np.exp(-(sdf_cortical + transition_width) / (transition_width * 0.3)))
        
    #     # Where cortical SDF is positive or near zero (at/near boundary): keep cortical detail
    #     # Where cortical SDF is very negative (interior holes): transition to filled SDF
    #     sdf = blend_weight * sdf_cortical + (1 - blend_weight) * sdf_filled
    # else:
    #     sdf = sdf_filled

    # Optionally blend with cortical ribbon detail
    if cortical_detail_weight > 0:
        # Compute SDF from the original cortical mask (with hole filling but not convex hull)
        cortical_mask = binary_fill_holes(closed_mask)
        sdf_cortical = signed_distance(cortical_mask)
        
        # Blend the two SDFs
        # cortical_detail_weight=0 -> pure filled SDF
        # cortical_detail_weight=1 -> pure cortical SDF
        sdf = (1 - cortical_detail_weight) * sdf_filled + cortical_detail_weight * sdf_cortical
    else:
        sdf = sdf_filled
    
    # sdf = gaussian_filter(sdf, sigma=1)

    if smooth_weights_sigma is not None:
        # Convert to smooth weight (sigmoid)
        weights = 1.0 / (1.0 + np.exp(-sdf / smooth_weights_sigma))
    else:
        weights = sdf
        if clip is not None:
            weights = np.clip(weights, -clip, clip)
    
    # Restore 3D shape if needed
    if squeeze_z:
        weights = weights[:, :, np.newaxis]
    
    # Convert back to a nifti image if the input was a nifti image
    if passed_nbimage:
        weights = nb.Nifti1Image(weights.astype(np.float32), affine=affine, header=header)
    
    return weights
