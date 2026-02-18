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

def compute_signed_distance_weight(img, img_smth_gauss=1, pctl_cut=90, closing_radius=5, smooth_weights_sigma=15, min_object_size=100):
    """
    Docstring for compute_signed_distance_weight
    
    :param img: Description
    :param img_smth_gauss: Description
    :param pctl_cut: Description
    :param closing_radius: Description
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
    
    #convert back to a nifti image if the input was a nifti image
    if passed_nbimage:
        weights = nb.Nifti1Image(weights, affine=affine, header=header)        
    
    return weights
