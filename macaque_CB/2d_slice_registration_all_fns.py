
import os
import shutil
import nighres
import numpy
import nibabel
from PIL import Image
from scipy.signal import convolve2d
# import math
from nighres.io import load_volume, save_volume
# import scipy.ndimage
# from nibabel import processing
# import subprocess



# code by @pilou, using nighres; adapted, modularized, and extended by @csteele

# file parameters
subject = 'zefir'
# inputdir = '/path/to/input/dir/'
# prefix = '_Image_'
# suffix = '.vsi - 20x'
# format = '.tif'
output_dir = '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/slice_reg_perSliceTemplate_image_weights/'
zfill_num = 4
per_slice_template = True #use a median of the slice and adjacent slices to create a slice-specific template for anchoring the registration
run_syn = True #include some nonlinear reg in the initial regs as well (best to do this with hand-created slice data)
rescale=20

all_image_fnames = ['/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_01_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_01_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_02_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_02_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_03_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_03_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_04_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_04_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_05_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_05_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_06_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_06_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_07_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_07_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_08_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_08_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_09_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_09_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_10_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_10_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_11_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_11_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_12_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_12_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_13_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_13_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_14_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_14_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_15_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_16_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_17_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_18_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_19_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_20_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_21_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_22_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_23_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_24_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_25_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_26_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_27_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_28_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_29_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_30_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_31_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_32_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_33_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_34_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_35_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_36_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_37_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_38_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_39_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_40_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_41_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_42_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/slide_43_Image_01_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_43_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_44_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_45_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_46_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_48_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_49_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_50_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_51_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_52_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_53_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_54_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_55_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_56_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_57_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_58_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_59_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_60_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_61_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_62_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_63_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_64_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_65_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_66_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_67_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_68_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_69_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_70_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_71_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_72_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_73_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_74_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_75_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_76_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_77_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_78_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/slide_80_Image_01_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_79_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_80_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_81_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_82_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_83_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_84_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_85_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_86_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_87_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_88_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_89_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_90_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_91_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_92_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_93_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/slide_96_Image_01_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_94_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_95_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_96_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_97_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_01_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_02_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_03_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_04_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_05_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_06_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_07_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_08_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_09_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_10_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_11_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_12_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_13_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_14_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_15_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_16_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_17_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_18_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_19_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_20_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_21_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_22_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_23_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_24_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_25_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_26_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_27_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_28_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_29_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_30_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_31_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_32_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_33_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_34_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_35_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_36_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_37_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_38_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_39_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_40_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_41_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_42_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_43_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_44_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_45_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_46_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_47_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_48_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_49_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_50_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_51_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_52_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_53_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_54_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_55_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_56_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_57_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_58_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_59_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_60_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_61_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_61_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_62_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_62_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_63_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_63_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_64_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_64_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_65_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_65_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_66_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_66_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_67_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_67_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_68_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_68_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_69_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_69_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_70_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_70_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_71_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_71_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_72_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_72_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_73_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_73_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_74_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_74_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_75_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_75_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/slide_176_Image_01_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/slide_176_Image_01_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_76_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_76_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_77_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_77_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_78_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_78_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_79_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_79_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_80_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_80_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_81_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_81_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_82_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_83_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_83_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_84_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_84_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_85_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_85_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_86_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_86_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_87_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_87_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_88_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_88_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_89_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_89_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_90_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_90_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_91_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_91_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_92_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_92_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_93_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_93_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_94_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_94_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_95_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_95_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_96_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_96_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_97_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_97_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_98_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_98_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_99_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_99_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_01_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_01_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_02_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_02_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_03_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_03_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_04_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_04_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_05_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_05_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_06_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_06_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_07_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_07_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_08_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_08_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_09_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_09_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_10_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_10_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_11_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_11_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_12_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_12_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_13_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_13_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_14_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_14_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_15_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_15_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_16_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_16_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_17_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_17_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_18_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_18_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_19_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_19_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_20_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_20_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_21_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_22_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_22_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_23_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_23_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_24_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_24_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_25_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_25_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_26_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_26_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_27_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_27_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_28_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_28_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_29_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_29_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_30_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_30_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_31_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_31_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_32_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_32_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_33_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_33_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_34_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_34_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_35_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_36_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_36_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_37_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_37_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_38_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_38_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_39_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_39_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_39_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_40_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_40_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_40_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_41_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_41_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_41_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_42_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_42_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_42_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_43_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_43_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_43_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_44_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_44_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_44_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_45_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_45_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_45_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_46_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_46_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_46_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_47_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_47_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_47_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_48_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_48_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_48_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_49_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_49_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_49_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_50_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_50_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_50_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_51_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_51_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_51_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_52_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_52_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_52_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_53_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_53_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_53_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_54_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_54_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_54_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_55_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_55_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_55_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_56_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_56_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_56_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_57_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_57_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_57_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_58_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_58_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_59_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_59_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_59_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_60_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_60_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_61_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_61_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_61_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_61_-_20x_04_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_62_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_62_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_62_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_62_-_20x_04_cellCount_29_downsample_10p002um_pix.tif']

all_image_fnames = all_image_fnames[0:200] #for testing
all_image_names = [os.path.basename(image).split('.')[0] for image in all_image_fnames] #remove the .tif extension to comply with formatting below


if not os.path.exists(output_dir):
     os.makedirs(output_dir)

def generate_gaussian_weights(slice_order_idxs, gauss_std=3):
    """
    Generates Gaussian weights for the given slice indices, ensuring the weights sum to 1.
    This should be agnostic to the order in which the slice_order_indices are input.
    
    Parameters:
    - slice_order_idxs: list of slice indices to generate weights for.
    - gauss_std: Standard deviation of the Gaussian distribution, controls the spread of the weights.
    
    Returns:
    - out_weights: Array of Gaussian weights corresponding to the input slice indices, summing to 1.
    """
    import numpy as np
    from scipy import signal

    # Ensure slice_order_idxs is a numpy array
    slice_order_idxs = np.array(slice_order_idxs)
    
    # Insert 0 into the correct "middle" location, preserving the original order
    if 0 not in slice_order_idxs:
        mid_index = np.argmin(np.abs(slice_order_idxs))  # Insert 0 in the central position relative to other slices
        slice_order_idxs = np.insert(slice_order_idxs, mid_index + 1, 0)
    
    # Define the range of indices to cover both positive and negative slices symmetrically
    max_idx = np.max(np.abs(slice_order_idxs))
    num_vals = max_idx * 2 + 1  # Total number of values in the symmetric Gaussian
    
    # Generate a symmetric Gaussian, centered at 0
    gaussian_window = signal.windows.gaussian(num_vals, std=gauss_std)
    
    # Extract the weights corresponding to the absolute slice indices
    out_weights = np.zeros(slice_order_idxs.shape)
    
    for i, slice_idx in enumerate(slice_order_idxs):
        # Use the absolute value of the slice index to get the corresponding weight
        out_weights[i] = gaussian_window[max_idx + slice_idx]
    
    # Normalize the weights to sum to 1
    return out_weights / out_weights.sum()


def coreg_multislice(output_dir,subject,all_image_fnames,template,target_slice_offet_list=[-1,-2,-3], 
                     zfill_num=4, input_source_file_tag='coreg0nl', reg_level_tag='coreg1nl',run_syn=True,
                     run_rigid=True,previous_target_tag=None,scaling_factor=64,image_weights=True):
    ''' Co-register to slices before/after
    target_offset_list: negative values indicate slices prior to the current, positive after
    '''
    all_image_names = [os.path.basename(image_fname).split('.')[0] for image_fname in all_image_fnames]

    if image_weights:
        image_weights = generate_gaussian_weights(target_slice_offet_list) #function adds the input slice (slice 0) in the weighting
    else:
        image_weights = None
    if type(template) is list: #we have a list of templates, one for each slice
        per_slice_template = True
    else:
        per_slice_template = False
        targets = [template]
    for idx,img in enumerate(all_image_fnames):
        img = os.path.basename(img).split('.')[0]
        # current image
        previous_tail = f'_{input_source_file_tag}_ants-def0.nii.gz'
        nifti = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img+previous_tail

        sources = [nifti]
        if per_slice_template:
            targets = [template[idx]]
        if previous_target_tag is not None:
            tail = f'_{previous_target_tag}_ants-def0.nii.gz' #if we want to use the previous iteration rather than building from scratch every time (useful for windowing)
        else:
            tail = f'_{reg_level_tag}_ants-def0.nii.gz'
        # append additional images as additional targets to stabilize reg
        for slice_offset in target_slice_offet_list:
            if slice_offset < 0: #we add registration targets for the slices that came before
                if idx > numpy.abs(slice_offset + 1):        
                    prev1 = output_dir+subject+'_'+str(idx+slice_offset).zfill(zfill_num)+'_'+all_image_names[idx+slice_offset]+tail
                    sources.append(nifti)
                    targets.append(prev1)
            elif slice_offset > 0: #we add registration targets for the slices that come afterwards
                 if idx < len(all_image_fnames)-1*slice_offset:
                    prev1 = output_dir+subject+'_'+str(idx+slice_offset).zfill(zfill_num)+'_'+all_image_names[idx+slice_offset]+tail
                    sources.append(nifti)
                    targets.append(prev1)
        
        output = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img+"_"+reg_level_tag
        coreg_output = nighres.registration.embedded_antspy_2d_multi(source_images=sources, 
                        target_images=targets,
                        image_weights=image_weights,
                        run_rigid=run_rigid,
                        rigid_iterations=1000,
                        run_affine=False,
                        run_syn=run_syn,
                        coarse_iterations=2000,
                        medium_iterations=1000, fine_iterations=200,
					    scaling_factor=scaling_factor,
                        cost_function='MutualInformation',
                        interpolation='Linear',
                        regularization='High',
                        convergence=1e-6,
                        mask_zero=False,
                        ignore_affine=True, ignore_orient=True, ignore_res=True,
                        save_data=True, overwrite=False,
                        file_name=output)

def coreg_multislice_reverse(output_dir,subject,all_image_fnames,template,target_slice_offet_list=[1,2,3], 
                             zfill_num=4, input_source_file_tag='coreg1nl', reg_level_tag='coreg2nl',run_syn=True,
                             run_rigid=True, previous_target_tag=None,scaling_factor=64,image_weights=True):
    ''' Co-register to slices before/after
    target_offset_list: negative values indicate slices prior to the current, positive after
    differs in that we reverse the list (and the idx) and the offsets are the opposite sign
    TODO: can likely be combined with standard, with some more thought.
    '''
    all_image_names = [os.path.basename(image_fname).split('.')[0] for image_fname in all_image_fnames]
    
    if image_weights:
        image_weights = generate_gaussian_weights(target_slice_offet_list)
    else:
        image_weights = None
    if type(template) is list: #we have a list of templates, one for each slice
        per_slice_template = True
    else:
        per_slice_template = False
        targets = [template]
    for idx,img in reversed(list(enumerate(all_image_fnames))):
        img = os.path.basename(img).split('.')[0]
        # current image
        previous_tail = f'_{input_source_file_tag}_ants-def0.nii.gz'
        nifti = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img+previous_tail

        sources = [nifti]

        if per_slice_template:
            targets = [template[idx]]
        if previous_target_tag is not None:
            tail = f'_{previous_target_tag}_ants-def0.nii.gz' #if we want to use the previous iteration rather than building from scratch every time (useful for windowing)
        else:
            tail = f'_{reg_level_tag}_ants-def0.nii.gz'

        # append additional images as additional targets to stabilize reg
        for slice_offset in target_slice_offet_list:
            if slice_offset < 0: #we add registration targets for the slices that came before
                if idx > numpy.abs(slice_offset + 1):        
                    prev1 = output_dir+subject+'_'+str(idx+slice_offset).zfill(zfill_num)+'_'+all_image_names[idx+slice_offset]+tail
                    sources.append(nifti)
                    targets.append(prev1)
            elif slice_offset > 0: #we add registration targets for the slices that come afterwards
                 if idx < len(all_image_fnames)-1*slice_offset:
                    prev1 = output_dir+subject+'_'+str(idx+slice_offset).zfill(zfill_num)+'_'+all_image_names[idx+slice_offset]+tail
                    sources.append(nifti)
                    targets.append(prev1)
        
        output = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img+"_"+reg_level_tag
        coreg_output = nighres.registration.embedded_antspy_2d_multi(source_images=sources, 
                        target_images=targets,
                        image_weights=image_weights,
                        run_rigid=run_rigid,
                        rigid_iterations=1000,
                        run_affine=False,
                        run_syn=run_syn,
                        coarse_iterations=2000,
                        medium_iterations=1000, fine_iterations=200,
					    scaling_factor=scaling_factor,
                        cost_function='MutualInformation',
                        interpolation='Linear',
                        regularization='High',
                        convergence=1e-6,
                        mask_zero=False,
                        ignore_affine=True, ignore_orient=True, ignore_res=True,
                        save_data=True, overwrite=False,
                        file_name=output)

def generate_stack_and_template(output_dir,subject,all_image_fnames,zfill_num=4,reg_level_tag='coreg12nl',
                                per_slice_template=False,missing_idxs_to_fill=None):
    #we can also output a per_slice_template based on the median of the current and neighbouring slices
    stack = []
    stack_tail = f'_{reg_level_tag}_stack.nii.gz'
    img_stack = output_dir+subject+stack_tail
    template_tail = f'_{reg_level_tag}_template.nii.gz'
    template = output_dir+subject+template_tail

    template_list = []

    img_tail = f'_{reg_level_tag}_ants-def0.nii.gz'
    # if (os.path.isfile(img_stack)):
    if False:
        print('Stacking was already completed for this level: {}'.format(template))
    else:
        #this can only handle if there is a single missing slice between two good slices
        #and that they are not at the start or end of the stack (or it will crash)
        for idx,img_name in enumerate(all_image_fnames):
            img_name = os.path.basename(img_name).split('.')[0]
            reg = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+img_tail
            stack.append(nighres.io.load_volume(reg).get_fdata())

        img = numpy.stack(stack,axis=-1)

        missing_idxs_to_fill = None #XXX SETTING TO NONE FOR TESTING TODO
        #now we fill any missing data with the mean of the neighbouring slices
        if missing_idxs_to_fill is not None and len(missing_idxs_to_fill)>0:
            missing_idxs_to_fill.sort() #sort it
            missing_slices_interpolated = []
            missing_idxs_pre = numpy.array(missing_idxs_to_fill)-1
            missing_idxs_post = numpy.array(missing_idxs_to_fill)+1
            
            for idx,img_idx in enumerate(missing_idxs_pre):
                img_name = all_image_fnames[img_idx]
                img_name = os.path.basename(img_name).split('.')[0]
                reg = output_dir+subject+'_'+str(img_idx).zfill(zfill_num)+'_'+img_name+img_tail
                _t = nighres.io.load_volume(reg).get_fdata()
                if idx==0:
                    pre_d = numpy.zeros(_t.shape+(len(missing_idxs_to_fill),))
                pre_d[...,idx] = _t

            for idx,img_idx in enumerate(missing_idxs_post):
                img_name = all_image_fnames[img_idx]
                img_name = os.path.basename(img_name).split('.')[0]
                reg = output_dir+subject+'_'+str(img_idx).zfill(zfill_num)+'_'+img_name+img_tail
                _t = nighres.io.load_volume(reg).get_fdata()
                if idx==0:
                    post_d = numpy.zeros(_t.shape+(len(missing_idxs_to_fill),))
                post_d[...,idx] = _t
            
            missing_slices_interpolated = .5*(pre_d+post_d)

            #now we can fill the slices with the interpolated value
            for idx,missing_idx in enumerate(missing_idxs_to_fill):
                print(idx)
                print(missing_idx)
                print(numpy.shape(missing_slices_interpolated))
                print(numpy.shape(img))
                img[...,missing_idx] = missing_slices_interpolated[...,idx]

        header = nibabel.Nifti1Header()
        header.set_data_shape(img.shape)
        
        nifti = nibabel.Nifti1Image(img,affine=None,header=header)
        save_volume(img_stack,nifti)

        if per_slice_template:
            num_slices = img.shape[-1]
            for idx,img_name in enumerate(all_image_fnames):
                img_name = os.path.basename(img_name).split('.')[0]
                slice_template_fname = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+template_tail
                if idx == 0: #if at the front, take the first two only
                    slice_template = numpy.median(img[...,0:2],axis=-1)
                elif idx == num_slices-1: #if at the end, take the last two only
                    slice_template = numpy.median(img[...,-2:],axis=-1)
                else: #take one on each side and the current slice
                    start = idx-1
                    stop = idx+2
                    slice_template = numpy.median(img[...,start:stop],axis=-1)

                header.set_data_shape(slice_template.shape)
                nifti = nibabel.Nifti1Image(slice_template,affine=None,header=header)
                nifti.update_header()
                save_volume(slice_template_fname,nifti)
                template_list.append(slice_template_fname)            

        img = numpy.median(img,axis=2)
        nifti = nibabel.Nifti1Image(img,affine=None,header=header)
        save_volume(template,nifti)
        print('Stacking: done - {}'.format(template))
    if per_slice_template:
        return template_list
    else:
        return template
    
# import numpy as np
# import nighres
# from sklearn.metrics import mutual_info_score

# # Calculate Mutual Information between two images
# def calculate_MI(slice1, slice2):
#     """Calculate mutual information between two 2D slices."""
#     slice1_flat = slice1.ravel()
#     slice2_flat = slice2.ravel()
#     return mutual_info_score(slice1_flat, slice2_flat)

# # Function to check if a slice is an outlier based on MI with its neighbors
# def is_outlier(slice, neighbor_slices, threshold=0.2):
#     """Check if the slice is an outlier compared to its neighbors using MI."""
#     mi_scores = [calculate_MI(slice, neighbor) for neighbor in neighbor_slices]
#     mean_mi = np.mean(mi_scores)
#     return np.any(np.array(mi_scores) < mean_mi * (1 - threshold))  # Detect outlier if any MI is too low


def select_best_reg_by_MI(output_dir,subject,all_image_fnames,template_tag='coreg0nl',
                          zfill_num=zfill_num,reg_level_tag1='coreg1nl', reg_level_tag2='coreg2nl',reg_output_tag='coreg12nl',per_slice_template=False,
                          overwrite=True):
    '''
    Use MI to determine best registration (forwards or backwards) and select going forward
    reg_output_tag identifies the best registration outputs
    '''
    template_tail = f'_{template_tag}_template.nii.gz'
    out_tail = f'_{reg_output_tag}'
    tag1_tail = f'_{reg_level_tag1}'
    tag2_tail = f'_{reg_level_tag2}'
    
    for idx,img_name in enumerate(all_image_fnames):
        img_name = os.path.basename(img_name).split('.')[0]

        if not per_slice_template:
            template = output_dir+subject+template_tail #we use the generally defined template
        output = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+out_tail+'_ants-def0.nii.gz'
        if (not os.path.isfile(output)) or overwrite:
            if per_slice_template: #or,we use individual templates
                template = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+template_tail
            slice1 = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+tag1_tail+'_ants-def0.nii.gz'
            slice2 = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+tag2_tail+'_ants-def0.nii.gz'
        
            curr1 = nighres.io.load_volume(slice1).get_fdata()
            curr2 = nighres.io.load_volume(slice2).get_fdata()
            curr = nighres.io.load_volume(template).get_fdata()
            
            p1,v1 = numpy.histogram(curr1.flatten(), bins=100, density=True)
            p2,v2 = numpy.histogram(curr2.flatten(), bins=100, density=True)
            pc,vc = numpy.histogram(curr.flatten(), bins=100, density=True)

            # normalize histograms to 1
            p1 = p1/numpy.sum(p1)
            p2 = p2/numpy.sum(p2)
            pc = pc/numpy.sum(pc)
            
            p1c,v1,vc = numpy.histogram2d(curr1.flatten(), curr.flatten(), bins=100, density=True)
            p2c,v2,vc = numpy.histogram2d(curr2.flatten(), curr.flatten(), bins=100, density=True)
        
            # normalize joint histograms to 1
            p1c = p1c / numpy.sum(p1c)
            p2c = p2c / numpy.sum(p2c)
            
            p1pc = numpy.outer(p1,pc)
            p2pc = numpy.outer(p2,pc)
                
            mi1c = numpy.sum(p1c*numpy.log(p1c/(p1pc),where=(p1c*p1pc>0)))
            mi2c = numpy.sum(p2c*numpy.log(p2c/(p2pc),where=(p2c*p2pc>0)))
        
            print("MI: "+str(mi1c)+", "+str(mi2c))
            
            # copy the best result
            mapping= output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+out_tail+'_ants-map.nii.gz'
            inverse= output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+out_tail+'_ants-invmap.nii.gz'
            if (mi1c>mi2c): 
                mapping1= output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+tag1_tail+'_ants-map.nii.gz'
                inverse1= output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+tag1_tail+'_ants-invmap.nii.gz'
                shutil.copyfile(mapping1, mapping)
                shutil.copyfile(inverse1, inverse)
                shutil.copyfile(slice1, output)
            else:
                mapping2= output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+tag2_tail+'_ants-map.nii.gz'
                inverse2= output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img_name+tag2_tail+'_ants-invmap.nii.gz'
                shutil.copyfile(mapping2, mapping)
                shutil.copyfile(inverse2, inverse)
                shutil.copyfile(slice2, output)

            # cleanup files
            try:
                os.remove(mapping1)
                print(f"File {mapping1} deleted successfully.")
            except:
                pass

            try:
                os.remove(mapping2)
                print(f"File {mapping2} deleted successfully.")
            except:
                pass

            try:
                os.remove(inverse1)
                print(f"File {inverse1} deleted successfully.")
            except:
                pass

            try:
                os.remove(inverse2)
                print(f"File {inverse2} deleted successfully.")
            except:
                pass

            try:
                os.remove(slice1)
                print(f"File {slice1} deleted successfully.")
            except:
                pass

            try:
                os.remove(slice2)
                print(f"File {slice2} deleted successfully.")
            except:
                pass


            

# 0. Convert to nifti
for idx,img_orig in enumerate(all_image_fnames):
    img = os.path.basename(img_orig).split('.')[0] 
    output = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img+'.nii.gz'
    
    if (os.path.isfile(output)):
        print('1. Extract lightness: done')
        nifti = output
    else:
        print('image '+str(img_orig))
        # get the TIFF image
        slice_name = str(img_orig)
        if os.path.isfile(slice_name):
            slice_img = Image.open(slice_name)
            
            slice_img = numpy.array(slice_img)
            
            # crop: use various options
            image = slice_img
            slice_li = numpy.pad(image,pad_width=((0,rescale),(0,rescale)),mode='edge')
            
            ## alternative using 2d convolution to preserve cell counts (meaning is still the same here)
            kernel = numpy.ones((rescale,rescale)) #2d convolution kernel, all 1s
            slice_img = convolve2d(image,kernel,mode='full')[::rescale,::rescale] #can divide by rescale if we want the mean, otherwise sum is good (total cell count)

            #exceptions that need fixing, since rigid reg does not seem to address big flips
            if 'TP1' in img_orig: #we have files named the same within the subdirs, so we must specify specifically 
                if 'Image_11_-_20x_01_cellCount' in img_orig:
                    slice_img = numpy.flip(slice_img,axis=0) #flip x
            
            header = nibabel.Nifti1Header()
            header.set_data_shape(slice_img.shape)
            
            affine = numpy.eye(4)
            affine[0,3] = -slice_img.shape[0]/2.0
            affine[1,3] = -slice_img.shape[1]/2.0
                  
            nifti = nibabel.Nifti1Image(slice_img,affine=affine,header=header)
            save_volume(output,nifti)
                 
        else:
            print('file '+slice_name+' not found')
            
# 1. Find largeest image as baseline
largest = -1
size= 0
for idx,img in enumerate(all_image_fnames):
    img = os.path.basename(img).split('.')[0]
    nifti = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img+'.nii.gz'
    shape = nighres.io.load_volume(nifti).header.get_data_shape()
    
    if shape[0]*shape[1]>size:
        size = shape[0]*shape[1]
        largest = idx
        
template = output_dir+subject+'_'+str(largest).zfill(zfill_num)+'_'+os.path.basename(all_image_fnames[largest]).split('.')[0]+'.nii.gz'    

print(f"\tUsing the following image as the template for size: {template}")

# initial step to bring all images into the same space of our 2d template
for idx,img in enumerate(all_image_fnames):
    img = os.path.basename(img).split('.')[0]
    nifti = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img+'.nii.gz'

    sources = [nifti]
    targets = [template]
        
    output = output_dir+subject+'_'+str(idx).zfill(zfill_num)+'_'+img+'_coreg0nl.nii.gz'
    coreg1nl = nighres.registration.embedded_antspy_2d_multi(source_images=sources, 
                    target_images=targets,
                    run_rigid=False,
                    run_affine=False,
                    run_syn=False,
                    scaling_factor=64,
                    cost_function='MutualInformation',
                    interpolation='Linear',
                    regularization='High',
                    convergence=1e-6,
                    mask_zero=False,
                    ignore_affine=False, ignore_orient=False, ignore_res=False,
                    save_data=True, overwrite=False,
                    file_name=output)

template = generate_stack_and_template(output_dir,subject,all_image_fnames,zfill_num=4,reg_level_tag='coreg0nl',
                                       missing_idxs_to_fill=None)
## ****************************** Iteration 1
# in all cases, we go:
#   - forwards
#   - backwards
#   - select the best registration
#   - generate a template, using a per-slice template helps quite a bit
#   - window in front and behind, forwards
#   - window in front and behind, backwards
#   - select the best registration
#   - generate a template

# STEP 1: Rigid + Syn
num_reg_iterations = 10
template_tag = 'coreg0nl' #initial template tag, which we update with each loop

missing_idxs_to_fill = [32,59,120,160,189,228] #these are the slice indices with missing or terrible data, fill with mean of neigbours
for iter in range(num_reg_iterations): #here we always go back to the original coreg0 images, just refining our target template
    iter_tag = f"_rigsyn_{iter}"
    if (iter == 0):
        first_run_slice_template = False
    else:
        first_run_slice_template = per_slice_template

    coreg_multislice(output_dir,subject,all_image_fnames,template,target_slice_offet_list=[-1,-2,-3], 
                    zfill_num=zfill_num, input_source_file_tag='coreg0nl', reg_level_tag='coreg1nl'+iter_tag,run_syn=run_syn) 
    coreg_multislice_reverse(output_dir,subject,all_image_fnames,template,target_slice_offet_list=[1,2,3], 
                            zfill_num=zfill_num, input_source_file_tag='coreg0nl', reg_level_tag='coreg2nl'+iter_tag,run_syn=run_syn)
    
    print(iter)
    print(template_tag)
    print(first_run_slice_template)
    select_best_reg_by_MI(output_dir,subject,all_image_fnames,template_tag=template_tag,
                        zfill_num=zfill_num,reg_level_tag1='coreg1nl'+iter_tag, reg_level_tag2='coreg2nl'+iter_tag,
                        reg_output_tag='coreg12nl'+iter_tag,per_slice_template=first_run_slice_template) #use the per slice templates after the first round, if requested
    template = generate_stack_and_template(output_dir,subject,all_image_fnames,
                                        zfill_num=4,reg_level_tag='coreg12nl'+iter_tag,per_slice_template=per_slice_template,
                                        missing_idxs_to_fill=missing_idxs_to_fill)
    template_tag = 'coreg12nl'+iter_tag
    
    coreg_multislice(output_dir,subject,all_image_fnames,template,target_slice_offet_list=[-1,-2,-3,1,2,3], 
                    zfill_num=zfill_num, input_source_file_tag='coreg0nl', 
                    previous_target_tag = 'coreg12nl'+iter_tag,reg_level_tag='coreg12nl_win1'+iter_tag,run_syn=run_syn) 
    coreg_multislice_reverse(output_dir,subject,all_image_fnames,template,target_slice_offet_list=[-1,-2,-3,1,2,3], 
                    zfill_num=zfill_num, input_source_file_tag='coreg0nl', 
                    previous_target_tag = 'coreg12nl'+iter_tag,reg_level_tag='coreg12nl_win2'+iter_tag,run_syn=run_syn)
    
    select_best_reg_by_MI(output_dir,subject,all_image_fnames,template_tag=template_tag,
                        zfill_num=zfill_num,reg_level_tag1='coreg12nl_win1'+iter_tag, reg_level_tag2='coreg12nl_win2'+iter_tag,
                        reg_output_tag='coreg12nl_win12'+iter_tag,per_slice_template=per_slice_template)
    template = generate_stack_and_template(output_dir,subject,all_image_fnames,
                                        zfill_num=4,reg_level_tag='coreg12nl_win12'+iter_tag,per_slice_template=per_slice_template,
                                        missing_idxs_to_fill=missing_idxs_to_fill)
    template_tag = 'coreg12nl_win12'+iter_tag
    
final_reg_level_tag = 'coreg12nl_win12'+iter_tag
step1_iter_tag = iter_tag

# # STEP 2: Syn only
# num_syn_reg_iterations = 10
# for iter in range(num_syn_reg_iterations):
#     #for the nonlinear step, we base our registrations on the previous ones instead of going back to the original images, starting with the previous step and 
#     # then using the output from each successive step
#     iter_tag = f"{step1_iter_tag}_syn_{iter}"
#     coreg_multislice(output_dir,subject,all_image_fnames,template,target_slice_offet_list=[-1,-2,-3], 
#                     zfill_num=zfill_num, input_source_file_tag=final_reg_level_tag, reg_level_tag='coreg1nl'+iter_tag,run_syn=True,run_rigid=False) 
#     coreg_multislice_reverse(output_dir,subject,all_image_fnames,template,target_slice_offet_list=[1,2,3], 
#                             zfill_num=zfill_num, input_source_file_tag=final_reg_level_tag, reg_level_tag='coreg2nl'+iter_tag,run_syn=True,run_rigid=False) 
    
#     select_best_reg_by_MI(output_dir,subject,all_image_fnames,template_tag=template_tag,
#                         zfill_num=zfill_num,reg_level_tag1='coreg1nl'+iter_tag, reg_level_tag2='coreg2nl'+iter_tag,
#                         reg_output_tag='coreg12nl'+iter_tag,per_slice_template=per_slice_template)
#     template = generate_stack_and_template(output_dir,subject,all_image_fnames,
#                                         zfill_num=4,reg_level_tag='coreg12nl'+iter_tag,per_slice_template=per_slice_template,
#                                         missing_idxs_to_fill=missing_idxs_to_fill)
#     template_tag = 'coreg12nl'+iter_tag
#     # print(template)

#     coreg_multislice(output_dir,subject,all_image_fnames,template,target_slice_offet_list=[-1,-2,1,2], 
#                     zfill_num=zfill_num, input_source_file_tag='coreg12nl'+iter_tag, 
#                     previous_target_tag = 'coreg12nl'+iter_tag,reg_level_tag='coreg12nl_win1'+iter_tag,run_syn=True,run_rigid=False) 
#     coreg_multislice_reverse(output_dir,subject,all_image_fnames,template,target_slice_offet_list=[-1,-2,1,2], 
#                     zfill_num=zfill_num, input_source_file_tag='coreg12nl'+iter_tag, 
#                     previous_target_tag = 'coreg12nl'+iter_tag,reg_level_tag='coreg12nl_win2'+iter_tag,run_syn=True,run_rigid=False)

#     select_best_reg_by_MI(output_dir,subject,all_image_fnames,template_tag=template_tag,
#                         zfill_num=zfill_num,reg_level_tag1='coreg12nl_win1'+iter_tag, reg_level_tag2='coreg12nl_win2'+iter_tag,
#                         reg_output_tag='coreg12nl_win12'+iter_tag,per_slice_template=per_slice_template)
#     template = generate_stack_and_template(output_dir,subject,all_image_fnames,
#                                         zfill_num=4,reg_level_tag='coreg12nl_win12'+iter_tag,per_slice_template=per_slice_template,
#                                         missing_idxs_to_fill=missing_idxs_to_fill)
#     final_reg_level_tag = 'coreg12nl_win12'+iter_tag
#     template_tag = 'coreg12nl_win12'+iter_tag
