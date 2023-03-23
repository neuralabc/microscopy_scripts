
import nighres
import numpy
import math
import nibabel
from PIL import Image
from nighres.io import load_volume, save_volume
import scipy.ndimage
from scipy.signal import convolve2d
import os
from nibabel import processing
import subprocess
import shutil


# code by @pilou, using nighres

# file parameters
subject = 'zefir'
# inputdir = '/path/to/input/dir/'
# prefix = '_Image_'
# suffix = '.vsi - 20x'
# format = '.tif'
output_dir = '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/slice_reg/'


# image slices to consider
# images = ['01','02','05','06','07','09','10','11','12','14','15','16','17','19','20','21','22']
all_image_fnames = ['/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_01_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_01_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_02_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_02_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_03_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_03_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_04_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_04_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_05_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_05_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_06_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_06_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_07_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_07_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_08_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_08_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_09_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_09_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_10_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_10_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_11_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_11_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_12_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_12_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_13_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_13_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_14_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_14_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_15_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_16_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_17_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_18_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_19_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_20_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_21_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_22_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_23_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_24_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_25_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_26_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_27_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_28_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_29_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_30_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_31_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_32_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_33_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_34_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_35_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_36_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_37_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_38_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_39_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_40_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_41_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_42_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/slide_43_Image_01_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_43_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_44_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_45_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_46_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_48_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_49_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_50_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_51_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_52_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_53_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_54_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_55_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_56_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_57_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_58_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_59_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_60_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_61_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_62_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_63_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_64_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_65_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_66_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_67_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_68_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_69_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_70_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_71_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_72_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_73_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_74_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_75_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_76_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_77_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_78_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/slide_80_Image_01_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_79_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_80_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_81_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_82_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_83_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_84_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_85_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_86_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_87_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_88_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_89_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_90_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_91_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_92_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_93_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/slide_96_Image_01_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_94_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_95_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_96_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP1/_Image_97_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_01_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_02_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_03_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_04_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_05_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_06_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_07_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_08_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_09_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_10_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_11_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_12_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_13_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_14_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_15_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_16_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_17_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_18_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_19_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_20_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_21_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_22_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_23_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_24_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_25_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_26_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_27_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_28_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_29_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_30_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_31_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_32_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_33_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_34_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_35_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_36_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_37_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_38_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_39_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_40_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_41_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_42_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_43_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_44_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_45_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_46_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_47_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_48_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_49_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_50_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_51_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_52_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_53_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_54_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_55_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_56_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_57_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_58_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_59_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_60_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_61_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_61_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_62_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_62_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_63_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_63_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_64_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_64_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_65_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_65_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_66_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_66_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_67_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_67_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_68_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_68_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_69_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_69_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_70_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_70_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_71_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_71_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_72_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_72_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_73_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_73_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_74_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_74_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_75_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_75_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/slide_176_Image_01_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/slide_176_Image_01_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_76_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_76_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_77_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_77_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_78_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_78_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_79_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_79_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_80_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_80_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_81_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_81_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_82_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_83_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_83_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_84_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_84_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_85_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_85_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_86_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_86_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_87_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_87_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_88_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_88_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_89_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_89_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_90_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_90_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_91_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_91_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_92_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_92_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_93_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_93_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_94_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_94_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_95_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_95_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_96_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_96_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_97_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_97_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_98_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_98_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_99_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP2/_Image_99_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_01_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_01_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_02_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_02_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_03_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_03_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_04_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_04_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_05_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_05_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_06_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_06_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_07_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_07_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_08_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_08_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_09_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_09_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_10_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_10_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_11_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_11_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_12_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_12_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_13_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_13_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_14_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_14_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_15_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_15_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_16_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_16_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_17_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_17_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_18_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_18_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_19_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_19_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_20_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_20_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_21_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_22_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_22_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_23_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_23_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_24_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_24_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_25_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_25_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_26_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_26_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_27_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_27_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_28_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_28_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_29_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_29_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_30_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_30_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_31_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_31_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_32_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_32_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_33_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_33_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_34_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_34_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_35_-_20x_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_36_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_36_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_37_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_37_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_38_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_38_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_39_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_39_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_39_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_40_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_40_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_40_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_41_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_41_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_41_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_42_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_42_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_42_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_43_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_43_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_43_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_44_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_44_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_44_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_45_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_45_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_45_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_46_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_46_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_46_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_47_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_47_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_47_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_48_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_48_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_48_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_49_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_49_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_49_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_50_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_50_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_50_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_51_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_51_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_51_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_52_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_52_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_52_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_53_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_53_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_53_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_54_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_54_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_54_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_55_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_55_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_55_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_56_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_56_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_56_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_57_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_57_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_57_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_58_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_58_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_59_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_59_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_59_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_60_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_60_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_61_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_61_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_61_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_61_-_20x_04_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_62_-_20x_01_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_62_-_20x_02_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_62_-_20x_03_cellCount_29_downsample_10p002um_pix.tif', '/data/data_drive/Macaque_CB/processing/results_from_cell_counts/TP3/_Image_62_-_20x_04_cellCount_29_downsample_10p002um_pix.tif']
all_image_names = [os.path.basename(image).split('.')[0] for image in all_image_fnames] #remove the .tif extension to comply with formatting below
rescale=10

if not os.path.exists(output_dir):
     os.makedirs(output_dir)

# 0. Convert to nifti
for idx,img_orig in enumerate(all_image_fnames):
    img = os.path.basename(img_orig).split('.')[0] 
    output = output_dir+subject+'_'+str(idx).zfill(2)+'_'+img+'.nii.gz'
    
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
            
            ## REMOVED, TODO: test if continues to work as expected            
            # # use a for loop :(
            # slice_crop = numpy.zeros((math.ceil(image.shape[0]/rescale),math.ceil(image.shape[1]/rescale),rescale*rescale))
            # for dx in range(rescale):
            #     for dy in range(rescale):
            #         slice_crop[:,:,dx+dy*rescale] = slice_li[dx:rescale*math.ceil(image.shape[0]/rescale):rescale,dy:rescale*math.ceil(image.shape[1]/rescale):rescale]
            
            # # save the median as base contrast (mean used here)
            # slice_img = numpy.mean(slice_crop,axis=2)
            
            ## alternative using 2d convolution to preserve cell counts (meaning is still the same here)
            kernel = numpy.ones((rescale,rescale)) #2d convolution kernel, all 1s
            slice_img = convolve2d(image,kernel,mode='full')[::rescale,::rescale] #can divide by rescale if we want the mean, otherwise sum is good (total cell count)

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
    nifti = output_dir+subject+'_'+str(idx).zfill(2)+'_'+img+'.nii.gz'
    shape = nighres.io.load_volume(nifti).header.get_data_shape()
    
    if shape[0]*shape[1]>size:
        size = shape[0]*shape[1]
        largest = idx
        
template = output_dir+subject+'_'+str(largest).zfill(2)+'_'+os.path.basename(all_image_fnames[largest]).split('.')[0]+'.nii.gz'    

print(f"\tUsing the following image as the template for size: {template}")

# initial step to bring all images into the same space of our 2d template
for idx,img in enumerate(all_image_fnames):
    img = os.path.basename(img).split('.')[0]
    nifti = output_dir+subject+'_'+str(idx).zfill(2)+'_'+img+'.nii.gz'

    sources = [nifti]
    targets = [template]
        
    output = output_dir+subject+'_'+str(idx).zfill(2)+'_'+img+'_coreg0nl.nii.gz'
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

img_stack = output_dir+subject+'_coreg0nl_stack.nii.gz'
template = output_dir+subject+'_coreg0nl_template.nii.gz'
if (os.path.isfile(template)):
    print('1. Template: done')
else:
    stack=[]
    for idx,img in enumerate(all_image_fnames):
        img = os.path.basename(img).split('.')[0]
        reg = output_dir+subject+'_'+str(idx).zfill(2)+'_'+img+'_coreg0nl_ants-def0.nii.gz'
        print(reg)
        stack.append(nighres.io.load_volume(reg).get_fdata())
        
        header = nibabel.Nifti1Header()
        header.set_data_shape(img.shape)
        
        nifti = nibabel.Nifti1Image(img,affine=None,header=header)
        nifti.update_header()
        save_volume(img_stack,nifti)

        img = numpy.stack(stack,axis=-1)
        img = numpy.mean(img,axis=2)
        header = nibabel.Nifti1Header()
        header.set_data_shape(img.shape)
        
        nifti = nibabel.Nifti1Image(img,affine=None,header=header)
        save_volume(template,nifti)
    print('1. Template: done - {}'.format(template))


# 2. Co-register based on previous slices
for idx,img in enumerate(all_image_fnames):
        img = os.path.basename(img).split('.')[0]
        # current image
        nifti = output_dir+subject+'_'+str(idx).zfill(2)+'_'+img+'_coreg0nl_ants-def0.nii.gz'
        sources = [nifti]
        targets = [template]
        # previous image
        if idx>0:
            prev1 = output_dir+subject+'_'+str(idx-1).zfill(2)+'_'+all_image_names[idx-1]+'_coreg1nl_ants-def0.nii.gz'
            sources.append(nifti)
            targets.append(prev1)
        
        # second previous image
        if idx>1:
            prev2 = output_dir+subject+'_'+str(idx-2).zfill(2)+'_'+all_image_names[idx-2]+'_coreg1nl_ants-def0.nii.gz'
            sources.append(nifti)
            targets.append(prev2)
        
        # third previous image
        if idx>2:
            prev3 = output_dir+subject+'_'+str(idx-3).zfill(2)+'_'+all_image_names[idx-3]+'_coreg1nl_ants-def0.nii.gz'
            sources.append(nifti)
            targets.append(prev3)
        
        output = output_dir+subject+'_'+str(idx).zfill(2)+'_'+img+'_coreg1nl.nii.gz'
        coreg1nl = nighres.registration.embedded_antspy_2d_multi(source_images=sources, 
                        target_images=targets,
                        run_rigid=True,
                        rigid_iterations=1000,
                        run_affine=False,
                        run_syn=True,
                        coarse_iterations=2000,
                        medium_iterations=1000, fine_iterations=200,
					    scaling_factor=64,
                        cost_function='MutualInformation',
                        interpolation='Linear',
                        regularization='High',
                        convergence=1e-6,
                        mask_zero=False,
                        ignore_affine=True, ignore_orient=True, ignore_res=True,
                        save_data=True, overwrite=False,
                        file_name=output)

stack = []
img_stack = output_dir+subject+'_coreg1nl_stack.nii.gz'
template = output_dir+subject+'_coreg1nl_template.nii.gz'
if (os.path.isfile(img_stack)):
        print('4. Stacking: done')
else:
    for idx,img in enumerate(all_image_fnames):
        img = os.path.basename(img).split('.')[0]
        reg = output_dir+subject+'_'+str(idx).zfill(2)+'_'+img+'_coreg1nl_ants-def0.nii.gz'
        stack.append(nighres.io.load_volume(reg).get_fdata())
    
    img = numpy.stack(stack,axis=-1)
    header = nibabel.Nifti1Header()
    header.set_data_shape(img.shape)
    
    nifti = nibabel.Nifti1Image(img,affine=None,header=header)
    save_volume(img_stack,nifti)

    img = numpy.mean(img,axis=2)
        
    nifti = nibabel.Nifti1Image(img,affine=None,header=header)
    save_volume(template,nifti)
    print('4. Stacking: done - {}'.format(template))


# 3. Co-register based on previous slices, this time in reverse
for idx,img in reversed(list(enumerate(all_image_fnames))):
        img = os.path.basename(img).split('.')[0]
        # current image
        nifti = output_dir+subject+'_'+str(idx).zfill(2)+'_'+img+'_coreg0nl_ants-def0.nii.gz' #change to 0 here after conversation w/ @pilou
     
        sources = [nifti]
        targets = [template]
        # next image
        if idx<len(all_image_fnames)-1:
            next1 = output_dir+subject+'_'+str(idx+1).zfill(2)+'_'+all_image_names[idx+1]+'_coreg2nl_ants-def0.nii.gz'
            sources.append(nifti)
            targets.append(next1)
        
        # second next image
        if idx<len(all_image_fnames)-2:
            next2 = output_dir+subject+'_'+str(idx+2).zfill(2)+'_'+all_image_names[idx+2]+'_coreg2nl_ants-def0.nii.gz'
            sources.append(nifti)
            targets.append(next2)
        
        # third next image
        if idx<len(all_image_fnames)-3:
            next3 = output_dir+subject+'_'+str(idx+3).zfill(2)+'_'+all_image_names[idx+3]+'_coreg2nl_ants-def0.nii.gz'
            sources.append(nifti)
            targets.append(next3)
        
        output = output_dir+subject+'_'+str(idx).zfill(2)+'_'+img+'_coreg2nl.nii.gz'
        coreg1nl = nighres.registration.embedded_antspy_2d_multi(source_images=sources, 
                        target_images=targets,
                        run_rigid=True,
                        rigid_iterations=1000,
                        run_affine=False,
                        run_syn=True,
                        coarse_iterations=2000,
                        medium_iterations=1000, fine_iterations=200,
					    scaling_factor=64,
                        cost_function='MutualInformation',
                        interpolation='Linear',
                        regularization='High',
                        convergence=1e-6,
                        mask_zero=False,
                        ignore_affine=True, ignore_orient=True, ignore_res=True,
                        save_data=True, overwrite=False,
                        file_name=output)

stack = []
img_stack = output_dir+subject+'_coreg2nl_stack.nii.gz'
template = output_dir+subject+'_coreg2nl_template.nii.gz'
if (os.path.isfile(img_stack)):
        print('4. Stacking: done')
else:
    for idx,img in enumerate(all_image_fnames):
        img = os.path.basename(img).split('.')[0]
        reg = output_dir+subject+'_'+str(idx).zfill(2)+'_'+img+'_coreg2nl_ants-def0.nii.gz'
        stack.append(nighres.io.load_volume(reg).get_fdata())
    
    img = numpy.stack(stack,axis=-1)
    header = nibabel.Nifti1Header()
    header.set_data_shape(img.shape)
    
    nifti = nibabel.Nifti1Image(img,affine=None,header=header)
    save_volume(img_stack,nifti)

    img = numpy.mean(img,axis=2)
        
    nifti = nibabel.Nifti1Image(img,affine=None,header=header)
    save_volume(template,nifti)
    print('4. Stacking: done - {}'.format(template))



# 4.select the highest MI result for output
for idx,img in enumerate(all_image_fnames):
    img = os.path.basename(img).split('.')[0]

    template = output_dir+subject+'_coreg0nl_template.nii.gz'
    
    output = output_dir+subject+'_'+str(idx).zfill(2)+'_'+img+'_coreg12nl_ants-def0.nii.gz'
    if (not os.path.isfile(output)):
        slice1 = output_dir+subject+'_'+str(idx).zfill(2)+'_'+img+'_coreg1nl_ants-def0.nii.gz'
        slice2 = output_dir+subject+'_'+str(idx).zfill(2)+'_'+img+'_coreg2nl_ants-def0.nii.gz'
    
        curr1 = nighres.io.load_volume(slice1).get_fdata()
        curr2 = nighres.io.load_volume(slice2).get_fdata()
        curr = nighres.io.load_volume(template).get_fdata()
        
        p1,v1 = numpy.histogram(curr1.flatten(), bins=100, density=True)
        p2,v2 = numpy.histogram(curr2.flatten(), bins=100, density=True)
        pc,vc = numpy.histogram(curr.flatten(), bins=100, density=True)
        
        p1c,v1,vc = numpy.histogram2d(curr1.flatten(), curr.flatten(), bins=100, density=True)
        p2c,v2,vc = numpy.histogram2d(curr2.flatten(), curr.flatten(), bins=100, density=True)
    
        p1pc = numpy.outer(p1,pc)
        p2pc = numpy.outer(p2,pc)
             
        mi1c = numpy.sum(p1c*numpy.log(p1c/(p1pc),where=(p1c*p1pc>0)))
        mi2c = numpy.sum(p2c*numpy.log(p2c/(p2pc),where=(p2c*p2pc>0)))
    
        print("MI: "+str(mi1c)+", "+str(mi2c))
        
        # copy the best result
        mapping= output_dir+subject+'_'+str(idx).zfill(2)+'_'+img+'_coreg12nl_ants-map.nii.gz'
        inverse= output_dir+subject+'_'+str(idx).zfill(2)+'_'+img+'_coreg12nl_ants-invmap.nii.gz'
        if (mi1c>mi2c): 
            mapping1= output_dir+subject+'_'+str(idx).zfill(2)+'_'+img+'_coreg1nl_ants-map.nii.gz'
            inverse1= output_dir+subject+'_'+str(idx).zfill(2)+'_'+img+'_coreg1nl_ants-invmap.nii.gz'
            shutil.copyfile(mapping1, mapping)
            shutil.copyfile(inverse1, inverse)
            shutil.copyfile(slice1, output)
        else:
            mapping2= output_dir+subject+'_'+str(idx).zfill(2)+'_'+img+'_coreg2nl_ants-map.nii.gz'
            inverse2= output_dir+subject+'_'+str(idx).zfill(2)+'_'+img+'_coreg2nl_ants-invmap.nii.gz'
            shutil.copyfile(mapping2, mapping)
            shutil.copyfile(inverse2, inverse)
            shutil.copyfile(slice2, output)

stack = []
img_stack = output_dir+subject+'_coreg12nl_stack.nii.gz'
template = output_dir+subject+'_coreg12nl_template.nii.gz'
if (os.path.isfile(img_stack)):
        print('4. Stacking: done')
else:
    for idx,img in enumerate(all_image_fnames):
        img = os.path.basename(img).split('.')[0]
        reg = output_dir+subject+'_'+str(idx).zfill(2)+'_'+img+'_coreg12nl_ants-def0.nii.gz'
        stack.append(nighres.io.load_volume(reg).get_fdata())
    
    img = numpy.stack(stack,axis=-1)
    header = nibabel.Nifti1Header()
    header.set_data_shape(img.shape)
    
    nifti = nibabel.Nifti1Image(img,affine=None,header=header)
    save_volume(img_stack,nifti)

    img = numpy.mean(img,axis=2)
        
    nifti = nibabel.Nifti1Image(img,affine=None,header=header)
    save_volume(template,nifti)
    print('4. Stacking: done - {}'.format(template))
