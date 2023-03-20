import matplotlib.pyplot as plt
import skimage.io as io
from glob import glob
from os.path import join


in_dir = './'
fs = glob(join(in_dir,'*.tif'))

for f in fs:
    print(f)
    plt.figure(figsize=(15,15))
    plt.imshow(io.imread(f),cmap='Greys')
    plt.title(f)
    plt.show()
    input("close the plot window and then press enter to show the next image")
    plt.close()
