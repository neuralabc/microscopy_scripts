
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
subject = 'test'
inputdir = '/path/to/input/dir/'
prefix = '_Image_'
suffix = '.vsi - 20x'
format = '.tif'
output_dir = 'path/to/out_dir'


# image slices to consider
images = ['01','02','05','06','07','09','10','11','12','14','15','16','17','19','20','21','22']

rescale=10

# 0. Convert to nifti
for idx,img in enumerate(images):
    
    output = output_dir+subject+'_'+img+'.nii.gz'
    if (os.path.isfile(output)):
        print('1. Extract lightness: done')
        nifti = output
    else:
        print('image '+str(prefix+img+suffix+format))
        # get the TIFF image
        slice_name = str(prefix+img+suffix+format)
        if os.path.isfile(inputdir+slice_name):
            slice_img = Image.open(inputdir+slice_name)
            
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
            print('file '+inputdir+slice_name+' not found')
            
# 1. Find largeest image as baseline
largest = -1
size= 0
for idx,img in enumerate(images):
    
    nifti = output_dir+subject+'_'+img+'.nii.gz'
    shape = nighres.io.load_volume(nifti).header.get_data_shape()
    
    if shape[0]*shape[1]>size:
        size = shape[0]*shape[1]
        largest = idx
        
template = output_dir+subject+'_'+images[largest]+'.nii.gz'    

for idx,img in enumerate(images):
    
    nifti = output_dir+subject+'_'+img+'.nii.gz'

    sources = [nifti]
    targets = [template]
        
    output = output_dir+subject+'_'+img+'_coreg0nl.nii.gz'
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

template = output_dir+subject+'_coreg0nl_template.nii.gz'
if (os.path.isfile(template)):
    print('1. Template: done')
else:
    stack=[]
    for idx,img in enumerate(images):
        reg = output_dir+subject+'_'+img+'_coreg0nl_ants-def0.nii.gz'
        stack.append(nighres.io.load_volume(reg).get_fdata())
        
        img = numpy.stack(stack,axis=-1)
        img = numpy.mean(img,axis=2)
        header = nibabel.Nifti1Header()
        header.set_data_shape(img.shape)
        
        nifti = nibabel.Nifti1Image(img,affine=None,header=header)
        save_volume(template,nifti)

# 2. Co-register based on previous slices
for idx,img in enumerate(images):
        # current image
        nifti = output_dir+subject+'_'+img+'_coreg0nl_ants-def0.nii.gz'
     
        sources = [nifti]
        targets = [template]
        # previous image
        if idx>0:
            prev1 = output_dir+subject+'_'+images[idx-1]+'_coreg1nl_ants-def0.nii.gz'
            sources.append(nifti)
            targets.append(prev1)
        
        # second previous image
        if idx>1:
            prev2 = output_dir+subject+'_'+images[idx-2]+'_coreg1nl_ants-def0.nii.gz'
            sources.append(nifti)
            targets.append(prev2)
        
        # third previous image
        if idx>2:
            prev3 = output_dir+subject+'_'+images[idx-3]+'_coreg1nl_ants-def0.nii.gz'
            sources.append(nifti)
            targets.append(prev3)
        
        output = output_dir+subject+'_'+img+'_coreg1nl.nii.gz'
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
    for idx,img in enumerate(images):
        reg = output_dir+subject+'_'+img+'_coreg1nl_ants-def0.nii.gz'
        stack.append(nighres.io.load_volume(reg).get_fdata())
    
    img = numpy.stack(stack,axis=-1)
    header = nibabel.Nifti1Header()
    header.set_data_shape(img.shape)
    
    nifti = nibabel.Nifti1Image(img,affine=None,header=header)
    save_volume(img_stack,nifti)

    img = numpy.mean(img,axis=2)
        
    nifti = nibabel.Nifti1Image(img,affine=None,header=header)
    save_volume(template,nifti)


# 3. Co-register based on previous slices, thjis time in reverse
for idx,img in reversed(list(enumerate(images))):
        # current image
        nifti = output_dir+subject+'_'+img+'_coreg1nl_ants-def0.nii.gz'
     
        sources = [nifti]
        targets = [template]
        # next image
        if idx<len(images)-1:
            next1 = output_dir+subject+'_'+images[idx+1]+'_coreg2nl_ants-def0.nii.gz'
            sources.append(nifti)
            targets.append(next1)
        
        # second next image
        if idx<len(images)-2:
            next2 = output_dir+subject+'_'+images[idx+2]+'_coreg2nl_ants-def0.nii.gz'
            sources.append(nifti)
            targets.append(next2)
        
        # third next image
        if idx<len(images)-3:
            next3 = output_dir+subject+'_'+images[idx+3]+'_coreg2nl_ants-def0.nii.gz'
            sources.append(nifti)
            targets.append(next3)
        
        output = output_dir+subject+'_'+img+'_coreg2nl.nii.gz'
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
    for idx,img in enumerate(images):
        reg = output_dir+subject+'_'+img+'_coreg2nl_ants-def0.nii.gz'
        stack.append(nighres.io.load_volume(reg).get_fdata())
    
    img = numpy.stack(stack,axis=-1)
    header = nibabel.Nifti1Header()
    header.set_data_shape(img.shape)
    
    nifti = nibabel.Nifti1Image(img,affine=None,header=header)
    save_volume(img_stack,nifti)

    img = numpy.mean(img,axis=2)
        
    nifti = nibabel.Nifti1Image(img,affine=None,header=header)
    save_volume(template,nifti)


# 4.select the highest MI result for output
for idx,img in enumerate(images):
    template = output_dir+subject+'_coreg0nl_template.nii.gz'
    
    output = output_dir+subject+'_'+img+'_coreg12nl_ants-def0.nii.gz'
    if (not os.path.isfile(output)):
        slice1 = output_dir+subject+'_'+img+'_coreg1nl_ants-def0.nii.gz'
        slice2 = output_dir+subject+'_'+img+'_coreg2nl_ants-def0.nii.gz'
    
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
        mapping= output_dir+subject+'_'+img+'_coreg12nl_ants-map.nii.gz'
        inverse= output_dir+subject+'_'+img+'_coreg12nl_ants-invmap.nii.gz'
        if (mi1c>mi2c): 
            mapping1= output_dir+subject+'_'+img+'_coreg1nl_ants-map.nii.gz'
            inverse1= output_dir+subject+'_'+img+'_coreg1nl_ants-invmap.nii.gz'
            shutil.copyfile(mapping1, mapping)
            shutil.copyfile(inverse1, inverse)
            shutil.copyfile(slice1, output)
        else:
            mapping2= output_dir+subject+'_'+img+'_coreg2nl_ants-map.nii.gz'
            inverse2= output_dir+subject+'_'+img+'_coreg2nl_ants-invmap.nii.gz'
            shutil.copyfile(mapping2, mapping)
            shutil.copyfile(inverse2, inverse)
            shutil.copyfile(slice2, output)

stack = []
img_stack = output_dir+subject+'_coreg12nl_stack.nii.gz'
template = output_dir+subject+'_coreg12nl_template.nii.gz'
if (os.path.isfile(img_stack)):
        print('4. Stacking: done')
else:
    for idx,img in enumerate(images):
        reg = output_dir+subject+'_'+img+'_coreg12nl_ants-def0.nii.gz'
        stack.append(nighres.io.load_volume(reg).get_fdata())
    
    img = numpy.stack(stack,axis=-1)
    header = nibabel.Nifti1Header()
    header.set_data_shape(img.shape)
    
    nifti = nibabel.Nifti1Image(img,affine=None,header=header)
    save_volume(img_stack,nifti)

    img = numpy.mean(img,axis=2)
        
    nifti = nibabel.Nifti1Image(img,affine=None,header=header)
    save_volume(template,nifti)
