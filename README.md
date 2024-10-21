# microscopcy_scripts
Collection of scripts to automate microscopy collection/processing

## QuPath
run scripts for all images in project with:
`QuPath script <full_path_to_script.groovy> -p <full_path_to_project_file.qpproj>`

## 2D registrations
- require `nighres` and additional tools
- depend on multiple iterations of forward and back registrations

### 2d_slice_registration_all_fns.py
- full registration approach, but not full resolution so that you can test this relatively quickly before pushing to full reg

### 2d_slice_registration_all_fns_fullres.py
- as above, but in full res to generate 50um images (`rescale` value is set differently)