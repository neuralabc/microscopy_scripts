# microscopy_scripts
Collection of scripts to automate microscopy collection/processing

## QuPath
run scripts for all images in project with:
`QuPath script <full_path_to_script.groovy> -p <full_path_to_project_file.qpproj>`

## 2D registrations
- require `nighres` and additional tools
- depend on multiple iterations of forward and back registrations

### Outputs
Concatenation Order to Bring Data into Final Space
To bring data from native slice space to the final registered space, the maps should be concatenated (composed) in the following order:

**Step	Map & File Tag	          Purpose**
1	    coreg0nl	                Initial translation to common field of view
2	    cascade_0	                Coarse cascading registration for rough alignment
3	    coreg12nl_win12_rigsyn_0	STAGE1 iter 0, best rigid+SyN wide window
4	    coreg12nl_win12_rigsyn_1	STAGE1 iter 1
5	    coreg12nl_win12_rigsyn_2	STAGE1 iter 2
6	    coreg12nl_win12_rigsyn_3	STAGE1 iter 3
7	    coreg12nl_win12_rigsyn_4	STAGE1 iter 4 (final rigsyn; this is final_rigsyn_reg_level_tag)
8	    ..._groupwise_iter3	      Groupwise iter 3 (first chained iter, groupwise_first_chained_iter=3)
9	    ..._groupwise_iter4	      Groupwise iter 4
…	…	…
15	..._groupwise_iter9	Groupwise iter 9 (final)

**Note:** The intermediate coreg1nl, coreg2nl, coreg12nl, coreg12nl_win1, and coreg12nl_win2 maps at each rigsyn iteration are intermediate/candidate maps and should note be used. Only the coreg12nl_win12 maps from each iteration represent the selected best registration and are what we chain. The coreg1nl/coreg2nl etc. files are candidates that are present on disk but superseded by the winner-take-all coreg12nl_win12 copy.
