"""Runtime compatibility shims for running the slice registration in the uv environment.

Import this BEFORE slice_registration_functions. It:
1. limits per-process threading (ITK/OpenMP/BLAS) so that many workers do not oversubscribe the machine
2. raises this process's OOM score so that the kernel kills our jobs first under memory pressure
   (children created by ProcessPoolExecutor inherit the value)
3. maps the ANTsPy <0.5 internal API names (ants.utils.*) onto the current ones (ants.internal.*)
4. provides the nighres functions used by embedded_antspy_smthMod: the two pure-python path helpers and
   load_volume/save_volume (nighres >=1.5 needs a Java/JCC build and is not on PyPI)

None of the library files are modified.
"""
import os

THREADS_PER_WORKER = os.environ.get('SLICEREG_THREADS_PER_WORKER', '1')
for _var in ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', 'OMP_NUM_THREADS',
             'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(_var, THREADS_PER_WORKER)

try:
    with open('/proc/self/oom_score_adj', 'w') as f:
        f.write(os.environ.get('SLICEREG_OOM_SCORE_ADJ', '1000'))
except OSError as e:
    print(f'env_compat: could not set oom_score_adj ({e})')

import ants
import ants.utils
import ants.internal

if not hasattr(ants.utils, '_int_antsProcessArguments'):
    ants.utils._int_antsProcessArguments = ants.internal.process_arguments
if not hasattr(ants.utils, 'get_lib_fn'):
    ants.utils.get_lib_fn = ants.internal.get_lib_fn


# copied from nighres/utils.py (github.com/nighres/nighres, master)
def _output_dir_4saving(output_dir=None, rootfile=None):
    if (output_dir is None or output_dir == ''):
        if rootfile is None:
            output_dir = os.getcwd()
        else:
            output_dir = os.path.dirname(rootfile)
            if (output_dir is None or output_dir == ''):
                output_dir = os.getcwd()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not (output_dir[-1] == os.path.sep):
        output_dir += os.path.sep

    if not os.access(output_dir, os.W_OK | os.X_OK):
        raise ValueError(f"Cannot write to {output_dir}, please specify a different output_dir.")

    print("\nOutputs will be saved to {0}".format(output_dir))
    return output_dir


def _fname_4saving(file_name=None, rootfile=None, suffix=None, ext=None, module='output'):
    file_ext = 'nii.gz'
    if file_name is None:
        if isinstance(rootfile, str):
            file_name = os.path.basename(rootfile)
            if suffix is None:
                suffix = 'out'
        else:
            file_name = module + '.nii.gz'

    if len(file_name) <= 1:
        raise ValueError("Empty string for file_name. Check if your inputs "
                         "exist, or try to specify the file_name "
                         "parameter for saving.")

    split_name = file_name.split('.')
    if len(split_name) == 1:
        base = split_name[0]
    else:
        file_ext = split_name.pop(-1)
        if file_ext == 'gz':
            file_ext = split_name.pop(-1) + '.gz'
        base = split_name.pop(0)
        while split_name:
            base += '.' + split_name.pop(0)
        if ext is None:
            ext = file_ext

    if ext is None:
        ext = 'nii.gz'

    if suffix is not None:
        fullname = base + '_' + suffix + '.' + ext
    else:
        fullname = base + '.' + ext

    return fullname


import embedded_antspy_smthMod as _emb
from slice_registration_functions import load_volume, save_volume  # nibabel stand-ins for nighres.io
for _name, _fn in [('_output_dir_4saving', _output_dir_4saving), ('_fname_4saving', _fname_4saving),
                   ('load_volume', load_volume), ('save_volume', save_volume)]:
    if not hasattr(_emb, _name):
        setattr(_emb, _name, _fn)
