"""Methods of one slice-registration run, written from what that run recorded: <run_dir>/QC/METHODS.md.

The text is generated for each run, so that it describes what was done (the steps that ran, their settings and
their results), not what the code does by default. Every value has a source, listed in Appendix A:
  run     run_settings.json (settings and run-time values; older runs record fewer keys)
  log     the run log(s): the antsRegistration arguments of every registration (grouped by level), the pose-graph
          lines, the scaling factor, skip / fallback / exit messages
  script  the copy of the main script in the run folder (module constants and call keywords, parsed with ast)
  code    statements about function internals, each checked against the source at the run's git commit
          (Appendix B; a statement whose check fails is marked UNVERIFIED in the text)
  files   outputs and sidecars (QC/metrics.json, QC/mri/*/qc.json, atlas_labels.json, *_counts_stack.json,
          *_coreg0nl_jitter_poses.csv, *_granular_mask.json)
References were checked against the publisher or PubMed Central records (2026-10-29).

qc_report.py calls this module and shows the result in the QC page; by hand:
  uv run python macaque_CB/methods_report.py --run-dir <run> [--check]
--check prints the code checks (Appendix B) and exits 1 if one fails.
"""
import argparse
import ast
import collections
import glob
import html
import inspect
import json
import os
import re
import subprocess
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SUBJECT = 'zefir'

REFERENCES = {
    'avants2008': 'Avants BB, Epstein CL, Grossman M, Gee JC (2008). Symmetric diffeomorphic image registration with '
                  'cross-correlation: evaluating automated labeling of elderly and neurodegenerative brain. Medical Image '
                  'Analysis 12(1):26-41. doi:10.1016/j.media.2007.06.004',
    'avants2011': 'Avants BB, Tustison NJ, Song G, Cook PA, Klein A, Gee JC (2011). A reproducible evaluation of ANTs '
                  'similarity metric performance in brain image registration. NeuroImage 54(3):2033-2044. '
                  'doi:10.1016/j.neuroimage.2010.09.025',
    'tustison2021': 'Tustison NJ, Cook PA, Holbrook AJ, Johnson HJ, Muschelli J, Devenyi GA, Duda JT, Das SR, Cullen NC, '
                    'Gillen DL, Yassa MA, Stone JR, Gee JC, Avants BB (2021). The ANTsX ecosystem for quantitative '
                    'biological and medical imaging. Scientific Reports 11:9068. doi:10.1038/s41598-021-87564-6',
    'huntenburg2018': 'Huntenburg JM, Steele CJ, Bazin P-L (2018). Nighres: processing tools for high-resolution '
                      'neuroimaging. GigaScience 7(7):giy082. doi:10.1093/gigascience/giy082',
    'maes1997': 'Maes F, Collignon A, Vandermeulen D, Marchal G, Suetens P (1997). Multimodality image registration by '
                'maximization of mutual information. IEEE Transactions on Medical Imaging 16(2):187-198. '
                'doi:10.1109/42.563664',
    'mattes2003': 'Mattes D, Haynor DR, Vesselle H, Lewellen TK, Eubank W (2003). PET-CT image registration in the chest '
                  'using free-form deformations. IEEE Transactions on Medical Imaging 22(1):120-128. '
                  'doi:10.1109/TMI.2003.809072',
    'huber1964': 'Huber PJ (1964). Robust estimation of a location parameter. The Annals of Mathematical Statistics '
                 '35(1):73-101. doi:10.1214/aoms/1177703732',
    'ashburner2000': 'Ashburner J, Friston KJ (2000). Voxel-based morphometry - the methods. NeuroImage 11(6):805-821. '
                     'doi:10.1006/nimg.2000.0582',
    'reveley2017': 'Reveley C, Gruslys A, Ye FQ, Glen D, Samaha J, Russ BE, Saad Z, Seth AK, Leopold DA, Saleem KS '
                   '(2017). Three-dimensional digital template atlas of the macaque brain. Cerebral Cortex '
                   '27(9):4463-4477. doi:10.1093/cercor/bhw248',
    'saleem2021': 'Saleem KS, Avram AV, Glen D, Yen CC-C, Ye FQ, Komlosh M, Basser PJ (2021). High-resolution mapping '
                  'and digital atlas of subcortical regions in the macaque monkey based on matched MAP-MRI and '
                  'histology. NeuroImage 245:118759. doi:10.1016/j.neuroimage.2021.118759',
}

# statements about function internals: (id, file, regex that must match the source at the run's commit, statement)
CODE_CHECKS = [
    ('downsample_sum', 'slice_registration_functions.py', r'block_reduce\(padded_image, block_size=\(rescale, rescale\), func=numpy\.sum\)',
     'downsampling sums the counts in rescale x rescale blocks'),
    ('downsample_pad', 'slice_registration_functions.py', r'def downsample_image\(image, rescale, prop_pad=0\.2',
     'each side is padded by 20% of the image size before the block sum'),
    ('flip_exception', 'run_slice_registration_optimized_newReg_sdf_v2_local.py', r"'Image_11_-_20x_01_cellCount' in img_orig:\s*\n\s*slice_img = numpy\.flip\(slice_img,axis=0\)",
     'section Image_11_-_20x_01 of TP1 is mirrored along the first image axis before registration'),
    ('centre_affine', 'slice_registration_functions.py', r"ignore_orient=True,",
     'coreg0nl (do_reg) replaces the header orientation by a centred grid (ignore_orient=True)'),
    ('zero_iter_rigid', 'embedded_antspy_smthMod.py', r"if run_rigid is False and run_affine is False and run_syn is False:[\s\S]{0,400}?\[0,1\.0,2\]",
     'without transforms, antsRegistration is called with a rigid stage of 0 iterations (resampling only)'),
    ('median3', 'slice_registration_functions.py', r"start = idx-1\s*\n\s*stop = idx\+2\s*\n\s*slice_template = numpy\.median\(img\[\.\.\.,start:stop\],axis=-1\)",
     'per-section template = median of the section and its two neighbours'),
    ('stack_median', 'slice_registration_functions.py', r"img = numpy\.median\(img,axis=2\)",
     'stack template = median over all sections'),
    ('missing_neighbours', 'slice_registration_functions.py', r"missing_idxs_pre = numpy\.array\(missing_idxs_to_fill\)-1\s*\n\s*missing_idxs_post = numpy\.array\(missing_idxs_to_fill\)\+1",
     'an excluded section is interpolated from the sections immediately before and after it'),
    ('interp_refine', 'slice_registration_functions.py', r"def compute_intermediate_slice\(pre_img, post_img, current_img=None, idx=None, delete_intermediate_files=True,\s*\n\s*reg_refinement_iterations=7",
     'the interpolation refines the average over 7 iterations'),
    ('interp_blur', 'slice_registration_functions.py', r"blur_scales = numpy\.linspace\(0,1,reg_refinement_iterations-1\)\[::-1\]",
     'the Gaussian blur of the running average decreases linearly from 1 to 0 pixel'),
    ('interp_unsharp', 'slice_registration_functions.py', r"avg = compute_histogram_matched_slice\(avg, img1\.get_fdata\(\),img2\.get_fdata\(\)\)[\s\S]{0,600}?unsharp_mask\(avg, sigma=sigma, strength=strength\)",
     'the last iteration matches the histogram to the neighbours and applies an unsharp mask'),
    ('sharpen_stats', 'slice_registration_functions.py', r"detail_energy = np\.mean\(np\.abs\(laplace\(img\)\)\)[\s\S]{0,200}?contrast = np\.std\(img\)",
     'sharpening multipliers: inverse trimmed means of the mean absolute Laplacian and of the SD'),
    ('cascade_ants', 'slice_registration_functions.py', r"reg_aligned = do_reg_ants\(\[source\], \[target\], file_name=output, output_dir=temp_out_dir, run_syn=run_syn",
     'the cascade registers each section to its already registered neighbour (do_reg_ants)'),
    ('do_reg_ants_syn', 'slice_registration_functions.py', r"def do_reg_ants\(sources, targets, run_rigid=True, run_syn=False,\s*\n.*\n\s*mask_zero=False, syn_flow_sigma=3, syn_total_sigma=0",
     'do_reg_ants: ANTsPy Rigid, then SyNOnly with flow sigma 3 and total sigma 0'),
    ('stage1_targets', 'slice_registration_extensions.py', r"targets = \[template\[idx\]\][\s\S]{0,700}?targets\.append\(stack_template\)",
     'STAGE1 targets: per-section template and stack template, then the neighbours'),
    ('mi_select', 'slice_registration_functions.py', r"bins = numpy\.linspace\(min_val, max_val, 100\)[\s\S]{0,400}?mi1c = mutual_info_score\(curr1_binned, curr_binned\)",
     'the two STAGE1 windows are compared by mutual information (100 bins) with the per-section template'),
    ('gw_local', 'slice_registration_extensions.py', r"slice_sigma = local_template_window / 2\.0",
     'groupwise local template: Gaussian weights over sections, sigma = window / 2'),
    ('gw_sdw', 'slice_registration_functions.py', r"weights = 1\.0 / \(1\.0 \+ np\.exp\(-signed_dist / sigma\)\)",
     'groupwise template weights: logistic function of the signed distance to the tissue boundary'),
    ('gw_mask_pct', 'slice_registration_functions.py', r"threshold = np\.percentile\(img_2d\[img_2d > 0\], 5\)",
     'groupwise tissue mask: values above the 5th percentile of the non-zero values'),
    ('gw_source', 'slice_registration_extensions.py', r"if use_deformed_source_after_iteration is None or iteration < use_deformed_source_after_iteration:\s*\n\s*source_fnames = orig_images_fnames",
     'groupwise sources: the STAGE1 sections before the switch iteration, the previous output after it'),
    ('pg_distances', 'slice_posegraph.py', r"DEFAULT_DISTANCES = \(1, 2, 3, 4, 6, 8, 12, 16\)",
     'pose graph: pairs at section distances 1, 2, 3, 4, 6, 8, 12 and 16'),
    ('pg_huber', 'slice_posegraph.py', r"def _huber_lsq\(A, b, w, n_iter=10, k_mad=2\.5\)",
     'pose graph: Huber reweighting, 10 iterations, threshold 2.5 x MAD'),
    ('pg_weight', 'slice_posegraph.py', r"weight 1/d",
     'pose graph: pair weight 1/d'),
    ('pg_pairfit', 'slice_jitter.py', r"ants\.registration\(fi, mv, 'Rigid', aff_metric='GC', aff_random_sampling_rate=1\.0,\s*\n\s*aff_iterations=\(200, 100, 50\), aff_shrink_factors=\(4, 2, 1\),\s*\n\s*aff_smoothing_sigmas=\(2, 1, 0\)",
     'pose graph pair fit: ANTsPy Rigid, global correlation, all samples, 200/100/50 iterations, shrink 4/2/1, sigma 2/1/0'),
    ('weights_norm', 'embedded_antspy_smthMod.py', r"weights\.append\(image_weights\[idx\]/weight_sum\)",
     'the metric weights of a registration are divided by their sum'),
    ('neighbour_default', 'slice_registration_extensions.py', r"else:\s*\n\s*neighbour_tail = previous_tail",
     'without a neighbour tag, the STAGE1 neighbours are the source-level (coreg0nl) sections'),
    ('gw_update', 'slice_registration_extensions.py', r"images = registered_images\s*\n\s*images_fnames = new_images_fnames",
     'groupwise: the templates of each iteration use the output of the previous iteration'),
    ('gw_mask_morph', 'slice_registration_functions.py', r"remove_small_objects\(mask\.astype\(bool\), min_size=min_size\)[\s\S]{0,200}?remove_small_holes[\s\S]{0,300}?binary_erosion",
     'groupwise tissue mask: small objects removed, holes filled, eroded'),
    ('pg_exclude', 'slice_posegraph.py', r"valid\[\[s for s in exclude if 0 <= s < n\]\] = False",
     'pose graph: excluded sections are not valid pair members'),
    ('mri_ref_refine', 'mri_register.py', r"chosen = 'refined from reference'",
     'MRI affine with a stored reference: the reference refined on the stack is used'),
    ('chain', 'slice_registration_functions.py', r"chain = \['coreg0nl', final_rigsyn_col\]",
     'transform chain: coreg0nl -> final STAGE1 level -> groupwise iterations'),
]

ANTS_SIGMA_NOTE = ('antsRegistration reads smoothing sigmas without a unit suffix as voxels (ANTs '
                   'antsRegistrationTemplateHeader.h: no "mm"/"vox" suffix -> physical units false).')


# ----------------------------------------------------------------------------------------------------------------
# sources
# ----------------------------------------------------------------------------------------------------------------
def git_source(commit, fname):
    """File content at the run's commit, else the working tree (with the reason)."""
    if commit and commit != 'unknown':
        try:
            top = subprocess.run(['git', '-C', HERE, 'rev-parse', '--show-toplevel'], capture_output=True, text=True,
                                 check=True).stdout.strip()
            rel = os.path.relpath(os.path.join(HERE, fname), top)
            return subprocess.run(['git', '-C', top, 'show', f'{commit}:{rel}'], capture_output=True, text=True,
                                  check=True).stdout, f'git {commit}'
        except Exception:
            pass
    p = os.path.join(HERE, fname)
    if os.path.isfile(p):
        with open(p) as fh:
            return fh.read(), 'working tree (commit not available)'
    return None, 'not found'


def run_code_checks(commit):
    out = []
    cache = {}
    for cid, fname, pat, text in CODE_CHECKS:
        if fname not in cache:
            cache[fname] = git_source(commit, fname)
        src, where = cache[fname]
        ok = bool(src and re.search(pat, src))
        out.append(dict(id=cid, file=fname, source=where, ok=ok, statement=text))
    return out


def parse_logs(run_dir, cache_dir):
    """Summary of all run logs (cached by size + mtime: the logs can be several GB)."""
    logs = sorted(glob.glob(os.path.join(run_dir, '*_log_*.log')))
    key = [[os.path.basename(f), os.path.getsize(f), int(os.path.getmtime(f))] for f in logs]
    cf = os.path.join(cache_dir, 'methods_log.json')
    if os.path.isfile(cf):
        try:
            with open(cf) as fh:
                c = json.load(fh)
            if c.get('key') == key:
                return c['summary']
        except Exception:
            pass
    pat_args = re.compile(r' - INFO - (\[.*\])\s*$')
    pat_ts = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)')
    regs = collections.OrderedDict()
    s = dict(logs=[], scaling_factor=None, posegraph=[], posegraph_before=None, jitter=[], fallback_mean=0,
             skips=collections.Counter(), messages={})
    for f in logs:
        first = last = None
        with open(f, errors='replace') as fh:
            for line in fh:
                m = pat_ts.match(line)
                if m:
                    first = first or m.group(1)
                    last = m.group(1)
                if "'--metric'" in line:
                    m = pat_args.search(line)
                    if not m:
                        continue
                    try:
                        args = ast.literal_eval(m.group(1))
                    except Exception:
                        continue
                    out = args[args.index('--output') + 1] if '--output' in args else ''
                    tag = re.sub(r'^.*?_pix_?', '', out.split('/')[-1].strip('[]').split(',')[0])
                    tag = re.sub(r'(_tmp.*|_ants.*|_syn.*|\.nii.*)$', '', tag)
                    stages = []
                    for i, a in enumerate(args):
                        a = a.strip()
                        if a == '--transform':
                            stages.append(dict(transform=args[i + 1].replace(' ', ''), metrics=[]))
                        elif a == '--metric' and stages:
                            mm = re.match(r'(\w+)\[[^,]*,[^,]*,([0-9.]+),(\w+),(\w+),([0-9.]+)\]', args[i + 1])
                            if mm:
                                stages[-1]['metrics'].append((mm.group(1), float(mm.group(2)), mm.group(3),
                                                              mm.group(4), mm.group(5)))
                        elif a in ('--convergence', '--smoothing-sigmas', '--shrink-factors') and stages:
                            stages[-1][a.strip('-')] = args[i + 1].replace(' ', '')
                    sig = json.dumps([{k: (v if k != 'metrics' else [list(x[:1]) + list(x[2:]) for x in v][:1] +
                                           [len(v)]) for k, v in st.items()} for st in stages])
                    g = regs.setdefault(tag, dict(n=0, settings=collections.Counter(), weights=collections.Counter()))
                    g['n'] += 1
                    g['settings'][sig] += 1
                    if stages:
                        g['weights'][json.dumps([round(x[1], 3) for x in stages[0]['metrics']])] += 1
                    continue
                if 'Scaling factor set to' in line:
                    s['scaling_factor'] = int(line.rsplit(':', 1)[1])
                elif 'Pose-graph iteration' in line:
                    s['posegraph'].append(line.split(' - WARNING - ', 1)[-1].strip())
                elif 'Pose-graph correction: adjacent NCC median' in line:
                    s['posegraph_before'] = line.split(' - WARNING - ', 1)[-1].strip()
                elif 'Jitter iteration' in line:
                    s['jitter'].append(line.split(' - WARNING - ', 1)[-1].strip())
                elif 'Falling back to simple mean' in line:
                    s['fallback_mean'] += 1
                elif 'exists, skipping' in line or 'already done, skipping' in line or 'already exist, skipping' in line:
                    s['skips'][line.split(' - WARNING - ', 1)[-1].strip()[:90]] += 1
                else:
                    for k in ('OME-Zarr export skipped', 'OME-Zarr written', 'MRI registration finished',
                              'MRI registration skipped', 'Count-preserving stack finished', 'Tissue mask written',
                              'QC report written', 'Tissue mask failed', 'OME-Zarr export failed', 'QC report failed'):
                        if k in line:
                            s['messages'][k] = line.split(' - WARNING - ', 1)[-1].strip()
        s['logs'].append(dict(file=os.path.basename(f), first=first, last=last))
    s['registrations'] = {t: dict(n=g['n'], settings=[[json.loads(k), v] for k, v in g['settings'].most_common(3)],
                                  weights=[[json.loads(k), v] for k, v in g['weights'].most_common(3)])
                          for t, g in regs.items()}
    s['skips'] = dict(s['skips'])
    with open(cf, 'w') as fh:
        json.dump(dict(key=key, summary=s), fh, indent=1)
    return s


def script_facts(run_dir):
    """Module constants (all assignments, in order) and call keywords from the script copy in the run folder."""
    cands = sorted(glob.glob(os.path.join(run_dir, 'run_slice_registration*.py')))
    if not cands:
        return None
    f = cands[0]
    with open(f) as fh:
        src = fh.read()
    tree = ast.parse(src)
    assigns = collections.defaultdict(list)
    calls = collections.defaultdict(list)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                assigns[node.targets[0].id].append((node.lineno, ast.literal_eval(node.value)))
            except Exception:
                pass
        elif isinstance(node, ast.Call):
            fn = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, 'attr', None)
            if fn:
                kw = {}
                for k in node.keywords:
                    if k.arg is None:
                        continue
                    try:
                        kw[k.arg] = ast.literal_eval(k.value)
                    except Exception:
                        kw[k.arg] = ast.unparse(k.value)
                calls[fn].append((node.lineno, kw))
    for v in assigns.values():
        v.sort()
    for v in calls.values():
        v.sort(key=lambda t: t[0])
    return dict(file=os.path.basename(f), assigns=dict(assigns), calls=dict(calls), src=src)


def first(sf, name, default=None):
    v = sf['assigns'].get(name) if sf else None
    return v[0][1] if v else default


def ants_defaults():
    try:
        import ants
        sig = inspect.signature(ants.registration)
        keep = ('aff_metric', 'aff_sampling', 'aff_iterations', 'aff_shrink_factors', 'aff_smoothing_sigmas',
                'syn_metric', 'syn_sampling', 'reg_iterations', 'grad_step')
        return ants.__version__, {k: sig.parameters[k].default for k in keep if k in sig.parameters}
    except Exception:
        return None, {}


# ----------------------------------------------------------------------------------------------------------------
# text helpers
# ----------------------------------------------------------------------------------------------------------------
class Doc:
    def __init__(self):
        self.lines = []
        self.params = []   # (parameter, value, source)
        self.cited = []

    def p(self, text=''):
        self.lines.append(text)

    def h(self, level, text):
        self.lines += ['', '#' * level + ' ' + text, '']

    def par(self, name, value, source):
        self.params.append((name, value, source))
        return value

    def cite(self, *keys):
        for k in keys:
            if k not in self.cited:
                self.cited.append(k)
        return '(' + '; '.join(self._short(k) for k in keys) + ')'

    @staticmethod
    def _short(k):
        r = REFERENCES[k]
        authors = r.split(' (')[0].split(', ')
        year = re.search(r'\((\d{4})\)', r).group(1)
        first_author = authors[0].split(' ')[0]
        return f'{first_author} et al., {year}' if len(authors) > 2 else (
            f'{first_author} and {authors[1].split(" ")[0]}, {year}' if len(authors) == 2 else f'{first_author}, {year}')


def fmt_stage(st):
    """One antsRegistration stage (from the log) as text."""
    m = st.get('metrics', [])
    met = m[0] if m else ['?']
    n = m[-1] if m and isinstance(m[-1], int) else 1
    mname = {'MI': 'mutual information', 'CC': 'cross-correlation', 'Mattes': 'Mattes mutual information'}.get(met[0], met[0])
    detail = ''
    if len(met) >= 4:
        detail = (f', {met[1]} bins' if met[0] in ('MI', 'Mattes') else f', radius {met[1]}') + \
                 f', {met[2].lower()} sampling of {float(met[3]) * 100:.0f}% of the points'
    return (f"{st['transform']}; {mname}{detail}; {n} weighted metric term{'s' if n > 1 else ''}; iterations "
            f"{st.get('convergence', '?')}; shrink factors {st.get('shrink-factors', '?')}; smoothing sigmas "
            f"{st.get('smoothing-sigmas', '?')} voxels")


def level_settings(logs, pattern):
    """Most common registration settings over the log tags that match pattern; (stages, n registrations, tags)."""
    tags = [t for t in (logs or {}).get('registrations', {}) if re.fullmatch(pattern, t)]
    if not tags:
        return None, 0, []
    cnt = collections.Counter()
    for t in tags:
        for st, n in logs['registrations'][t]['settings']:
            cnt[json.dumps(st)] += n
    best = json.loads(cnt.most_common(1)[0][0])
    return best, sum(logs['registrations'][t]['n'] for t in tags), tags


def md_to_html(md):
    """Minimal Markdown (headings, paragraphs, lists, tables, code, bold) to HTML for the QC page."""
    out, para, lst, tbl = [], [], [], []

    def inline(t):
        t = html.escape(t)
        t = re.sub(r'`([^`]+)`', r'<code>\1</code>', t)
        return re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', t)

    def flush():
        if para:
            out.append('<p>' + inline(' '.join(para)) + '</p>')
            para.clear()
        if lst:
            out.append('<ul>' + ''.join(f'<li>{inline(x)}</li>' for x in lst) + '</ul>')
            lst.clear()
        if tbl:
            rows = [[c.strip() for c in r.strip('|').split('|')] for r in tbl if not re.fullmatch(r'\|[\s:|-]+\|', r)]
            h = ''.join(f'<th>{inline(c)}</th>' for c in rows[0])
            b = ''.join('<tr>' + ''.join(f'<td>{inline(c)}</td>' for c in r) + '</tr>' for r in rows[1:])
            out.append(f'<div class="tablewrap"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>')
            tbl.clear()
    for line in md.splitlines():
        if line.startswith('#'):
            flush()
            n = len(line) - len(line.lstrip('#'))
            out.append(f'<h{min(n + 1, 6)}>{inline(line[n:].strip())}</h{min(n + 1, 6)}>')
        elif line.startswith('|'):
            if para or lst:
                flush()
            tbl.append(line)
        elif line.startswith('- '):
            if para or tbl:
                flush()
            lst.append(line[2:])
        elif line.startswith('  ') and lst:
            lst[-1] += ' ' + line.strip()
        elif not line.strip():
            flush()
        else:
            if lst or tbl:
                flush()
            para.append(line.strip())
    flush()
    return '\n'.join(out)


# ----------------------------------------------------------------------------------------------------------------
# the document
# ----------------------------------------------------------------------------------------------------------------
def build(run_dir, subject=SUBJECT, missing=None):
    run_dir = run_dir.rstrip('/') + '/'
    qc_dir = os.path.join(run_dir, 'QC')
    cache_dir = os.path.join(qc_dir, 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    rs = {}
    if os.path.isfile(os.path.join(run_dir, 'run_settings.json')):
        with open(os.path.join(run_dir, 'run_settings.json')) as fh:
            rs = json.load(fh)
    commit = rs.get('git_commit', 'unknown')
    logs = parse_logs(run_dir, cache_dir)
    sf = script_facts(run_dir)
    checks = run_code_checks(commit)
    ok = {c['id']: c['ok'] for c in checks}

    def chk(cid, text):
        return text if ok.get(cid) else f'{text} [UNVERIFIED: code check "{cid}" failed, Appendix B]'

    from qc_report import find_levels
    levels = find_levels(run_dir, subject)
    final = next((lv for lv in reversed(levels) if 'groupwise_iter' in lv), levels[-1] if levels else None)
    rescale = rs.get('rescale')
    px_um = (rs.get('in_plane_res_um') or [None])[0]
    pipe = rs.get('pipeline', {})
    in_um = (pipe.get('in_plane_res_um_input') or [first(sf, 'in_plane_res_x', 10)])[0]
    thick = pipe.get('section_thickness_um', first(sf, 'in_plane_res_z', 50))
    missing = rs.get('missing_idxs_to_fill', missing) or []
    n_sec = rs.get('n_sections')
    if n_sec is None:   # older runs: from the pose table or the final stack
        jp = os.path.join(run_dir, f'{subject}_coreg0nl_jitter_poses.csv')
        try:
            if os.path.isfile(jp):
                with open(jp) as fh:
                    n_sec = sum(1 for _ in fh) - 1
            elif final:
                import nibabel
                n_sec = nibabel.load(os.path.join(run_dir, f'{subject}_{final}_stack.nii.gz')).shape[2]
        except Exception:
            n_sec = None
    sdf = pipe.get('signed_distance_mode', first(sf, 'use_signed_distance_weighting_for_registration'))
    use_res = pipe.get('use_resolution_in_registration', first(sf, 'use_resolution_in_registration'))
    d = Doc()

    # ---------------------------------------------------------------- header
    d.p('# Methods: reconstruction of the cell-count section stack')
    d.p()
    d.p(f'Generated {time.strftime("%Y-%m-%d %H:%M")} by `methods_report.py` from the records of the run in '
        f'`{run_dir}`. The text describes the steps that this run executed, with the parameters that it used. '
        'Appendix A lists every parameter with its source; Appendix B lists the checks of statements about code '
        'internals against the source code at the commit of the run.')
    d.p()
    sess = logs.get('logs', [])
    d.p(f'- Main script: `{os.path.basename(rs.get("script", sf["file"] if sf else "?"))}`; git commit `{commit}`'
        + ('' if rs.get('git_dirty') is None else
           (' (the working tree had uncommitted changes at the start of the run)' if rs.get('git_dirty') else
            ' (no uncommitted changes)')) + '.')
    if sess:
        d.p(f'- Run log(s): {len(sess)}; first entry {sess[0]["first"]}, last entry {sess[-1]["last"]}.'
            + (f' The run was resumed: {sum(logs["skips"].values())} steps were found complete and skipped.'
               if logs.get('skips') else ''))
    if rs.get('packages'):
        d.p('- Software: ' + ', '.join(f'{k} {v}' for k, v in rs['packages'].items()) + '.')
    else:
        av, _ = ants_defaults()
        d.p('- Software versions were not recorded by this run (recorded from 2026-10-29 on); the environment that '
            f'wrote this report has ANTsPy {av}.')

    # ---------------------------------------------------------------- 1 data
    d.h(2, '1. Input data')
    d.p(f'The input consisted of {d.par("number of sections", n_sec, "run" if rs.get("n_sections") else "files")} '
        f'consecutive sections of the cerebellum of subject {subject}, each represented as an image of detected cell '
        f'counts per {d.par("input pixel size (um)", in_um, "script")} um pixel (the output of the automated cell '
        f'detection), with a nominal section thickness of {d.par("section thickness (um)", thick, "script")} um. '
        f'Image values are integer counts of detected cells per pixel.')
    d.p()
    if rescale:
        d.p(chk('downsample_sum', f'Each section was downsampled by a factor of {d.par("rescale", rescale, "run")} in '
            f'both in-plane directions by summing the counts in non-overlapping {rescale} x {rescale} pixel blocks, '
            f'which gives {d.par("in-plane voxel size (um)", px_um, "run")} um pixels and preserves the total cell '
            f'count of each section.') + ' ' +
            chk('downsample_pad', 'Before the block sum, each side of the image was zero-padded by 20% of its size '
                '(and up to a multiple of the factor), so that tissue near the image border remains inside the '
                'registration canvas.'))
        if sdf is False:
            d.p()
            d.p('Registration used the downsampled count images directly (the signed-distance image mode of the '
                f'pipeline was off; cost function `{pipe.get("cost_function", "MutualInformation")}`).')
    flip_in = [f for f in rs.get('input_files', []) if 'Image_11_-_20x_01_cellCount' in f and
               ('TP1' in f or '/testProject/' in f)]
    if flip_in or not rs.get('input_files'):
        d.p()
        d.p(chk('flip_exception', 'The image `Image_11_-_20x_01` of TP1 was mirrored along the first image axis before '
                'registration, because its image was mirrored relative to its neighbours and a rigid registration '
                'cannot correct a reflection.') + ('' if flip_in else ' (This run did not record its input paths; the '
                'rule applies if that file was in the input list.)'))
    if missing:
        d.p()
        d.p(f'Sections {d.par("excluded / interpolated sections", ", ".join(map(str, missing)), "run")} (indices '
            'from 0) were missing or of insufficient quality. They were carried through the registration but, in '
            'every stack, their content was replaced by an interpolation between the two adjacent sections '
            '(Section 3). These interpolated sections contain no measured data and were excluded from the '
            'cell-count summaries (Sections 9 and 10).')

    # ---------------------------------------------------------------- 2 placement and pose
    d.h(2, '2. Common canvas, initial placement and rigid pose correction')
    sfac = rs.get('scaling_factor', logs.get('scaling_factor'))
    canvas = rs.get('canvas_shape')
    d.p(f'All sections were resampled onto the grid of the section image with the largest number of pixels'
        + (f' ({canvas[0]} x {canvas[1]} pixels)' if canvas else '') + '. ' +
        chk('centre_affine', 'For this initial placement (level `coreg0nl`), the header orientation was replaced by a '
            'grid centred on the image centre, so that the centres of all sections coincide.') + ' ' +
        chk('zero_iter_rigid', 'The registration call of this step contained no optimised transform: '
            'antsRegistration was run with a rigid stage of zero iterations, which only resamples the section '
            'onto the canvas.'))
    st, n, _ = level_settings(logs, r'coreg0nl')
    if st:
        d.p()
        d.p(f'Logged settings of this step ({n} registrations): ' + '; '.join(fmt_stage(s) for s in st) + '.')
    if sfac:
        d.par('multi-resolution scaling factor (STAGE1 / groupwise pyramid)', sfac, 'run' if rs.get('scaling_factor') else 'log')
    if rs.get('jitter_correction'):
        method = rs.get('jitter_method')
        if method == 'posegraph':
            d.p()
            d.p('The rigid poses of the sections were then corrected with a pose graph. For each section, rigid '
                'registrations to the sections at distances ' +
                chk('pg_distances', '1, 2, 3, 4, 6, 8, 12 and 16') + ' along the stack were computed ' +
                chk('pg_pairfit', '(ANTsPy rigid registration with the global correlation metric on all pixels; '
                    '200, 100 and 50 iterations at shrink factors 4, 2 and 1 with Gaussian smoothing of 2, 1 and 0 '
                    'voxels)') + f' {d.cite("tustison2021")}. The rotations and then the translations of all '
                'sections were estimated jointly by weighted least squares over all pairs, ' +
                chk('pg_weight', 'each pair weighted by the inverse of its distance') + ', with robust ' +
                chk('pg_huber', 'Huber reweighting of the residuals (threshold 2.5 times the median absolute '
                    'deviation, 10 reweighting iterations)') + f' {d.cite("huber1964")}; the mean rotation and the mean '
                'translation were constrained to zero. ' + chk('pg_exclude', 'Excluded sections did not enter the '
                'pair measurements.') + ' Each rotation acts about the tissue centroid of its section. The low-frequency part '
                f'of the pose sequence (Gaussian, sigma {d.par("pose slow-path sigma (sections)", rs.get("jitter_sigma"), "run")} '
                'sections) was removed from the estimated poses, so that the correction affects only the '
                'section-to-section misalignment and preserves the slow path of the stack (its overall position and '
                f'curvature). The estimation ran in {d.par("pose-graph passes", rs.get("jitter_iterations"), "run")} passes, '
                'each on the stack corrected by the previous passes. Pairs that crossed a block of sections mounted at a common angle '
                'constrained the whole block, which a fit to the nearest neighbours alone cannot do. The poses were '
                'folded into the `coreg0nl` maps and deformed sections, so that they are part of the transform '
                'chain.')
            if logs.get('posegraph'):
                b = re.search(r'NCC median ([0-9.]+)', logs.get('posegraph_before') or '')
                rows = []
                for line in logs['posegraph']:
                    m = re.search(r"iteration (\d+)/(\d+): \{'n_pairs': (\d+), 'rotation_residual_deg_median': ([0-9.e-]+), "
                                  r"'downweighted_pairs': (\d+)\}; adjacent NCC median ([0-9.]+), rotation SD ([0-9.]+) deg, "
                                  r"shift SD ([0-9.]+) mm", line)
                    if m:
                        rows.append(m.groups())
                if rows:
                    d.p()
                    d.p(f'In this run, the median NCC of adjacent sections was {b.group(1) if b else "?"} before the '
                        'correction; per pass: ' + '; '.join(
                            f'pass {r[0]}: {r[2]} pairs ({r[4]} down-weighted), median rotation residual '
                            f'{float(r[3]):.2f} deg, adjacent NCC {r[5]}, total rotation SD {r[6]} deg, total shift SD '
                            f'{r[7]} mm' for r in rows) + '.')
        else:
            d.p()
            d.p(f'The rigid poses were then corrected by fitting each section to the mean of its neighbours '
                f'(method `{method}`, {rs.get("jitter_iterations")} iterations, slow path removed with sigma '
                f'{rs.get("jitter_sigma")} sections).')
        jp = os.path.join(run_dir, f'{subject}_coreg0nl_jitter_poses.csv')
        if os.path.isfile(jp):
            import pandas as pd
            P = pd.read_csv(jp)
            keep = ~P.slice_idx.isin(missing)
            d.p()
            d.p(f'Applied pose corrections (measured sections): rotation SD {P.theta_deg[keep].std():.2f} deg '
                f'(maximum |rotation| {P.theta_deg[keep].abs().max():.1f} deg); translation SD '
                f'{np.hypot(P.tx_mm[keep], P.ty_mm[keep]).std():.3f} mm.')

    # ---------------------------------------------------------------- 3 templates and missing sections
    d.h(2, '3. Templates and interpolation of excluded sections')
    d.p('After each registration level, the deformed sections were stacked, and ' +
        chk('stack_median', 'a stack template was formed as the voxel-wise median over all sections') + '. After the '
        'cascade and after each STAGE1 pass, ' +
        chk('median3', 'a template was also formed for every section as the voxel-wise median of the section and its '
            'two neighbours (the two outermost sections used the median of two)') + '. These templates were the '
        'registration targets of the next STAGE1 pass.')
    if missing:
        mmeth = pipe.get('missing_slice_interp_method', 'intermediate_nonlin_mean')
        d.p()
        d.p(chk('missing_neighbours', 'Each excluded section was replaced by an image interpolated from the two '
                'sections adjacent to it.') + ' ' +
            (('The two neighbours were first registered rigidly to their average. ' +
              chk('interp_refine', 'The average was then refined iteratively (7 iterations): both neighbours were '
                  'registered to the current average (ANTsPy rigid, then SyN) and averaged again,') + ' ' +
              chk('interp_blur', 'with a Gaussian blur of the average that decreased linearly from 1 pixel to 0 '
                  'over the iterations.') + ' ' +
              chk('interp_unsharp', 'In the last iteration, the histogram of the average was matched to the '
                  'neighbours, and, where sharpening multipliers were given (below), an unsharp mask was applied '
                  'with radius and strength derived from the neighbours.'))
             if mmeth == 'intermediate_nonlin_mean' else f'Interpolation method: `{mmeth}`.'))
        shp = rs.get('sharpening')
        nshp = len([c for c in (sf or {}).get('calls', {}).get('generate_stack_and_template', [])
                    if 'sigma_multiplier' in c[1]])
        if shp or nshp:
            d.p(chk('sharpen_stats', 'The unsharp-mask parameters were scaled by dataset-wide multipliers: the '
                    'inverse of the 5%-trimmed mean, over all sections, of the mean absolute Laplacian (radius) and '
                    'of the intensity standard deviation (strength).')
                + (f' Values: sigma multiplier {shp.get("sigma_multiplier"):.4g}, strength multiplier '
                   f'{shp.get("strength_multiplier"):.4g}.' if shp else '')
                + (' In this script, the multipliers were passed only to the stacking of the `coreg0nl` level; '
                   'the later levels interpolated without the unsharp mask.' if nshp == 1 else ''))
        if logs.get('fallback_mean'):
            d.p(f'The interpolation failed {logs["fallback_mean"]} times and fell back to the plain mean of the '
                'neighbours.')

    # ---------------------------------------------------------------- 4 cascade
    if any(lv.startswith('cascade_') for lv in levels):
        d.h(2, '4. Initial non-linear alignment (cascade)')
        av, ad = ants_defaults()
        d.p(chk('cascade_ants', 'A cascade of pairwise registrations started at the middle section of the stack (the '
                'nearest non-excluded section) and proceeded outwards in both directions; each section was '
                'registered (from its `coreg0nl` image) to its already registered neighbour.') + ' ' +
            chk('do_reg_ants_syn', 'Each registration consisted of an ANTsPy rigid registration followed by a SyN '
                f'registration (ANTsPy `SyNOnly`, flow sigma 3, total sigma 0) {d.cite("avants2008", "tustison2021")}.')
            + (f' The remaining parameters were the ANTsPy {av} defaults: rigid stage with the '
               f'{ad.get("aff_metric")} metric ({ad.get("aff_sampling")} bins), iterations '
               f'{ad.get("aff_iterations")}, shrink factors {ad.get("aff_shrink_factors")} and smoothing sigmas '
               f'{ad.get("aff_smoothing_sigmas")} voxels; SyN stage with the {ad.get("syn_metric")} metric, '
               f'gradient step {ad.get("grad_step")} and iterations {ad.get("reg_iterations")}. '
               '(Version of the environment that wrote this report; see the software line above for the run.)'
               if ad else '') +
            ' The cascade result was used only to build the templates for the next stage; the registrations of '
            'all later stages started again from the `coreg0nl` sections.')

    # ---------------------------------------------------------------- 5 STAGE1
    s1_iters = rs.get('stage1', {}).get('iterations', first(sf, 'num_reg_iterations', 5))
    if any(re.fullmatch(r'coreg12nl_win12_rigsyn_\d+', lv) for lv in levels):
        d.h(2, '5. Template-guided registration (STAGE1)')
        offs = (sf or {}).get('assigns', {}).get('slice_offset_list_forward', [])
        offr = (sf or {}).get('assigns', {}).get('slice_offset_list_reverse', [])
        gstd = [c[1].get('gauss_std') for c in (sf or {}).get('calls', {}).get('generate_gaussian_weights', [])]
        reg_preset = rs.get('stage1', {}).get('regularization', 'Medium')
        d.p(f'STAGE1 consisted of {d.par("STAGE1 levels", s1_iters, "run" if rs.get("stage1") else "script")} levels. '
            'At every level, each section was registered again from its `coreg0nl` image (so that deformations '
            'did not accumulate across levels) to a weighted set of targets: ' +
            chk('stage1_targets', 'its template and the stack template of the previous level, and its '
                'neighbouring sections') + ' (' + chk('neighbour_default', 'their `coreg0nl` images') + '). The weights '
            'followed a Gaussian of the section offset; the per-section template and the stack template each received '
            'the weight of offset 0, and ' + chk('weights_norm', 'all weights of a registration were divided by their '
            'sum') + '. Each registration was a '
            'rigid stage followed by a SyN stage, both driven by the weighted sum of the mutual-information metrics '
            f'of all targets {d.cite("avants2008", "avants2011", "maes1997")}, run through a modified copy of the '
            f'multi-contrast ANTs wrapper of nighres (`embedded_antspy_2d_multi`) {d.cite("huntenburg2018")}.')
        if len(offs) >= 2 and len(offr) >= 2 and len(gstd) >= 4:
            d.p()
            d.p(f'Each level ran two passes, and each pass registered every section twice, with a backward-weighted '
                f'and a forward-weighted window of neighbours. Pass 1: offsets {offs[0][1]} and {offr[0][1]}, Gaussian '
                f'sigma {gstd[0]} sections. Pass 2: offsets {offs[1][1]} and {offr[1][1]}, Gaussian sigma {gstd[2]} '
                'sections. ' + chk('mi_select', 'After each pass, the result of the window with the higher mutual '
                                   'information (100 bins) with the per-section template was kept') +
                ', and new templates were formed (Section 3). The templates of pass 1 were the targets of pass 2.')
            d.par('STAGE1 pass-1 offsets (back / forward)', f'{offs[0][1]} / {offr[0][1]}', 'script')
            d.par('STAGE1 pass-2 offsets (back / forward)', f'{offs[1][1]} / {offr[1][1]}', 'script')
            d.par('STAGE1 Gaussian sigma of the weights (sections)', f'{gstd[0]} / {gstd[2]}', 'script')
        st1, n1, _ = level_settings(logs, r'coreg[12]nl_rigsyn_\d+')
        st, n, _ = level_settings(logs, r'coreg12nl_win[12]_rigsyn_\d+')
        if st1:
            d.p()
            d.p(f'Registration settings logged for pass 1 ({n1} registrations): ' + ' | '.join(fmt_stage(x) for x in st1)
                + '.')
        if st:
            d.p()
            d.p(f'Registration settings logged for pass 2 ({n} registrations; SyN preset '
                f'`{d.par("STAGE1 SyN preset", reg_preset, "run" if rs.get("stage1") else "script")}`): ' +
                ' | '.join(fmt_stage(s) for s in st) + '. ' + ANTS_SIGMA_NOTE)
            w = None
            for t in logs['registrations']:
                if re.fullmatch(r'coreg12nl_win1_rigsyn_\d+', t):
                    w = logs['registrations'][t]['weights'][0][0]
                    break
            if w:
                d.p(f'Metric weights of an interior section (pass 2, backward window; per-section template, stack template, then the neighbours from the farthest back): {w}.')
        s1_last = rs.get('stage1', {}).get('syn_last_level', True)
        d.p()
        d.p(f'The maps of the last STAGE1 level (`coreg12nl_win12_rigsyn_{s1_iters - 1}`) form the second element of '
            'the transform chain' + ('' if s1_last else ' (this level was rigid only)') + '.')

    # ---------------------------------------------------------------- 6 groupwise
    gw_levels = [lv for lv in levels if 'groupwise_iter' in lv]
    if gw_levels:
        g = rs.get('groupwise') or {}
        gcall = ((sf or {}).get('calls', {}).get('groupwise_stack_optimization_embedded_antspy_v2') or [[0, {}]])[0][1]
        gsrc = 'run' if g else 'script'
        n_gw = d.par('groupwise iterations', g.get('iterations', rs.get('groupwise_iterations')), 'run')
        sw = d.par('groupwise: first iteration with the previous output as source',
                   g.get('use_deformed_source_after_iteration', gcall.get('use_deformed_source_after_iteration')), gsrc)
        lt = d.par('groupwise: first iteration with local templates',
                   g.get('use_local_template_after_iteration', gcall.get('use_local_template_after_iteration')), gsrc)
        win = d.par('groupwise: local template half-window (sections)',
                    g.get('local_template_window', gcall.get('local_template_window')), gsrc)
        d.h(2, '6. Groupwise refinement')
        d.p(f'The sections of the last STAGE1 level were refined in {n_gw} groupwise iterations. In each iteration, '
            'every section was registered with a SyN transformation to a template computed from the current '
            f'stack. In iterations 0 to {lt - 1}, the template was the weighted mean of all sections; from iteration '
            f'{lt} on, each section had a local template, ' +
            chk('gw_local', f'the weighted mean of the sections within +/-{win} sections, with Gaussian weights of '
                f'the section distance (sigma {win / 2:g} sections)') + '. ' +
            chk('gw_sdw', 'In both cases, each section contributed with spatial weights given by a logistic function '
                '(scale 3 pixels) of the signed distance to its tissue boundary') + ', ' +
            chk('gw_mask_pct', 'with tissue defined as values above the 5th percentile of the non-zero values') + ' ' +
            chk('gw_mask_morph', '(objects under 100 pixels removed, holes filled, eroded by 1 pixel)') + '; ' +
            chk('gw_update', 'the templates of each iteration were computed from the output of the previous one') +
            '. ' + chk('gw_source', f'In iterations 0 to {sw - 1}, the source of each registration was the STAGE1 '
                       f'section; from iteration {sw} on, it was the output of the previous iteration') +
            f'. The transform chain therefore starts with iteration {sw - 1} (registered from the STAGE1 section) '
            f'and continues with iterations {sw} to {n_gw - 1}; the maps of earlier iterations were not kept.')
        st, n, _ = level_settings(logs, r'.*_groupwise_iter\d+')
        if st:
            d.p()
            d.p(f'Logged settings ({n} registrations): ' + ' | '.join(fmt_stage(s) for s in st) +
                f'. The SyN parameters [gradient step, update-field sigma, total-field sigma] were set for a smooth '
                f'deformation {d.cite("avants2008")}. ' + ANTS_SIGMA_NOTE)

    # ---------------------------------------------------------------- 7 chain and stack
    d.h(2, '7. Transform chain and final stack')
    n_chain = 2 + max(0, len(gw_levels) - 3) if gw_levels else 2
    d.p(chk('chain', f'The final position of each section is given by the sequential application of the coordinate '
            f'maps `coreg0nl` -> `coreg12nl_win12_rigsyn_{s1_iters - 1}` -> groupwise iterations') +
        f' ({n_chain} maps for `{final}`). Each map is a pull map: it stores, for each output pixel, the source '
        'pixel coordinates. The maps of all levels are listed per section in `' + subject + '_forward_maps.csv` '
        'and `' + subject + '_inverse_maps.csv`. Each map application resampled the image with linear '
        'interpolation, so the final sections were interpolated once per map in the chain.')
    d.p()
    d.p(f'The final stack `{subject}_{final}_stack.nii.gz` has {px_um} x {px_um} um in-plane voxels and {thick} um '
        'between sections. Its values are counts per native pixel area at the corresponding native position '
        '(no Jacobian modulation), i.e. a density; they are suited to visualisation, masks and boundaries.')

    # ---------------------------------------------------------------- 8 tissue mask
    if final and os.path.isfile(os.path.join(run_dir, f'{subject}_{final}_tissue_mask.nii.gz')):
        d.h(2, '8. Tissue mask')
        d.p('A tissue mask was written for the final level: voxels with more than 2 detected cells per 50 x 50 um '
            'of section area (the threshold scales with the voxel area; about 16,000 cells/mm^3 for 50 um '
            'sections), with holes inside each section filled, because white matter has a low cell density but '
            'belongs to the tissue.')

    # ---------------------------------------------------------------- 9 counts
    cj = os.path.join(run_dir, f'{subject}_{final}_counts_stack.json') if final else ''
    if cj and os.path.isfile(cj):
        with open(cj) as fh:
            c = json.load(fh)
        k = c.get('check', {})
        d.h(2, '9. Count-preserving stack')
        d.p('Because the warps change the local area, the sum of the final stack over a section or a region is not '
            'the number of cells in it. A count-preserving stack was therefore computed: each native section was '
            'passed through the same chain of maps, and the result of each step was multiplied by the absolute '
            'Jacobian determinant of that map (the derivative of the source pixel coordinates with respect to the '
            'output pixel index, by central differences; zero where the map samples outside the source canvas). '
            'This is the change-of-variables rule that preserves integrals, the same correction as the modulation '
            f'of spatially normalised tissue maps by the Jacobian determinants {d.cite("ashburner2000")}. An image of '
            'ones passed through the same steps gives the native area per final pixel, so that the density of a '
            'region is the ratio of its summed counts to its summed native area times the section thickness.')
        if k:
            d.p()
            d.p(f'Check: per measured section, the ratio of the modulated total to the native total deviated from 1 '
                f'by a median of {100 * k["modulated_ratio_abs_dev_median"]:.2f}% (95th percentile '
                f'{100 * k["modulated_ratio_abs_dev_p95"]:.2f}%, maximum {100 * k["modulated_ratio_abs_dev_max"]:.2f}%), '
                f'against a median of {100 * k["unmodulated_ratio_abs_dev_median"]:.2f}% (maximum '
                f'{100 * k["unmodulated_ratio_abs_dev_max"]:.1f}%) for the unmodulated final stack. The largest share '
                f'of the counts on pixels with a negative Jacobian determinant (folding) was '
                f'{k.get("folded_count_fraction_max", float("nan")):.4f}.')

    # ---------------------------------------------------------------- 10 MRI and labels
    mq = sorted(glob.glob(os.path.join(qc_dir, 'mri', '*', 'qc.json')))
    if mq:
        with open(mq[-1]) as fh:
            q = json.load(fh)
        aff, nl = q.get('tables', {}).get('affine', {}), q.get('tables', {}).get('nonlinear', {})
        d.h(2, '10. Registration to the D99 template')
        d.p(f'The stack of level `{q.get("level")}` was registered to the cerebellum and brainstem of the D99 '
            f'macaque template {d.cite("reveley2017", "saleem2021")} (cropped, 0.25 mm). D99 is a different animal; '
            'this registration places the stack in a common space and transfers atlas labels, and it does not '
            'change the reconstruction. ' + (
                chk('mri_ref_refine', 'The affine transformation (without shear) was a stored reference affine, refined '
                    'on this stack; the unrefined reference is reported for comparison.')
                if aff.get('chosen') == 'refined from reference' else
                ('The affine transformation was given as an input.' if aff.get('chosen') == 'given' else
                 f'The affine transformation (without shear) came from a full search; the candidate '
                 f'"{aff.get("chosen")}" was chosen by mutual information.')))
        if aff:
            d.p()
            d.p('Affine results: ' + '; '.join(f'{kk} = {v if isinstance(v, str) else f"{v:.4g}"}'
                                                for kk, v in aff.items() if kk != 'chosen') + '.')
        if nl:
            d.p()
            d.p(f'A non-linear SyN registration followed {d.cite("avants2008")}: {nl.get("SyN", "")}. ' +
                '; '.join(f'{kk} = {v if isinstance(v, str) else f"{v:.4g}"}' for kk, v in nl.items() if kk != 'SyN')
                + f'. Mutual information here is the Mattes formulation {d.cite("mattes2003")}.')
        lq = os.path.join(os.path.dirname(mq[-1]), 'atlas_labels.json')
        if os.path.isfile(lq):
            with open(lq) as fh:
                L = json.load(fh)
            d.p()
            d.p('**D99 labels (work in progress, not validated).** The D99 v2.0 labels were transferred into the '
                'stack grid with the inverse transformations and nearest-label interpolation. ' +
                ' '.join(L.get('notes', [])[1:]))
            summ = L.get('tables', {}).get('D99 labels: summary', {})
            if summ:
                d.p()
                d.p('| measure | value |')
                d.p('|---|---|')
                for kk, v in summ.items():
                    d.p(f'| {kk} | {v if isinstance(v, str) else f"{v:.4g}"} |')
    msg = logs.get('messages', {})
    if 'OME-Zarr export skipped' in msg:
        d.p()
        d.p(f'The OME-Zarr export was not performed in this run ({msg["OME-Zarr export skipped"]}).')

    # ---------------------------------------------------------------- 11 granular mask (when present)
    gj = sorted(glob.glob(os.path.join(run_dir, f'{subject}_*_granular_mask.json')))
    if gj:
        with open(gj[-1]) as fh:
            gm = json.load(fh)
        d.h(2, '11. Granular-layer mask')
        d.p(gm.get('methods', 'A granular-layer mask was computed (see its json sidecar).'))

    # ---------------------------------------------------------------- QC
    mf = os.path.join(qc_dir, 'metrics.json')
    if os.path.isfile(mf):
        with open(mf) as fh:
            M = json.load(fh)
        d.h(2, 'Quality control')
        d.p(f'Continuity was measured between adjacent sections on a {M.get("qc_grid_mm", 0.2):.2f} mm in-plane grid '
            '(block sums of the stack): normalised cross-correlation (NCC); NCC after the same in-plane smoothing '
            'of every level (blur-equalised NCC, because NCC increases with interpolation blur); and the Dice '
            'overlap of adjacent tissue masks.')
        d.p()
        d.p('| level | NCC median | NCC p5 | blur-eq. NCC median | blur-eq. NCC p5 | Dice median | Dice p5 |')
        d.p('|---|---|---|---|---|---|---|')
        for lv, m in M.get('levels', {}).items():
            if lv in ('coreg0nl', 'cascade_0') or lv == M.get('final_level') or re.fullmatch(r'coreg12nl_win12_rigsyn_\d+', lv):
                d.p(f'| {lv} | {m["ncc_median"]:.3f} | {m["ncc_p5"]:.3f} | {m["blur_eq_ncc_median"]:.3f} | '
                    f'{m["blur_eq_ncc_p5"]:.3f} | {m["mask_dice_median"]:.3f} | {m["mask_dice_p5"]:.3f} |')

    # ---------------------------------------------------------------- limitations
    d.h(2, 'Notes and limitations')
    d.p('- The initial placement (`coreg0nl`) is a centre alignment without optimisation; the rigid alignment of '
        'the sections comes from the pose correction (when it ran) and from the rigid stages of STAGE1.')
    d.p('- The reconstruction has no external shape reference (no blockface images); its overall shape follows '
        'the stack itself. The D99 registration does not feed back into it.')
    d.p('- Interpolated sections are synthetic and must not be used as data.')
    if any(not c['ok'] for c in checks):
        d.p('- Some statements about code internals could not be verified against the source at the run commit '
            '(marked UNVERIFIED; Appendix B).')

    # ---------------------------------------------------------------- references and appendices
    d.h(2, 'References')
    for k in sorted(d.cited, key=lambda k: REFERENCES[k]):
        d.p(f'- {REFERENCES[k]}')
    d.h(2, 'Appendix A. Parameters and their sources')
    d.p('Sources: run = run_settings.json; log = run log; script = copy of the main script in the run folder; '
        'files = outputs and sidecars.')
    d.p()
    d.p('| parameter | value | source |')
    d.p('|---|---|---|')
    for name, v, src in d.params:
        d.p(f'| {name} | {v} | {src} |')
    for t, g in (logs.get('registrations') or {}).items():
        pass
    d.h(2, 'Appendix B. Checks of statements about code internals')
    d.p(f'Source: the files at git commit `{commit}` (or the working tree where the commit is not available).')
    d.p()
    d.p('| check | statement | file | source | result |')
    d.p('|---|---|---|---|---|')
    for c in checks:
        d.p(f'| {c["id"]} | {c["statement"]} | {c["file"]} | {c["source"]} | {"verified" if c["ok"] else "FAILED"} |')
    md = '\n'.join(d.lines).replace('\n\n\n', '\n\n') + '\n'
    return md, checks


def write(run_dir, subject=SUBJECT, missing=None):
    md, checks = build(run_dir, subject, missing)
    out = os.path.join(run_dir.rstrip('/'), 'QC', 'METHODS.md')
    with open(out, 'w') as fh:
        fh.write(md)
    return out, md, checks


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--run-dir', required=True)
    p.add_argument('--subject', default=SUBJECT)
    p.add_argument('--check', action='store_true', help='print the code checks; exit 1 if one fails')
    a = p.parse_args()
    out, md, checks = write(a.run_dir, a.subject)
    print(f'written: {out}')
    if a.check:
        for c in checks:
            print(f'{"ok    " if c["ok"] else "FAILED"} {c["id"]:20s} {c["source"]:30s} {c["statement"]}')
        raise SystemExit(0 if all(c['ok'] for c in checks) else 1)


if __name__ == '__main__':
    main()
