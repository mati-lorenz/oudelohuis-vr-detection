# -*- coding: utf-8 -*-
"""
Mutual information between each neuron's continuous deconvolved trace
and four continuous behavioral/state variables: position along the VR
corridor, running speed, pupil area, and the first principal component
of face-video motion (see infotheory/behavior_signals.py for how each
is extracted and aligned to the imaging frame rate).

This is computed over the WHOLE session (not trial-windowed), since
these are continuously-varying state variables rather than trial-locked
task events -- the natural companion to the stim/choice MI analyses
elsewhere in this pipeline, and the information-theoretic upgrade of
the simple runspeed-correlation check in plot_spike_exploratory.py (MI
also catches nonlinear relationships that a Pearson correlation would
miss -- see behavior_mi.py's module docstring).

Fully parallelized: per-neuron MI computation (infotheory/behavior_mi.py)
uses joblib across neurons, and sessions are processed in parallel via
joblib as well (nested-parallelism-aware, see n_jobs_sessions/n_jobs_
neurons below).

If a session is missing video tracking (no videodata) or behavioral
tracking (no zpos_F/runspeed_F), the corresponding variable(s) are
skipped for that session with a warning, rather than failing the whole
run -- results are still pooled across whichever sessions DID have each
variable.

Produces one figure (2x2 grid, one panel per behavioral variable) plus
a per-neuron summary CSV per variable.
"""

#%%
import os, sys
os.chdir('/u/g/glorenz/Documents/Research/Code/external_repos/oudelohuis-vr-detection')
sys.path.insert(0, '/u/g/glorenz/Documents/Research/Code/external_repos/oudelohuis-vr-detection')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']  # avoid 'findfont: Arial not found' warning
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from loaddata.session_info import filter_sessions, report_sessions
from infotheory import InfoTheoryParams, BinningParams, BiasCorrectionParams, ParallelParams
from infotheory.behavior_signals import get_behavior_trace
from infotheory.behavior_mi import compute_behavior_mi
from infotheory.celldata_utils import (
    get_area_label, ordered_groups, bar_by_group, AREA_COLORS, LABEL_LINESTYLES, DEFAULT_LABEL_ORDER)
from utils.cellselection_lib import filter_nearlabeled_layer23
from qc_lib import load_qc_pass

#%% ------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
protocol = ['DN']
calciumversion = 'deconv'

behavior_vars = ['position', 'runspeed', 'pupil_area', 'video_pc1']
video_pc_column = None   # set explicitly once you know your dataset's face-motion PC
                          # column name, e.g. 'motSVD_0' -- see behavior_signals.py

# Per-BEHAVIORAL-VARIABLE bin count override (see behavior_mi.compute_behavior_mi's
# target_n_bins/target_method docstring) -- these bin the behavioral trace only, NOT
# the neuron responses (those stay at params.binning.n_bins below, deliberately kept
# coarse). Position and running speed are smooth, densely-sampled, whole-session
# traces with plenty of samples to support a finer discretization without the
# small-sample bias concerns that keep per-trial neural response binning low;
# pupil_area/video_pc1 are noisier proxies where over-binning mostly adds noise, so
# they're left closer to the neuron bin count. A variable not listed here falls back
# to params.binning.n_bins (the old, single-n_bins-for-everything behavior).
target_n_bins = {
    'position': 8,
    'runspeed': 8,
    'pupil_area': 8,
    'video_pc1': 8,
}

# Per-BEHAVIORAL-VARIABLE binning-method override (see behavior_mi.compute_behavior_mi's
# target_n_bins/target_method docstring). Position and runspeed are better served by
# EQUAL-WIDTH bins here rather than the equipopulated default: equipopulated binning
# places edges at quantiles, and both variables have heavy ties (runspeed sits at
# exactly 0 for a large fraction of samples when the mouse is stationary; position can
# repeat if the animal pauses at the same corridor location) -- ties collapse
# equipopulated edges (see discretize.equipopulated_edges), silently capping the actual
# number of bins far below what you asked for (e.g. requesting 32 can silently collapse
# to 16 or fewer -- check the 'n_target_bins_used' column in the output CSV/prints if
# unsure). Equal-width bins don't have this collapse failure mode: a tied value just
# makes one bin's occupancy large rather than eliminating bin edges. A variable not
# listed here falls back to params.binning.method ('equipopulated').
target_method = {
    'position': 'equal_width',
    'runspeed': 'equal_width',
    'pupil_area': 'equal_width',
    'video_pc1': 'equal_width',
}

params = InfoTheoryParams(
    binning=BinningParams(method='equipopulated', n_bins=4),   # neuron response binning
    bias=BiasCorrectionParams(panzeri_treves=True, random_state=0),
    parallel=ParallelParams(n_jobs=1, backend='loky', verbose=0),  # neuron-level n_jobs set below
)
compute_significance = False   # shuffle-test p-values per neuron; expensive at population scale

# n_jobs at the SESSION level (primary parallelism, one worker per session).
# Keep n_jobs_neurons small when running many sessions in parallel to avoid
# oversubscribing CPUs with nested parallel pools; bump n_jobs_neurons if
# you only have 1-2 sessions.
n_jobs_sessions = -1
n_jobs_neurons = 1

output_dir = './output/infotheory_plots'
os.makedirs(output_dir, exist_ok=True)

# QC filter (in addition to the anatomical V1/PM filter): produced by
# run_qc_summary.py. Run that script first; set to None to skip QC
# filtering (e.g. for a quick look before QC has been run).
qc_csv_path = os.path.join(output_dir, 'qc_per_cell.csv')

#%% ------------------------------------------------------------------
# Load sessions (shallow)
# ----------------------------------------------------------------------
sessions, nSessions = filter_sessions(
    protocols=protocol, load_behaviordata=False, load_calciumdata=False,
    load_videodata=False, min_trials=100)
report_sessions(sessions)

#%% ------------------------------------------------------------------
# Per-session computation (parallelized across sessions)
# ----------------------------------------------------------------------
def process_session(ses, session_index, behavior_vars, calciumversion, params,
                     video_pc_column, compute_significance, n_jobs_neurons, qc_csv_path,
                     target_n_bins, target_method):
    print(f'\n=== Session {session_index + 1}: {ses.session_id} ===')

    ses.load_data(load_behaviordata=True, load_calciumdata=True,
                  load_videodata=True, calciumversion=calciumversion)
    calciumdata = np.asarray(ses.calciumdata)
    area, label, _ = get_area_label(ses.celldata)

    # V1/PM unl cells must be near a labeled cell (50um) and in layer 2/3
    # (depth<300um); labeled cells and cells outside V1/PM pass through
    # unaffected (see utils/cellselection_lib.py). Combined with the QC
    # pass/fail flag from run_qc_summary.py (silent/artifact/noisy cells).
    # Apply BEFORE computing MI so 'neuron_index' below stays aligned with
    # the filtered area/label.
    idx_anat = filter_nearlabeled_layer23(ses, radius=50, depth_thr=300)
    idx_qc = load_qc_pass(ses, qc_csv_path=qc_csv_path)
    idx_valid = idx_anat  & idx_qc
    print(f'  [{ses.session_id}] {np.sum(idx_valid)}/{len(idx_valid)} neurons kept '
          f'(anatomical: {np.sum(idx_anat)}, QC: {np.sum(idx_qc)})')
    calciumdata = calciumdata[:, idx_valid]
    area, label = area[idx_valid], label[idx_valid]

    # per-(area,label) neuron counts THIS session -- for direct comparison
    # against run_dimensionality_estimate.py's dimensionality_group_status.csv,
    # since that script requires >= min_neurons_per_group PER SESSION (no
    # pooling across sessions) while this script pools across sessions before
    # plotting, so a group can look fine here while being skipped there.
    group_status_rows = []
    for a in np.unique(area):
        for l in DEFAULT_LABEL_ORDER:
            n_neurons_group = int(np.sum((area == a) & (label == l)))
            if n_neurons_group > 0:
                group_status_rows.append({'area': a, 'label': l, 'n_neurons_group': n_neurons_group,
                                           'session_id': ses.session_id})
    df_group_status = pd.DataFrame(group_status_rows)

    params_local = InfoTheoryParams(
        binning=params.binning, bias=params.bias,
        parallel=ParallelParams(n_jobs=n_jobs_neurons, backend=params.parallel.backend,
                                 verbose=params.parallel.verbose))

    per_var_df = {}
    for var_name in behavior_vars:
        try:
            trace = get_behavior_trace(ses, var_name, video_pc_column=video_pc_column)
        except ValueError as e:
            print(f'  [{ses.session_id}] SKIPPING "{var_name}": {e}')
            continue
        df = compute_behavior_mi(calciumdata, trace, params=params_local,
                                  compute_significance=compute_significance,
                                  target_n_bins=target_n_bins.get(var_name),
                                  target_method=target_method.get(var_name))
        df['area'] = area[df['neuron_index'].to_numpy()]
        df['label'] = label[df['neuron_index'].to_numpy()]
        per_var_df[var_name] = df
        print(f'  [{ses.session_id}] {var_name}: mean I_pt = {df["I_pt"].mean():.4f} bits '
              f'over {len(df)} neurons ({df["n_target_bins_used"].iloc[0]} target bins '
              f'[{target_method.get(var_name, params.binning.method)}], '
              f'{df["n_bins_used"].iloc[0]} response bins)')

    for attr in ('calciumdata', 'videodata', 'behaviordata'):
        if hasattr(ses, attr):
            delattr(ses, attr)

    return {'session_id': ses.session_id, 'per_var_df': per_var_df, 'group_status': df_group_status}


session_results = Parallel(n_jobs=n_jobs_sessions, backend='loky', verbose=10)(
    delayed(process_session)(
        ses, ises, behavior_vars, calciumversion, params,
        video_pc_column, compute_significance, n_jobs_neurons, qc_csv_path,
        target_n_bins, target_method)
    for ises, ses in enumerate(sessions)
)

#%% ------------------------------------------------------------------
# Group status per session -- check this first if a group (e.g. PMunl)
# looks different here than in run_dimensionality_estimate.py: this
# script pools neurons across ALL sessions before plotting/pooling MI, so
# a group with few neurons per session can still look healthy here while
# being skipped entirely there (which enforces min_neurons_per_group
# PER SESSION, no pooling). Comparing the two tables side by side tells
# you whether a "missing" group is a real filtering difference or just
# the pooling-vs-per-session distinction.
# ----------------------------------------------------------------------
df_group_status_all = pd.concat([r['group_status'] for r in session_results if len(r['group_status'])],
                                 ignore_index=True)
df_group_status_all.to_csv(os.path.join(output_dir, 'mi_group_status.csv'), index=False)

print('\nGroup status per session (check this first if a group is missing from the results):')
print(df_group_status_all.sort_values(['area', 'label', 'session_id']).to_string(index=False))
print('\nTotal pooled neurons per group (summed across sessions -- this is what actually goes into the plot):')
print(df_group_status_all.groupby(['area', 'label'])['n_neurons_group'].sum().to_string())

#%% ------------------------------------------------------------------
# Pool across sessions per behavioral variable
# ----------------------------------------------------------------------
all_df_by_var = {var: [] for var in behavior_vars}
for res in session_results:
    for var_name, df in res['per_var_df'].items():
        df = df.copy()
        df['session_id'] = res['session_id']
        all_df_by_var[var_name].append(df)

pooled_by_var = {}
for var_name, dfs in all_df_by_var.items():
    if dfs:
        pooled = pd.concat(dfs, ignore_index=True)
        pooled_by_var[var_name] = pooled
        pooled.to_csv(os.path.join(output_dir, f'behavior_mi_{var_name}.csv'), index=False)
        print(f'\nSaved {var_name} MI table ({len(pooled)} neurons) to '
              f'{os.path.join(output_dir, f"behavior_mi_{var_name}.csv")}')
    else:
        print(f'\nNo sessions had data available for "{var_name}" -- skipping in the figure.')

#%% ------------------------------------------------------------------
# Plot: 2x2 grid, one panel per behavioral variable, MI distribution per
# (area, label) group
# ----------------------------------------------------------------------
n_vars = len(behavior_vars)
n_rows = int(np.ceil(n_vars / 2))
fig, axes = plt.subplots(n_rows, 2, figsize=(13, 5 * n_rows), squeeze=False, sharey=True)

for i, var_name in enumerate(behavior_vars):
    ax = axes[i // 2, i % 2]
    if var_name not in pooled_by_var:
        ax.set_title(f'{var_name} (no data available)')
        ax.axis('off')
        continue
    df = pooled_by_var[var_name]
    values_by_group = {(a, l): sub['I_pt'].to_numpy()
                        for (a, l), sub in df.groupby(['area', 'label'])}
    bar_by_group(ax, values_by_group, ylabel=f'MI with {var_name} (bits, PT-corrected)')
    ax.set_title(var_name)

# hide any unused trailing subplot (odd number of variables)
if n_vars % 2 == 1:
    axes[-1, -1].axis('off')

fig.suptitle(f'Neuron-behavior mutual information ({"+".join(protocol)}, {calciumversion}, '
             f'{nSessions} sessions)')
fig.tight_layout()
fig.savefig(os.path.join(output_dir, 'behavior_mi_by_area_label.png'), dpi=150, bbox_inches='tight')
print(f'\nSaved figure to {os.path.join(output_dir, "behavior_mi_by_area_label.png")}')
plt.close(fig)
