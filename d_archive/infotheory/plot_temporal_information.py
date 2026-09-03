# -*- coding: utf-8 -*-
"""
Plot time- (or, for VR/detection protocols, position-) resolved
single-cell information about stimulus and choice, split by brain area
and labeling status (unl / lab).

For each session, builds a (K trials, N neurons, T bins) tensor
(protocol-aware: time-locked for IM/GR/GN, spatial for VR/DM/DN/DP, see
infotheory/tensor_utils.py), computes PT-corrected MI(stim) and
MI(choice) at every bin for every neuron (infotheory/temporal.py,
parallelized across neurons), then pools neurons across sessions within
each (area, label) group and plots the group-average time/position
course with a neuron-to-neuron SEM band.

Two figures are saved: one for stimulus information, one for choice
information (if `choice_var` is configured).
"""

#%%
import os, sys
os.chdir('/u/g/glorenz/Documents/Research/Code/external_repos/oudelohuis-vr-detection')
sys.path.insert(0, '/u/g/glorenz/Documents/Research/Code/external_repos/oudelohuis-vr-detection')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from loaddata.session_info import filter_sessions, report_sessions
from infotheory import (
    InfoTheoryParams, BinningParams, BiasCorrectionParams, ParallelParams,
    compute_tensor_for_session, compute_time_resolved_information, get_trial_labels,
)
from infotheory.celldata_utils import (
    get_area_label, ordered_groups, AREA_COLORS, LABEL_LINESTYLES, DEFAULT_LABEL_ORDER)
from utils.cellselection_lib import filter_nearlabeled_layer23
from qc_lib import load_qc_pass

#%% ------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
protocol = ['DN']       # keep protocols with the SAME response axis
                                     # (all spatial here); don't mix spatial and
                                     # time-locked protocols in one run
calciumversion = 'deconv'

params = InfoTheoryParams(
    binning=BinningParams(method='equipopulated', n_bins=4),
    bias=BiasCorrectionParams(panzeri_treves=True),  # shuffle settings unused here
    parallel=ParallelParams(n_jobs=-1, backend='loky', verbose=0),
    stim_var='signal',
    stim_binarize_threshold=0,
    choice_var='lickResponse',
    trial_mask_var=None,
)

# spatial-protocol tensor window (only used for VR/DM/DN/DP sessions)
s_pre, s_post, binsize = -60, 80, 10
# time-locked tensor window (only used for IM/GR/GN sessions)
t_pre, t_post = -1, 2

max_neurons_per_group_per_session = None   # subsample cap for speed; None = no cap
random_state = 0

# n_jobs at the SESSION level (primary parallelism, one worker per session).
# Keep n_jobs_neurons=1 inside each worker when running many sessions in
# parallel to avoid oversubscribing CPUs with nested parallel pools; bump
# n_jobs_neurons instead if you only have 1-2 sessions.
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
# Per session: compute tensor + time/space-resolved MI, then accumulate
# per (area, label) group (parallelized across sessions)
# ----------------------------------------------------------------------
def process_session(ses, session_index, nSessions, calciumversion, params, n_jobs_neurons,
                     t_pre, t_post, s_pre, s_post, binsize,
                     max_neurons_per_group_per_session, random_state, qc_csv_path):
    print(f'\n=== Session {session_index + 1}/{nSessions}: {ses.session_id} ===')
    rng_local = np.random.default_rng(random_state + session_index)

    tensor, axis, axis_label = compute_tensor_for_session(
        ses, calciumversion=calciumversion,
        t_pre=t_pre, t_post=t_post,
        s_pre=s_pre, s_post=s_post, binsize=binsize, keepraw=False)

    params_local = InfoTheoryParams(
        binning=params.binning, bias=params.bias,
        parallel=ParallelParams(n_jobs=n_jobs_neurons, backend=params.parallel.backend,
                                 verbose=params.parallel.verbose),
        stim_var=params.stim_var, stim_binarize_threshold=params.stim_binarize_threshold,
        choice_var=params.choice_var, trial_mask_var=params.trial_mask_var,
    )

    stim, choice, mask = get_trial_labels(ses, params_local)
    tensor = tensor[mask, :, :]
    stim = stim[mask]
    choice = choice[mask] if choice is not None else None

    area, label, _ = get_area_label(ses.celldata)

    # V1/PM unl cells must be near a labeled cell (50um) and in layer 2/3
    # (depth<300um); labeled cells and cells outside V1/PM pass through
    # unaffected (see utils/cellselection_lib.py). Combined with the QC
    # pass/fail flag from run_qc_summary.py (silent/artifact/noisy cells).
    idx_anat = filter_nearlabeled_layer23(ses, radius=50, depth_thr=300, lateral_only=True)
    idx_qc = load_qc_pass(ses, qc_csv_path=qc_csv_path)
    idx_valid = idx_anat & idx_qc
    tensor = tensor[:, idx_valid, :]
    area, label = area[idx_valid], label[idx_valid]

    # report how many neurons per (area, label) group survive the
    # filtering step
    print(f'  [{ses.session_id}] {np.sum(idx_valid)}/{len(idx_valid)} neurons kept '
          f'(anatomical: {np.sum(idx_anat)}, QC: {np.sum(idx_qc)}); by group:')
    for a in np.unique(area):
        for l in DEFAULT_LABEL_ORDER:
            n = int(np.sum((area == a) & (label == l)))
            if n > 0:
                print(f'    {a:>4s} - {l:<4s}: {n}')

    stim_mi, choice_mi = compute_time_resolved_information(
        tensor, stim, choice, params=params_local)   # (N, T) each

    group_stim_local = {}
    group_choice_local = {}
    for a in np.unique(area):
        for l in DEFAULT_LABEL_ORDER:
            idx = np.where((area == a) & (label == l))[0]
            if len(idx) == 0:
                continue
            if max_neurons_per_group_per_session is not None and len(idx) > max_neurons_per_group_per_session:
                idx = rng_local.choice(idx, size=max_neurons_per_group_per_session, replace=False)
            key = (a, l)
            group_stim_local[key] = stim_mi[idx, :]
            if choice_mi is not None:
                group_choice_local[key] = choice_mi[idx, :]

    for attr in ('calciumdata', 'videodata', 'behaviordata'):
        if hasattr(ses, attr):
            delattr(ses, attr)

    return {'axis': axis, 'axis_label': axis_label,
            'group_stim': group_stim_local, 'group_choice': group_choice_local}


session_results = Parallel(n_jobs=n_jobs_sessions, backend='loky', verbose=10)(
    delayed(process_session)(
        ses, ises, nSessions, calciumversion, params, n_jobs_neurons,
        t_pre, t_post, s_pre, s_post, binsize,
        max_neurons_per_group_per_session, random_state, qc_csv_path)
    for ises, ses in enumerate(sessions)
)

#%% ------------------------------------------------------------------
# Check all sessions share the same response axis, then pool per group
# ----------------------------------------------------------------------
reference_axis = session_results[0]['axis']
reference_axis_label = session_results[0]['axis_label']
for res, ses in zip(session_results, sessions):
    assert res['axis_label'] == reference_axis_label and len(res['axis']) == len(reference_axis), (
        f"Session {ses.session_id} has a different response axis "
        f"({res['axis_label']}, {len(res['axis'])} bins) than previous sessions "
        f"({reference_axis_label}, {len(reference_axis)} bins). Run "
        f"time-locked and spatial protocols in separate script runs.")

group_stim_mi = {}
group_choice_mi = {}
for res in session_results:
    for key, mi in res['group_stim'].items():
        group_stim_mi.setdefault(key, []).append(mi)
    for key, mi in res['group_choice'].items():
        group_choice_mi.setdefault(key, []).append(mi)

group_stim_mi = {k: np.concatenate(v, axis=0) for k, v in group_stim_mi.items()}
if group_choice_mi:
    group_choice_mi = {k: np.concatenate(v, axis=0) for k, v in group_choice_mi.items()}

#%% ------------------------------------------------------------------
# Plotting helper
# ----------------------------------------------------------------------
def plot_group_timecourses(group_mi, axis, axis_label, title, save_path):
    fig, ax = plt.subplots(figsize=(7, 5))
    areas_present = sorted(set(a for a, l in group_mi.keys()))
    pairs = ordered_groups(areas_present)

    for a, l in pairs:
        if (a, l) not in group_mi:
            continue
        mi = group_mi[(a, l)]   # (n_neurons, T)
        n = mi.shape[0]
        mean = np.nanmean(mi, axis=0)
        sem = np.nanstd(mi, axis=0) / np.sqrt(np.maximum(n, 1))

        color = AREA_COLORS.get(a, 'gray')
        ls = LABEL_LINESTYLES.get(l, '-')
        ax.plot(axis, mean, color=color, linestyle=ls, label=f'{a} ({l}, n={n})')
        ax.fill_between(axis, mean - sem, mean + sem, color=color, alpha=0.15)

    if 'Time' in axis_label:
        ax.axvline(0, color='k', linestyle=':', linewidth=1)
    ax.set_xlabel(axis_label)
    ax.set_ylabel('Mutual information (bits, PT-corrected)')
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved figure to {save_path}')
    plt.close(fig)


#%% ------------------------------------------------------------------
# Stimulus information time/position course
# ----------------------------------------------------------------------
plot_group_timecourses(
    group_stim_mi, reference_axis, reference_axis_label,
    title=f'Stimulus information over time/position ({"+".join(protocol)}, {calciumversion})',
    save_path=os.path.join(output_dir, 'stim_information_timecourse_by_area_label.png'))

#%% ------------------------------------------------------------------
# Choice information time/position course (if configured)
# ----------------------------------------------------------------------
if group_choice_mi:
    plot_group_timecourses(
        group_choice_mi, reference_axis, reference_axis_label,
        title=f'Choice information over time/position ({"+".join(protocol)}, {calciumversion})',
        save_path=os.path.join(output_dir, 'choice_information_timecourse_by_area_label.png'))

#%% ------------------------------------------------------------------
# Also save the underlying per-neuron time-courses (long format) for
# further analysis/re-plotting without recomputation
# ----------------------------------------------------------------------
rows = []
for (a, l), mi in group_stim_mi.items():
    for n in range(mi.shape[0]):
        rows.append({'area': a, 'label': l, 'neuron_in_group_idx': n,
                      'variable': 'stim', **{f'bin_{i}': v for i, v in enumerate(mi[n, :])}})
if group_choice_mi:
    for (a, l), mi in group_choice_mi.items():
        for n in range(mi.shape[0]):
            rows.append({'area': a, 'label': l, 'neuron_in_group_idx': n,
                         'variable': 'choice', **{f'bin_{i}': v for i, v in enumerate(mi[n, :])}})

df_long = pd.DataFrame(rows)
df_long.to_csv(os.path.join(output_dir, 'timeresolved_information_per_neuron.csv'), index=False)
pd.Series(reference_axis, name='bin_center').to_csv(
    os.path.join(output_dir, 'timeresolved_information_bin_axis.csv'), index=False)
print(f'\nSaved per-neuron time-courses to '
      f'{os.path.join(output_dir, "timeresolved_information_per_neuron.csv")}')
