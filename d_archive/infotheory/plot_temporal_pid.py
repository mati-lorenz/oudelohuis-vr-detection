# -*- coding: utf-8 -*-
"""
Plot the time- (or position-) resolved partial information decomposition
(PID) of stimulus and choice information carried by single neurons,
split by brain area and labeling status (unl / lab).

For each neuron, at every time/space bin, decomposes I(response;
stim, choice) into redundant / unique-to-stim / unique-to-choice /
synergistic components (infotheory/temporal_pid.py, parallelized across
neurons), then pools neurons across sessions within each (area, label)
group and plots the group-average time/position course with a
neuron-to-neuron SEM band.

Like plot_temporal_breakdown.py, results are CACHED to disk per session
(infotheory/cache.py) since this is a relatively expensive analysis
(shuffle-subtraction bias correction, on by default, multiplies the
cost by params.bias.pid_n_shuffles + 1): if a cache file matching the
current session + parameters already exists, it is loaded directly;
otherwise the PID time-course is computed and cached. Delete the
relevant file in `cache_dir` (or set `force_recompute = True`) to force
a recompute.
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
    compute_tensor_for_session, compute_time_resolved_pid, get_trial_labels,
    cache_path, load_or_compute,
)
from infotheory.celldata_utils import (
    get_area_label, ordered_groups, DEFAULT_LABEL_ORDER)
from infotheory.temporal_pid import PID_TERMS
from utils.cellselection_lib import filter_nearlabeled_layer23
from qc_lib import load_qc_pass

#%% ------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
protocol = ['DN']       # keep protocols with the SAME response axis
calciumversion = 'deconv'

params = InfoTheoryParams(
    binning=BinningParams(method='equipopulated', n_bins=4),
    bias=BiasCorrectionParams(panzeri_treves=False,          # unused here
                               pid_shuffle_correction=True, pid_n_shuffles=30,
                               random_state=0),
    parallel=ParallelParams(n_jobs=-1, backend='loky', verbose=0),
    stim_var='stimcat',
    # collapse the 3-category 'stimcat' (C/N/M) into a binary "signal
    # present or not" variable -- see params.py for the alternative
    # stim_binarize_threshold route using the continuous 'signal' column
    stim_value_map={'C': 0, 'N': 1, 'M': 1},
    choice_var='lickResponse',
    trial_mask_var=None,
)

s_pre, s_post, binsize = -60, 80, 10     # spatial protocols (VR/DM/DN/DP)
t_pre, t_post = -1, 2                    # time-locked protocols (IM/GR/GN)

max_neurons_per_group_per_session = 150   # subsample cap for speed; None = no cap
random_state = 0

# n_jobs at the SESSION level (primary parallelism, one worker per session).
# Keep n_jobs_neurons=1 inside each worker when running many sessions in
# parallel to avoid oversubscribing CPUs with nested parallel pools (loky
# inside loky); bump n_jobs_neurons instead if you only have 1-2 sessions.
n_jobs_sessions = -1
n_jobs_neurons = 1

cache_dir = './output/infotheory_cache'
output_dir = './output/infotheory_plots'
os.makedirs(output_dir, exist_ok=True)
force_recompute = False

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
# Per session: load from cache, or compute the time-resolved PID for a
# (subsampled) set of neurons per (area, label) group, then cache it
# (parallelized across sessions; cache lookup/write happens inside each
# worker, keyed by session_id so there's no collision between workers)
# ----------------------------------------------------------------------
term_keys_all = list(PID_TERMS) + [f'{k}_shuffcorr' for k in PID_TERMS] \
    if params.bias.pid_shuffle_correction else list(PID_TERMS)


def process_session(ses, session_index, nSessions, calciumversion, params, n_jobs_neurons,
                     s_pre, s_post, binsize, t_pre, t_post,
                     max_neurons_per_group_per_session, random_state, cache_dir, force_recompute,
                     qc_csv_path):
    print(f'\n=== Session {session_index + 1}/{nSessions}: {ses.session_id} ===')

    key_params = dict(
        session_id=ses.session_id, protocol=ses.protocol, calciumversion=calciumversion,
        n_bins=params.binning.n_bins, binning_method=params.binning.method,
        stim_var=params.stim_var, stim_value_map=params.stim_value_map,
        stim_binarize_threshold=params.stim_binarize_threshold,
        choice_var=params.choice_var, trial_mask_var=params.trial_mask_var,
        pid_shuffle_correction=params.bias.pid_shuffle_correction,
        pid_n_shuffles=params.bias.pid_n_shuffles, random_state=random_state,
        max_neurons_per_group=max_neurons_per_group_per_session,
        s_pre=s_pre, s_post=s_post, binsize=binsize, t_pre=t_pre, t_post=t_post,
        # QC file mtime, so the cache busts automatically if run_qc_summary.py
        # is re-run and the pass/fail flags change:
        qc_mtime=os.path.getmtime(qc_csv_path) if qc_csv_path and os.path.exists(qc_csv_path) else None,
    )
    path = cache_path(cache_dir, f'{ses.session_id}_temporal_pid', **key_params)

    params_local = InfoTheoryParams(
        binning=params.binning, bias=params.bias,
        parallel=ParallelParams(n_jobs=n_jobs_neurons, backend=params.parallel.backend,
                                 verbose=params.parallel.verbose),
        stim_var=params.stim_var, stim_value_map=params.stim_value_map,
        stim_binarize_threshold=params.stim_binarize_threshold,
        choice_var=params.choice_var, trial_mask_var=params.trial_mask_var,
    )

    def _compute(ses=ses):
        tensor, axis, axis_label = compute_tensor_for_session(
            ses, calciumversion=calciumversion,
            t_pre=t_pre, t_post=t_post,
            s_pre=s_pre, s_post=s_post, binsize=binsize, keepraw=False)

        stim, choice, mask = get_trial_labels(ses, params_local)
        tensor = tensor[mask, :, :]
        stim = stim[mask]
        choice = choice[mask]

        area, label, _ = get_area_label(ses.celldata)

        # V1/PM unl cells must be near a labeled cell (50um) and in layer 2/3
        # (depth<300um); labeled cells and cells outside V1/PM pass through
        # unaffected (see utils/cellselection_lib.py). Combined with the QC
        # pass/fail flag from run_qc_summary.py (silent/artifact/noisy cells).
        idx_anat = filter_nearlabeled_layer23(ses, radius=50, depth_thr=300)
        idx_qc = load_qc_pass(ses, qc_csv_path=qc_csv_path)
        idx_valid = idx_anat & idx_qc
        tensor = tensor[:, idx_valid, :]
        area, label = area[idx_valid], label[idx_valid]

        # report how many neurons per (area, label) group survive the
        # filtering step, before any further subsampling below
        print(f'  [{ses.session_id}] {np.sum(idx_valid)}/{len(idx_valid)} neurons kept '
              f'(anatomical: {np.sum(idx_anat)}, QC: {np.sum(idx_qc)}); by group:')
        for a in np.unique(area):
            for l in DEFAULT_LABEL_ORDER:
                n = int(np.sum((area == a) & (label == l)))
                if n > 0:
                    print(f'    {a:>4s} - {l:<4s}: {n}')

        rng = np.random.default_rng(random_state)

        # subsample neurons per (area,label) group for speed, same
        # pattern as plot_temporal_information.py
        keep_idx = []
        keep_group = []
        for a in np.unique(area):
            for l in DEFAULT_LABEL_ORDER:
                idx = np.where((area == a) & (label == l))[0]
                if len(idx) == 0:
                    continue
                if (max_neurons_per_group_per_session is not None
                        and len(idx) > max_neurons_per_group_per_session):
                    idx = rng.choice(idx, size=max_neurons_per_group_per_session, replace=False)
                keep_idx.append(idx)
                keep_group.extend([(a, l)] * len(idx))
        keep_idx = np.concatenate(keep_idx) if keep_idx else np.array([], dtype=int)

        tensor_sub = tensor[:, keep_idx, :]
        pid_out = compute_time_resolved_pid(tensor_sub, stim, choice, params=params_local)

        return {'pid_out': pid_out, 'axis': axis, 'axis_label': axis_label,
                'group': keep_group, 'n_neurons': len(keep_idx)}

    result = load_or_compute(path, _compute, force_recompute=force_recompute)
    print(f"  [{ses.session_id}] {result['n_neurons']} neurons (post-subsample), "
          f"{len(result['axis'])} bins")

    for attr in ('calciumdata', 'videodata', 'behaviordata'):
        if hasattr(ses, attr):
            delattr(ses, attr)

    return result


session_results = Parallel(n_jobs=n_jobs_sessions, backend='loky', verbose=10)(
    delayed(process_session)(
        ses, ises, nSessions, calciumversion, params, n_jobs_neurons,
        s_pre, s_post, binsize, t_pre, t_post,
        max_neurons_per_group_per_session, random_state, cache_dir, force_recompute, qc_csv_path)
    for ises, ses in enumerate(sessions)
)

#%% ------------------------------------------------------------------
# Check all sessions share the same response axis, then pool per group
# ----------------------------------------------------------------------
reference_axis = session_results[0]['axis']
reference_axis_label = session_results[0]['axis_label']
for res, ses in zip(session_results, sessions):
    assert res['axis_label'] == reference_axis_label and len(res['axis']) == len(reference_axis), (
        f"Session {ses.session_id}'s response axis doesn't match previous "
        f"sessions -- run time-locked and spatial protocols separately.")

group_terms = {}   # (area, label) -> {term_name: list of (n_neurons, T) arrays}
for result in session_results:
    for a in {g[0] for g in result['group']}:
        for l in DEFAULT_LABEL_ORDER:
            sel = np.array([g == (a, l) for g in result['group']], dtype=bool)
            if not np.any(sel):
                continue
            key = (a, l)
            for term in term_keys_all:
                group_terms.setdefault(key, {}).setdefault(term, []).append(
                    result['pid_out'][term][sel, :])

group_terms = {key: {term: np.concatenate(v, axis=0) for term, v in terms.items()}
               for key, terms in group_terms.items()}

print('\nTotal pooled neurons per (area, label) group:')
for (a, l), terms in sorted(group_terms.items()):
    print(f'  {a:>4s} - {l:<4s}: {terms[term_keys_all[0]].shape[0]}')

#%% ------------------------------------------------------------------
# Save per-neuron long-format table
# ----------------------------------------------------------------------
rows = []
for (a, l), terms in group_terms.items():
    n_neurons = terms[term_keys_all[0]].shape[0]
    for n in range(n_neurons):
        row = {'area': a, 'label': l, 'neuron_in_group_idx': n}
        row.update({f'bin_{i}_{term}': terms[term][n, i]
                    for term in term_keys_all for i in range(len(reference_axis))})
        rows.append(row)
# (kept compact: full per-bin CSV can get wide; also save a tidy long version)
tidy_rows = []
for (a, l), terms in group_terms.items():
    n_neurons = terms[term_keys_all[0]].shape[0]
    for n in range(n_neurons):
        for i in range(len(reference_axis)):
            tidy_rows.append({
                'area': a, 'label': l, 'neuron_in_group_idx': n, 'bin_index': i,
                **{term: terms[term][n, i] for term in term_keys_all},
            })
df_tidy = pd.DataFrame(tidy_rows)
df_tidy.to_csv(os.path.join(output_dir, 'temporal_pid_per_neuron.csv'), index=False)
pd.Series(reference_axis, name='bin_center').to_csv(
    os.path.join(output_dir, 'temporal_pid_bin_axis.csv'), index=False)
print(f'\nSaved per-neuron PID time-courses to '
      f'{os.path.join(output_dir, "temporal_pid_per_neuron.csv")}')

#%% ------------------------------------------------------------------
# Plot: grid of (area x label), each subplot showing the 4 PID
# components plus I_total as a dashed reference
# ----------------------------------------------------------------------
suffix = '_shuffcorr' if params.bias.pid_shuffle_correction else ''
TERM_COLORS = {
    f'unique_stim{suffix}': 'tab:blue',
    f'unique_choice{suffix}': 'tab:purple',
    f'redundancy{suffix}': 'tab:orange',
    f'synergy{suffix}': 'tab:green',
}

areas_present = sorted({a for a, l in group_terms.keys()})
pairs_grid = ordered_groups(areas_present)
n_areas = len(areas_present)
n_labels = len(DEFAULT_LABEL_ORDER)

fig, axes = plt.subplots(n_areas, n_labels, figsize=(5.5 * n_labels, 3.2 * n_areas),
                          squeeze=False, sharex=True, sharey=False)

for a, l in pairs_grid:
    if (a, l) not in group_terms:
        continue
    i = areas_present.index(a)
    j = DEFAULT_LABEL_ORDER.index(l)
    ax = axes[i, j]
    terms = group_terms[(a, l)]
    n_neurons = terms[term_keys_all[0]].shape[0]

    for term, color in TERM_COLORS.items():
        mean = np.nanmean(terms[term], axis=0)
        sem = np.nanstd(terms[term], axis=0) / np.sqrt(max(n_neurons, 1))
        ax.plot(reference_axis, mean, color=color, label=term.replace(suffix, ''))
        ax.fill_between(reference_axis, mean - sem, mean + sem, color=color, alpha=0.15)

    i_total_key = f'I_total{suffix}'
    mean_total = np.nanmean(terms[i_total_key], axis=0)
    ax.plot(reference_axis, mean_total, color='k', linestyle='--', linewidth=1.5, label='I_total')
    ax.axhline(0, color='gray', linewidth=0.5)
    if 'Time' in reference_axis_label:
        ax.axvline(0, color='gray', linestyle=':', linewidth=1)

    ax.set_title(f'{a} - {l}  (n={n_neurons})', fontsize=9)
    ax.set_xlabel(reference_axis_label)
    ax.set_ylabel('Information (bits)')

handles, labels_ = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels_, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.04), fontsize=9)
fig.suptitle(f'Stimulus/choice PID over time/position '
             f'({"+".join(protocol)}, {calciumversion}, {nSessions} sessions)', y=1.08, fontsize=12)
fig.tight_layout()

fig_path = os.path.join(output_dir, 'temporal_pid_by_area_label.png')
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f'Saved figure to {fig_path}')
plt.close(fig)
