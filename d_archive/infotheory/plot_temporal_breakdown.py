# -*- coding: utf-8 -*-
"""
Plot the time- (or position-) resolved Pola et al. (2003) pairwise
information breakdown (I_lin, I_sig_sim, I_cor_indep, I_cor_dep, summing
exactly to I_full at every bin), split by brain area and labeling status
(unl / lab).

This is the most expensive analysis in the pipeline (per pair: T bins x
4 mutual-information estimates with surrogate resampling), so results
are CACHED to disk per session (infotheory/cache.py): if a cache file
matching the current session + parameters already exists, it is loaded
directly and only the aggregation/plotting is redone; otherwise the
breakdown is computed and the cache is written. Delete the relevant file
in `cache_dir` (or set `force_recompute = True` below) to force a
recompute, e.g. after changing `n_bins`, the pairs-per-group cap, or the
response window.

Neuron pairs are sampled WITHIN each (area, label) group (both cells of
a pair share area & label), see
infotheory.celldata_utils.sample_pairs_within_groups, so the resulting
per-group breakdown is directly interpretable as "the pairwise breakdown
among V1-unl cells", etc.
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
    compute_tensor_for_session, compute_time_resolved_breakdown, get_trial_labels,
    cache_path, load_or_compute,
)
from infotheory.celldata_utils import (
    get_area_label, ordered_groups, sample_pairs_within_groups, DEFAULT_LABEL_ORDER)
from utils.cellselection_lib import filter_nearlabeled_layer23
from qc_lib import load_qc_pass

#%% ------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
protocol = ['DN']        # keep protocols with the SAME response axis
calciumversion = 'deconv'

params = InfoTheoryParams(
    binning=BinningParams(method='equipopulated', n_bins=4),
    bias=BiasCorrectionParams(panzeri_treves=True, random_state=0),  # shuffle settings unused here
    parallel=ParallelParams(n_jobs=-1, backend='loky', verbose=0),
    stim_var='signal',
    stim_binarize_threshold=0,
    choice_var=None,       # the Pola breakdown is computed against ONE variable at a
                            # time (set to 'lickResponse' and re-run for choice)
    trial_mask_var=None,
)

s_pre, s_post, binsize = -60, 80, 10     # spatial protocols (VR/DM/DN/DP)
t_pre, t_post = -1, 2                    # time-locked protocols (IM/GR/GN)

max_pairs_per_group = 150   # cap on pairs sampled within each (area,label) group, per session
random_state = 0

# n_jobs at the SESSION level (primary parallelism, one worker per session).
# Keep n_jobs_pairs=1 inside each worker when running many sessions in
# parallel to avoid oversubscribing CPUs with nested parallel pools (loky
# inside loky); bump n_jobs_pairs instead if you only have 1-2 sessions.
n_jobs_sessions = -1
n_jobs_pairs = 1

cache_dir = './output/infotheory_cache'
output_dir = './output/infotheory_plots'
os.makedirs(output_dir, exist_ok=True)
force_recompute = False     # set True to ignore existing cache files

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
# Per session: load from cache, or compute the time-resolved breakdown
# for a sample of within-group pairs, then cache it (parallelized across
# sessions; cache lookup/write happens inside each worker, keyed by
# session_id so there's no collision between workers)
# ----------------------------------------------------------------------
def process_session(ses, session_index, nSessions, calciumversion, params, n_jobs_pairs,
                     s_pre, s_post, binsize, t_pre, t_post,
                     max_pairs_per_group, random_state, cache_dir, force_recompute, qc_csv_path):
    print(f'\n=== Session {session_index + 1}/{nSessions}: {ses.session_id} ===')

    # everything that affects the RESULT goes into the cache key -- if
    # you change any of these and re-run, a new cache file is used
    # automatically (old ones are simply left on disk, unused)
    key_params = dict(
        session_id=ses.session_id, protocol=ses.protocol, calciumversion=calciumversion,
        n_bins=params.binning.n_bins, binning_method=params.binning.method,
        stim_var=params.stim_var, trial_mask_var=params.trial_mask_var,
        max_pairs_per_group=max_pairs_per_group, random_state=random_state,
        s_pre=s_pre, s_post=s_post, binsize=binsize, t_pre=t_pre, t_post=t_post,
        # QC file mtime, so the cache busts automatically if run_qc_summary.py
        # is re-run and the pass/fail flags change:
        qc_mtime=os.path.getmtime(qc_csv_path) if qc_csv_path and os.path.exists(qc_csv_path) else None,
    )
    path = cache_path(cache_dir, f'{ses.session_id}_temporal_breakdown', **key_params)

    params_local = InfoTheoryParams(
        binning=params.binning, bias=params.bias,
        parallel=ParallelParams(n_jobs=n_jobs_pairs, backend=params.parallel.backend,
                                 verbose=params.parallel.verbose),
        stim_var=params.stim_var, stim_binarize_threshold=params.stim_binarize_threshold,
        choice_var=params.choice_var, trial_mask_var=params.trial_mask_var,
    )

    def _compute(ses=ses):
        tensor, axis, axis_label = compute_tensor_for_session(
            ses, calciumversion=calciumversion,
            t_pre=t_pre, t_post=t_post,
            s_pre=s_pre, s_post=s_post, binsize=binsize, keepraw=False)

        stim, _, mask = get_trial_labels(ses, params_local)
        tensor = tensor[mask, :, :]
        stim = stim[mask]

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
        # filtering step, before pairs are sampled below
        print(f'  [{ses.session_id}] {np.sum(idx_valid)}/{len(idx_valid)} neurons kept '
              f'(anatomical: {np.sum(idx_anat)}, QC: {np.sum(idx_qc)}); by group:')
        for a in np.unique(area):
            for l in DEFAULT_LABEL_ORDER:
                n = int(np.sum((area == a) & (label == l)))
                if n > 0:
                    print(f'    {a:>4s} - {l:<4s}: {n}')

        rng = np.random.default_rng(random_state)
        pairs, pair_group = sample_pairs_within_groups(
            area, label, max_pairs_per_group=max_pairs_per_group, rng=rng)

        print(f'  [{ses.session_id}] pairs sampled after filtering:')
        pair_group_arr = np.array(pair_group, dtype=object)
        for a in np.unique(area):
            for l in DEFAULT_LABEL_ORDER:
                n = int(np.sum([g == (a, l) for g in pair_group_arr]))
                if n > 0:
                    print(f'    {a:>4s} - {l:<4s}: {n} pairs')

        df = compute_time_resolved_breakdown(tensor, stim, pairs, params=params_local)
        group_lookup = dict(zip(range(len(pairs)), pair_group))
        df['area'] = df['pair_index'].map(lambda p: group_lookup[p][0])
        df['label'] = df['pair_index'].map(lambda p: group_lookup[p][1])

        return {'df': df, 'axis': axis, 'axis_label': axis_label, 'n_pairs': len(pairs)}

    result = load_or_compute(path, _compute, force_recompute=force_recompute)

    df = result['df'].copy()
    df['session_id'] = ses.session_id
    print(f"  [{ses.session_id}] {result['n_pairs']} pairs, {len(result['axis'])} bins")

    for attr in ('calciumdata', 'videodata', 'behaviordata'):
        if hasattr(ses, attr):
            delattr(ses, attr)

    return {'df': df, 'axis': result['axis'], 'axis_label': result['axis_label']}


session_results = Parallel(n_jobs=n_jobs_sessions, backend='loky', verbose=10)(
    delayed(process_session)(
        ses, ises, nSessions, calciumversion, params, n_jobs_pairs,
        s_pre, s_post, binsize, t_pre, t_post,
        max_pairs_per_group, random_state, cache_dir, force_recompute, qc_csv_path)
    for ises, ses in enumerate(sessions)
)

#%% ------------------------------------------------------------------
# Check all sessions share the same response axis, then pool
# ----------------------------------------------------------------------
reference_axis = session_results[0]['axis']
reference_axis_label = session_results[0]['axis_label']
for res, ses in zip(session_results, sessions):
    assert res['axis_label'] == reference_axis_label and len(res['axis']) == len(reference_axis), (
        f"Session {ses.session_id}'s response axis doesn't match previous "
        f"sessions -- run time-locked and spatial protocols separately.")

df_all = pd.concat([res['df'] for res in session_results], ignore_index=True)
df_all.to_csv(os.path.join(output_dir, 'temporal_breakdown_all_pairs.csv'), index=False)

#%% ------------------------------------------------------------------
# Aggregate: mean breakdown term per (area, label, bin) across all pairs
# and sessions (linear averaging preserves the exact I_full = sum(terms)
# identity at the group level too)
# ----------------------------------------------------------------------
term_cols = ['I_full', 'I_lin', 'I_sig_sim', 'I_cor_indep', 'I_cor_dep']
df_agg = (df_all.groupby(['area', 'label', 'bin_index'])[term_cols]
          .mean().reset_index())
n_pairs_series = (df_all.groupby(['area', 'label', 'bin_index'])['pair_index']
                   .nunique().rename('n_pairs_pooled').reset_index())
df_agg = df_agg.merge(n_pairs_series, on=['area', 'label', 'bin_index'], how='left')

df_agg.to_csv(os.path.join(output_dir, 'temporal_breakdown_summary.csv'), index=False)

#%% ------------------------------------------------------------------
# Plot: grid of (area x label), each subplot showing the 4 breakdown
# terms plus I_full as a dashed reference
# ----------------------------------------------------------------------
TERM_COLORS = {
    'I_lin': 'tab:blue',
    'I_sig_sim': 'tab:orange',
    'I_cor_indep': 'tab:green',
    'I_cor_dep': 'tab:red',
}

areas_present = sorted(df_agg['area'].unique())
pairs_grid = ordered_groups(areas_present)
n_areas = len(areas_present)
n_labels = len(DEFAULT_LABEL_ORDER)

fig, axes = plt.subplots(n_areas, n_labels, figsize=(5.5 * n_labels, 3.2 * n_areas),
                          squeeze=False, sharex=True, sharey=True)

for a, l in pairs_grid:
    sub = df_agg[(df_agg.area == a) & (df_agg.label == l)].sort_values('bin_index')
    if len(sub) == 0:
        continue
    i = areas_present.index(a)
    j = DEFAULT_LABEL_ORDER.index(l)
    ax = axes[i, j]

    x = reference_axis[sub['bin_index'].to_numpy()]
    for term, color in TERM_COLORS.items():
        ax.plot(x, sub[term], color=color, label=term)
    ax.plot(x, sub['I_full'], color='k', linestyle='--', linewidth=1.5, label='I_full (sum)')
    ax.axhline(0, color='gray', linewidth=0.5)

    if 'Time' in reference_axis_label:
        ax.axvline(0, color='gray', linestyle=':', linewidth=1)

    n_pairs_here = df_all[(df_all.area == a) & (df_all.label == l)]['pair_index'].nunique()
    ax.set_title(f'{a} - {l}  (n_pairs~{n_pairs_here})', fontsize=9)
    ax.set_xlabel(reference_axis_label)
    ax.set_ylabel('Information (bits)')

handles, labels_ = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels_, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.04), fontsize=9)
fig.suptitle(f'Pairwise information breakdown over time/position '
             f'({"+".join(protocol)}, {calciumversion}, {nSessions} sessions)', y=1.08, fontsize=12)
fig.tight_layout()

fig_path = os.path.join(output_dir, 'temporal_breakdown_by_area_label.png')
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f'\nSaved figure to {fig_path}')
print(f'Saved aggregated table to {os.path.join(output_dir, "temporal_breakdown_summary.csv")}')
plt.close(fig)
