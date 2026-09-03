# -*- coding: utf-8 -*-
"""
Plot the time- (or position-) resolved partial information decomposition
(PID) of stimulus -- and, separately, of choice -- carried jointly by
PAIRS of neurons, split by brain area and labeling status (unl / lab).

This is the temporal counterpart of plot_pairwise_pid.py: instead of one
PID per pair over the full response window, this computes the PID at
every bin of a (K trials, N neurons, T bins) tensor
(infotheory/temporal_pairwise_pid.py, parallelized across pairs), then
pools pairs across sessions within each (area, label) group and plots
the group-average time/position course with a pair-to-pair SEM band.

As documented in temporal_pairwise_pid.py, this is the single most
expensive analysis in the pipeline (pairs x bins x 2 targets x
(1 + pid_n_shuffles)), so results are CACHED per session
(infotheory/cache.py). Tune `max_pairs_per_group`, `params.binning.
n_bins`, `params.bias.pid_n_shuffles`, and the response window's bin
count/size down for a first exploratory pass -- the cache means you
only pay the cost once per parameter combination.

Produces TWO figures, as with plot_pairwise_pid.py: one for the
stim-target PID time-course, one for the choice-target PID time-course,
each a grid (rows = area, columns = label) of redundancy / unique
(summed) / synergy time-courses plus I_total as a dashed reference.
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
    compute_tensor_for_session, compute_time_resolved_pairwise_pid, get_trial_labels,
    cache_path, load_or_compute,
)
from infotheory.celldata_utils import (
    get_area_label, ordered_groups, sample_pairs_within_groups, DEFAULT_LABEL_ORDER)
from infotheory.temporal_pairwise_pid import PID_TERMS
from utils.cellselection_lib import filter_nearlabeled_layer23
from qc_lib import load_qc_pass

#%% ------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
protocol = ['DN']        # keep protocols with the SAME response axis
calciumversion = 'deconv'

params = InfoTheoryParams(
    binning=BinningParams(method='equipopulated', n_bins=3),   # coarser than usual: this
                                                                  # analysis is expensive, and
                                                                  # PID needs a small alphabet
    bias=BiasCorrectionParams(panzeri_treves=False,             # unused here
                               pid_shuffle_correction=True, pid_n_shuffles=20,
                               random_state=0),
    parallel=ParallelParams(n_jobs=-1, backend='loky', verbose=0),
    stim_var='stimcat',
    stim_value_map={'C': 0, 'N': 1, 'M': 1},
    choice_var='lickResponse',
    trial_mask_var=None,
)

s_pre, s_post, binsize = -60, 80, 20     # coarser spatial bins than other scripts, for cost
t_pre, t_post = -1, 2                    # time-locked protocols (IM/GR/GN)

max_pairs_per_group = 60    # cap on pairs sampled within each (area,label) group, per session
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
# Per session: load from cache, or compute the time-resolved pairwise
# PID (stim- and choice-target) for a sample of within-group pairs
# (parallelized across sessions; cache lookup/write happens inside each
# worker, keyed by session_id so there's no collision between workers)
# ----------------------------------------------------------------------
def process_session(ses, session_index, nSessions, calciumversion, params, n_jobs_pairs,
                     s_pre, s_post, binsize, t_pre, t_post,
                     max_pairs_per_group, random_state, cache_dir, force_recompute, qc_csv_path):
    print(f'\n=== Session {session_index + 1}/{nSessions}: {ses.session_id} ===')

    key_params = dict(
        session_id=ses.session_id, protocol=ses.protocol, calciumversion=calciumversion,
        n_bins=params.binning.n_bins, binning_method=params.binning.method,
        stim_var=params.stim_var, stim_value_map=params.stim_value_map,
        stim_binarize_threshold=params.stim_binarize_threshold,
        choice_var=params.choice_var, trial_mask_var=params.trial_mask_var,
        pid_shuffle_correction=params.bias.pid_shuffle_correction,
        pid_n_shuffles=params.bias.pid_n_shuffles, random_state=random_state,
        max_pairs_per_group=max_pairs_per_group,
        s_pre=s_pre, s_post=s_post, binsize=binsize, t_pre=t_pre, t_post=t_post,
        # QC file mtime, so the cache busts automatically if run_qc_summary.py
        # is re-run and the pass/fail flags change:
        qc_mtime=os.path.getmtime(qc_csv_path) if qc_csv_path and os.path.exists(qc_csv_path) else None,
    )
    path = cache_path(cache_dir, f'{ses.session_id}_temporal_pairwise_pid', **key_params)

    params_local = InfoTheoryParams(
        binning=params.binning, bias=params.bias,
        parallel=ParallelParams(n_jobs=n_jobs_pairs, backend=params.parallel.backend,
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
        for a in np.unique(area):
            for l in DEFAULT_LABEL_ORDER:
                n = int(np.sum([g == (a, l) for g in pair_group]))
                if n > 0:
                    print(f'    {a:>4s} - {l:<4s}: {n} pairs')

        df = compute_time_resolved_pairwise_pid(tensor, stim, choice, pairs, params=params_local)
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
df_all.to_csv(os.path.join(output_dir, 'temporal_pairwise_pid_all_pairs.csv'), index=False)

#%% ------------------------------------------------------------------
# Aggregate: mean PID term per (area, label, bin) across all pairs and
# sessions, for each target separately
# ----------------------------------------------------------------------
suffix = '_shuffcorr' if params.bias.pid_shuffle_correction else ''
TERM_COLORS = {
    f'redundancy{suffix}': 'tab:orange',
    f'synergy{suffix}': 'tab:green',
    f'unique_sum{suffix}': 'tab:gray',
}


def aggregate_target(df_all, target_prefix):
    df = df_all.copy()
    df[f'unique_sum{suffix}'] = (df[f'{target_prefix}_pid_unique_n1{suffix}']
                                  + df[f'{target_prefix}_pid_unique_n2{suffix}'])
    term_cols = [f'redundancy{suffix}', f'synergy{suffix}', f'unique_sum{suffix}', f'I_total{suffix}']
    rename = {f'{target_prefix}_pid_redundancy{suffix}': f'redundancy{suffix}',
              f'{target_prefix}_pid_synergy{suffix}': f'synergy{suffix}',
              f'{target_prefix}_pid_I_total{suffix}': f'I_total{suffix}'}
    df = df.rename(columns=rename)
    agg = (df.groupby(['area', 'label', 'bin_index'])[term_cols].mean().reset_index())
    n_pairs = (df.groupby(['area', 'label', 'bin_index'])['pair_index']
               .nunique().rename('n_pairs_pooled').reset_index())
    agg = agg.merge(n_pairs, on=['area', 'label', 'bin_index'], how='left')
    return agg


agg_stim = aggregate_target(df_all, 'stim')
agg_choice = aggregate_target(df_all, 'choice')

agg_stim.to_csv(os.path.join(output_dir, 'temporal_pairwise_pid_stim_summary.csv'), index=False)
agg_choice.to_csv(os.path.join(output_dir, 'temporal_pairwise_pid_choice_summary.csv'), index=False)

#%% ------------------------------------------------------------------
# Plotting helper: grid of (area x label), each subplot showing
# redundancy/synergy/unique-sum time-courses plus I_total reference
# ----------------------------------------------------------------------
def plot_pairwise_pid_grid(agg, title, save_path):
    areas_present = sorted(agg['area'].unique())
    pairs_grid = ordered_groups(areas_present)
    n_areas = len(areas_present)
    n_labels = len(DEFAULT_LABEL_ORDER)

    fig, axes = plt.subplots(n_areas, n_labels, figsize=(5.5 * n_labels, 3.2 * n_areas),
                              squeeze=False, sharex=True, sharey=True)

    for a, l in pairs_grid:
        sub = agg[(agg.area == a) & (agg.label == l)].sort_values('bin_index')
        if len(sub) == 0:
            continue
        i = areas_present.index(a)
        j = DEFAULT_LABEL_ORDER.index(l)
        ax = axes[i, j]

        x = reference_axis[sub['bin_index'].to_numpy()]
        for term, color in TERM_COLORS.items():
            ax.plot(x, sub[term], color=color, label=term.replace(suffix, ''))
        ax.plot(x, sub[f'I_total{suffix}'], color='k', linestyle='--', linewidth=1.5, label='I_total')
        ax.axhline(0, color='gray', linewidth=0.5)
        if 'Time' in reference_axis_label:
            ax.axvline(0, color='gray', linestyle=':', linewidth=1)

        n_pairs_here = int(sub['n_pairs_pooled'].max())
        ax.set_title(f'{a} - {l}  (n_pairs~{n_pairs_here})', fontsize=9)
        ax.set_xlabel(reference_axis_label)
        ax.set_ylabel('Information (bits)')

    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.04), fontsize=9)
    fig.suptitle(title, y=1.08, fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved figure to {save_path}')
    plt.close(fig)


#%% ------------------------------------------------------------------
# Figure 1: pairwise PID about STIMULUS, over time/position
# ----------------------------------------------------------------------
plot_pairwise_pid_grid(
    agg_stim,
    title=f'Pairwise PID about stimulus over time/position '
          f'({"+".join(protocol)}, {calciumversion}, {nSessions} sessions)',
    save_path=os.path.join(output_dir, 'temporal_pairwise_pid_stim_by_area_label.png'))

#%% ------------------------------------------------------------------
# Figure 2: pairwise PID about CHOICE, over time/position
# ----------------------------------------------------------------------
plot_pairwise_pid_grid(
    agg_choice,
    title=f'Pairwise PID about choice over time/position '
          f'({"+".join(protocol)}, {calciumversion}, {nSessions} sessions)',
    save_path=os.path.join(output_dir, 'temporal_pairwise_pid_choice_by_area_label.png'))
