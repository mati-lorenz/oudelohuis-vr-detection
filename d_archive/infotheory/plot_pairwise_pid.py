# -*- coding: utf-8 -*-
"""
Compute the partial information decomposition (PID) of STIMULUS and,
separately, of CHOICE, carried jointly by pairs of neurons -- i.e. here
the two neurons in a pair are the PID SOURCES and the task variable
(stim or choice) is the TARGET, the mirror image of the single-cell PID
in run_singlecell_infotheory.py (where the neuron is the target and
stim/choice are the sources). This is the "N=2 PID" analysis in Lorenz
et al. (2025)'s Fig 2 (their Syn/Red bar panels for neuron pairs about
a task variable).

Produces TWO figures (as requested): one summarizing the stim-target
PID, one summarizing the choice-target PID, each a grouped bar chart of
mean +/- SEM redundancy / synergy / unique information (as % of the
pair's total joint information about that target) per (area, label)
group. Neuron pairs are sampled WITHIN each (area, label) group (see
infotheory.celldata_utils.sample_pairs_within_groups), same convention
as plot_temporal_breakdown.py.

Results are cached per session (infotheory/cache.py) since pairwise PID
with shuffle-subtraction bias correction for two targets is a
non-trivial amount of computation; delete the relevant file in
`cache_dir` (or set `force_recompute = True`) to force a recompute.
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
    compute_respmat_for_session, compute_pairwise_pid, get_trial_labels,
    cache_path, load_or_compute,
)
from infotheory.celldata_utils import (
    get_area_label, ordered_groups, sample_pairs_within_groups,
    AREA_COLORS, DEFAULT_LABEL_ORDER)
from utils.cellselection_lib import filter_nearlabeled_layer23
from qc_lib import load_qc_pass

#%% ------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
protocol = ['DN']
calciumversion = 'deconv'

params = InfoTheoryParams(
    binning=BinningParams(method='equipopulated', n_bins=3),
    bias=BiasCorrectionParams(panzeri_treves=False,   # unused here
                               pid_shuffle_correction=True, pid_n_shuffles=30,
                               random_state=0),
    parallel=ParallelParams(n_jobs=-1, backend='loky', verbose=0),
    stim_var='stimcat',
    stim_value_map={'C': 0, 'N': 1, 'M': 1},   # collapse to binary "signal present" (see params.py)
    choice_var='lickResponse',
    trial_mask_var=None,
)

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
# Per session: load from cache, or compute pairwise PID (stim-target and
# choice-target) for a sample of within-group pairs, then cache it
# (parallelized across sessions; cache lookup/write happens inside each
# worker, keyed by session_id so there's no collision between workers)
# ----------------------------------------------------------------------
def process_session(ses, session_index, nSessions, calciumversion, params, n_jobs_pairs,
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
        # QC file mtime, so the cache busts automatically if run_qc_summary.py
        # is re-run and the pass/fail flags change:
        qc_mtime=os.path.getmtime(qc_csv_path) if qc_csv_path and os.path.exists(qc_csv_path) else None,
    )
    path = cache_path(cache_dir, f'{ses.session_id}_pairwise_pid', **key_params)

    params_local = InfoTheoryParams(
        binning=params.binning, bias=params.bias,
        parallel=ParallelParams(n_jobs=n_jobs_pairs, backend=params.parallel.backend,
                                 verbose=params.parallel.verbose),
        stim_var=params.stim_var, stim_value_map=params.stim_value_map,
        stim_binarize_threshold=params.stim_binarize_threshold,
        choice_var=params.choice_var, trial_mask_var=params.trial_mask_var,
    )

    def _compute(ses=ses):
        respmat = compute_respmat_for_session(ses, calciumversion=calciumversion, keepraw=False)
        stim, choice, mask = get_trial_labels(ses, params_local)
        respmat = np.asarray(respmat)[:, mask]
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
        print(f'  [{ses.session_id}] {np.sum(idx_valid)}/{len(idx_valid)} neurons kept '
              f'(anatomical: {np.sum(idx_anat)}, QC: {np.sum(idx_qc)})')
        respmat = respmat[idx_valid, :]
        area, label = area[idx_valid], label[idx_valid]

        rng = np.random.default_rng(random_state)
        pairs, pair_group = sample_pairs_within_groups(
            area, label, max_pairs_per_group=max_pairs_per_group, rng=rng)

        df = compute_pairwise_pid(respmat, stim, choice, params=params_local, pairs=pairs)
        group_lookup = dict(zip(range(len(pairs)), pair_group))
        df['area'] = df.index.map(lambda p: group_lookup[p][0])
        df['label'] = df.index.map(lambda p: group_lookup[p][1])

        return {'df': df, 'n_pairs': len(pairs)}

    result = load_or_compute(path, _compute, force_recompute=force_recompute)
    df = result['df'].copy()
    df['session_id'] = ses.session_id
    print(f"  [{ses.session_id}] {result['n_pairs']} pairs")

    for attr in ('calciumdata', 'videodata', 'behaviordata'):
        if hasattr(ses, attr):
            delattr(ses, attr)

    return df


session_dfs = Parallel(n_jobs=n_jobs_sessions, backend='loky', verbose=10)(
    delayed(process_session)(
        ses, ises, nSessions, calciumversion, params, n_jobs_pairs,
        max_pairs_per_group, random_state, cache_dir, force_recompute, qc_csv_path)
    for ises, ses in enumerate(sessions)
)

df_all = pd.concat(session_dfs, ignore_index=True)
df_all.to_csv(os.path.join(output_dir, 'pairwise_pid_all_pairs.csv'), index=False)

#%% ------------------------------------------------------------------
# Aggregate: for each target (stim, choice), express redundancy/
# synergy/unique as % of that pair's I_total, then average within each
# (area, label) group
# ----------------------------------------------------------------------
def summarize_target(df_all, target_prefix, min_i_total=0.01):
    """
    Express redundancy/synergy/unique as % of each pair's I_total, then
    average within each (area, label) group.

    `min_i_total` (bits) excludes pairs whose total joint information
    about the target is too close to zero for the % ratio to be
    meaningful -- without this guard, a handful of near-zero-information
    pairs can produce huge, noisy percentages (dividing by ~0) that
    dominate the group average. Tune this up if group averages still
    look dominated by a few outlier pairs; check `sub['<target>_pid_
    I_total_shuffcorr']`'s distribution if in doubt.
    """
    total = df_all[f'{target_prefix}_pid_I_total_shuffcorr']
    valid = total.abs() > min_i_total
    sub = df_all[valid].copy()
    total = sub[f'{target_prefix}_pid_I_total_shuffcorr']

    sub['pct_redundancy'] = 100 * sub[f'{target_prefix}_pid_redundancy_shuffcorr'] / total
    sub['pct_synergy'] = 100 * sub[f'{target_prefix}_pid_synergy_shuffcorr'] / total
    sub['pct_unique'] = 100 * (sub[f'{target_prefix}_pid_unique_n1_shuffcorr']
                               + sub[f'{target_prefix}_pid_unique_n2_shuffcorr']) / total

    summary = (sub.groupby(['area', 'label'])[['pct_redundancy', 'pct_synergy', 'pct_unique']]
               .agg(['mean', 'sem']))
    n_pairs = sub.groupby(['area', 'label']).size().rename('n_pairs')
    return summary, n_pairs, sub


summary_stim, n_pairs_stim, sub_stim = summarize_target(df_all, 'stim')
summary_choice, n_pairs_choice, sub_choice = summarize_target(df_all, 'choice')

print(f'\nStim-target: {len(sub_stim)}/{len(df_all)} pairs retained '
      f'(|I_total_shuffcorr| > threshold); {len(df_all) - len(sub_stim)} excluded as ~zero-information.')
print(f'Choice-target: {len(sub_choice)}/{len(df_all)} pairs retained; '
      f'{len(df_all) - len(sub_choice)} excluded as ~zero-information.')

summary_stim.to_csv(os.path.join(output_dir, 'pairwise_pid_stim_summary.csv'))
summary_choice.to_csv(os.path.join(output_dir, 'pairwise_pid_choice_summary.csv'))

#%% ------------------------------------------------------------------
# Plotting helper: grouped bar chart of %Red / %Syn / %Unique per
# (area, label) group
# ----------------------------------------------------------------------
TERM_COLORS = {'pct_redundancy': 'tab:orange', 'pct_synergy': 'tab:green', 'pct_unique': 'tab:gray'}
TERM_LABELS = {'pct_redundancy': 'Redundancy', 'pct_synergy': 'Synergy', 'pct_unique': 'Unique (sum)'}


def plot_pid_summary(summary, n_pairs, title, save_path):
    groups_present = list(summary.index)
    areas_present = sorted({a for a, l in groups_present})
    ordered = [g for g in ordered_groups(areas_present) if g in groups_present]

    n_groups = len(ordered)
    n_terms = len(TERM_COLORS)
    x = np.arange(n_groups)
    width = 0.8 / n_terms

    fig, ax = plt.subplots(figsize=(max(6, 1.2 * n_groups), 5))

    for t_idx, (term, color) in enumerate(TERM_COLORS.items()):
        means = [summary.loc[g, (term, 'mean')] for g in ordered]
        sems = [summary.loc[g, (term, 'sem')] for g in ordered]
        offset = (t_idx - (n_terms - 1) / 2) * width
        ax.bar(x + offset, means, width=width, yerr=sems, color=color,
               label=TERM_LABELS[term], capsize=3)

    ax.axhline(0, color='k', linewidth=0.8)
    xtick_labels = [f'{a}\n{l}\n(n={n_pairs.loc[(a, l)]})' for a, l in ordered]
    ax.set_xticks(x)
    ax.set_xticklabels(xtick_labels, fontsize=8)
    ax.set_ylabel('% of joint information about target')
    ax.set_title(title)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f'Saved figure to {save_path}')
    plt.close(fig)


#%% ------------------------------------------------------------------
# Figure 1: pairwise PID about STIMULUS
# ----------------------------------------------------------------------
plot_pid_summary(
    summary_stim, n_pairs_stim,
    title=f'Pairwise PID about stimulus ({"+".join(protocol)}, {calciumversion}, {nSessions} sessions)',
    save_path=os.path.join(output_dir, 'pairwise_pid_stim.png'))

#%% ------------------------------------------------------------------
# Figure 2: pairwise PID about CHOICE
# ----------------------------------------------------------------------
plot_pid_summary(
    summary_choice, n_pairs_choice,
    title=f'Pairwise PID about choice ({"+".join(protocol)}, {calciumversion}, {nSessions} sessions)',
    save_path=os.path.join(output_dir, 'pairwise_pid_choice.png'))
