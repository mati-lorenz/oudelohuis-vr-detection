# -*- coding: utf-8 -*-
"""
Dimensionality-reduction pipeline, STEP 5: information content of the
low-D population code, per (area, label) group.

This is the "information" stage of the roadmap, closing the loop back to
the single-cell/pairwise information-theoretic pipeline this whole
dimensionality-reduction detour branched off from: does the population's
low-D latent structure (from Step 2's GPFA fit) carry MORE, LESS, or
DIFFERENT stimulus/choice information than individual neurons?

Deliberately reuses Step 2's ALREADY-FITTED GPFA latents (the saved
trajectories_<session_id>.npz files) rather than refitting anything --
per the original DR-methods comparison, GPFA/FA latents are the right
input for information-theoretic analysis specifically BECAUSE they're
unsupervised (fit without reference to stim/choice), unlike TDR's axes,
which are fit USING stim/choice and would make an information estimate
on them partly circular.

For each GPFA latent dimension, collapses each trial's trajectory to a
single scalar (mean over `response_window`, in the SAME axis units as
Steps 3-4's denoise/condition windows) and computes shuffle-corrected
mutual information against stim and against choice, using equipopulated
binning (matching plot_response_distributions.py's binning convention)
and a permutation significance test (same pattern as Step 4's jPCA
significance test, including Benjamini-Hochberg correction across all
tests -- see that script's docstring for why the correction matters).

VALIDATED (see development notes): the shuffle-corrected MI estimator
recovers ~0 MI (not systematically positive) on synthetic label-independent
data, and correctly detects genuine signal with a low p-value on synthetic
label-dependent data -- shuffle correction removes the same upward small-
sample bias that Panzeri-Treves correction targets in the original
single-cell pipeline (see Panzeri et al. 2007, already cited in this
project's earlier discussion of sufficient-sample-size literature).

Produces, per session:
  - info_<session_id>.npz : per group, per latent dimension: MI_raw,
    MI_corrected, p_value for stim and for choice
  - a quick-look figure per group: MI_corrected per latent dimension,
    stim vs choice, pooled across sessions, with FDR-corrected
    significance annotated -- directly comparable in spirit to this
    project's single-cell MI figures (plot_temporal_information.py etc.)
    but at the population-latent level instead of per-neuron
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
from infotheory import (compute_tensor_for_session, get_trial_labels, InfoTheoryParams,
                        BinningParams, BiasCorrectionParams, ParallelParams,
                        cache_path, load_or_compute)
from infotheory.discretize import equipopulated_edges

#%% ------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
protocol = ['DN']
calciumversion = 'deconv'

stim_var = 'signal'
stim_binarize_threshold = 0
choice_var = 'lickResponse'
trial_mask_var = None

# where Step 2 (run_gpfa_trajectories.py) saved its output -- this script
# reads that, it does not recompute GPFA
trajectory_dir = os.path.join('./output/dimred_plots', 'trajectories')

# which portion of the trial to collapse each trial's latent trajectory
# into a scalar response, in the SAME axis units as Steps 3-4's windows.
# Default matches the informative window identified from
# tdr_r2_over_time.png during Step 3 -- adjust per your own data.
response_window_range = (0, 40)

n_bins = 4              # equipopulated bins for the MI estimate
n_shuffles = 200         # for shuffle bias-correction + significance test
fdr_alpha = 0.05
random_state = 0
n_jobs_sessions = -1

cache_dir = './output/dimred_cache'
force_recompute = False

# Bump whenever _compute()'s return schema/logic changes without a matching
# key_params change -- see run_tdr_encoding.py's docstring for why this
# exists (hit this exact bug once already in this pipeline).
_INFO_COMPUTE_VERSION = 1

output_dir = './output/dimred_plots'
info_dir = os.path.join(output_dir, 'information')
os.makedirs(info_dir, exist_ok=True)


#%% ------------------------------------------------------------------
# Core: shuffle-corrected discrete MI estimator
# ----------------------------------------------------------------------
def _naive_mi(binned_response, label, n_bins):
    """Plug-in (naive, upward-biased at small sample sizes) MI estimate, in bits."""
    labels_u = np.unique(label)
    joint = np.zeros((n_bins, len(labels_u)))
    for i, lv in enumerate(labels_u):
        for b in range(n_bins):
            joint[b, i] = np.sum((binned_response == b) & (label == lv))
    total = joint.sum()
    if total == 0:
        return 0.0
    joint = joint / total
    pr = joint.sum(axis=1, keepdims=True)
    pl = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = joint * np.log2(joint / (pr @ pl))
    return float(np.nansum(terms[joint > 0]))


def compute_mi_discrete(response, label, n_bins=4, n_shuffles=200, random_state=0):
    """
    Shuffle-corrected mutual information (bits) between a continuous
    per-trial scalar `response` and a discrete per-trial `label`
    (stim or choice). Validated during development: gives ~0 (not
    systematically positive) MI_corrected on label-independent synthetic
    data, and correctly low p-values on label-dependent synthetic data.

    Returns dict: mi_raw, mi_corrected, null_mi_mean, null_mi_std, p_value
    """
    response = np.asarray(response, dtype=float)
    label = np.asarray(label)
    valid = ~np.isnan(response)
    response, label = response[valid], label[valid]

    if len(np.unique(label)) < 2 or len(response) < 2 * n_bins:
        return {'mi_raw': np.nan, 'mi_corrected': np.nan, 'null_mi_mean': np.nan,
                'null_mi_std': np.nan, 'p_value': np.nan, 'n_trials': len(response)}

    edges, _ = equipopulated_edges(response, n_bins)
    binned = np.clip(np.digitize(response, edges[1:-1]), 0, n_bins - 1)

    mi_raw = _naive_mi(binned, label, n_bins)
    rng = np.random.default_rng(random_state)
    null_mi = np.empty(n_shuffles)
    for i in range(n_shuffles):
        null_mi[i] = _naive_mi(binned, rng.permutation(label), n_bins)

    mi_corrected = mi_raw - null_mi.mean()
    p_value = float((1 + np.sum(null_mi >= mi_raw)) / (1 + n_shuffles))
    return {'mi_raw': mi_raw, 'mi_corrected': float(mi_corrected), 'null_mi_mean': float(null_mi.mean()),
            'null_mi_std': float(null_mi.std()), 'p_value': p_value, 'n_trials': len(response)}


#%% ------------------------------------------------------------------
# Load sessions (shallow) -- only needed for stim/choice trial labels and
# the tensor's own axis (to resolve response_window_range); GPFA latents
# themselves come from the saved npz files, not recomputed here
# ----------------------------------------------------------------------
sessions, nSessions = filter_sessions(
    protocols=protocol, load_behaviordata=False, load_calciumdata=False,
    load_videodata=False, min_trials=100)
report_sessions(sessions)

params = InfoTheoryParams(
    binning=BinningParams(method='equipopulated', n_bins=n_bins),
    bias=BiasCorrectionParams(panzeri_treves=False),
    parallel=ParallelParams(n_jobs=1, backend='loky', verbose=0),
    stim_var=stim_var, stim_binarize_threshold=stim_binarize_threshold,
    choice_var=choice_var, trial_mask_var=trial_mask_var,
)


#%% ------------------------------------------------------------------
# Per session: load Step 2's saved GPFA latents, collapse to per-trial
# scalars, compute MI against stim/choice per latent dimension per group
# ----------------------------------------------------------------------
def process_session(ses, session_index, nSessions, trajectory_dir, params,
                     response_window_range, n_bins, n_shuffles, random_state,
                     info_dir, cache_dir, force_recompute):
    print(f'\n=== Session {session_index + 1}/{nSessions}: {ses.session_id} ===')

    npz_path = os.path.join(trajectory_dir, f'trajectories_{ses.session_id}.npz')
    if not os.path.exists(npz_path):
        print(f'  [{ses.session_id}] no GPFA trajectory file at {npz_path} -- run '
              f'run_gpfa_trajectories.py first. Skipping this session.')
        return {'session_id': ses.session_id, 'groups': [], 'group_results': {}}

    key_params = dict(
        compute_version=_INFO_COMPUTE_VERSION,
        session_id=ses.session_id, protocol=ses.protocol,
        stim_var=params.stim_var, stim_binarize_threshold=params.stim_binarize_threshold,
        choice_var=params.choice_var, trial_mask_var=params.trial_mask_var,
        response_window_range=str(response_window_range), n_bins=n_bins,
        n_shuffles=n_shuffles, random_state=random_state,
        trajectory_npz_mtime=os.path.getmtime(npz_path),
    )
    path = cache_path(cache_dir, f'{ses.session_id}_info_dimred', **key_params)

    def _compute(ses=ses):
        data = np.load(npz_path, allow_pickle=True)
        axis = data['axis']

        stim, choice, mask = get_trial_labels(ses, params)

        # which group keys were actually fit by Step 2 (exclude the
        # auxiliary *_trial_keep_idx entries also stored in the npz)
        fitted_group_keys = sorted({
            k for k in data.files
            if not k.endswith('_trial_keep_idx') and k not in ('axis', 'axis_label')
        })

        response_mask = (axis >= response_window_range[0]) & (axis <= response_window_range[1])
        if not np.any(response_mask):
            print(f'  [{ses.session_id}] WARNING: response_window_range={response_window_range} matches no '
                  f'bins on this session\'s axis ({axis.min():.1f} to {axis.max():.1f}) -- using whole window.')
            response_mask = np.ones(len(axis), dtype=bool)

        group_results = {}
        for group_key in fitted_group_keys:
            traj = data[group_key]  # (n_trials_kept, x_dim, T)
            trial_keep_key = f'{group_key}_trial_keep_idx'
            trial_keep_idx = data[trial_keep_key] if trial_keep_key in data.files else \
                np.ones(traj.shape[0], dtype=bool)

            # align stim/choice/mask (full trial set) down to the trials
            # Step 2 actually kept for this group (some were dropped there
            # for excessive NaN, see run_gpfa_trajectories.py)
            stim_g = stim[mask][trial_keep_idx] if np.sum(mask) == len(trial_keep_idx) else None
            choice_g = choice[mask][trial_keep_idx] if (choice is not None and stim_g is not None) else None
            if stim_g is None:
                print(f'    [{ses.session_id} {group_key}] trial count mismatch between trial labels and '
                      f'saved trajectory -- skipping (likely a stale trajectory file from a different '
                      f'trial-masking configuration; re-run run_gpfa_trajectories.py).')
                continue

            n_trials, x_dim, T = traj.shape
            window_idx = response_mask[:T] if len(response_mask) >= T else \
                np.ones(T, dtype=bool)  # length mismatch fallback (different tensor window than Steps 3-4)
            scalar_responses = np.nanmean(traj[:, :, window_idx], axis=2)  # (n_trials, x_dim)

            dim_results = []
            for d in range(x_dim):
                mi_stim = compute_mi_discrete(scalar_responses[:, d], stim_g, n_bins=n_bins,
                                               n_shuffles=n_shuffles, random_state=random_state)
                mi_choice = compute_mi_discrete(scalar_responses[:, d], choice_g, n_bins=n_bins,
                                                 n_shuffles=n_shuffles, random_state=random_state + 1) \
                    if choice_g is not None else None
                dim_results.append({'dim': d, 'mi_stim': mi_stim, 'mi_choice': mi_choice})

            print(f'    [{ses.session_id} {group_key}] {x_dim} latent dims, '
                  f'{scalar_responses.shape[0]} trials, mean MI_stim(corrected)='
                  f'{np.nanmean([r["mi_stim"]["mi_corrected"] for r in dim_results]):.3f} bits')
            group_results[group_key] = {'dim_results': dim_results, 'n_trials': n_trials, 'x_dim': x_dim}

        return {'group_results': group_results, 'fitted_group_keys': list(group_results.keys())}

    result = load_or_compute(path, _compute, force_recompute=force_recompute)

    np.savez(os.path.join(info_dir, f'info_{ses.session_id}.npz'),
             **{f'{gk}_dim{r["dim"]}_mi_stim_corrected': r['mi_stim']['mi_corrected']
                for gk, gres in result['group_results'].items() for r in gres['dim_results']},
             **{f'{gk}_dim{r["dim"]}_mi_stim_pvalue': r['mi_stim']['p_value']
                for gk, gres in result['group_results'].items() for r in gres['dim_results']})

    return {'session_id': ses.session_id, 'groups': result['fitted_group_keys'],
            'group_results': result['group_results']}


session_results = Parallel(n_jobs=n_jobs_sessions, backend='loky', verbose=10)(
    delayed(process_session)(
        ses, ises, nSessions, trajectory_dir, params,
        response_window_range, n_bins, n_shuffles, random_state,
        info_dir, cache_dir, force_recompute)
    for ises, ses in enumerate(sessions)
)

print('\nDone. Per-session information files:')
for res in session_results:
    print(f"  {res['session_id']}: groups {res['groups']}")

#%% ------------------------------------------------------------------
# Pool across sessions per group, apply FDR correction across ALL
# (session, group, dim, variable) tests -- same rationale as Step 4
# ----------------------------------------------------------------------
rows = []
for res in session_results:
    for group_key in res['groups']:
        gres = res['group_results'][group_key]
        for r in gres['dim_results']:
            for var_name in ('mi_stim', 'mi_choice'):
                m = r[var_name]
                if m is None or np.isnan(m['mi_corrected']):
                    continue
                rows.append({'session_id': res['session_id'], 'group': group_key, 'dim': r['dim'],
                            'variable': var_name, 'mi_corrected': m['mi_corrected'], 'p_value': m['p_value']})
df = pd.DataFrame(rows)

if len(df):
    pvals = df['p_value'].to_numpy()
    n_tests = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresholds = (np.arange(1, n_tests + 1) / n_tests) * fdr_alpha
    below = ranked <= thresholds
    n_sig = int(np.max(np.where(below)[0]) + 1) if np.any(below) else 0
    fdr_cutoff = ranked[n_sig - 1] if n_sig > 0 else 0.0
    df['significant_fdr'] = df['p_value'] <= fdr_cutoff

    print(f'\nFDR correction (alpha={fdr_alpha}, n_tests={n_tests}): {n_sig}/{n_tests} tests survive '
          f'(p <= {fdr_cutoff:.4f}); uncorrected p<{fdr_alpha}: {int(np.sum(pvals < fdr_alpha))}/{n_tests}')

    df.to_csv(os.path.join(info_dir, 'information_all_tests.csv'), index=False)

    #%% --------------------------------------------------------------
    # Quick-look: MI_corrected per latent dim, stim vs choice, pooled
    # across sessions per group, FDR significance marked
    # ------------------------------------------------------------------
    groups_present = sorted(df['group'].unique())
    fig, axes = plt.subplots(1, len(groups_present), figsize=(4.5 * len(groups_present), 4), squeeze=False)
    axes = axes[0]

    for j, group_key in enumerate(groups_present):
        ax = axes[j]
        sub = df[df.group == group_key]
        for var_name, color, offset in (('mi_stim', 'tab:blue', -0.15), ('mi_choice', 'tab:orange', 0.15)):
            sub_v = sub[sub.variable == var_name]
            if not len(sub_v):
                continue
            agg = sub_v.groupby('dim')['mi_corrected'].agg(['mean', 'sem']).reset_index()
            sig_any = sub_v.groupby('dim')['significant_fdr'].any().reindex(agg['dim']).to_numpy()
            ax.bar(agg['dim'] + offset, agg['mean'], width=0.3, yerr=agg['sem'],
                   color=color, alpha=0.8, label=var_name.replace('mi_', ''), capsize=3)
            for x, y, sig in zip(agg['dim'] + offset, agg['mean'], sig_any):
                if sig:
                    ax.text(x, y, '*', ha='center', va='bottom', fontsize=10)
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_title(group_key, fontsize=9)
        ax.set_xlabel('GPFA latent dimension', fontsize=8)
        if j == 0:
            ax.set_ylabel('MI (bits, shuffle-corrected)', fontsize=8)
        ax.legend(fontsize=7)

    fig.suptitle('Population-latent information about stim/choice, per group '
                 '(* = FDR-significant, pooled across sessions)', y=1.03)
    fig.tight_layout()
    fig_path = os.path.join(output_dir, 'information_quicklook.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'Saved quick-look information figure to {fig_path}')
    print(f'Saved full test table to {os.path.join(info_dir, "information_all_tests.csv")}')
else:
    print('\nNo valid MI estimates -- check that run_gpfa_trajectories.py has been run and its output '
          'is available at the configured trajectory_dir.')
