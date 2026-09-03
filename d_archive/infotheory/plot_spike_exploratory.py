# -*- coding: utf-8 -*-
"""
Exploratory analysis of the deconvolved "spike" (calcium-event) data,
covering the standard set of spike-train QC/exploratory analyses that
are actually meaningful for calcium imaging (see
infotheory/spike_stats.py's module docstring for what's included and
why some classic electrophysiology analyses -- e.g. millisecond-scale
ISI distributions -- are NOT included here). This is meant to be run
BEFORE the information-theoretic pipeline: several of its outputs
(reliability, sparsity, event rate) are natural neuron-filtering
criteria for the later analyses.

Analyses (all split by brain area and labeling status, unl/lab):
  1. Event rate & sparsity distributions
  2. Inter-event-interval (IEI) distribution + regularity (CV)
  3. Fano factor distribution
  4. Autocorrelogram of the deconvolved trace
  5. Population coupling (Okun et al. 2015-style)
  6. Split-half reliability of the trial-averaged tuning profile
  7. Pairwise signal vs. noise correlations
  8. Correlation with running speed (movement-artifact check), if
     behavioral tracking is available for this protocol

Produces 5 figures plus a per-neuron summary CSV (useful for filtering
neurons before further analysis) and a per-pair CSV (signal/noise
correlations).

Note on cost: IEI and the autocorrelogram loop per-neuron in plain
Python (O(T) per neuron), so they are computed on a random SUBSAMPLE of
up to `max_neurons_for_iei_autocorr` neurons per session (tracked
explicitly so the subsample's area/label identity is known -- see the
per-session loop below). Everything else (event rate, sparsity, Fano
factor, population coupling, reliability) is vectorized and computed
for the FULL population.
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
from infotheory import compute_respmat_for_session, compute_tensor_for_session, get_trial_labels
from infotheory.params import InfoTheoryParams
from infotheory import spike_stats as ss
from infotheory.celldata_utils import (
    get_area_label, ordered_groups, sample_pairs_within_groups, bar_by_group,
    AREA_COLORS, LABEL_LINESTYLES, DEFAULT_LABEL_ORDER)
from utils.cellselection_lib import filter_nearlabeled_layer23
from qc_lib import load_qc_pass

#%% ------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
protocol = ['DN']
calciumversion = 'deconv'

params = InfoTheoryParams(
    stim_var='stimcat', stim_value_map={'C': 0, 'N': 1, 'M': 1},
    choice_var='lickResponse', trial_mask_var=None,
)

max_neurons_for_iei_autocorr = 300   # per session, subsample cap (these loop per-neuron)
max_pairs_per_group = 300            # for signal/noise correlations
window_sec_fano = 1.0                # Fano-factor window
max_lag_sec_autocorr = 2.0
n_reliability_splits = 10
random_state = 0

# Confirmed imaging frame rate (Hz), per experimenter -- overrides the
# ts_F-based auto-detection in spike_stats.get_frame_rate (which was
# giving a close-but-not-exact ~5.6 Hz estimate). Set to None to fall
# back to auto-detection instead.
fs_override = 5.35

output_dir = './output/infotheory_plots'
os.makedirs(output_dir, exist_ok=True)

# QC filter (in addition to the anatomical V1/PM filter): produced by
# run_qc_summary.py. Run that script first; set to None to skip QC
# filtering (e.g. for a quick look before QC has been run).
qc_csv_path = os.path.join(output_dir, 'qc_per_cell.csv')

#%% ------------------------------------------------------------------
# Load sessions
# ----------------------------------------------------------------------
sessions, nSessions = filter_sessions(
    protocols=protocol, load_behaviordata=False, load_calciumdata=False,
    load_videodata=False, min_trials=100)
report_sessions(sessions)

#%% ------------------------------------------------------------------
# Per-session computation
# ----------------------------------------------------------------------
def process_session(ses, session_index, params, calciumversion,
                     max_neurons_for_iei_autocorr, max_pairs_per_group,
                     window_sec_fano, max_lag_sec_autocorr, n_reliability_splits,
                     random_state, n_jobs_neurons=1, fs_override=None, qc_csv_path=None):
    """
    Run everything for ONE session and return a small, fully picklable
    dict of results (plain numpy arrays / lists / floats -- no Session
    object), so this function can be dispatched via joblib.Parallel
    across sessions without pickling issues.
    """
    print(f'\n=== Session {session_index + 1}: {ses.session_id} ===')
    rng = np.random.default_rng(random_state + session_index)  # independent, reproducible per-session stream

    ses.load_data(load_behaviordata=True, load_calciumdata=True, calciumversion=calciumversion)
    fs = fs_override if fs_override is not None else ss.get_frame_rate(ses)
    calciumdata = np.asarray(ses.calciumdata)
    area, label, _ = get_area_label(ses.celldata)

    # V1/PM unl cells must be near a labeled cell (50um) and in layer 2/3
    # (depth<300um); labeled cells and cells outside V1/PM pass through
    # unaffected (see utils/cellselection_lib.py). Combined with the QC
    # pass/fail flag from run_qc_summary.py (silent/artifact/noisy cells).
    # Applied here, before any per-neuron stats are computed, so
    # respmat/tensor (computed from `ses` below) are filtered the same way
    # further down.
    idx_anat = filter_nearlabeled_layer23(ses, radius=50, depth_thr=300)
    idx_qc = load_qc_pass(ses, qc_csv_path=qc_csv_path)
    idx_valid = idx_anat & idx_qc
    print(f'  [{ses.session_id}] {np.sum(idx_valid)}/{len(idx_valid)} neurons kept '
          f'(anatomical: {np.sum(idx_anat)}, QC: {np.sum(idx_qc)})')
    calciumdata = calciumdata[:, idx_valid]
    area, label = area[idx_valid], label[idx_valid]
    n_neurons_total = calciumdata.shape[1]

    # --- full-population, vectorized stats ---
    rate = ss.event_rate(calciumdata, fs)                  # magnitude-weighted, a.u./s
    active_rate = ss.active_frame_rate(calciumdata, fs)     # count-based, consistent with IEI/sparsity
    spars = ss.sparsity(calciumdata)
    fano = ss.fano_factor(calciumdata, fs, window_sec=window_sec_fano)                 # magnitude-weighted
    active_fano = ss.fano_factor(calciumdata, fs, window_sec=window_sec_fano, threshold=0.0)  # count-based

    # sanity check: flag when the two rate definitions disagree a lot,
    # which signals the deconvolved trace isn't ~0/1-per-frame (see
    # spike_stats.event_rate's docstring)
    with np.errstate(divide='ignore', invalid='ignore'):
        rate_ratio = np.nanmedian(rate[active_rate > 0] / active_rate[active_rate > 0]) \
            if np.any(active_rate > 0) else np.nan
    if np.isfinite(rate_ratio) and rate_ratio > 5:
        print(f'  [{ses.session_id}] NOTE: median(event_rate / active_frame_rate) = {rate_ratio:.1f} '
              f'-- the deconvolved trace has active-frame magnitudes well above 1, so "event_rate" '
              f'is NOT a literal events/sec count here. Use "active_frame_rate" for anything meant '
              f'to be consistent with the IEI/sparsity panels.')

    runspeed_corr = None
    if hasattr(ses, 'runspeed_F') and ses.runspeed_F is not None:
        runspeed_corr = ss.runspeed_correlation(calciumdata, ses.runspeed_F)

    respmat = np.asarray(compute_respmat_for_session(
        ses, calciumversion=calciumversion, keepraw=True))[idx_valid, :]
    tensor, t_axis, t_axis_label = compute_tensor_for_session(
        ses, calciumversion=calciumversion, keepraw=True)
    tensor = tensor[:, idx_valid, :]

    stim, choice, mask = get_trial_labels(ses, params)
    respmat_m = np.asarray(respmat)[:, mask]
    tensor_m = tensor[mask, :, :]
    stim_m = stim[mask]

    coupling = ss.population_coupling(respmat_m, n_trials=len(stim_m))
    reliability = ss.split_half_reliability(tensor_m, n_splits=n_reliability_splits,
                                             random_state=random_state + session_index)

    # --- subsampled-population, per-neuron-loop stats (IEI + autocorrelogram) ---
    subsample_idx = np.arange(n_neurons_total)
    if max_neurons_for_iei_autocorr is not None and n_neurons_total > max_neurons_for_iei_autocorr:
        subsample_idx = rng.choice(n_neurons_total, size=max_neurons_for_iei_autocorr, replace=False)
    calciumdata_sub = calciumdata[:, subsample_idx]
    area_sub = area[subsample_idx]
    label_sub = label[subsample_idx]

    _, cv_iei_sub, _ = ss.iei_stats(calciumdata_sub, fs, n_jobs=n_jobs_neurons)
    lags, ac_sub = ss.autocorrelogram(calciumdata_sub, fs, max_lag_sec=max_lag_sec_autocorr,
                                       n_jobs=n_jobs_neurons)

    iei_pooled_per_group = {}
    for a in np.unique(area_sub):
        for l in DEFAULT_LABEL_ORDER:
            idx_local = np.where((area_sub == a) & (label_sub == l))[0]
            if len(idx_local) == 0:
                continue
            pooled_this_group = []
            for n_local in idx_local:
                ev = np.where(calciumdata_sub[:, n_local] > 0)[0]
                if len(ev) >= 2:
                    pooled_this_group.append(np.diff(ev) / fs)
            if pooled_this_group:
                iei_pooled_per_group[(a, l)] = np.concatenate(pooled_this_group)

    # --- pairwise signal/noise correlations ---
    pairs, pair_group = sample_pairs_within_groups(
        area, label, max_pairs_per_group=max_pairs_per_group, rng=rng)
    signal_corr, noise_corr = ss.signal_noise_correlations(respmat_m, stim_m, pairs)

    print(f'  [{ses.session_id}] fs={fs:.2f} Hz, {n_neurons_total} neurons, '
          f'{len(subsample_idx)} subsampled for IEI/autocorr, {len(pairs)} pairs sampled')

    for attr in ('calciumdata', 'videodata', 'behaviordata'):
        if hasattr(ses, attr):
            delattr(ses, attr)

    return {
        'session_id': ses.session_id, 'fs': fs,
        'area': area, 'label': label,
        'rate': rate, 'active_rate': active_rate, 'sparsity': spars,
        'fano': fano, 'active_fano': active_fano, 'coupling': coupling,
        'reliability': reliability, 'runspeed_corr': runspeed_corr,
        'area_sub': area_sub, 'label_sub': label_sub,
        'cv_iei_sub': cv_iei_sub, 'lags': lags, 'ac_sub': ac_sub,
        'iei_pooled_per_group': iei_pooled_per_group,
        'pair_group': pair_group, 'signal_corr': signal_corr, 'noise_corr': noise_corr,
    }


# n_jobs at the SESSION level (primary parallelism -- each worker handles
# one whole session). Keep n_jobs_neurons=1 inside each worker when running
# multiple sessions in parallel, to avoid oversubscribing CPUs with nested
# parallel pools; bump n_jobs_neurons instead if you only have 1-2 sessions.
n_jobs_sessions = -1
n_jobs_neurons = 1

session_results = Parallel(n_jobs=n_jobs_sessions, backend='loky', verbose=10)(
    delayed(process_session)(
        ses, ises, params, calciumversion,
        max_neurons_for_iei_autocorr, max_pairs_per_group,
        window_sec_fano, max_lag_sec_autocorr, n_reliability_splits,
        random_state, n_jobs_neurons=n_jobs_neurons, fs_override=fs_override,
        qc_csv_path=qc_csv_path)
    for ises, ses in enumerate(sessions)
)

#%% ------------------------------------------------------------------
# Aggregate results across sessions into (area, label) groups
# ----------------------------------------------------------------------
group_scalar = {}
group_iei_cv = {}
group_iei_pooled = {}
group_autocorr = {}
group_pairwise = {}
reference_lags = None
fs_values = []

for res in session_results:
    fs_values.append(res['fs'])
    area, label = res['area'], res['label']

    for a in np.unique(area):
        for l in DEFAULT_LABEL_ORDER:
            idx = np.where((area == a) & (label == l))[0]
            if len(idx) == 0:
                continue
            key = (a, l)
            d = group_scalar.setdefault(key, {
                'rate': [], 'active_rate': [], 'sparsity': [], 'fano': [], 'active_fano': [],
                'coupling': [], 'reliability': [], 'runspeed_corr': []})
            d['rate'].append(res['rate'][idx])
            d['active_rate'].append(res['active_rate'][idx])
            d['sparsity'].append(res['sparsity'][idx])
            d['fano'].append(res['fano'][idx])
            d['active_fano'].append(res['active_fano'][idx])
            d['coupling'].append(res['coupling'][idx])
            d['reliability'].append(res['reliability'][idx])
            if res['runspeed_corr'] is not None:
                d['runspeed_corr'].append(res['runspeed_corr'][idx])

    area_sub, label_sub = res['area_sub'], res['label_sub']
    lags = res['lags']
    ac_sub = res['ac_sub']
    if reference_lags is None:
        reference_lags = lags
    elif len(lags) != len(reference_lags) or not np.allclose(lags, reference_lags):
        ac_sub = np.vstack([np.interp(reference_lags, lags, row) for row in ac_sub])

    for a in np.unique(area_sub):
        for l in DEFAULT_LABEL_ORDER:
            idx_local = np.where((area_sub == a) & (label_sub == l))[0]
            if len(idx_local) == 0:
                continue
            key = (a, l)
            group_iei_cv.setdefault(key, []).append(res['cv_iei_sub'][idx_local])
            group_autocorr.setdefault(key, []).append(ac_sub[idx_local, :])
            if key in res['iei_pooled_per_group']:
                group_iei_pooled.setdefault(key, []).append(res['iei_pooled_per_group'][key])

    for k, key in enumerate(res['pair_group']):
        d = group_pairwise.setdefault(key, {'signal_corr': [], 'noise_corr': []})
        d['signal_corr'].append(res['signal_corr'][k])
        d['noise_corr'].append(res['noise_corr'][k])

if len(set(np.round(fs_values, 1))) > 1:
    print(f'\nWARNING: frame rate varies across sessions ({sorted(set(fs_values))} Hz); '
          f'the autocorrelogram was interpolated onto a common lag grid.')

#%% ------------------------------------------------------------------
# Build per-neuron summary table (full-population stats; IEI CV is NaN
# for neurons outside the per-session IEI/autocorrelogram subsample)
# ----------------------------------------------------------------------
rows = []
for (a, l), d in group_scalar.items():
    rate_cat = np.concatenate(d['rate'])
    active_rate_cat = np.concatenate(d['active_rate'])
    spars_cat = np.concatenate(d['sparsity'])
    fano_cat = np.concatenate(d['fano'])
    active_fano_cat = np.concatenate(d['active_fano'])
    coup_cat = np.concatenate(d['coupling'])
    rel_cat = np.concatenate(d['reliability'])
    rs_cat = np.concatenate(d['runspeed_corr']) if d['runspeed_corr'] else np.full_like(rate_cat, np.nan)
    for i in range(len(rate_cat)):
        rows.append({'area': a, 'label': l,
                      'mean_activity_per_sec': rate_cat[i],   # magnitude-weighted (was 'event_rate_hz')
                      'active_frame_rate_hz': active_rate_cat[i],  # count-based, consistent with IEI/sparsity
                      'sparsity': spars_cat[i],
                      'fano_factor_magnitude': fano_cat[i],   # magnitude-weighted (was 'fano_factor')
                      'fano_factor_active_frames': active_fano_cat[i],  # count-based, comparable to Poisson~1
                      'population_coupling': coup_cat[i],
                      'split_half_reliability': rel_cat[i], 'runspeed_corr': rs_cat[i]})
df_summary = pd.DataFrame(rows)
df_summary.to_csv(os.path.join(output_dir, 'spike_exploratory_per_neuron_summary.csv'), index=False)
print(f'\nSaved per-neuron summary to '
      f'{os.path.join(output_dir, "spike_exploratory_per_neuron_summary.csv")}')

rows_iei = []
for (a, l), cv_list in group_iei_cv.items():
    cv_cat = np.concatenate(cv_list)
    for v in cv_cat:
        rows_iei.append({'area': a, 'label': l, 'iei_cv': v})
pd.DataFrame(rows_iei).to_csv(os.path.join(output_dir, 'spike_exploratory_iei_cv.csv'), index=False)

rows_pw = []
for (a, l), d in group_pairwise.items():
    for sc, nc in zip(d['signal_corr'], d['noise_corr']):
        rows_pw.append({'area': a, 'label': l, 'signal_corr': sc, 'noise_corr': nc})
pd.DataFrame(rows_pw).to_csv(os.path.join(output_dir, 'spike_exploratory_pairwise_corr.csv'), index=False)
print(f'Saved pairwise signal/noise correlation table to '
      f'{os.path.join(output_dir, "spike_exploratory_pairwise_corr.csv")}')

#%% ------------------------------------------------------------------
# Figure 1: event rate, sparsity, IEI, Fano factor (2x2 grid)
#
# NOTE: the top-left panel plots `active_frame_rate` (count-based,
# bounded by the frame rate), NOT the magnitude-weighted `event_rate` --
# active_frame_rate is the one directly consistent with the IEI panel
# (median IEI ~= 1/active_frame_rate under near-Poisson statistics),
# whereas event_rate can be much larger whenever the deconvolved
# trace's active-frame values are well above 1 (see
# spike_stats.event_rate's docstring). The magnitude-weighted version
# is still saved in the per-neuron CSV as 'mean_activity_per_sec' if
# you want it.
# ----------------------------------------------------------------------
# NOTE: Fano factor panel plots `active_fano` (count-based, binarized
# active-frame events per window), NOT the magnitude-weighted `fano` --
# the magnitude-weighted version is dominated by the SQUARE of typical
# event amplitude for heavy-tailed/zero-inflated data and is not
# comparable to the classic Poisson-count "~1" intuition (see
# spike_stats.fano_factor's docstring). Both are saved in the CSV.
# ----------------------------------------------------------------------
active_rate_by_group = {k: np.concatenate(v['active_rate']) for k, v in group_scalar.items()}
spars_by_group = {k: np.concatenate(v['sparsity']) for k, v in group_scalar.items()}
active_fano_by_group = {k: np.concatenate(v['active_fano']) for k, v in group_scalar.items()}
iei_pooled_by_group = {k: np.concatenate(v) for k, v in group_iei_pooled.items()}

fig1, axes1 = plt.subplots(2, 2, figsize=(13, 9))
bar_by_group(axes1[0, 0], active_rate_by_group, ylabel='Active-frame rate (events/s)')
bar_by_group(axes1[0, 1], spars_by_group, ylabel='Sparsity (fraction of frames active)')
bar_by_group(axes1[1, 0], iei_pooled_by_group, ylabel='Inter-event interval (s)', log_y=True)
bar_by_group(axes1[1, 1], active_fano_by_group,
              ylabel=f'Fano factor of active-frame counts ({window_sec_fano:.0f}s windows)',
              zero_line=False)
axes1[1, 1].axhline(1.0, color='k', linestyle=':', linewidth=1)
fig1.suptitle(f'Basic activity statistics ({"+".join(protocol)}, {calciumversion}, {nSessions} sessions)')
fig1.tight_layout()
fig1.savefig(os.path.join(output_dir, 'spike_exploratory_basic_stats.png'), dpi=150, bbox_inches='tight')
print(f'Saved figure to {os.path.join(output_dir, "spike_exploratory_basic_stats.png")}')
plt.close(fig1)

#%% ------------------------------------------------------------------
# Figure 2: autocorrelogram (mean +/- SEM per group) -- unchanged, this
# is a time-course plot, not a distribution, so the line+band style
# already reads cleanly with multiple groups
# ----------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(7, 5))
for (a, l), ac_list in group_autocorr.items():
    if not ac_list:
        continue
    ac_all = np.concatenate(ac_list, axis=0)
    mean_ac = np.nanmean(ac_all, axis=0)
    sem_ac = np.nanstd(ac_all, axis=0) / np.sqrt(max(ac_all.shape[0], 1))
    color = AREA_COLORS.get(a, 'gray')
    ls = LABEL_LINESTYLES.get(l, '-')
    ax2.plot(reference_lags, mean_ac, color=color, linestyle=ls, label=f'{a} ({l}, n={ac_all.shape[0]})')
    ax2.fill_between(reference_lags, mean_ac - sem_ac, mean_ac + sem_ac, color=color, alpha=0.15)
ax2.axhline(0, color='gray', linewidth=0.5)
ax2.set_xlabel('Lag (s)')
ax2.set_ylabel('Autocorrelation')
ax2.set_title(f'Deconvolved-trace autocorrelogram ({"+".join(protocol)}, {calciumversion})')
ax2.legend(fontsize=8)
fig2.tight_layout()
fig2.savefig(os.path.join(output_dir, 'spike_exploratory_autocorrelogram.png'), dpi=150, bbox_inches='tight')
print(f'Saved figure to {os.path.join(output_dir, "spike_exploratory_autocorrelogram.png")}')
plt.close(fig2)

#%% ------------------------------------------------------------------
# Figure 3: population coupling & split-half reliability
# ----------------------------------------------------------------------
coupling_by_group = {k: np.concatenate(v['coupling']) for k, v in group_scalar.items()}
reliability_by_group = {k: np.concatenate(v['reliability']) for k, v in group_scalar.items()}

fig3, axes3 = plt.subplots(1, 2, figsize=(13, 5))
bar_by_group(axes3[0], coupling_by_group, ylabel='Population coupling (r)')
bar_by_group(axes3[1], reliability_by_group, ylabel='Split-half reliability (r)')
fig3.suptitle(f'Population coupling & trial-to-trial reliability '
              f'({"+".join(protocol)}, {calciumversion}, {nSessions} sessions)')
fig3.tight_layout()
fig3.savefig(os.path.join(output_dir, 'spike_exploratory_coupling_reliability.png'), dpi=150, bbox_inches='tight')
print(f'Saved figure to {os.path.join(output_dir, "spike_exploratory_coupling_reliability.png")}')
plt.close(fig3)

#%% ------------------------------------------------------------------
# Figure 4: pairwise signal vs. noise correlations
# ----------------------------------------------------------------------
signal_by_group = {k: np.array(v['signal_corr']) for k, v in group_pairwise.items()}
noise_by_group = {k: np.array(v['noise_corr']) for k, v in group_pairwise.items()}

fig4, axes4 = plt.subplots(1, 2, figsize=(13, 5))
bar_by_group(axes4[0], signal_by_group, ylabel='Signal correlation (r)')
bar_by_group(axes4[1], noise_by_group, ylabel='Noise correlation (r)')
fig4.suptitle(f'Pairwise signal & noise correlations ({"+".join(protocol)}, {calciumversion}, {nSessions} sessions)')
fig4.tight_layout()
fig4.savefig(os.path.join(output_dir, 'spike_exploratory_signal_noise_corr.png'), dpi=150, bbox_inches='tight')
print(f'Saved figure to {os.path.join(output_dir, "spike_exploratory_signal_noise_corr.png")}')
plt.close(fig4)

#%% ------------------------------------------------------------------
# Figure 5 (optional): running-speed correlation
# ----------------------------------------------------------------------
runspeed_by_group = {k: np.concatenate(v['runspeed_corr']) for k, v in group_scalar.items() if v['runspeed_corr']}
if runspeed_by_group:
    fig5, ax5 = plt.subplots(figsize=(7, 5))
    bar_by_group(ax5, runspeed_by_group, ylabel='Correlation with running speed (r)')
    ax5.set_title(f'Locomotion-related activity ({"+".join(protocol)}, {calciumversion}, {nSessions} sessions)')
    fig5.tight_layout()
    fig5.savefig(os.path.join(output_dir, 'spike_exploratory_runspeed_corr.png'), dpi=150, bbox_inches='tight')
    print(f'Saved figure to {os.path.join(output_dir, "spike_exploratory_runspeed_corr.png")}')
    plt.close(fig5)
else:
    print('\nNo runspeed data available for this protocol/session set -- skipped Figure 5.')
