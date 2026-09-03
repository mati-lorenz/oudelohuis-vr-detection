# -*- coding: utf-8 -*-
"""
Fits the shared-gain (doubly stochastic / modulated Poisson) population
model to each session's trial responses (see infotheory/gain_model.py
for the model and estimators), in two variants:

  Model A: G(t) left completely unconstrained (estimated per trial via
      the precision-weighted estimator).
  Model B: G(t) constrained to a log-linear function of ONE measured
      behavioral variable at a time (running speed, pupil area, video
      PC1, position), fit via a pooled Poisson GLM with offset.

Then asks: how much of the UNCONSTRAINED gain's variance does each
behavioral variable explain? And, as an independent consistency check
on the whole shared-gain framework: does the fitted sigma_G^2 correctly
predict the empirical pairwise noise-correlation structure
(infotheory.spike_stats.signal_noise_correlations)?

Note on units: this treats the deconvolved trace as if it were Poisson
counts. As established in plot_spike_exploratory.py's investigation,
your deconvolved values are NOT literal spike counts (they have
active-frame magnitudes ~100x too large for that) -- but the Poisson
GLM machinery here is used in its "quasi-likelihood" sense: it still
correctly estimates the MEAN-modulation structure (the beta
coefficients and the shape of G(t)) even when the absolute scale of
"counts" is off, because a Poisson GLM's mean-value fit is a valid
(if not maximally efficient) M-estimator for any distribution with the
same variance-to-mean relationship the model assumes. The absolute
sigma_G_sq magnitude should be interpreted with that caveat in mind;
the relative comparisons (which behavioral variable explains gain
variance best; predicted-vs-observed noise correlation SHAPE) are more
robust to it than the absolute numbers.

Fully parallelized across sessions via joblib.
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
    InfoTheoryParams, compute_respmat_for_session, get_trial_labels,
    estimate_tuning_curves_cv, estimate_gain_unconstrained, fit_poisson_gain_glm,
    predicted_noise_correlation, signal_noise_correlations,
    estimate_neuron_coupling_strength, predicted_noise_correlation_coupling,
)
from infotheory.behavior_signals import get_behavior_trace
from infotheory.celldata_utils import get_area_label

#%% ------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
protocol = ['DN']
calciumversion = 'deconv'

params = InfoTheoryParams(
    stim_var='stimcat', stim_value_map={'C': 0, 'N': 1, 'M': 1},
    choice_var='lickResponse', trial_mask_var=None,
)

behavior_vars = ['position', 'runspeed', 'pupil_area', 'video_pc1']
video_pc_column = None

n_cv_folds = 5
max_pairs_for_noise_check = 300   # random pairs used for the predicted-vs-observed check
random_state = 0

n_jobs_sessions = -1

output_dir = './output/infotheory_plots'
os.makedirs(output_dir, exist_ok=True)

#%% ------------------------------------------------------------------
# Per-trial behavior summarization (mirrors compute_respmat_for_session's
# protocol-aware windowing, but applied to a continuous behavioral trace
# instead of the neural calciumdata)
# ----------------------------------------------------------------------
def compute_behavior_trial_values(session, behavior_trace, s_resp_start=0, s_resp_stop=20,
                                   t_pre=0.0, t_post=1.0):
    """
    One value per trial, summarizing `behavior_trace` (already aligned
    to session.ts_F) over that trial's response window -- spatial
    window (s_resp_start..s_resp_stop cm into the stimulus zone) for
    VR/detection protocols, or a fixed time window (t_pre..t_post s
    after trial onset) for time-locked protocols. Implemented directly
    (not via utils.psth) for the time-locked case, to avoid depending
    on an unconfirmed function signature there; the spatial case reuses
    the exact same compute_respmat_space call used elsewhere in this
    pipeline, applied to a single-column "neuron".
    """
    from infotheory.session_utils import TIME_LOCKED_PROTOCOLS, SPATIAL_PROTOCOLS

    protocol_ = session.protocol
    if protocol_ in SPATIAL_PROTOCOLS:
        from utils.psth import compute_respmat_space
        behavior_df = pd.DataFrame({'beh': behavior_trace})
        resp = compute_respmat_space(
            behavior_df, session.ts_F, session.trialdata['stimStart'],
            session.zpos_F, session.trialnum_F,
            s_resp_start=s_resp_start, s_resp_stop=s_resp_stop,
            method='mean', subtr_baseline=False)
        return np.asarray(resp, dtype=float).reshape(-1)

    elif protocol_ in TIME_LOCKED_PROTOCOLS:
        ts_F = np.asarray(session.ts_F)
        onsets = session.trialdata['tOnset'].to_numpy()
        vals = np.full(len(onsets), np.nan)
        for k, onset in enumerate(onsets):
            mask = (ts_F >= onset + t_pre) & (ts_F <= onset + t_post)
            if np.any(mask):
                vals[k] = np.nanmean(behavior_trace[mask])
        return vals

    else:
        raise ValueError(f"Don't know how to compute per-trial behavior values for "
                          f"protocol '{protocol_}'.")


#%% ------------------------------------------------------------------
# Load sessions (shallow)
# ----------------------------------------------------------------------
sessions, nSessions = filter_sessions(
    protocols=protocol, load_behaviordata=False, load_calciumdata=False,
    load_videodata=False, min_trials=100)
report_sessions(sessions)

#%% ------------------------------------------------------------------
# Per-session gain-model fitting (parallelized across sessions)
# ----------------------------------------------------------------------
def process_session(ses, session_index, params, calciumversion, behavior_vars,
                     video_pc_column, n_cv_folds, max_pairs_for_noise_check, random_state):
    print(f'\n=== Session {session_index + 1}: {ses.session_id} ===')

    ses.load_data(load_behaviordata=True, load_calciumdata=True,
                  load_videodata=True, calciumversion=calciumversion)
    respmat = compute_respmat_for_session(ses, calciumversion=calciumversion, keepraw=True)
    stim, choice, mask = get_trial_labels(ses, params)

    respmat_arr = np.asarray(respmat)   # (N, K) native orientation
    K = len(stim)
    if respmat_arr.shape[1] == K:
        respmat_t = respmat_arr.T[mask, :]     # -> (K_masked, N)
    else:
        respmat_t = respmat_arr[mask, :]
    stim_m = stim[mask]
    N = respmat_t.shape[1]

    # --- Model A: unconstrained gain ---
    mu_hat = estimate_tuning_curves_cv(respmat_t, stim_m, n_folds=n_cv_folds,
                                        random_state=random_state + session_index)
    gain_A = estimate_gain_unconstrained(respmat_t, mu_hat)
    print(f'  [{ses.session_id}] Model A: sigma_G_sq = {gain_A["sigma_G_sq"]:.4f} '
          f'(raw {gain_A["sigma_G_sq_raw"]:.4f})')

    # --- Model B: per behavioral variable ---
    model_b_results = {}
    for var_name in behavior_vars:
        try:
            trace = get_behavior_trace(ses, var_name, video_pc_column=video_pc_column)
            trial_vals = compute_behavior_trial_values(ses, trace)
            trial_vals_m = trial_vals[mask]
            valid = np.isfinite(trial_vals_m)
            if valid.sum() < 20:
                raise ValueError(f'too few valid trials ({valid.sum()}) with finite {var_name}')

            behavior_z = (trial_vals_m[valid] - np.nanmean(trial_vals_m[valid])) \
                / (np.nanstd(trial_vals_m[valid]) + 1e-12)
            res_b = fit_poisson_gain_glm(respmat_t[valid, :], mu_hat[valid, :], behavior_z)

            corr_with_A = np.corrcoef(res_b['G_hat_B'], gain_A['G_hat'][valid])[0, 1]
            model_b_results[var_name] = {
                'beta0': res_b['beta0'], 'beta1': res_b['beta1'],
                'pseudo_r2': res_b['pseudo_r2'], 'corr_with_gain_A': corr_with_A,
                'n_trials_used': int(valid.sum()),
            }
            print(f'  [{ses.session_id}] Model B ({var_name}): beta1={res_b["beta1"]:.4f}, '
                  f'pseudo_r2={res_b["pseudo_r2"]:.4f}, corr(G_B,G_A)={corr_with_A:.4f}')
        except ValueError as e:
            print(f'  [{ses.session_id}] SKIPPING behavior variable "{var_name}": {e}')
            continue

    # --- Per-neuron coupling strength + noise-correlation consistency check ---
    #
    # NOTE: uses estimate_neuron_coupling_strength / predicted_noise_
    # correlation_coupling, NOT the uniform-sensitivity predicted_noise_
    # correlation used in an earlier version of this script -- the
    # uniform model assumes every neuron has identical unit sensitivity
    # to the shared gain, which at this pipeline's actual deconvolved-
    # trace magnitudes (tens to hundreds) makes it saturate toward
    # predicting correlation ~1 almost regardless of the true coupling
    # strength. Validated on simulated data with heterogeneous, mixed-
    # sign coupling: the uniform model's predicted-vs-observed
    # correlation was actually NEGATIVE (-0.04) with a ~100x magnitude
    # overestimate, while the per-neuron coupling-strength model
    # recovered the true structure almost exactly (corr=0.998) -- see
    # tests/test_gain_model_canonical.py.
    area, label, _ = get_area_label(ses.celldata)
    rng = np.random.default_rng(random_state + session_index)
    all_pairs = list(zip(rng.integers(0, N, max_pairs_for_noise_check),
                          rng.integers(0, N, max_pairs_for_noise_check)))
    all_pairs = [(i, j) for i, j in all_pairs if i != j]

    coupling_c, coupling_c_sq = estimate_neuron_coupling_strength(respmat_t, mu_hat, stim_m)
    mu_bar = mu_hat.mean(axis=0)   # trial-averaged mean per neuron, for the prediction formula
    signal_corr, noise_corr = signal_noise_correlations(respmat_t, stim_m, all_pairs)
    predicted_corr = np.array([predicted_noise_correlation_coupling(
        mu_bar[i], mu_bar[j], coupling_c[i], coupling_c[j]) for i, j in all_pairs])

    for attr in ('calciumdata', 'videodata', 'behaviordata'):
        if hasattr(ses, attr):
            delattr(ses, attr)

    return {
        'session_id': ses.session_id,
        'sigma_G_sq_A': gain_A['sigma_G_sq'], 'sigma_G_sq_A_raw': gain_A['sigma_G_sq_raw'],
        'G_hat_A': gain_A['G_hat'],
        'model_b': model_b_results,
        'coupling_c': coupling_c, 'area': area, 'label': label,
        'observed_noise_corr': noise_corr, 'predicted_noise_corr': predicted_corr,
        'n_neurons': N, 'n_trials': len(stim_m),
    }


session_results = Parallel(n_jobs=n_jobs_sessions, backend='loky', verbose=10)(
    delayed(process_session)(
        ses, ises, params, calciumversion, behavior_vars, video_pc_column,
        n_cv_folds, max_pairs_for_noise_check, random_state)
    for ises, ses in enumerate(sessions)
)

#%% ------------------------------------------------------------------
# Aggregate: per-session gain-model summary table
# ----------------------------------------------------------------------
rows = []
for res in session_results:
    row = {'session_id': res['session_id'], 'n_neurons': res['n_neurons'], 'n_trials': res['n_trials'],
           'sigma_G_sq_A': res['sigma_G_sq_A'], 'sigma_G_sq_A_raw': res['sigma_G_sq_A_raw']}
    for var_name in behavior_vars:
        if var_name in res['model_b']:
            b = res['model_b'][var_name]
            row[f'{var_name}_beta1'] = b['beta1']
            row[f'{var_name}_pseudo_r2'] = b['pseudo_r2']
            row[f'{var_name}_corr_with_gain_A'] = b['corr_with_gain_A']
        else:
            row[f'{var_name}_beta1'] = np.nan
            row[f'{var_name}_pseudo_r2'] = np.nan
            row[f'{var_name}_corr_with_gain_A'] = np.nan
    rows.append(row)
df_summary = pd.DataFrame(rows)
df_summary.to_csv(os.path.join(output_dir, 'gain_model_session_summary.csv'), index=False)
print('\n' + df_summary.to_string(index=False))
print(f'\nSaved session summary to {os.path.join(output_dir, "gain_model_session_summary.csv")}')

#%% ------------------------------------------------------------------
# Figure 1: how much of the unconstrained gain's variance does each
# behavioral variable explain (mean +/- SEM of pseudo_r2 across sessions)
# ----------------------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(7, 5))
x = np.arange(len(behavior_vars))
means = [df_summary[f'{v}_pseudo_r2'].mean() for v in behavior_vars]
sems = [df_summary[f'{v}_pseudo_r2'].sem() for v in behavior_vars]
ax1.bar(x, means, yerr=sems, capsize=4, color='tab:blue', alpha=0.8)
ax1.set_xticks(x)
ax1.set_xticklabels(behavior_vars)
ax1.set_ylabel('Pseudo-R^2 of Model B (behavior-driven gain)')
ax1.set_title(f'How well does each behavioral variable explain the population gain?\n'
              f'({"+".join(protocol)}, {calciumversion}, {nSessions} sessions)')
fig1.tight_layout()
fig1.savefig(os.path.join(output_dir, 'gain_model_behavior_r2.png'), dpi=150, bbox_inches='tight')
print(f'Saved figure to {os.path.join(output_dir, "gain_model_behavior_r2.png")}')
plt.close(fig1)

#%% ------------------------------------------------------------------
# Figure 2: example session's Ghat_A(t) time-course
# ----------------------------------------------------------------------
best_session = max(session_results, key=lambda r: len(r['model_b']))

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(best_session['G_hat_A'], color='k', linewidth=1, label='G_hat_A (unconstrained)', alpha=0.8)
ax2.axhline(1.0, color='gray', linestyle=':', linewidth=1)
ax2.set_xlabel('Trial')
ax2.set_ylabel('Estimated gain')
ax2.set_title(f'Unconstrained population gain over trials -- {best_session["session_id"]}')
ax2.legend(fontsize=8)
fig2.tight_layout()
fig2.savefig(os.path.join(output_dir, 'gain_model_example_timecourse.png'), dpi=150, bbox_inches='tight')
print(f'Saved figure to {os.path.join(output_dir, "gain_model_example_timecourse.png")}')
plt.close(fig2)

#%% ------------------------------------------------------------------
# Figure 3: predicted vs observed noise correlation (independent
# consistency check on the shared-gain model), pooled across sessions
# ----------------------------------------------------------------------
all_observed = np.concatenate([r['observed_noise_corr'] for r in session_results])
all_predicted = np.concatenate([r['predicted_noise_corr'] for r in session_results])
valid = np.isfinite(all_observed) & np.isfinite(all_predicted)

fig3, ax3 = plt.subplots(figsize=(6, 6))
ax3.scatter(all_predicted[valid], all_observed[valid], s=4, alpha=0.3, color='tab:blue')
lims = [min(all_predicted[valid].min(), all_observed[valid].min()),
        max(all_predicted[valid].max(), all_observed[valid].max())]
ax3.plot(lims, lims, 'k--', linewidth=1, label='unity')
ax3.set_xlabel('Predicted noise correlation (shared-gain model)')
ax3.set_ylabel('Observed noise correlation')
r_check = np.corrcoef(all_predicted[valid], all_observed[valid])[0, 1]
ax3.set_title(f'Shared-gain model consistency check (r={r_check:.3f})\n'
              f'({"+".join(protocol)}, {calciumversion}, {nSessions} sessions)')
ax3.legend(fontsize=8)
fig3.tight_layout()
fig3.savefig(os.path.join(output_dir, 'gain_model_noise_corr_check.png'), dpi=150, bbox_inches='tight')
print(f'Saved figure to {os.path.join(output_dir, "gain_model_noise_corr_check.png")}')
print(f'\nPredicted-vs-observed noise correlation agreement: r = {r_check:.4f}')
plt.close(fig3)

#%% ------------------------------------------------------------------
# Figure 4: per-neuron coupling strength distribution, by area/label --
# the heterogeneity this whole per-neuron reformulation was built to
# capture (Okun et al. 2015-style: neurons vary widely, including in
# SIGN, in how strongly they track the shared population fluctuation)
# ----------------------------------------------------------------------
from infotheory.celldata_utils import bar_by_group

coupling_by_group = {}
for res in session_results:
    for a, l, c in zip(res['area'], res['label'], res['coupling_c']):
        coupling_by_group.setdefault((a, l), []).append(c)
coupling_by_group = {k: np.array(v) for k, v in coupling_by_group.items()}

fig4, ax4 = plt.subplots(figsize=(8, 5))
bar_by_group(ax4, coupling_by_group, ylabel='Per-neuron coupling strength (c_i)')
ax4.set_title(f'Heterogeneity of coupling to the shared population factor\n'
              f'({"+".join(protocol)}, {calciumversion}, {nSessions} sessions)')
fig4.tight_layout()
fig4.savefig(os.path.join(output_dir, 'gain_model_coupling_by_area_label.png'), dpi=150, bbox_inches='tight')
print(f'Saved figure to {os.path.join(output_dir, "gain_model_coupling_by_area_label.png")}')
plt.close(fig4)
