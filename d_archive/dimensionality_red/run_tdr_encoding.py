# -*- coding: utf-8 -*-
"""
Dimensionality-reduction pipeline, STEP 3: Targeted Dimensionality
Reduction (TDR, Mante, Sussillo, Shenoy & Newsome 2013), per (area, label)
group.

Unlike GPFA (Step 2, unsupervised -- finds whatever low-D structure best
explains the data), TDR is EXPLICITLY built for the encoding question: at
every time bin, each PCA-reduced dimension's response is regressed onto
the task variables

    x_d(k, t) = b0_d(t) + b_stim_d(t) * stim(k) + b_choice_d(t) * choice(k) + noise

giving one regression-coefficient VECTOR per task variable per time bin
(length n_pca_components), then MAPPED BACK to neuron space via the PCA
loadings for interpretability. These are denoised by averaging over a
chosen time window (`denoise_window`), then orthogonalized against each
other via QR decomposition (so the "stimulus axis" and "choice axis" are
not just correlated copies of each other, per Mante et al. 2013's
Methods) -- population activity projected onto these axes directly
answers "how much does this population encode stimulus vs. choice, and
how does that encoding evolve over the trial."

WHY PCA FIRST (`use_pca_denoise`, default True): fitting the regression
directly on raw single-trial, single-neuron activity (the original version
of this script) gave near-zero R^2 (~0.01-0.02) in every group -- not
because encoding is absent, but because single-trial per-neuron calcium
signals are extremely noisy, and the regression has no way to average that
out. Reducing to the top `n_pca_components` PCs first (fit once per group
on all trials pooled) averages across neurons before regression, which is
exactly what Mante et al.'s own pipeline does; it should raise R^2
substantially if there's real linear encoding to find. Set
`use_pca_denoise=False` to fall back to the original raw-neuron regression
for comparison.

Axis sign is fixed so that projecting onto a task-variable axis is
POSITIVE when that variable is in its "high" state (see `_fix_axis_sign`)
-- this matters because QR's sign is otherwise arbitrary, and without
fixing it, averaging projected trajectories ACROSS SESSIONS (each session
has its own independently-fitted axes, on its own neurons) could cancel
out real signal rather than average it.

Condition means with fewer than `min_trials_per_condition` trials are
dropped from the output/plot entirely (rather than shown as noisy,
unreliable lines) -- a condition with only a handful of trials produces a
condition-mean trajectory that's mostly sampling noise, not signal.

Uses the SAME group-selection convention as the rest of this pipeline:
filter_nearlabeled_layer23 (anatomical) + load_qc_pass (QC) +
filter_target_groups (which area/label groups you want).

Produces, per session:
  - tdr_<session_id>.npz : per group, the fitted axes (stim_axis,
    choice_axis, offset_axis, each length n_neurons_group -- already
    mapped back from PCA space if use_pca_denoise=True), the
    condition-averaged (stim x choice) trajectories projected onto those
    axes, the time/position axis, and the regression R^2 per time bin
    (how much of the (PCA-reduced, if enabled) population variance the 3
    regressors explain -- a QC-style diagnostic on the encoding model
    itself, and the number most worth checking first)
  - a quick-look state-space figure per group: stim-axis vs choice-axis
    trajectory, one line per (session, condition), so encoding structure
    is visible directly rather than inferred from an unsupervised
    projection (contrast with Step 2's GPFA quick-look)
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
from sklearn.decomposition import PCA

from loaddata.session_info import filter_sessions, report_sessions
from infotheory import (compute_tensor_for_session, get_trial_labels, InfoTheoryParams,
                        BinningParams, BiasCorrectionParams, ParallelParams,
                        cache_path, load_or_compute)
from infotheory.celldata_utils import get_area_label, DEFAULT_LABEL_ORDER
from utils.cellselection_lib import filter_nearlabeled_layer23, filter_target_groups
from infotheory.qc_lib import load_qc_pass

#%% ------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
protocol = ['DN']
calciumversion = 'deconv'

# Which (area, label) groups to include -- same convention as Steps 1-2.
target_areas = None
target_labels = None

qc_csv_path = os.path.join('./output/dimred_plots', 'qc_per_cell.csv')

# tensor window -- match Steps 1-2 if you want a directly comparable window
s_pre, s_post, binsize = -60, 80, 10
t_pre, t_post = -1, 2   # used instead of s_pre/s_post for time-locked protocols (IM/GR/GN)

# task variables to regress against -- same convention as the info-theoretic
# scripts (plot_temporal_information.py etc.)
stim_var = 'signal'
stim_binarize_threshold = 0
choice_var = 'lickResponse'
trial_mask_var = None

# time window (as a slice of the tensor's own bin indices) to average
# regression coefficients over before orthogonalizing -- None = use the
# WHOLE window. Narrowing this to the window where task information is
# actually present (e.g. late in the trial, near the choice) gives a much
# less noisy axis than averaging over bins that are pure baseline.
# Which portion of the trial to average regression coefficients over,
# specified in the tensor's own axis UNITS (cm for position-locked
# protocols, seconds for time-locked) rather than raw bin indices --
# resolved against the actual returned axis at runtime (see process_session),
# so it's robust to the exact binning convention rather than assuming which
# integer index a given position/time falls into.
# Default (0, 40): set from inspecting tdr_r2_over_time.png, where R^2
# consistently peaks just after the stimulus zone (position 0) across
# every group, decaying back to baseline by ~+50-80cm -- adjust these
# bounds by eye per your own run of that figure. None = whole window
# (the original default, averages over a lot of uninformative baseline).
denoise_window_range = (0, 40)

# PCA pre-denoising before regression -- see module docstring for why this
# matters (raw single-neuron regression gave near-zero R^2). n_pca_components
# reads Step 1's FA optimal dimensionality by default, same convention as
# GPFA's xdim_source; set pca_source='fixed' to use pca_components_fixed instead
use_pca_denoise = True
pca_source = 'from_step1'
pca_components_fixed = 10
max_pca_components = 20
dimensionality_summary_path = os.path.join('./output/dimred_plots', 'dimensionality_optimal_summary.csv')

# condition means built from fewer than this many trials are dropped
# entirely (too noisy to trust) rather than plotted
min_trials_per_condition = 10
n_shuffles_significance = 200   # for tdr_r2_significance_test, see that function's docstring

min_neurons_per_group = 10
random_state = 0
n_jobs_sessions = -1

cache_dir = './output/dimred_cache'
force_recompute = False

# Bump this whenever _compute()'s internal logic or RETURN SCHEMA changes
# (new fields, renamed fields, etc.) even if no user-facing config parameter
# changed -- load_or_compute only busts its cache when something in
# key_params changes, so a pure logic/schema change with no matching
# key_params change would otherwise silently return a stale pickle missing
# the new fields (hit exactly this once: added 'r2_windowed_mean' to the
# return dict without bumping anything here, and the old cache got reused,
# crashing on the missing key downstream).
_TDR_COMPUTE_VERSION = 3

output_dir = './output/dimred_plots'
tdr_dir = os.path.join(output_dir, 'tdr')
os.makedirs(tdr_dir, exist_ok=True)


#%% ------------------------------------------------------------------
# Core TDR fitting: per-timepoint regression -> denoise -> orthogonalize
# ----------------------------------------------------------------------
def fit_tdr_axes(X, stim, choice, denoise_window_idx=None):
    """
    X : (n_neurons, n_trials, T), should be z-scored per neuron already
        (see process_session -- TDR regression coefficients aren't
        comparable across neurons with very different response scales
        otherwise, and the orthogonalization step assumes a common scale).
    stim, choice : (n_trials,) task variables, any numeric encoding
        (binary 0/1 is standard and what get_trial_labels gives you here).
    denoise_window_idx : slice, boolean array (len T), or None. Which time
        bins to average the per-timepoint regression coefficients over
        before orthogonalizing. None = average over the whole window.
        Typically passed in as a boolean mask resolved from
        denoise_window_range against the actual axis (see process_session).

    Returns dict:
        offset_axis, stim_axis, choice_axis : (n_neurons,) RAW (pre-
            orthogonalization) regression coefficients, denoise-window-
            averaged -- kept for sign-fixing and diagnostics.
        stim_axis_orth, choice_axis_orth : (n_neurons,) QR-orthogonalized,
            unit-norm, SIGN-FIXED (see _fix_axis_sign) axes -- these are
            what you actually project data onto.
        betas : (n_neurons, T, 3) raw per-timepoint coefficients
            [offset, stim, choice], for inspecting how encoding evolves
            over time before denoising.
        r2 : (T,) fraction of population variance the 3 regressors
            explain at each timepoint -- a diagnostic on the regression
            model itself, not on the axes.
    """
    n_neurons, n_trials, T = X.shape
    design = np.column_stack([np.ones(n_trials), stim, choice])  # (K, 3)

    betas = np.zeros((n_neurons, T, 3))
    r2 = np.zeros(T)
    for t in range(T):
        Y = X[:, :, t].T  # (K, n_neurons)
        beta, residuals, rank, sv = np.linalg.lstsq(design, Y, rcond=None)  # beta: (3, n_neurons)
        betas[:, t, :] = beta.T
        Y_hat = design @ beta
        ss_res = np.sum((Y - Y_hat) ** 2)
        ss_tot = np.sum((Y - Y.mean(axis=0, keepdims=True)) ** 2)
        r2[t] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    window = denoise_window_idx if denoise_window_idx is not None else slice(None)
    mean_betas = betas[:, window, :].mean(axis=1)  # (n_neurons, 3)
    offset_axis, stim_axis, choice_axis = mean_betas[:, 0], mean_betas[:, 1], mean_betas[:, 2]

    # QR-orthogonalize [stim_axis, choice_axis] against each other (Mante
    # et al. 2013 Methods): Q's columns are unit-norm and mutually
    # orthogonal, spanning the same 2D subspace as the raw axes
    task_axes = np.column_stack([stim_axis, choice_axis])  # (n_neurons, 2)
    Q, R = np.linalg.qr(task_axes)
    stim_axis_orth, choice_axis_orth = Q[:, 0].copy(), Q[:, 1].copy()

    # fix sign: QR's sign is arbitrary, but "projecting onto the stim axis
    # should be positive when stim is high" is not -- align the
    # orthogonalized axis with the RAW regression coefficient's sign so
    # projections are consistently interpretable, and so averaging across
    # independently-fit sessions doesn't cancel real signal
    stim_axis_orth = _fix_axis_sign(stim_axis_orth, stim_axis)
    choice_axis_orth = _fix_axis_sign(choice_axis_orth, choice_axis)

    return {'offset_axis': offset_axis, 'stim_axis': stim_axis, 'choice_axis': choice_axis,
            'stim_axis_orth': stim_axis_orth, 'choice_axis_orth': choice_axis_orth,
            'betas': betas, 'r2': r2}


def fit_tdr_axes_pca(X, stim, choice, n_components, denoise_window_idx=None, random_state=0):
    """
    PCA-denoised TDR (see module docstring for why): fits PCA on the
    pooled (trial x time) samples, regresses in that reduced space via
    fit_tdr_axes, then maps the resulting axes back to neuron space via
    the PCA loading matrix for interpretability.

    X : (n_neurons, n_trials, T), z-scored per neuron.
    n_components : int, number of PCs to reduce to before regression.

    Returns the same dict as fit_tdr_axes, PLUS:
        stim_axis_orth_neurons, choice_axis_orth_neurons : (n_neurons,)
            unit-norm axes mapped back to neuron space (pca.components_.T
            @ axis_pca_space, renormalized) -- these are what go in the
            saved npz for interpretability, but projections should still
            be computed in PCA space (see project_condition_pca below) to
            avoid a redundant, error-prone extra transform.
        pca : the fitted sklearn PCA object, needed to transform
            condition-mean trajectories into the same reduced space before
            projecting them onto the PCA-space axes.
    """
    n_neurons, n_trials, T = X.shape
    n_components = min(n_components, n_neurons, n_trials - 1)

    X_pooled = X.transpose(1, 2, 0).reshape(n_trials * T, n_neurons)  # (K*T, n_neurons)
    X_pooled = StandardScaler().fit_transform(X_pooled)
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X_pooled)

    X_pca = pca.transform(X_pooled).reshape(n_trials, T, n_components).transpose(2, 0, 1)  # (n_components, K, T)

    result = fit_tdr_axes(X_pca, stim, choice, denoise_window_idx=denoise_window_idx)

    for key in ('stim_axis_orth', 'choice_axis_orth'):
        axis_pca_space = result[key]  # (n_components,)
        axis_neuron_space = pca.components_.T @ axis_pca_space  # (n_neurons,)
        norm = np.linalg.norm(axis_neuron_space)
        result[f'{key}_neurons'] = axis_neuron_space / norm if norm > 0 else axis_neuron_space

    result['pca'] = pca
    result['n_pca_components'] = n_components
    result['pct_variance_explained'] = float(np.sum(pca.explained_variance_ratio_))
    result['X_pca'] = X_pca  # exposed so the significance test can reuse the PCA transform across shuffles
    return result


def compute_windowed_r2(X, stim, choice, window_mask):
    """
    Regression R^2 at each timepoint in window_mask, averaged -- the same
    metric TDR reports as 'mean windowed R^2', factored out here so the
    significance test below can call it repeatedly without redoing the
    full axis-fitting/orthogonalization machinery (irrelevant to a
    significance test on R^2 itself).

    X : (n_neurons_or_pcs, n_trials, T). window_mask : boolean, length T.
    """
    n_trials = X.shape[1]
    design = np.column_stack([np.ones(n_trials), stim, choice])
    r2_list = []
    for t in np.where(window_mask)[0]:
        Y = X[:, :, t].T
        beta, *_ = np.linalg.lstsq(design, Y, rcond=None)
        Y_hat = design @ beta
        ss_res = np.sum((Y - Y_hat) ** 2)
        ss_tot = np.sum((Y - Y.mean(axis=0, keepdims=True)) ** 2)
        r2_list.append(1 - ss_res / ss_tot if ss_tot > 0 else np.nan)
    return float(np.nanmean(r2_list))


def tdr_r2_significance_test(X, stim, choice, window_mask, n_shuffles=200, random_state=0):
    """
    Significance test for the windowed R^2 (previously missing -- Steps 4
    and 5 both got shuffle-based significance tests during development,
    this one didn't, despite being exactly the kind of small-sample
    regression where that matters). Null: jointly permute the (stim,
    choice) trial pairing (same permutation applied to both, so their
    real co-occurrence structure is preserved but decoupled from the
    neural data) -- tests "does knowing this trial's condition explain
    neural variance beyond chance," matching the same logic as Step 5's
    label-permutation MI test.

    Returns dict: real_r2, null_r2_mean, null_r2_std, p_value
    """
    if window_mask is None:
        window_mask = np.ones(X.shape[2], dtype=bool)
    real_r2 = compute_windowed_r2(X, stim, choice, window_mask)
    rng = np.random.default_rng(random_state)
    null_r2 = np.empty(n_shuffles)
    n_trials = len(stim)
    for i in range(n_shuffles):
        perm = rng.permutation(n_trials)
        null_r2[i] = compute_windowed_r2(X, stim[perm], choice[perm], window_mask)
    p_value = float((1 + np.sum(null_r2 >= real_r2)) / (1 + n_shuffles))
    return {'real_r2': real_r2, 'null_r2_mean': float(np.nanmean(null_r2)),
            'null_r2_std': float(np.nanstd(null_r2)), 'p_value': p_value}


def _fix_axis_sign(axis_orth, axis_raw):
    """Flip axis_orth's sign if it points opposite to the raw (pre-orthogonalized) axis."""
    return axis_orth if np.dot(axis_orth, axis_raw) >= 0 else -axis_orth


def project_onto_axis(X, axis):
    """
    X : (n_neurons, T) -- typically a condition-averaged trajectory.
    axis : (n_neurons,) unit-norm axis (e.g. stim_axis_orth).
    Returns (T,) scalar projection over time.
    """
    return axis @ X


#%% ------------------------------------------------------------------
# Load sessions (shallow)
# ----------------------------------------------------------------------
sessions, nSessions = filter_sessions(
    protocols=protocol, load_behaviordata=False, load_calciumdata=False,
    load_videodata=False, min_trials=100)
report_sessions(sessions)

params = InfoTheoryParams(
    binning=BinningParams(method='equipopulated', n_bins=4),   # unused by TDR itself, kept for get_trial_labels' signature
    bias=BiasCorrectionParams(panzeri_treves=False),            # unused by TDR
    parallel=ParallelParams(n_jobs=1, backend='loky', verbose=0),
    stim_var=stim_var, stim_binarize_threshold=stim_binarize_threshold,
    choice_var=choice_var, trial_mask_var=trial_mask_var,
)

if use_pca_denoise and pca_source == 'from_step1':
    if not os.path.exists(dimensionality_summary_path):
        raise FileNotFoundError(
            f"pca_source='from_step1' but {dimensionality_summary_path} doesn't exist -- "
            f"run run_dimensionality_estimate.py first, or set pca_source='fixed'.")
    df_dim = pd.read_csv(dimensionality_summary_path)
    df_dim_fa = df_dim[df_dim['method'] == 'fa']
    n_pca_components_by_group = (df_dim_fa.groupby(['area', 'label'])['n_components_optimal']
                                  .median().round().astype(int).clip(upper=max_pca_components).to_dict())
elif use_pca_denoise:
    n_pca_components_by_group = {}   # falls back to pca_components_fixed via .get() default below
else:
    n_pca_components_by_group = {}


def _get_n_pca(area_name, label_name):
    if pca_source == 'fixed':
        return pca_components_fixed
    return int(n_pca_components_by_group.get((area_name, label_name), pca_components_fixed))


#%% ------------------------------------------------------------------
# Per session: filter to requested groups, fit TDR axes + condition-
# averaged projections per group
# ----------------------------------------------------------------------
def process_session(ses, session_index, nSessions, calciumversion, qc_csv_path,
                     target_areas, target_labels, s_pre, s_post, binsize, t_pre, t_post,
                     params, denoise_window_range, min_neurons_per_group,
                     use_pca_denoise, n_pca_components_by_group, max_pca_components,
                     min_trials_per_condition, n_shuffles_significance,
                     random_state, tdr_dir, cache_dir, force_recompute):
    print(f'\n=== Session {session_index + 1}/{nSessions}: {ses.session_id} ===')

    key_params = dict(
        compute_version=_TDR_COMPUTE_VERSION,
        session_id=ses.session_id, protocol=ses.protocol, calciumversion=calciumversion,
        target_areas=target_areas, target_labels=target_labels,
        s_pre=s_pre, s_post=s_post, binsize=binsize, t_pre=t_pre, t_post=t_post,
        stim_var=params.stim_var, stim_binarize_threshold=params.stim_binarize_threshold,
        choice_var=params.choice_var, trial_mask_var=params.trial_mask_var,
        denoise_window_range=str(denoise_window_range), min_neurons_per_group=min_neurons_per_group,
        use_pca_denoise=use_pca_denoise, max_pca_components=max_pca_components,
        min_trials_per_condition=min_trials_per_condition, n_shuffles_significance=n_shuffles_significance,
        random_state=random_state, radius=50, depth_thr=300,
        qc_mtime=os.path.getmtime(qc_csv_path) if qc_csv_path and os.path.exists(qc_csv_path) else None,
    )
    path = cache_path(cache_dir, f'{ses.session_id}_tdr', **key_params)

    def _compute(ses=ses):
        tensor, axis, axis_label = compute_tensor_for_session(
            ses, calciumversion=calciumversion,
            t_pre=t_pre, t_post=t_post,
            s_pre=s_pre, s_post=s_post, binsize=binsize, keepraw=False)
        # tensor: (K trials, N neurons, T bins)

        stim, choice, mask = get_trial_labels(ses, params)
        tensor = tensor[mask, :, :]
        stim = stim[mask]
        choice = choice[mask] if choice is not None else np.zeros_like(stim)

        area, label, _ = get_area_label(ses.celldata)

        # V1/PM unl cells must be near a labeled cell (50um) and in layer 2/3
        # (depth<300um); labeled cells and cells outside V1/PM pass through
        # unaffected (see utils/cellselection_lib.py). Combined with the QC
        # pass/fail flag from run_qc_summary.py, and with the requested
        # (area, label) group restriction.
        idx_anat = filter_nearlabeled_layer23(ses, radius=50, depth_thr=300)
        idx_qc = load_qc_pass(ses, qc_csv_path=qc_csv_path)
        idx_target = filter_target_groups(area, label, target_areas, target_labels)
        idx_valid = idx_anat & idx_qc & idx_target
        print(f'  [{ses.session_id}] {np.sum(idx_valid)}/{len(idx_valid)} neurons kept '
              f'(anatomical: {np.sum(idx_anat)}, QC: {np.sum(idx_qc)}, target group: {np.sum(idx_target)})')

        tensor = tensor[:, idx_valid, :]
        area, label = area[idx_valid], label[idx_valid]

        # resolve denoise_window_range (axis units) into a boolean mask
        # against the ACTUAL axis this session/tensor produced -- robust to
        # binning convention (edges vs centers) rather than assuming a
        # fixed integer index range
        if denoise_window_range is not None:
            denoise_mask = (axis >= denoise_window_range[0]) & (axis <= denoise_window_range[1])
            if not np.any(denoise_mask):
                print(f'    WARNING [{ses.session_id}]: denoise_window_range={denoise_window_range} matches no '
                      f'bins on this session\'s axis (range: {axis.min():.1f} to {axis.max():.1f}) -- '
                      f'falling back to the whole window for this session.')
                denoise_mask = None
        else:
            denoise_mask = None

        group_results = {}
        fitted_group_keys = []
        for a in np.unique(area):
            for l in DEFAULT_LABEL_ORDER:
                idx_group = np.where((area == a) & (label == l))[0]
                n_neurons_group = len(idx_group)
                if n_neurons_group < min_neurons_per_group:
                    continue

                group_key = f'{a}_{l}'
                X = tensor[:, idx_group, :].transpose(1, 0, 2)  # (n_neurons, n_trials, T)

                # NaN handling -- same rationale as Step 2 (position-binned
                # windows leave NaN where a trial never reached that bin)
                frac_nan_per_trial = np.mean(np.isnan(X), axis=(0, 2))
                trial_keep = frac_nan_per_trial <= 0.5
                X = X[:, trial_keep, :]
                stim_g, choice_g = stim[trial_keep], choice[trial_keep]
                t_idx = np.arange(X.shape[2])
                for n in range(X.shape[0]):
                    for k in range(X.shape[1]):
                        trace = X[n, k, :]
                        valid = ~np.isnan(trace)
                        if valid.sum() == 0:
                            trace[:] = 0.0
                        elif valid.sum() < len(trace):
                            trace[:] = np.interp(t_idx, t_idx[valid], trace[valid])
                        X[n, k, :] = trace

                if X.shape[1] < 10 or len(np.unique(stim_g)) < 2:
                    print(f'    [{ses.session_id} {group_key}] skipping: too few trials or no '
                          f'stim variation ({X.shape[1]} trials, {len(np.unique(stim_g))} stim levels)')
                    continue

                # z-score each neuron across trials+time (TDR regression
                # coefficients need a common scale across neurons)
                mu = X.mean(axis=(1, 2), keepdims=True)
                sd = X.std(axis=(1, 2), keepdims=True)
                sd[sd == 0] = 1.0
                Xz = (X - mu) / sd

                print(f'    fitting TDR {group_key}: n_neurons={n_neurons_group}, n_trials={X.shape[1]}, '
                      f'PCA-denoise={"on" if use_pca_denoise else "off"}')

                if use_pca_denoise:
                    n_comp = min(_get_n_pca(a, l), max_pca_components)
                    axes = fit_tdr_axes_pca(Xz, stim_g, choice_g, n_components=n_comp,
                                             denoise_window_idx=denoise_mask, random_state=random_state)
                    print(f'    [{ses.session_id} {group_key}] PCA: {axes["n_pca_components"]} components, '
                          f'{axes["pct_variance_explained"]:.1%} variance retained; '
                          f'regression R^2 range over time: {np.nanmin(axes["r2"]):.2f}-{np.nanmax(axes["r2"]):.2f}')
                    # project condition means in the SAME (PCA) space the axes were fit in
                    pca = axes['pca']
                    stim_axis_for_proj = axes['stim_axis_orth']       # (n_pca_components,)
                    choice_axis_for_proj = axes['choice_axis_orth']   # (n_pca_components,)
                    stim_axis_out = axes['stim_axis_orth_neurons']    # (n_neurons,) for saving/interpretability
                    choice_axis_out = axes['choice_axis_orth_neurons']
                    X_for_significance = axes['X_pca']   # reuse the already-fit PCA transform, no refit needed
                else:
                    axes = fit_tdr_axes(Xz, stim_g, choice_g, denoise_window_idx=denoise_mask)
                    print(f'    [{ses.session_id} {group_key}] regression R^2 range over time: '
                          f'{np.nanmin(axes["r2"]):.2f}-{np.nanmax(axes["r2"]):.2f}')
                    pca = None
                    stim_axis_for_proj = axes['stim_axis_orth']
                    choice_axis_for_proj = axes['choice_axis_orth']
                    stim_axis_out = axes['stim_axis_orth']
                    choice_axis_out = axes['choice_axis_orth']
                    X_for_significance = Xz

                # significance test for the windowed R^2 (previously missing
                # -- see module docstring / null_hypotheses_framework.md):
                # is the windowed R^2 bigger than chance given this many
                # PCA dims/trials, via joint stim+choice label permutation
                sig_test = tdr_r2_significance_test(
                    X_for_significance, stim_g, choice_g, denoise_mask,
                    n_shuffles=n_shuffles_significance, random_state=random_state)
                sig_flag = '**' if sig_test['p_value'] < 0.05 else '  '
                print(f'    [{ses.session_id} {group_key}] windowed R^2 significance: '
                      f'real={sig_test["real_r2"]:.3f}, null={sig_test["null_r2_mean"]:.3f}, '
                      f'p={sig_test["p_value"]:.3f} {sig_flag}')

                # condition-averaged trajectories, projected onto both axes.
                # Conditions built from fewer than min_trials_per_condition
                # trials are dropped -- too noisy to trust (see module docstring).
                conditions = {}
                n_dropped_conditions = 0
                for stim_val in np.unique(stim_g):
                    for choice_val in np.unique(choice_g):
                        idx_c = (stim_g == stim_val) & (choice_g == choice_val)
                        n_trials_c = int(np.sum(idx_c))
                        if n_trials_c < min_trials_per_condition:
                            if n_trials_c > 0:
                                n_dropped_conditions += 1
                            continue
                        cond_mean = np.nanmean(Xz[:, idx_c, :], axis=1)  # (n_neurons, T)
                        if use_pca_denoise:
                            cond_mean_proj_space = pca.transform(cond_mean.T).T  # (n_pca_components, T)
                        else:
                            cond_mean_proj_space = cond_mean
                        conditions[f'stim{stim_val}_choice{choice_val}'] = {
                            'stim_proj': project_onto_axis(cond_mean_proj_space, stim_axis_for_proj),
                            'choice_proj': project_onto_axis(cond_mean_proj_space, choice_axis_for_proj),
                            'n_trials': n_trials_c,
                        }
                if n_dropped_conditions > 0:
                    print(f'    [{ses.session_id} {group_key}] dropped {n_dropped_conditions} condition(s) '
                          f'with < {min_trials_per_condition} trials')

                r2_windowed_mean = float(np.nanmean(axes['r2'][denoise_mask])) if denoise_mask is not None \
                    else float(np.nanmean(axes['r2']))
                group_results[group_key] = {
                    'stim_axis_orth': stim_axis_out, 'choice_axis_orth': choice_axis_out,
                    'r2': axes['r2'], 'r2_windowed_mean': r2_windowed_mean, 'conditions': conditions,
                    'r2_null_mean': sig_test['null_r2_mean'], 'r2_p_value': sig_test['p_value'],
                }
                fitted_group_keys.append(group_key)

        return {'group_results': group_results, 'fitted_group_keys': fitted_group_keys,
                'axis': axis, 'axis_label': axis_label}

    cache_hit = os.path.exists(path) and not force_recompute
    result = load_or_compute(path, _compute, force_recompute=force_recompute)

    npz_path = os.path.join(tdr_dir, f'tdr_{ses.session_id}.npz')
    npz_payload = {'axis': result['axis'], 'axis_label': result['axis_label']}
    for group_key, gres in result['group_results'].items():
        npz_payload[f'{group_key}_stim_axis'] = gres['stim_axis_orth']
        npz_payload[f'{group_key}_choice_axis'] = gres['choice_axis_orth']
        npz_payload[f'{group_key}_r2'] = gres['r2']
        npz_payload[f'{group_key}_r2_windowed_mean'] = gres['r2_windowed_mean']
        npz_payload[f'{group_key}_r2_null_mean'] = gres['r2_null_mean']
        npz_payload[f'{group_key}_r2_p_value'] = gres['r2_p_value']
        for cond_key, cres in gres['conditions'].items():
            npz_payload[f'{group_key}_{cond_key}_stim_proj'] = cres['stim_proj']
            npz_payload[f'{group_key}_{cond_key}_choice_proj'] = cres['choice_proj']
    np.savez(npz_path, **npz_payload)
    print(f'  [{ses.session_id}] saved TDR results to {npz_path} '
          f'({"loaded from cache" if cache_hit else "freshly computed"})')

    for attr in ('calciumdata', 'videodata', 'behaviordata'):
        if hasattr(ses, attr):
            delattr(ses, attr)

    return {'session_id': ses.session_id, 'groups': result['fitted_group_keys'],
            'group_results': result['group_results'], 'axis': result['axis'],
            'axis_label': result['axis_label']}


session_results = Parallel(n_jobs=n_jobs_sessions, backend='loky', verbose=10)(
    delayed(process_session)(
        ses, ises, nSessions, calciumversion, qc_csv_path,
        target_areas, target_labels, s_pre, s_post, binsize, t_pre, t_post,
        params, denoise_window_range, min_neurons_per_group,
        use_pca_denoise, n_pca_components_by_group, max_pca_components,
        min_trials_per_condition, n_shuffles_significance,
        random_state, tdr_dir, cache_dir, force_recompute)
    for ises, ses in enumerate(sessions)
)

print('\nDone. Per-session TDR files:')
for res in session_results:
    print(f"  {res['session_id']}: groups {res['groups']}")

#%% ------------------------------------------------------------------
# Quick-look figure: state-space plot (stim-axis vs choice-axis
# projection), one subplot per group, one line per (session, condition) --
# this is the direct encoding-structure counterpart to Step 2's
# unsupervised GPFA quick-look. Distinct stim/choice conditions get
# distinct linestyles so encoding separability is visible directly.
# ----------------------------------------------------------------------
LINESTYLES_BY_CONDITION = {}  # populated dynamically below, kept consistent within a figure


def get_linestyle(cond_key):
    styles = ['-', '--', ':', '-.']
    if cond_key not in LINESTYLES_BY_CONDITION:
        LINESTYLES_BY_CONDITION[cond_key] = styles[len(LINESTYLES_BY_CONDITION) % len(styles)]
    return LINESTYLES_BY_CONDITION[cond_key]


all_group_keys = sorted({g for res in session_results for g in res['groups']})

# FDR correction across ALL (session, group) R^2 significance tests --
# same rationale/procedure as Steps 4 and 5 (see run_jpca_computation.py
# and null_hypotheses_framework.md for why this matters).
sig_entries = [(res['session_id'], gk, res['group_results'][gk]['r2_p_value'])
               for res in session_results for gk in res['groups']]
fdr_significant = set()
if sig_entries:
    pvals = np.array([e[2] for e in sig_entries])
    n_tests = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresholds = (np.arange(1, n_tests + 1) / n_tests) * 0.05
    below = ranked <= thresholds
    n_sig = int(np.max(np.where(below)[0]) + 1) if np.any(below) else 0
    fdr_cutoff = ranked[n_sig - 1] if n_sig > 0 else 0.0
    fdr_significant = {(sig_entries[order[i]][0], sig_entries[order[i]][1]) for i in range(n_sig)}
    print(f'\nTDR R^2 significance (windowed, joint stim+choice label permutation): '
          f'{int(np.sum(pvals < 0.05))}/{n_tests} uncorrected p<0.05; '
          f'{n_sig}/{n_tests} survive FDR correction (p <= {fdr_cutoff:.4f})')
    for sid, gk, p in sig_entries:
        print(f'  {sid} {gk}: p={p:.3f} {"** FDR-sig" if (sid, gk) in fdr_significant else ""}')
if all_group_keys:
    n_groups = len(all_group_keys)
    fig, axes_arr = plt.subplots(1, n_groups, figsize=(4.5 * n_groups, 4.5), squeeze=False)
    axes_arr = axes_arr[0]

    for j, group_key in enumerate(all_group_keys):
        ax = axes_arr[j]
        any_plotted = False
        r2_windowed_means = []
        for res in session_results:
            if group_key not in res['groups']:
                continue
            gres = res['group_results'][group_key]
            r2_windowed_means.append(gres['r2_windowed_mean'])
            for cond_key, cres in gres['conditions'].items():
                ls = get_linestyle(cond_key)
                line, = ax.plot(cres['stim_proj'], cres['choice_proj'], linestyle=ls,
                                 alpha=0.7, linewidth=1.2,
                                 label=f"{res['session_id']} {cond_key} (n={cres['n_trials']})")
                ax.scatter(cres['stim_proj'][0], cres['choice_proj'][0], marker='o', s=20,
                           color=line.get_color())
                any_plotted = True

        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        # mean of the WINDOWED per-session R^2 (i.e. within denoise_window_range,
        # matching what the plotted axes were actually fit from), averaged
        # ACROSS all sessions with this group -- not just the last session
        # in the loop (both were bugs in the previous version)
        r2_title = np.mean(r2_windowed_means) if r2_windowed_means else np.nan
        n_fdr_sig = sum(1 for res in session_results if group_key in res['groups']
                        and (res['session_id'], group_key) in fdr_significant)
        ax.set_title(f'{group_key}\n(mean windowed R^2={r2_title:.2f}, n={len(r2_windowed_means)} sessions, '
                     f'{n_fdr_sig} FDR-sig)' if any_plotted else group_key, fontsize=9)
        ax.set_xlabel('stimulus axis projection', fontsize=8)
        ax.set_ylabel('choice axis projection', fontsize=8)
        if any_plotted:
            ax.legend(fontsize=5, loc='best')

    fig.suptitle('TDR state-space trajectories by condition (stim x choice), per session', y=1.03)
    fig.tight_layout()
    fig_path = os.path.join(output_dir, 'tdr_quicklook.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'\nSaved quick-look TDR figure to {fig_path}')

#%% ------------------------------------------------------------------
# R^2(t) diagnostic: the MEAN R^2 reported above can hide a real, much
# higher peak at a specific point in the trial if it's averaged together
# with a long uninformative baseline period -- this figure shows the full
# per-timepoint curve, one line per session, so you can check for that
# directly and use it to set `denoise_window_range` to just the informative
# window (which both raises achievable R^2 and gives cleaner axes) rather
# than averaging regression coefficients over the whole trial by default.
# ----------------------------------------------------------------------
if all_group_keys:
    fig2, axes2 = plt.subplots(1, n_groups, figsize=(4 * n_groups, 3.5), squeeze=False)
    axes2 = axes2[0]

    for j, group_key in enumerate(all_group_keys):
        ax = axes2[j]
        for res in session_results:
            if group_key not in res['groups']:
                continue
            r2_t = res['group_results'][group_key]['r2']
            x_axis = res['axis'][:len(r2_t)] if res['axis'] is not None else np.arange(len(r2_t))
            ax.plot(x_axis, r2_t, alpha=0.7, linewidth=1, label=res['session_id'])
        ax.axhline(0, color='gray', linewidth=0.5)
        ax.set_title(group_key, fontsize=9)
        ax.set_xlabel(session_results[0].get('axis_label', 'time/position bin'), fontsize=8)
        if j == 0:
            ax.set_ylabel('regression R^2', fontsize=8)
        ax.legend(fontsize=5, loc='best')

    fig2.suptitle('Regression R^2 over time/position, per session -- '
                  'check for a peak the whole-window mean might be hiding', y=1.03)
    fig2.tight_layout()
    fig2_path = os.path.join(output_dir, 'tdr_r2_over_time.png')
    fig2.savefig(fig2_path, dpi=150, bbox_inches='tight')
    print(f'Saved R^2(t) diagnostic figure to {fig2_path}')
