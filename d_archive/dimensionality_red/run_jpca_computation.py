# -*- coding: utf-8 -*-
"""
Dimensionality-reduction pipeline, STEP 4: jPCA (Churchland, Cunningham,
Kaufman et al. 2012) -- rotational dynamics, per (area, label) group.

This is the "computation" stage of the roadmap: Step 1 asked how many
dimensions the population needs, Step 2 (GPFA) described single-trial
trajectories without any hypothesis about their structure, Step 3 (TDR)
asked how much of the population's activity is explained by task
variables. jPCA asks a specific, falsifiable structural question: does
the population's condition-averaged dynamics contain a genuinely
ROTATIONAL component -- i.e. is there a 2D plane in which the state
rotates around the origin at roughly constant angular velocity, as
opposed to purely expanding/contracting or task-variable-locked motion?

Method (Churchland et al. 2012 Methods):
  1. PCA-reduce the condition-averaged (stim x choice) trajectories to
     `n_pca_components` dims (same Step-1-sourced convention as GPFA/TDR).
  2. Subtract the CROSS-CONDITION mean at each timepoint from every
     condition's trajectory -- this removes the condition-independent
     signal (e.g. a shared ramp present in every condition) so what's left
     isolates the condition-DEPENDENT dynamics, which is what's tested for
     rotational structure.
  3. Fit an unconstrained linear dynamical system dx/dt = M x via least
     squares on the (state, numerical-derivative) pairs pooled across all
     conditions and timepoints.
  4. Extract the skew-symmetric part M_skew = (M - M^T) / 2 -- a purely
     skew-symmetric matrix generates PURE rotation (no expansion/
     contraction) in its eigenplanes. `rotational_strength` (||M_skew||_F
     / ||M||_F) reports how much of the fitted dynamics is captured by
     this rotational component vs. the symmetric (expansive/contractive)
     part -- the headline diagnostic: values near 1 mean the dynamics are
     essentially rotational, values near 0 mean M is close to symmetric
     (no meaningful rotation) and the resulting plot below shouldn't be
     over-interpreted as showing "rotation."
  5. Eigendecompose M_skew (real skew-symmetric -> purely imaginary
     eigenvalues in conjugate pairs); the pair with largest |eigenvalue|
     (fastest rotation) defines the dominant rotational plane. Rotation
     direction is fixed to a consistent sign convention across sessions
     (see `_fix_rotation_direction`) -- the plane's absolute orientation
     is still session-specific/arbitrary (same caveat as GPFA's latent
     basis), but at least "clockwise vs counterclockwise" is comparable.

Uses the SAME group-selection convention as the rest of this pipeline:
filter_nearlabeled_layer23 (anatomical) + load_qc_pass (QC) +
filter_target_groups (which area/label groups you want) + the same
condition-building (stim x choice, min_trials_per_condition-guarded) as
run_tdr_encoding.py.

Produces, per session:
  - jpca_<session_id>.npz : per group, the jPCA plane basis (in PCA
    space), rotational_strength, dominant rotation frequency, and each
    condition's trajectory projected onto the plane (jPC1, jPC2 over time)
  - a quick-look figure per group: jPC1 vs jPC2 trajectory, one line per
    (session, condition), with rotational_strength annotated so you can
    see at a glance which sessions/groups actually show rotational
    structure vs. which don't
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

target_areas = None
target_labels = None

qc_csv_path = os.path.join('./output/dimred_plots', 'qc_per_cell.csv')

s_pre, s_post, binsize = -60, 80, 10
t_pre, t_post = -1, 2

stim_var = 'signal'
stim_binarize_threshold = 0
choice_var = 'lickResponse'
trial_mask_var = None

min_trials_per_condition = 10   # same guard as run_tdr_encoding.py -- see that script's docstring

# PCA reduction before jPCA -- same Step-1-sourced convention as GPFA/TDR.
# jPCA classically uses ~6-8 PCs (Churchland et al. 2012), but that assumes
# many more conditions/timepoints than typically available here. VALIDATED
# during development: with ~2-4 conditions x ~T position bins (i.e. FEWER
# samples than the d^2 free parameters in the unconstrained M being fit),
# pure noise gives rotational_strength ~0.85-0.90 on average -- the
# shuffle-test p-value (see fit_jpca_with_significance) is what actually
# distinguishes real rotation from this, NOT the raw rotational_strength
# number. Keeping n_pca_components small relative to n_conditions*T also
# helps directly -- consider capping max_pca_components at 3-4 if the
# p-values come back consistently non-significant.
pca_source = 'from_step1'
pca_components_fixed = 8
max_pca_components = 12
dimensionality_summary_path = os.path.join('./output/dimred_plots', 'dimensionality_optimal_summary.csv')

min_neurons_per_group = 10
n_shuffles = 200   # for the rotational_strength significance test -- see fit_jpca_with_significance
random_state = 0
n_jobs_sessions = -1

cache_dir = './output/dimred_cache'
force_recompute = False

# Bump this whenever _compute()'s internal logic or RETURN SCHEMA changes,
# even if no user-facing config parameter changed -- see run_tdr_encoding.py
# for why this matters (load_or_compute only busts its cache on a key_params
# change, not a logic/schema change).
_JPCA_COMPUTE_VERSION = 2   # v2: added phase-randomization null alongside time-permutation null

output_dir = './output/dimred_plots'
jpca_dir = os.path.join(output_dir, 'jpca')
os.makedirs(jpca_dir, exist_ok=True)


#%% ------------------------------------------------------------------
# Core jPCA fitting
# ----------------------------------------------------------------------
def fit_jpca(condition_trajectories):
    """
    condition_trajectories : dict of cond_key -> (n_pca_dims, T) array,
        ALREADY PCA-reduced condition-averaged trajectories.

    Returns dict:
        M : (d, d) fitted unconstrained dynamics matrix (dx/dt = M x)
        M_skew : (d, d) skew-symmetric part of M
        rotational_strength : ||M_skew||_F / ||M||_F, in [0, 1] -- the
            headline diagnostic (see module docstring)
        plane_basis : (d, 2) orthonormal basis for the dominant rotational
            plane, in PCA space
        frequency : dominant angular frequency (in units of 1/timestep)
        condition_projections : dict cond_key -> (2, T) trajectory
            projected onto plane_basis, mean-subtracted per the jPCA
            convention (condition-independent signal removed)
    """
    cond_keys = list(condition_trajectories.keys())
    trajs = np.stack([condition_trajectories[k] for k in cond_keys], axis=0)  # (n_cond, d, T)
    n_cond, d, T = trajs.shape

    # subtract cross-condition mean at each timepoint (isolates condition-
    # DEPENDENT dynamics, per Churchland et al. 2012 Methods)
    cross_cond_mean = trajs.mean(axis=0, keepdims=True)  # (1, d, T)
    trajs_centered = trajs - cross_cond_mean

    # guard against near-degenerate dimensions: the requested n_pca_components
    # was chosen for SINGLE-TRIAL variance (Step 1), which can legitimately
    # exceed how many dims carry real CONDITION-AVERAGED, mean-subtracted
    # dynamics. Validated during development: including even a couple of
    # near-zero-variance dims here inflates rotational_strength toward
    # ~0.8-0.9 REGARDLESS of whether the real dynamics is rotational,
    # because least-squares fits large spurious coefficients on
    # near-singular directions. Drop dims whose variance is below 1% of
    # the top dim's before fitting M.
    dim_var = trajs_centered.var(axis=(0, 2))  # (d,)
    keep_dims = dim_var >= 0.01 * dim_var.max()
    n_dropped_dims = int(np.sum(~keep_dims))
    if n_dropped_dims > 0:
        trajs_centered = trajs_centered[:, keep_dims, :]
        d = trajs_centered.shape[1]

    # numerical derivative (central differences, dropping the edge points
    # where it isn't defined) and matching state values, pooled across
    # conditions and time
    X_list, dX_list = [], []
    for c in range(n_cond):
        x = trajs_centered[c].T  # (T, d)
        dx = (x[2:, :] - x[:-2, :]) / 2.0  # central difference, (T-2, d)
        X_list.append(x[1:-1, :])
        dX_list.append(dx)
    X = np.concatenate(X_list, axis=0)   # (n_samples, d)
    dX = np.concatenate(dX_list, axis=0)  # (n_samples, d)

    # fit dX ≈ X @ M.T  =>  lstsq solves for M.T directly
    M_T, *_ = np.linalg.lstsq(X, dX, rcond=None)
    M = M_T.T

    M_skew = (M - M.T) / 2.0
    norm_M = np.linalg.norm(M, ord='fro')
    rotational_strength = float(np.linalg.norm(M_skew, ord='fro') / norm_M) if norm_M > 0 else 0.0

    eigvals, eigvecs = np.linalg.eig(M_skew)
    # purely imaginary eigenvalues come in +-i*omega conjugate pairs (plus
    # possibly zero eigenvalues if d is odd); pick the pair with the
    # largest |omega| = the fastest/dominant rotation
    order = np.argsort(-np.abs(eigvals.imag))
    top_idx = order[0]
    omega = float(np.abs(eigvals[top_idx].imag))
    v = eigvecs[:, top_idx]  # complex eigenvector, eigenvalue +-i*omega

    v_re, v_im = v.real, v.imag
    # Gram-Schmidt orthonormalize (theoretically already near-orthogonal
    # for a true skew-symmetric eigenpair, but this makes it robust to
    # floating point / arbitrary overall complex phase from eig())
    v_re = v_re / (np.linalg.norm(v_re) + 1e-12)
    v_im = v_im - (v_im @ v_re) * v_re
    v_im = v_im / (np.linalg.norm(v_im) + 1e-12)
    plane_basis = np.column_stack([v_re, v_im])  # (d, 2)

    plane_basis = _fix_rotation_direction(plane_basis, M_skew)

    condition_projections = {}
    for i, k in enumerate(cond_keys):
        condition_projections[k] = plane_basis.T @ trajs_centered[i]  # (2, T)

    return {'M': M, 'M_skew': M_skew, 'rotational_strength': rotational_strength,
            'plane_basis': plane_basis, 'frequency': omega, 'n_dropped_dims': n_dropped_dims,
            'condition_projections': condition_projections}


def _shuffle_time_within_condition(condition_trajectories, rng):
    """
    Null-hypothesis surrogate #1 (general-purpose): independently permute
    the T time labels WITHIN each condition. This destroys the local
    (state, derivative) relationship the fit actually depends on (dx/dt is
    computed between now-randomly-adjacent timepoints) while preserving
    each condition's marginal distribution of visited states -- a fairer
    null than i.i.d. noise, since it's built from the real data's own
    amplitude/covariance structure. Catches gross overfitting (more free
    parameters in M than samples).

    NOTE: circularly SHIFTING (rolling) each condition instead of fully
    permuting was tried during development and rejected -- a circular
    shift only relabels which absolute time index holds a given (x, dx)
    pair without changing which points are locally adjacent, so it barely
    perturbs the fit at all and produces an almost-real-value null (not a
    valid null). Full permutation is what actually breaks the fitted
    relationship.

    LIMITATION (why null #2 below also exists): this null destroys ALL
    temporal smoothness, not just cross-dimension coordination. Elsayed &
    Cunningham (2017)-style confound: two same-frequency oscillatory
    signals trace an ellipse (look exactly like rotation to this fit) for
    ANY relative phase, with zero true cross-dimension coordination
    required. A null that destroys smoothness entirely can under-detect
    this specific confound, because it makes even the null surrogates
    look nothing like smooth oscillatory data. Validated during
    development: on a synthetic phase-shifted-copy "traveling wave" (a
    trivial, uncoordinated phenomenon that nonetheless gives
    rotational_strength=0.84), this null's mean was 0.77 -- correct in
    that instance, but visibly less strict than null #2's 0.85, i.e. a
    less specific test for this exact confound.
    """
    shuffled = {}
    for k, traj in condition_trajectories.items():
        T = traj.shape[1]
        perm = rng.permutation(T)
        shuffled[k] = traj[:, perm]
    return shuffled


def _phase_randomize_conditions(condition_trajectories, rng):
    """
    Null-hypothesis surrogate #2 (Elsayed & Cunningham-style, targeted):
    for each PCA dimension independently, randomize its Fourier PHASE
    while preserving its power spectrum (hence its own autocorrelation/
    smoothness) exactly. This destroys cross-dimension timing
    relationships -- the specific thing true rotation requires -- while
    preserving exactly the "simple" statistic (each dimension's own
    smoothness) that a trivial phenomenon like a traveling wave or a
    shared-frequency-but-independent-phase oscillation would also have.

    Validated during development: on a synthetic "traveling wave" (dim 2 =
    dim 1 time-shifted by a lag -- a trivial phenomenon, no coordinated
    dynamics, yet rotational_strength=0.84), this null's mean was 0.85 --
    correctly flags the result as unremarkable, and is the more targeted/
    stricter test for this confound than full time permutation (whose
    null mean was a visibly weaker 0.77 on the same data).
    """
    shuffled = {}
    for k, traj in condition_trajectories.items():
        d, T = traj.shape
        out = np.zeros_like(traj)
        for i in range(d):
            f = np.fft.rfft(traj[i])
            mag = np.abs(f)
            phases = rng.uniform(0, 2 * np.pi, len(f))
            phases[0] = 0  # keep DC real
            if T % 2 == 0:
                phases[-1] = 0  # keep Nyquist real for even-length signals
            out[i] = np.fft.irfft(mag * np.exp(1j * phases), n=T)
        shuffled[k] = out
    return shuffled


def fit_jpca_with_significance(condition_trajectories, n_shuffles=200, random_state=0):
    """
    fit_jpca, PLUS TWO shuffle-based significance tests on
    rotational_strength -- see null_hypotheses_framework.md for why both
    are needed, not just one:
      1. Time permutation (_shuffle_time_within_condition): general-purpose,
         catches gross overfitting.
      2. Phase randomization (_phase_randomize_conditions): targeted at the
         specific confound of band-limited/oscillatory signals looking
         rotational regardless of true cross-dimension coordination.

    A result should only be treated as evidence of genuine rotational
    structure if it clears BOTH nulls (both p-values < 0.05), not just the
    more permissive one -- see `significant_both` below.

    IMPORTANT: validated during development that at realistic sample sizes
    for this pipeline (few conditions x ~T position bins, PCA-reduced to
    d~6-12 dims -- i.e. FEWER samples than the d^2 free parameters in the
    unconstrained M being fit), rotational_strength alone is NOT
    trustworthy: pure white noise at this sample size gives
    rotational_strength ~0.85-0.90 on average, indistinguishable by eye
    from genuine rotation. The p-values here are what actually tell you
    whether a given rotational_strength means anything; the raw number by
    itself does not.

    Returns the same dict as fit_jpca, PLUS:
        null_rotational_strength_mean/std, p_value : from the time-
            permutation null (kept under the original names for backward
            compatibility with existing summary/plotting code)
        null_rotational_strength_mean_phase/std_phase, p_value_phase :
            from the phase-randomization null
        significant_both : bool, True iff BOTH p-values < 0.05
    """
    result = fit_jpca(condition_trajectories)
    rng = np.random.default_rng(random_state)

    null_rs_perm = np.empty(n_shuffles)
    null_rs_phase = np.empty(n_shuffles)
    for i in range(n_shuffles):
        null_rs_perm[i] = fit_jpca(_shuffle_time_within_condition(condition_trajectories, rng))['rotational_strength']
        null_rs_phase[i] = fit_jpca(_phase_randomize_conditions(condition_trajectories, rng))['rotational_strength']

    p_perm = float((1 + np.sum(null_rs_perm >= result['rotational_strength'])) / (1 + n_shuffles))
    p_phase = float((1 + np.sum(null_rs_phase >= result['rotational_strength'])) / (1 + n_shuffles))

    result['null_rotational_strength_mean'] = float(null_rs_perm.mean())
    result['null_rotational_strength_std'] = float(null_rs_perm.std())
    result['p_value'] = p_perm
    result['null_rotational_strength_mean_phase'] = float(null_rs_phase.mean())
    result['null_rotational_strength_std_phase'] = float(null_rs_phase.std())
    result['p_value_phase'] = p_phase
    result['significant_both'] = bool(p_perm < 0.05 and p_phase < 0.05)
    return result


def _fix_rotation_direction(plane_basis, M_skew):
    """
    Fix rotation direction to a consistent sign convention (counter-
    clockwise in the returned (jPC1, jPC2) plane) across independently-
    fit sessions -- otherwise "clockwise in session A" and "counter-
    clockwise in session B" could be the SAME underlying rotation just
    reported with flipped axes, making cross-session comparison
    meaningless even for the one thing (direction) that should be
    comparable. The plane's absolute orientation remains session-specific
    regardless (same caveat as GPFA's latent basis).
    """
    v1, v2 = plane_basis[:, 0], plane_basis[:, 1]
    # d(v1)/dt under the fitted dynamics should point along +v2 for a
    # counterclockwise rotation in the (v1, v2) plane; flip v2 if not
    dv1 = M_skew @ v1
    if np.dot(dv1, v2) < 0:
        plane_basis = plane_basis.copy()
        plane_basis[:, 1] = -plane_basis[:, 1]
    return plane_basis


#%% ------------------------------------------------------------------
# Load sessions (shallow) + Step 1 dimensionality summary (if used)
# ----------------------------------------------------------------------
sessions, nSessions = filter_sessions(
    protocols=protocol, load_behaviordata=False, load_calciumdata=False,
    load_videodata=False, min_trials=100)
report_sessions(sessions)

params = InfoTheoryParams(
    binning=BinningParams(method='equipopulated', n_bins=4),
    bias=BiasCorrectionParams(panzeri_treves=False),
    parallel=ParallelParams(n_jobs=1, backend='loky', verbose=0),
    stim_var=stim_var, stim_binarize_threshold=stim_binarize_threshold,
    choice_var=choice_var, trial_mask_var=trial_mask_var,
)

if pca_source == 'from_step1':
    if not os.path.exists(dimensionality_summary_path):
        raise FileNotFoundError(
            f"pca_source='from_step1' but {dimensionality_summary_path} doesn't exist -- "
            f"run run_dimensionality_estimate.py first, or set pca_source='fixed'.")
    df_dim = pd.read_csv(dimensionality_summary_path)
    df_dim_fa = df_dim[df_dim['method'] == 'fa']
    n_pca_components_by_group = (df_dim_fa.groupby(['area', 'label'])['n_components_optimal']
                                  .median().round().astype(int).clip(upper=max_pca_components).to_dict())
else:
    n_pca_components_by_group = {}


def _get_n_pca(area_name, label_name):
    if pca_source == 'fixed':
        return pca_components_fixed
    return int(n_pca_components_by_group.get((area_name, label_name), pca_components_fixed))


#%% ------------------------------------------------------------------
# Per session: filter to requested groups, fit jPCA per group
# ----------------------------------------------------------------------
def process_session(ses, session_index, nSessions, calciumversion, qc_csv_path,
                     target_areas, target_labels, s_pre, s_post, binsize, t_pre, t_post,
                     params, min_trials_per_condition, min_neurons_per_group,
                     max_pca_components, n_shuffles, random_state, jpca_dir, cache_dir, force_recompute):
    print(f'\n=== Session {session_index + 1}/{nSessions}: {ses.session_id} ===')

    key_params = dict(
        compute_version=_JPCA_COMPUTE_VERSION,
        session_id=ses.session_id, protocol=ses.protocol, calciumversion=calciumversion,
        target_areas=target_areas, target_labels=target_labels,
        s_pre=s_pre, s_post=s_post, binsize=binsize, t_pre=t_pre, t_post=t_post,
        stim_var=params.stim_var, stim_binarize_threshold=params.stim_binarize_threshold,
        choice_var=params.choice_var, trial_mask_var=params.trial_mask_var,
        min_trials_per_condition=min_trials_per_condition, min_neurons_per_group=min_neurons_per_group,
        pca_source=pca_source, pca_components_fixed=pca_components_fixed, max_pca_components=max_pca_components,
        n_shuffles=n_shuffles, random_state=random_state, radius=50, depth_thr=300,
        qc_mtime=os.path.getmtime(qc_csv_path) if qc_csv_path and os.path.exists(qc_csv_path) else None,
    )
    path = cache_path(cache_dir, f'{ses.session_id}_jpca', **key_params)

    def _compute(ses=ses):
        tensor, axis, axis_label = compute_tensor_for_session(
            ses, calciumversion=calciumversion,
            t_pre=t_pre, t_post=t_post,
            s_pre=s_pre, s_post=s_post, binsize=binsize, keepraw=False)

        stim, choice, mask = get_trial_labels(ses, params)
        tensor = tensor[mask, :, :]
        stim = stim[mask]
        choice = choice[mask] if choice is not None else np.zeros_like(stim)

        area, label, _ = get_area_label(ses.celldata)

        idx_anat = filter_nearlabeled_layer23(ses, radius=50, depth_thr=300)
        idx_qc = load_qc_pass(ses, qc_csv_path=qc_csv_path)
        idx_target = filter_target_groups(area, label, target_areas, target_labels)
        idx_valid = idx_anat & idx_qc & idx_target
        print(f'  [{ses.session_id}] {np.sum(idx_valid)}/{len(idx_valid)} neurons kept '
              f'(anatomical: {np.sum(idx_anat)}, QC: {np.sum(idx_qc)}, target group: {np.sum(idx_target)})')

        tensor = tensor[:, idx_valid, :]
        area, label = area[idx_valid], label[idx_valid]

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

                # NaN handling -- same rationale as Steps 2-3
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

                # z-score per neuron (consistent scale for PCA, same as TDR)
                mu = X.mean(axis=(1, 2), keepdims=True)
                sd = X.std(axis=(1, 2), keepdims=True)
                sd[sd == 0] = 1.0
                Xz = (X - mu) / sd

                # condition-averaged trajectories (stim x choice), dropping
                # any condition with too few trials -- same guard as TDR
                conditions_raw = {}
                for stim_val in np.unique(stim_g):
                    for choice_val in np.unique(choice_g):
                        idx_c = (stim_g == stim_val) & (choice_g == choice_val)
                        if np.sum(idx_c) < min_trials_per_condition:
                            continue
                        conditions_raw[f'stim{stim_val}_choice{choice_val}'] = np.nanmean(Xz[:, idx_c, :], axis=1)  # (n_neurons, T)

                if len(conditions_raw) < 2:
                    print(f'    [{ses.session_id} {group_key}] skipping: fewer than 2 conditions '
                          f'survive the min_trials_per_condition={min_trials_per_condition} guard')
                    continue

                n_comp = min(_get_n_pca(a, l), max_pca_components, n_neurons_group - 1)
                # fit PCA on all conditions' timepoints pooled
                all_cond_concat = np.concatenate(list(conditions_raw.values()), axis=1).T  # (n_cond*T, n_neurons)
                pca = PCA(n_components=n_comp, random_state=random_state)
                pca.fit(all_cond_concat)

                conditions_pca = {k: pca.transform(v.T).T for k, v in conditions_raw.items()}  # (n_pca, T) each

                print(f'    fitting jPCA {group_key}: n_neurons={n_neurons_group}, '
                      f'{len(conditions_pca)} conditions, {n_comp} PCs '
                      f'({100 * np.sum(pca.explained_variance_ratio_):.0f}% var)')
                jpca_result = fit_jpca_with_significance(conditions_pca, n_shuffles=n_shuffles,
                                                          random_state=random_state)
                if jpca_result['n_dropped_dims'] > 0:
                    print(f'    [{ses.session_id} {group_key}] dropped {jpca_result["n_dropped_dims"]}/{n_comp} '
                          f'near-zero-variance PCA dims before fitting M (condition-averaged, mean-subtracted '
                          f'variance too small relative to the top dim -- see fit_jpca docstring)')
                sig_perm = '*' if jpca_result['p_value'] < 0.05 else ' '
                sig_phase = '*' if jpca_result['p_value_phase'] < 0.05 else ' '
                both_flag = '** BOTH' if jpca_result['significant_both'] else '  '
                print(f'    [{ses.session_id} {group_key}] rotational_strength={jpca_result["rotational_strength"]:.2f}, '
                      f'p_permutation={jpca_result["p_value"]:.3f}{sig_perm}, '
                      f'p_phase_randomization={jpca_result["p_value_phase"]:.3f}{sig_phase} {both_flag}, '
                      f'frequency={jpca_result["frequency"]:.3f}')

                group_results[group_key] = {
                    'rotational_strength': jpca_result['rotational_strength'],
                    'null_rotational_strength_mean': jpca_result['null_rotational_strength_mean'],
                    'null_rotational_strength_std': jpca_result['null_rotational_strength_std'],
                    'p_value': jpca_result['p_value'],
                    'null_rotational_strength_mean_phase': jpca_result['null_rotational_strength_mean_phase'],
                    'null_rotational_strength_std_phase': jpca_result['null_rotational_strength_std_phase'],
                    'p_value_phase': jpca_result['p_value_phase'],
                    'significant_both': jpca_result['significant_both'],
                    'frequency': jpca_result['frequency'],
                    'condition_projections': jpca_result['condition_projections'],
                    'n_conditions': len(conditions_pca), 'n_pca_components': n_comp,
                    'pct_variance_explained': float(np.sum(pca.explained_variance_ratio_)),
                }
                fitted_group_keys.append(group_key)

        return {'group_results': group_results, 'fitted_group_keys': fitted_group_keys,
                'axis': axis, 'axis_label': axis_label}

    cache_hit = os.path.exists(path) and not force_recompute
    result = load_or_compute(path, _compute, force_recompute=force_recompute)

    npz_path = os.path.join(jpca_dir, f'jpca_{ses.session_id}.npz')
    npz_payload = {'axis': result['axis'], 'axis_label': result['axis_label']}
    for group_key, gres in result['group_results'].items():
        npz_payload[f'{group_key}_rotational_strength'] = gres['rotational_strength']
        npz_payload[f'{group_key}_null_rotational_strength_mean'] = gres['null_rotational_strength_mean']
        npz_payload[f'{group_key}_p_value'] = gres['p_value']
        npz_payload[f'{group_key}_null_rotational_strength_mean_phase'] = gres['null_rotational_strength_mean_phase']
        npz_payload[f'{group_key}_p_value_phase'] = gres['p_value_phase']
        npz_payload[f'{group_key}_significant_both'] = gres['significant_both']
        npz_payload[f'{group_key}_frequency'] = gres['frequency']
        for cond_key, proj in gres['condition_projections'].items():
            npz_payload[f'{group_key}_{cond_key}_jpc'] = proj  # (2, T)
    np.savez(npz_path, **npz_payload)
    print(f'  [{ses.session_id}] saved jPCA results to {npz_path} '
          f'({"loaded from cache" if cache_hit else "freshly computed"})')

    for attr in ('calciumdata', 'videodata', 'behaviordata'):
        if hasattr(ses, attr):
            delattr(ses, attr)

    return {'session_id': ses.session_id, 'groups': result['fitted_group_keys'],
            'group_results': result['group_results']}


session_results = Parallel(n_jobs=n_jobs_sessions, backend='loky', verbose=10)(
    delayed(process_session)(
        ses, ises, nSessions, calciumversion, qc_csv_path,
        target_areas, target_labels, s_pre, s_post, binsize, t_pre, t_post,
        params, min_trials_per_condition, min_neurons_per_group,
        max_pca_components, n_shuffles, random_state, jpca_dir, cache_dir, force_recompute)
    for ises, ses in enumerate(sessions)
)

print('\nDone. Per-session jPCA files:')
for res in session_results:
    print(f"  {res['session_id']}: groups {res['groups']}")

print('\nRotational strength summary (0=no rotation, 1=purely rotational), per session x group -- '
      'TWO p-values per test, see null_hypotheses_framework.md for why:')
for res in session_results:
    for group_key in res['groups']:
        gres = res['group_results'][group_key]
        sig_perm = '*' if gres['p_value'] < 0.05 else ' '
        sig_phase = '*' if gres['p_value_phase'] < 0.05 else ' '
        print(f"  {res['session_id']} {group_key}: rotational_strength={gres['rotational_strength']:.2f}, "
              f"p_perm={gres['p_value']:.3f}{sig_perm}, p_phase={gres['p_value_phase']:.3f}{sig_phase}, "
              f"n_conditions={gres['n_conditions']}, PCA var={gres['pct_variance_explained']:.0%}")


def _bh_fdr(pvals, alpha=0.05):
    """Benjamini-Hochberg FDR correction. Returns (n_significant, cutoff_pvalue, order)."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha
    below = ranked <= thresholds
    n_sig = int(np.max(np.where(below)[0]) + 1) if np.any(below) else 0
    cutoff = ranked[n_sig - 1] if n_sig > 0 else 0.0
    return n_sig, cutoff, order


# --- Multiple-comparisons correction (Benjamini-Hochberg FDR), run
# SEPARATELY on each null's p-values across ALL session x group tests.
# Eyeballing individual p<0.05 flags across a whole grid like this is
# exactly the situation FDR correction exists for: with ~N tests at
# alpha=0.05, ~0.05*N "significant" hits are expected by chance alone even
# under the pure null. The FINAL verdict (survives_fdr_both, below)
# requires a result to survive FDR correction under BOTH nulls
# independently -- the most conservative, most defensible criterion this
# script can offer for "this is genuine rotational structure."
all_entries = [(res['session_id'], group_key,
                res['group_results'][group_key]['p_value'],
                res['group_results'][group_key]['p_value_phase'])
               for res in session_results for group_key in res['groups']]

if all_entries:
    alpha = 0.05
    pvals_perm = np.array([e[2] for e in all_entries])
    pvals_phase = np.array([e[3] for e in all_entries])
    n_tests = len(all_entries)

    n_sig_perm, cutoff_perm, order_perm = _bh_fdr(pvals_perm, alpha)
    n_sig_phase, cutoff_phase, order_phase = _bh_fdr(pvals_phase, alpha)
    fdr_sig_perm = {(all_entries[order_perm[i]][0], all_entries[order_perm[i]][1]) for i in range(n_sig_perm)}
    fdr_sig_phase = {(all_entries[order_phase[i]][0], all_entries[order_phase[i]][1]) for i in range(n_sig_phase)}
    fdr_sig_both = fdr_sig_perm & fdr_sig_phase

    print(f'\nMultiple-comparisons correction (Benjamini-Hochberg FDR, alpha={alpha}, n_tests={n_tests}):')
    print(f'  Permutation null:        {int(np.sum(pvals_perm < alpha))}/{n_tests} uncorrected p<{alpha}; '
          f'{n_sig_perm}/{n_tests} survive FDR (p <= {cutoff_perm:.4f})')
    print(f'  Phase-randomization null: {int(np.sum(pvals_phase < alpha))}/{n_tests} uncorrected p<{alpha}; '
          f'{n_sig_phase}/{n_tests} survive FDR (p <= {cutoff_phase:.4f})')
    print(f'  FINAL VERDICT -- survive FDR correction under BOTH nulls: {len(fdr_sig_both)}/{n_tests}')
    if fdr_sig_both:
        print('  Surviving (session, group):')
        for sid, gk in sorted(fdr_sig_both):
            print(f'    {sid}, {gk}')
    else:
        print('  No tests survive FDR correction under both nulls -- no session/group in this dataset shows '
              'evidence for genuine rotational dynamics that survives the full battery of controls.')

#%% ------------------------------------------------------------------
# Quick-look figure: jPC1 vs jPC2 trajectory, one subplot per group, one
# line per (session, condition). Rotational_strength is annotated per
# session in the legend -- LOW values mean the "circular" appearance (if
# any) isn't backed by genuinely rotational dynamics and shouldn't be
# over-interpreted.
# ----------------------------------------------------------------------
all_group_keys = sorted({g for res in session_results for g in res['groups']})
if all_group_keys:
    n_groups = len(all_group_keys)
    fig, axes_arr = plt.subplots(1, n_groups, figsize=(4.5 * n_groups, 4.5), squeeze=False)
    axes_arr = axes_arr[0]

    for j, group_key in enumerate(all_group_keys):
        ax = axes_arr[j]
        any_plotted = False
        for res in session_results:
            if group_key not in res['groups']:
                continue
            gres = res['group_results'][group_key]
            rs = gres['rotational_strength']
            p_perm = gres['p_value']
            p_phase = gres['p_value_phase']
            both = '**' if gres['significant_both'] else ('*' if (p_perm < 0.05 or p_phase < 0.05) else 'ns')
            for cond_key, proj in gres['condition_projections'].items():
                line, = ax.plot(proj[0, :], proj[1, :], alpha=0.7, linewidth=1.2,
                                 label=f"{res['session_id']} (R={rs:.2f}, p_perm={p_perm:.2f}, "
                                       f"p_phase={p_phase:.2f} {both})"
                                 if cond_key == list(gres['condition_projections'])[0] else None)
                ax.scatter(proj[0, 0], proj[1, 0], marker='o', s=20, color=line.get_color())
                any_plotted = True

        ax.axhline(0, color='gray', linewidth=0.5)
        ax.axvline(0, color='gray', linewidth=0.5)
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_title(group_key, fontsize=9)
        ax.set_xlabel('jPC1', fontsize=8)
        ax.set_ylabel('jPC2', fontsize=8)
        if any_plotted:
            ax.legend(fontsize=6, loc='best')

    fig.suptitle('jPCA rotational-plane trajectories, per session (R = rotational_strength; '
                 'p_perm = time-permutation null, p_phase = phase-randomization null; '
                 '**=significant under both, *=one only, ns=neither)', y=1.03)
    fig.tight_layout()
    fig_path = os.path.join(output_dir, 'jpca_quicklook.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'\nSaved quick-look jPCA figure to {fig_path}')
