# -*- coding: utf-8 -*-
"""
Dimensionality-reduction pipeline, STEP 2: GPFA single-trial trajectories,
per (area, label) group.

*** DATA-FORMAT ASSUMPTION -- READ BEFORE RUNNING ***
GPFA (Yu et al. 2009) is defined for point-process (spike count) data: its
observation model is a linear-Gaussian mapping from a smooth GP latent onto
per-neuron spike counts. This script uses `elephant.gpfa.GPFA`, the
standard/validated implementation, which expects spike trains (neo.SpikeTrain
objects), not continuous dF/F or deconvolved traces directly.

Since this pipeline uses calciumversion='deconv' (spike-deconvolved calcium,
already a spike-count PROXY rather than raw fluorescence), this script treats
each time bin's deconvolved value as a spike count and manufactures synthetic
neo.SpikeTrain objects with that many spikes placed at the bin center -- a
common, defensible way to feed deconvolved calcium into spike-based tools,
but a modeling choice, not a neutral default. Before trusting the output:
  - Sanity-check that deconvolved values are non-negative and roughly
    countlike (see `report_deconv_sanity` below, run automatically per
    session and printed to the console).
  - Consider whether `bin_size` (set from your tensor's binsize below)
    is coarse enough that per-bin values look like small integer counts
    rather than continuous amplitudes -- if not, GPFA's Gaussian-tail
    approximation to Poisson counts is on shakier ground.
If this assumption doesn't hold for your data, set `use_elephant_gpfa =
False` below to fall back to a lighter-weight approximation: per-trial
Gaussian temporal smoothing followed by a single shared FA basis fit
across all (smoothed) timepoints and trials pooled together. This is NOT
real GPFA (no single-trial GP noise model, no proper cross-validated
single-trial log-likelihood) but gives you a smooth single-trial
trajectory in a low-D space with no extra dependencies, useful as a
placeholder while you decide whether the elephant/spike-train route is
worth the extra machinery for your data.

Requires: pip install elephant neo quantities   (only if use_elephant_gpfa=True)

Uses the SAME group-selection convention as the rest of this pipeline:
filter_nearlabeled_layer23 (anatomical) + load_qc_pass (QC) +
filter_target_groups (which area/label groups you want).

x_dim (latent dimensionality) per group defaults to the FA optimal
n_components found in Step 1 (dimensionality_optimal_summary.csv), capped
at max_xdim for cost -- run run_dimensionality_estimate.py first, or set
xdim_source='fixed' with a manual xdim_fixed value to skip that dependency.

Produces, per session:
  - trajectories_<session_id>.npz : dict of group_key -> (n_trials, x_dim, T) array,
    plus the time axis and trial metadata needed to relate trajectories
    back to task variables in later (encoding/computation) stages
  - one trajectory figure per group: condition-averaged trajectory in the
    top 2 latent dimensions (colored by stimulus/choice if available)
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
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import FactorAnalysis

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

# Which (area, label) groups to include -- same convention as Step 1.
target_areas = None
target_labels = None

qc_csv_path = os.path.join('./output/infotheory_plots', 'qc_per_cell.csv')

# tensor window (spatial protocol; match Step 1 / the temporal info scripts
# if you want trajectories over the same window as everything else)
s_pre, s_post, binsize = -60, 80, 10
t_pre, t_post = -1, 2   # used instead of s_pre/s_post for time-locked protocols (IM/GR/GN)

use_elephant_gpfa = True   # False -> lightweight smoothed-FA fallback, see module docstring
gpfa_bin_size_ms = 20      # nominal duration assigned to EACH ORIGINAL tensor bin (position or
                           # time bin) for the elephant spike-time construction -- with the fix
                           # below, this now maps 1:1 to GPFA's own output bins (no re-binning),
                           # so its numeric value has no real "ms" meaning for position-locked
                           # data; it just needs to be self-consistent, which it is

# latent dimensionality per group: 'from_step1' reads
# dimensionality_optimal_summary.csv (FA column) capped at max_xdim; 'fixed'
# uses xdim_fixed for every group regardless of Step 1 results
xdim_source = 'from_step1'
xdim_fixed = 8
max_xdim = 12
dimensionality_summary_path = os.path.join('./output/dimred_plots', 'dimensionality_optimal_summary.csv')

min_neurons_per_group = 10
random_state = 0
n_jobs_sessions = -1

cache_dir = './output/dimred_cache'
force_recompute = False   # set True to ignore existing cache files and refit everything

# Bump whenever _compute()'s internal logic or output schema changes, even
# if no user-facing config parameter changed -- see run_tdr_encoding.py's
# docstring for why this exists (hit this exact stale-cache bug once
# already in this pipeline).
_GPFA_COMPUTE_VERSION = 2   # v2: added report_shared_variance diagnostic

output_dir = './output/dimred_plots'
trajectory_dir = os.path.join(output_dir, 'trajectories')
os.makedirs(trajectory_dir, exist_ok=True)


#%% ------------------------------------------------------------------
# Sanity check: are deconvolved values plausible as spike-count proxies?
# ----------------------------------------------------------------------
def report_deconv_sanity(X, session_id, group_key):
    """
    X: (n_neurons, n_trials, T) tensor for one group. Prints basic
    diagnostics relevant to treating X as spike counts for GPFA.
    """
    frac_negative = np.mean(X < 0)
    frac_zero = np.mean(X == 0)
    frac_noninteger = np.mean(np.abs(X - np.round(X)) > 1e-6)
    if frac_negative > 0.01:
        print(f'    WARNING [{session_id} {group_key}]: {frac_negative:.1%} of deconvolved '
              f'values are negative -- spike-count treatment for GPFA is questionable here.')
    print(f'    [{session_id} {group_key}] deconv sanity: {frac_zero:.1%} zero, '
          f'{frac_noninteger:.1%} non-integer-valued (informational only).')


def report_shared_variance(X, session_id, group_key, random_state=0):
    """
    Elsayed & Cunningham (2017)-style diagnostic: is there genuine SHARED
    single-trial covariation across neurons, beyond each neuron's own
    condition-mean PSTH plus independent noise? Simple alternative
    explanation being ruled out: single-trial GPFA "trajectories" are just
    per-neuron PSTH + independent noise, with no real coordinated
    population-level single-trial fluctuation to describe.

    Cheap relative to refitting GPFA on shuffled data many times (not
    done here) -- this is a covariance-based proxy: after subtracting each
    neuron's own condition-mean PSTH, compute what fraction of the total
    residual variance the LEADING eigenvector of the neuron x neuron
    covariance captures ("shared variance fraction"). Compare against a
    null where each neuron's trial assignment is shuffled independently
    (preserves each neuron's own residual time-series shape, destroys
    which trial each neuron's fluctuation co-occurred with -- i.e.
    destroys genuine shared single-trial covariation specifically).

    X : (n_neurons, n_trials, T). Prints and returns (real_frac, null_frac).
    """
    n_neurons, n_trials, T = X.shape
    psth = X.mean(axis=1, keepdims=True)  # (n_neurons, 1, T)
    resid = X - psth

    def shared_frac(R):
        flat = R.reshape(n_neurons, -1)
        cov = np.cov(flat)
        eigvals = np.linalg.eigvalsh(cov)
        return float(eigvals.max() / eigvals.sum()) if eigvals.sum() > 0 else 0.0

    real_frac = shared_frac(resid)
    rng = np.random.default_rng(random_state)
    resid_null = resid.copy()
    for n in range(n_neurons):
        resid_null[n] = resid[n, rng.permutation(n_trials), :]
    null_frac = shared_frac(resid_null)

    ratio = real_frac / null_frac if null_frac > 0 else np.inf
    flag = '' if ratio > 1.5 else '  NOTE: little evidence of genuine shared single-trial structure ' \
                                   'beyond independent per-neuron noise around the PSTH here.'
    print(f'    [{session_id} {group_key}] shared variance fraction: real={real_frac:.3f}, '
          f'null={null_frac:.3f} (ratio={ratio:.1f}x){flag}')
    return real_frac, null_frac


def clean_nan_tensor(X, session_id, group_key, max_nan_frac_per_trial=0.5):
    """
    Handle NaNs in a (n_neurons, n_trials, T) tensor before it reaches
    either GPFA path -- neither elephant's spike-time construction nor
    sklearn's FactorAnalysis accept NaN input. NaNs are expected/normal
    here: with position-binned spatial windows (s_pre/s_post/binsize),
    not every trial covers the full window (short trials don't reach the
    edges), so compute_tensor_for_session leaves those bins as NaN (see
    the "Mean of empty slice" warning from utils/psth.py -- that's this).

    Strategy, per neuron per trial:
      1. Linearly interpolate interior NaN runs from the surrounding
         valid samples.
      2. Fill any remaining NaNs at the very start/end of the trial
         (interpolation can't extrapolate) via nearest-value fill.
      3. If a trial is STILL more than `max_nan_frac_per_trial` NaN after
         that (e.g. the trial never entered the window at all), drop the
         whole trial rather than fabricate most of its trace.

    Returns
    -------
    X_clean : (n_neurons, n_trials_kept, T), no NaNs
    trial_keep_idx : boolean array, len n_trials, which original trials survived
    """
    n_neurons, n_trials, T = X.shape
    X = X.copy()

    frac_nan_per_trial = np.mean(np.isnan(X), axis=(0, 2))  # (n_trials,)
    trial_keep_idx = frac_nan_per_trial <= max_nan_frac_per_trial
    n_dropped = int(np.sum(~trial_keep_idx))
    if n_dropped > 0:
        print(f'    [{session_id} {group_key}] dropping {n_dropped}/{n_trials} trials '
              f'(>{max_nan_frac_per_trial:.0%} NaN, likely never entered the response window)')
    X = X[:, trial_keep_idx, :]

    t_idx = np.arange(T)
    for n in range(X.shape[0]):
        for k in range(X.shape[1]):
            trace = X[n, k, :]
            valid = ~np.isnan(trace)
            if valid.sum() == 0:
                trace[:] = 0.0  # entire trace NaN despite passing the trial-level filter (all-NaN neuron) -> zero-fill
            elif valid.sum() < T:
                trace[:] = np.interp(t_idx, t_idx[valid], trace[valid])  # interpolate interior + edge-extrapolate via np.interp's flat extrapolation
            X[n, k, :] = trace

    remaining_nan = np.sum(np.isnan(X))
    if remaining_nan > 0:
        print(f'    [{session_id} {group_key}] WARNING: {remaining_nan} NaN values remain after '
              f'cleaning -- investigate before trusting this group\'s trajectories.')

    return X, trial_keep_idx


#%% ------------------------------------------------------------------
# GPFA fitting: elephant-based (real GPFA) or smoothed-FA fallback
# ----------------------------------------------------------------------
def fit_gpfa_elephant(X, bin_size_ms, x_dim, random_state=0):
    """
    X: (n_neurons, n_trials, T) tensor of deconvolved "counts", one value
    per ORIGINAL tensor bin (position or time bin, whichever the tensor
    was built with). Builds synthetic neo.SpikeTrain objects (see module
    docstring for the modeling assumption) and fits elephant.gpfa.GPFA.

    IMPORTANT: dt is set equal to bin_size_ms itself, i.e. each ORIGINAL
    tensor bin maps to exactly one GPFA output bin (T_gpfa == T). This
    used to instead derive a fake sampling rate from the tensor's spatial
    `binsize` (cm) via a `1/(binsize/100)` guess with no real justification
    -- that manufactured several GPFA bins per original bin (e.g. ~5x with
    the default s_pre/s_post/binsize), asking GPFA to resolve more temporal
    structure than the data actually supports, which produced jagged,
    self-crossing single-trial trajectories that then averaged into a
    frayed, hard-to-read mean. Tying dt to bin_size_ms directly removes
    that source of manufactured noise -- there's no real "sampling rate"
    for spatially-binned data to guess at.

    Returns: trajectories, array (n_trials, x_dim, T_gpfa_bins) -- T_gpfa
    should now equal T (the original tensor's bin count) except for
    integer rounding at trial edges.
    """
    import warnings
    import neo
    import quantities as pq
    from elephant.gpfa import GPFA

    n_neurons, n_trials, T = X.shape
    dt = bin_size_ms / 1000.0  # seconds -- ties 1 tensor bin to exactly 1 GPFA bin, no re-binning
    t_edges = np.arange(T + 1) * dt

    spiketrains = []
    for k in range(n_trials):
        trial_trains = []
        for n in range(n_neurons):
            times = []
            for t in range(T):
                count = int(round(max(X[n, k, t], 0)))
                if count > 0:
                    bin_center = (t_edges[t] + t_edges[t + 1]) / 2
                    # jitter identical-count spikes slightly within the bin so
                    # they aren't literally simultaneous (GPFA/elephant is fine
                    # with this either way, but avoids degenerate duplicate
                    # timestamps in the spike train)
                    times.extend(bin_center + np.linspace(-dt / 4, dt / 4, count))
            st = neo.SpikeTrain(np.sort(times) * pq.s, t_start=0 * pq.s, t_stop=t_edges[-1] * pq.s)
            trial_trains.append(st)
        spiketrains.append(trial_trains)

    gpfa = GPFA(bin_size=bin_size_ms * pq.ms, x_dim=x_dim)

    # elephant's internal FA-initialization step tries to cut each trial
    # into fixed-length sub-segments (its own default segLength, typically
    # 20 bins) to bootstrap the initial parameter guess before EM. With T
    # now correctly reflecting the tensor's true (often <20) bin count per
    # trial -- rather than the ~5x-inflated bin count from the old fs
    # hack -- every trial is shorter than that default, so elephant emits
    # one UserWarning per trial and falls back to segLength=Inf (whole
    # trial as one segment). This does NOT affect the actual EM fit that
    # follows (see "Fitting GPFA model..." after this call) -- it only
    # means the initial guess EM starts from is less refined. Suppressed
    # here (replaced with one summary line) since it's expected given a
    # correctly-sized T, not a sign of a new problem.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        trajectories = gpfa.fit_transform(spiketrains)  # list of (x_dim, T_gpfa) arrays, one per trial
    n_seglength_warnings = sum(1 for w in caught if 'segLength' in str(w.message))
    if n_seglength_warnings > 0:
        print(f'    (elephant: {n_seglength_warnings} trial(s) shorter than its internal segLength '
              f'init heuristic -- falling back to whole-trial segments, EM fit unaffected)')
    other_warnings = [w for w in caught if 'segLength' not in str(w.message)]
    for w in other_warnings:
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)

    T_gpfa = trajectories[0].shape[1]
    out = np.full((n_trials, x_dim, T_gpfa), np.nan)
    for k, traj in enumerate(trajectories):
        out[k, :, :traj.shape[1]] = traj
    return out


def fit_gpfa_fallback(X, x_dim, smooth_sigma_bins=2, random_state=0):
    """
    NOT real GPFA -- see module docstring. Gaussian-smooths each trial's
    trace in time, then fits ONE FactorAnalysis basis on all (smoothed)
    timepoints x trials pooled together, and projects every trial through
    it. Gives a smooth single-trial trajectory with no extra dependencies.

    X: (n_neurons, n_trials, T). Returns: (n_trials, x_dim, T)
    """
    n_neurons, n_trials, T = X.shape
    X_smooth = gaussian_filter1d(X, sigma=smooth_sigma_bins, axis=2)

    X_pooled = X_smooth.transpose(1, 2, 0).reshape(n_trials * T, n_neurons)  # (K*T, N)
    fa = FactorAnalysis(n_components=x_dim, random_state=random_state)
    fa.fit(X_pooled)
    latents = fa.transform(X_pooled)  # (K*T, x_dim)
    return latents.reshape(n_trials, T, x_dim).transpose(0, 2, 1)  # (K, x_dim, T)


#%% ------------------------------------------------------------------
# Load sessions (shallow) + Step 1 dimensionality summary (if used)
# ----------------------------------------------------------------------
sessions, nSessions = filter_sessions(
    protocols=protocol, load_behaviordata=False, load_calciumdata=False,
    load_videodata=False, min_trials=100)
report_sessions(sessions)

if xdim_source == 'from_step1':
    if not os.path.exists(dimensionality_summary_path):
        raise FileNotFoundError(
            f"xdim_source='from_step1' but {dimensionality_summary_path} doesn't exist -- "
            f"run run_dimensionality_estimate.py first, or set xdim_source='fixed'.")
    df_dim = pd.read_csv(dimensionality_summary_path)
    df_dim_fa = df_dim[df_dim['method'] == 'fa']
    # median across sessions per group, as a single per-group x_dim
    xdim_by_group = (df_dim_fa.groupby(['area', 'label'])['n_components_optimal']
                      .median().round().astype(int).clip(upper=max_xdim).to_dict())
else:
    xdim_by_group = {}


def get_xdim(area_name, label_name):
    if xdim_source == 'fixed':
        return xdim_fixed
    return int(xdim_by_group.get((area_name, label_name), xdim_fixed))


#%% ------------------------------------------------------------------
# Per session: filter to requested groups, fit trajectories per group
# ----------------------------------------------------------------------
def process_session(ses, session_index, nSessions, calciumversion, qc_csv_path,
                     target_areas, target_labels, s_pre, s_post, binsize, t_pre, t_post,
                     use_elephant_gpfa, gpfa_bin_size_ms, min_neurons_per_group,
                     random_state, trajectory_dir, cache_dir, force_recompute,
                     xdim_source, xdim_fixed, max_xdim):
    print(f'\n=== Session {session_index + 1}/{nSessions}: {ses.session_id} ===')

    # everything that affects the RESULT goes into the cache key -- if you
    # change any of these and re-run, a new cache file is used automatically
    # (old ones are simply left on disk, unused); GPFA fitting is the
    # expensive step (~1min/session), so this is what actually saves the time
    key_params = dict(
        compute_version=_GPFA_COMPUTE_VERSION,
        session_id=ses.session_id, protocol=ses.protocol, calciumversion=calciumversion,
        target_areas=target_areas, target_labels=target_labels,
        s_pre=s_pre, s_post=s_post, binsize=binsize, t_pre=t_pre, t_post=t_post,
        use_elephant_gpfa=use_elephant_gpfa, gpfa_bin_size_ms=gpfa_bin_size_ms,
        min_neurons_per_group=min_neurons_per_group, random_state=random_state,
        xdim_source=xdim_source, xdim_fixed=xdim_fixed, max_xdim=max_xdim,
        radius=50, depth_thr=300,  # fixed anatomical filter params, included for completeness
        # QC file mtime, so the cache busts automatically if run_qc_summary.py
        # is re-run and the pass/fail flags change:
        qc_mtime=os.path.getmtime(qc_csv_path) if qc_csv_path and os.path.exists(qc_csv_path) else None,
    )
    path = cache_path(cache_dir, f'{ses.session_id}_gpfa_trajectories', **key_params)

    def _compute(ses=ses):
        tensor, axis, axis_label = compute_tensor_for_session(
            ses, calciumversion=calciumversion,
            t_pre=t_pre, t_post=t_post,
            s_pre=s_pre, s_post=s_post, binsize=binsize, keepraw=False)
        # tensor: (K trials, N neurons, T bins) per this pipeline's convention

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

        group_trajectories = {}
        fitted_group_keys = []   # ONLY actual (n_trials, x_dim, T) trajectory keys, excluding the
                                  # auxiliary *_trial_keep_idx boolean arrays also stored in the npz
        for a in np.unique(area):
            for l in DEFAULT_LABEL_ORDER:
                idx_group = np.where((area == a) & (label == l))[0]
                n_neurons_group = len(idx_group)
                if n_neurons_group < min_neurons_per_group:
                    continue

                group_key = f'{a}_{l}'
                X = tensor[:, idx_group, :].transpose(1, 0, 2)  # (n_neurons, n_trials, T)
                report_deconv_sanity(X, ses.session_id, group_key)
                X, trial_keep_idx = clean_nan_tensor(X, ses.session_id, group_key)
                if X.shape[1] < 2:
                    print(f'    [{ses.session_id} {group_key}] skipping: fewer than 2 trials survive NaN cleaning')
                    continue

                # Elsayed & Cunningham (2017)-style check: is there genuine
                # shared single-trial covariation to describe here at all,
                # or is a single-trial "trajectory" just PSTH + independent
                # per-neuron noise? See null_hypotheses_framework.md.
                shared_var_real, shared_var_null = report_shared_variance(
                    X, ses.session_id, group_key, random_state=random_state)

                x_dim = get_xdim(a, l)
                x_dim = min(x_dim, n_neurons_group - 1) if n_neurons_group > 1 else 1
                print(f'    fitting {group_key}: n_neurons={n_neurons_group}, n_trials={X.shape[1]}, x_dim={x_dim}, '
                      f'method={"elephant GPFA" if use_elephant_gpfa else "smoothed-FA fallback"}')

                try:
                    if use_elephant_gpfa:
                        traj = fit_gpfa_elephant(X, bin_size_ms=gpfa_bin_size_ms, x_dim=x_dim,
                                                  random_state=random_state)
                    else:
                        traj = fit_gpfa_fallback(X, x_dim=x_dim, random_state=random_state)
                    group_trajectories[group_key] = traj
                    group_trajectories[f'{group_key}_trial_keep_idx'] = trial_keep_idx
                    group_trajectories[f'{group_key}_shared_variance_real'] = shared_var_real
                    group_trajectories[f'{group_key}_shared_variance_null'] = shared_var_null
                    fitted_group_keys.append(group_key)
                except ImportError:
                    print(f'    elephant/neo/quantities not installed -- falling back to smoothed-FA '
                          f'for {group_key}. Run `pip install elephant neo quantities` for real GPFA.')
                    group_trajectories[group_key] = fit_gpfa_fallback(X, x_dim=x_dim, random_state=random_state)
                    group_trajectories[f'{group_key}_trial_keep_idx'] = trial_keep_idx
                    group_trajectories[f'{group_key}_shared_variance_real'] = shared_var_real
                    group_trajectories[f'{group_key}_shared_variance_null'] = shared_var_null
                    fitted_group_keys.append(group_key)

        return {'group_trajectories': group_trajectories, 'fitted_group_keys': fitted_group_keys,
                'axis': axis, 'axis_label': axis_label}

    cache_hit = os.path.exists(path) and not force_recompute
    result = load_or_compute(path, _compute, force_recompute=force_recompute)

    # always (re)write the npz from the cached/computed result, so it stays
    # in sync even on a cache hit (cheap -- this is not the expensive step)
    npz_path = os.path.join(trajectory_dir, f'trajectories_{ses.session_id}.npz')
    np.savez(npz_path, axis=result['axis'], axis_label=result['axis_label'], **result['group_trajectories'])
    print(f'  [{ses.session_id}] saved trajectories to {npz_path} '
          f'({"loaded from cache" if cache_hit else "freshly computed"})')

    for attr in ('calciumdata', 'videodata', 'behaviordata'):
        if hasattr(ses, attr):
            delattr(ses, attr)

    return {'session_id': ses.session_id, 'groups': result['fitted_group_keys'],
            'axis': result['axis'], 'axis_label': result['axis_label']}


session_results = Parallel(n_jobs=n_jobs_sessions, backend='loky', verbose=10)(
    delayed(process_session)(
        ses, ises, nSessions, calciumversion, qc_csv_path,
        target_areas, target_labels, s_pre, s_post, binsize, t_pre, t_post,
        use_elephant_gpfa, gpfa_bin_size_ms, min_neurons_per_group,
        random_state, trajectory_dir, cache_dir, force_recompute,
        xdim_source, xdim_fixed, max_xdim)
    for ises, ses in enumerate(sessions)
)

print('\nDone. Per-session trajectory files:')
for res in session_results:
    print(f"  {res['session_id']}: groups {res['groups']}")

#%% ------------------------------------------------------------------
# Quick-look figure: condition-averaged trajectory in the top 2 latent
# dims, one subplot per (session, group) that was fit -- a sanity-check
# plot, not the final encoding/computation analysis (that's Step 3+).
# One line per session; color is shared between each session's trajectory
# line and its trial-window-start marker (explicitly matched via
# line.get_color(), since plot()/scatter() aren't guaranteed to draw from
# the same point in the axes' color cycle), and labeled by session_id so
# the legend actually has something to show.
# ----------------------------------------------------------------------
all_group_keys = sorted({g for res in session_results for g in res['groups']})
if all_group_keys:
    n_groups = len(all_group_keys)
    fig, axes = plt.subplots(1, n_groups, figsize=(4 * n_groups, 4), squeeze=False)
    axes = axes[0]

    for j, group_key in enumerate(all_group_keys):
        ax = axes[j]
        any_plotted = False
        for res in session_results:
            if group_key not in res['groups']:
                continue
            npz_path = os.path.join(trajectory_dir, f"trajectories_{res['session_id']}.npz")
            data = np.load(npz_path, allow_pickle=True)
            traj = data[group_key]  # (n_trials, x_dim, T)
            if traj.shape[1] < 2:
                continue
            mean_traj = np.nanmean(traj, axis=0)  # (x_dim, T)

            # variance explained by the 2 plotted dims, relative to all x_dim
            # fitted dims (elephant's GPFA orthonormalizes/orders latents by
            # variance, so dims 0-1 are its top 2 -- but if x_dim is large and
            # variance is spread fairly evenly across dims, a 2D projection can
            # still miss most of the structure, which is worth knowing when
            # judging whether this plot is a fair summary)
            var_per_dim = np.nanvar(traj, axis=(0, 2))  # (x_dim,)
            pct_top2 = 100 * var_per_dim[:2].sum() / var_per_dim.sum() if var_per_dim.sum() > 0 else np.nan

            line, = ax.plot(mean_traj[0, :], mean_traj[1, :], alpha=0.7, linewidth=1,
                             label=f"{res['session_id']} ({pct_top2:.0f}% var)")
            ax.scatter(mean_traj[0, 0], mean_traj[1, 0], marker='o', s=20,
                       color=line.get_color())  # trial-window start, same color as its line
            any_plotted = True

        ax.set_title(group_key, fontsize=9)
        ax.set_xlabel('latent dim 1', fontsize=8)
        ax.set_ylabel('latent dim 2', fontsize=8)
        if any_plotted:
            ax.legend(fontsize=6, loc='best')

    fig.suptitle('Condition-averaged trajectories (top 2 latent dims), per session', y=1.03)
    fig.tight_layout()
    fig_path = os.path.join(output_dir, 'trajectories_quicklook.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'\nSaved quick-look trajectory figure to {fig_path}')
