# -*- coding: utf-8 -*-
"""
spike_stats.py
================
Exploratory descriptive statistics for calcium-imaging "spike"
(deconvolved activity) data -- the calcium-imaging analogues of
standard electrophysiology spike-train QC/exploratory analyses (event
rate, ISI/IEI statistics, Fano factor, autocorrelation, signal/noise
correlations, population coupling, split-half reliability, movement-
artifact checks), adapted to the practical realities of calcium
imaging:

  - deconvolved "spike" traces are inferred, continuous-valued
    ESTIMATES of spiking activity (not directly observed spikes), at
    the imaging frame rate (typically 15-30 Hz) -- much coarser than
    electrophysiology sampling, so anything relying on precise spike
    TIMING (e.g. millisecond-resolution ISI distributions, spike-phase
    coding) is not meaningful here. What IS meaningful is
    coarser-timescale statistics: event rate, event sparsity,
    inter-event-interval statistics at frame-rate resolution, Fano
    factor over reasonably long windows, and the shape of the
    deconvolved trace's autocorrelation.
  - GCaMP-style indicators have slow decay kinetics (typically
    hundreds of ms), so even a well-deconvolved trace can show residual
    autocorrelation reflecting incomplete deconvolution or genuine
    burst structure -- the autocorrelogram here is a diagnostic (e.g.
    for sanity-checking deconvolution quality), not a claim about
    millisecond-precision spike timing.

None of these functions require trial structure except
`split_half_reliability` (needs a trial tensor) and
`signal_noise_correlations` (needs a trial response matrix + stimulus
labels); the rest operate directly on the continuous, session-long
deconvolved trace.

`iei_stats` and `autocorrelogram` loop per-neuron in plain Python (each
neuron's event-detection / autocorrelation is a short, cheap operation,
but with thousands of simultaneously recorded neurons the loop itself
dominates), so both accept `n_jobs`/`backend` and dispatch via
joblib.Parallel, matching the rest of this package.
"""

from joblib import Parallel, delayed

import numpy as np


def get_frame_rate(session, fallback=None):
    """
    Best-effort inference of the imaging frame rate (Hz) for a loaddata
    Session. Tries, in order: a 'fs'/'imaging_rate'/'framerate'
    sessiondata column, then the median frame interval of
    `session.ts_F`. Pass `fallback` to avoid raising if neither is
    available (e.g. for a quick synthetic-data test).
    """
    if hasattr(session, 'sessiondata'):
        for col in ('fs', 'imaging_rate', 'framerate'):
            if col in session.sessiondata.columns:
                return float(session.sessiondata[col].iloc[0])
    if hasattr(session, 'ts_F'):
        dt = np.median(np.diff(np.asarray(session.ts_F)))
        if dt > 0:
            return float(1.0 / dt)
    if fallback is not None:
        return fallback
    raise ValueError('Could not infer imaging frame rate for this session; '
                      'pass fallback=... explicitly.')


def event_rate(calciumdata, fs):
    """
    Per-neuron mean "event rate" (events/sec): the deconvolved trace's
    time-average times the frame rate. This is a MAGNITUDE-WEIGHTED
    quantity -- only equal to a literal "events per second" count if
    the deconvolved trace is close to a 0/1-per-frame event indicator.

    If your deconvolved trace instead has continuous, larger-magnitude
    values (common depending on the deconvolution algorithm/scaling),
    this number is better read as "mean activity per second" (a.u./s),
    NOT a literal spike/event count -- and it can then legitimately
    come out much larger than the frame rate itself, and much larger
    than `active_frame_rate` below. Compare the two: if
    `event_rate / active_frame_rate` is >> 1 (e.g. >5-10x), that's a
    sign the trace's active-frame magnitude is well above 1 and this
    number should NOT be read as an events/sec count -- use
    `active_frame_rate` instead for anything you want to interpret as
    a literal rate (e.g. for cross-checking against `iei_stats`, which
    is defined the same way: threshold at >0, count frames).

    Parameters
    ----------
    calciumdata : array or DataFrame (T samples, N neurons)
    fs : float, Hz

    Returns
    -------
    rate : 1D array, length N
    """
    X = np.asarray(calciumdata, dtype=float)
    return np.nanmean(X, axis=0) * fs


def active_frame_rate(calciumdata, fs, threshold=0.0):
    """
    Per-neuron rate of ACTIVE frames (frames with activity > threshold),
    in events/sec -- i.e. `sparsity(calciumdata, threshold) * fs`.

    Unlike `event_rate` (mean(trace)*fs, magnitude-weighted), this is
    bounded above by `fs` (at most one "active-frame event" per frame)
    and uses the EXACT SAME >threshold frame-counting definition as
    `iei_stats` and `sparsity` -- so it is the number directly
    comparable/consistent with those: under a roughly memoryless
    (Poisson-like) assumption, `1 / active_frame_rate` should be in the
    same ballpark as the median IEI from `iei_stats`. If you see
    `event_rate` reporting hundreds of "events/sec" while `iei_stats`
    reports a median IEI of hundreds of milliseconds to seconds, THAT
    is the mismatch this function resolves -- use `active_frame_rate`
    as "the" event rate whenever you want a number consistent with the
    rest of this module's event-based statistics.
    """
    return sparsity(calciumdata, threshold=threshold) * fs


def sparsity(calciumdata, threshold=0.0):
    """Per-neuron fraction of frames with activity above `threshold`."""
    X = np.asarray(calciumdata, dtype=float)
    return np.mean(X > threshold, axis=0)


def iei_stats(calciumdata, fs, threshold=0.0, max_neurons=None, rng=None,
              n_jobs=1, backend='loky'):
    """
    Per-neuron inter-event-interval (IEI) statistics: median IEI (s)
    and coefficient of variation (CV = std/mean of IEIs) -- CV ~ 1 is
    "Poisson-like" irregular, CV < 1 more regular, CV > 1 more bursty
    (here at frame-rate resolution, see module docstring).

    Parameters
    ----------
    n_jobs, backend : parallelization across neurons (joblib.Parallel);
        n_jobs=1 (default) runs serially -- set n_jobs=-1 for large N.

    Returns
    -------
    median_iei : 1D array, length N (seconds; NaN if <2 events)
    cv_iei : 1D array, length N (NaN if <2 events)
    iei_pooled : 1D array, ALL inter-event-intervals pooled across the
        requested neurons (for a population-level histogram)
    """
    X = np.asarray(calciumdata, dtype=float)
    T, N = X.shape
    neuron_idx = np.arange(N)
    if max_neurons is not None and N > max_neurons:
        rng = rng or np.random.default_rng()
        neuron_idx = rng.choice(N, size=max_neurons, replace=False)

    def _one_neuron(n):
        event_frames = np.where(X[:, n] > threshold)[0]
        if len(event_frames) < 2:
            return np.nan, np.nan, None
        ieis = np.diff(event_frames) / fs
        med = np.median(ieis)
        m = np.mean(ieis)
        cv = np.std(ieis) / m if m > 0 else np.nan
        return med, cv, ieis

    results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_one_neuron)(n) for n in neuron_idx)

    median_iei = np.full(N, np.nan)
    cv_iei = np.full(N, np.nan)
    pooled = []
    for n, (med, cv, ieis) in zip(neuron_idx, results):
        median_iei[n] = med
        cv_iei[n] = cv
        if ieis is not None:
            pooled.append(ieis)

    iei_pooled = np.concatenate(pooled) if pooled else np.array([])
    return median_iei, cv_iei, iei_pooled


def fano_factor(calciumdata, fs, window_sec=1.0, threshold=None):
    """
    Per-neuron Fano factor (variance / mean of activity summed in
    non-overlapping `window_sec`-long windows across the whole
    session). Fano factor ~1 is consistent with Poisson-like count
    variability; >1 over-dispersed/bursty, <1 under-dispersed/regular.

    IMPORTANT (magnitude vs. count): by default (`threshold=None`),
    this sums the RAW deconvolved trace within each window -- a
    MAGNITUDE-WEIGHTED quantity, exactly analogous to `event_rate` (see
    its docstring). If active-frame values are well above 1 (check via
    `event_rate`/`active_frame_rate`'s ratio), the variance is dominated
    by the SQUARE of the typical event amplitude, which can inflate the
    Fano factor into the tens, hundreds, or more even for
    otherwise-ordinary event TIMING statistics -- this is a real
    property of magnitude-weighted sums of heavy-tailed/zero-inflated
    data, not a sign anything is broken, but it also means a Fano
    factor computed this way is NOT comparable to the textbook "~1 for
    Poisson" intuition, which is about EVENT COUNTS, not summed
    magnitudes.

    Pass `threshold` (e.g. `threshold=0.0`) to instead binarize the
    trace (active/inactive per frame) BEFORE windowing -- a COUNT-BASED
    Fano factor of active frames per window, consistent with
    `active_frame_rate`/`sparsity`/`iei_stats`, and the one to use if
    you want a number that's actually comparable to the classic
    Poisson-count intuition.

    Parameters
    ----------
    threshold : float or None
        if given, `calciumdata` is thresholded to a 0/1 active-frame
        indicator (activity > threshold) BEFORE summing into windows.
    """
    X = np.asarray(calciumdata, dtype=float)
    if threshold is not None:
        X = (X > threshold).astype(float)
    T, N = X.shape
    win = max(1, int(round(window_sec * fs)))
    n_windows = T // win
    if n_windows < 2:
        return np.full(N, np.nan)
    X_trim = X[:n_windows * win, :].reshape(n_windows, win, N).sum(axis=1)
    mean = X_trim.mean(axis=0)
    var = X_trim.var(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ff = np.where(mean > 0, var / mean, np.nan)
    return ff


def autocorrelogram(calciumdata, fs, max_lag_sec=2.0, max_neurons=None, rng=None,
                     n_jobs=1, backend='loky'):
    """
    Per-neuron (or per-neuron-subsample) normalized autocorrelation of
    the deconvolved trace, for lags 0..max_lag_sec.

    Parameters
    ----------
    n_jobs, backend : parallelization across neurons (joblib.Parallel);
        n_jobs=1 (default) runs serially -- set n_jobs=-1 for large N.

    Returns
    -------
    lags : 1D array (s)
    ac : 2D array (n_neurons_used, n_lags); ac[:, 0] == 1 by construction
    """
    X = np.asarray(calciumdata, dtype=float)
    T, N = X.shape
    neuron_idx = np.arange(N)
    if max_neurons is not None and N > max_neurons:
        rng = rng or np.random.default_rng()
        neuron_idx = rng.choice(N, size=max_neurons, replace=False)

    max_lag = int(round(max_lag_sec * fs))
    lags = np.arange(max_lag + 1) / fs

    def _one_neuron(n):
        x = X[:, n] - np.mean(X[:, n])
        denom = np.sum(x ** 2)
        row = np.full(max_lag + 1, np.nan)
        if denom == 0:
            return row
        for lag in range(max_lag + 1):
            row[lag] = 1.0 if lag == 0 else np.sum(x[:-lag] * x[lag:]) / denom
        return row

    rows = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(_one_neuron)(n) for n in neuron_idx)
    ac = np.vstack(rows)
    return lags, ac


def signal_noise_correlations(respmat, stim, pairs):
    """
    Pairwise SIGNAL correlation (correlation of trial-averaged,
    per-stimulus mean responses across stimulus conditions) and NOISE
    correlation (correlation of trial-by-trial residuals after
    subtracting each trial's stimulus-conditional mean) for a set of
    neuron pairs -- the same signal/noise correlation concept underlying
    the Pola et al. (2003) breakdown (infotheory/breakdown.py), computed
    here directly as a simple, model-free exploratory summary.

    Parameters
    ----------
    respmat : array (N neurons, K trials) or (K, N) -- auto-oriented
        against len(stim)
    stim : 1D array, length K
    pairs : list of (i, j) index tuples

    Returns
    -------
    signal_corr : 1D array, length len(pairs)
    noise_corr : 1D array, length len(pairs)
    """
    X = np.asarray(respmat, dtype=float)
    stim = np.asarray(stim)
    K = len(stim)
    if X.shape[0] == K and X.shape[1] != K:
        pass
    elif X.shape[1] == K and X.shape[0] != K:
        X = X.T
    else:
        raise ValueError('Could not unambiguously orient respmat; pass (K, N).')

    stim_values = np.unique(stim)
    means = np.zeros((len(stim_values), X.shape[1]))
    residual = np.zeros_like(X)
    for si, sv in enumerate(stim_values):
        idx = stim == sv
        means[si, :] = X[idx, :].mean(axis=0)
        residual[idx, :] = X[idx, :] - means[si, :]

    signal_corr = np.full(len(pairs), np.nan)
    noise_corr = np.full(len(pairs), np.nan)
    for k, (i, j) in enumerate(pairs):
        if means[:, i].std() > 0 and means[:, j].std() > 0:
            signal_corr[k] = np.corrcoef(means[:, i], means[:, j])[0, 1]
        if residual[:, i].std() > 0 and residual[:, j].std() > 0:
            noise_corr[k] = np.corrcoef(residual[:, i], residual[:, j])[0, 1]

    return signal_corr, noise_corr


def population_coupling(respmat, n_trials):
    """
    Per-neuron population coupling: Pearson correlation between each
    neuron's trial-by-trial response and the trial-by-trial POPULATION
    MEAN response EXCLUDING that neuron (Okun et al. 2015-style
    metric), computed at the trial-response-matrix scale (one value per
    trial per neuron). High coupling = a neuron's activity tracks
    population-wide fluctuations; can reflect shared behavioral/arousal
    state, imaging artifacts (e.g. z-motion), or genuine widespread
    network coupling.

    Parameters
    ----------
    respmat : array (N neurons, K trials) or (K, N)
    n_trials : int
        the KNOWN number of trials (e.g. `len(stim)`), used to
        determine orientation unambiguously. Earlier versions of this
        function guessed the orientation by assuming trials was the
        larger axis -- WRONG for calcium imaging, where sessions
        routinely have far more simultaneously recorded neurons
        (thousands) than trials (hundreds), the opposite of what that
        heuristic assumed. Getting this wrong silently returns a
        coupling vector of the wrong length (indexed by trial instead
        of by neuron), which fails loudly downstream but can also fail
        silently if the two happen to be nearly equal -- always pass
        `n_trials` explicitly.

    Returns
    -------
    coupling : 1D array, length N
    """
    X = np.asarray(respmat, dtype=float)
    if X.shape[0] == n_trials and X.shape[1] != n_trials:
        pass  # already (K, N)
    elif X.shape[1] == n_trials and X.shape[0] != n_trials:
        X = X.T   # was (N, K)
    else:
        raise ValueError(
            f'Could not unambiguously orient respmat of shape {X.shape} '
            f'against n_trials={n_trials}.')
    K, N = X.shape

    # vectorized across all N neurons at once (no Python loop): for each
    # neuron n, pop_excl[:, n] = (sum of all OTHER neurons' response on
    # that trial) / (N-1), then correlate column-wise against X
    total = X.sum(axis=1)                                    # (K,)
    pop_excl = (total[:, None] - X) / max(N - 1, 1)           # (K, N)

    x_centered = X - X.mean(axis=0, keepdims=True)
    p_centered = pop_excl - pop_excl.mean(axis=0, keepdims=True)
    cov = (x_centered * p_centered).sum(axis=0)
    std_x = np.sqrt((x_centered ** 2).sum(axis=0))
    std_p = np.sqrt((p_centered ** 2).sum(axis=0))

    with np.errstate(divide='ignore', invalid='ignore'):
        coupling = np.where((std_x > 0) & (std_p > 0), cov / (std_x * std_p), np.nan)
    return coupling


def split_half_reliability(tensor, n_splits=10, random_state=None):
    """
    Per-neuron split-half reliability: correlation between the
    trial-averaged temporal (time- or position-locked) response profile
    computed from two random halves of the trials, averaged over
    `n_splits` random splits for stability. A standard measure of how
    reliably time/position-locked a neuron's response is, useful for
    filtering neurons before further tuning/information analyses.

    Parameters
    ----------
    tensor : array (K trials, N neurons, T bins)
    n_splits : int
    random_state : optional int

    Returns
    -------
    reliability : 1D array, length N (mean Pearson r across splits)
    """
    K, N, T = tensor.shape
    rng = np.random.default_rng(random_state)
    r_accum = np.zeros(N)
    r_count = np.zeros(N)

    for _ in range(n_splits):
        perm = rng.permutation(K)
        half1, half2 = perm[:K // 2], perm[K // 2:2 * (K // 2)]
        mean1 = np.nanmean(tensor[half1, :, :], axis=0)   # (N, T)
        mean2 = np.nanmean(tensor[half2, :, :], axis=0)   # (N, T)
        for n in range(N):
            a, b = mean1[n, :], mean2[n, :]
            if np.std(a) > 0 and np.std(b) > 0:
                r_accum[n] += np.corrcoef(a, b)[0, 1]
                r_count[n] += 1

    with np.errstate(invalid='ignore', divide='ignore'):
        reliability = np.where(r_count > 0, r_accum / np.maximum(r_count, 1), np.nan)
    return reliability


def runspeed_correlation(calciumdata, runspeed_trace):
    """
    Per-neuron Pearson correlation between the continuous deconvolved
    trace and the (imaging-rate-interpolated) running speed trace -- a
    standard movement-artifact / locomotion-modulation QC check for
    calcium imaging in behaving animals.
    """
    X = np.asarray(calciumdata, dtype=float)
    v = np.asarray(runspeed_trace, dtype=float)
    N = X.shape[1]
    corr = np.full(N, np.nan)
    if np.std(v) == 0:
        return corr
    for n in range(N):
        if np.std(X[:, n]) > 0:
            corr[n] = np.corrcoef(X[:, n], v)[0, 1]
    return corr
