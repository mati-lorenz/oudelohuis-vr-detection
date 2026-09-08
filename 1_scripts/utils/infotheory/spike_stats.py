# -*- coding: utf-8 -*-
"""
spike_stats.py
================
Exploratory descriptive statistics for calcium-imaging "spike"
(deconvolved activity) data -- ported from the lab's own spike_stats.py
almost unchanged (it already operated on plain arrays, not anything
loaddata-specific, apart from frame-rate lookup). The calcium-imaging
analogues of standard electrophysiology spike-train QC/exploratory
analyses (event rate, ISI/IEI statistics, Fano factor, autocorrelation,
signal/noise correlations, population coupling, split-half reliability,
movement-artifact checks), adapted to the practical realities of
calcium imaging:

  - deconvolved "spike" traces are inferred, continuous-valued
    ESTIMATES of spiking activity (not directly observed spikes), at
    the imaging frame rate (typically 5-30 Hz) -- much coarser than
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
labels) -- neither is used by 2b_activity_statistics.py (no trial
structure involved there by design; those two are for later 2_ steps
once a response tensor/respmat exists). The rest operate directly on
the continuous, session-long deconvolved trace.

`iei_stats` and `autocorrelogram` loop per-neuron in plain Python (each
neuron's event-detection / autocorrelation is a short, cheap operation,
but with thousands of simultaneously recorded neurons the loop itself
dominates), so both accept `n_jobs`/`backend` and dispatch via
joblib.Parallel, matching the rest of this package.
"""
from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed


def get_frame_rate(session, fallback: float | None = None) -> float:
    """Best-effort inference of the imaging frame rate (Hz) for a
    Session. Tries a 'fs'/'imaging_rate'/'framerate' sessiondata column
    first (this project's make_fake_data.py always sets 'fs'), then
    falls back to `fallback` if given, else raises."""
    if getattr(session, "sessiondata", None) is not None:
        for col in ("fs", "imaging_rate", "framerate"):
            if col in session.sessiondata.columns:
                return float(session.sessiondata[col].iloc[0])
    if fallback is not None:
        return fallback
    raise ValueError("Could not infer imaging frame rate for this session; "
                      "pass fallback=... explicitly.")


def event_rate(calciumdata, fs: float) -> np.ndarray:
    """Per-neuron mean "event rate" (events/sec): the deconvolved
    trace's time-average times the frame rate. MAGNITUDE-WEIGHTED --
    only a literal events/sec count if the trace is close to a 0/1-per-
    frame indicator. If your deconvolved trace has larger-magnitude
    continuous values instead, read this as "mean activity per second"
    (a.u./s). Compare against `active_frame_rate`: if
    event_rate / active_frame_rate is >> 1 (e.g. >5-10x), the trace's
    active-frame magnitude is well above 1 and this number should NOT
    be read as an events/sec count."""
    X = np.asarray(calciumdata, dtype=float)
    return np.nanmean(X, axis=0) * fs


def active_frame_rate(calciumdata, fs: float, threshold: float = 0.0) -> np.ndarray:
    """Per-neuron rate of ACTIVE frames (activity > threshold),
    events/sec -- `sparsity(calciumdata, threshold) * fs`. Unlike
    `event_rate`, bounded above by `fs` and uses the exact same
    >threshold frame-counting definition as `iei_stats`/`sparsity`, so
    it's the number directly comparable to those (median IEI ~=
    1/active_frame_rate under roughly memoryless statistics)."""
    return sparsity(calciumdata, threshold=threshold) * fs


def sparsity(calciumdata, threshold: float = 0.0) -> np.ndarray:
    """Per-neuron fraction of frames with activity above `threshold`."""
    X = np.asarray(calciumdata, dtype=float)
    return np.mean(X > threshold, axis=0)


def iei_stats(calciumdata, fs: float, threshold: float = 0.0, max_neurons: int | None = None,
              rng=None, n_jobs: int = 1, backend: str = "loky"):
    """Per-neuron inter-event-interval (IEI) statistics: median IEI (s)
    and coefficient of variation (CV = std/mean of IEIs) -- CV~1
    "Poisson-like" irregular, CV<1 more regular, CV>1 more bursty (at
    frame-rate resolution, see module docstring).

    Returns
    -------
    median_iei : 1D array, length N (NaN if <2 events)
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

    results = Parallel(n_jobs=n_jobs, backend=backend)(delayed(_one_neuron)(n) for n in neuron_idx)

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


def fano_factor(calciumdata, fs: float, window_sec: float = 1.0, threshold: float | None = None) -> np.ndarray:
    """Per-neuron Fano factor (variance/mean of activity summed in
    non-overlapping `window_sec` windows across the whole session).
    ~1 is Poisson-like; >1 over-dispersed/bursty; <1 under-dispersed.

    IMPORTANT: by default (`threshold=None`) this sums the RAW trace --
    MAGNITUDE-WEIGHTED, same caveat as `event_rate`: if active-frame
    values are well above 1, variance is dominated by the SQUARE of the
    typical event amplitude and this can inflate into the tens/hundreds
    even for ordinary event timing -- not a bug, but not comparable to
    the textbook "~1 for Poisson counts" intuition. Pass `threshold`
    (e.g. 0.0) to binarize first for a COUNT-based Fano factor that IS
    comparable to that intuition."""
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
    with np.errstate(divide="ignore", invalid="ignore"):
        ff = np.where(mean > 0, var / mean, np.nan)
    return ff


def autocorrelogram(calciumdata, fs: float, max_lag_sec: float = 2.0, max_neurons: int | None = None,
                     rng=None, n_jobs: int = 1, backend: str = "loky"):
    """Per-neuron (or per-subsample) normalized autocorrelation of the
    deconvolved trace, for lags -max_lag_sec..+max_lag_sec. The
    estimator here (Pearson correlation of the trace against itself
    shifted by `lag` samples) is symmetric for a real-valued signal --
    ac(-k) == ac(k) exactly -- so the negative half is the positive
    half mirrored, not recomputed; this only doubles the OUTPUT size,
    not the compute cost.

    Returns
    -------
    lags : 1D array (s), from -max_lag_sec to +max_lag_sec, lags[0] < 0
    ac : 2D array (n_neurons_used, n_lags); ac[:, len(lags)//2] == 1
        (the lag=0 column) by construction
    """
    X = np.asarray(calciumdata, dtype=float)
    T, N = X.shape
    neuron_idx = np.arange(N)
    if max_neurons is not None and N > max_neurons:
        rng = rng or np.random.default_rng()
        neuron_idx = rng.choice(N, size=max_neurons, replace=False)

    max_lag = int(round(max_lag_sec * fs))
    pos_lags = np.arange(max_lag + 1) / fs

    def _one_neuron(n):
        x = X[:, n] - np.mean(X[:, n])
        denom = np.sum(x ** 2)
        row = np.full(max_lag + 1, np.nan)
        if denom == 0:
            return row
        for lag in range(max_lag + 1):
            row[lag] = 1.0 if lag == 0 else np.sum(x[:-lag] * x[lag:]) / denom
        return row

    rows = Parallel(n_jobs=n_jobs, backend=backend)(delayed(_one_neuron)(n) for n in neuron_idx)
    pos_ac = np.vstack(rows)                          # (n, max_lag+1), lag 0..max_lag
    full_ac = np.concatenate([pos_ac[:, :0:-1], pos_ac], axis=1)   # mirror -> -max_lag..max_lag
    full_lags = np.concatenate([-pos_lags[:0:-1], pos_lags])
    return full_lags, full_ac


def population_coupling(respmat, n_trials: int) -> np.ndarray:
    """Per-neuron population coupling: Pearson correlation between each
    neuron's trial-by-trial response and the trial-by-trial POPULATION
    MEAN response EXCLUDING that neuron (Okun et al. 2015-style).
    `n_trials` must be passed explicitly (imaging sessions routinely
    have far more neurons than trials, so orientation can't be guessed
    from shape alone)."""
    X = np.asarray(respmat, dtype=float)
    if X.shape[0] == n_trials and X.shape[1] != n_trials:
        pass
    elif X.shape[1] == n_trials and X.shape[0] != n_trials:
        X = X.T
    else:
        raise ValueError(f"Could not unambiguously orient respmat of shape {X.shape} "
                          f"against n_trials={n_trials}.")
    K, N = X.shape
    total = X.sum(axis=1)
    pop_excl = (total[:, None] - X) / max(N - 1, 1)
    x_centered = X - X.mean(axis=0, keepdims=True)
    p_centered = pop_excl - pop_excl.mean(axis=0, keepdims=True)
    cov = (x_centered * p_centered).sum(axis=0)
    std_x = np.sqrt((x_centered ** 2).sum(axis=0))
    std_p = np.sqrt((p_centered ** 2).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((std_x > 0) & (std_p > 0), cov / (std_x * std_p), np.nan)


def split_half_reliability(tensor, n_splits: int = 10, random_state: int | None = None) -> np.ndarray:
    """Per-neuron split-half reliability: correlation between the
    trial-averaged response profile from two random trial halves,
    averaged over `n_splits` splits. tensor: (K trials, N neurons, T bins)."""
    K, N, T = tensor.shape
    rng = np.random.default_rng(random_state)
    r_accum = np.zeros(N)
    r_count = np.zeros(N)
    for _ in range(n_splits):
        perm = rng.permutation(K)
        half1, half2 = perm[:K // 2], perm[K // 2:2 * (K // 2)]
        mean1 = np.nanmean(tensor[half1, :, :], axis=0)
        mean2 = np.nanmean(tensor[half2, :, :], axis=0)
        for n in range(N):
            a, b = mean1[n, :], mean2[n, :]
            if np.std(a) > 0 and np.std(b) > 0:
                r_accum[n] += np.corrcoef(a, b)[0, 1]
                r_count[n] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(r_count > 0, r_accum / np.maximum(r_count, 1), np.nan)


def runspeed_correlation(calciumdata, runspeed_trace) -> np.ndarray:
    """Per-neuron Pearson correlation between the deconvolved trace and
    the (imaging-rate-aligned) running speed trace -- movement-artifact
    / locomotion-modulation QC check."""
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
