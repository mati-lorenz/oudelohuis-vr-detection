# -*- coding: utf-8 -*-
"""
corr_lib.py
============
Frame-level Pearson correlation between every cell's activity and one
behavioral trace (running speed, pupil, lick rate, position ...), on the
imaging clock, with a circular-shift null.

Why circular shifts (not trial shuffles)? Both neural activity and
behavior are strongly autocorrelated in time, so a plain label shuffle
gives a null that is far too narrow (every cell looks significant). Here
the behavior trace is rolled by a random lag (>= `min_shift` frames,
wrapping around the session): the null keeps the behavior's
autocorrelation and its relation to the trial structure, and destroys
only the moment-to-moment alignment with the cell. A cell is called
significant if its observed r lies outside the central (1 - alpha)
interval of ITS OWN null (2.5/97.5 percentiles), the same criterion as
roc.auc_permutation_test.

Multiple comparisons: ~5% of cells are expected to pass by chance; no
correction across cells/variables is applied.

Implementation: for every shift, r for all cells follows from a handful
of matrix-vector products with masked sums, so 500 shifts x 1000 cells x
20k frames takes seconds.
"""
from __future__ import annotations

import numpy as np


def _masked_r(A, A2, b, valid):
    """r between every column of A and b over frames where `valid`."""
    w = valid.astype(float)
    n = w.sum()
    if n < 10:
        return np.full(A.shape[1], np.nan), int(n)
    bw = np.where(valid, b, 0.0)
    Sa, Saa = w @ A, w @ A2
    Sb, Sbb, Sab = bw.sum(), (bw * bw).sum(), bw @ A
    cov = Sab / n - (Sa / n) * (Sb / n)
    va = Saa / n - (Sa / n) ** 2
    vb = Sbb / n - (Sb / n) ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / np.sqrt(va * vb)
    r = np.where((va > 1e-12) & (vb > 1e-12), r, np.nan)
    return r, int(n)


def corr_with_shift_null(A, b, frame_ok=None, n_shifts: int = 500, min_shift: int = 100,
                         rng: np.random.Generator | None = None, alpha: float = 0.05,
                         lags=(0,)) -> dict:
    """Correlate each column of A (T frames x N cells) with b (T,).

    frame_ok : optional (T,) bool -- frames allowed in the correlation
        (e.g. engaged frames). b is set to NaN outside it BEFORE shifting.
    lags : integer frame lags to consider (default only 0). A POSITIVE lag
        k correlates the cell at frame t with the behavior at t-k, i.e.
        "activity FOLLOWS behavior by k frames" (as in Chen et al. 2013,
        cross-correlation lags). With several lags the statistic is the
        r of the lag with the largest |r| (signed), and the circular-shift
        null uses the SAME max-over-lags statistic, so choosing the best
        lag does not inflate significance.
    Returns dict: r (statistic), lag (frames, chosen lag per cell),
    r_by_lag (len(lags), N), null_lo, null_hi, p_value (two-sided, +1
    corrected), significant, n_frames.
    """
    rng = rng or np.random.default_rng()
    lags = np.atleast_1d(np.asarray(lags, int))
    A = np.asarray(A, float)
    b = np.asarray(b, float).copy()
    T, N = A.shape
    ok = np.ones(T, bool) if frame_ok is None else np.asarray(frame_ok, bool)
    b[~ok] = np.nan
    A2 = A * A

    def stat(bb):
        """(N,) signed r at the lag with max |r|, and which lag; plus (L,N)."""
        rl = np.empty((len(lags), N))
        n = 0
        for i, k in enumerate(lags):
            bk = np.roll(bb, int(k)) if k else bb
            rl[i], n = _masked_r(A, A2, bk, ok & np.isfinite(bk))
        if len(lags) == 1:
            return rl[0], np.zeros(N, int), rl, n
        idx = np.nanargmax(np.where(np.isnan(rl), -1.0, np.abs(rl)), axis=0)
        return rl[idx, np.arange(N)], idx, rl, n

    r, idx, r_by_lag, n = stat(b)
    lag_chosen = lags[idx]
    if n < 10 or T <= 2 * min_shift + 1 + int(np.abs(lags).max()):
        nan = np.full(N, np.nan)
        return dict(r=r, lag=lag_chosen, r_by_lag=r_by_lag, null_lo=nan, null_hi=nan, p_value=nan,
                    significant=np.zeros(N, bool), n_frames=n)
    shifts = rng.integers(min_shift, T - min_shift - int(np.abs(lags).max()), size=n_shifts)
    null = np.empty((n_shifts, N))
    for i, sh in enumerate(shifts):
        null[i] = stat(np.roll(b, int(sh)))[0]
    lo = np.nanpercentile(null, 100 * alpha / 2, axis=0)
    hi = np.nanpercentile(null, 100 * (1 - alpha / 2), axis=0)
    p = (1.0 + (np.abs(np.nan_to_num(null)) >= np.abs(np.nan_to_num(r))[None, :]).sum(0)) / (n_shifts + 1.0)
    sig = ((r < lo) | (r > hi)) & np.isfinite(r)
    return dict(r=r, lag=lag_chosen, r_by_lag=r_by_lag, null_lo=lo, null_hi=hi,
                p_value=np.where(np.isfinite(r), p, np.nan), significant=sig, n_frames=n)
