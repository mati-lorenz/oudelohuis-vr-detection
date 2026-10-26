# -*- coding: utf-8 -*-
"""
tensors.py
===========
Vectorized helpers shared by 2d_linear_encod / 2g_behavior_correlation /
2b_activity_statistics (raster part): put behavior on the imaging clock, locate every
imaging frame relative to its trial's stimulus onset (a POSITION, see
psth.py's docstring), and average activity per trial inside position
windows or position bins.

Everything here works on the IMAGING clock (`ts_F`, ~8 Hz) with one
sparse-matrix product per call, so it is fast for thousands of cells and
does not depend on psth.build_position_binned_trial_tensor.

Frame inclusion (same convention as 2c/neural_encoding): a frame counts
only if its timestamp lies inside an ENGAGED trial's [tStart, tEnd]
(continuous.restrict_to_engaged), and it is then assigned to the engaged
trial whose stimStart is nearest in position, provided it lies inside the
requested window around that stimStart.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse


# --------------------------------------------------------------------- #
# Behavior -> imaging frames
# --------------------------------------------------------------------- #
def behavior_to_imaging_frames(beh_ts, values, ts_F, mode: str = "mean") -> np.ndarray:
    """Aggregate a behavior trace (fine clock `beh_ts`) into one value per
    imaging frame.

    Each imaging frame owns the interval between the midpoints to its
    neighbouring frames. mode="mean": NaN-aware mean of the samples in the
    interval (NaN if none valid). mode="rate": sum of the samples divided
    by the interval length in seconds (events/s, for lick events).
    """
    beh_ts = np.asarray(beh_ts, dtype=float)
    values = np.asarray(values, dtype=float)
    ts_F = np.asarray(ts_F, dtype=float)
    mid = (ts_F[1:] + ts_F[:-1]) / 2.0
    edges = np.concatenate([[ts_F[0] - (mid[0] - ts_F[0])], mid, [ts_F[-1] + (ts_F[-1] - mid[-1])]])
    pos = np.searchsorted(beh_ts, edges)
    ok = np.isfinite(values)
    csum = np.concatenate([[0.0], np.cumsum(np.where(ok, values, 0.0))])
    ccnt = np.concatenate([[0], np.cumsum(ok)])
    sums = csum[pos[1:]] - csum[pos[:-1]]
    cnts = ccnt[pos[1:]] - ccnt[pos[:-1]]
    if mode == "rate":
        dur = np.diff(edges)
        return np.where(cnts > 0, sums / dur, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnts > 0, sums / cnts, np.nan)


def interp_to_imaging_frames(beh_ts, values, ts_F) -> np.ndarray:
    """Linear interpolation of a monotone trace (zpos) onto ts_F."""
    return np.interp(np.asarray(ts_F, float), np.asarray(beh_ts, float), np.asarray(values, float))


# --------------------------------------------------------------------- #
# Frame -> (trial, relative position)
# --------------------------------------------------------------------- #
def engaged_frame_mask(ts_F, trialdata: pd.DataFrame) -> np.ndarray:
    """True for imaging frames inside an engaged trial's [tStart, tEnd]."""
    ts_F = np.asarray(ts_F, float)
    tr = trialdata
    if "engaged" in tr.columns:
        tr = tr[tr["engaged"] == 1]
    if tr.empty:
        return np.zeros(len(ts_F), bool)
    t0 = tr["tStart"].to_numpy(float)
    t1 = tr["tEnd"].to_numpy(float)
    order = np.argsort(t0)
    t0, t1 = t0[order], t1[order]
    i = np.searchsorted(t0, ts_F, side="right") - 1
    ok = i >= 0
    return ok & (ts_F <= t1[np.clip(i, 0, None)])


def trial_frame_mask(ts_F, trialdata: pd.DataFrame, trial_numbers) -> np.ndarray:
    """True for imaging frames inside the [tStart, tEnd] of the trials whose trialNumber is in
    `trial_numbers` (engaged flag ignored): the generalisation of engaged_frame_mask used by the
    disengaged-trial tests."""
    ts_F = np.asarray(ts_F, float)
    tr = trialdata[trialdata["trialNumber"].isin(list(trial_numbers))]
    if tr.empty:
        return np.zeros(len(ts_F), bool)
    t0 = tr["tStart"].to_numpy(float)
    t1 = tr["tEnd"].to_numpy(float)
    order = np.argsort(t0)
    t0, t1 = t0[order], t1[order]
    i = np.searchsorted(t0, ts_F, side="right") - 1
    ok = i >= 0
    return ok & (ts_F <= t1[np.clip(i, 0, None)])


def position_relative_to_stim(zpos_F, stim_starts, s_pre: float, s_post: float):
    """For every frame: index of the nearest stimStart (position) and the
    position relative to it. Frames outside [-s_pre, +s_post) get k=-1 and
    rel=NaN. `stim_starts` must be sorted ascending (monotone zpos)."""
    zpos_F = np.asarray(zpos_F, float)
    ss = np.asarray(stim_starts, float)
    j = np.searchsorted(ss, zpos_F)
    cand = np.stack([np.clip(j - 1, 0, len(ss) - 1), np.clip(j, 0, len(ss) - 1)])
    d = zpos_F[None, :] - ss[cand]
    pick = np.abs(d).argmin(0)
    k = cand[pick, np.arange(len(zpos_F))]
    rel = zpos_F - ss[k]
    out = (rel < -s_pre) | (rel >= s_post)
    k = np.where(out, -1, k)
    rel = np.where(out, np.nan, rel)
    return k, rel


# --------------------------------------------------------------------- #
# Trial x cell x bin averaging
# --------------------------------------------------------------------- #
def _bin_matrix(k, b, n_trials, n_bins, frame_ok):
    rows = (k * n_bins + b)[frame_ok]
    cols = np.where(frame_ok)[0]
    return sparse.csr_matrix((np.ones(len(rows)), (rows, cols)),
                             shape=(n_trials * n_bins, len(frame_ok)))


def build_position_tensor(activity, k, rel, frame_ok, n_trials: int, s_pre: float, s_post: float,
                          binsize: float):
    """Mean activity per (trial, cell, position bin).

    activity : (T, N) on the imaging clock
    k, rel   : outputs of position_relative_to_stim (trial index into the
               stim_starts array used there; -1 = none)
    frame_ok : (T,) bool -- extra frame filter (engaged etc.)
    Returns (tensor (n_trials, N, B) with NaN where a trial has no frame
    in a bin, counts (n_trials, B), bin_centers (B,)).
    """
    edges = np.arange(-s_pre, s_post + binsize / 2, binsize)
    n_bins = len(edges) - 1
    b = np.clip(np.floor((np.nan_to_num(rel, nan=0.0) + s_pre) / binsize).astype(int), 0, n_bins - 1)
    ok = frame_ok & (k >= 0) & np.isfinite(rel)
    M = _bin_matrix(np.clip(k, 0, None), b, n_trials, n_bins, ok)
    sums = M @ np.asarray(activity, float)
    counts = np.asarray(M.sum(1)).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sums / counts[:, None]
    tensor = mean.reshape(n_trials, n_bins, -1).transpose(0, 2, 1)
    return tensor, counts.reshape(n_trials, n_bins), (edges[:-1] + edges[1:]) / 2


def trial_window_means(activity, k, rel, frame_ok, n_trials: int, windows: dict,
                       equalize_quantile: float | None = None, return_info: bool = False):
    """Mean activity per (trial, cell) in each position window
    (name -> (start_cm, end_cm) relative to stimStart). Returns dict
    name -> (n_trials, N) with NaN for trials with no frame in the window.
    Windows may overlap.

    equalize_quantile=None: plain mean over all frames in the window
    (what 2c_information does). This is a TIME-weighted average along
    position: a trial that slows down inside the window (lick trials do)
    spends more frames at the slow part, so (a) it averages more frames
    -- for skewed, spike-like activity the mean of n frames has a
    different distribution shape for different n, which rank statistics
    (AUC) and MI read as "information" -- and (b) it over-weights the
    positions where it slowed down, so any position-tuned cell looks
    choice-/speed-dependent.

    equalize_quantile=q (e.g. 0.1): POSITION-MATCHED sampling. n_target =
    the q-quantile of the per-trial frame counts; every trial is sampled at
    the same n_target evenly spaced positions in the window (midpoints of
    n_target equal-width sub-windows), taking the nearest imaging frame to
    each. Every trial then averages exactly n_target single frames at the
    same positions. Trials with a sub-window more than one sub-window
    width away from any frame (too few frames) become NaN.
    """
    activity = np.asarray(activity, float)
    out, info = {}, {}
    for name, (lo, hi) in windows.items():
        ok = frame_ok & (k >= 0) & np.isfinite(rel) & (rel >= lo) & (rel < hi)
        rows = np.clip(k, 0, None)[ok]
        cols = np.where(ok)[0]
        cnt = np.bincount(rows, minlength=n_trials)
        if equalize_quantile is None:
            M = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_trials, len(ok)))
            sums = M @ activity
            with np.errstate(invalid="ignore", divide="ignore"):
                out[name] = np.where(cnt[:, None] > 0, sums / cnt[:, None], np.nan)
            info[name] = dict(n_target=None, frac_trials_dropped=float(np.mean(cnt == 0)))
            continue
        have = cnt[cnt > 0]
        n_target = max(1, int(np.floor(np.quantile(have, equalize_quantile)))) if len(have) else 1
        step = (hi - lo) / n_target
        targets = lo + (np.arange(n_target) + 0.5) * step
        order = np.argsort(rows, kind="stable")
        cols_s = cols[order]
        starts = np.concatenate([[0], np.cumsum(cnt)])
        res = np.full((n_trials, activity.shape[1]), np.nan)
        for t in np.where(cnt > 0)[0]:
            f = cols_s[starts[t]:starts[t + 1]]
            r = rel[f]
            srt = np.argsort(r)
            f, r = f[srt], r[srt]
            j = np.clip(np.searchsorted(r, targets), 1, len(r)) if len(r) > 1 else np.zeros(n_target, int)
            if len(r) > 1:
                left_closer = np.abs(r[j - 1] - targets) <= np.abs(r[np.minimum(j, len(r) - 1)] - targets)
                nearest = np.where(left_closer, j - 1, np.minimum(j, len(r) - 1))
            else:
                nearest = j
            if np.abs(r[nearest] - targets).max() > step:
                continue
            res[t] = activity[f[nearest]].mean(0)
        out[name] = res
        info[name] = dict(n_target=n_target, frac_trials_dropped=float(np.mean(~np.isfinite(res[:, 0]))))
    return (out, info) if return_info else out


def trial_window_profiles(activity, k, rel, frame_ok, n_trials: int, window: tuple, n_bins: int):
    """Position-matched response PROFILE per trial: activity of the nearest
    imaging frame to each of `n_bins` evenly spaced positions (sub-window
    midpoints) inside `window` (start_cm, end_cm). Returns (n_trials, N,
    n_bins); trials with a position more than one sub-window width from
    any frame are NaN. Same sampling as trial_window_means(equalize=...),
    but keeping the within-window time course (what the Chen et al. 2013
    template discriminant uses)."""
    activity = np.asarray(activity, float)
    lo, hi = window
    step = (hi - lo) / n_bins
    targets = lo + (np.arange(n_bins) + 0.5) * step
    ok = frame_ok & (k >= 0) & np.isfinite(rel) & (rel >= lo) & (rel < hi)
    rows, cols = np.clip(k, 0, None)[ok], np.where(ok)[0]
    cnt = np.bincount(rows, minlength=n_trials)
    order = np.argsort(rows, kind="stable")
    cols_s = cols[order]
    starts = np.concatenate([[0], np.cumsum(cnt)])
    out = np.full((n_trials, activity.shape[1], n_bins), np.nan)
    for t in np.where(cnt > 0)[0]:
        f = cols_s[starts[t]:starts[t + 1]]
        r = rel[f]
        srt = np.argsort(r)
        f, r = f[srt], r[srt]
        j = np.clip(np.searchsorted(r, targets), 1, len(r)) if len(r) > 1 else np.zeros(n_bins, int)
        if len(r) > 1:
            left = np.abs(r[j - 1] - targets) <= np.abs(r[np.minimum(j, len(r) - 1)] - targets)
            nearest = np.where(left, j - 1, np.minimum(j, len(r) - 1))
        else:
            nearest = j
        if np.abs(r[nearest] - targets).max() > step:
            continue
        out[t] = activity[f[nearest]].T
    return out


def zscore_frames(activity, frame_ok):
    """Z-score each cell over the included frames (mean 0, SD 1)."""
    A = np.asarray(activity, float)
    m = A[frame_ok].mean(0)
    s = A[frame_ok].std(0)
    s = np.where(s > 0, s, np.nan)
    return (A - m) / s
