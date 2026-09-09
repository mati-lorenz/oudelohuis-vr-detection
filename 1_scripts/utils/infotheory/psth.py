# -*- coding: utf-8 -*-
"""
Trial-aligned averaging ("PSTH"-style) for continuous signals -- adapted
from the lab's `psth.py` (Matthijs Oude Lohuis, 2023) and cross-checked
against the real pipeline's tensor_utils.py/behavior_signals.py, which
settle a schema question this module got wrong in an earlier version:

    `stimStart` (and rewardZoneStart/rewardZoneEnd) in trialdata are
    POSITIONS, on the SAME (possibly session-accumulating) scale as
    `zpos` in behaviordata/videodata -- NOT timestamps. The real
    pipeline always compares them directly: `zpos_F - z_T[k]` where
    `z_T = trialdata['stimStart']` (see compute_tensor_space's callers
    in tensor_utils.py and behavior_signals.py). A session's `zpos` can
    still span many thousands of cm if it accumulates rather than
    resets every trial/lap -- each trial's own `stimStart` sits at the
    matching point on that same accumulating scale, so the subtraction
    still lands in a small, sensible range (e.g. -80 to +60 cm).

Two alignment domains:
    - position: bin by (zpos - trial['stimStart']) -- direct
      subtraction, exactly like the real pipeline. Since `zpos` is
      monotonically non-decreasing, `searchsorted` finds each trial's
      relevant sample range directly on the POSITION axis -- exact, no
      time-window guess needed.
    - time: bin by (t - t_onset[trial]), where t_onset is the wall-clock
      time at which the position trace first reaches trial['stimStart']
      -- derived via `derive_onset_time`, since stimStart itself isn't a
      timestamp.

`continuous` should be the FULL session trace, not one restricted to a
trial's own [tStart, tEnd] (e.g. via `restrict_to_engaged`) or to
engaged time only -- a window around trial k's stimulus onset
legitimately needs samples from before tStart (the tail end of the
previous trial / the ITI) and after tEnd. Restrict by ENGAGEMENT via
which trials you pass in `trialdata` (only engaged trials), not by
restricting `continuous` itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binned_statistic


def derive_onset_time(trialdata: pd.DataFrame, continuous: pd.DataFrame,
                       position_value_col: str = "stimStart", position_col: str = "zpos",
                       time_col: str = "ts", out_col: str = "_onset_time") -> pd.DataFrame:
    """For each trial, the wall-clock time at which the continuous
    position trace first reaches trial[position_value_col] -- since
    that column is a POSITION (see module docstring), the time-domain
    alignment (`align_trials_time`) needs this derived before it has
    anything to anchor on. Uses `searchsorted` on `continuous[position_col]`,
    which must be monotonically non-decreasing (true for a running/VR
    odometer, even one that accumulates across the whole session)."""
    pos = continuous[position_col].to_numpy()
    ts = continuous[time_col].to_numpy()
    n = len(pos)
    onset_times = []
    for _, trial in trialdata.iterrows():
        idx = int(np.clip(np.searchsorted(pos, trial[position_value_col]), 0, n - 1))
        onset_times.append(ts[idx])
    trialdata = trialdata.copy()
    trialdata[out_col] = onset_times
    return trialdata


def align_trials_time(trialdata: pd.DataFrame, continuous: pd.DataFrame, columns: list[str],
                       t_pre: float, t_post: float, binsize: float,
                       onset_col: str = "_onset_time", rate_columns: list[str] | None = None) -> pd.DataFrame:
    """One row per (trial, time bin, variable): mean of that variable in
    that bin, time relative to `onset_col` -- an actual TIMESTAMP column
    (default `_onset_time`, as produced by `derive_onset_time`; `stimStart`
    itself is a position, not a time -- see module docstring). Long
    format so it's easy to merge with trial metadata (outcome, signal,
    ...) afterward for grouped plotting.

    `rate_columns` (e.g. ["lick"]): sum instead of mean, then divide by
    binsize to get events/second -- the right quantity for a 0/1 event
    channel like individual lick detections."""
    rate_columns = set(rate_columns or [])
    binedges = np.arange(t_pre, t_post + binsize, binsize)
    bincenters = (binedges[:-1] + binedges[1:]) / 2
    ts = continuous["ts"].to_numpy()

    rows = []
    for _, trial in trialdata.iterrows():
        onset = trial[onset_col]
        lo, hi = np.searchsorted(ts, [onset + t_pre, onset + t_post])
        if hi <= lo:
            continue
        rel_t = ts[lo:hi] - onset
        window = continuous.iloc[lo:hi]
        for col in columns:
            statistic = "sum" if col in rate_columns else "mean"
            stat, _, _ = binned_statistic(rel_t, window[col].to_numpy(), statistic=statistic, bins=binedges)
            if col in rate_columns:
                stat = stat / binsize
            rows.extend({"trialNumber": trial.get("trialNumber", np.nan), "t_rel": c,
                         "variable": col, "value": v} for c, v in zip(bincenters, stat))
    return pd.DataFrame(rows)


def align_trials_position(trialdata: pd.DataFrame, continuous: pd.DataFrame, columns: list[str],
                           s_pre: float, s_post: float, binsize: float,
                           position_onset_col: str = "stimStart", position_col: str = "zpos",
                           rate_columns: list[str] | None = None) -> pd.DataFrame:
    """One row per (trial, position bin, variable): mean of that
    variable, position relative to trial[position_onset_col] -- a
    direct subtraction against `continuous[position_col]`, matching the
    real pipeline's `zpos_F - z_T[k]` convention exactly (both columns
    are positions on the same scale, whether or not that scale resets
    every trial). `continuous[position_col]` must be monotonically
    non-decreasing for the `searchsorted` lookup below to be correct.

    `rate_columns` (e.g. ["lick"]): sum instead of mean, then divide by
    binsize to get events per unit of `position_col` (e.g. licks/cm).
    """
    rate_columns = set(rate_columns or [])
    binedges = np.arange(s_pre, s_post + binsize, binsize)
    bincenters = (binedges[:-1] + binedges[1:]) / 2
    pos = continuous[position_col].to_numpy()

    rows = []
    for _, trial in trialdata.iterrows():
        z0 = trial[position_onset_col]
        lo, hi = np.searchsorted(pos, [z0 + s_pre, z0 + s_post])
        if hi <= lo:
            continue
        window = continuous.iloc[lo:hi]
        rel_pos = window[position_col].to_numpy() - z0
        for col in columns:
            statistic = "sum" if col in rate_columns else "mean"
            stat, _, _ = binned_statistic(rel_pos, window[col].to_numpy(), statistic=statistic, bins=binedges)
            if col in rate_columns:
                stat = stat / binsize
            rows.extend({"trialNumber": trial.get("trialNumber", np.nan), "s_rel": c,
                         "variable": col, "value": v} for c, v in zip(bincenters, stat))
    return pd.DataFrame(rows)


def compute_position_bin_indices(trialdata: pd.DataFrame, continuous: pd.DataFrame,
                                  s_pre: float, s_post: float, binsize: float,
                                  position_onset_col: str = "stimStart",
                                  position_col: str = "zpos") -> tuple:
    """Precompute, ONCE per session, which `continuous` row-indices
    (frames) fall in each position bin, pooled across ALL trials --
    i.e. everything `compute_position_binned_information` needs EXCEPT
    which columns to pull values from. Exists because that function
    was originally called once per (cell, predictor) pair, redoing this
    identical trial-windowing loop every single time even though it
    doesn't depend on which cell/predictor is being analyzed -- with
    hundreds of cells x several predictors, that redundant looping
    dominated the runtime. Call this once per session, then
    `mi_from_position_bins` per (cell, predictor) pair, reusing the
    same bin_frame_indices -- the actual fix used by
    2f_lag_information.py.

    Returns
    -------
    bincenters : 1D array
    bin_frame_indices : list of 1D int arrays, one per bin -- pooled
        `continuous` row-positions (i.e. .iloc-style integer indices,
        not the DataFrame's .loc index) falling in that bin, across
        every trial
    """
    binedges = np.arange(s_pre, s_post + binsize, binsize)
    bincenters = (binedges[:-1] + binedges[1:]) / 2
    pos = continuous[position_col].to_numpy()

    pooled_idx = [[] for _ in bincenters]
    for _, trial in trialdata.iterrows():
        z0 = trial[position_onset_col]
        lo, hi = np.searchsorted(pos, [z0 + s_pre, z0 + s_post])
        if hi <= lo:
            continue
        rel_pos = pos[lo:hi] - z0
        bin_idx = np.digitize(rel_pos, binedges) - 1
        valid = (bin_idx >= 0) & (bin_idx < len(bincenters))
        frame_idx = np.arange(lo, hi)[valid]
        for b in np.unique(bin_idx[valid]):
            pooled_idx[b].append(frame_idx[bin_idx[valid] == b])

    bin_frame_indices = [np.concatenate(chunks) if chunks else np.array([], dtype=int)
                          for chunks in pooled_idx]
    return bincenters, bin_frame_indices


def mi_from_position_bins(activity: np.ndarray, reference: np.ndarray, bincenters, bin_frame_indices,
                           mi_bins: int = 12, n_shuffles: int = 0, min_samples_per_bin: int = 30,
                           rng=None) -> pd.DataFrame:
    """The per-(cell, predictor) half of `compute_position_binned_information`,
    given the session's bin_frame_indices already computed once by
    `compute_position_bin_indices` -- just slices and computes MI per
    bin, no trial looping. Same return format (s_rel, n_samples, mi_bits)."""
    from .info_theory import mutual_information_hist, mutual_information_shuffle

    rows = []
    for s_rel, idx in zip(bincenters, bin_frame_indices):
        if len(idx) == 0:
            continue
        a, r = activity[idx], reference[idx]
        valid = np.isfinite(a) & np.isfinite(r)
        a, r = a[valid], r[valid]
        if len(a) < min_samples_per_bin:
            continue
        if n_shuffles > 0:
            mi = mutual_information_shuffle(r, a, bins=mi_bins, n_shuffles=n_shuffles, rng=rng)
            mi_bits = mi.bits_corrected
        else:
            mi_bits = mutual_information_hist(r, a, bins=mi_bins)
        rows.append({"s_rel": s_rel, "n_samples": len(a), "mi_bits": mi_bits})
    return pd.DataFrame(rows)


def compute_position_binned_information(trialdata: pd.DataFrame, continuous: pd.DataFrame,
                                          activity_col: str, reference_col: str,
                                          s_pre: float, s_post: float, binsize: float,
                                          position_onset_col: str = "stimStart", position_col: str = "zpos",
                                          mi_bins: int = 12, n_shuffles: int = 0,
                                          min_samples_per_bin: int = 30, rng=None) -> pd.DataFrame:
    """The information-theoretic analogue of `align_trials_position`:
    instead of averaging a variable within each position bin,
    pools every RAW SAMPLE (every individual frame, from every trial)
    that falls in a given position bin and computes mutual information
    between `activity_col` and `reference_col` from that pooled sample
    set -- i.e. a position-resolved MI curve, not a position-resolved
    mean. `reference_col` should already be whatever lag-shifted version
    of a variable the caller wants tested (this function does no
    shifting itself -- shift the continuous DataFrame's column BEFORE
    calling, e.g. via `continuous[col].shift(lag_frames)`).

    PERFORMANCE NOTE: if calling this for MANY (activity_col,
    reference_col) pairs against the SAME trialdata/continuous (e.g.
    many cells x several predictors), this redoes an expensive
    per-trial windowing loop every single call even though that
    windowing doesn't depend on which columns are being compared --
    use `compute_position_bin_indices` (once per session) +
    `mi_from_position_bins` (once per pair) instead; see
    2f_lag_information.py, which hit exactly this cost with hundreds of
    cells.

    `n_shuffles=0` (default): plain histogram MI, no bias-correction
    shuffle test -- fast, appropriate for lag-SEARCH use (relative
    ordering across lags is what matters, not calibrated bits). Pass
    n_shuffles>0 for the final, reported version (shuffle-corrected
    bits, matching this project's convention elsewhere).

    `min_samples_per_bin`: a position bin with fewer pooled samples than
    this is skipped (returns no row for it) rather than computing MI on
    too few points to be meaningful.

    Returns
    -------
    DataFrame: one row per position bin with enough pooled samples --
    s_rel (bin center), n_samples, mi_bits (mi_bits_corrected if
    n_shuffles>0, else the raw/uncorrected estimate)
    """
    from .info_theory import mutual_information_hist, mutual_information_shuffle

    binedges = np.arange(s_pre, s_post + binsize, binsize)
    bincenters = (binedges[:-1] + binedges[1:]) / 2
    pos = continuous[position_col].to_numpy()
    activity = continuous[activity_col].to_numpy()
    reference = continuous[reference_col].to_numpy()

    pooled_activity = [[] for _ in bincenters]
    pooled_reference = [[] for _ in bincenters]

    for _, trial in trialdata.iterrows():
        z0 = trial[position_onset_col]
        lo, hi = np.searchsorted(pos, [z0 + s_pre, z0 + s_post])
        if hi <= lo:
            continue
        rel_pos = pos[lo:hi] - z0
        bin_idx = np.digitize(rel_pos, binedges) - 1
        act_window = activity[lo:hi]
        ref_window = reference[lo:hi]
        valid = np.isfinite(act_window) & np.isfinite(ref_window) & (bin_idx >= 0) & (bin_idx < len(bincenters))
        for b in np.unique(bin_idx[valid]):
            mask = valid & (bin_idx == b)
            pooled_activity[b].append(act_window[mask])
            pooled_reference[b].append(ref_window[mask])

    rows = []
    for b, s_rel in enumerate(bincenters):
        if not pooled_activity[b]:
            continue
        a = np.concatenate(pooled_activity[b])
        r = np.concatenate(pooled_reference[b])
        if len(a) < min_samples_per_bin:
            continue
        if n_shuffles > 0:
            mi = mutual_information_shuffle(r, a, bins=mi_bins, n_shuffles=n_shuffles, rng=rng)
            mi_bits = mi.bits_corrected
        else:
            mi_bits = mutual_information_hist(r, a, bins=mi_bins)
        rows.append({"s_rel": s_rel, "n_samples": len(a), "mi_bits": mi_bits})
    return pd.DataFrame(rows)
