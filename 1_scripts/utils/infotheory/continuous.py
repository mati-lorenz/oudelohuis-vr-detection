# -*- coding: utf-8 -*-
"""
Helpers for continuous behavior traces (running speed, position, pupil
size, video motion energy): merging behaviordata+videodata onto one
timebase, restricting to engaged-trial time windows, basic pupil
artifact rejection, and position/window-based aggregation. Shared by
1c_behavior and any later step that needs the same continuous-trace
handling (1d_performance_predictors will).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def check_zpos_consistency(behaviordata: pd.DataFrame, videodata: pd.DataFrame | None,
                            tolerance: float = 0.05) -> dict | None:
    """QC check: if both behaviordata and videodata carry their own
    `zpos`, how much do they actually disagree after nearest-timestamp
    alignment? `merge_behavior_video` always keeps behaviordata's copy
    and discards video's -- this quantifies whether that discard could
    matter (e.g. if the two clocks drift, or video's position tracking
    is independently noisy). Returns None if video has no zpos to
    compare against."""
    if (videodata is None or videodata.empty or "zpos" not in videodata.columns
            or "zpos" not in behaviordata.columns):
        return None

    left = behaviordata.sort_values("ts")[["ts", "zpos"]]
    right = videodata.sort_values("ts")[["ts", "zpos"]].rename(columns={"zpos": "zpos_video"})
    merged = pd.merge_asof(left, right, on="ts", direction="nearest", tolerance=tolerance)
    diff = (merged["zpos"] - merged["zpos_video"]).abs().dropna()
    if diff.empty:
        return None
    return {"n_compared": len(diff), "median_abs_diff": float(diff.median()),
            "p95_abs_diff": float(diff.quantile(0.95)), "max_abs_diff": float(diff.max())}


def merge_behavior_video(behaviordata: pd.DataFrame, videodata: pd.DataFrame | None,
                          tolerance: float = 0.05) -> pd.DataFrame:
    """Merge behaviordata and videodata onto behaviordata's timebase
    (nearest-timestamp match -- the two are rarely sampled on exactly
    the same clock). Returns behaviordata unchanged if there's no video
    for this session."""
    if videodata is None or videodata.empty:
        return behaviordata.copy()

    left = behaviordata.sort_values("ts")
    right = videodata.sort_values("ts")
    # Some columns (notably zpos) can legitimately exist in both -- keep
    # behaviordata's copy and drop video's duplicate rather than letting
    # merge_asof silently rename both to `_x`/`_y` suffixes, which would
    # make neither match the plain column name downstream.
    video_cols = [c for c in right.columns if c != "ts" and c not in left.columns]
    return pd.merge_asof(left, right[["ts", *video_cols]], on="ts",
                          direction="nearest", tolerance=tolerance)


def build_calcium_continuous(calciumdata: pd.DataFrame, ts_F, behaviordata: pd.DataFrame) -> pd.DataFrame:
    """Build a continuous DataFrame on the CALCIUM imaging clock (ts_F,
    typically ~5-30Hz), with 'zpos' interpolated from behaviordata's own
    much finer clock (~50-100Hz) via exact linear interpolation
    (np.interp) rather than a nearest-timestamp merge_asof -- cheap and
    exact regardless of how many cell columns `calciumdata` has (unlike
    merge_behavior_video's merge_asof, which would need to carry
    potentially thousands of cell columns through the join).

    The result has 'ts', 'zpos', plus every column of `calciumdata`
    (cell_id names) unchanged -- directly usable with
    `compute_trial_position_window_means`/`bin_by_position`, exactly
    like the behavior/video continuous traces elsewhere in this
    package, since those only ever need 'ts'/'zpos' plus arbitrary
    value columns.
    """
    ts_bhv = behaviordata["ts"].to_numpy()
    zpos_bhv = behaviordata["zpos"].to_numpy()
    zpos_F = np.interp(np.asarray(ts_F, dtype=float), ts_bhv, zpos_bhv)
    out = calciumdata.copy()
    out.insert(0, "zpos", zpos_F)
    out.insert(0, "ts", np.asarray(ts_F, dtype=float))
    return out


def restrict_to_engaged(continuous: pd.DataFrame, trialdata: pd.DataFrame) -> pd.DataFrame:
    """Keep only samples that fall within an engaged trial's [tStart, tEnd]
    window. Uses time windows (not a trialNumber column) so it works
    regardless of whether the continuous data carries its own trial
    numbering."""
    if "engaged" not in trialdata.columns:
        return continuous.copy()

    engaged_trials = trialdata[trialdata["engaged"] == 1]
    if engaged_trials.empty:
        return continuous.iloc[0:0].copy()

    ts = continuous["ts"].to_numpy()
    keep = np.zeros(len(continuous), dtype=bool)
    for _, trial in engaged_trials.iterrows():
        keep |= (ts >= trial["tStart"]) & (ts <= trial["tEnd"])
    return continuous.loc[keep].copy()


def remove_pupil_outliers(continuous: pd.DataFrame, column: str = "pupil_area",
                           z_thresh: float = 5.0) -> pd.DataFrame:
    """Null out (not drop -- other channels' samples at that timestamp
    are still useful) pupil values more than `z_thresh` SDs from the
    session mean, the same blink-artifact rule the earlier QC scripts
    used."""
    if column not in continuous.columns:
        return continuous
    continuous = continuous.copy()
    z = stats.zscore(continuous[column], nan_policy="omit")
    continuous.loc[(z > z_thresh) | (z < -z_thresh), column] = np.nan
    return continuous


def compute_trial_position_window_means(trialdata: pd.DataFrame, continuous: pd.DataFrame,
                                         columns: list[str], s_start: float, s_end: float,
                                         position_onset_col: str = "stimStart", position_col: str = "zpos",
                                         rate_columns: list[str] | None = None) -> pd.DataFrame:
    """Per-trial mean of each of `columns` within
    [trial[position_onset_col] + s_start, trial[position_onset_col] + s_end]
    -- a POSITION window in cm (e.g. s_start=0, s_end=20 for "the 20cm
    starting at stimulus onset"), NOT a time window. Since `stimStart`
    is itself a position (see psth.py's module docstring), this is a
    direct subtraction against `continuous[position_col]`, exactly like
    `psth.align_trials_position`. `continuous[position_col]` must be
    monotonically non-decreasing so the `searchsorted` lookup below is
    correct -- which also means only the relevant slice of the
    continuous trace is ever touched per trial, not the whole session.

    `rate_columns` (e.g. ["lick"]): report a RATE in events per cm
    (sum of events divided by the window width `s_end - s_start`)
    instead of the mean of a 0/1 indicator, which would just be
    "fraction of samples with a lick" -- not directly interpretable
    without knowing the sampling density.

    This is the "one independent-ish sample per trial" unit used for MI
    and regression -- raw ~50Hz samples are heavily autocorrelated, so
    pooling them directly would vastly overstate the effective sample
    size and bias significance tests."""
    rate_columns = set(rate_columns or [])
    pos = continuous[position_col].to_numpy()
    window_width = s_end - s_start
    rows = []
    for _, trial in trialdata.iterrows():
        z0 = trial[position_onset_col]
        lo, hi = np.searchsorted(pos, [z0 + s_start, z0 + s_end])
        row = {"trialNumber": trial.get("trialNumber", np.nan)}
        window_has_data = hi > lo
        window = continuous.iloc[lo:hi] if window_has_data else None
        for col in columns:
            if not window_has_data:
                row[col] = np.nan
            elif col in rate_columns:
                row[col] = window[col].sum() / window_width
            else:
                row[col] = window[col].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def compute_trial_window_means(trialdata: pd.DataFrame, continuous: pd.DataFrame,
                                columns: list[str], pre_s: float = 2.0, post_s: float = 1.0,
                                time_col: str = "_onset_time", rate_columns: list[str] | None = None) -> pd.DataFrame:
    """Per-trial mean of each of `columns` in a [-pre_s, +post_s] window
    around `time_col`. `time_col` must be an actual TIMESTAMP column
    (default `_onset_time`, as produced by `psth.derive_onset_time` --
    NOT `stimStart` directly, which is a position in this project's
    schema; see psth.py's module docstring). One row per trial; trials
    with no samples in their window get NaN.

    `rate_columns` (e.g. ["lick"]): instead of the mean of a 0/1
    indicator, report a RATE in events/second (sum of events divided by
    the window's duration) -- the right quantity for a binary event
    channel like individual lick detections, where "mean" would just be
    "fraction of samples with a lick" (meaningless without knowing the
    sampling rate) rather than an interpretable lick rate.

    This is the "one independent-ish sample per trial" unit used for MI
    and regression -- raw ~50Hz samples are heavily autocorrelated
    (a speed sample at t and t+20ms are nearly the same value), so
    pooling them directly would vastly overstate the effective sample
    size and bias significance tests."""
    rate_columns = set(rate_columns or [])
    ts = continuous["ts"].to_numpy()
    window_s = pre_s + post_s
    rows = []
    for _, trial in trialdata.iterrows():
        center = trial[time_col]
        window = (ts >= center - pre_s) & (ts <= center + post_s)
        row = {"trialNumber": trial.get("trialNumber", np.nan)}
        for col in columns:
            if not window.any():
                row[col] = np.nan
            elif col in rate_columns:
                row[col] = continuous.loc[window, col].sum() / window_s
            else:
                row[col] = continuous.loc[window, col].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def bin_by_position(continuous: pd.DataFrame, columns: list[str], bins: np.ndarray,
                     position_col: str = "zpos", rate_columns: list[str] | None = None) -> pd.DataFrame:
    """Mean +/- SEM of each of `columns`, binned by `position_col`. One
    row per bin (bin center + stats) -- for corridor/position-tuning
    plots.

    `rate_columns` (e.g. ["lick"]): report events per unit of
    `position_col` (e.g. licks/cm) instead of the mean -- sum of events
    in the bin divided by the bin width."""
    rate_columns = set(rate_columns or [])
    binned = pd.cut(continuous[position_col], bins=bins)
    grouped = continuous.groupby(binned, observed=True)

    mean_cols = [c for c in columns if c not in rate_columns]
    means = grouped[mean_cols].mean() if mean_cols else pd.DataFrame(index=grouped.size().index)
    sems = grouped[mean_cols].sem() if mean_cols else pd.DataFrame(index=grouped.size().index)
    out = means.add_suffix("_mean").join(sems.add_suffix("_sem"))

    if rate_columns:
        sums = grouped[list(rate_columns)].sum()
        widths = pd.Series([interval.length for interval in sums.index], index=sums.index)
        for col in rate_columns:
            out[f"{col}_mean"] = sums[col] / widths

    out["bin_center"] = [interval.mid for interval in grouped.size().index]
    return out.reset_index(drop=True)
