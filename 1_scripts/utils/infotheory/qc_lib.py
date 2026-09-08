# -*- coding: utf-8 -*-
"""
qc_lib.py
==========
Sanity checks for neural recordings: flags silent neurons, likely
artifacts, and excessively noisy cells -- with WHICH criterion flagged
each cell kept visible, not just a pass/fail bit, so bad sessions/
areas/populations can be spotted before running downstream analyses.

Ported from the lab's own qc_lib.py, adapted to a functional style
(operating on plain arrays/DataFrames rather than mutating a Session
object in place) and to this project's schema:
  - `flag_bad_cells`'s Fano-factor check used the lab's ses.respmat
    (trial response matrix); this project doesn't have a trial-response-
    matrix pipeline yet (no info-theoretic step has needed one so far),
    so the Fano check here uses spike_stats.fano_factor computed
    directly on the continuous deconvolved trace instead (thresholded,
    i.e. a count-based Fano factor of active frames per window) --
    conceptually the same "how stable is this neuron's activity"
    question, just computed over fixed time windows across the whole
    session rather than per-trial.

Threshold defaults below are the lab's own starting points -- tune per
dataset/imaging system; that tuning is exactly what
2b_activity_statistics.py's plots are for.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skew as _skew

from . import spike_stats as ss

# ------------------------------------------------------------------------- #
# Default thresholds -- tune these per dataset / imaging system
# ------------------------------------------------------------------------- #
RATE_THR = 0.01        # min fraction of frames with above-baseline signal (silent-neuron cutoff)
NOISE_THR = 100.0       # max celldata['noise_level']
FANO_THR = 3.0          # max (count-based) Fano factor across fixed time windows -- NOTE: this is
                        # recalibrated from the lab's original 1e3 default, which was for a
                        # MAGNITUDE-weighted, per-trial Fano factor (can legitimately reach the
                        # hundreds/thousands for heavy-tailed event amplitudes -- see
                        # spike_stats.fano_factor's docstring). The count-based Fano this port
                        # actually uses (thresholded at >0, fixed time windows, no trial structure)
                        # lives on a much smaller natural scale (order ~1 for regular activity,
                        # with a bursty/unstable tail reaching several-fold higher) -- 1e3 would
                        # never trigger at all. Re-tune this per dataset like every other threshold here.
SKEW_MIN = -1.0         # traces shouldn't be strongly left-skewed (saturation/clipping artifact)
FLAT_STD_THR = 1e-6     # ~zero variance = dead/silent channel
NAN_FRAC_THR = 0.01     # max tolerated fraction of NaN samples

REASON_COLS = ["flat", "lowrate", "nan", "lowskew", "highfano", "noisy"]
REASON_LABELS = {
    "flat": "flat trace (std too low)", "lowrate": "low activity rate",
    "nan": "too many NaNs", "lowskew": "low/negative skew (saturation)",
    "highfano": "unstable across time (high Fano)", "noisy": "high noise_level",
}
REASON_COLORS = {
    "flat": "tab:gray", "lowrate": "tab:blue", "nan": "tab:red",
    "lowskew": "tab:orange", "highfano": "tab:purple", "noisy": "tab:brown",
}


def compute_qc_metrics(calciumdata, fs: float, fano_window_sec: float = 1.0) -> pd.DataFrame:
    """Per-neuron QC metrics computed directly from the continuous
    deconvolved trace. Returns a DataFrame (one row per neuron, in
    calciumdata's column order) with:
        qc_rate      : fraction of frames with above-baseline signal
                       (baseline = 20th percentile + 2*std, matching a
                       simple "clearly above noise floor" criterion --
                       NOT the same as spike_stats.sparsity's >0
                       threshold, which is more permissive)
        qc_std       : std of the raw trace (flags flat/dead channels)
        qc_skew      : skewness of the raw trace
        qc_nan_frac  : fraction of NaN samples in the trace
        qc_fano      : count-based Fano factor (spike_stats.fano_factor,
                       thresholded at >0) over fano_window_sec windows
    """
    X = np.asarray(calciumdata, dtype=float)
    N = X.shape[1]
    nan_frac = np.mean(np.isnan(X), axis=0)

    with np.errstate(invalid="ignore"):
        baseline = np.nanpercentile(X, 20, axis=0)
        thresholded = X > (baseline + 2 * np.nanstd(X, axis=0))
        qc_rate = np.nanmean(thresholded, axis=0)

    qc_std = np.nanstd(X, axis=0)
    qc_skew = np.asarray(_skew(X, axis=0, nan_policy="omit"))
    qc_fano = ss.fano_factor(X, fs, window_sec=fano_window_sec, threshold=0.0)

    return pd.DataFrame({"qc_rate": qc_rate, "qc_std": qc_std, "qc_skew": qc_skew,
                          "qc_nan_frac": nan_frac, "qc_fano": qc_fano})


def flag_bad_cells(celldata_with_qc: pd.DataFrame, rate_thr: float = RATE_THR,
                    noise_thr: float = NOISE_THR, fano_thr: float = FANO_THR,
                    skew_min: float = SKEW_MIN, flat_std_thr: float = FLAT_STD_THR,
                    nan_frac_thr: float = NAN_FRAC_THR) -> pd.DataFrame:
    """Combine QC metrics (from `compute_qc_metrics`, already joined
    onto celldata -- see 2b_activity_statistics.py) into pass/fail
    flags, keeping each individual reason separate. Requires
    'qc_rate'/'qc_std'/'qc_skew'/'qc_nan_frac'/'qc_fano' columns; uses
    'noise_level' if present (celldata's own column), else skips that
    check.

    Returns a copy of `celldata_with_qc` with added columns:
        qc_flag_<reason>  : one boolean column per REASON_COLS entry
        qc_fail_reason     : comma-joined string of active reasons, or 'pass'
        qc_pass             : True iff no reason flag is active
    """
    df = celldata_with_qc.copy()
    for col in ["qc_rate", "qc_std", "qc_skew", "qc_nan_frac"]:
        assert col in df.columns, f"Missing '{col}' -- run compute_qc_metrics first and join it onto celldata."

    fano_col = df["qc_fano"] if "qc_fano" in df.columns else pd.Series(np.nan, index=df.index)
    noise_col = df["noise_level"] if "noise_level" in df.columns else pd.Series(np.nan, index=df.index)

    flags = {
        "flat": (df["qc_std"] < flat_std_thr).to_numpy(),
        "lowrate": (df["qc_rate"] < rate_thr).to_numpy(),
        "nan": (df["qc_nan_frac"] > nan_frac_thr).to_numpy(),
        "lowskew": (df["qc_skew"] < skew_min).to_numpy(),
        "highfano": (fano_col > fano_thr).to_numpy(),
        "noisy": (noise_col > noise_thr).to_numpy(),
    }
    for reason in REASON_COLS:
        df[f"qc_flag_{reason}"] = flags[reason]

    reasons_df = pd.DataFrame({r: flags[r] for r in REASON_COLS}, index=df.index)
    df["qc_fail_reason"] = reasons_df.apply(
        lambda row: ",".join(row.index[row.to_numpy()]) if row.any() else "pass", axis=1)
    df["qc_pass"] = ~np.any(list(flags.values()), axis=0)
    return df


def summarize_qc(celldata_qc: pd.DataFrame, groupby=("session_id", "roi_name", "labeled")) -> pd.DataFrame:
    """Pass/fail + per-reason breakdown, grouped by any combination of
    celldata columns. Requires flag_bad_cells to have been run first
    (i.e. `celldata_qc` has 'qc_pass' and the 'qc_flag_*' columns)."""
    assert "qc_pass" in celldata_qc.columns, "Run flag_bad_cells first."
    groupby = [g for g in groupby if g in celldata_qc.columns]
    agg_dict = {"n_total": ("qc_pass", "size"), "n_pass": ("qc_pass", "sum")}
    agg_dict.update({r: (f"qc_flag_{r}", "sum") for r in REASON_COLS})
    summary = celldata_qc.groupby(groupby).agg(**agg_dict).reset_index()
    summary["frac_pass"] = summary["n_pass"] / summary["n_total"]
    return summary.sort_values("frac_pass")
