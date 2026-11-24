# -*- coding: utf-8 -*-
"""
tuning_common.py
================
The shared core of 1f (everything vs PUPIL) and 1g (everything vs SPEED), so that the two scripts mirror each other: the same
y variables, the same bin logic, the same levels and the same figures, with only the x variable different.

Levels (rows of the tuning table; `level` column)
    whole_trial   every time sample (frame) of the selected trials, pooled (frames are autocorrelated, which does not bias a mean
                  curve; it only makes p-values over frames meaningless -- this level is descriptive)
    <window>      one value per trial: the mean of x and of every y in a position window (cm relative to the stimulus onset),
                  e.g. pre_stim / stim / reward. Also the trial's response (lick_response = P(lick)).
    sections      the INDEPENDENT-ish alternative to frames for the whole trial: the trial is cut into non-overlapping SECTION_WIDTH cm
                  sections (SECTION_RANGE) and every (trial, section) mean is one sample. Use these (not the frames) for any test.
Bins: quantiles of x WITHIN each session (equal occupancy, ties broken in time order); with stop_bin (speed) bin 0 = stationary
(x <= stop_value) and the rest are quantiles of the moving samples. Per unit (session / mouse) the bin means are computed first, then
mean +- SEM over units; pooled = sums over all sessions (sample-weighted), no error bars (project convention).

EQUAL-WIDTH BINS (scheme="equal_width", x_bins_fixed): an alternative to the default quantile scheme (scheme="quantile"),
switched on by passing `fixed_range=(lo, hi)` to session_tuning. Bin edges are FIXED, equal-width and the SAME in every
session (not re-derived per session like the quantiles), so bin k means the same physical range of x everywhere and the
x-axis spacing is linear -- at the cost of uneven sample counts per bin for a skewed/zero-inflated x (which is exactly
why the quantile scheme is the default; see the project's equipopulated-vs-equispaced discussion). stop_bin still carves
out the same floor bin first; values outside [lo, hi) are dropped (-1), as elsewhere in the project's fixed-width binning
(2j/2k's position bins). Both schemes can be computed side by side (session_tuning returns both, tagged by the `scheme`
column) without one overwriting the other; plot_tuning_figures(..., scheme=...) picks which one a given figure call draws.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata

from .continuous import compute_trial_position_window_means
from .plotting import PROTOCOL_COLORS, save_fig

TUNING_VARS = ["runspeed", "pupil_area", "pupil_xpos", "pupil_ypos", "motionenergy", "lick", "pos_rel", "videoPC_0", "videoPC_1"]
NO_WINDOW_Y = ["pos_rel"]            # the position inside a fixed window is its centre
VAR_LABELS = {"runspeed": "speed (cm/s)", "pupil_area": "pupil area (z)", "pupil_xpos": "pupil x (z)",
              "pupil_ypos": "pupil y (z)", "motionenergy": "motion energy (z)", "lick": "lick rate",
              "videoPC_0": "video PC 1 (z)", "videoPC_1": "video PC 2 (z)", "zpos": "position (cm)",
              "pos_rel": "position rel. to stimulus (cm)", "lick_rate": "lick rate (Hz, smoothed)",
              "lick_response": "P(lick) (trial response)"}


# ----------------------------------------------------------------------------------------------------------------------
# session level
# ----------------------------------------------------------------------------------------------------------------------
def x_bins(x, n_bins, stop_bin=False, stop_value=1.0):
    """Bin index of every value (-1 = invalid). stop_bin: bin 0 = stationary (x <= stop_value), the others = equal-occupancy
    quantiles (by rank, ties broken in time order) of the moving values; else n_bins quantiles of everything."""
    x = np.asarray(x, float)
    b = np.full(len(x), -1, dtype=int)
    ok = np.isfinite(x)
    if stop_bin:
        stat = ok & (x <= stop_value)
        b[stat] = 0
        mov, first, n_mov = ok & ~stat, 1, n_bins - 1
    else:
        mov, first, n_mov = ok, 0, n_bins
    n = int(mov.sum())
    if n >= n_mov:
        r = rankdata(x[mov], method="ordinal")
        b[mov] = first + np.minimum((r - 1) * n_mov // n, n_mov - 1)
    return b


def x_bins_fixed(x, lo, hi, n_bins, stop_bin=False, stop_value=1.0):
    """Bin index via FIXED, EQUAL-WIDTH edges over [lo, hi) -- the SAME physical range in every session, unlike
    x_bins' within-session equal-occupancy quantiles. Values outside [lo, hi) are dropped (-1), as elsewhere in the
    project's fixed-width binning (e.g. 2j/2k's position bins) -- a rare extreme sample is excluded, not clipped
    into the edge bin. stop_bin: bin 0 = stationary (x <= stop_value, same definition as x_bins), the remaining
    n_bins-1 bins are equal-width over [lo, hi) of the MOVING values; else n_bins equal-width bins over [lo, hi)."""
    x = np.asarray(x, float)
    b = np.full(len(x), -1, dtype=int)
    ok = np.isfinite(x)
    if stop_bin:
        stat = ok & (x <= stop_value)
        b[stat] = 0
        mov, first, nb = ok & ~stat, 1, n_bins - 1
    else:
        mov, first, nb = ok, 0, n_bins
    step = (hi - lo) / nb
    v = mov & (x >= lo) & (x < hi)
    b[v] = first + np.clip(np.floor((x[v] - lo) / step).astype(int), 0, nb - 1)
    return b


def _bin_rows(level, b, n_bins, x, Y, min_n):
    rows = []
    for name, y in Y.items():
        ok = (b >= 0) & np.isfinite(y) & np.isfinite(x)
        for k in range(n_bins):
            m = ok & (b == k)
            n = int(m.sum())
            if n >= min_n:
                rows.append(dict(level=level, y=name, bin=k, n=n, sum_x=float(x[m].sum()), sum_y=float(y[m].sum())))
    return rows


def tuning_rows(level, x, Y, min_n, n_bins, stop_bin=False, stop_value=1.0):
    """Bin sums (n, sum_x, sum_y) of every y in the WITHIN-SESSION QUANTILE bins of x (x_bins)."""
    b = x_bins(x, n_bins, stop_bin, stop_value)
    return _bin_rows(level, b, n_bins, x, Y, min_n)


def tuning_rows_fixed(level, x, Y, min_n, lo, hi, n_bins, stop_bin=False, stop_value=1.0):
    """Bin sums as tuning_rows, but with FIXED, EQUAL-WIDTH bins (x_bins_fixed) over [lo, hi) instead of
    within-session quantiles -- see the module docstring."""
    b = x_bins_fixed(x, lo, hi, n_bins, stop_bin, stop_value)
    return _bin_rows(level, b, n_bins, x, Y, min_n)


def add_lick_rate(merged_full, smooth_s):
    """Smoothed lick rate (Hz) = boxcar of the lick events, on the whole trace (before any trial restriction)."""
    if "lick" not in merged_full.columns:
        return merged_full
    dt = np.median(np.diff(merged_full["ts"].to_numpy())) if "ts" in merged_full.columns and len(merged_full) > 2 else np.nan
    fs = 1.0 / dt if np.isfinite(dt) and dt > 0 else 1.0
    k = max(1, int(round(smooth_s * fs)))
    merged_full["lick_rate"] = np.convolve(merged_full["lick"].fillna(0).to_numpy(float), np.ones(k) / k, mode="same") * fs
    return merged_full


def sampling_rate(merged_full):
    dt = np.median(np.diff(merged_full["ts"].to_numpy())) if "ts" in merged_full.columns and len(merged_full) > 2 else np.nan
    return 1.0 / dt if np.isfinite(dt) and dt > 0 else 1.0


def add_pos_rel(merged, eng):
    """Position inside the trial relative to the stimulus onset (cm) for every sample (NaN outside the trials)."""
    if "zpos" in merged.columns and "stimStart" in eng.columns:
        ts_m, z_m = merged["ts"].to_numpy(), merged["zpos"].to_numpy(float)
        pos_rel = np.full(len(merged), np.nan)
        for _, tr in eng.iterrows():
            m = (ts_m >= tr["tStart"]) & (ts_m <= tr["tEnd"])
            pos_rel[m] = z_m[m] - tr["stimStart"]
        merged["pos_rel"] = pos_rel
    return merged


def session_tuning(merged, eng, *, x_var, y_vars, rate_vars, fs, windows, section_range, section_width, n_bins,
                   stop_bin=False, stop_value=1.0, min_frames=30, min_trials=5, fixed_range=None):
    """All levels of one session. `merged` = the samples of the selected trials (with pos_rel / lick_rate added), `eng` = the
    selected trials' rows of trialdata. Returns a DataFrame of bin sums (level, y, bin, n, sum_x, sum_y, scheme).
    fixed_range=(lo, hi): when given, ALSO compute every level's bins with FIXED, equal-width edges over [lo, hi)
    (tuning_rows_fixed; same n_bins / stop_bin floor bin as the quantile scheme) alongside the default within-session
    quantile bins -- both are returned, tagged by the `scheme` column ("quantile" / "equal_width"), neither overwriting
    the other (see the module docstring)."""
    ys = [v for v in y_vars if v in merged.columns and merged[v].notna().any()]
    kw = dict(n_bins=n_bins, stop_bin=stop_bin, stop_value=stop_value)
    x = merged[x_var].to_numpy(dtype=float)
    Y = {v: merged[v].to_numpy(dtype=float) * (fs if v in rate_vars else 1.0) for v in ys}      # lick events -> Hz
    rows = tuning_rows("whole_trial", x, Y, min_frames, **kw)
    fixed_rows = tuning_rows_fixed("whole_trial", x, Y, min_frames, *fixed_range, **kw) if fixed_range else []

    resp = eng.set_index("trialNumber")["lickResponse"] if "lickResponse" in eng.columns else None
    ys_w = [v for v in ys if v not in NO_WINDOW_Y]

    def window_table(lo, hi):
        pw = compute_trial_position_window_means(eng, merged, columns=[x_var] + ys_w, s_start=lo, s_end=hi, rate_columns=rate_vars)
        return pw if (not pw.empty and x_var in pw.columns) else None

    for wname, (lo, hi) in windows.items():
        pw = window_table(lo, hi)
        if pw is None:
            continue
        Yw = {v: pw[v].to_numpy(float) for v in ys_w if v in pw.columns}
        if resp is not None and "trialNumber" in pw.columns:
            Yw["lick_response"] = pw["trialNumber"].map(resp).to_numpy(float)
        rows += tuning_rows(wname, pw[x_var].to_numpy(float), Yw, min_trials, **kw)
        if fixed_range:
            fixed_rows += tuning_rows_fixed(wname, pw[x_var].to_numpy(float), Yw, min_trials, *fixed_range, **kw)

    secs = []
    for lo in np.arange(section_range[0], section_range[1] - section_width + 1e-9, section_width):
        pw = window_table(float(lo), float(lo + section_width))
        if pw is not None:
            secs.append(pw)
    if secs:
        ps = pd.concat(secs, ignore_index=True)
        xs = ps[x_var].to_numpy(float)
        Ys = {v: ps[v].to_numpy(float) for v in ys_w if v in ps.columns}
        rows += tuning_rows("sections", xs, Ys, min_trials, **kw)
        if fixed_range:
            fixed_rows += tuning_rows_fixed("sections", xs, Ys, min_trials, *fixed_range, **kw)
    out = pd.DataFrame(rows)
    out["scheme"] = "quantile"
    if fixed_range and fixed_rows:
        outf = pd.DataFrame(fixed_rows)
        outf["scheme"] = "equal_width"
        out = pd.concat([out, outf], ignore_index=True)
    return out


def spearman_with_x(merged, x_var, y_vars):
    """Spearman rho between x and every y over all samples (autocorrelated: descriptive only). `lick` is reported from the
    smoothed lick_rate when present."""
    rows = []
    for y in y_vars:
        col = "lick_rate" if y == "lick" and "lick_rate" in merged.columns else y
        if col in merged.columns and col != x_var and merged[col].notna().any():
            rows.append(dict(var_a=x_var, var_b=y, rho=float(merged[[x_var, col]].corr(method="spearman").iloc[0, 1])))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------------
# unit level and figures
# ----------------------------------------------------------------------------------------------------------------------
def unit_curves(tun, unit_keys, agg, x_col):
    """pooled: sums over all sessions (exact sample-weighted); session / mouse: sums inside the unit, then mean +- SEM over units.
    `x_col` names the column that holds the mean x of the bin. Grouped by `scheme` too (quantile / equal_width, see
    session_tuning) when present, so the two binning schemes never get summed together under the same bin index."""
    keys = ["level", "y", "bin"] + (["scheme"] if "scheme" in tun.columns else [])
    g = tun.groupby(["protocol"] + unit_keys + keys)[["n", "sum_x", "sum_y"]].sum().reset_index()
    g[x_col], g["ymean"] = g["sum_x"] / g["n"], g["sum_y"] / g["n"]
    out = (g.groupby(["protocol"] + keys)
             .agg(**{x_col: (x_col, "mean"), "ymean": ("ymean", "mean"), "sem": ("ymean", "sem"), "n_units": ("ymean", "size"),
                     "n": ("n", "sum")}).reset_index())
    out["sem"] = out["sem"].fillna(0.0)
    if agg == "pooled":
        out["sem"] = 0.0
    return out


def corr_summary(sp, agg):
    """Mean +- SEM over units of the Spearman rho (mouse: per-mouse mean first)."""
    s = sp.copy()
    if agg == "mouse":
        s = s.groupby(["protocol", "animal_id", "var_a", "var_b"])["rho"].mean().reset_index()
    out = s.groupby(["protocol", "var_a", "var_b"])["rho"].agg(rho="mean", sem="sem", n_units="size").reset_index()
    out["sem"] = out["sem"].fillna(0.0)
    return out


def draw_curve(ax, sub, protocol, x_col, agg, label=None):
    sub = sub.sort_values("bin")
    ax.errorbar(sub[x_col], sub["ymean"], yerr=sub["sem"] if agg != "pooled" else None, color=PROTOCOL_COLORS[protocol],
                marker="o", markersize=4, lw=1.4, capsize=2.5, elinewidth=0.9, label=protocol if label is None else label)


def plot_tuning_figures(curves, *, figdir, names, x_col, x_label, x_name, protocols, agg, unit, ys, windows, rho_lookup,
                        bins_note, trial_note, section_range, section_width, var_labels=VAR_LABELS, x_mode="value",
                        stop_bin=False, scheme="quantile"):
    """The three mirrored figures: whole trial (frames), per window, sections.
    names = (whole_trial, by_window, sections) file stems.
    x_mode "value": x = the mean x of each bin (cm/s, z, ...). "quantile": x = the QUANTILE (bin number, equal spacing; with
    stop_bin bin 0 is labelled "stop"), the mean value of x is not used at all.
    scheme: "quantile" (default, within-session equal-occupancy bins) or "equal_width" (fixed, equal-width bins shared
    across sessions -- see x_bins_fixed / session_tuning's fixed_range). Selects which of the two binning schemes this
    call draws, when `curves` carries both (i.e. session_tuning was given fixed_range)."""
    if curves is None or not len(curves):
        return
    if "scheme" in curves.columns:
        curves = curves[curves["scheme"] == scheme]
        if not len(curves):
            return
    if x_mode == "quantile":
        curves = curves.copy()
        curves["q"] = curves["bin"] + (0 if stop_bin else 1)
        x_col = "q"
        x_label = f"{x_name} quantile (within session)"
        x_note = "x = quantile, not the mean value"
    else:
        x_label = f"{x_label}, mean of the {'equal-width' if scheme == 'equal_width' else 'quantile'} bin"
        x_note = None

    def _ticks(ax):
        if x_mode == "quantile":
            qs = sorted(curves["q"].unique())
            ax.set_xticks(qs)
            ax.set_xticklabels(["stop" if (stop_bin and q == 0) else str(int(q)) for q in qs], fontsize=7)
    ys_present = [y for y in ys if (curves["y"] == y).any()]
    err = f"mean +- SEM over {unit}s" if agg != "pooled" else "pooled over sessions"
    if x_note:
        err += f"; {x_note}"
    proto_txt = "/".join(protocols)

    def grid_one_level(level, fname, title, xlabel, ys_lvl):
        lv = curves[curves["level"] == level]
        if lv.empty or not ys_lvl:
            return
        ncol = min(4, len(ys_lvl))
        nrow = int(np.ceil(len(ys_lvl) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 3.4 * nrow), squeeze=False)
        for ax, y in zip(axes.ravel(), ys_lvl):
            for protocol in protocols:
                sub = lv[(lv["y"] == y) & (lv["protocol"] == protocol)]
                if sub.empty:
                    continue
                r = rho_lookup(protocol, y) if rho_lookup is not None and level == "whole_trial" else np.nan
                draw_curve(ax, sub, protocol, x_col, agg, label=f"{protocol}  ρ={r:+.2f}" if np.isfinite(r) else protocol)
            ax.legend(frameon=False, fontsize=7, loc="best", handlelength=1.2)
            ax.set_title(var_labels.get(y, y), fontsize=9)
            ax.set_xlabel(xlabel, fontsize=8)
            _ticks(ax)
            ax.grid(alpha=0.2, lw=0.5)
        for ax in axes.ravel()[len(ys_lvl):]:
            ax.axis("off")
        fig.suptitle(title, y=1.0, fontsize=10)
        fig.tight_layout()
        save_fig(fig, figdir, fname)
        plt.close(fig)

    ys_nowin = [y for y in ys_present if y not in NO_WINDOW_Y]
    grid_one_level("whole_trial", names[0],
                   f"Tuning to {x_name}, whole trial (every frame of the {proto_txt} {trial_note}, pooled). {bins_note}. {err}; ρ = Spearman with {x_name}",
                   x_label, ys_present)

    windows_present = [w for w in windows if (curves["level"] == w).any()]
    ys_win = [y for y in list(ys_nowin) + ["lick_response"] if (curves["y"] == y).any()]
    if windows_present and ys_win:
        fig, axes = plt.subplots(len(windows_present), len(ys_win), figsize=(2.9 * len(ys_win), 2.6 * len(windows_present)),
                                 squeeze=False, sharey="col")
        for r, w in enumerate(windows_present):
            for c, y in enumerate(ys_win):
                ax = axes[r][c]
                for protocol in protocols:
                    sub = curves[(curves["level"] == w) & (curves["y"] == y) & (curves["protocol"] == protocol)]
                    if len(sub):
                        draw_curve(ax, sub, protocol, x_col, agg)
                ax.grid(alpha=0.2, lw=0.5)
                _ticks(ax)
                if r == 0:
                    ax.set_title(var_labels.get(y, y), fontsize=9)
                if r == len(windows_present) - 1:
                    ax.set_xlabel(x_label, fontsize=8)
                if c == 0:
                    lo, hi = windows[w]
                    ax.set_ylabel(f"{w}\n[{lo:g}, {hi:g}] cm", fontsize=8)
        axes[0][0].legend(frameon=False, fontsize=7, loc="upper left")
        fig.suptitle(f"Tuning to {x_name} per trial window (window means of the {trial_note}; {bins_note}; y shared per column; {err})",
                     y=1.0, fontsize=10)
        fig.tight_layout()
        save_fig(fig, figdir, names[1])
        plt.close(fig)

    grid_one_level("sections", names[2],
                   f"Tuning to {x_name}, whole trial as {section_width:g} cm SECTION means ({section_range[0]:g}..{section_range[1]:g} cm from the "
                   f"stimulus onset; one sample per trial and section; {trial_note}). {bins_note}. {err}",
                   x_label, ys_nowin)
