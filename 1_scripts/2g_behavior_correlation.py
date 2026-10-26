# -*- coding: utf-8 -*-
"""
2g_behavior_correlation
========================
Which cells follow running speed, pupil size, licking and position?
(correlation analysis after Chen et al. 2013). DN sessions only.

For every included cell and every behavioral variable (VARIABLES below)
the Pearson correlation r between the cell's activity and the variable is
computed FRAME BY FRAME on the imaging clock, over engaged-trial frames:

    runspeed     mean running speed in each imaging frame
    pupil_area   mean pupil area (blink outliers removed, as in 1c/2c)
    lick         lick events/s in each frame, smoothed (LICK_SMOOTH_S)
    position     position relative to the stimulus onset (cm), frames within
                 [-S_PRE, +S_POST) of an engaged trial's stimStart. This is
                 a LINEAR correlation: it finds cells whose activity ramps up
                 or down along the corridor, not cells with a localized peak
                 (that is 2b_activity_statistics (raster part)).

Each variable except position is analysed twice:
    <var>            raw frame-level correlation
    <var>|position   correlation of the RESIDUALS after subtracting, for both the
                     cell and the behavior, their mean at each position bin
                     (CONTROL_POSITION_BIN_CM) relative to stimStart. Running,
                     licking and pupil are all locked to the trial structure, so
                     any cell with a position-locked response correlates with all
                     of them in the raw version; the residual version asks whether
                     the cell follows the behavior's FLUCTUATIONS beyond what the
                     corridor position already predicts.

Lags: the correlation is evaluated at every lag in LAGS_S (default 0-0.5 s;
positive = the cell FOLLOWS the behavior, since calcium/deconvolved signals lag
behavior, cf. the lag windows of Chen et al. 2013) and the lag with the largest
|r| is kept (`lag_s` in the table). The null uses the same best-lag statistic.

Classification (behavior_class.csv): each cell is assigned the behavior in
CLASS_VARIABLES with the largest significant r (on the CLASS_SUFFIX version),
"mixed" if the runner-up significant behavior is within CLASS_MARGIN, and
"unclassified" if none is significant -- the analogue of the touch / whisking /
unclassified cells of Chen et al. (they test the difference of r by bootstrap
over trials; here a fixed margin). 2b_activity_statistics (raster part) reads this table.

Significance: circular-shift null. The behavioral trace is rolled by a
random lag (>= MIN_SHIFT_S seconds, wrapping around the session) N_SHIFTS
times and r recomputed; a cell is significant if its r lies outside the
central 95% of ITS OWN null. The shift keeps the autocorrelation of both
signals, which a trial shuffle would destroy (making everything look
significant). Z-scoring the activity first would not change r (Pearson r
is scale invariant), so raw deconvolved activity is used.

Scope / aggregation (project rules): `--animals`, `--sessions` choose mice/sessions (a restricted run writes
to out/selection/<tag>/, never over the full-dataset out/ that 2b reads); `--aggregate pooled|session|mouse`
sets the unit of the group bar plots (pooled = all cells, no error bars; session / mouse = bar is mean +- SEM
over units, dots = units coloured by mouse). The per-cell tables, the cell-level histograms / scatters and
behavior_class.csv are always cell-level and do not depend on the aggregate. Sessions are processed one per
worker (rule #3); sessions without video simply lack pupil_area.

Relation to 2c/2d: this is a FRAME-level Pearson correlation on the imaging clock (continuous, no binning), while
2c/2d work on trial-window means. Same filters (1b included sessions, engaged trials, cells passing 2a AND 2b), same
predictor definitions; the different clock is deliberate (it is what allows lags and the circular-shift null).

Caveats
-------
* ~5% of cells pass by chance and no correction across cells/variables is applied; compare fractions with the 5% line.
* Running, pupil, licking and position are not independent. A cell correlated with running will often also correlate
  with lick. Look at the r-vs-r scatter (figure 3) before concluding a cell "encodes licking".
* Frame-level r is dominated by slow co-fluctuations (state); it says nothing about causality or timing
  (see 2f_lag_information for lags).

Outputs
-------
2_pipeline/2g_behavior_correlation/
    out/    behavior_correlation_per_cell.csv   one row per (session, cell, variable)
            behavior_class.csv                  one row per cell: class (a CLASS_VARIABLES entry, "mixed" or
                                                "unclassified"), r and lag per behavior  (read by 2b)
            behavior_correlation_by_group[__<aggregate>].csv   per group x variable (x unit): n_cells, mean r,
                                                mean |r|, fraction significant (pos / neg), p_vs_chance (pooled)
            behavior_class_by_group[__<aggregate>].csv         per group x class (x unit): fraction of cells
            figures/*.png
    out/selection/<tag>/...                     the same for a restricted run
    store/  corr_session_<hash>.pkl             per-session caches
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.stats import binomtest

from infotheory import project_config as cfg
from infotheory.pipeline import get_pipeline_paths
from infotheory.session import discover_sessions
from infotheory.neural_encoding import load_cell_inclusion
from infotheory.continuous import merge_behavior_video, remove_pupil_outliers
from infotheory.celldata_utils import SIX_GROUP_ORDER, SIX_GROUP_LABELS, SIX_GROUP_COLORS
from infotheory.groups import six_panel_figure, group_key
from infotheory.plotting import set_style, save_fig, clear_figures
from infotheory.corr_lib import corr_with_shift_null
from infotheory.scope import add_scope_args, scope_from_args, select_sessions, scoped_dirs, data_stamp, unit_noun
from infotheory.session_jobs import run_sessions
from infotheory.unit_plots import group_panels, mouse_palette
from infotheory import tensors as tn
from infotheory import cell_tables as ct

PROTOCOL = "DN"

# variable -> how to put it on the imaging clock: "mean" (average of samples in the
# frame), "rate" (events/s), or "position" (cm relative to stimStart).
# Add e.g. "motionenergy": "mean" to include another column of videodata.
VARIABLES = {"runspeed": "mean", "pupil_area": "mean", "lick": "rate", "position": "position"}
LICK_SMOOTH_S = 0.5              # Gaussian smoothing (SD, seconds) of the lick rate
S_PRE, S_POST = 80.0, 60.0       # position window around stimStart for "position"
CONTROL_POSITION_BIN_CM = 5.0    # bin for the "<var>|position" residual version; None = skip it

LAGS_S = (0.0, 0.5)              # lag range (s) searched; positive = activity FOLLOWS behavior
                                 # (Chen et al. 2013 used 0-0.57 s). (0.0, 0.0) = zero lag only.
CLASS_VARIABLES = ["runspeed", "pupil_area", "lick"]   # behaviors a cell can be classified by
CLASS_SUFFIX = "|position"       # classify on the position-residual correlations ("" = raw)
CLASS_MARGIN = 0.05              # best r must beat the runner-up by this much, else "mixed"

N_SHIFTS = 500
MIN_SHIFT_S = 20.0               # smallest circular shift, seconds
ALPHA = 0.05
RANDOM_STATE = 0

MIN_CELLS_PER_UNIT = 5
JOBLIB_VERBOSE = 5

parser = argparse.ArgumentParser()
add_scope_args(parser)
parser.add_argument("--n-shifts", type=int, default=N_SHIFTS)
args = parser.parse_args()
scope = scope_from_args(args)
AGG = scope.aggregate
UNIT = unit_noun(AGG)
UNITS = "mice" if UNIT == "mouse" else f"{UNIT}s"
AGG_SUFFIX = "" if AGG == "pooled" else f"__{AGG}"
UNIT_KEYS = {"pooled": [], "session": ["session_id"], "mouse": ["animal_id"]}[AGG]
n_shifts = args.n_shifts

paths = get_pipeline_paths(__file__)
table_dir, figdir = scoped_dirs(paths, scope)
clear_figures(figdir)
set_style()
print(scope.describe())

# --------------------------------------------------------------------- #
# 1. Included sessions (1b) and cells (2a + 2b)
# --------------------------------------------------------------------- #
fits_path = paths.out_from("1b_psychometric") / "psychometric_fits.csv"
cell_dist_out = paths.out_from("2a_cell_distribution")
activity_out = paths.out_from("2b_activity_statistics")
for f, hint in [(fits_path, "1b_psychometric.py"), (cell_dist_out / "celldata_combined.csv", "2a_cell_distribution.py"),
                (activity_out / "qc_per_cell.csv", "2b_activity_statistics.py")]:
    if not f.exists():
        raise SystemExit(f"Run {hint} first -- expected {f}")
fits_df = pd.read_csv(fits_path)
included_all = set(fits_df[fits_df["included"] & (fits_df["protocol"] == PROTOCOL)]["session_id"])
all_triples = discover_sessions([PROTOCOL])
triples = select_sessions(all_triples, scope)
selected = {f"{a}_{d}": (p, a, d) for p, a, d in triples}
session_ids = [s for s in selected if s in included_all]
if not session_ids:
    raise SystemExit("None of the selected sessions passed 1b_psychometric's inclusion criteria.")
mouse_colors = mouse_palette(a for _, a, _ in all_triples)
cell_inclusion = load_cell_inclusion(cell_dist_out, activity_out)
cells_by_session = {sid: sorted(sub.loc[sub["include"], "cell_id"]) for sid, sub in cell_inclusion.groupby("session_id")
                    if sid in session_ids}
session_ids = [s for s in session_ids if cells_by_session.get(s)]
if not session_ids:
    raise SystemExit("No included cells in the selected sessions.")
print(f"{len(session_ids)} sessions, {sum(len(v) for v in cells_by_session.values())} included cells")
stamps = {s: data_stamp(*selected[s]) for s in session_ids}
qc_mtime = int((activity_out / "qc_per_cell.csv").stat().st_mtime)


# --------------------------------------------------------------------- #
# 2. Per-session computation (one session per worker, cached)
# --------------------------------------------------------------------- #
def behavior_on_frames(ses, frame_ok):
    """Dict variable -> (T,) trace on the imaging clock."""
    need_video = any(v not in ("lick", "position") and ses.videodata is not None for v in VARIABLES)
    merged = merge_behavior_video(ses.behaviordata, ses.videodata if need_video else None)
    merged = remove_pupil_outliers(merged)
    ts_F = ses.ts_F
    dt = float(np.median(np.diff(ts_F)))
    traces = {}
    for var, mode in VARIABLES.items():
        if mode == "position":
            zpos_F = tn.interp_to_imaging_frames(ses.behaviordata["ts"], ses.behaviordata["zpos"], ts_F)
            stim = ses.trialdata[ses.trialdata["engaged"] == 1]["stimStart"].to_numpy()
            _, rel = tn.position_relative_to_stim(zpos_F, stim, S_PRE, S_POST)
            traces[var] = rel
        elif var not in merged.columns:
            continue
        else:
            x = tn.behavior_to_imaging_frames(merged["ts"], merged[var], ts_F, mode=mode)
            if var == "lick" and LICK_SMOOTH_S:
                x = gaussian_filter1d(np.nan_to_num(x), LICK_SMOOTH_S / dt)
            traces[var] = x
    return traces, dt


def residualize_by_position(X, rel, mask, bin_cm):
    """Subtract the per-position-bin mean (over `mask` frames) from X (T,) or (T,N);
    frames outside `mask` become NaN (1D) / 0 (2D, they are excluded via the mask)."""
    from scipy import sparse
    X2 = X[:, None] if X.ndim == 1 else X
    n_bins = int(np.ceil((S_PRE + S_POST) / bin_cm))
    b = np.clip(np.floor((np.nan_to_num(rel, nan=0.0) + S_PRE) / bin_cm).astype(int), 0, n_bins - 1)
    idx = np.where(mask)[0]
    M = sparse.csr_matrix((np.ones(len(idx)), (b[idx], idx)), shape=(n_bins, len(X2)))
    cnt = np.asarray(M.sum(1)).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        means = (M @ np.nan_to_num(X2)) / cnt[:, None]
    res = np.where(mask[:, None], X2 - np.nan_to_num(means[b]), 0.0)
    return (np.where(mask, res[:, 0], np.nan) if X.ndim == 1 else res)


def process_session(ses, seed):
    sid = ses.session_id
    cell_ids = cells_by_session.get(sid, [])
    if (not cell_ids or ses.calciumdata is None or ses.ts_F is None or ses.trialdata is None
            or ses.behaviordata is None or "engaged" not in ses.trialdata.columns):
        return None
    rng = np.random.default_rng(seed)
    A = ses.calciumdata.loc[:, cell_ids].to_numpy(dtype=float)
    frame_ok = tn.engaged_frame_mask(ses.ts_F, ses.trialdata)
    traces, dt = behavior_on_frames(ses, frame_ok)
    min_shift = int(round(MIN_SHIFT_S / dt))
    lags = np.arange(int(round(LAGS_S[0] / dt)), int(round(LAGS_S[1] / dt)) + 1)
    jobs = [(var, b, A, frame_ok) for var, b in traces.items()]
    if CONTROL_POSITION_BIN_CM and "position" in traces:
        rel = traces["position"]
        in_win = frame_ok & np.isfinite(rel)
        A_res = residualize_by_position(A, rel, in_win, CONTROL_POSITION_BIN_CM)
        for var, b in traces.items():
            if var != "position":
                b_res = residualize_by_position(np.where(np.isfinite(b), b, 0.0), rel,
                                                in_win & np.isfinite(b), CONTROL_POSITION_BIN_CM)
                jobs.append((f"{var}|position", b_res, A_res, in_win & np.isfinite(b_res)))
    records = []
    for var, b, Amat, ok in jobs:
        res = corr_with_shift_null(Amat, b, ok, n_shifts=n_shifts, min_shift=min_shift,
                                   rng=rng, alpha=ALPHA, lags=lags)
        records.append(pd.DataFrame({
            "session_id": sid, "cell_id": cell_ids, "variable": var,
            "r": res["r"], "lag_s": res["lag"] * dt, "null_lo": res["null_lo"], "null_hi": res["null_hi"],
            "p_value": res["p_value"], "significant": res["significant"], "n_frames": res["n_frames"]}))
    return pd.concat(records, ignore_index=True) if records else None


results = run_sessions(
    session_ids, process_session, [PROTOCOL], paths.store, "corr_session",
    key_params=dict(version="2g-v1", variables=VARIABLES, lick_smooth=LICK_SMOOTH_S, control_bin=CONTROL_POSITION_BIN_CM,
                    s_pre=S_PRE, s_post=S_POST, n_shifts=n_shifts, lags_s=LAGS_S, min_shift_s=MIN_SHIFT_S,
                    alpha=ALPHA, qc_mtime=qc_mtime, fits_mtime=int(fits_path.stat().st_mtime)),
    load_kwargs=dict(load_behaviordata=True, load_videodata=True, load_celldata=True, load_calciumdata=True),
    cells_by_session=cells_by_session, base_seed=RANDOM_STATE, n_jobs=scope.n_jobs, recompute=args.recompute,
    verbose=JOBLIB_VERBOSE, extra_key_by_session=stamps, flatten=False)
results = [r for r in results if r is not None and len(r)]
if not results:
    raise SystemExit("No correlation results (no sessions with usable cells).")
corr = pd.concat(results, ignore_index=True)


# --------------------------------------------------------------------- #
# 3. Attach area/label -> six groups; save tables
# --------------------------------------------------------------------- #
cd = pd.read_csv(cell_dist_out / "celldata_combined.csv", low_memory=False)
corr = ct.attach_cell_meta(corr, cd)
corr["abs_r"] = corr["r"].abs()
corr["sig_pos"] = (corr["significant"] & (corr["r"] > 0)).astype(float)
corr["sig_neg"] = (corr["significant"] & (corr["r"] < 0)).astype(float)
corr["significant"] = corr["significant"].astype(bool)
corr.to_csv(table_dir / "behavior_correlation_per_cell.csv", index=False)
print(f"{len(corr)} (session, cell, variable) rows; "
      f"{corr['cell_id'].nunique()} cells in {corr['session_id'].nunique()} sessions")
GROUP_KEYS = [group_key(g) for g in SIX_GROUP_ORDER]
agg_note = {"pooled": "all cells pooled (no error bars)", "session": "bar = mean +- SEM over sessions, dots = sessions",
            "mouse": "bar = mean +- SEM over mice, dots = mice"}[AGG]


# --------------------------------------------------------------------- #
# 3b. Behavior classification (Chen et al. 2013: touch / whisking / unclassified)
# --------------------------------------------------------------------- #
class_vars = [f"{v}{CLASS_SUFFIX}" for v in CLASS_VARIABLES if f"{v}{CLASS_SUFFIX}" in set(corr["variable"])]
cls_df = None
if class_vars:
    sub = corr[corr["variable"].isin(class_vars)]
    r_w = sub.pivot_table(index=["session_id", "cell_id"], columns="variable", values="r")
    s_w = sub.pivot_table(index=["session_id", "cell_id"], columns="variable", values="significant").astype(bool)
    lag_w = sub.pivot_table(index=["session_id", "cell_id"], columns="variable", values="lag_s")
    r_sig = r_w.where(s_w)                       # r only where significant
    best = r_sig.abs().fillna(-1.0).idxmax(axis=1)
    best_abs = r_sig.abs().max(axis=1)
    second = r_sig.abs().apply(lambda row: np.sort(row.dropna().to_numpy())[-2]
                               if row.notna().sum() >= 2 else np.nan, axis=1)
    cls_name = best.str.replace(CLASS_SUFFIX, "", regex=False) if CLASS_SUFFIX else best
    klass = pd.Series("unclassified", index=r_w.index)
    klass[best_abs.notna()] = cls_name[best_abs.notna()]
    klass[(best_abs - second) < CLASS_MARGIN] = "mixed"
    klass[best_abs.isna()] = "unclassified"
    cls_df = pd.DataFrame({"class": klass}).join(r_w.add_prefix("r_")).join(lag_w.add_prefix("lag_s_")).reset_index()
    cls_df = ct.attach_cell_meta(cls_df, cd)
    cls_df.to_csv(table_dir / "behavior_class.csv", index=False)
    class_order = [v for v in CLASS_VARIABLES if v in set(cls_df["class"])] + ["mixed", "unclassified"]
    for c in class_order:
        cls_df[f"is_{c}"] = (cls_df["class"] == c).astype(float)
    t = ct.unit_group_summary(cls_df, [], AGG, [f"is_{c}" for c in class_order])
    t.to_csv(table_dir / f"behavior_class_by_group{AGG_SUFFIX}.csv", index=False)
    print(f"\nBehavior classes (mean fraction of cells per group{'' if AGG == 'pooled' else f', mean over {UNITS}'}; "
          f"classified on {CLASS_SUFFIX or 'raw'} correlations):")
    print(t.groupby("group")[[f"is_{c}" for c in class_order]].mean().reindex(GROUP_KEYS).round(2)
          .rename(columns=lambda c: c[3:]).to_string())

summ = ct.unit_group_summary(corr, ["variable"], AGG, ["r", "abs_r", "sig_pos", "sig_neg"], sig_col="significant")
if AGG == "pooled":
    summ["p_vs_chance"] = [binomtest(int(s_), int(n), ALPHA).pvalue if n > 0 else np.nan
                           for s_, n in zip(summ["n_sig"], summ["n_cells"])]
summ.to_csv(table_dir / f"behavior_correlation_by_group{AGG_SUFFIX}.csv", index=False)

variables = [v for v in VARIABLES if v in set(corr["variable"])]
variables += [f"{v}|position" for v in VARIABLES if f"{v}|position" in set(corr["variable"])]
unit_mean = lambda col: (summ.pivot_table(index="variable", columns="group", values=col, aggfunc="mean")  # noqa: E731
                         .reindex(index=variables, columns=GROUP_KEYS))
print(f"\nFraction of cells significantly correlated (chance = 5%), "
      f"{'pooled' if AGG == 'pooled' else f'mean over {UNITS}'}:")
print(unit_mean("frac_sig").round(2).to_string())
print("\nMean r:")
print(unit_mean("r").round(3).to_string())


# --------------------------------------------------------------------- #
# 4. Figures (always the six groups)
# --------------------------------------------------------------------- #
# 1. r histograms (cell level, not aggregated): rows = variables, columns = six groups
bins = np.linspace(-1, 1, 51)
fig, axes = plt.subplots(len(variables), 6, figsize=(16, 2.3 * len(variables)), sharex=True, squeeze=False)
for i, var in enumerate(variables):
    for j, g in enumerate(SIX_GROUP_ORDER):
        ax = axes[i, j]
        sub = corr[(corr["variable"] == var) & (corr["group"] == group_key(g))]
        sub = sub[np.isfinite(sub["r"])]
        ax.hist(sub.loc[~sub["significant"], "r"], bins=bins, color="#bbbbbb")
        ax.hist(sub.loc[sub["significant"], "r"], bins=bins, color=SIX_GROUP_COLORS[g])
        ax.axvline(0, color="k", lw=0.6)
        ax.text(0.98, 0.95, f"{sub['significant'].mean():.0%} sig\nn={len(sub)}" if len(sub) else "no cells",
                transform=ax.transAxes, ha="right", va="top", fontsize=7)
        if i == 0:
            ax.set_title(SIX_GROUP_LABELS[g], color=SIX_GROUP_COLORS[g], fontsize=10)
        if j == 0:
            ax.set_ylabel(f"{var}\ncells")
        if i == len(variables) - 1:
            ax.set_xlabel("correlation r")
fig.suptitle("Cell-behavior correlation (colored = outside 95% of circular-shift null; all cells, not aggregated)")
fig.tight_layout()
save_fig(fig, figdir, "1_r_hist_by_variable")
plt.close(fig)

# 2. fraction significant per group, one panel per variable, at the chosen unit; then split by sign of r
corr["sig_f"] = corr["significant"].astype(float)
group_panels(corr, "variable", variables, "sig_f", AGG, figdir, "2_frac_significant_by_group",
             f"Fraction of cells significantly correlated with behavior; dashed = chance ({ALPHA:.0%}); {agg_note}",
             ylabel="fraction of cells significant", hline=ALPHA, mouse_colors=mouse_colors,
             min_cells_per_unit=MIN_CELLS_PER_UNIT)
signed = pd.concat([corr.assign(sign="r > 0", frac=corr["sig_pos"]), corr.assign(sign="r < 0", frac=corr["sig_neg"])],
                   ignore_index=True)
group_panels(signed, "variable", variables, "frac", AGG, figdir, "2b_frac_significant_by_sign",
             f"Significant cells split by the sign of r; dashed = {ALPHA / 2:.1%} per sign expected by chance; {agg_note}",
             ylabel="fraction of cells", hline=ALPHA / 2, series_col="sign", series_order=["r > 0", "r < 0"],
             series_colors={"r > 0": "#d62728", "r < 0": "#1f77b4"}, mouse_colors=mouse_colors,
             min_cells_per_unit=MIN_CELLS_PER_UNIT)

# 3. r vs r scatters (cell level): which cells follow several variables
wide = corr.pivot_table(index=["session_id", "cell_id", "group"], columns="variable",
                        values=["r", "significant"]).reset_index()
base_pairs = [("runspeed", "lick"), ("runspeed", "pupil_area"), ("lick", "pupil_area")]
pairs = [(a, b) for a, b in base_pairs] + [(f"{a}|position", f"{b}|position") for a, b in base_pairs]
pairs = [(a, b) for a, b in pairs if a in variables and b in variables]
for xv, yv in pairs:
    fig, axd = six_panel_figure(sharex=True, sharey=True)
    for g, ax in axd.items():
        sub = wide[wide[("group", "")] == group_key(g)]
        x, y = sub[("r", xv)], sub[("r", yv)]
        sx, sy = sub[("significant", xv)] > 0.5, sub[("significant", yv)] > 0.5
        ax.scatter(x[~(sx | sy)], y[~(sx | sy)], s=5, color="#bbbbbb")
        ax.scatter(x[sx & ~sy], y[sx & ~sy], s=6, color="#1f77b4")
        ax.scatter(x[sy & ~sx], y[sy & ~sx], s=6, color="#d62728")
        ax.scatter(x[sx & sy], y[sx & sy], s=8, color="#6a0dad")
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.text(0.03, 0.95, f"n={len(sub)}", transform=ax.transAxes, fontsize=7, va="top")
        ax.set_xlabel(f"r {xv}", fontsize=8)
        ax.set_ylabel(f"r {yv}", fontsize=8)
    fig.suptitle(f"grey = neither, blue = {xv} only, red = {yv} only, purple = both")
    fig.tight_layout()
    save_fig(fig, figdir, f"3_r_scatter_{xv.replace('|', '_given_')}_vs_{yv.replace('|', '_given_')}")
    plt.close(fig)

# 4. per-unit heatmaps of the fraction significant (pooled: a single row)
ukey = {"pooled": None, "session": "session_id", "mouse": "animal_id"}[AGG]
hm = corr.assign(_unit=corr[ukey] if ukey else "all")
fig, axes = plt.subplots(1, len(variables), figsize=(3.4 * len(variables), 0.35 * hm["_unit"].nunique() + 2),
                         squeeze=False)
for ax, var in zip(axes[0], variables):
    piv = (hm[hm["variable"] == var].groupby(["_unit", "group"])["sig_f"].mean().unstack("group")
           .reindex(columns=GROUP_KEYS))
    im = ax.imshow(piv.to_numpy(float), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(6))
    ax.set_xticklabels([SIX_GROUP_LABELS[g] for g in SIX_GROUP_ORDER], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels(piv.index if ax is axes[0, 0] else [], fontsize=6)
    ax.set_title(var, fontsize=9)
fig.colorbar(im, ax=axes[0].tolist(), label="fraction significant", shrink=0.8)
fig.suptitle(f"Fraction of significantly correlated cells per {UNIT}")
save_fig(fig, figdir, "4_per_unit_heatmap")
plt.close(fig)

# 5. class fractions per group (cf. Chen et al. 2013, Fig. 1h): one panel per class, at the chosen unit
if cls_df is not None:
    class_long = pd.concat([cls_df.assign(cls=c, frac=cls_df[f"is_{c}"]) for c in class_order], ignore_index=True)
    group_panels(class_long, "cls", class_order, "frac", AGG, figdir, "5_behavior_class_fractions",
                 f"Fraction of cells per behavior class ({'position-residual' if CLASS_SUFFIX else 'raw'} r); {agg_note}",
                 ylabel="fraction of cells", mouse_colors=mouse_colors, min_cells_per_unit=MIN_CELLS_PER_UNIT)

    # 6. lag distribution of the significant cells (cell level; activity follows behavior if lag > 0)
    sig_lags = corr[corr["significant"] & corr["variable"].isin(class_vars)]
    if len(sig_lags):
        fig, ax = plt.subplots(figsize=(5, 3.2))
        for v in class_vars:
            ax.hist(sig_lags.loc[sig_lags["variable"] == v, "lag_s"], bins=np.linspace(-0.05, LAGS_S[1] + 0.1, 12),
                    histtype="step", lw=1.5, label=v)
        ax.set_xlabel("best lag (s); positive = activity follows behavior")
        ax.set_ylabel("significant cells")
        ax.legend(fontsize=7, frameon=False)
        fig.tight_layout()
        save_fig(fig, figdir, "6_best_lag_hist")
        plt.close(fig)

print(f"\nWrote tables to {table_dir}, figures to {figdir}")
