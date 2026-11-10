# -*- coding: utf-8 -*-
"""
2j_pupil_activity_shape
========================
What is the SHAPE of the relationship between pupil area and neural activity (e.g. an inverted U), and is that
shape the same across NEURONS, SESSIONS and MICE?  DN sessions only (calcium + video).

Per cell, frame by frame on the imaging clock (engaged trials, frames within [-S_PRE, +S_POST) cm of stimStart):

  x = pupil area (blink outliers removed, z-scored within the session -- the project convention for face quantities),
      re-standardized over the analysed frames.
  y = the cell's deconvolved activity, z-scored per cell. Two versions (VARIANTS):
        raw        the activity itself
        |position  the activity minus its mean at each corridor position (CONTROL_POSITION_BIN_CM) relative to
                   stimStart, then re-standardized. Pupil is locked to the trial structure, so any cell with a
                   position-locked response "follows" pupil in the raw version; the residual version asks about
                   the cell's fluctuations around its position-locked response (same logic as 2g).
  LAG_S: activity is paired with the pupil LAG_S seconds earlier (0 = simultaneous; positive = activity follows).

A. TUNING CURVE   mean activity (SD units) in N_BINS pupil QUANTILE bins, bins defined within the session so every
                  session spans the same percentiles of ITS pupil range (x axis = percentile of the pupil range,
                  not absolute size, which differs between mice and sessions).
B. SHAPE TESTS    y ~ 1 + z + z^2 per cell (z = standardized pupil).
                    beta2    < 0 inverted U, > 0 U
                    slopes at the low / high end of the range (RANGE_PCT percentiles): inverted U needs low > 0 and
                    high < 0 (turning point inside the range); U the opposite
                    dR2      variance explained by the quadratic beyond the linear model
                    r_lin    plain linear correlation (an inverted U has r ~ 0)
                  Significance = CIRCULAR-SHIFT null (as in 2g): the pupil trace is rolled by a random lag
                  (>= MIN_SHIFT_S) N_SHIFTS times and the same statistics recomputed, so the autocorrelation of both
                  signals is kept (frames are not independent, a plain OLS p-value would be far too small).
                  Each cell is then assigned a SHAPE CLASS:
                    inverted_U / U     quadratic term significant AND the slopes change sign inside the range
                    increasing / decreasing   not a hump, but a significant (monotone or saturating) trend
                    flat               nothing significant
                  Model-free companion: `center_minus_edges` = mean of the middle bins minus mean of the outer bins of
                  the tuning curve (> 0 inverted-U-like, < 0 U-like), no fit involved.
C. RELIABILITY    the tuning curve of EVEN and ODD trials separately (same bin edges): per-cell split-half curve
                  correlation, and across cells the split-half correlation of center_minus_edges. If a cell's shape does
                  not even replicate in its own other half, nothing can be said about consistency between cells.
D. CONSISTENCY    -- across CELLS: fraction of cells per shape class, PCA of the cells' curves (PC1 variance share =
                  how much of the cell-to-cell variety is ONE shared shape, and the PC1/PC2 shapes themselves);
                  -- across SESSIONS and MICE: the mean curve of each session (mouse) with at least
                  MIN_CELLS_PER_UNIT cells in the group, leave-one-out correlation of each unit's curve with the average
                  of all OTHER units (shape agreement), how many units have the same sign of beta2 /
                  center_minus_edges, Wilcoxon of the unit means against 0 (>= MIN_UNITS units).
                  Note that the curve correlation also rewards a shared monotone component; the signed measures
                  (beta2, center_minus_edges) are the ones that speak to the hump specifically.
Always all the groups of the depth rule (the six + deep PM): shapes can differ between V1 / PM / deep PM and between
labeled and unlabeled cells.

Project-wide rules
    * selection : --animals / --sessions restrict the sessions; a restricted run writes to out/selection/<tag>/.
    * aggregate : --aggregate pooled|session|mouse = the unit of the mean-curve figure and of the group bar plots
                  (default session). The consistency tables always report cells, sessions AND mice.
    * memory    : each session is loaded, reduced to a small per-cell table, cached in store/ and freed.
    * cells     : 1b-included sessions; cells passing 2a (proximity, depth rule) and 2b QC.

Reads: 1b_psychometric psychometric_fits.csv, 2a celldata_combined.csv, 2b qc_per_cell.csv.
Loads: raw behaviordata/videodata/calciumdata, one session at a time.

Outputs
-------
2_pipeline/2j_pupil_activity_shape/
    out/    pupil_activity_shape_per_cell.csv     one row per (variant, session, cell): n_frames, beta1, beta2, slopes,
                    turning point, r_lin, dR2, p-values, shape class, center_minus_edges, split-half r, curve c0..cN
                    (bin means), half curves h1_*, h2_*, pupil bin centers zc*
            shape_class_by_group[__<aggregate>].csv   fraction of cells per class, at the chosen unit
            shape_consistency_by_group.csv            reliability + consistency across cells / sessions / mice
            shape_unit_curves.csv                     mean curve per group x session / mouse
            shape_pca.csv                             PC shapes and variance shares per group
            figures/*.png
    store/  shape_session_<hash>.pkl     per-session results (reused unless --recompute)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, spearmanr

from infotheory import project_config as cfg
from infotheory.pipeline import get_pipeline_paths
from infotheory.session import discover_sessions
from infotheory.neural_encoding import load_cell_inclusion
from infotheory.continuous import merge_behavior_video, remove_pupil_outliers, zscore_per_session
from infotheory.celldata_utils import SIX_GROUP_ORDER, SIX_GROUP_LABELS, SIX_GROUP_COLORS
from infotheory.groups import six_panel_figure, group_key
from infotheory.plotting import set_style, save_fig, clear_figures
from infotheory.scope import add_scope_args, scope_from_args, select_sessions, scoped_dirs, data_stamp, unit_noun
from infotheory.session_jobs import run_sessions_units
from infotheory.unit_plots import group_panels, mouse_palette
from infotheory import tensors as tn
from infotheory import cell_tables as ct

PROTOCOL = "DN"

# --- what is analysed ------------------------------------------------------------------------------------
S_PRE, S_POST = 80.0, 60.0            # position window around stimStart (cm); frames outside are not used
VARIANTS = ["raw", "|position"]
CONTROL_POSITION_BIN_CM = 5.0         # bin of the position control for "|position"
LAG_S = 0.0                           # activity follows pupil by this many seconds (>= 0)
N_BINS = 8                            # pupil quantile bins (within session)
RANGE_PCT = (5, 95)                   # percentiles of z defining the low / high end of the range
MIN_FRAMES = 400                      # analysed frames needed in a session
N_SHIFTS = 200
MIN_SHIFT_S = 20.0
ALPHA = 0.05
RANDOM_STATE = 0
CLASSES = ["inverted_U", "U", "increasing", "decreasing", "flat"]
CLASS_COLORS = {"inverted_U": "#d62728", "U": "#1f77b4", "increasing": "#ff9896", "decreasing": "#aec7e8",
                "flat": "#bbbbbb"}

# --- local -----------------------------------------------------------------------------------------------
MIN_CELLS_PER_UNIT = 5                # cells in a group needed for a session / mouse mean curve
MIN_UNITS = 4                         # sessions / mice needed for a Wilcoxon test
MIN_UNITS_LOO = 3                     # sessions / mice needed for the leave-one-out correlation
VARIANT_TAG = {"raw": "raw", "|position": "pos"}
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
cell_group = dict(zip(cell_inclusion["cell_id"], cell_inclusion["group"]))      # per-group cell / pair caps (--max-cells / --max-pairs)
cells_by_session = {sid: sorted(sub.loc[sub["include"], "cell_id"]) for sid, sub in cell_inclusion.groupby("session_id")
                    if sid in session_ids}
session_ids = [s for s in session_ids if cells_by_session.get(s)]
if not session_ids:
    raise SystemExit("No included cells in the selected sessions.")
print(f"{len(session_ids)} sessions, {sum(len(v) for v in cells_by_session.values())} included cells")
stamps = {s: data_stamp(*selected[s]) for s in session_ids}
qc_mtime = int((activity_out / "qc_per_cell.csv").stat().st_mtime)

CURVE = [f"c{i}" for i in range(N_BINS)]
HALF1 = [f"h1_{i}" for i in range(N_BINS)]
HALF2 = [f"h2_{i}" for i in range(N_BINS)]
ZC = [f"zc{i}" for i in range(N_BINS)]
MID = [i for i in range(N_BINS) if N_BINS // 4 <= i < N_BINS - N_BINS // 4]
EDGE = [i for i in range(N_BINS) if i not in MID]
XB = (np.arange(N_BINS) + 0.5) / N_BINS          # pupil percentile of each bin centre


# --------------------------------------------------------------------- #
# 2. Per-session computation (one session per worker, cached)
# --------------------------------------------------------------------- #
def residualize_by_position(X, rel, mask, bin_cm):
    """Subtract the per-position-bin mean (over `mask` frames) from X (T, N); frames outside `mask` become 0."""
    from scipy import sparse
    n_bins = int(np.ceil((S_PRE + S_POST) / bin_cm))
    b = np.clip(np.floor((np.nan_to_num(rel, nan=0.0) + S_PRE) / bin_cm).astype(int), 0, n_bins - 1)
    idx = np.where(mask)[0]
    M = sparse.csr_matrix((np.ones(len(idx)), (b[idx], idx)), shape=(n_bins, len(X)))
    cnt = np.asarray(M.sum(1)).ravel()
    with np.errstate(invalid="ignore", divide="ignore"):
        means = (M @ np.nan_to_num(X)) / cnt[:, None]
    return np.where(mask[:, None], X - np.nan_to_num(means[b]), 0.0)


def fit_all(z, w, Y, Y2):
    """Quadratic and linear OLS of every column of Y (T, N; zeros where w = 0) on the pupil z (T,) over the frames with
    weight w in {0, 1}. Returns beta1, beta2 (per SD / SD^2 of the pupil), r_lin, dR2 and the standardization (mu, sd)."""
    n = w.sum()
    mu = (w * z).sum() / n
    sd = np.sqrt((w * (z - mu) ** 2).sum() / n)
    zz = np.where(w > 0, (z - mu) / sd, 0.0)
    X = np.column_stack([np.ones_like(zz), zz, zz ** 2])
    Xw = X * w[:, None]
    XtY = Xw.T @ Y
    XtX = Xw.T @ X
    beta = np.linalg.solve(XtX, XtY)
    bl = np.linalg.solve(XtX[:2, :2], XtY[:2])
    Yw, Y2w = w @ Y, w @ Y2
    sst = Y2w - Yw ** 2 / n
    with np.errstate(invalid="ignore", divide="ignore"):
        r2q = 1 - (Y2w - (beta * XtY).sum(0)) / sst
        r2l = 1 - (Y2w - (bl * XtY[:2]).sum(0)) / sst
    return dict(beta1=beta[1], beta2=beta[2], r_lin=np.sign(bl[1]) * np.sqrt(np.clip(r2l, 0, None)),
                dR2=r2q - r2l, mu=mu, sd=sd)


def bin_means(Y, mask, b):
    return np.vstack([Y[mask & (b == i)].mean(0) if (mask & (b == i)).any() else np.full(Y.shape[1], np.nan)
                      for i in range(N_BINS)])


def process_session(ses, seed, new_cells, new_pairs, req_cells):
    sid = ses.session_id
    cell_ids = list(new_cells)          # only the cells not yet in the ledger (session_jobs.run_sessions_units)
    if (not cell_ids or ses.calciumdata is None or ses.ts_F is None or ses.trialdata is None or ses.videodata is None
            or ses.behaviordata is None or "engaged" not in ses.trialdata.columns):
        return None
    merged = merge_behavior_video(ses.behaviordata, ses.videodata)
    if "pupil_area" not in merged.columns:
        return None
    merged = zscore_per_session(remove_pupil_outliers(merged), ["pupil_area"])
    ts_F = ses.ts_F
    dt = float(np.median(np.diff(ts_F)))
    pup = tn.behavior_to_imaging_frames(merged["ts"], merged["pupil_area"], ts_F, mode="mean")
    A = ses.calciumdata.loc[:, cell_ids].to_numpy(dtype=float)
    lag = int(round(LAG_S / dt))
    if lag > 0:                                     # pair pupil at t with activity at t + lag
        A = np.vstack([A[lag:], np.full((lag, A.shape[1]), np.nan)])
    zpos_F = tn.interp_to_imaging_frames(ses.behaviordata["ts"], ses.behaviordata["zpos"], ts_F)
    stim = np.sort(ses.trialdata[ses.trialdata["engaged"] == 1]["stimStart"].to_numpy(float))
    k, rel = tn.position_relative_to_stim(zpos_F, stim, S_PRE, S_POST)
    ok = tn.engaged_frame_mask(ts_F, ses.trialdata) & np.isfinite(pup) & (k >= 0) & np.isfinite(A).all(1)
    if ok.sum() < MIN_FRAMES:
        return None
    A = np.nan_to_num(A)
    w = ok.astype(float)
    T = len(pup)
    mu_p, sd_p = pup[ok].mean(), pup[ok].std()
    zp = np.where(ok, (pup - mu_p) / sd_p, 0.0)      # re-standardized over the analysed frames
    edges = np.quantile(zp[ok], np.linspace(0, 1, N_BINS + 1))
    b = np.clip(np.searchsorted(edges[1:-1], zp, side="right"), 0, N_BINS - 1)
    zc = np.array([np.median(zp[ok & (b == i)]) if (ok & (b == i)).any() else np.nan for i in range(N_BINS)])
    z_lo, z_hi = np.percentile(zp[ok], RANGE_PCT)
    odd = (k % 2 == 1)
    rng = np.random.default_rng(seed)
    min_shift = int(round(MIN_SHIFT_S / dt))
    shifts = rng.integers(min_shift, max(min_shift + 1, T - min_shift), n_shifts)
    out = []
    for var in VARIANTS:
        Y = tn.zscore_frames(A, ok)
        if var == "|position":
            Y = tn.zscore_frames(residualize_by_position(np.nan_to_num(Y), rel, ok, CONTROL_POSITION_BIN_CM), ok)
        good = np.isfinite(Y[ok]).all(0)
        if not good.any():
            continue
        Y = np.where(ok[:, None], np.nan_to_num(Y[:, good]), 0.0)
        Y2 = Y ** 2
        obs = fit_all(zp, w, Y, Y2)
        nulls = {"r_lin": [], "beta2": [], "dR2": []}
        for s in shifts:
            zs, ws = np.roll(zp, s), w * np.roll(w, s)
            f = fit_all(zs, ws, Y * ws[:, None], Y2 * ws[:, None])
            for key in nulls:
                nulls[key].append(f[key])
        nulls = {kk: np.vstack(v) for kk, v in nulls.items()}
        p_lin = (1 + (np.abs(nulls["r_lin"]) >= np.abs(obs["r_lin"])).sum(0)) / (1 + n_shifts)
        p_quad = (1 + (np.abs(nulls["beta2"]) >= np.abs(obs["beta2"])).sum(0)) / (1 + n_shifts)
        p_dR2 = (1 + (nulls["dR2"] >= obs["dR2"]).sum(0)) / (1 + n_shifts)
        zlo_s, zhi_s = (z_lo - obs["mu"]) / obs["sd"], (z_hi - obs["mu"]) / obs["sd"]
        with np.errstate(invalid="ignore", divide="ignore"):
            vertex = obs["mu"] + obs["sd"] * (-obs["beta1"] / (2 * obs["beta2"]))
        curve = bin_means(Y, ok, b)
        h1 = bin_means(Y, ok & ~odd, b)
        h2 = bin_means(Y, ok & odd, b)
        df = pd.DataFrame({
            "session_id": sid, "cell_id": np.asarray(cell_ids)[good], "variant": var, "n_frames": int(ok.sum()),
            "beta1": obs["beta1"], "beta2": obs["beta2"], "slope_low": obs["beta1"] + 2 * obs["beta2"] * zlo_s,
            "slope_high": obs["beta1"] + 2 * obs["beta2"] * zhi_s, "turning_z": vertex, "r_lin": obs["r_lin"],
            "dR2": obs["dR2"], "p_lin": p_lin, "p_quad": p_quad, "p_dR2": p_dR2, "z_lo": z_lo, "z_hi": z_hi})
        for cols, arr in ((CURVE, curve), (HALF1, h1), (HALF2, h2)):
            df[cols] = arr.T
        df[ZC] = np.tile(zc, (len(df), 1))
        out.append(df)
    return {"df": pd.concat(out, ignore_index=True)} if out else None


results = run_sessions_units(
    session_ids, process_session, [PROTOCOL], paths.store, "shape_session",
    key_params=dict(version="2j-v2", variants=VARIANTS, ctrl=CONTROL_POSITION_BIN_CM, lag_s=LAG_S, n_bins=N_BINS,
                    range_pct=RANGE_PCT, min_frames=MIN_FRAMES, n_shifts=n_shifts, min_shift_s=MIN_SHIFT_S,
                    s_pre=S_PRE, s_post=S_POST, qc_mtime=qc_mtime, fits_mtime=int(fits_path.stat().st_mtime)),
    load_kwargs=dict(load_behaviordata=True, load_videodata=True, load_celldata=True, load_calciumdata=True),
    cells_by_session=cells_by_session, parts={"df": "cell"}, unwrap="df", groups_by_cell=cell_group, max_cells=scope.max_cells,
    base_seed=RANDOM_STATE, n_jobs=scope.n_jobs, recompute=args.recompute,
    verbose=JOBLIB_VERBOSE, extra_key_by_session=stamps)
results = [r for r in results if r is not None and len(r)]
if not results:
    raise SystemExit("No results (sessions without video / pupil, or too few analysable frames).")
cells = pd.concat(results, ignore_index=True)


# --------------------------------------------------------------------- #
# 3. Cell table: groups, shape class, model-free shape metrics, split-half reliability
# --------------------------------------------------------------------- #
cd = pd.read_csv(cell_dist_out / "celldata_combined.csv", low_memory=False)
cells = ct.attach_cell_meta(cells, cd)
GROUP_KEYS = [group_key(g) for g in SIX_GROUP_ORDER]
cells = cells[cells["group"].isin(GROUP_KEYS)].reset_index(drop=True)
if cells.empty:
    raise SystemExit("No cells in any group.")


def row_corr(a, b):
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(1) / np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1))


sig_q, sig_l = cells["p_quad"] < ALPHA, cells["p_lin"] < ALPHA
inv_u = sig_q & (cells["beta2"] < 0) & (cells["slope_low"] > 0) & (cells["slope_high"] < 0)
u_sh = sig_q & (cells["beta2"] > 0) & (cells["slope_low"] < 0) & (cells["slope_high"] > 0)
mono = ~(inv_u | u_sh) & (sig_q | sig_l)
cells["cls"] = np.select([inv_u, u_sh, mono & (cells["beta1"] > 0), mono & (cells["beta1"] <= 0)],
                         ["inverted_U", "U", "increasing", "decreasing"], default="flat")
C = cells[CURVE].to_numpy(float)
cells["center_minus_edges"] = np.nanmean(C[:, MID], 1) - np.nanmean(C[:, EDGE], 1)
cells["peak_bin"] = np.where(np.isfinite(C).all(1), np.nanargmax(np.where(np.isfinite(C), C, -np.inf), axis=1), -1)
H1, H2 = cells[HALF1].to_numpy(float), cells[HALF2].to_numpy(float)
cells["center_minus_edges_h1"] = np.nanmean(H1[:, MID], 1) - np.nanmean(H1[:, EDGE], 1)
cells["center_minus_edges_h2"] = np.nanmean(H2[:, MID], 1) - np.nanmean(H2[:, EDGE], 1)
cells["split_half_curve_r"] = row_corr(H1, H2)
for c in CLASSES:
    cells[f"is_{c}"] = (cells["cls"] == c).astype(float)
cells.to_csv(table_dir / "pupil_activity_shape_per_cell.csv", index=False)
print(f"{cells['cell_id'].nunique()} cells x {len(VARIANTS)} variants, {cells['session_id'].nunique()} sessions")

agg_note = {"pooled": "all cells pooled (no error bars)", "session": "bar = mean +- SEM over sessions, dots = sessions",
            "mouse": "bar = mean +- SEM over mice, dots = mice"}[AGG]
t = ct.unit_group_summary(cells, ["variant"], AGG, [f"is_{c}" for c in CLASSES])
t.to_csv(table_dir / f"shape_class_by_group{AGG_SUFFIX}.csv", index=False)
print(f"\nShape classes, mean fraction of cells per group ({'pooled' if AGG == 'pooled' else f'mean over {UNITS}'}); "
      f"chance for each of inverted_U / U is ~{ALPHA / 2:.1%} x P(slopes change sign):")
for var in VARIANTS:
    print(f"  variant {var}")
    print(t[t["variant"] == var].groupby("group")[[f"is_{c}" for c in CLASSES]].mean().reindex(GROUP_KEYS).round(2)
          .rename(columns=lambda c: c[3:]).to_string())


# --------------------------------------------------------------------- #
# 4. Consistency across cells / sessions / mice
# --------------------------------------------------------------------- #
def unit_curve_table(df, level):
    key = {"session": "session_id", "mouse": "animal_id"}[level]
    g = df.groupby(key)
    mean, size = g[CURVE].mean(), g.size()
    return mean[size >= MIN_CELLS_PER_UNIT], g[["beta2", "center_minus_edges"]].mean().loc[mean.index[size >= MIN_CELLS_PER_UNIT]], \
        size[size >= MIN_CELLS_PER_UNIT]


def loo_r(M):
    out = []
    for i in range(len(M)):
        oth = np.delete(M, i, 0).mean(0)
        out.append(np.corrcoef(M[i], oth)[0, 1] if M[i].std() > 0 and oth.std() > 0 else np.nan)
    return np.array(out)


def safe_wilcoxon(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < MIN_UNITS or np.allclose(v, 0):
        return np.nan
    return wilcoxon(v).pvalue


cons_rows, unit_rows, pca_rows = [], [], []
for var in VARIANTS:
    for gk in GROUP_KEYS:
        sub = cells[(cells["variant"] == var) & (cells["group"] == gk)]
        row = dict(variant=var, group=gk, n_cells=len(sub), n_sessions=sub["session_id"].nunique(),
                   n_mice=sub["animal_id"].nunique())
        if len(sub):
            for c in CLASSES:
                row[f"frac_{c}"] = float((sub["cls"] == c).mean())
            row["median_split_half_curve_r"] = float(np.nanmedian(sub["split_half_curve_r"]))
            row["frac_split_half_r_positive"] = float(np.nanmean(sub["split_half_curve_r"] > 0))
            ok_h = sub[["center_minus_edges_h1", "center_minus_edges_h2"]].dropna()
            row["split_half_rho_center_minus_edges"] = (spearmanr(ok_h.iloc[:, 0], ok_h.iloc[:, 1])[0]
                                                        if len(ok_h) >= 5 else np.nan)
            row["mean_beta2"] = float(sub["beta2"].mean())
            row["mean_center_minus_edges"] = float(sub["center_minus_edges"].mean())
            Cs = sub[CURVE].to_numpy(float)
            Cs = Cs[np.isfinite(Cs).all(1)]
            if len(Cs) >= 5:                             # PCA of the cells' curves (no centering across cells)
                u, s, vt = np.linalg.svd(Cs, full_matrices=False)
                share = s ** 2 / (s ** 2).sum()
                row["pc1_var_share"], row["pc2_var_share"] = float(share[0]), float(share[1])
                mean_shape = Cs.mean(0)
                for pc in range(2):
                    v = vt[pc] * (1.0 if pc > 0 or np.dot(vt[pc], mean_shape) >= 0 else -1.0)
                    pca_rows.append(dict(variant=var, group=gk, pc=pc + 1, var_share=float(share[pc]),
                                         **{f"b{i}": v[i] for i in range(N_BINS)}))
            for level in ("session", "mouse"):
                M, stat, size = unit_curve_table(sub, level)
                for unit, curve in M.iterrows():
                    unit_rows.append(dict(variant=var, group=gk, level=level, unit=unit, n_cells=int(size[unit]),
                                          **{f"b{i}": curve.iloc[i] for i in range(N_BINS)}))
                row[f"n_{level}_units"] = len(M)
                if len(M) >= MIN_UNITS_LOO:
                    loo = loo_r(M.to_numpy(float))
                    row[f"{level}_loo_r_median"] = float(np.nanmedian(loo))
                    row[f"{level}_loo_r_positive"] = int(np.nansum(loo > 0))
                if len(M) >= MIN_UNITS:
                    row[f"{level}_beta2_same_sign_as_mean"] = int((np.sign(stat["beta2"]) == np.sign(stat["beta2"].mean())).sum())
                    row[f"{level}_cme_same_sign_as_mean"] = int((np.sign(stat["center_minus_edges"])
                                                                 == np.sign(stat["center_minus_edges"].mean())).sum())
                    row[f"{level}_p_beta2_wilcoxon"] = safe_wilcoxon(stat["beta2"])
                    row[f"{level}_p_cme_wilcoxon"] = safe_wilcoxon(stat["center_minus_edges"])
        cons_rows.append(row)
cons = pd.DataFrame(cons_rows)
cons.to_csv(table_dir / "shape_consistency_by_group.csv", index=False)
unit_curves = pd.DataFrame(unit_rows)
unit_curves.to_csv(table_dir / "shape_unit_curves.csv", index=False)
pca_df = pd.DataFrame(pca_rows)
pca_df.to_csv(table_dir / "shape_pca.csv", index=False)
show = ["group", "n_cells", "n_sessions", "n_mice", "median_split_half_curve_r", "pc1_var_share", "mean_beta2",
        "session_loo_r_median", "mouse_loo_r_median", "mouse_p_beta2_wilcoxon"]
for var in VARIANTS:
    print(f"\nConsistency, variant {var}:")
    print(cons[cons["variant"] == var].set_index("group")[[c for c in show[1:] if c in cons.columns]].round(3).to_string())


# --------------------------------------------------------------------- #
# 5. Figures
# --------------------------------------------------------------------- #
def unit_mean_curves(sub, aggregate):
    """(unit label, animal, curve) per analysis unit at the chosen aggregate; pooled: one entry per cell."""
    if aggregate == "pooled":
        return [("cell", None, sub[CURVE].to_numpy(float))]
    key = "session_id" if aggregate == "session" else "animal_id"
    g = sub.groupby(key)
    return [(u, d["animal_id"].iloc[0], d[CURVE].mean().to_numpy(float)) for u, d in g if len(d) >= MIN_CELLS_PER_UNIT]


for var in VARIANTS:
    tag = VARIANT_TAG[var]
    vc = cells[cells["variant"] == var]

    # 1. mean tuning curves per group at the chosen unit (shared y)
    fig, axd = six_panel_figure(figsize_per_panel=(3.3, 3.0), sharex=True, sharey=True)
    for g, ax in axd.items():
        sub = vc[vc["group"] == group_key(g)]
        ax.axhline(0, color="k", lw=0.5)
        if sub.empty:
            ax.text(0.5, 0.5, "no cells", transform=ax.transAxes, ha="center", color="grey")
            continue
        ents = unit_mean_curves(sub, AGG)
        if AGG == "pooled":
            M = ents[0][2]
            mean = np.nanmean(M, 0)
            sem = np.nanstd(M, 0) / np.sqrt(np.isfinite(M).sum(0))
        else:
            for _, animal, curve in ents:
                ax.plot(XB, curve, color=mouse_colors.get(animal, "grey"), lw=0.9, alpha=0.7)
            M = np.vstack([e[2] for e in ents]) if ents else np.full((1, N_BINS), np.nan)
            mean = np.nanmean(M, 0)
            sem = np.nanstd(M, 0) / np.sqrt(max(len(ents), 1))
        ax.plot(XB, mean, color=SIX_GROUP_COLORS[g], lw=2.4)
        ax.fill_between(XB, mean - sem, mean + sem, color=SIX_GROUP_COLORS[g], alpha=0.25)
        fr = {c: (sub["cls"] == c).mean() for c in ("inverted_U", "U")}
        ax.text(0.03, 0.97, f"n={len(sub)} cells\ninv-U {fr['inverted_U']:.0%} | U {fr['U']:.0%}",
                transform=ax.transAxes, fontsize=7, va="top")
        ax.set_xlabel("pupil percentile (within session)", fontsize=8)
        ax.set_ylabel("activity (SD)", fontsize=8)
    fig.suptitle(f"Activity vs pupil ({'position-residual' if var == '|position' else 'raw'} activity, lag {LAG_S:g} s); "
                 f"thick = mean over {'cells' if AGG == 'pooled' else UNITS} +- SEM, thin = one {UNIT}", y=1.0)
    fig.tight_layout()
    save_fig(fig, figdir, f"1_mean_curves_{tag}")
    plt.close(fig)

    # 1b. per-mouse mean curves (always at the mouse level)
    fig, axd = six_panel_figure(figsize_per_panel=(3.3, 3.0), sharex=True, sharey=True)
    for g, ax in axd.items():
        ax.axhline(0, color="k", lw=0.5)
        sub = unit_curves[(unit_curves["variant"] == var) & (unit_curves["group"] == group_key(g))
                          & (unit_curves["level"] == "mouse")]
        for _, r in sub.iterrows():
            ax.plot(XB, [r[f"b{i}"] for i in range(N_BINS)], color=mouse_colors.get(r["unit"], "grey"), lw=1.4,
                    label=r["unit"])
        ax.set_xlabel("pupil percentile (within session)", fontsize=8)
        ax.set_ylabel("activity (SD)", fontsize=8)
        if len(sub):
            ax.text(0.03, 0.97, f"{len(sub)} mice", transform=ax.transAxes, fontsize=7, va="top")
    fig.suptitle("One line per mouse (cells of a mouse pooled; mice with >= %d cells in the group)" % MIN_CELLS_PER_UNIT, y=1.0)
    fig.tight_layout()
    save_fig(fig, figdir, f"1b_mouse_curves_{tag}")
    plt.close(fig)

    # 2. heatmap of every cell's curve, cells sorted by their peak bin
    fig, axd = six_panel_figure(figsize_per_panel=(3.3, 3.2), sharex=True, sharey=False)
    allC = vc[CURVE].to_numpy(float)
    vmax = float(np.nanpercentile(np.abs(allC), 99)) or 1.0       # one symmetric colour scale for all groups
    for g, ax in axd.items():
        sub = vc[vc["group"] == group_key(g)]
        Cg = sub[CURVE].to_numpy(float)
        Cg = Cg[np.isfinite(Cg).all(1)]
        if not len(Cg):
            ax.text(0.5, 0.5, "no cells", transform=ax.transAxes, ha="center", color="grey")
            continue
        order = np.lexsort((-Cg.max(1), Cg.argmax(1)))
        im = ax.imshow(Cg[order], aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       extent=[0, 1, len(Cg), 0], interpolation="nearest")
        ax.set_xlabel("pupil percentile", fontsize=8)
        ax.set_ylabel("cells (sorted by peak bin)", fontsize=8)
    fig.suptitle("Every cell's tuning curve to pupil (rows = cells)", y=1.0)
    fig.tight_layout(rect=[0, 0, 0.92, 0.97])
    cax = fig.add_axes([0.94, 0.25, 0.012, 0.5])
    fig.colorbar(im, cax=cax, label="activity (SD)")
    save_fig(fig, figdir, f"2_curve_heatmaps_{tag}")
    plt.close(fig)

    # 3. shape class fractions at the chosen unit
    class_long = pd.concat([vc.assign(cls_name=c, frac=vc[f"is_{c}"]) for c in CLASSES], ignore_index=True)
    group_panels(class_long, "cls_name", CLASSES, "frac", AGG, figdir, f"3_shape_class_fractions_{tag}",
                 f"Fraction of cells per shape class (circular-shift null, alpha {ALPHA}); {agg_note}",
                 ylabel="fraction of cells", mouse_colors=mouse_colors, min_cells_per_unit=MIN_CELLS_PER_UNIT)

    # 4. shape coefficients per group at the chosen unit
    metrics = ["beta2", "center_minus_edges", "r_lin", "dR2"]
    mlabel = {"beta2": "quadratic coefficient beta2 (<0 inverted U)", "center_minus_edges": "centre minus edges (>0 inverted-U-like)",
              "r_lin": "linear r", "dR2": "R2 gain of quadratic over linear"}
    long = pd.concat([vc.assign(metric=mlabel[m], val=vc[m]) for m in metrics], ignore_index=True)
    group_panels(long, "metric", [mlabel[m] for m in metrics], "val", AGG, figdir, f"4_shape_coefficients_{tag}",
                 f"Shape coefficients per group; {agg_note}", ylabel="value", hline=0, ncols=2,
                 mouse_colors=mouse_colors, min_cells_per_unit=MIN_CELLS_PER_UNIT)

    # 5. PCA shapes
    fig, axd = six_panel_figure(figsize_per_panel=(3.3, 3.0), sharex=True, sharey=True)
    for g, ax in axd.items():
        sub = pca_df[(pca_df["variant"] == var) & (pca_df["group"] == group_key(g))]
        ax.axhline(0, color="k", lw=0.5)
        for _, r in sub.iterrows():
            ax.plot(XB, [r[f"b{i}"] for i in range(N_BINS)], lw=2.0 if r["pc"] == 1 else 1.4,
                    ls="-" if r["pc"] == 1 else "--", color=SIX_GROUP_COLORS[g],
                    label=f"PC{int(r['pc'])} ({r['var_share']:.0%})")
        if len(sub):
            ax.legend(fontsize=7, frameon=False, loc="upper left")
        ax.set_xlabel("pupil percentile", fontsize=8)
        ax.set_ylabel("PC loading", fontsize=8)
    fig.suptitle("Main shapes across cells (PCA of the tuning curves; PC1 = shared shape, sign set to the mean curve)", y=1.0)
    fig.tight_layout()
    save_fig(fig, figdir, f"5_pca_shapes_{tag}")
    plt.close(fig)

    # 6. reliability and consistency
    fig, axes = plt.subplots(1, 2, figsize=(5.2 + 0.55 * len(GROUP_KEYS) * 1.6, 3.8))
    rng_j = np.random.default_rng(0)
    ax = axes[0]
    data = [vc.loc[vc["group"] == gk, "split_half_curve_r"].dropna().to_numpy() for gk in GROUP_KEYS]
    bp = ax.boxplot(data, positions=range(len(GROUP_KEYS)), widths=0.6, showfliers=False, patch_artist=True)
    for patch, g in zip(bp["boxes"], SIX_GROUP_ORDER):
        patch.set_facecolor(SIX_GROUP_COLORS[g])
        patch.set_alpha(0.6)
    ax.axhline(0, color="grey", ls="--", lw=1)
    ax.set_ylabel("even-vs-odd trials: curve correlation (per cell)")
    ax.set_title("Within-cell reliability of the tuning curve", fontsize=9)
    ax = axes[1]
    for xi, gk in enumerate(GROUP_KEYS):
        for level, size, marker in (("session", 14, "o"), ("mouse", 55, "D")):
            M, _, _ = unit_curve_table(cells[(cells["variant"] == var) & (cells["group"] == gk)], level) \
                if ((cells["variant"] == var) & (cells["group"] == gk)).any() else (pd.DataFrame(), None, None)
            if len(M) >= MIN_UNITS_LOO:
                loo = loo_r(M.to_numpy(float))
                animals = [u if level == "mouse" else str(u).split("_")[0] for u in M.index]
                xs = xi + (-0.18 if level == "session" else 0.22) + rng_j.uniform(-0.07, 0.07, len(loo))
                ax.scatter(xs, loo, s=size, marker=marker, c=[mouse_colors.get(a, "grey") for a in animals],
                           edgecolor="k", lw=0.4, zorder=3)
    ax.axhline(0, color="grey", ls="--", lw=1)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("leave-one-out curve correlation\n(unit vs mean of the other units)")
    ax.set_title("Across units: circles = sessions, diamonds = mice", fontsize=9)
    for ax in axes:
        ax.set_xticks(range(len(GROUP_KEYS)))
        ax.set_xticklabels([SIX_GROUP_LABELS[g] for g in SIX_GROUP_ORDER], rotation=35, ha="right", fontsize=7)
    fig.tight_layout()
    save_fig(fig, figdir, f"6_reliability_consistency_{tag}")
    plt.close(fig)

print(f"\nWrote tables to {table_dir}, figures to {figdir}")
