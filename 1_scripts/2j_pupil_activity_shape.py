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
B. SHAPE TESTS    y ~ 1 + z + z^2 per cell (z = standardized pupil), CROSS-VALIDATED: the quadratic (and nested
                  linear) model is fit on ODD trials and scored on EVEN trials and vice versa (fit_train /
                  score_test / fit_cv), SSE and SST pooled over both directions before computing R^2. beta1/beta2
                  are the MEAN of the two half-fits. This replaces fitting and scoring on the SAME frames (the old
                  design): an in-sample quadratic R^2 can only ever be >= the linear R^2 (a nested model), so dR2
                  was mechanically non-negative and partly just "the quadratic absorbing noise with its extra
                  parameter" -- here dR2 (and r2q/r2l individually, not written out but implicit in dR2/r_lin) CAN
                  come out negative when the shape does not generalize, which is the point.
                    beta2    < 0 inverted U, > 0 U (mean of the odd-fit and even-fit beta2)
                    slopes at the low / high end of the range (RANGE_PCT percentiles): inverted U needs low > 0 and
                    high < 0 (turning point inside the range); U the opposite
                    dR2      CROSS-VALIDATED variance explained by the quadratic beyond the linear model (held-out
                             R^2 of the quadratic minus held-out R^2 of the linear; can be negative)
                    r_lin    plain linear correlation, from the held-out R^2 of the linear-only model (an inverted
                             U has r ~ 0)
                  Significance = CIRCULAR-SHIFT null (as in 2g): the pupil trace is rolled by a random lag
                  (>= MIN_SHIFT_S) N_SHIFTS times and the SAME cross-validated fit_cv recomputed at every shift (not
                  just the in-sample fit), so the autocorrelation of both signals is kept (frames are not
                  independent, a plain OLS p-value would be far too small) AND the null is of the honest, held-out
                  statistic, not the inflated in-sample one.
                  SUPPLEMENTARY 5-FOLD CROSS-CHECK: every beta1/beta2/r_lin/dR2/p_* above also has a `_5fold`
                  twin, from the SAME fit_cv generalized to K=5 folds (fold = trial index mod 5): train on 4/5 of
                  the trials, score the held-out 1/5, pooled SSE/SST over the 5 held-out scorings, coefficients
                  averaged over the 5 training fits. The 2-fold (odd/even) numbers stay the headline (unchanged,
                  used by every figure and the shape classification); `_5fold` is there to confirm the 2-fold
                  numbers are not an artifact of that particular split -- see figure 1c.
                  FIT-ON-ONE-HALF, SHOW-THE-OTHER-HALF'S-POINTS (figure 1c): NOT the quadratic model (that is what
                  beta_quadratic/dR2 above already test) -- h2/h1 (per cell, per bin; odd trials' own bin means,
                  no shape assumed) is plotted as a line against h1/h2 (the OTHER half's own, independently computed
                  bin means) as points, so the curve shown is always being compared against data it never saw.
                  Each cell is then assigned a SHAPE CLASS:
                    inverted_U / U     quadratic term significant AND the slopes change sign inside the range
                    increasing / decreasing   not a hump, but a significant (monotone or saturating) trend
                    flat               nothing significant
                  Model-free companion: `center_minus_edges` = mean of the middle bins minus mean of the outer bins of
                  the tuning curve (> 0 inverted-U-like, < 0 U-like), no fit involved. Not cross-validated: MID/EDGE
                  are FIXED bin positions (not picked from the data, unlike e.g. 2k's `modulation`'s data-dependent
                  peak/trough bin), so there is no selection bias to correct here -- its own reliability is the
                  split-half check in section C below.
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
                    (bin means), half curves h1_*/h2_* (even/odd actual bin means; also the figure-1c "fit", idea 1 --
                    no shape assumed), the _5fold twins of beta1/beta2/r_lin/dR2/p_* (idea 2/3, supplementary
                    cross-check of the quadratic model), pupil bin centers zc*
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
HALF1 = [f"h1_{i}" for i in range(N_BINS)]        # actual bin means of the EVEN trials -- also the "fit" shown for idea 1
                                                    # (no shape assumed: the curve IS the other half's own bin means)
HALF2 = [f"h2_{i}" for i in range(N_BINS)]        # actual bin means of the ODD trials -- ditto
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


def fit_train(z, w, Y):
    """Quadratic (and nested linear) OLS of every column of Y (T, N) on the pupil z (T,), TRAIN frames only
    (weight w in {0, 1}). Returns beta (3, N), bl (2, N) and the standardization (mu, sd) -- all fit on these
    same TRAIN frames only, so a caller scoring held-out frames never leaks test information into mu/sd/beta."""
    n = w.sum()
    mu = (w * z).sum() / n
    sd = np.sqrt((w * (z - mu) ** 2).sum() / n)
    zz = np.where(w > 0, (z - mu) / sd, 0.0)
    X = np.column_stack([np.ones_like(zz), zz, zz ** 2])
    Xw = X * w[:, None]
    XtX = Xw.T @ X
    beta = np.linalg.solve(XtX, Xw.T @ Y)
    bl = np.linalg.solve(XtX[:2, :2], Xw.T[:2] @ Y)
    return beta, bl, mu, sd


def score_test(beta, bl, mu, sd, z, w, Y, Y2):
    """Held-out SSE (quadratic, linear model) and SST of Y on TEST frames (weight w), scoring a model TRAINED
    elsewhere (beta/bl/mu/sd come from fit_train on a DIFFERENT, disjoint set of frames)."""
    n = w.sum()
    zz = np.where(w > 0, (z - mu) / sd, 0.0)
    X = np.column_stack([np.ones_like(zz), zz, zz ** 2])
    grand = (w @ Y) / n
    sse_q = (w[:, None] * (Y - X @ beta) ** 2).sum(0)
    sse_l = (w[:, None] * (Y - X[:, :2] @ bl) ** 2).sum(0)
    sst = (w @ Y2) - 2 * grand * (w @ Y) + n * grand ** 2
    return sse_q, sse_l, sst


def fit_cv(z, w, fold, Y, Y2):
    """GENERAL K-FOLD cross-validated quadratic-vs-pupil fit (`fold`: integer array of fold ids 0..K-1 per frame).
    For every fold f: fit on every OTHER fold (fit_train), score the held-out fold f (score_test); SSE/SST are
    POOLED over all K held-out scorings before computing R^2 -- never an average of K separate ratios (project
    convention). This is the held-out counterpart of the old fit_all, which fit and scored on the SAME frames and
    so could only ever make the quadratic look AS GOOD OR BETTER than the linear model (R^2 of a nested model
    never decreases in-sample). Here dR2 (and r2q, r2l individually) CAN come out negative.
    beta1, beta2 = the MEAN of the K folds' own TRAINING fits (idea 3: "average curve across the different fits of
    the different subsets") -- never a coefficient fit on the same frames it is scored against. mu, sd = mean of
    the K folds' own standardization.
    K=2 with fold = odd trials (0/1) reproduces the original odd/even design exactly; K=5 (fold = trial index % 5)
    is the standard 5-fold CV (idea 2: "divide the data in 5 parts, train on 4/5, measure the held-out loss, mean
    across the 5"), used here as a second, independent cross-validated estimate of the same quantities.
    (Idea 1's "fit on one half, show the OTHER half's points" figure does NOT use this quadratic fit -- it just
    shows each half's own bin means (HALF1/HALF2 below) against the other, with no shape assumed.)"""
    K = int(fold.max()) + 1
    sse_q = sse_l = sst = 0.0
    betas1, betas2, bl1s, mus, sds = [], [], [], [], []
    for f in range(K):
        tr_w, te_w = w * (fold != f), w * (fold == f)
        beta, bl, mu, sd = fit_train(z, tr_w, Y)
        sq, sl, ss = score_test(beta, bl, mu, sd, z, te_w, Y, Y2)
        sse_q, sse_l, sst = sse_q + sq, sse_l + sl, sst + ss
        betas1.append(beta[1]); betas2.append(beta[2]); bl1s.append(bl[1]); mus.append(mu); sds.append(sd)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2q = 1 - sse_q / sst
        r2l = 1 - sse_l / sst
    beta1, beta2, bl1 = np.mean(betas1, axis=0), np.mean(betas2, axis=0), np.mean(bl1s, axis=0)
    return dict(beta1=beta1, beta2=beta2, r_lin=np.sign(bl1) * np.sqrt(np.clip(r2l, 0, None)), dR2=r2q - r2l,
                mu=float(np.mean(mus)), sd=float(np.mean(sds)), K=K)


def bin_means(Y, mask, b):
    return np.vstack([Y[mask & (b == i)].mean(0) if (mask & (b == i)).any() else np.full(Y.shape[1], np.nan)
                      for i in range(N_BINS)])


def process_session(ses, seed, new_cells, new_pairs, req_cells):
    sid = ses.session_id
    cell_ids = list(new_cells)          # the requested cells (session_jobs.run_sessions_units)
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
    fold5 = np.mod(k, 5)                  # 5-fold split (idea 2), same trial-index logic as the odd/even 2-fold split
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
        obs = fit_cv(zp, w, odd, Y, Y2)                              # 2-fold (odd/even): headline, unchanged
        obs5 = fit_cv(zp, w, fold5, Y, Y2)                          # 5-fold: supplementary cross-check (idea 2/3)
        nulls = {"r_lin": [], "beta2": [], "dR2": []}
        nulls5 = {"r_lin": [], "beta2": [], "dR2": []}
        for s in shifts:
            zs, ws = np.roll(zp, s), w * np.roll(w, s)
            f = fit_cv(zs, ws, odd, Y, Y2)
            f5 = fit_cv(zs, ws, fold5, Y, Y2)
            for key in nulls:
                nulls[key].append(f[key])
                nulls5[key].append(f5[key])
        nulls = {kk: np.vstack(v) for kk, v in nulls.items()}
        nulls5 = {kk: np.vstack(v) for kk, v in nulls5.items()}
        p_lin = (1 + (np.abs(nulls["r_lin"]) >= np.abs(obs["r_lin"])).sum(0)) / (1 + n_shifts)
        p_quad = (1 + (np.abs(nulls["beta2"]) >= np.abs(obs["beta2"])).sum(0)) / (1 + n_shifts)
        p_dR2 = (1 + (nulls["dR2"] >= obs["dR2"]).sum(0)) / (1 + n_shifts)
        p_lin5 = (1 + (np.abs(nulls5["r_lin"]) >= np.abs(obs5["r_lin"])).sum(0)) / (1 + n_shifts)
        p_quad5 = (1 + (np.abs(nulls5["beta2"]) >= np.abs(obs5["beta2"])).sum(0)) / (1 + n_shifts)
        p_dR25 = (1 + (nulls5["dR2"] >= obs5["dR2"]).sum(0)) / (1 + n_shifts)
        zlo_s, zhi_s = (z_lo - obs["mu"]) / obs["sd"], (z_hi - obs["mu"]) / obs["sd"]
        with np.errstate(invalid="ignore", divide="ignore"):
            vertex = obs["mu"] + obs["sd"] * (-obs["beta1"] / (2 * obs["beta2"]))
        curve = bin_means(Y, ok, b)
        h1 = bin_means(Y, ok & ~odd, b)    # actual EVEN-trial bin means
        h2 = bin_means(Y, ok & odd, b)     # actual ODD-trial bin means
        df = pd.DataFrame({
            "session_id": sid, "cell_id": np.asarray(cell_ids)[good], "variant": var, "n_frames": int(ok.sum()),
            "beta1": obs["beta1"], "beta2": obs["beta2"], "slope_low": obs["beta1"] + 2 * obs["beta2"] * zlo_s,
            "slope_high": obs["beta1"] + 2 * obs["beta2"] * zhi_s, "turning_z": vertex, "r_lin": obs["r_lin"],
            "dR2": obs["dR2"], "p_lin": p_lin, "p_quad": p_quad, "p_dR2": p_dR2, "z_lo": z_lo, "z_hi": z_hi,
            "beta1_5fold": obs5["beta1"], "beta2_5fold": obs5["beta2"], "r_lin_5fold": obs5["r_lin"],
            "dR2_5fold": obs5["dR2"], "p_lin_5fold": p_lin5, "p_quad_5fold": p_quad5, "p_dR2_5fold": p_dR25})
        for cols, arr in ((CURVE, curve), (HALF1, h1), (HALF2, h2)):
            df[cols] = arr.T
        df[ZC] = np.tile(zc, (len(df), 1))
        out.append(df)
    return {"df": pd.concat(out, ignore_index=True)} if out else None


results = run_sessions_units(
    session_ids, process_session, [PROTOCOL], paths.store, "shape_session",
    key_params=dict(version="2j-v5", variants=VARIANTS, ctrl=CONTROL_POSITION_BIN_CM, lag_s=LAG_S, n_bins=N_BINS,
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

    # 1c. idea 1: one half's own bin-mean curve (no shape assumed -- not the quadratic model, which is tested
    #     separately via beta_quadratic/dR2) drawn against the OTHER half's own actual bin means: a "fit" that has
    #     never seen the points it is laid over. HALF2/HALF1 (odd curve vs even points) and HALF1/HALF2 (even curve
    #     vs odd points); both directions shown so neither half is privileged. Also reports the 5-fold dR2 next to
    #     the 2-fold (odd/even) dR2 as a convergence check (idea 2) -- that statistic IS quadratic-based, unlike
    #     this figure's curves.
    fig, axd = six_panel_figure(figsize_per_panel=(3.3, 3.0), sharex=True, sharey=True)
    for g, ax in axd.items():
        sub = vc[vc["group"] == group_key(g)]
        ax.axhline(0, color="k", lw=0.5)
        if sub.empty:
            ax.text(0.5, 0.5, "no cells", transform=ax.transAxes, ha="center", color="grey")
            continue
        for fit_cols, pt_cols, fit_lab, pt_lab, color in (
                (HALF2, HALF1, "odd trials' own curve", "even points", "#d62728"),
                (HALF1, HALF2, "even trials' own curve", "odd points", "#1f77b4")):
            fit_m = np.nanmean(sub[fit_cols].to_numpy(float), 0)
            pt_m = np.nanmean(sub[pt_cols].to_numpy(float), 0)
            pt_sem = (np.nanstd(sub[pt_cols].to_numpy(float), 0)
                      / np.sqrt(np.isfinite(sub[pt_cols].to_numpy(float)).sum(0)))
            ax.plot(XB, fit_m, color=color, lw=2.0, label=fit_lab)
            ax.errorbar(XB, pt_m, yerr=pt_sem, color=color, marker="o", markersize=4, lw=0, elinewidth=1.1,
                        capsize=2, label=pt_lab)
        r2_2f, r2_5f = sub["dR2"].mean(), sub["dR2_5fold"].mean()
        ax.text(0.03, 0.97, f"n={len(sub)} cells\ndR2 2-fold {r2_2f:+.3f} | 5-fold {r2_5f:+.3f}",
                transform=ax.transAxes, fontsize=7, va="top")
        ax.set_xlabel("pupil percentile (within session)", fontsize=8)
        ax.set_ylabel("activity (SD)", fontsize=8)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.suptitle("Each half's own tuning curve (no shape assumed), shown against the OTHER half's own points "
                 "(mean over cells); dR2 (quadratic model): 2-fold (odd/even, headline) vs 5-fold (supplementary) "
                 "agreement", y=1.0, fontsize=9.5)
    fig.tight_layout()
    save_fig(fig, figdir, f"1c_cv_fit_vs_heldout_points_{tag}")
    plt.close(fig)

    # 2. heatmap of every cell's curve -- idea 4: cells SORTED by their peak bin on the ODD-trial curve (h2),
    #    values DISPLAYED are the EVEN-trial curve (h1): the sort order and the colors shown never come from the
    #    same trials, so a heatmap that "looks structured" is not sorting noise to look structured.
    fig, axd = six_panel_figure(figsize_per_panel=(3.3, 3.2), sharex=True, sharey=False)
    allH = vc[HALF1].to_numpy(float)
    vmax = float(np.nanpercentile(np.abs(allH), 99)) or 1.0       # one symmetric colour scale for all groups
    for g, ax in axd.items():
        sub = vc[vc["group"] == group_key(g)]
        H2g = sub[HALF2].to_numpy(float)    # ODD-trial curve: used ONLY to pick the sort order
        H1g = sub[HALF1].to_numpy(float)    # EVEN-trial curve: what is actually DISPLAYED
        keep = np.isfinite(H2g).all(1) & np.isfinite(H1g).all(1)
        H2g, H1g = H2g[keep], H1g[keep]
        if not len(H1g):
            ax.text(0.5, 0.5, "no cells", transform=ax.transAxes, ha="center", color="grey")
            continue
        order = np.lexsort((-H2g.max(1), H2g.argmax(1)))     # sort by the ODD-trial peak bin
        im = ax.imshow(H1g[order], aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       extent=[0, 1, len(H1g), 0], interpolation="nearest")
        ax.set_xlabel("pupil percentile", fontsize=8)
        ax.set_ylabel("cells (sorted by ODD-trial peak)", fontsize=8)
    fig.suptitle("Every cell's EVEN-trial tuning curve (rows = cells), sorted by the ODD-trial peak bin "
                 "(idea 4: never sort and display the same trials)", y=1.0, fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 0.92, 0.97])
    cax = fig.add_axes([0.94, 0.25, 0.012, 0.5])
    fig.colorbar(im, cax=cax, label="activity (SD), even trials")
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
