# -*- coding: utf-8 -*-
"""
2l_cross_area_tuning
====================
TUNING OF CELLS TO THE ACTIVITY OF ANOTHER POPULATION, e.g. how V1 cells fire in the stim window depending on the activity of
PM-lab cells in the pre-stim window. DN sessions, noise trials only (NOISE_TRIALS_ONLY).

For every session and every SOURCE group (SOURCE_GROUPS, e.g. "PM_lab"):
    source activity of a trial = mean over the source cells (>= MIN_SOURCE_CELLS in the session) of the cell's trial response in the
                                 SOURCE window (each cell's trial responses z-scored across trials first, so no cell dominates)
    source bins                = N_SOURCE_BINS equal-occupancy bins of that value across the session's trials (bin 0 = lowest source
                                 activity, last bin = highest)
    curve of a TARGET cell     = mean of the cell's z-scored trial response in the TARGET window in each source bin
for every pair of windows in WINDOW_PAIRS (source window, target window); windows are cm relative to the stimulus onset
(WINDOWS). Target cells = all included cells of ALL groups (8 with deep PM) except the cells of the source group itself.

Several versions (MODES) are computed, each removing one more shared influence:
    raw                    source bins over all trials. A relation here may simply be that both populations follow the stimulus / the
                           animal's state (higher stimulus strength, more arousal -> more activity in both areas).
    within_signal          source bins defined WITHIN each stimulus level (signal strength), null shuffles within levels.
    resid_behavior         running speed, pupil area, motion energy and lick rate (trial means in the same window as the response)
                           are regressed out of BOTH the source and every target response; bins/curves are of the residuals.
    resid_behavior+signal  the same + the stimulus level (one dummy per signal level); null shuffles within levels.
    resid_global           the GLOBAL component is regressed out of both: the mean z-response of all cells outside the source set in
                           the same window (what all areas share, e.g. a brain-wide gain).
    resid_all              behavior + stimulus level + global component.
    A relation that survives resid_all is a coupling beyond the shared stimulus / behavioral state / global gain. Regression is
    linear (OLS with intercept, trials with a missing value are left out); nonlinear shared influences are not removed.
CONTROL SOURCES (same analysis, same modes, to calibrate what the real source adds):
    CONTROL_SOURCE_GROUPS  other real groups used as the source (e.g. AL_unl: does PM-lab predict V1 more than AL does?)
    RANDOM_CONTROL         for every source group a matched-size set of random cells from the other groups (a source with no
                           group identity: if the real source predicts the targets no better than this, nothing is specific to it)
    Per-source figures are made for FIGURE_SOURCES only; the cross-mode / cross-source summary figures 7_ and 8_ include all.
Per target cell: eta2 (shuffle-corrected), permutation p-value (tuned = p < ALPHA, ~5 % by chance), diff_last_first (target response
in the highest minus the lowest source bin; >0 = the cell fires more when the source is more active), split-half r of the curve
(odd vs even trials), modulation.  The same-window pairs (e.g. prestim -> prestim) are simultaneous = they include shared slow
fluctuations; prestim -> stim / reward are predictive (the source precedes the target by one window).

Project-wide rules
    * selection : --animals / --sessions restrict the sessions (a restricted run writes to out/selection/<tag>/); --max-cells caps the
                  cells per group and session (it also thins the source population)
    * aggregate : --aggregate pooled|session|mouse = the unit of the group bar plots and of the curve bands
    * memory    : one session in RAM per worker; per-session results cached in store/ (no ledger)
    * cells     : 1b-included sessions; cells passing 2a (proximity, depth rule) and 2b QC.

Reads: 1b psychometric_fits.csv, 2a celldata_combined.csv, 2b qc_per_cell.csv.  Loads: raw behavior / calcium, one session at a time.

Outputs
-------
2_pipeline/2l_cross_area_tuning/
    out/    cross_tuning_curves.csv        per (source, mode, window pair, cell, source bin) -- only if WRITE_CELL_CURVES (large)
            cross_tuning_stats.csv         per (source, mode, window pair, cell): eta2_corr, p, diff_last_first, split_half_r, group ...
            cross_tuning_by_group[__<agg>].csv, cross_tuning_unit_curves[__<agg>].csv
            figures/<source>_<mode>/1_mean_curves.png (one panel per window pair, one line per target group; x = source bin, low -> high),
                    2_fraction_tuned.png, 3_variance_explained.png, 4_effect_high_minus_low.png, 5_split_half_reliability.png,
                    6_by_group_<pair>.png (one panel per target group; thin = sessions / mice / example cells)
            figures/7_effect_across_controls_<pair>.png, 8_fraction_tuned_across_controls_<pair>.png: x = mode (what is regressed
                    out), one line per source (real, control group, random), one panel per target group
    store/  cross_tuning_session_<hash>.pkl
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata


from infotheory.pipeline import get_pipeline_paths
from infotheory.continuous import merge_behavior_video, remove_pupil_outliers, zscore_per_session
from infotheory.session import discover_sessions
from infotheory.neural_encoding import load_cell_inclusion
from infotheory.celldata_utils import SIX_GROUP_ORDER, SIX_GROUP_LABELS, SIX_GROUP_COLORS
from infotheory.groups import group_key
from infotheory.plotting import set_style, save_fig, clear_figures
from infotheory.scope import add_scope_args, scope_from_args, select_sessions, scoped_dirs, data_stamp, unit_noun
from infotheory.session_jobs import run_sessions_units
from infotheory.unit_plots import group_panels, mouse_palette
from infotheory.trial_tuning import tune_trials, pack_stats, restrict_result
from infotheory import tensors as tn
from infotheory import cell_tables as ct

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom")
PROTOCOL = "DN"
NOISE_TRIALS_ONLY = True               # True: only the noise trials (0 < signal < 100 %) of the engaged trials
SOURCE_GROUPS = ["PMdeep_lab"]             # group keys (see 2k / SIX_GROUP_ORDER: V1_lab, V1_unl, PM_lab, PM_unl, AL_unl, RSP_unl, PMdeep_lab, PMdeep_unl)
CONTROL_SOURCE_GROUPS = ["AL_unl"]     # other real groups run as the source for comparison
CONTROL_MATCH_SIZE = True              # control groups are subsampled to the number of cells of the real source (a source with 300 cells
                                       # gives a much cleaner population signal than one with 10) and only run in sessions that have the real source
N_CONTROL_DRAWS = 3                    # random draws per control (their results are averaged in the group means)
RANDOM_CONTROL = True                  # + for each source group a matched-size random set of cells as source
FIGURE_SOURCES = SOURCE_GROUPS         # sources that get the full per-source figure set (all appear in the summary figures 7_ / 8_)
WRITE_CELL_CURVES = False              # also write cross_tuning_curves.csv (one row per cell x bin x source x mode x pair: large)
WINDOWS = {"prestim": (-30.0, -10.0), "stim": (0.0, 20.0), "reward": (25.0, 45.0)}      # cm rel. to stimStart
WINDOW_PAIRS = [("prestim", "prestim"), ("prestim", "stim"), ("prestim", "reward")]    # (source window, target window)
MODES = ["raw", "within_signal", "resid_behavior", "resid_behavior+signal", "resid_global", "resid_all"]
MODE_REGRESS = {"raw": [], "within_signal": [], "resid_behavior": ["behavior"], "resid_behavior+signal": ["behavior", "signal"],
                "resid_global": ["global"], "resid_all": ["behavior", "signal", "global"]}
LICK_SMOOTH_S = 1.0
N_SOURCE_BINS = 4
MIN_SOURCE_CELLS = 5                   # source cells needed in a session
MIN_TRIALS = 40                        # trials needed in a session
MIN_TRIALS_PER_BIN = 6                 # trials needed in each source bin (after pooling the stimulus levels in within_signal)
MIN_TRIALS_PER_LEVEL = 8               # within_signal: stimulus levels with fewer trials (>= 2 per bin) are left out
S_PRE, S_POST = 40.0, 60.0             # frames within [-S_PRE, +S_POST) cm of the stimulus onset are loaded for the window means
N_PERM = 200
ALPHA = 0.05
RANDOM_STATE = 0
MIN_CELLS_PER_UNIT = 5
N_EXAMPLE_CELLS = 40                   # pooled mode, figure 6: thin lines = this many random cells per group
JOBLIB_VERBOSE = 5

parser = argparse.ArgumentParser()
add_scope_args(parser)
parser.add_argument("--n-perm", type=int, default=N_PERM)
args = parser.parse_args()
scope = scope_from_args(args)
AGG = scope.aggregate
UNIT = unit_noun(AGG)
UNITS = "mice" if UNIT == "mouse" else f"{UNIT}s"
AGG_SUFFIX = "" if AGG == "pooled" else f"__{AGG}"
n_perm = args.n_perm
UNIT_KEYS = {"pooled": [], "session": ["animal_id", "session_id"], "mouse": ["animal_id"]}[AGG]
TRIAL_NOTE = "noise trials (0 < signal < 100 %)" if NOISE_TRIALS_ONLY else "all engaged trials"
GROUP_KEYS = [group_key(g) for g in SIX_GROUP_ORDER]
for g in SOURCE_GROUPS + CONTROL_SOURCE_GROUPS:
    if g not in GROUP_KEYS:
        raise SystemExit(f"Unknown source group {g}; choose from {GROUP_KEYS}")

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
cell_group = dict(zip(cell_inclusion["cell_id"], cell_inclusion["group"]))
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
def zcols(M):
    """z-score each column across trials (NaN-aware)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return (M - np.nanmean(M, 0)) / np.nanstd(M, 0)


def source_bins(src, levels, mode, nb):
    """Bin index (-1 = not used) of the per-trial source activity: equal-occupancy bins over all trials (raw) or within each
    stimulus level (within_signal; levels with too few trials are dropped)."""
    out = np.full(len(src), -1, dtype=int)
    fin = np.isfinite(src)
    groups = [fin] if mode == "raw" else [fin & (levels == lv) for lv in np.unique(levels[fin])]
    for m in groups:
        n = int(m.sum())
        if n < (nb if mode == "raw" else max(2 * nb, MIN_TRIALS_PER_LEVEL)):
            continue
        r = rankdata(src[m], method="ordinal")
        out[m] = np.minimum((r - 1) * nb // n, nb - 1)
    return out


def resid(Y, X):
    """Residuals of the columns of Y (n, k) after OLS on X (n, p) with intercept; rows with any non-finite value are NaN."""
    Y = np.asarray(Y, float)
    X = np.column_stack([np.ones(len(Y)), X])
    ok = np.isfinite(Y).all(1) & np.isfinite(X).all(1)
    out = np.full(Y.shape, np.nan)
    if ok.sum() > X.shape[1] + 5:
        beta = np.linalg.lstsq(X[ok], Y[ok], rcond=None)[0]
        out[ok] = Y[ok] - X[ok] @ beta
    return out


def coupling_r(s, T, strata, rng, n_perm):
    """Pearson r between the per-trial source activity s (n,) and every target response in T (n, N); two-sided permutation
    p-value (source labels permuted across trials, within `strata` if given). This is the linear (signed) version of the
    4-bin curve: more power for a monotonic relation, none for a non-monotonic one."""
    ok = np.isfinite(s) & np.isfinite(T).all(1)
    N = T.shape[1]
    if ok.sum() < MIN_TRIALS:
        return np.full(N, np.nan), np.full(N, np.nan)
    s0, T0 = s[ok], T[ok]
    sd, Td = s0.std(), T0.std(0)
    if sd == 0:
        return np.full(N, np.nan), np.full(N, np.nan)
    s0 = (s0 - s0.mean()) / sd
    with np.errstate(invalid="ignore", divide="ignore"):
        T0 = (T0 - T0.mean(0)) / np.where(Td > 0, Td, np.nan)
    n = len(s0)
    r = s0 @ T0 / n
    st = None if strata is None else np.asarray(strata)[ok]
    cnt = np.zeros(N)
    for _ in range(n_perm):
        if st is None:
            sp = rng.permutation(s0)
        else:
            sp = s0.copy()
            for l in np.unique(st):
                i = np.where(st == l)[0]
                sp[i] = rng.permutation(s0[i])
        cnt += np.abs(sp @ T0 / n) >= np.abs(r)
    return r, (1 + cnt) / (1 + n_perm)


def behavior_trial_means(ses, sel, ts_F, dt, k, rel, ok):
    """Window means per trial of speed, pupil area, motion energy and lick rate (z-scored across trials), per window:
    {window: (n_trials, <= 4)} or None if no video."""
    if ses.videodata is None:
        return None
    merged = zscore_per_session(remove_pupil_outliers(merge_behavior_video(ses.behaviordata, ses.videodata)))
    cols = []
    for c in ("runspeed", "pupil_area", "motionenergy"):
        if c in merged.columns:
            cols.append(tn.behavior_to_imaging_frames(merged["ts"], merged[c], ts_F, "mean"))
    if "lick" in merged.columns:
        ev = np.nan_to_num(tn.behavior_to_imaging_frames(merged["ts"], merged["lick"], ts_F, "rate"))
        kb = max(1, int(round(LICK_SMOOTH_S / dt)))
        cols.append(np.convolve(ev, np.ones(kb) / kb, mode="same"))
    if not cols:
        return None
    B = np.column_stack(cols)
    fin = np.isfinite(B)
    num = tn.trial_window_means(np.where(fin, B, 0.0), k, rel, ok, len(sel), WINDOWS)       # NaN-aware window means
    den = tn.trial_window_means(fin.astype(float), k, rel, ok, len(sel), WINDOWS)
    out = {}
    for w in WINDOWS:
        M = zcols(num[w] / np.where(den[w] > 0, den[w], np.nan))
        out[w] = M[:, np.isfinite(M).mean(0) > 0.5]
    return out


def process_session(ses, seed, new_cells, new_pairs, req_cells):
    sid = ses.session_id
    cell_ids = list(new_cells)
    td = ses.trialdata
    if (not cell_ids or ses.calciumdata is None or ses.ts_F is None or td is None or ses.behaviordata is None
            or "engaged" not in td.columns or "signal" not in td.columns):
        return dict(info=pd.DataFrame([dict(session_id=sid, status="missing data")]))
    info = []
    sel = td[td["engaged"] == 1]
    if NOISE_TRIALS_ONLY:
        sel = sel[(sel["signal"] > 0) & (sel["signal"] < 100)]
    sel = sel.sort_values("stimStart").reset_index(drop=True)
    if len(sel) < MIN_TRIALS:
        return dict(info=pd.DataFrame([dict(session_id=sid, status="too few trials", n_trials=len(sel))]))
    ts_F = np.asarray(ses.ts_F, float)
    dt = float(np.median(np.diff(ts_F)))
    A = ses.calciumdata.loc[:, cell_ids].to_numpy(dtype=float)
    zpos_F = tn.interp_to_imaging_frames(ses.behaviordata["ts"], ses.behaviordata["zpos"], ts_F)
    k, rel = tn.position_relative_to_stim(zpos_F, sel["stimStart"].to_numpy(float), S_PRE, S_POST)
    ok = tn.trial_frame_mask(ts_F, td, sel["trialNumber"]) & (k >= 0) & np.isfinite(A).all(1)
    if ok.sum() < 400:
        return dict(info=pd.DataFrame([dict(session_id=sid, status="too few frames", n_trials=len(sel))]))
    Y = tn.zscore_frames(np.nan_to_num(A), ok)
    good = np.isfinite(Y[ok]).all(0)
    if not good.any():
        return dict(info=pd.DataFrame([dict(session_id=sid, status="no active cells", n_trials=len(sel))]))
    ids = np.asarray(cell_ids)[good]
    Y = np.where(ok[:, None], np.nan_to_num(Y[:, good]), 0.0)
    R = tn.trial_window_means(Y, k, rel, ok, len(sel), WINDOWS)
    Zw = {w: zcols(R[w]) for w in WINDOWS}                         # every cell's trial responses, z-scored across trials
    beh = behavior_trial_means(ses, sel, ts_F, dt, k, rel, ok) if any("behavior" in v for v in MODE_REGRESS.values()) else None
    grp = np.array([cell_group.get(c) for c in ids])
    levels = sel["signal"].to_numpy(float)
    lv = np.unique(levels)
    dummies = np.column_stack([(levels == l).astype(float) for l in lv[1:]]) if len(lv) > 1 else np.zeros((len(levels), 0))
    odd = (np.arange(len(sel)) % 2 == 1)
    rng = np.random.default_rng((int(seed), 7))
    n_cells = len(ids)

    # source sets: (name, kind, draw, source cell index, target cell index)
    specs = []
    for sg in SOURCE_GROUPS:
        is_src = grp == sg
        if is_src.sum() < MIN_SOURCE_CELLS:
            info.append(dict(session_id=sid, source=sg, status="too few source cells", n_trials=len(sel),
                             n_source_cells=int(is_src.sum())))
            continue
        n = int(is_src.sum())
        specs.append((sg, "source", 0, np.where(is_src)[0], np.where(~is_src)[0]))
        pool = np.where(~is_src)[0]
        if RANDOM_CONTROL and len(pool) > 2 * n:
            for d in range(N_CONTROL_DRAWS):
                pick = np.sort(rng.choice(pool, n, replace=False))
                specs.append((f"random[{sg}]", "control_random", d, pick, np.setdiff1d(pool, pick)))
        for cg in CONTROL_SOURCE_GROUPS:
            is_c = grp == cg
            if is_c.sum() < max(n if CONTROL_MATCH_SIZE else MIN_SOURCE_CELLS, MIN_SOURCE_CELLS):
                info.append(dict(session_id=sid, source=f"{cg}[match {sg}]", status="too few control cells",
                                 n_trials=len(sel), n_source_cells=int(is_c.sum())))
                continue
            ci = np.where(is_c)[0]
            for d in range(N_CONTROL_DRAWS if CONTROL_MATCH_SIZE else 1):
                pick = np.sort(rng.choice(ci, n, replace=False)) if CONTROL_MATCH_SIZE else ci
                specs.append((f"{cg}[match {sg}]", "control_group", d, pick, np.where(~is_c)[0]))

    curves, stats = [], []
    for name, kind, draw, src_idx, tgt_idx in specs:
        rest_idx = np.setdiff1d(np.arange(n_cells), src_idx)
        glob = {w: zcols(np.nanmean(Zw[w][:, rest_idx], axis=1)[:, None])[:, 0] for w in WINDOWS}       # global component
        for ws, wt in WINDOW_PAIRS:
            for mode in MODES:
                reg = MODE_REGRESS[mode]
                row = dict(session_id=sid, source=name, kind=kind, draw=draw, mode=mode, wpair=f"{ws}->{wt}", n_trials=len(sel),
                           n_source_cells=len(src_idx))
                if "behavior" in reg and (beh is None or beh[ws].shape[1] == 0 or beh[wt].shape[1] == 0):
                    info.append(dict(row, status="no video / behavior"))
                    continue

                def design(w):
                    cols = []
                    if "behavior" in reg:
                        cols.append(beh[w])
                    if "signal" in reg:
                        cols.append(dummies)
                    if "global" in reg:
                        cols.append(glob[w][:, None])
                    return np.column_stack(cols) if cols else None

                src = np.nanmean(Zw[ws][:, src_idx], axis=1)
                T = Zw[wt][:, tgt_idx]
                if reg:
                    src = resid(src[:, None], design(ws))[:, 0]
                    T = resid(T, design(wt))
                xb = source_bins(src, levels, "within_signal" if mode == "within_signal" else "raw", N_SOURCE_BINS)
                strata = levels if (mode == "within_signal" or "signal" in reg) else None
                res = tune_trials(T, xb, N_SOURCE_BINS, odd, rng, n_perm, MIN_TRIALS, MIN_TRIALS_PER_BIN, strata=strata)
                if mode == "within_signal" and dummies.shape[1]:        # r: remove the stimulus-level means from both
                    src_r, T_r = resid(src[:, None], dummies)[:, 0], resid(T, dummies)
                else:
                    src_r, T_r = src, T
                r_vec, p_vec = coupling_r(src_r, T_r, strata, rng, n_perm)
                cnt_ = np.bincount(xb[xb >= 0], minlength=N_SOURCE_BINS)
                row.update(n_used_trials=int((xb >= 0).sum()), min_bin_trials=int(cnt_.min()))
                if res is None:
                    info.append(dict(row, status="too few trials in a source bin / used"))
                    continue
                info.append(dict(row, status="ok"))
                r_, s_ = pack_stats(sid, ids[tgt_idx][res["good"]], row["wpair"], restrict_result(res), N_SOURCE_BINS)
                g_ = res["good"]
                tag = dict(source=name, kind=kind, draw=draw, mode=mode, wpair=row["wpair"])
                stats.append(s_.assign(r=r_vec[g_], p_r=p_vec[g_], **tag))
                if name in FIGURE_SOURCES or kind == "control_random" and WRITE_CELL_CURVES:
                    curves.append(r_.assign(**tag))
    info = pd.DataFrame(info)
    if not stats:
        return dict(info=info)
    out = dict(stats=pd.concat(stats, ignore_index=True), info=info)
    out["curves"] = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(
        columns=["session_id", "cell_id", "variable", "bin", "n", "mean", "mean_h1", "mean_h2", "source", "kind", "mode", "wpair"])
    return out


results = run_sessions_units(
    session_ids, process_session, [PROTOCOL], paths.store, "cross_tuning_session",
    key_params=dict(version="2l-v3", ctl=(CONTROL_MATCH_SIZE, N_CONTROL_DRAWS), noise=NOISE_TRIALS_ONLY, sources=(SOURCE_GROUPS, CONTROL_SOURCE_GROUPS, RANDOM_CONTROL, FIGURE_SOURCES, WRITE_CELL_CURVES), regress=MODE_REGRESS, lick_smooth=LICK_SMOOTH_S, windows=WINDOWS, pairs=WINDOW_PAIRS,
                    modes=MODES, nb=N_SOURCE_BINS, mins=(MIN_SOURCE_CELLS, MIN_TRIALS, MIN_TRIALS_PER_BIN, MIN_TRIALS_PER_LEVEL),
                    s=(S_PRE, S_POST), n_perm=n_perm, qc_mtime=qc_mtime),
    load_kwargs=dict(load_behaviordata=True, load_videodata=True, load_celldata=True, load_calciumdata=True),
    cells_by_session=cells_by_session, parts={"curves": "cell", "stats": "cell"}, groups_by_cell=cell_group,
    max_cells=scope.max_cells, base_seed=RANDOM_STATE, n_jobs=scope.n_jobs, recompute=args.recompute,
    verbose=JOBLIB_VERBOSE, extra_key_by_session=stamps)
results = [r for r in results if r]
# --- why are sessions (not) in the results? ---
status = pd.concat([r["info"] for r in results if "info" in r], ignore_index=True) if results else pd.DataFrame()
if len(status):
    status.to_csv(table_dir / "cross_tuning_session_status.csv", index=False)
    cols = [c for c in ["source", "mode", "wpair", "status"] if c in status.columns]
    print("\nSession status (number of sessions per outcome; a session counts once per source and mode, pooled over window pairs):")
    print(status.fillna({"source": "-", "mode": "-", "wpair": "-"}).groupby(["source", "mode", "status"])["session_id"].nunique()
          .rename("n_sessions").to_string())
    if "n_source_cells" in status.columns:
        print("\nSource cells per session (sessions with fewer than MIN_SOURCE_CELLS are skipped):")
        print(status.dropna(subset=["n_source_cells"]).drop_duplicates(["session_id", "source"])
              [["session_id", "source", "n_source_cells", "n_trials"]].sort_values("n_source_cells").to_string(index=False))
results = [r for r in results if "stats" in r]
if not results:
    raise SystemExit(f"No results (sessions without >= {MIN_SOURCE_CELLS} cells of {SOURCE_GROUPS} or too few trials); see the status table above.")

# --------------------------------------------------------------------- #
# 3. Tables
# --------------------------------------------------------------------- #
cd = pd.read_csv(cell_dist_out / "celldata_combined.csv", low_memory=False)
stats = ct.attach_cell_meta(pd.concat([r["stats"] for r in results], ignore_index=True), cd)
stats = stats[stats["group"].isin(GROUP_KEYS)].reset_index(drop=True)
if stats.empty:
    raise SystemExit("No target cells in any group.")
stats["tuned"] = stats["p_value"] < ALPHA
meta = stats[["session_id", "cell_id", "group", "animal_id"]].drop_duplicates(["session_id", "cell_id"])
_cu = [r["curves"] for r in results if len(r["curves"])]       # sessions without a figure-source result return an empty frame
if not _cu:
    raise SystemExit(f"No curves for FIGURE_SOURCES={FIGURE_SOURCES}; see the status table above.")
curves = pd.concat(_cu, ignore_index=True).merge(meta, on=["session_id", "cell_id"], how="inner")
curves = curves.astype({"bin": int, "mean": float, "mean_h1": float, "mean_h2": float, "n": float})
del results
stats.to_csv(table_dir / "cross_tuning_stats.csv", index=False, float_format="%.4g")
if WRITE_CELL_CURVES:
    curves.to_csv(table_dir / "cross_tuning_curves.csv", index=False, float_format="%.4g")
stats["combo"] = stats["source"] + "|" + stats["mode"] + "|" + stats["wpair"]
tg = ct.unit_group_summary(stats, ["source", "mode", "wpair"], AGG, ["eta2_corr", "diff_last_first", "split_half_r", "modulation", "r"],
                           sig_col="tuned")
tg.to_csv(table_dir / f"cross_tuning_by_group{AGG_SUFFIX}.csv", index=False)
for (sg, md), t_ in tg.groupby(["source", "mode"]):
    if sg not in FIGURE_SOURCES:
        continue
    print(f"\nFraction of target cells tuned to {sg} activity (p<{ALPHA}; chance ~{ALPHA:.0%}), mode '{md}', per target group:")
    print(t_.groupby(["wpair", "group"])["frac_sig"].mean().unstack("group").reindex(columns=GROUP_KEYS).round(2).to_string())

# --- statistical tests at the level of SESSIONS (the independent unit; cells within a session are not) ---
from scipy import stats as sps


def bh(pv):
    """Benjamini-Hochberg q-values."""
    pv = np.asarray(pv, float)
    q = np.full(len(pv), np.nan)
    ok = np.isfinite(pv)
    if ok.sum():
        o = np.argsort(pv[ok])
        ranked = pv[ok][o] * ok.sum() / (np.arange(ok.sum()) + 1)
        q[np.where(ok)[0][o]] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    return q


MIN_UNITS_TEST = 4
sess = (stats.groupby(["source", "kind", "mode", "wpair", "group", "animal_id", "session_id"])
        .agg(r=("r", "mean"), eta2=("eta2_corr", "mean"), n_cells=("cell_id", "nunique")).reset_index())
sess = sess[sess["n_cells"] >= MIN_CELLS_PER_UNIT]
trows = []
for (md_, wp_, g_), d in sess.groupby(["mode", "wpair", "group"]):
    for sg in SOURCE_GROUPS:
        real = d[d["source"] == sg].set_index("session_id")
        if len(real) < MIN_UNITS_TEST:
            continue
        comps = [("vs_zero", None)] + [("vs_" + c, c) for c in [f"random[{sg}]"] + [f"{c}[match {sg}]" for c in CONTROL_SOURCE_GROUPS]]
        for lab, ctl in comps:
            if ctl is None:
                x = real["r"]
                mice = real.groupby("animal_id")["r"].mean()
            else:
                c = d[d["source"] == ctl].set_index("session_id")["r"]
                common = real.index.intersection(c.index)
                if len(common) < MIN_UNITS_TEST:
                    continue
                x = real.loc[common, "r"] - c.loc[common]
                mice = pd.concat([x, real.loc[common, "animal_id"]], axis=1).groupby("animal_id")["r"].mean()
            if x.std() == 0 or not np.isfinite(x.std()):
                continue
            try:
                pw = sps.wilcoxon(x).pvalue
            except Exception:
                pw = np.nan
            trows.append(dict(source=sg, comparison=lab, mode=md_, wpair=wp_, group=g_, n_sessions=len(x), n_mice=len(mice),
                              mean_r=float(x.mean()), sem=float(x.sem()), p_t=float(sps.ttest_1samp(x, 0).pvalue), p_wilcoxon=pw,
                              p_mice=float(sps.ttest_1samp(mice, 0).pvalue) if len(mice) >= 3 and mice.std() > 0 else np.nan))
tests = pd.DataFrame(trows)
if len(tests):
    tests["q_t"] = np.nan
    for _, ix in tests.groupby(["comparison", "mode"]).groups.items():      # BH over target groups x window pairs
        tests.loc[ix, "q_t"] = bh(tests.loc[ix, "p_t"])
    tests.to_csv(table_dir / "cross_tuning_tests.csv", index=False, float_format="%.4g")
    print("\nSession-level tests (mean coupling r; 'vs_zero' = real source vs 0; 'vs_<control>' = real minus control, paired by session;"
          " q = BH-corrected over groups x window pairs within comparison and mode). Smallest p:")
    show = tests.sort_values("p_t").head(25)[["source", "comparison", "mode", "wpair", "group", "n_sessions", "n_mice", "mean_r", "p_t", "q_t", "p_mice"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3g}"))

agg_note = {"pooled": "all cells pooled (no error bars)", "session": "bar = mean +- SEM over sessions, dots = sessions",
            "mouse": "bar = mean +- SEM over mice, dots = mice"}[AGG]
BIN_LABELS = [f"Q{i + 1}" for i in range(N_SOURCE_BINS)]
BIN_LABELS[0] += " (low)"
BIN_LABELS[-1] += " (high)"


def unit_mean_curves(d):
    """Mean curve per target group: mean over units of the unit's mean over its cells, SEM over units (pooled: over cells)."""
    if AGG == "pooled":
        g = d.groupby(["group", "bin"])["mean"].agg(["mean", "sem", "size"]).reset_index()
        return g.rename(columns={"size": "n_units"})
    per = d.groupby(["group"] + UNIT_KEYS + ["bin"])["mean"].agg(["mean", "size"]).reset_index()
    per = per[per["size"] >= MIN_CELLS_PER_UNIT]
    g = per.groupby(["group", "bin"])["mean"].agg(["mean", "sem", "size"]).reset_index()
    return g.rename(columns={"size": "n_units"})


unit_rows = []
rs = np.random.default_rng(0)
for (sg, md), st_ in stats.groupby(["source", "mode"]):
    if sg not in FIGURE_SOURCES:
        continue
    cu_ = curves[(curves["source"] == sg) & (curves["mode"] == md)]
    sub = f"{sg}_{md}"
    fdir = figdir / sub
    fdir.mkdir(parents=True, exist_ok=True)
    pairs = [f"{a}->{b}" for a, b in WINDOW_PAIRS if (st_["wpair"] == f"{a}->{b}").any()]
    ttl = f"source = {sg} ({md}); target = all other groups; {PROTOCOL}, {TRIAL_NOTE}"
    st2 = st_.assign(variable=st_["wpair"])
    # 1: mean curves per window pair
    ncols = min(3, len(pairs))
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.6 * nrows), squeeze=False, sharey=True)
    for ax, wp in zip(axes.ravel(), pairs):
        g = unit_mean_curves(cu_[cu_["wpair"] == wp])
        for grp in SIX_GROUP_ORDER:
            s = g[g["group"] == group_key(grp)].sort_values("bin")
            if s.empty:
                continue
            col = SIX_GROUP_COLORS[grp]
            ax.plot(s["bin"], s["mean"], color=col, marker="o", markersize=3, lw=1.3, label=SIX_GROUP_LABELS[grp])
            if np.isfinite(s["sem"]).any():
                ax.fill_between(s["bin"], s["mean"] - s["sem"].fillna(0), s["mean"] + s["sem"].fillna(0), color=col, alpha=0.15, lw=0)
            unit_rows.append(s.assign(source=sg, mode=md, wpair=wp))
        ax.axhline(0, color="grey", lw=0.6, ls="--")
        ax.set_xticks(range(N_SOURCE_BINS))
        ax.set_xticklabels(BIN_LABELS, fontsize=7)
        ax.set_title(wp.replace("->", " -> ") + " (source -> target window)", fontsize=9)
        ax.set_xlabel(f"{sg} activity in the source window", fontsize=8)
    for ax in axes.ravel()[len(pairs):]:
        ax.axis("off")
    for r_ in range(nrows):
        axes[r_][0].set_ylabel("target response (SD of trials)", fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle(f"Target-cell response by source-population activity -- {ttl}; "
                 f"{'mean +- SEM over cells' if AGG == 'pooled' else f'mean +- SEM over {UNITS}'}; y shared", y=1.0, fontsize=9)
    fig.tight_layout()
    save_fig(fig, fdir, "1_mean_curves")
    plt.close(fig)
    # 2-5: group bar panels
    kw = dict(mouse_colors=mouse_colors, min_cells_per_unit=MIN_CELLS_PER_UNIT, ncols=3, panel_size=(4.6, 3.2))
    group_panels(st2.assign(sig=st2["tuned"].astype(float)), "variable", pairs, "sig", AGG, fdir, "2_fraction_tuned",
                 f"Fraction of target cells tuned to the source activity -- {ttl}; dashed = chance; {agg_note}",
                 ylabel=f"fraction of cells p<{ALPHA}", hline=ALPHA, **kw)
    group_panels(st2, "variable", pairs, "eta2_corr", AGG, fdir, "3_variance_explained",
                 f"Variance of the target response explained by the source bins (shuffle-corrected) -- {ttl}; {agg_note}",
                 ylabel="eta2 (corrected)", hline=0, **kw)
    group_panels(st2, "variable", pairs, "diff_last_first", AGG, fdir, "4_effect_high_minus_low",
                 f"Target response in the highest minus the lowest source bin (SD; >0 = fires more when {sg} is more active) -- {ttl}; {agg_note}",
                 ylabel="high - low source (SD)", hline=0, **kw)
    group_panels(st2, "variable", pairs, "r", AGG, fdir, "5b_signed_coupling_r",
                 f"Signed linear coupling r between target response and source activity (>0 = more active when {sg} is more active) -- {ttl}; {agg_note}",
                 ylabel="coupling r", hline=0, **kw)
    group_panels(st2, "variable", pairs, "split_half_r", AGG, fdir, "5_split_half_reliability",
                 f"Split-half reliability of the curve (odd vs even trials) -- {ttl}; {agg_note}",
                 ylabel="split-half r", hline=0, **kw)
    # 6: per window pair, one panel per target group
    ukeys = UNIT_KEYS if AGG != "pooled" else ["session_id", "cell_id"]
    for wp in pairs:
        d = cu_[cu_["wpair"] == wp]
        g = unit_mean_curves(d)
        fig, axes = plt.subplots(2, 4, figsize=(4.2 * 4, 3.3 * 2), squeeze=False, sharex=True, sharey=True)
        allv = []
        for ax, grp in zip(axes.ravel(), SIX_GROUP_ORDER):
            key = group_key(grp)
            col = SIX_GROUP_COLORS[grp]
            dd = d[d["group"] == key]
            ax.set_title(SIX_GROUP_LABELS[grp], fontsize=9)
            if dd.empty:
                continue
            if AGG == "pooled":
                keys = dd[ukeys].drop_duplicates()
                keys = keys.iloc[rs.permutation(len(keys))[:N_EXAMPLE_CELLS]]
                thin = dd.merge(keys, on=ukeys).groupby(ukeys + ["bin"])["mean"].mean().reset_index()
                thin["animal_id"] = None
            else:
                thin = dd.groupby(["animal_id"] + [k_ for k_ in ukeys if k_ != "animal_id"] + ["bin"])["mean"].agg(
                    ["mean", "size"]).reset_index()
                thin = thin[thin["size"] >= MIN_CELLS_PER_UNIT]
            for _, u in thin.groupby(ukeys):
                u = u.sort_values("bin")
                lc = col if AGG == "pooled" else mouse_colors.get(u["animal_id"].iloc[0], "grey")
                ax.plot(u["bin"], u["mean"], color=lc, lw=0.7, alpha=0.25 if AGG == "pooled" else 0.6)
            allv += list(thin["mean"].dropna())
            s = g[g["group"] == key].sort_values("bin")
            if not s.empty:
                ax.plot(s["bin"], s["mean"], color=col if AGG == "pooled" else "black", marker="o", markersize=3, lw=2)
                if np.isfinite(s["sem"]).any():
                    ax.fill_between(s["bin"], s["mean"] - s["sem"].fillna(0), s["mean"] + s["sem"].fillna(0), color="grey", alpha=0.3, lw=0)
                allv += list(s["mean"] + s["sem"].fillna(0)) + list(s["mean"] - s["sem"].fillna(0))
            ax.axhline(0, color="grey", lw=0.6, ls="--")
            ax.set_xticks(range(N_SOURCE_BINS))
            ax.set_xticklabels(BIN_LABELS, fontsize=6)
        allv = [x_ for x_ in allv if np.isfinite(x_)]
        if allv:
            lo_, hi_ = np.percentile(allv, [1, 99])
            pad = 0.1 * (hi_ - lo_ + 1e-9)
            axes[0][0].set_ylim(lo_ - pad, hi_ + pad)
        for r_ in range(2):
            axes[r_][0].set_ylabel("target response (SD)", fontsize=9)
        what = {"pooled": f"thin = {N_EXAMPLE_CELLS} random cells, thick = mean",
                "session": "thin = sessions (colour = mouse), black = mean +- SEM", "mouse": "thin = mice, black = mean +- SEM"}[AGG]
        fig.suptitle(f"{wp.replace('->', ' -> ')}: {ttl}; {what}; y shared (1-99 percentile)", fontsize=9, y=1.0)
        fig.tight_layout()
        save_fig(fig, fdir, f"6_by_group_{wp.replace('->', '_to_')}")
        plt.close(fig)

# 7 / 8: effect across modes (what is regressed out) and across sources, one figure per window pair
SRC_ORDER = (FIGURE_SOURCES + [f"{c}[match {x}]" for x in FIGURE_SOURCES for c in CONTROL_SOURCE_GROUPS]
             + [f"random[{x}]" for x in FIGURE_SOURCES if RANDOM_CONTROL])
SRC_STYLE = {n: (c, ls) for n, c, ls in zip(SRC_ORDER, ["black", "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"],
                                             ["-", "-", "--", "-", "--", "-"])}
mode_x = {m: i for i, m in enumerate(MODES)}
for wp in [f"{a}->{b}" for a, b in WINDOW_PAIRS]:
    for metric, ylab, fname, hl in [("eta2_corr", "eta2 (corrected)", "7_effect_across_controls", 0),
                                    ("frac_sig", f"fraction of cells p<{ALPHA}", "8_fraction_tuned_across_controls", ALPHA),
                                    ("r", "mean signed coupling r (residuals)", "9_signed_coupling_across_controls", 0)]:
        t_ = tg[tg["wpair"] == wp]
        if t_.empty:
            continue
        fig, axes = plt.subplots(2, 4, figsize=(4.4 * 4, 3.4 * 2), squeeze=False, sharex=True, sharey=True)
        for ax, grp in zip(axes.ravel(), SIX_GROUP_ORDER):
            key = group_key(grp)
            ax.set_title(SIX_GROUP_LABELS[grp], fontsize=9)
            for src_n in SRC_ORDER:
                d = t_[(t_["group"] == key) & (t_["source"] == src_n)]
                if d.empty:
                    continue
                g = d.groupby("mode")[metric].agg(["mean", "sem"]).reindex(MODES)
                col, ls = SRC_STYLE[src_n]
                x = np.arange(len(MODES))
                ax.errorbar(x, g["mean"], yerr=g["sem"] if AGG != "pooled" else None, color=col, ls=ls, marker="o", ms=3,
                            lw=1.3, capsize=2, label=src_n)
            ax.axhline(hl, color="grey", lw=0.6, ls="--")
            ax.set_xticks(range(len(MODES)))
            ax.set_xticklabels(MODES, fontsize=6, rotation=45, ha="right")
        axes[0][0].legend(frameon=False, fontsize=7)
        for r_ in range(2):
            axes[r_][0].set_ylabel(ylab, fontsize=9)
        fig.suptitle(f"{wp.replace('->', ' -> ')}: {ylab} of target cells vs source activity, as more shared influences are regressed "
                     f"out (left -> right); one line per source (a source group is not its own target); {PROTOCOL}, {TRIAL_NOTE}; "
                     f"{'mean +- SEM over ' + UNITS if AGG != 'pooled' else 'pooled cells'}", fontsize=8, y=1.0)
        fig.tight_layout()
        save_fig(fig, figdir, f"{fname}_{wp.replace('->', '_to_')}")
        plt.close(fig)

if unit_rows:
    pd.concat(unit_rows, ignore_index=True).to_csv(table_dir / f"cross_tuning_unit_curves{AGG_SUFFIX}.csv", index=False)
print(f"\nWrote tables to {table_dir}, figures to {figdir}")
