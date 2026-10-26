# -*- coding: utf-8 -*-
"""
3h_pair_correlations_strategies
===============================
Pairwise correlations over POSITION (cm from stimulus onset) and over TIME (s from stimulus onset), for a large sample of
pairs (150 per session x area x label, sampled here as in 3g), computed with TWO ways of aggregating the calcium frames of
a bin, and with TWO kinds of bins:

  aggregation of the frames of a bin
    frame_pooled  the frames (time points) of ALL valid trials in the bin are POOLED across trials into one long sample and
                  the correlation is computed over all of them; every frame carries its trial's target label.
                  signal_corr = correlation of the target-bin means, noise_corr = correlation of the frame residuals after
                  removing the target-bin mean, raw_corr = plain Pearson correlation over frames. Frames of one trial are
                  not independent (checked with a trial-shuffle null: the null spread is close to the naive 1/sqrt(n)),
                  frame-level correlations are smaller than trial-mean ones because frame noise is not averaged out.
    trial_mean    one value per TRIAL: the trial's mean activity of each cell in the bin; the correlation is computed fully
                  across trials (signal / noise / raw as in 3a and 3d).
  bins
    fixed         non-overlapping bins (position BINSIZE cm / time BINSIZE s): as 3d.
    sliding       overlapping windows (position WINDOW cm moved by STEP cm / time WINDOW s moved by STEP s): the
                  same data are re-used by neighbouring windows, so the curve is smoother and less noisy (neighbouring
                  points are NOT independent -- read the shape, not point-to-point differences). Window centre on the x axis.

TARGETS (what the signal / noise correlation is conditioned on): each stimulus strategy of stim_binning.py
(`zero_to80_above80`, `pct_0-20_20-80_80-100`, `zero_vs_nonzero`), `choice`, and `all_levels` = EVERY stimulus value (0, 20,
..., 100) is its own bin. signal_corr needs >= 3 target bins (a correlation of 2 points is always +-1): it exists for the
stimulus strategies with 3 bins and for all_levels (6 bins, the best-defined tuning similarity), and it is empty for the
2-bin targets.

FILTERS (the project's, as 3a-3g): sessions = 1b-included (psychometric_fits.csv), restricted by --animals / --sessions;
cells = 2a proximity + 2b QC (load_cell_inclusion); pairs = within (area, label), MAX_PAIRS_PER_GROUP per (session, area,
label); ENGAGED trials only and a frame is used only if it lies inside its OWN engaged trial's [tStart, tEnd] (bins that
reach into the neighbouring trial do not pick up its frames: far bins therefore lose trials); trials per (bin, target): finite
activity and label, target bins with < MIN_TRIALS_PER_TARGET_BIN trials dropped, >= MIN_TRIALS_ANALYSIS trials, 2 labels
(the pair_trials rule of 3a-3g; it is cell-independent here, so it is applied once per bin and target).

Project-wide rules (infotheory/project_config.py) -- what applies here
    * selection : --animals / --sessions restrict the sessions; a restricted run writes to out/selection/<tag>/.
    * aggregate : --aggregate pooled|session|mouse. Curves = per group, binning, axis, method, target, bin: the mean over
                  units of the unit's mean over its pairs (units with < MIN_PAIRS_PER_UNIT pairs left out), band = SEM over
                  units; pooled: mean over all pairs, band = SEM over pairs (pairs treated as independent). Six groups always.
    * memory    : one session in RAM per worker (--n-jobs); correlations of all pairs of a session are computed together
                  (standardised matrices) and reduced to per-(session, group) sums, so no per-pair table is written.
    * binning   : stimulus / choice targets and the trial rule shared with 3a-3g; position bins on the same grid as 3d.
    * axes      : --axes position time (default both). Time = frame timestamp minus the time the odometer reaches the
                  trial's stimStart (psth.derive_onset_time). KNOWN CAVEAT (as 2f): windows far from onset can reach into
                  the neighbouring trial; the own-trial-window filter removes those frames, so the far bins have few trials.

Outputs
-------
2_pipeline/3h_pair_correlations_strategies/
    out/    pair_correlation_moments.csv               per (session, binning, axis, method, target, bin, group): n_trials,
                                                       n_frames and, for each metric, n / sum / sum of squares over the pairs
            pair_correlations_strategies_by_group[__<aggregate>].csv   curves per group x binning x axis x method x target x bin
            figures/<binning>/<method>/<target>/1_<axis>_correlations.png   signal / noise / raw (rows) x area (columns:
                                                       V1, PM, AL, RSP; V1 and PM show two lines, lab and unl);
                                                       position plots shade 3a's windows (-30..-10, 0..20, 25..45 cm)
            figures/<binning>/<method>/<target>/2_<axis>_<metric>.png   the same, one metric per figure (raw = Pearson)
    out/selection/<tag>/...                            the same for a restricted run
    store/  strategy_corr_<hash>.pkl                   per-session caches
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse

from infotheory import project_config as cfg
from infotheory.pipeline import get_pipeline_paths
from infotheory.session import discover_sessions
from infotheory.celldata_utils import SIX_GROUP_ORDER, SIX_GROUP_LABELS, SIX_GROUP_COLORS
from infotheory.groups import group_key, add_six_group_column
from infotheory.plotting import set_style, save_fig, clear_figures
from infotheory.scope import add_scope_args, scope_from_args, select_sessions, scoped_dirs, data_stamp, unit_noun
from infotheory.session_jobs import run_sessions
from infotheory.neural_encoding import build_session_tables, load_cell_inclusion
from infotheory.continuous import build_calcium_continuous
from infotheory.psth import derive_onset_time
from infotheory.unit_stats import stable_seed
from infotheory.stim_binning import (STIM_BINNING_STRATEGIES, DEFAULT_STRATEGIES, CHOICE_TARGET, target_signature,
                                     target_bin_names, describe_strategies)
from infotheory import pair_tables as pt
from infotheory import position_info as pi

PROTOCOL = "DN"
# fixed bins and sliding windows, per axis (position in cm, time in s)
AXES_DEF = {"position": dict(pre=-60.0, post=80.0, binsize=20.0, window=20.0, step=5.0,
                             label="position rel. to stimulus onset (cm)"),
            "time": dict(pre=-4.0, post=6.0, binsize=1.0, window=2.0, step=0.5,
                         label="time rel. to stimulus onset (s)")}
BINNINGS = ["fixed", "sliding"]
METHODS = ["frame_pooled", "trial_mean"]
WINDOWS = {"pre_stim": (-30.0, -10.0), "stim": (0.0, 20.0), "reward": (25.0, 45.0)}   # 3a's windows (cm), shaded on position plots
MAX_PAIRS_PER_GROUP = 200    # per (session, area, label)
MIN_PAIRS_PER_UNIT = 5
MIN_FRAMES_POOLED = 30       # frame_pooled: frames needed in the bin (after the trial rule)
RANDOM_STATE = 0
BINNING_STRATEGIES = list(DEFAULT_STRATEGIES) + ["all_levels"]
INCLUDE_CHOICE_TARGET = True
JOBLIB_VERBOSE = 5
METRICS = ["signal_corr", "noise_corr", "raw_corr"]
SHARE_Y = True               # panels of one metric share the y range

parser = argparse.ArgumentParser()
add_scope_args(parser)
parser.add_argument("--strategies", nargs="+", default=None, choices=sorted(STIM_BINNING_STRATEGIES))
parser.add_argument("--max-pairs", type=int, default=MAX_PAIRS_PER_GROUP, help="pairs per (session, area, label)")
parser.add_argument("--axes", nargs="+", default=["position", "time"], choices=list(AXES_DEF))
parser.add_argument("--binnings", nargs="+", default=BINNINGS, choices=BINNINGS)
parser.add_argument("--position-bin", type=float, default=AXES_DEF["position"]["binsize"], help="fixed position bin (cm)")
parser.add_argument("--position-window", type=float, default=AXES_DEF["position"]["window"], help="sliding window (cm)")
parser.add_argument("--position-step", type=float, default=AXES_DEF["position"]["step"], help="sliding step (cm)")
parser.add_argument("--time-bin", type=float, default=AXES_DEF["time"]["binsize"], help="fixed time bin (s)")
parser.add_argument("--time-window", type=float, default=AXES_DEF["time"]["window"], help="sliding window (s)")
parser.add_argument("--time-step", type=float, default=AXES_DEF["time"]["step"], help="sliding step (s)")
args = parser.parse_args()
for ax_ in ("position", "time"):
    AXES_DEF[ax_]["binsize"] = getattr(args, f"{ax_}_bin")
    AXES_DEF[ax_]["window"] = getattr(args, f"{ax_}_window")
    AXES_DEF[ax_]["step"] = getattr(args, f"{ax_}_step")
    r_ = AXES_DEF[ax_]["window"] / AXES_DEF[ax_]["step"]
    if abs(r_ - round(r_)) > 1e-9:
        raise SystemExit(f"{ax_}: the sliding window must be a multiple of the step")
scope = scope_from_args(args)
AGG = scope.aggregate
UNIT = unit_noun(AGG)
AGG_SUFFIX = "" if AGG == "pooled" else f"__{AGG}"
STRATEGIES = args.strategies or BINNING_STRATEGIES
TARGETS = STRATEGIES + ([CHOICE_TARGET] if INCLUDE_CHOICE_TARGET else [])
AXES = list(args.axes)
BINS_RUN = list(args.binnings)
GROUP_KEYS = [group_key(g) for g in SIX_GROUP_ORDER]
GCODE = {k: i for i, k in enumerate(GROUP_KEYS)}
NG = len(GROUP_KEYS)

paths = get_pipeline_paths(__file__)
table_dir, figdir = scoped_dirs(paths, scope)
clear_figures(figdir)
for _sub in figdir.glob("*/"):
    clear_figures(_sub)
set_style()
print(scope.describe())
print("Targets (stimulus binning strategies + choice):\n" + describe_strategies(TARGETS))

# --------------------------------------------------------------------- #
# 1. Inputs (as 3g)
# --------------------------------------------------------------------- #
fits_path = paths.out_from("1b_psychometric") / "psychometric_fits.csv"
cell_dist_out = paths.out_from("2a_cell_distribution")
activity_out = paths.out_from("2b_activity_statistics")
for f, hint in [(fits_path, "1b_psychometric.py"), (cell_dist_out / "celldata_combined.csv", "2a_cell_distribution.py"),
                (activity_out / "qc_per_cell.csv", "2b_activity_statistics.py")]:
    if not f.exists():
        raise SystemExit(f"Run {hint} first -- expected {f}")
fits_df = pd.read_csv(fits_path)
mu_sigma_lookup = fits_df.set_index("session_id")[["mu", "sigma"]].to_dict("index")
included_all = set(fits_df[fits_df["included"] & (fits_df["protocol"] == PROTOCOL)]["session_id"])
all_triples = discover_sessions([PROTOCOL])
selected = {f"{a}_{d}": (p, a, d) for p, a, d in select_sessions(all_triples, scope)}
session_ids = [s for s in selected if s in included_all]
if not session_ids:
    raise SystemExit("None of the selected sessions passed 1b_psychometric's inclusion criteria.")
inclusion = load_cell_inclusion(cell_dist_out, activity_out)
cells_by_session = {sid: set(sub.loc[sub["include"], "cell_id"]) for sid, sub in inclusion.groupby("session_id")
                    if sid in session_ids}
session_ids = [s for s in session_ids if cells_by_session.get(s)]
if not session_ids:
    raise SystemExit("No selected session has included cells.")
stamps = {s: data_stamp(*selected[s]) for s in session_ids}
print(f"{len(session_ids)} sessions")


# --------------------------------------------------------------------- #
# 2. Vectorised correlations of all pairs of a bin
# --------------------------------------------------------------------- #
def zscore(X):
    sd = X.std(0)
    sd = np.where(sd < 1e-12, np.nan, sd)
    return (X - X.mean(0)) / sd


def pair_corr(Z, I, J):
    return (Z[:, I] * Z[:, J]).mean(0)


def group_means(X, y, n_lab):
    """Label means (n_lab x cells) and the label-residuals of X (indicator-matrix product)."""
    M = sparse.csr_matrix((np.ones(len(y)), (y, np.arange(len(y)))), shape=(n_lab, len(y)))
    cnt = np.asarray(M.sum(1)).ravel()
    G = (M @ X) / np.maximum(cnt, 1)[:, None]
    return G, X - G[y]


def moments(v, gcode):
    """per group: n finite, sum, sum of squares of the pair values v."""
    ok = np.isfinite(v) & (gcode >= 0)
    g, x = gcode[ok], v[ok]
    return (np.bincount(g, minlength=NG).astype(float), np.bincount(g, x, NG), np.bincount(g, x * x, NG))


def metrics_for(X, y, n_lab, I, J, gcode):
    """signal / noise / raw correlation of every pair for the samples X (n x cells) with integer labels y."""
    out = {"raw_corr": moments(pair_corr(zscore(X), I, J), gcode)}
    G, R = group_means(X, y, n_lab)
    out["noise_corr"] = moments(pair_corr(zscore(R), I, J), gcode)
    if n_lab >= 3:
        out["signal_corr"] = moments(pair_corr(zscore(G), I, J), gcode)
    else:
        z = np.zeros(NG)
        out["signal_corr"] = (z, z, z)
    return out


def valid_trials(lab, has_frames):
    """Rows (trials) usable for a target in a bin: the cell-independent part of pair_trials."""
    ok = has_frames & np.isfinite(lab)
    if ok.any():
        vals, counts = np.unique(lab[ok], return_counts=True)
        small = vals[counts < cfg.MIN_TRIALS_PER_TARGET_BIN]
        if len(small):
            ok &= ~np.isin(lab, small)
    if ok.sum() < cfg.MIN_TRIALS_ANALYSIS or len(np.unique(lab[ok])) < 2:
        return None
    return ok


# --------------------------------------------------------------------- #
# 3. Per session
# --------------------------------------------------------------------- #
def process_session(ses, seed):
    sid = ses.session_id
    cells = cells_by_session.get(sid)
    if not cells or ses.celldata is None or ses.calciumdata is None or ses.behaviordata is None:
        return []
    celldata = ses.celldata[ses.celldata["cell_id"].isin(cells)].reset_index(drop=True)
    if len(celldata) < 2:
        return []
    pairs = pt.sample_session_pairs(celldata, sid, np.random.default_rng(stable_seed(sid, "pairs3h", base=RANDOM_STATE)),
                                    args.max_pairs)
    if pairs.empty:
        return []
    pairs = add_six_group_column(pairs, area_var="roi_name", label_var="labeled_grouped")
    pairs = pairs[pairs["group"].isin(GCODE)].reset_index(drop=True)
    if pairs.empty:
        return []
    cells = sorted(set(pairs["cell_i"]) | set(pairs["cell_j"]))
    col_of = {c: k for k, c in enumerate(cells)}
    I = pairs["cell_i"].map(col_of).to_numpy()
    J = pairs["cell_j"].map(col_of).to_numpy()
    gcode = pairs["group"].map(GCODE).to_numpy()

    tabs = build_session_tables(ses, mu_sigma_lookup, {"stim": (0.0, 20.0)}, included_cell_ids=set(cells))
    if "stim" not in tabs or tabs["stim"].empty or "signal" not in tabs["stim"].columns:
        return []
    cs = set(cells)
    tt = tabs["stim"]
    meta = pt.add_target_columns(tt[[c for c in tt.columns if c not in cs]], [t for t in TARGETS if t != CHOICE_TARGET])
    td = ses.trialdata
    trials = (td[td["engaged"] == 1] if "engaged" in td.columns else td).reset_index(drop=True)
    cont = build_calcium_continuous(ses.calciumdata[cells], ses.ts_F, ses.behaviordata)
    trials = derive_onset_time(trials, cont)
    ts_all = cont["ts"].to_numpy(float)
    t0, t1 = trials["tStart"].to_numpy(float), trials["tEnd"].to_numpy(float)
    tmeta = trials[["trialNumber"]].merge(meta, on="trialNumber", how="left")
    LAB = {t: tmeta[pt.target_column(t)].to_numpy(float) for t in TARGETS}
    A = cont[cells].to_numpy(float)
    bad = ~np.isfinite(A).all(0)
    if bad.any():           # cells with non-finite samples: their pairs are skipped
        print(f"  [{sid}] WARNING: {int(bad.sum())} cells with non-finite activity, their pairs are skipped")
        keep = ~(bad[I] | bad[J])
        I, J, gcode = I[keep], J[keep], gcode[keep]
        A = np.nan_to_num(A)
    nt = len(trials)
    rows = []
    for axis in AXES:
        ad = AXES_DEF[axis]
        if axis == "position":
            coord, anchor = cont["zpos"].to_numpy(float), trials["stimStart"].to_numpy(float)
        else:
            coord, anchor = ts_all, trials["_onset_time"].to_numpy(float)
        for binning in BINS_RUN:
            base = ad["binsize"] if binning == "fixed" else ad["step"]
            m = 1 if binning == "fixed" else int(round(ad["window"] / ad["step"]))
            fi, tr, bb, centers = pi.frame_bin_index(anchor, coord, ad["pre"], ad["post"], base)
            if len(fi):     # keep a frame only if it lies inside ITS OWN engaged trial's [tStart, tEnd]
                own = (ts_all[fi] >= t0[tr]) & (ts_all[fi] <= t1[tr])
                fi, tr, bb = fi[own], tr[own], bb[own]
            if len(fi) == 0:
                continue
            for k in range(len(centers) - m + 1):
                s_rel = (centers[k] + centers[k + m - 1]) / 2.0
                sel = (bb >= k) & (bb < k + m)
                f_w, t_w = fi[sel], tr[sel]
                if len(f_w) == 0:
                    continue
                cnt = np.bincount(t_w, minlength=nt).astype(float)
                S = sparse.csr_matrix((np.ones(len(t_w)), (t_w, np.arange(len(t_w)))), shape=(nt, len(t_w)))
                Aw = A[f_w]
                TM = (S @ Aw) / np.maximum(cnt, 1)[:, None]            # trial means (nt x cells)
                for target in TARGETS:
                    lab = LAB[target]
                    ok = valid_trials(lab, cnt > 0)
                    if ok is None:
                        continue
                    base_rec = dict(session_id=sid, animal_id=ses.animal_id, binning=binning, axis=axis, target=target,
                                    s_rel=s_rel)
                    # ---- trial_mean: one value per trial
                    _, yt = np.unique(lab[ok], return_inverse=True)
                    res = metrics_for(TM[ok], yt, int(yt.max()) + 1, I, J, gcode)
                    rows.append((dict(base_rec, method="trial_mean", n_trials=int(ok.sum()),
                                      n_frames=int(cnt[ok].sum())), res))
                    # ---- frame_pooled: all frames of the valid trials
                    fm = ok[t_w]
                    if fm.sum() >= MIN_FRAMES_POOLED:
                        _, yf = np.unique(lab[t_w[fm]], return_inverse=True)
                        res = metrics_for(Aw[fm], yf, int(yf.max()) + 1, I, J, gcode)
                        rows.append((dict(base_rec, method="frame_pooled", n_trials=int(ok.sum()),
                                          n_frames=int(fm.sum())), res))
    recs = []                # one record per group
    for base_rec, res in rows:
        for g in range(NG):
            if res["raw_corr"][0][g] == 0 and res["noise_corr"][0][g] == 0:
                continue
            r = dict(base_rec, group=GROUP_KEYS[g])
            for mname in METRICS:
                n, s_, ss = res[mname]
                r[f"{mname}_n"], r[f"{mname}_sum"], r[f"{mname}_ss"] = n[g], s_[g], ss[g]
            recs.append(r)
    print(f"  [{sid}] {len(pairs)} pairs, {len(recs)} rows")
    return recs


records = run_sessions(
    session_ids, process_session, [PROTOCOL], paths.store, "strategy_corr",
    key_params=dict(axes={a: AXES_DEF[a] for a in AXES}, binnings=BINS_RUN, targets=target_signature(TARGETS),
                    min_trials=cfg.MIN_TRIALS_ANALYSIS, min_per_bin=cfg.MIN_TRIALS_PER_TARGET_BIN, version="3h-v4",
                    frames=MIN_FRAMES_POOLED, max_pairs=args.max_pairs, fits_mtime=int(fits_path.stat().st_mtime)),
    load_kwargs=dict(load_behaviordata=True, load_videodata=True, load_celldata=True, load_calciumdata=True),
    cells_by_session=cells_by_session, base_seed=RANDOM_STATE, n_jobs=scope.n_jobs, recompute=args.recompute,
    verbose=JOBLIB_VERBOSE, extra_key_by_session=stamps, flatten=True)
if not records:
    raise SystemExit("No pair had enough valid trials in any bin.")
mom = pd.DataFrame(records)
mom.to_csv(table_dir / "pair_correlation_moments.csv", index=False)
print(f"\n{len(mom)} moment rows; pairs per (session, group): median "
      f"{int(mom.groupby(['session_id', 'group'])['raw_corr_n'].max().median())}")


# --------------------------------------------------------------------- #
# 4. Curves
# --------------------------------------------------------------------- #
KEYS = ["binning", "axis", "method", "target", "group", "s_rel"]


def build_curves(m):
    """Mean and SEM per KEYS: pooled = over pairs (from the moments); session / mouse = over the units' pair means."""
    unit_col = {"pooled": None, "session": "session_id", "mouse": "animal_id"}[AGG]
    cols = [f"{x}_{k}" for x in METRICS for k in ("n", "sum", "ss")]
    if unit_col is None:
        g = m.groupby(KEYS)[cols].sum()
        out = pd.DataFrame(index=g.index)
        for x in METRICS:
            n, s_, ss = g[f"{x}_n"], g[f"{x}_sum"], g[f"{x}_ss"]
            mean = s_ / n.where(n > 0)
            var = (ss / n.where(n > 0) - mean ** 2).clip(lower=0) * n / (n - 1).where(n > 1)
            out[x] = mean
            out[f"{x}_sem"] = np.sqrt(var / n.where(n > 0)).fillna(0.0)
        out["n_pairs"] = g["raw_corr_n"]
        out["n_units"] = g["raw_corr_n"]
        return out.reset_index()
    u = m.groupby(KEYS + [unit_col])[cols].sum().reset_index()
    res = None
    for x in METRICS:
        v = u[f"{x}_sum"] / u[f"{x}_n"].where(u[f"{x}_n"] >= MIN_PAIRS_PER_UNIT)
        d = u[KEYS].assign(v=v, n=u[f"{x}_n"]).dropna(subset=["v"])
        g = d.groupby(KEYS)
        part = pd.DataFrame({x: g["v"].mean(), f"{x}_sem": g["v"].sem().fillna(0.0)})
        if x == "raw_corr":
            part["n_units"] = g.size()
            part["n_pairs"] = g["n"].sum()
        res = part if res is None else res.join(part, how="outer")
    return res.reset_index()


curves = build_curves(mom)
curves.to_csv(table_dir / f"pair_correlations_strategies_by_group{AGG_SUFFIX}.csv", index=False)
agg_note = {"pooled": "mean +- SEM over all pairs", "session": "mean +- SEM over sessions",
            "mouse": "mean +- SEM over mice"}[AGG]
fr = mom.groupby(["binning", "axis", "method"])["n_frames"].median().unstack("method")
print("Median frames per (bin, target) by binning, axis and method:\n" + fr.round(0).to_string())


# --------------------------------------------------------------------- #
# 5. Figures
# --------------------------------------------------------------------- #
def draw_lines(ax, c, value, color, label):
    c = c.sort_values("s_rel").dropna(subset=[value])
    if c.empty:
        return
    ax.plot(c["s_rel"], c[value], color=color, label=label, lw=1.5)
    ax.fill_between(c["s_rel"], c[value] - c[f"{value}_sem"], c[value] + c[f"{value}_sem"], color=color, alpha=0.18, lw=0)


AREAS = []
for g in SIX_GROUP_ORDER:
    if g[0] not in AREAS:
        AREAS.append(g[0])
GROUPS_OF_AREA = {a: [g for g in SIX_GROUP_ORDER if g[0] == a] for a in AREAS}     # V1: lab + unl, PM: lab + unl, AL, RSP
METRIC_TITLE = {"raw_corr": "raw (Pearson) correlation", "noise_corr": "noise correlation",
                "signal_corr": "signal correlation"}


def area_panel(ax, cur, metric, axis, area):
    """One area: one line per project group of the area (V1 / PM: lab and unl)."""
    for g in GROUPS_OF_AREA[area]:
        c = cur[cur["group"] == group_key(g)]
        draw_lines(ax, c, metric, SIX_GROUP_COLORS[g],
                   f"{SIX_GROUP_LABELS[g]} (n={int(c['n_pairs'].max()) if len(c) else 0})")
    if axis == "position":      # the analysis windows of 3a-3c, to compare with the window-based numbers
        for lo, hi in WINDOWS.values():
            ax.axvspan(lo, hi, color="grey", alpha=0.15, lw=0, zorder=0)
    ax.axvline(0, color="black", ls=":", lw=1)
    ax.axhline(0, color="grey", lw=0.7)
    ax.set_xlabel(AXES_DEF[axis]["label"])
    ax.set_ylabel(metric)
    ax.set_title(area, fontsize=10)
    ax.legend(fontsize=6, frameon=False)


def metric_figure(cur, axis, target, method, binning, metrics, name, outdir):
    metrics = [x for x in metrics if cur[x].notna().any()]
    if not metrics:
        return
    fig, axes = plt.subplots(len(metrics), len(AREAS), figsize=(4.2 * len(AREAS), 3.3 * len(metrics)), squeeze=False,
                             sharey="row" if SHARE_Y else False)
    for i, metric in enumerate(metrics):
        for j, area in enumerate(AREAS):
            area_panel(axes[i, j], cur, metric, axis, area)
            if len(metrics) > 1 and i > 0:
                axes[i, j].set_title("")
    ad = AXES_DEF[axis]
    bn = (f"fixed bins {ad['binsize']:g}" if binning == "fixed" else
          f"sliding window {ad['window']:g} step {ad['step']:g}") + (" cm" if axis == "position" else " s")
    fig.suptitle(f"{method}, {bn}: {', '.join(METRIC_TITLE[x] for x in metrics)} over {axis} -- {target} "
                 f"[{' | '.join(target_bin_names(target))}]; {agg_note}", y=1.0, fontsize=9)
    fig.tight_layout()
    save_fig(fig, outdir, name)
    plt.close(fig)


for binning in BINS_RUN:
    for method in METHODS:
        for target in TARGETS:
            tdir = figdir / binning / method / target
            tdir.mkdir(parents=True, exist_ok=True)
            for axis in AXES:
                cur = curves[(curves["binning"] == binning) & (curves["axis"] == axis) & (curves["method"] == method)
                             & (curves["target"] == target)]
                if cur.empty:
                    continue
                metric_figure(cur, axis, target, method, binning, METRICS, f"1_{axis}_correlations", tdir)
                for metric in METRICS:
                    metric_figure(cur, axis, target, method, binning, [metric], f"2_{axis}_{metric.replace('_corr', '')}", tdir)
print(f"\nWrote tables to {table_dir}, figures to {figdir}")
