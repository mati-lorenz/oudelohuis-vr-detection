# -*- coding: utf-8 -*-
"""
3a2_all_pairs_noise_correlations
=================================
NOISE CORRELATIONS OF *ALL* PAIRS of simultaneously recorded cells, from the full cell x cell matrices (matrix products),
instead of 3a's sample of MAX_PAIRS_PER_GROUP pairs. 3a / 3b / 3c are untouched; this step only needs the noise
correlation, so it has no signal correlation and none of the (per-pair, slow) conditioned shuffle tests.

For each session, window (pre_stim / stim / reward) and TARGET (the stimulus binning strategies of stim_binning.py + choice):
    residual   = a cell's window-mean response minus the mean of ITS OWN trials in the same target bin
    noise corr = Pearson correlation of the two cells' residuals, for every pair of included cells of the session
(same definition as 3a's `noise_corr`). All pairs at once: R is trials x cells, the matrix of sums R^T R and the
pairwise-complete corrections (cells with a non-finite response on some trials) give the whole correlation matrix.

Same dataset and trial rules as 3a: 1b-included DN sessions, ENGAGED trials, cells passing 2a's proximity / depth filters and
2b's QC (infotheory.neural_encoding.load_cell_inclusion); trials need a finite label, target bins with < MIN_TRIALS_PER_TARGET_BIN
trials are dropped, a pair needs >= MIN_TRIALS_ANALYSIS shared finite trials.
DIFFERENCE vs 3a (only matters for cells with non-finite responses): 3a computes the bin means and drops small bins on the trials
where BOTH cells of the pair are finite; here a cell's bin means use all of its own finite trials and the small-bin rule counts
trials with a finite label. With no non-finite responses the numbers are identical -- `--check N` compares N random pairs per
(window, target) with 3a's own code (pair_tables.pair_trials) and prints the largest difference.

Pairs: every pair of cells of a session (cross-area pairs included). Each pair is assigned to the SIX project groups of its two
cells, so the output is a 6 x 6 table of group pairs; the six bars of the figures are the DIAGONAL (both cells in the same group,
as in 3a). Off-diagonal pairs (e.g. V1-lab x V1-unl, V1 x PM) are in the tables.

Project-wide rules (infotheory/project_config.py) -- what applies here
    * selection : --animals / --sessions restrict the sessions; a restricted run writes to out/selection/<tag>/.
    * aggregate : --aggregate pooled|session|mouse. pooled = all pairs of a group together (mean over pairs), no error bars;
                  session = each session's mean over its pairs, mouse = a mouse's sessions' pairs pooled first; bar = mean +- SEM over
                  units, dots = units (units with fewer than MIN_PAIRS_PER_UNIT pairs in the group are left out). Always the six groups.
    * memory    : one session in RAM per worker (--n-jobs); only small summaries (counts, sums, histograms) leave the worker and are
                  cached per session in store/. A session's cell x cell matrix is trials x cells at most (hundreds of cells).
                  --save-matrices writes the full matrices to store/ (float32 .npy + cell table) for later steps.
    * binning   : targets / windows / strategies as 3a.

Outputs
-------
2_pipeline/3a2_all_pairs_noise_correlations/
    out/    noise_corr_per_session.csv            per (session, window, target, group pair): n_pairs, mean, SD of the noise correlation
            noise_corr_by_group_pair[__<aggregate>].csv   per (group pair, window, target) x unit: n_pairs, mean noise correlation
            noise_corr_check_vs_3a.csv            (with --check) largest |difference| to 3a's own computation
            figures/<target>/1_noise_correlation_by_group.png      the six bars per window (same-group pairs)
            figures/<target>/2_noise_correlation_distribution.png  distribution of the noise correlation, all pairs of the group
            figures/compare_targets_noise_corr.png
    out/selection/<tag>/...                       the same for a restricted run
    store/  noise_corr_all_<hash>.pkl             per-session summaries
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from infotheory import project_config as cfg
from infotheory.pipeline import get_pipeline_paths
from infotheory.session import discover_sessions
from infotheory.neural_encoding import load_cell_inclusion, build_session_tables
from infotheory.celldata_utils import SIX_GROUP_ORDER, SIX_GROUP_LABELS, SIX_GROUP_COLORS
from infotheory.groups import group_key, add_six_group_column
from infotheory.plotting import set_style, save_fig, clear_figures
from infotheory.scope import add_scope_args, scope_from_args, select_sessions, scoped_dirs, data_stamp, unit_noun
from infotheory.session_jobs import run_sessions_units
from infotheory.unit_stats import stable_seed
from infotheory.unit_plots import group_panels, mouse_palette
from infotheory.stim_binning import (STIM_BINNING_STRATEGIES, DEFAULT_STRATEGIES, CHOICE_TARGET, target_signature,
                                     target_bin_names, describe_strategies)
from infotheory import pair_tables as pt

PROTOCOL = "DN"
WINDOWS = {"pre_stim": (-30.0, -10.0), "stim": (0.0, 20.0), "reward": (25.0, 45.0)}   # as 3a

MIN_PAIRS_PER_UNIT = 3      # session / mouse aggregate: pairs needed in a group for a unit's value to be used
BINNING_STRATEGIES = list(DEFAULT_STRATEGIES)
INCLUDE_CHOICE_TARGET = True
SHARE_Y = True
HIST_BINS = np.linspace(-1.0, 1.0, 81)      # distribution of the noise correlation (bins of 0.025)
RANDOM_STATE = 0
JOBLIB_VERBOSE = 5

parser = argparse.ArgumentParser()
add_scope_args(parser)
parser.add_argument("--strategies", nargs="+", default=None, choices=sorted(STIM_BINNING_STRATEGIES),
                    help=f"stimulus binning strategies (default {BINNING_STRATEGIES})")
parser.add_argument("--check", type=int, default=0, help="compare N random pairs per (window, target) with 3a's computation")
parser.add_argument("--save-matrices", action="store_true", help="also write each session's full noise-correlation matrices to store/")
args = parser.parse_args()
scope = scope_from_args(args)
AGG = scope.aggregate
UNIT = unit_noun(AGG)
UNITS = "mice" if UNIT == "mouse" else f"{UNIT}s"
AGG_SUFFIX = "" if AGG == "pooled" else f"__{AGG}"
STRATEGIES = args.strategies or BINNING_STRATEGIES
TARGETS = STRATEGIES + ([CHOICE_TARGET] if INCLUDE_CHOICE_TARGET else [])
N_CHECK = args.check

paths = get_pipeline_paths(__file__)
table_dir, figdir = scoped_dirs(paths, scope)
clear_figures(figdir)
for _sub in figdir.glob("*/"):
    clear_figures(_sub)
set_style()
print(scope.describe())
print("Targets (stimulus binning strategies + choice):\n" + describe_strategies(TARGETS))

# --------------------------------------------------------------------- #
# 1. Inputs (as 3a)
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
MIN_TRIALS, MIN_PER_BIN = cfg.MIN_TRIALS_ANALYSIS, cfg.MIN_TRIALS_PER_TARGET_BIN


# --------------------------------------------------------------------- #
# 2. The all-pairs noise-correlation matrix
# --------------------------------------------------------------------- #
def noise_corr_matrix(X, lab):
    """Noise-correlation matrix (cells x cells, NaN diagonal / undefined pairs) of the trials x cells response matrix X
    for the trial labels `lab` (target bins). None when the trials do not qualify (see the module docstring)."""
    ok = np.isfinite(lab)
    if ok.any():
        vals, counts = np.unique(lab[ok], return_counts=True)
        small = vals[counts < MIN_PER_BIN]
        if len(small):
            ok &= ~np.isin(lab, small)
    if ok.sum() < MIN_TRIALS or len(np.unique(lab[ok])) < 2:
        return None
    Xo, lo = X[ok], lab[ok]
    M = np.isfinite(Xo)
    R = np.zeros_like(Xo)
    for v in np.unique(lo):                                   # residual = response - this cell's mean in the bin
        idx = lo == v
        cnt = M[idx].sum(0)
        mu = np.where(cnt > 0, np.where(M[idx], Xo[idx], 0.0).sum(0) / np.maximum(cnt, 1), 0.0)
        R[idx] = np.where(M[idx], Xo[idx] - mu, 0.0)
    Mf = M.astype(float)
    n = Mf.T @ Mf                                             # shared finite trials per pair
    Sx = R.T @ Mf                                             # [i, j] = sum over shared trials of R_i
    Sxy = R.T @ R                                             # zero-filled, so only shared trials contribute
    Sxx = (R ** 2).T @ Mf                                     # [i, j] = sum over shared trials of R_i^2
    with np.errstate(divide="ignore", invalid="ignore"):
        mx = Sx / n
        my = mx.T
        cov = Sxy / n - mx * my
        vx = Sxx / n - mx ** 2
        vy = vx.T
        C = cov / np.sqrt(vx * vy)
    C[(n < MIN_TRIALS) | (vx <= 1e-12) | (vy <= 1e-12)] = np.nan
    np.fill_diagonal(C, np.nan)
    return np.clip(C, -1.0, 1.0)


def noise_corr_3a(table, target, ci, cj):
    """3a's own computation for one pair (pair_trials + residuals), for --check."""
    d = pt.pair_trials(table, target, ci, cj)
    if d is None:
        return np.nan
    xi, xj, sb = d["x1"], d["x2"], d["label"]
    ri = xi - pd.Series(xi).groupby(sb).transform("mean").to_numpy()
    rj = xj - pd.Series(xj).groupby(sb).transform("mean").to_numpy()
    return np.nan if ri.std() == 0 or rj.std() == 0 else float(np.corrcoef(ri, rj)[0, 1])


# --------------------------------------------------------------------- #
# 3. Per session (one in RAM at a time): all pairs -> small summaries
# --------------------------------------------------------------------- #
def process_session(ses, seed, new_cells, new_pairs, req_cells):
    """Noise correlation of the NEW pairs only (session_jobs.run_sessions_units); one row per pair, window, target."""
    sid = ses.session_id
    cells = {c for q in new_pairs for c in q}
    if not cells or ses.celldata is None:
        return None
    rng = np.random.default_rng(seed)
    tables = build_session_tables(ses, mu_sigma_lookup, WINDOWS, included_cell_ids=cells)
    celldata = ses.celldata[ses.celldata["cell_id"].isin(cells)].reset_index(drop=True)
    celldata = add_six_group_column(celldata, area_var="roi_name", label_var="labeled")
    prows, checks = [], []
    for window, table in tables.items():
        if table.empty or "signal" not in table.columns:
            continue
        cell_ids = [c for c in celldata["cell_id"] if c in table.columns]
        if len(cell_ids) < 2:
            continue
        pos = {c: i for i, c in enumerate(cell_ids)}
        pr = [q for q in new_pairs if q[0] in pos and q[1] in pos]
        if not pr:
            continue
        pi = np.array([pos[q[0]] for q in pr])
        pj = np.array([pos[q[1]] for q in pr])
        grp = celldata.set_index("cell_id").loc[cell_ids, "group"].to_numpy()
        X = table[cell_ids].to_numpy(float)
        table = pt.add_target_columns(table, STRATEGIES)
        ga, gb = grp[pi], grp[pj]
        lo_g, hi_g = np.where(ga <= gb, ga, gb), np.where(ga <= gb, gb, ga)
        for target in TARGETS:
            lab = table[pt.target_column(target)].to_numpy(float) if pt.target_column(target) in table.columns else None
            if lab is None:
                continue
            C = noise_corr_matrix(X, lab)
            if C is None:
                continue
            if args.save_matrices:        # only the cells of the pairs computed in THIS call
                np.save(paths.store / f"noise_corr_matrix__{sid}__{window}__{target}.npy", C.astype(np.float32))
                pd.DataFrame({"cell_id": cell_ids, "group": grp}).to_csv(
                    paths.store / f"noise_corr_cells__{sid}__{window}.csv", index=False)
            r = C[pi, pj]
            ok = np.isfinite(r)
            prows.append(pd.DataFrame({"session_id": sid, "window": window, "target": target,
                                       "cell_i": [pr[k][0] for k in np.flatnonzero(ok)],
                                       "cell_j": [pr[k][1] for k in np.flatnonzero(ok)],
                                       "group_a": lo_g[ok], "group_b": hi_g[ok], "r": r[ok]}))
            if N_CHECK:
                pick = rng.choice(np.flatnonzero(ok), size=min(N_CHECK, int(ok.sum())), replace=False) if ok.any() else []
                diffs = []
                for k in pick:
                    ref = noise_corr_3a(table, target, cell_ids[pi[k]], cell_ids[pj[k]])
                    if np.isfinite(ref):
                        diffs.append(abs(ref - r[k]))
                if diffs:
                    checks.append(dict(session_id=sid, window=window, target=target, n_checked=len(diffs),
                                       max_abs_diff=max(diffs)))
    print(f"  [{sid}] {len(new_pairs)} new pairs among {len(cells)} cells")
    return dict(session_id=sid, animal_id=ses.animal_id,
                pairs=pd.concat(prows, ignore_index=True) if prows else pd.DataFrame(), checks=checks)


results = run_sessions_units(
    session_ids, process_session, [PROTOCOL], paths.store, "noise_corr_all",
    key_params=dict(windows=WINDOWS, targets=target_signature(TARGETS), min_trials=MIN_TRIALS, min_per_bin=MIN_PER_BIN,
                    hist_bins=len(HIST_BINS), check=N_CHECK, save_matrices=args.save_matrices, version="3a2-v2", face_z=cfg.NORMALIZE_FACE_PER_SESSION,
                    fits_mtime=int(fits_path.stat().st_mtime)),
    load_kwargs=dict(load_behaviordata=True, load_videodata=True, load_celldata=True, load_calciumdata=True),
    cells_by_session=cells_by_session, parts={"pairs": "pair"}, extras={"checks": "extend"},
    groups_by_cell=cell_group, max_cells=scope.max_cells, max_pairs=scope.max_pairs, base_seed=RANDOM_STATE, n_jobs=scope.n_jobs,
    recompute=args.recompute, verbose=JOBLIB_VERBOSE, extra_key_by_session=stamps)
results = [r for r in results if r is not None]
# group-pair summaries + histograms of the REQUESTED pairs, rebuilt from the per-pair rows (ledger: results per pair)
for r in results:
    r["rows"], r["hists"] = [], {}
    r.setdefault("checks", [])
    bp = r.get("pairs")
    if bp is None or not len(bp):
        continue
    for (w_, t_, a_, b_), g_ in bp.groupby(["window", "target", "group_a", "group_b"]):
        r["rows"].append(dict(session_id=r["session_id"], animal_id=r["animal_id"], window=w_, target=t_, group_a=a_,
                              group_b=b_, n_pairs=len(g_), sum_r=float(g_["r"].sum()), sum_r2=float((g_["r"] ** 2).sum())))
        if a_ == b_:
            r["hists"][(w_, t_, a_)] = np.histogram(g_["r"], HIST_BINS)[0]
rows = [x for r in results for x in r["rows"]]
if not rows:
    raise SystemExit("No session had enough valid trials.")
per_session = pd.DataFrame(rows)
per_session["mean_r"] = per_session["sum_r"] / per_session["n_pairs"]
per_session["sd_r"] = np.sqrt(np.maximum(per_session["sum_r2"] / per_session["n_pairs"] - per_session["mean_r"] ** 2, 0.0))
per_session.drop(columns=["sum_r", "sum_r2"]).to_csv(table_dir / "noise_corr_per_session.csv", index=False)
if N_CHECK:
    chk = pd.DataFrame([c for r in results for c in r["checks"]])
    chk.to_csv(table_dir / "noise_corr_check_vs_3a.csv", index=False)
    print(f"\nCheck vs 3a's own computation: {int(chk['n_checked'].sum())} pairs, largest |difference| = "
          f"{chk['max_abs_diff'].max():.2e}" if len(chk) else "\nCheck: no pair could be compared")
print(f"\n{int(per_session.loc[(per_session['window'] == next(iter(WINDOWS))) & (per_session['target'] == TARGETS[0]), 'n_pairs'].sum())} "
      f"pairs in total ({TARGETS[0]}, {next(iter(WINDOWS))} window)")

# --------------------------------------------------------------------- #
# 4. Units: pooled / session / mouse (pair-weighted), the six same-group bars
# --------------------------------------------------------------------- #
UNIT_KEY = {"pooled": [], "session": ["session_id"], "mouse": ["animal_id"]}[AGG]
u = (per_session.groupby(["group_a", "group_b", "window", "target"] + UNIT_KEY)[["n_pairs", "sum_r"]].sum().reset_index())
u["noise_corr"] = u["sum_r"] / u["n_pairs"]
if AGG != "pooled":
    u = u[u["n_pairs"] >= MIN_PAIRS_PER_UNIT]
    if AGG == "mouse":
        u["session_id"] = u["animal_id"]
    else:
        u["animal_id"] = u["session_id"].str.split("_").str[0]
else:
    u["session_id"], u["animal_id"] = "all", ""
u.drop(columns="sum_r").to_csv(table_dir / f"noise_corr_by_group_pair{AGG_SUFFIX}.csv", index=False)

same = u[u["group_a"] == u["group_b"]].rename(columns={"group_a": "group"})
agg_note = {"pooled": "all pairs pooled (no error bars)", "session": "bar = mean +- SEM over sessions, dots = sessions",
            "mouse": "bar = mean +- SEM over mice, dots = mice"}[AGG]
WIN = list(WINDOWS)
hist_all = {}
for r in results:
    for k, h in r["hists"].items():
        hist_all[k] = hist_all.get(k, 0) + h

for target in TARGETS:
    s_t = same[same["target"] == target]
    if s_t.empty:
        print(f"\n(no pairs for target {target!r})")
        continue
    tag = f"{target} [{' | '.join(target_bin_names(target))}]"
    tdir = figdir / target
    tdir.mkdir(parents=True, exist_ok=True)
    print(f"\n======== target: {tag} ========")
    print(f"Noise correlation by (window, group), same-group pairs{'' if AGG == 'pooled' else f', mean over {UNITS}'}:")
    print(s_t.groupby(["window", "group"])[["n_pairs", "noise_corr"]].agg({"n_pairs": "sum", "noise_corr": "mean"})
          .round(4).to_string())
    group_panels(s_t, "window", WIN, "noise_corr", AGG, tdir, "1_noise_correlation_by_group",
                 f"Noise correlation, ALL pairs -- {tag}; {agg_note}", ylabel="noise correlation",
                 mouse_colors=mouse_colors, min_cells_per_unit=1, ncols=len(WIN), panel_size=(4.6, 3.6), sharey=SHARE_Y)

    fig, axes = plt.subplots(1, len(WIN), figsize=(5 * len(WIN), 4), squeeze=False, sharex=True, sharey=SHARE_Y)
    mid = 0.5 * (HIST_BINS[:-1] + HIST_BINS[1:])
    for ax, w in zip(axes[0], WIN):
        for g in SIX_GROUP_ORDER:
            h = hist_all.get((w, target, group_key(g)))
            if h is None or h.sum() == 0:
                continue
            ax.plot(mid, h / h.sum() / np.diff(HIST_BINS), color=SIX_GROUP_COLORS[g], label=f"{SIX_GROUP_LABELS[g]} (n={int(h.sum())})")
        ax.axvline(0, color="grey", linewidth=0.7)
        ax.set_xlabel("noise correlation")
        ax.set_ylabel("density")
        ax.set_title(w)
        ax.legend(fontsize=6, frameon=False)
    fig.suptitle(f"Distribution of the noise correlation, all same-group pairs pooled (not aggregated) -- {PROTOCOL}, {tag}", y=1.03)
    fig.tight_layout()
    save_fig(fig, tdir, "2_noise_correlation_distribution")
    plt.close(fig)

group_panels(same, "window", WIN, "noise_corr", AGG, figdir, "compare_targets_noise_corr",
             f"Mean noise correlation, all pairs: stimulus binning strategies vs. choice; {agg_note}",
             ylabel="mean noise correlation", mouse_colors=mouse_colors, min_cells_per_unit=1, ncols=len(WIN),
             panel_size=(4.6, 3.6), sharey=SHARE_Y, series_col="target", series_order=TARGETS)

print(f"\nWrote tables to {table_dir}, figures to {figdir}")
