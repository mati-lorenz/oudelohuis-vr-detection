# -*- coding: utf-8 -*-
"""
3g_pair_hit_miss_correlations
=============================
Do pairs of cells fluctuate together differently on trials the animal detects (HIT: stimulus present, lick) and trials it
misses (MISS: stimulus present, no lick)? Following the logic of Safaai et al. (2025): noise correlations of pairs
within a projection class, split by behavioural outcome, with the controls that make the comparison fair.

DEFINITIONS
    trials      engaged trials with a stimulus (signal > 0) and a decision; HIT = lick, MISS = no lick (catch trials are
                not used). Window means as everywhere (WINDOWS: pre_stim as a control for a state difference that
                exists before the stimulus, stim as the question).
    noise corr  correlation of the two cells' residuals after removing the mean of each (stimulus bin, outcome) group,
                the stimulus bins being one `stim__<strategy>` binning (--strategy; default pct_0-20_20-80_80-100).
    conditioning  "stimulus" = the above; "stimulus+movement" = first regress each cell on running speed, pupil area and
                motion energy (window means, linear, fitted on the pair's trials of the session; lick rate is NOT used
                -- it IS the choice). Both conditionings use the same trials. Sessions without video: stimulus only.
    delta       r_hit - r_miss, per pair.
CONTROLS
    * trial matching  correct trials outnumber errors ~4:1 and a correlation from fewer trials is more variable and
                biased towards 0 by the residual step, so hits and misses are matched IN COUNT AND IN STIMULUS MIX: per
                stimulus bin, k = min(n_hit, n_miss) trials are drawn from each class (strata with k < 2 dropped), the
                correlation is computed on the drawn trials and averaged over --draws random draws. Pairs with fewer
                than MIN_MATCHED trials per class after matching are left out. The unmatched correlations
                (r_hit_unmatched, ...) are kept for comparison (figure 3).
    * distance  nearby cells are more correlated, and projection classes may differ in distance (see
                infotheory.pair_distance): within every (session, area) the labelled pairs are matched to unlabelled pairs
                of the same session and area at a similar distance (caliper --caliper microns, no replacement).
                Figure 4 compares all pairs with the distance-matched subset; AL / RSP have no labelled class and are
                left as they are. Distance = 3-D (xloc, yloc, depth).
    The pairs are sampled here (MAX_PAIRS_PER_GROUP per (session, area, label)), a much larger budget than 3a's (8), because
    the hit/miss contrast needs many pairs; they are not 3a's list (their noise correlations are not directly
    comparable with 3a's: trials differ -- stimulus present only, engaged -- and hits/misses are matched).

Project-wide rules (infotheory/project_config.py) -- what applies here
    * selection : --animals / --sessions restrict the sessions; a restricted run writes to out/selection/<tag>/.
    * aggregate : --aggregate pooled|session|mouse (pooled = all pairs of a group, no error bars; session / mouse = mean
                  over the unit's pairs first, units with < MIN_PAIRS_PER_UNIT pairs left out, bar = mean +- SEM over
                  units, dots = units by mouse). Always the six groups.
    * memory    : one session in RAM per worker (--n-jobs), cached per session in store/.
    * binning   : same trial rule (infotheory.pair_tables.pair_trials-style finite checks) and same stimulus bins as
                  3a-3c; the noise correlation here is 3a's definition applied to hit and miss trials separately.

CAVEAT  expect small effects; a difference between classes can come from the number of cells per area, distance, and
state; read figure 4 and the pre_stim panels before interpreting a stim-window difference.

Outputs
-------
2_pipeline/3g_pair_hit_miss_correlations/
    out/    hit_miss_pair_correlations.csv        one row per (session, window, conditioning, pair)
            hit_miss_by_group[__<aggregate>].csv  per group x window x conditioning x subset (all | distance-matched) (x unit)
            distance_matching.csv                 per (session, area, window, conditioning): distances before / after matching
            figures/<conditioning>/*.png, figures/compare_conditioning_delta.png
    out/selection/<tag>/...                       the same for a restricted run
    store/  hit_miss_<hash>.pkl                   per-session caches
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from infotheory import project_config as cfg
from infotheory.pipeline import get_pipeline_paths
from infotheory.session import discover_sessions
from infotheory.neural_encoding import build_session_tables, load_cell_inclusion
from infotheory.celldata_utils import SIX_GROUP_ORDER, SIX_GROUP_LABELS, SIX_GROUP_COLORS
from infotheory.groups import group_key
from infotheory.plotting import set_style, save_fig, clear_figures
from infotheory.scope import add_scope_args, scope_from_args, select_sessions, scoped_dirs, data_stamp, unit_noun
from infotheory.session_jobs import run_sessions
from infotheory.unit_stats import stable_seed
from infotheory.unit_plots import group_panels, mouse_palette
from infotheory.stim_binning import STIM_BINNING_STRATEGIES, target_bin_names
from infotheory import cell_tables as ct
from infotheory import pair_tables as pt
from infotheory import pair_distance as pd_
from infotheory import pair_temporal as ptt

PROTOCOL = "DN"
WINDOWS = {"pre_stim": (-30.0, -10.0), "stim": (0.0, 20.0)}
STRATEGY = "pct_0-20_20-80_80-100"
MAX_PAIRS_PER_GROUP = 150   # per (session, area, label)
N_DRAWS = 20                # random trial-matching draws averaged per pair
MIN_MATCHED = 8             # trials per class after matching
CALIPER_UM = 20.0           # distance-matching tolerance
MIN_PAIRS_PER_UNIT = 5
RANDOM_STATE = 0
JOBLIB_VERBOSE = 5
SHARE_Y = True              # the window panels of a group figure get the same y range
MOVEMENT_COLUMNS = ["cur_runspeed", "cur_pupil_area", "cur_motionenergy"]
CONDITIONINGS = ["stimulus", "stimulus+movement"]
DIST_EDGES = [0, 50, 100, 150, 250, 400, np.inf]
DIST_MIDS = [25, 75, 125, 200, 325, 500]

parser = argparse.ArgumentParser()
add_scope_args(parser)
parser.add_argument("--strategy", default=STRATEGY, choices=sorted(STIM_BINNING_STRATEGIES))
parser.add_argument("--max-pairs", type=int, default=MAX_PAIRS_PER_GROUP, help="pairs per (session, area, label)")
parser.add_argument("--draws", type=int, default=N_DRAWS)
parser.add_argument("--caliper", type=float, default=CALIPER_UM, help="distance-matching tolerance (microns)")
args = parser.parse_args()
scope = scope_from_args(args)
AGG = scope.aggregate
UNIT = unit_noun(AGG)
AGG_SUFFIX = "" if AGG == "pooled" else f"__{AGG}"
STIM_COL = ct.stim_predictor(args.strategy)

paths = get_pipeline_paths(__file__)
table_dir, figdir = scoped_dirs(paths, scope)
clear_figures(figdir)
for _sub in figdir.glob("*/"):
    clear_figures(_sub)
set_style()
print(scope.describe())
print(f"Stimulus bins ({args.strategy}): {' | '.join(target_bin_names(args.strategy))}")

# --------------------------------------------------------------------- #
# 1. Inputs
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
    raise SystemExit("No selected session passed the inclusion criteria.")
mouse_colors = mouse_palette(a for _, a, _ in all_triples)
inclusion = load_cell_inclusion(cell_dist_out, activity_out)
cells_by_session = {sid: set(sub.loc[sub["include"], "cell_id"]) for sid, sub in inclusion.groupby("session_id")
                    if sid in session_ids}
session_ids = [s for s in session_ids if cells_by_session.get(s)]
stamps = {s: data_stamp(*selected[s]) for s in session_ids}
print(f"{len(session_ids)} sessions")


# --------------------------------------------------------------------- #
# 2. Per session
# --------------------------------------------------------------------- #
def corr(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def residual(x, strata):
    """x minus the mean of its stratum."""
    s = np.asarray(strata)
    n = np.bincount(s)
    m = np.bincount(s, weights=x) / np.maximum(n, 1)
    return x - m[s]


def hit_miss_noise_corr(x1, x2, stim, hit, rng, draws, min_matched):
    """Noise correlations of hit and miss trials, stimulus-conditioned: matched in count and stimulus mix (mean over
    `draws` draws) and unmatched (all trials of each class). See the module docstring."""
    codes = pd.factorize(stim)[0]
    out = dict(n_hit=int(hit.sum()), n_miss=int((~hit).sum()), n_matched=0, r_hit=np.nan, r_miss=np.nan,
               r_hit_unmatched=np.nan, r_miss_unmatched=np.nan)
    for name, cls in (("hit", hit), ("miss", ~hit)):
        idx = np.flatnonzero(cls)
        if len(idx) >= min_matched:
            c = codes[idx]
            keep = np.bincount(c, minlength=codes.max() + 1)[c] >= 2
            if keep.sum() >= min_matched:
                i = idx[keep]
                out[f"r_{name}_unmatched"] = corr(residual(x1[i], pd.factorize(codes[i])[0]),
                                                  residual(x2[i], pd.factorize(codes[i])[0]))
    per = []
    for b in np.unique(codes):
        ih, im = np.flatnonzero(hit & (codes == b)), np.flatnonzero(~hit & (codes == b))
        k = min(len(ih), len(im))
        if k >= 2:
            per.append((ih, im, k))
    n_m = sum(k for _, _, k in per)
    if n_m < min_matched:
        return out
    out["n_matched"] = n_m
    strata = np.concatenate([np.full(k, j) for j, (_, _, k) in enumerate(per)])
    rh, rm = [], []
    for _ in range(draws):
        sh = np.concatenate([rng.choice(ih, k, replace=False) for ih, _, k in per])
        sm = np.concatenate([rng.choice(im, k, replace=False) for _, im, k in per])
        rh.append(corr(residual(x1[sh], strata), residual(x2[sh], strata)))
        rm.append(corr(residual(x1[sm], strata), residual(x2[sm], strata)))
    out["r_hit"], out["r_miss"] = float(np.nanmean(rh)), float(np.nanmean(rm))
    return out


def movement_residuals(sub, cells, mv):
    """Copy of `sub` where every cell column has the linear fit on the movement columns removed."""
    Z = sub[mv].to_numpy(float)
    Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1.0)
    X = np.column_stack([np.ones(len(Z)), Z])
    out = sub.copy()
    for c in cells:
        y = sub[c].to_numpy(float)
        ok = np.isfinite(y)
        if ok.sum() > X.shape[1] + 2:
            beta = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
            out[c] = y - X @ beta
    return out


def process_session(ses, seed):
    sid = ses.session_id
    cells = cells_by_session.get(sid)
    if not cells or ses.celldata is None:
        return []
    rng = np.random.default_rng(seed)
    celldata = ses.celldata[ses.celldata["cell_id"].isin(cells)].reset_index(drop=True)
    if len(celldata) < 2:
        return []
    pairs = pt.sample_session_pairs(celldata, sid, np.random.default_rng(stable_seed(sid, "pairs3g", base=RANDOM_STATE)),
                                    args.max_pairs)
    if pairs.empty:
        return []
    pairs["distance_um"] = pd_.pair_distances(celldata, pairs["cell_i"], pairs["cell_j"])
    need = set(pairs["cell_i"]) | set(pairs["cell_j"])
    tables = build_session_tables(ses, mu_sigma_lookup, WINDOWS, included_cell_ids=need)
    records = []
    for window, table in tables.items():
        if table.empty or "signal" not in table.columns:
            continue
        table = pt.add_target_columns(table, [args.strategy])
        sub = table[(table["signal"] > 0) & table["choice"].isin([0, 1]) & table[STIM_COL].notna()].reset_index(drop=True)
        mv = [c for c in MOVEMENT_COLUMNS if c in sub.columns and sub[c].notna().any()]
        if mv:
            sub = sub[np.isfinite(sub[mv].to_numpy(float)).all(axis=1)].reset_index(drop=True)
        if len(sub) < cfg.MIN_TRIALS_ANALYSIS:
            continue
        variants = {"stimulus": sub}
        if mv:
            variants["stimulus+movement"] = movement_residuals(sub, need, mv)
        for cond, tab in variants.items():
            stim, hit = tab[STIM_COL].to_numpy(float), tab["choice"].to_numpy(float) == 1
            for r in pairs.itertuples(index=False):
                if r.cell_i not in tab.columns or r.cell_j not in tab.columns:
                    continue
                x1, x2 = tab[r.cell_i].to_numpy(float), tab[r.cell_j].to_numpy(float)
                ok = np.isfinite(x1) & np.isfinite(x2)
                if ok.sum() < cfg.MIN_TRIALS_ANALYSIS:
                    continue
                res = hit_miss_noise_corr(x1[ok], x2[ok], stim[ok], hit[ok], rng, args.draws, MIN_MATCHED)
                records.append(dict(session_id=sid, animal_id=ses.animal_id, window=window, conditioning=cond,
                                    roi_name=r.roi_name, labeled_grouped=r.labeled_grouped, cell_i=r.cell_i,
                                    cell_j=r.cell_j, pair_rank=r.pair_rank, distance_um=r.distance_um, **res))
    print(f"  [{sid}] {len(pairs)} pairs, {len(records)} (window, conditioning, pair) rows")
    return records


records = run_sessions(
    session_ids, process_session, [PROTOCOL], paths.store, "hit_miss",
    key_params=dict(windows=WINDOWS, strategy=args.strategy, max_pairs=args.max_pairs, draws=args.draws,
                    min_matched=MIN_MATCHED, movement=MOVEMENT_COLUMNS, version="3g-v1",
                    min_trials=cfg.MIN_TRIALS_ANALYSIS, fits_mtime=int(fits_path.stat().st_mtime)),
    load_kwargs=dict(load_behaviordata=True, load_videodata=True, load_celldata=True, load_calciumdata=True),
    cells_by_session={s: cells_by_session[s] for s in session_ids}, base_seed=RANDOM_STATE, n_jobs=scope.n_jobs,
    recompute=args.recompute, verbose=JOBLIB_VERBOSE, extra_key_by_session=stamps, flatten=True)
if not records:
    raise SystemExit("No pair had enough hit and miss trials.")
table = pt.finalize_pair_table(pd.DataFrame(records))
table["delta"] = table["r_hit"] - table["r_miss"]
table["delta_unmatched"] = table["r_hit_unmatched"] - table["r_miss_unmatched"]
table["r_mean"] = (table["r_hit"] + table["r_miss"]) / 2
print(f"\n{len(table)} (session, window, conditioning, pair) rows; "
      f"{int(table['delta'].notna().sum())} with enough matched hit and miss trials")

# distance matching (within session x area, per window and conditioning, on the pairs that have a valid delta)
valid = table.dropna(subset=["delta"]).copy()
valid["dm_matched"] = False
dm_rows = []
for (w, cond), blk in valid.groupby(["window", "conditioning"]):
    flag = pd_.flag_distance_matched(blk, args.caliper, lambda k, w=w, cond=cond: stable_seed(*k, w, cond, "dm",
                                                                                              base=RANDOM_STATE))
    valid.loc[blk.index, "dm_matched"] = flag
    for (sid, roi), b in blk.groupby(["session_id", "roi_name"]):
        t, p = b[b["labeled_grouped"] == "lab"], b[b["labeled_grouped"] == "unl"]
        if t.empty or p.empty:
            continue
        m = flag.loc[b.index]
        dm_rows.append(dict(session_id=sid, roi_name=roi, window=w, conditioning=cond, n_lab=len(t), n_unl=len(p),
                            n_lab_matched=int(m[t.index].sum()), n_unl_matched=int(m[p.index].sum()),
                            median_dist_lab=t["distance_um"].median(), median_dist_unl_before=p["distance_um"].median(),
                            median_dist_unl_after=p.loc[m[p.index], "distance_um"].median()))
valid["distance_matched"] = valid["dm_matched"]
table = table.merge(valid[["session_id", "window", "conditioning", "cell_i", "cell_j", "dm_matched"]],
                    on=["session_id", "window", "conditioning", "cell_i", "cell_j"], how="left")
table.to_csv(table_dir / "hit_miss_pair_correlations.csv", index=False)
dm = pd.DataFrame(dm_rows)
dm.to_csv(table_dir / "distance_matching.csv", index=False)
if len(dm):
    s = dm[(dm["window"] == "stim") & (dm["conditioning"] == "stimulus")]
    print(f"\nDistance matching (stim window, stimulus conditioning): median distance of labelled pairs "
          f"{s['median_dist_lab'].median():.0f} um, of unlabelled pairs {s['median_dist_unl_before'].median():.0f} um "
          f"before and {s['median_dist_unl_after'].median():.0f} um after matching; "
          f"{int(s['n_lab_matched'].sum())}/{int(s['n_lab'].sum())} labelled pairs matched (caliper {args.caliper:g} um)")

VALUES = ["r_hit", "r_miss", "delta", "r_hit_unmatched", "r_miss_unmatched", "delta_unmatched", "r_mean", "distance_um"]
subsets = {"all pairs": valid, "distance-matched": valid[valid["dm_matched"]]}
by_group = pd.concat([ct.unit_group_summary(d, ["window", "conditioning"], AGG, VALUES).assign(subset=name)
                      for name, d in subsets.items()], ignore_index=True).rename(columns={"n_cells": "n_pairs"})
by_group.to_csv(table_dir / f"hit_miss_by_group{AGG_SUFFIX}.csv", index=False)

agg_note = {"pooled": "all pairs pooled (no error bars)", "session": "bar = mean +- SEM over sessions, dots = sessions",
            "mouse": "bar = mean +- SEM over mice, dots = mice"}[AGG]
WIN = list(WINDOWS)


def panels(df, name, value, ylabel, title, outdir, **kw):
    kw.setdefault("sharey", SHARE_Y)
    group_panels(df, "window", WIN, value, AGG, outdir, name, title, ylabel=ylabel, mouse_colors=mouse_colors,
                 min_cells_per_unit=MIN_PAIRS_PER_UNIT, ncols=len(WIN), panel_size=(5.2, 3.8), **kw)


# --------------------------------------------------------------------- #
# 3. Reports and figures, one set per conditioning
# --------------------------------------------------------------------- #
for cond in CONDITIONINGS:
    v = valid[valid["conditioning"] == cond]
    if v.empty:
        print(f"\n(no pairs for conditioning {cond!r})")
        continue
    cdir = figdir / cond.replace("+", "_plus_")
    cdir.mkdir(parents=True, exist_ok=True)
    print(f"\n======== conditioning: {cond} ========")
    per = ct.unit_group_summary(v, ["window"], AGG, ["r_hit", "r_miss", "delta"])
    print(f"Mean noise correlation, hit / miss / delta by (window, group){'' if AGG == 'pooled' else f', mean over {UNIT}s'}:")
    print(per.groupby(["window", "group"])[["r_hit", "r_miss", "delta"]].mean().round(4).to_string())
    long = pd.concat([v.assign(outcome="hit", r=v["r_hit"]), v.assign(outcome="miss", r=v["r_miss"])], ignore_index=True)
    panels(long, "1_hit_vs_miss", "r", "noise correlation (trial-matched)",
           f"Noise correlation on hit and miss trials -- {cond}; {agg_note}", cdir, series_col="outcome",
           series_order=["hit", "miss"], series_colors={"hit": "#2a9d8f", "miss": "#c0392b"}, hline=0.0)
    panels(v, "2_delta_hit_minus_miss", "delta", "r_hit - r_miss",
           f"Hit minus miss noise correlation -- {cond}; {agg_note}", cdir, hline=0.0)
    lt = pd.concat([v.assign(trials="matched (count + stimulus mix)", d=v["delta"]),
                    v.assign(trials="all trials", d=v["delta_unmatched"])], ignore_index=True)
    panels(lt, "3_effect_of_trial_matching", "d", "r_hit - r_miss",
           f"Effect of matching hit and miss trials -- {cond}; {agg_note}", cdir, series_col="trials",
           series_order=["all trials", "matched (count + stimulus mix)"], hline=0.0,
           series_colors={"all trials": "#999999", "matched (count + stimulus mix)": "#2a9d8f"})
    ls = pd.concat([d[d["conditioning"] == cond].assign(subset=name, d=lambda x: x["delta"]) for name, d in subsets.items()],
                   ignore_index=True)
    panels(ls, "4_distance_matched_delta", "d", "r_hit - r_miss",
           f"Hit minus miss, all pairs vs distance-matched labelled/unlabelled pairs -- {cond}; {agg_note}", cdir,
           series_col="subset", series_order=list(subsets), hline=0.0,
           series_colors={"all pairs": "#999999", "distance-matched": "#2a9d8f"})

    v = v.assign(s_rel=pd.cut(v["distance_um"], DIST_EDGES, labels=DIST_MIDS).astype(float))
    cur = ptt.position_curves(v.dropna(subset=["s_rel"]), ["r_mean"], ["window"], AGG, MIN_PAIRS_PER_UNIT)
    fig, axes = plt.subplots(1, len(WIN), figsize=(5.2 * len(WIN), 4.2), squeeze=False)
    for ax, w in zip(axes[0], WIN):
        for g in SIX_GROUP_ORDER:
            c = cur[(cur["window"] == w) & (cur["group"] == group_key(g))]
            ptt.draw_lines(ax, c, "r_mean", SIX_GROUP_COLORS[g], label=SIX_GROUP_LABELS[g], aggregate=AGG)
        ax.axhline(0, color="grey", lw=0.7)
        ax.set_xlabel("distance between the two cells (um; bin centres)")
        ax.set_ylabel("noise correlation (mean of hit and miss)")
        ax.set_title(w, fontsize=9)
        ax.legend(fontsize=6, frameon=False)
    fig.suptitle(f"Noise correlation vs distance -- {cond}; {agg_note}", y=1.02, fontsize=9)
    fig.tight_layout()
    save_fig(fig, cdir, "5_noise_correlation_vs_distance")
    plt.close(fig)

cmp_ = valid.copy()
panels(cmp_, "compare_conditioning_delta", "delta", "r_hit - r_miss",
       f"Hit minus miss noise correlation: stimulus vs stimulus+movement conditioning; {agg_note}", figdir,
       series_col="conditioning", series_order=CONDITIONINGS, hline=0.0,
       series_colors={"stimulus": "#4c72b0", "stimulus+movement": "#dd8452"})
print(f"\nWrote tables to {table_dir}, figures to {figdir}")
