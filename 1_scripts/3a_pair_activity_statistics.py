# -*- coding: utf-8 -*-
"""
3a_pair_activity_statistics
==============================
Pairwise correlation statistics between simultaneously recorded cells -- the "activity statistics" step of the
pairwise analyses (3a/3b/3c), preceding the information-theoretic Information Breakdown (3b, Pola et al. 2003) and
Partial Information Decomposition (3c) that build on the SAME pairs and the same conditioned-shuffle machinery.

For each sampled pair of cells (within an (area, label) group), per trial window (pre_stim / stim / reward, positions in
cm relative to stimStart, as 2c) and per TARGET (each stimulus binning strategy of stim_binning.py, plus "choice"):
    signal correlation : correlation between the two cells' per-target-bin mean responses (how similarly the pair
                         responds to the target ON AVERAGE)
    noise correlation  : correlation of the two cells' RESIDUALS (activity minus that trial's target-bin mean)
    raw correlation    : plain trial-by-trial correlation, no conditioning (signal + noise combined)
and the condition-conditioned shuffle test (infotheory.shuffling.conditioned_shuffle_test) in four modes:
    global      : is there ANY relationship (destroys everything, including shared tuning)
    stim        : is there a relationship BEYOND shared tuning to the target (shuffle analogue of the noise correlation)
    choice      : is there a relationship BEYOND shared choice
    stim_choice : is there a relationship BEYOND both jointly

Dataset: 1b-included sessions, ENGAGED trials, cells passing 2a's proximity filter AND 2b's QC
(infotheory.neural_encoding.load_cell_inclusion).

Project-wide rules (infotheory/project_config.py) -- what applies here
    * selection : --animals / --sessions restrict the sessions; a restricted run writes to out/selection/<tag>/ and
                  NEVER overwrites out/pair_list.csv, which 3b and 3c read.
    * y axes    : the window panels of every group figure share one y range (SHARE_Y); the pair-level scatter and
                  boxplot figures already did.
    * aggregate : --aggregate pooled|session|mouse = the unit of the group plots: pooled = all pairs of a group together,
                  no error bars; session / mouse = mean over the unit's pairs first (units with fewer than
                  MIN_PAIRS_PER_UNIT pairs in the group are left out), bar = mean +- SEM over units, dots = units
                  coloured by mouse. Always the six groups. The pair-level scatters and boxplots are not aggregated.
    * memory    : one session in RAM per worker (--n-jobs), reduced to small tables and cached per session in store/.
    * binning   : 3a, 3b and 3c use the SAME pairs (3a samples them once per session and writes pair_list.csv; 3b/3c read
                  it) and the SAME trials per (pair, target, window) (infotheory.pair_tables.pair_trials: both cells and
                  label finite, target bins with < MIN_TRIALS_PER_TARGET_BIN trials dropped, >= MIN_TRIALS_ANALYSIS trials,
                  2 labels) -- the rule of 2c/2d. The correlations here use the window-mean responses as they are; 3b/3c
                  cut them into RESPONSE_BINS quantile bins.
    CHANGES vs the previous 3a: pairs are sampled once per session (before: again for every window); target bins are
    counted AFTER removing non-finite responses (before: before); targets are the `stim__<strategy>` columns of 2c.
    Numbers are therefore not identical to old runs.

CAVEAT: MAX_PAIRS_PER_GROUP pairs per (session, area, label) is a small sample of the possible pairs (and few pairs per
session make session-level bars noisy); raise it if the compute budget allows (3b/3c can use a subset via --max-pairs).

Outputs
-------
2_pipeline/3a_pair_activity_statistics/
    out/    pair_list.csv                    the sampled pairs (session, area, label, cell_i, cell_j, pair_rank) -- read by 3b, 3c
            pair_correlations.csv            one row per (session, window, TARGET, pair): n_trials, signal / noise / raw
                                             correlation, each shuffle mode's corrected statistic and p-value
            pair_correlations_by_group[__<aggregate>].csv   per group x window x target (x unit): n_pairs, mean
                                             correlations, fraction of pairs with p<0.05 per shuffle mode
            figures/<target>/*.png           one figure set per target; figures/compare_*.png across targets
    out/selection/<tag>/...                  the same for a restricted run
    store/  pair_correlations_<hash>.pkl     per-session caches
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
from infotheory.shuffling import conditioned_shuffle_test, SHUFFLE_MODES
from infotheory.celldata_utils import SIX_GROUP_ORDER, SIX_GROUP_LABELS, SIX_GROUP_COLORS
from infotheory.groups import group_key
from infotheory.plotting import set_style, save_fig, clear_figures
from infotheory.scope import add_scope_args, scope_from_args, select_sessions, scoped_dirs, data_stamp, unit_noun
from infotheory.session_jobs import run_sessions
from infotheory.unit_stats import stable_seed
from infotheory.unit_plots import group_panels, mouse_palette
from infotheory.stim_binning import (STIM_BINNING_STRATEGIES, DEFAULT_STRATEGIES, CHOICE_TARGET, target_signature,
                                     target_bin_names, describe_strategies)
from infotheory import cell_tables as ct
from infotheory import pair_tables as pt

PROTOCOL = "DN"
WINDOWS = {"pre_stim": (-30.0, -10.0), "stim": (0.0, 20.0), "reward": (25.0, 45.0)}

MAX_PAIRS_PER_GROUP = 8     # per (session, area, label); 3b/3c use a subset (--max-pairs)
N_SHUFFLES = 10             # per shuffle test (4 modes x pairs x windows x targets of them per session)
MIN_PAIRS_PER_UNIT = 3      # session / mouse aggregate: pairs needed in a group for a unit's value to be used
ALPHA = 0.05
RANDOM_STATE = 0
BINNING_STRATEGIES = list(DEFAULT_STRATEGIES)   # stimulus targets (override with --strategies)
INCLUDE_CHOICE_TARGET = True
SHARE_Y = True              # the three windows of a group figure get the same y range
JOBLIB_VERBOSE = 5

parser = argparse.ArgumentParser()
add_scope_args(parser)
parser.add_argument("--strategies", nargs="+", default=None, choices=sorted(STIM_BINNING_STRATEGIES),
                    help=f"stimulus binning strategies (default {BINNING_STRATEGIES})")
parser.add_argument("--n-shuffles", type=int, default=N_SHUFFLES)
parser.add_argument("--max-pairs", type=int, default=MAX_PAIRS_PER_GROUP, help="pairs sampled per (session, area, label)")
args = parser.parse_args()
scope = scope_from_args(args)
AGG = scope.aggregate
UNIT = unit_noun(AGG)
UNITS = "mice" if UNIT == "mouse" else f"{UNIT}s"
AGG_SUFFIX = "" if AGG == "pooled" else f"__{AGG}"
STRATEGIES = args.strategies or BINNING_STRATEGIES
TARGETS = STRATEGIES + ([CHOICE_TARGET] if INCLUDE_CHOICE_TARGET else [])
N_SHUF = args.n_shuffles
MAX_PAIRS = args.max_pairs

paths = get_pipeline_paths(__file__)
table_dir, figdir = scoped_dirs(paths, scope)
clear_figures(figdir)
for _sub in figdir.glob("*/"):
    clear_figures(_sub)
set_style()
print(scope.describe())
print("Targets (stimulus binning strategies + choice):\n" + describe_strategies(TARGETS))

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


def corr_fn(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------- #
# 2. Per session (one in RAM at a time): sample pairs once, then correlations + shuffle tests
# --------------------------------------------------------------------- #
def process_session(ses, seed):
    sid = ses.session_id
    cells = set(cells_by_session.get(sid, []))
    if not cells or ses.celldata is None:
        return None
    rng = np.random.default_rng(seed)
    celldata = ses.celldata[ses.celldata["cell_id"].isin(cells)].reset_index(drop=True)
    if len(celldata) < 2:
        return None
    pairs = pt.sample_session_pairs(celldata, sid, np.random.default_rng(stable_seed(sid, "pairs", base=RANDOM_STATE)),
                                    MAX_PAIRS)
    tables = build_session_tables(ses, mu_sigma_lookup, WINDOWS, included_cell_ids=cells)
    records = []
    for window, table in tables.items():
        if table.empty or "signal" not in table.columns:
            continue
        table = pt.add_target_columns(table, STRATEGIES)
        for target in TARGETS:
            for r in pairs.itertuples(index=False):
                d = pt.pair_trials(table, target, r.cell_i, r.cell_j)
                if d is None:
                    continue
                xi, xj, sb, ch = d["x1"], d["x2"], d["label"], d["choice"]
                tuning_i = pd.Series(xi).groupby(sb).transform("mean").to_numpy()
                tuning_j = pd.Series(xj).groupby(sb).transform("mean").to_numpy()
                bm_i, bm_j = pd.Series(xi).groupby(sb).mean(), pd.Series(xj).groupby(sb).mean()
                row = dict(session_id=sid, animal_id=ses.animal_id, window=window, target=target,
                           roi_name=r.roi_name, labeled_grouped=r.labeled_grouped, cell_i=r.cell_i, cell_j=r.cell_j,
                           pair_rank=r.pair_rank, n_trials=len(xi), n_bins=len(bm_i),
                           signal_corr=corr_fn(bm_i.to_numpy(), bm_j.to_numpy()),
                           noise_corr=corr_fn(xi - tuning_i, xj - tuning_j), raw_corr=corr_fn(xi, xj))
                for mode in SHUFFLE_MODES:
                    res = conditioned_shuffle_test(corr_fn, xi, xj, mode=mode, stim=sb, choice=ch,
                                                   n_shuffles=N_SHUF, rng=rng)
                    row[f"shuffle_{mode}_corrected"] = res.stat_corrected
                    row[f"shuffle_{mode}_p"] = res.p_value
                records.append(row)
    print(f"  [{sid}] {len(pairs)} pairs, {len(records)} (window, target, pair) rows")
    return dict(session_id=sid, pairs=pairs, records=records)


results = run_sessions(
    session_ids, process_session, [PROTOCOL], paths.store, "pair_correlations",
    key_params=dict(windows=WINDOWS, max_pairs=MAX_PAIRS, n_shuffles=N_SHUF, targets=target_signature(TARGETS),
                    min_trials=cfg.MIN_TRIALS_ANALYSIS, min_per_bin=cfg.MIN_TRIALS_PER_TARGET_BIN, version="3a-v3",
                    fits_mtime=int(fits_path.stat().st_mtime)),
    load_kwargs=dict(load_behaviordata=True, load_videodata=True, load_celldata=True, load_calciumdata=True),
    cells_by_session=cells_by_session, base_seed=RANDOM_STATE, n_jobs=scope.n_jobs, recompute=args.recompute,
    verbose=JOBLIB_VERBOSE, extra_key_by_session=stamps, flatten=False)
results = [r for r in results if r is not None]
pair_list = pd.concat([r["pairs"] for r in results], ignore_index=True) if results else pd.DataFrame()
records = [row for r in results for row in r["records"]]
if not records:
    raise SystemExit("No pair had enough valid trials.")
pair_list.to_csv(table_dir / "pair_list.csv", index=False)
pair_table = pt.finalize_pair_table(pd.DataFrame(records))
for m in SHUFFLE_MODES:
    pair_table[f"sig_{m}"] = (pair_table[f"shuffle_{m}_p"] < ALPHA).astype(float)
pair_table.to_csv(table_dir / "pair_correlations.csv", index=False)
print(f"\n{len(pair_list)} pairs sampled; {len(pair_table)} (session, window, target, pair) rows")

VALUE_COLS = ["signal_corr", "noise_corr", "raw_corr"] + [f"sig_{m}" for m in SHUFFLE_MODES]
unit_gt = ct.unit_group_summary(pair_table, ["window", "target"], AGG, VALUE_COLS)
unit_gt = unit_gt.rename(columns={"n_cells": "n_pairs", **{f"sig_{m}": f"frac_p<{ALPHA}_{m}" for m in SHUFFLE_MODES}})
unit_gt.to_csv(table_dir / f"pair_correlations_by_group{AGG_SUFFIX}.csv", index=False)

GROUP_KEYS = [group_key(g) for g in SIX_GROUP_ORDER]
agg_note = {"pooled": "all pairs pooled (no error bars)", "session": "bar = mean +- SEM over sessions, dots = sessions",
            "mouse": "bar = mean +- SEM over mice, dots = mice"}[AGG]
WIN = list(WINDOWS)


def panels(df, name, value, ylabel, title, outdir, **kw):
    kw.setdefault("sharey", SHARE_Y)
    group_panels(df, "window", WIN, value, AGG, outdir, name, title, ylabel=ylabel, mouse_colors=mouse_colors,
                 min_cells_per_unit=MIN_PAIRS_PER_UNIT, ncols=len(WIN), panel_size=(4.6, 3.6), **kw)


# --------------------------------------------------------------------- #
# 3. Reports and figures, one set per target
# --------------------------------------------------------------------- #
for target in TARGETS:
    sub_t = pair_table[pair_table["target"] == target]
    if sub_t.empty:
        print(f"\n(no pairs for target {target!r})")
        continue
    tag = f"{target} [{' | '.join(target_bin_names(target))}]"
    tdir = figdir / target
    tdir.mkdir(parents=True, exist_ok=True)
    print(f"\n======== target: {tag} ========")
    per = ct.unit_group_summary(sub_t, ["window"], AGG, ["signal_corr", "noise_corr", "raw_corr"])
    print(f"Mean signal / noise / raw correlation by (window, group){'' if AGG == 'pooled' else f', mean over {UNITS}'}:")
    print(per.groupby(["window", "group"])[["signal_corr", "noise_corr", "raw_corr"]].mean().round(3).to_string())
    sig = sub_t.groupby(["window"] + ([] if AGG == "pooled" else [{"session": "session_id", "mouse": "animal_id"}[AGG]]))[
        [f"sig_{m}" for m in SHUFFLE_MODES]].mean().groupby("window").mean()
    print(f"\nFraction of pairs with p<{ALPHA} per shuffle mode (global should be the largest, stim_choice the smallest):")
    print(sig.rename(columns=lambda c: c[4:]).round(3).to_string())

    fig, axes = plt.subplots(1, len(WIN), figsize=(5 * len(WIN), 5), squeeze=False)    # 1: pair-level scatter
    for ax, w in zip(axes[0], WIN):
        sub = sub_t[sub_t["window"] == w].dropna(subset=["signal_corr", "noise_corr"])
        for g in SIX_GROUP_ORDER:
            a_ = sub[sub["group"] == group_key(g)]
            ax.scatter(a_["signal_corr"], a_["noise_corr"], s=10, alpha=0.4, color=SIX_GROUP_COLORS[g],
                       label=f"{SIX_GROUP_LABELS[g]} (n={len(a_)})")
        ax.axhline(0, color="grey", linewidth=0.7)
        ax.axvline(0, color="grey", linewidth=0.7)
        ax.set_xlabel("signal correlation")
        ax.set_ylabel("noise correlation")
        ax.set_title(w)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"Signal vs. noise correlation, per pair (not aggregated) -- {PROTOCOL}, {tag}", y=1.03)
    fig.tight_layout()
    save_fig(fig, tdir, "1_signal_vs_noise_correlation")
    plt.close(fig)

    panels(sub_t, "2_noise_correlation_by_group", "noise_corr", "noise correlation",
           f"Noise correlation -- {tag}; {agg_note}", tdir)
    panels(sub_t, "2b_signal_correlation_by_group", "signal_corr", "signal correlation",
           f"Signal correlation -- {tag}; {agg_note}", tdir)
    panels(sub_t, "2c_raw_correlation_by_group", "raw_corr", "raw correlation",
           f"Raw (trial-by-trial) correlation -- {tag}; {agg_note}", tdir)

    fig, axes = plt.subplots(1, len(WIN), figsize=(5 * len(WIN), 4.5), squeeze=False)    # 3: pair-level boxplots
    for ax, w in zip(axes[0], WIN):
        sub = sub_t[sub_t["window"] == w]
        if sub.empty:
            continue
        ax.boxplot([sub[f"shuffle_{m}_corrected"].dropna() for m in SHUFFLE_MODES], tick_labels=SHUFFLE_MODES, showfliers=False)
        ax.axhline(0, color="grey", linewidth=0.7)
        ax.set_ylabel("shuffle-corrected correlation")
        ax.set_title(w)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle(f"What survives each conditioned shuffle, all pairs (not aggregated) -- {PROTOCOL}, {tag}", y=1.03)
    fig.tight_layout()
    save_fig(fig, tdir, "3_shuffle_mode_comparison")
    plt.close(fig)

    modes_long = pd.concat([sub_t.assign(mode=m, corrected=sub_t[f"shuffle_{m}_corrected"]) for m in SHUFFLE_MODES],
                           ignore_index=True)
    panels(modes_long, "3b_shuffle_mode_by_group", "corrected", "shuffle-corrected correlation",
           f"Correlation surviving each conditioned shuffle -- {tag}; {agg_note}", tdir,
           series_col="mode", series_order=list(SHUFFLE_MODES))

# --------------------------------------------------------------------- #
# 4. Targets compared (the three stimulus binnings + choice), six groups at the chosen unit
# --------------------------------------------------------------------- #
for val, nm in (("noise_corr", "noise correlation"), ("signal_corr", "signal correlation")):
    panels(pair_table, f"compare_targets_{val}", val, f"mean {nm}",
           f"Mean {nm}: stimulus binning strategies vs. choice; {agg_note}", figdir, series_col="target",
           series_order=TARGETS)
if CHOICE_TARGET in TARGETS:
    for strat in STRATEGIES:
        two = pair_table[pair_table["target"].isin([strat, CHOICE_TARGET])]
        for val, nm in (("noise_corr", "noise"), ("signal_corr", "signal")):
            panels(two, f"{nm}_stim_vs_choice__{strat}", val, f"{nm} correlation",
                   f"{nm.capitalize()} correlation with the stimulus [{strat}] vs. with choice; {agg_note}", figdir,
                   series_col="target", series_order=[strat, CHOICE_TARGET],
                   series_colors={strat: "#4c72b0", CHOICE_TARGET: "#c44e52"})

print(f"\nWrote tables to {table_dir}, figures to {figdir}")
