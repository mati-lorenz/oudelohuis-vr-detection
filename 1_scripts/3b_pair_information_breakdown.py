# -*- coding: utf-8 -*-
"""
3b_pair_information_breakdown
================================
The Pola, Thiele, Hoffmann & Panzeri (2003) "exact" information breakdown, applied to pairs of simultaneously recorded
cells: how much of a pair's joint information about the target (stimulus binning or choice) is explained by each cell's
own tuning, how much is lost/gained from the two cells having similar/different tuning, and how much comes from genuine
trial-to-trial correlated activity.

DEFINITIONS (Pola, Thiele, Hoffmann & Panzeri 2003, Network 14:35; formulas and tests in infotheory/pair_information.py)
    I_total  : MI between the pair's JOINT response (r1, r2) and the target (direct method, shuffle-corrected)
    I_lin    : I_1 + I_2, the information if the two cells were independent informants (shuffle-corrected)
    I_sigsim : signal similarity, the closed form of Pola's eq. 10 on the plug-in probabilities
               = -KL( sum_s P(s) P(r1|s)P(r2|s)  ||  P(r1)P(r2) ). ALWAYS <= 0 (0 only when the two cells' tuning is
               unrelated); the more similar the tuning, the more negative (redundancy). Not bias-corrected (Pola et al.:
               hardly biased), which is what guarantees the sign.
    I_corind : stimulus-independent correlational term (eq. 11): interplay of signal similarity and noise correlation.
               Either sign (positive when they have opposite signs). Plug-in.
    I_cordep : stimulus-dependent correlational term (eq. 12): information from the target modulating the trial-to-trial
               correlation. >= 0 before bias correction; here the remainder I_total - I_lin - I_sigsim - I_corind (inherits the
               shuffle-corrected total), NOT floored: it can be slightly negative after correction. Bias treatment per term:
               pair_information.py.
    I_cor    = I_total - I_lin - I_sigsim (exact remainder; = I_corind + I_cordep) ;  RSI = I_total - I_lin = I_sigsim + I_cor
               (> 0 net synergy, < 0 net redundancy).   I_ind = I_lin + I_sigsim.
The four terms add up exactly to I_total (I_corind is the remainder after the others; its own formula agrees with the
remainder to 1e-15 on plug-in probabilities). Response binning: RESPONSE_BINS (3) quantile bins per cell.

Project-wide rules (infotheory/project_config.py) -- what applies here
    * selection : --animals / --sessions restrict the sessions; a restricted run writes to out/selection/<tag>/.
    * aggregate : --aggregate pooled|session|mouse = the unit of the group plots (pooled = all pairs of a group, no error
                  bars; session / mouse = mean over the unit's pairs first, units with < MIN_PAIRS_PER_UNIT pairs left
                  out, bar = mean +- SEM over units, dots = units coloured by mouse). Always the six groups.
    * memory    : one session in RAM per worker (--n-jobs), cached per session in store/.
    * binning   : the SAME pairs as 3a and 3c (read from 3a's out/pair_list.csv; --max-pairs uses the first N per
                  (session, area, label) in 3a's random order, so a smaller budget is a subset of the same pairs) and the
                  SAME trials per (pair, target, window) (infotheory.pair_tables.pair_trials), RESPONSE_BINS shared with
                  2c / 3c. Run 3a first (full dataset). 3b's cost per pair is ~40x 3a's, hence the default --max-pairs 5.
    CHANGES vs the previous 3b: I_sigsim is now the exact (always <= 0) closed form, I_corind / I_cordep are Pola's exact
    terms (before: surrogate-based, I_sigsim could come out positive); pairs come from 3a (before: sampled again, per window); target bins counted after removing
    non-finite responses; stimulus targets are 2c's `stim__<strategy>`.

Outputs
-------
2_pipeline/3b_pair_information_breakdown/
    out/    pair_information_breakdown.csv   one row per (session, window, TARGET, pair): I_total, I_lin, I_sigsim, I_cor,
                                             RSI, I_corind, I_cordep, I_cordep_raw, n_trials
            pair_information_breakdown_by_group[__<aggregate>].csv   per group x window x target (x unit): n_pairs, means,
                                             fraction of synergistic pairs
            figures/<target>/*.png, figures/compare_*.png
    out/selection/<tag>/...                  the same for a restricted run
    store/  breakdown_<hash>.pkl             per-session caches
"""
from __future__ import annotations

import argparse
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from infotheory import project_config as cfg
from infotheory.pipeline import get_pipeline_paths
from infotheory.session import discover_sessions
from infotheory.neural_encoding import load_cell_inclusion, build_session_tables
from infotheory.pair_information import compute_pair_breakdown
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

MAX_PAIRS_PER_GROUP = 5     # pairs per (session, area, label) taken from 3a's list (cost per pair is high)
R_BINS = cfg.RESPONSE_BINS  # shared with 2c / 3c
MI_SHUFFLES = 4             # bias-correction shuffles for every direct-method MI
MIN_PAIRS_PER_UNIT = 3
RANDOM_STATE = 0
BINNING_STRATEGIES = list(DEFAULT_STRATEGIES)
INCLUDE_CHOICE_TARGET = True
SHARE_Y = True              # the window panels of a group figure get the same y range
JOBLIB_VERBOSE = 5

parser = argparse.ArgumentParser()
add_scope_args(parser)
parser.add_argument("--strategies", nargs="+", default=None, choices=sorted(STIM_BINNING_STRATEGIES))
parser.add_argument("--max-pairs", type=int, default=MAX_PAIRS_PER_GROUP, help="pairs per (session, area, label), from 3a's list")
parser.add_argument("--mi-shuffles", type=int, default=MI_SHUFFLES)
args = parser.parse_args()
scope = scope_from_args(args)
AGG = scope.aggregate
UNIT = unit_noun(AGG)
UNITS = "mice" if UNIT == "mouse" else f"{UNIT}s"
AGG_SUFFIX = "" if AGG == "pooled" else f"__{AGG}"
STRATEGIES = args.strategies or BINNING_STRATEGIES
TARGETS = STRATEGIES + ([CHOICE_TARGET] if INCLUDE_CHOICE_TARGET else [])

paths = get_pipeline_paths(__file__)
table_dir, figdir = scoped_dirs(paths, scope)
clear_figures(figdir)
for _sub in figdir.glob("*/"):
    clear_figures(_sub)
set_style()
print(scope.describe())
print("Targets (stimulus binning strategies + choice):\n" + describe_strategies(TARGETS))

# --------------------------------------------------------------------- #
# 1. Inputs (pairs from 3a)
# --------------------------------------------------------------------- #
fits_path = paths.out_from("1b_psychometric") / "psychometric_fits.csv"
cell_dist_out = paths.out_from("2a_cell_distribution")
activity_out = paths.out_from("2b_activity_statistics")
pair_list_path = paths.out_from("3a_pair_activity_statistics") / "pair_list.csv"
for f, hint in [(fits_path, "1b_psychometric.py"), (cell_dist_out / "celldata_combined.csv", "2a_cell_distribution.py"),
                (activity_out / "qc_per_cell.csv", "2b_activity_statistics.py"),
                (pair_list_path, "3a_pair_activity_statistics.py (full dataset)")]:
    if not f.exists():
        raise SystemExit(f"Run {hint} first -- expected {f}")
fits_df = pd.read_csv(fits_path)
mu_sigma_lookup = fits_df.set_index("session_id")[["mu", "sigma"]].to_dict("index")
included_all = set(fits_df[fits_df["included"] & (fits_df["protocol"] == PROTOCOL)]["session_id"])
all_triples = discover_sessions([PROTOCOL])
selected = {f"{a}_{d}": (p, a, d) for p, a, d in select_sessions(all_triples, scope)}
session_ids = [s for s in selected if s in included_all]
mouse_colors = mouse_palette(a for _, a, _ in all_triples)
pairs_all = pt.load_pair_list(pair_list_path, session_ids, args.max_pairs)
missing = sorted(set(session_ids) - set(pairs_all["session_id"]))
if missing:
    print(f"WARNING: no pairs in 3a's list for {missing} (rerun 3a on the full dataset); skipped")
session_ids = [s for s in session_ids if s in set(pairs_all["session_id"])]
if not session_ids:
    raise SystemExit("No selected session has pairs in 3a's pair_list.csv.")
pairs_by_session = {sid: sub.reset_index(drop=True) for sid, sub in pairs_all.groupby("session_id")}
cells_by_session = {sid: sorted(set(sub["cell_i"]) | set(sub["cell_j"])) for sid, sub in pairs_by_session.items()}
print(f"{len(session_ids)} sessions, {len(pairs_all)} pairs (3a's list, pair_rank < {args.max_pairs})")
pair_hash = {s: hashlib.sha1(pairs_by_session[s].to_csv(index=False).encode()).hexdigest()[:10] for s in session_ids}
stamps = {s: f"{data_stamp(*selected[s])}-{pair_hash[s]}" for s in session_ids}


# --------------------------------------------------------------------- #
# 2. Per session
# --------------------------------------------------------------------- #
def process_session(ses, seed):
    sid = ses.session_id
    pairs = pairs_by_session.get(sid)
    if pairs is None or ses.celldata is None:
        return []
    rng = np.random.default_rng(seed)
    tables = build_session_tables(ses, mu_sigma_lookup, WINDOWS, included_cell_ids=set(cells_by_session[sid]))
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
                res = compute_pair_breakdown(d["x1"], d["x2"], d["label"], r_bins=R_BINS, mi_shuffles=args.mi_shuffles,
                                             rng=rng)
                records.append(dict(session_id=sid, animal_id=ses.animal_id, window=window, target=target,
                                    roi_name=r.roi_name, labeled_grouped=r.labeled_grouped, cell_i=r.cell_i,
                                    cell_j=r.cell_j, pair_rank=r.pair_rank, **res))
    print(f"  [{sid}] {len(records)} (window, target, pair) rows")
    return records


records = run_sessions(
    session_ids, process_session, [PROTOCOL], paths.store, "breakdown",
    key_params=dict(windows=WINDOWS, r_bins=R_BINS, mi_shuffles=args.mi_shuffles,
                    targets=target_signature(TARGETS), min_trials=cfg.MIN_TRIALS_ANALYSIS,
                    min_per_bin=cfg.MIN_TRIALS_PER_TARGET_BIN, version="3b-v7", fits_mtime=int(fits_path.stat().st_mtime)),
    load_kwargs=dict(load_behaviordata=True, load_videodata=True, load_celldata=True, load_calciumdata=True),
    cells_by_session=cells_by_session, base_seed=RANDOM_STATE, n_jobs=scope.n_jobs, recompute=args.recompute,
    verbose=JOBLIB_VERBOSE, extra_key_by_session=stamps, flatten=True)
if not records:
    raise SystemExit("No pair had enough valid trials.")
table = pt.finalize_pair_table(pd.DataFrame(records))
table["synergistic"] = (table["RSI"] > 0).astype(float)
table.to_csv(table_dir / "pair_information_breakdown.csv", index=False)
print(f"\n{len(table)} (session, window, target, pair) rows")

VALUE_COLS = ["I_total", "I_lin", "I_sigsim", "I_cor", "RSI", "I_corind", "I_cordep", "synergistic"]
by_group = ct.unit_group_summary(table, ["window", "target"], AGG, VALUE_COLS).rename(
    columns={"n_cells": "n_pairs", "synergistic": "frac_synergistic"})
by_group.to_csv(table_dir / f"pair_information_breakdown_by_group{AGG_SUFFIX}.csv", index=False)

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
COMP, COMP_COL = ["I_lin", "I_sigsim", "I_cor"], ["#4c72b0", "#dd8452", "#55a868"]
for target in TARGETS:
    sub_t = table[table["target"] == target]
    if sub_t.empty:
        print(f"\n(no pairs for target {target!r})")
        continue
    tag = f"{target} [{' | '.join(target_bin_names(target))}]"
    tdir = figdir / target
    tdir.mkdir(parents=True, exist_ok=True)
    print(f"\n======== target: {tag} ========")
    per = ct.unit_group_summary(sub_t, ["window"], AGG, ["I_total", "I_lin", "I_sigsim", "I_cor", "RSI"])
    print(f"Mean I_total / I_lin / I_sigsim / I_cor / RSI by (window, group){'' if AGG == 'pooled' else f', mean over {UNITS}'}:")
    print(per.groupby(["window", "group"])[["I_total", "I_lin", "I_sigsim", "I_cor", "RSI"]].mean().round(4).to_string())
    print(f"\n{int(sub_t['synergistic'].sum())}/{len(sub_t)} pairs show net synergy (RSI>0); the rest net redundancy.")

    panels(sub_t, "1_rsi_by_group", "RSI", "RSI (bits)",
           f"Redundancy-Synergy Index (>0 synergy, <0 redundancy) -- {tag}; {agg_note}", tdir, hline=0.0)
    panels(sub_t, "1b_itotal_by_group", "I_total", "I_total (bits)", f"Joint pair information -- {tag}; {agg_note}", tdir)

    fig, axes = plt.subplots(1, len(WIN), figsize=(5 * len(WIN), 4.5), squeeze=False)     # 2: components stacked
    for ax, w in zip(axes[0], WIN):
        sub = sub_t[sub_t["window"] == w]
        if sub.empty:
            continue
        pt.stacked_group_bars(ax, sub, COMP, dict(zip(COMP, COMP_COL)), AGG, mouse_colors, MIN_PAIRS_PER_UNIT, ylabel="bits",
                              total="I_total")
        ax.set_title(w)
        if w == WIN[0]:
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"Information breakdown components (stack to I_total; dots = {UNIT} I_total) -- {tag}; {agg_note}", y=1.03,
                 fontsize=9)
    fig.tight_layout()
    save_fig(fig, tdir, "2_component_breakdown")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(WIN), figsize=(5 * len(WIN), 5), squeeze=False)       # 3: pair-level scatter
    for ax, w in zip(axes[0], WIN):
        sub = sub_t[sub_t["window"] == w]
        for g in SIX_GROUP_ORDER:
            a_ = sub[sub["group"] == group_key(g)]
            ax.scatter(a_["I_corind"], a_["I_cordep"], s=10, alpha=0.4, color=SIX_GROUP_COLORS[g],
                       label=f"{SIX_GROUP_LABELS[g]} (n={len(a_)})")
        ax.legend(fontsize=7, frameon=False)
        ax.axhline(0, color="grey", linewidth=0.7)
        ax.axvline(0, color="grey", linewidth=0.7)
        ax.set_xlabel("I_corind")
        ax.set_ylabel("I_cordep")
        ax.set_title(w)
    fig.suptitle(f"Split of I_cor into I_corind and I_cordep, per pair (not aggregated) -- {tag}", y=1.03)
    fig.tight_layout()
    save_fig(fig, tdir, "3_corind_vs_cordep")
    plt.close(fig)

for val, nm in (("I_total", "pair information I_total (bits)"), ("RSI", "RSI"), ("I_cor", "I_cor (bits)")):
    panels(table, f"compare_targets_{val}", val, f"mean {nm}", f"Mean {nm}: stimulus binning strategies vs. choice; {agg_note}",
           figdir, series_col="target", series_order=TARGETS)

print(f"\nWrote tables to {table_dir}, figures to {figdir}")
