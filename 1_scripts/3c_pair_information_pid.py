# -*- coding: utf-8 -*-
"""
3c_pair_information_pid
=======================
Partial information decomposition (Williams & Beer 2010) of each pair's information about the target into redundancy,
unique information of each cell and synergy, with the I_min redundancy measure (REDUNDANCY_MEASURE; others are listed in
infotheory.pid.REDUNDANCY_MEASURES).
    I1, I2, I12 : plug-in MI of cell 1, cell 2 and the joint (r1, r2) with the target (NOT bias-corrected)
    redundancy  : I_min redundancy, bias-corrected by subtracting the mean over PID_SHUFFLES target-shuffled realisations
    unique1/2   : I1/I2 - redundancy
    synergy     : I12 - redundancy - unique1 - unique2
    redundancy_p_value : fraction of shuffles with redundancy >= observed (sig_redundancy = p < ALPHA)
Because I1/I2/I12 are plug-in and only the redundancy is corrected, the components carry the plug-in bias of I12; compare
across groups, not as absolute bits. 3b's RSI (I_total - I_1 - I_2) and 3c's synergy - redundancy are related but not equal.

Project-wide rules (infotheory/project_config.py) -- what applies here
    * selection : --animals / --sessions restrict the sessions; a restricted run writes to out/selection/<tag>/.
    * aggregate : --aggregate pooled|session|mouse (pooled = all pairs of a group; session / mouse = mean over the unit's
                  pairs first, units with < MIN_PAIRS_PER_UNIT pairs left out, bar = mean +- SEM, dots = units by mouse).
    * memory    : one session in RAM per worker (--n-jobs), cached per session in store/.
    * binning   : the SAME pairs as 3a / 3b (3a's out/pair_list.csv; --max-pairs keeps the first N per (session, area, label)),
                  the SAME trials (infotheory.pair_tables.pair_trials) and RESPONSE_BINS as 2c / 3b. Run 3a first.
    CHANGES vs the previous 3c: pairs come from 3a; shared trial rule and targets (`stim__<strategy>`, choice); scope,
    aggregate and parallel arguments; group plots follow --aggregate.

Outputs
-------
2_pipeline/3c_pair_information_pid/
    out/    pair_pid.csv                      one row per (session, window, target, pair): n_trials, I1, I2, I12, redundancy,
                                              unique1, unique2, synergy, redundancy_p_value, sig_redundancy
            pair_pid_by_group[__<aggregate>].csv
            figures/<target>/*.png, figures/compare_*.png
    out/selection/<tag>/...                   the same for a restricted run
    store/  pid_<hash>.pkl                    per-session caches
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
from infotheory.pid import pid_from_binned_responses, REDUNDANCY_MEASURES
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

MAX_PAIRS_PER_GROUP = 8     # pairs per (session, area, label) taken from 3a's list
R_BINS = cfg.RESPONSE_BINS  # shared with 2c / 3b
REDUNDANCY_MEASURE = "imin"
PID_SHUFFLES = 30           # target shuffles for the redundancy bias correction and p-value
ALPHA = 0.05
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
parser.add_argument("--pid-shuffles", type=int, default=PID_SHUFFLES)
parser.add_argument("--redundancy", default=REDUNDANCY_MEASURE, choices=sorted(REDUNDANCY_MEASURES))
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
                p = pid_from_binned_responses(d["x1"], d["x2"], d["label"], bins=R_BINS, redundancy_measure=args.redundancy,
                                              n_shuffles=args.pid_shuffles, rng=rng)
                res = dict(n_trials=p.n, I1=p.I1, I2=p.I2, I12=p.I12, redundancy=p.redundancy, unique1=p.unique1,
                           unique2=p.unique2, synergy=p.synergy, redundancy_p_value=p.redundancy_p_value)
                records.append(dict(session_id=sid, animal_id=ses.animal_id, window=window, target=target,
                                    roi_name=r.roi_name, labeled_grouped=r.labeled_grouped, cell_i=r.cell_i,
                                    cell_j=r.cell_j, pair_rank=r.pair_rank, **res))
    print(f"  [{sid}] {len(records)} (window, target, pair) rows")
    return records


records = run_sessions(
    session_ids, process_session, [PROTOCOL], paths.store, "pid",
    key_params=dict(windows=WINDOWS, r_bins=R_BINS, pid_shuffles=args.pid_shuffles, redundancy=args.redundancy,
                    targets=target_signature(TARGETS), min_trials=cfg.MIN_TRIALS_ANALYSIS,
                    min_per_bin=cfg.MIN_TRIALS_PER_TARGET_BIN, version="3c-v3", fits_mtime=int(fits_path.stat().st_mtime)),
    load_kwargs=dict(load_behaviordata=True, load_videodata=True, load_celldata=True, load_calciumdata=True),
    cells_by_session=cells_by_session, base_seed=RANDOM_STATE, n_jobs=scope.n_jobs, recompute=args.recompute,
    verbose=JOBLIB_VERBOSE, extra_key_by_session=stamps, flatten=True)
if not records:
    raise SystemExit("No pair had enough valid trials.")
table = pt.finalize_pair_table(pd.DataFrame(records))
table["sig_redundancy"] = (table["redundancy_p_value"] < ALPHA).astype(float)
table.to_csv(table_dir / "pair_pid.csv", index=False)
print(f"\n{len(table)} (session, window, target, pair) rows")

VALUE_COLS = ["I1", "I2", "I12", "redundancy", "unique1", "unique2", "synergy", "sig_redundancy"]
by_group = ct.unit_group_summary(table, ["window", "target"], AGG, VALUE_COLS).rename(
    columns={"n_cells": "n_pairs", "sig_redundancy": "frac_sig_redundancy"})
by_group.to_csv(table_dir / f"pair_pid_by_group{AGG_SUFFIX}.csv", index=False)

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
COMP, COMP_COL = ["redundancy", "unique1", "unique2", "synergy"], ["#8172b2", "#4c72b0", "#64b5cd", "#c44e52"]
for target in TARGETS:
    sub_t = table[table["target"] == target]
    if sub_t.empty:
        print(f"\n(no pairs for target {target!r})")
        continue
    tag = f"{target} [{' | '.join(target_bin_names(target))}]"
    tdir = figdir / target
    tdir.mkdir(parents=True, exist_ok=True)
    print(f"\n======== target: {tag} ========")
    per = ct.unit_group_summary(sub_t, ["window"], AGG, COMP + ["I12"])
    print(f"Mean PID components by (window, group){'' if AGG == 'pooled' else f', mean over {UNITS}'}:")
    print(per.groupby(["window", "group"])[COMP + ["I12"]].mean().round(4).to_string())
    print(f"\n{int(sub_t['sig_redundancy'].sum())}/{len(sub_t)} pairs have significant redundancy (p<{ALPHA}).")

    for comp in COMP:
        panels(sub_t, f"1_{comp}_by_group", comp, f"{comp} (bits)", f"PID {comp} -- {tag}; {agg_note}", tdir, hline=0.0)

    fig, axes = plt.subplots(1, len(WIN), figsize=(5 * len(WIN), 4.5), squeeze=False)
    for ax, w in zip(axes[0], WIN):
        sub = sub_t[sub_t["window"] == w]
        if sub.empty:
            continue
        pt.stacked_group_bars(ax, sub, COMP, dict(zip(COMP, COMP_COL)), AGG, mouse_colors, MIN_PAIRS_PER_UNIT, ylabel="bits",
                              total="I12")
        ax.set_title(w)
        if w == WIN[0]:
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"PID components (stack to I12; dots = {UNIT} I12) -- {tag}; {agg_note}", y=1.03, fontsize=9)
    fig.tight_layout()
    save_fig(fig, tdir, "2_component_breakdown")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(WIN), figsize=(5 * len(WIN), 5), squeeze=False)
    for ax, w in zip(axes[0], WIN):
        sub = sub_t[sub_t["window"] == w]
        for g in SIX_GROUP_ORDER:
            a_ = sub[sub["group"] == group_key(g)]
            ax.scatter(a_["redundancy"], a_["synergy"], s=10, alpha=0.4, color=SIX_GROUP_COLORS[g],
                       label=f"{SIX_GROUP_LABELS[g]} (n={len(a_)})")
        ax.legend(fontsize=7, frameon=False)
        ax.axhline(0, color="grey", linewidth=0.7)
        ax.axvline(0, color="grey", linewidth=0.7)
        ax.set_xlabel("redundancy (bits)")
        ax.set_ylabel("synergy (bits)")
        ax.set_title(w)
    fig.suptitle(f"Redundancy vs synergy, per pair (not aggregated) -- {tag}", y=1.03)
    fig.tight_layout()
    save_fig(fig, tdir, "3_redundancy_vs_synergy")
    plt.close(fig)

for val, nm in (("I12", "joint information I12 (bits)"), ("redundancy", "redundancy (bits)"), ("synergy", "synergy (bits)")):
    panels(table, f"compare_targets_{val}", val, f"mean {nm}", f"Mean {nm}: stimulus binning strategies vs. choice; {agg_note}",
           figdir, series_col="target", series_order=TARGETS)

print(f"\nWrote tables to {table_dir}, figures to {figdir}")
