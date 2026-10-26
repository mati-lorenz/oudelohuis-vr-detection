# -*- coding: utf-8 -*-
"""
3f_pair_information_pid_temporal
================================
Position-resolved version of 3c: partial information decomposition (redundancy, unique1, unique2, synergy; I_min
redundancy, shuffle-corrected; definitions and caveats in 3c's docstring) of each pair at every position bin relative to
stimulus onset instead of once per coarse window.

BINS: `fixed` = non-overlapping bins of BINSIZE cm; `sliding` = windows of WINDOW cm moved by STEP cm (neighbouring windows
share data: the curve is smoother and less noisy, its points are not independent). Each window = the trial's mean activity
over the frames of the engaged trial inside the window (a frame is used only if it lies in ITS OWN trial's [tStart, tEnd]).
PAIRS: MAX_PAIRS_PER_GROUP = 150 per (session, area, label), sampled here from the 2a + 2b included cells (as 3g / 3h), no
longer 3a's list of 5-8: the pairs therefore differ from 3a-3c's.
COST: one PID with PID_SHUFFLES = 30 shuffles per (pair, window, target), about 9 ms: with ~900 pairs per session, 4 targets
and 7 fixed + 25 sliding windows this is of the order of 20-30 minutes per session and worker. --binnings fixed / sliding
runs one kind only, --max-pairs / --pid-shuffles lower the cost.

Project-wide rules (infotheory/project_config.py) -- what applies here
    * selection : --animals / --sessions restrict the sessions; a restricted run writes to out/selection/<tag>/.
    * aggregate : --aggregate pooled|session|mouse. Curves = per group, binning, target and position bin, the mean over
                  units of the unit's mean over its pairs (units with < MIN_PAIRS_PER_UNIT pairs at that bin left out),
                  band = SEM over units; pooled: mean over all pairs, band = SEM over pairs (pairs treated as independent).
                  Always the six groups.
    * memory    : one session in RAM per worker (--n-jobs); per-pair values are reduced to per-(session, group) sums, no
                  per-pair table is written.
    * binning   : the same targets, trial rule and RESPONSE_BINS as 3a-3c. Helpers: infotheory.pair_temporal.

Outputs
-------
2_pipeline/3f_pair_information_pid_temporal/
    out/    pair_pid_temporal_moments.csv            per (session, binning, target, bin, group): n_trials and, for each value,
                                                     n / sum / sum of squares over the pairs
            pair_pid_temporal_by_group[__<aggregate>].csv   curves per group x binning x target x s_rel
            figures/<binning>/<target>/1_pid_vs_position.png, 2_group_comparison.png; figures/<binning>/compare_targets_*.png
    out/selection/<tag>/...                          the same for a restricted run
    store/  pid_temporal_<hash>.pkl            per-session caches
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from infotheory import project_config as cfg
from infotheory.pipeline import get_pipeline_paths
from infotheory.pid import pid_from_binned_responses, REDUNDANCY_MEASURES
from infotheory.celldata_utils import SIX_GROUP_ORDER, SIX_GROUP_LABELS, SIX_GROUP_COLORS
from infotheory.groups import group_key
from infotheory.plotting import set_style, save_fig, clear_figures
from infotheory.scope import add_scope_args, scope_from_args, scoped_dirs, unit_noun
from infotheory.session_jobs import run_sessions
from infotheory.stim_binning import (STIM_BINNING_STRATEGIES, DEFAULT_STRATEGIES, CHOICE_TARGET, target_signature,
                                     target_bin_names, describe_strategies)
from infotheory import pair_temporal as ptt

PROTOCOL = "DN"
S_PRE, S_POST, BINSIZE = -60.0, 80.0, 20.0
MAX_PAIRS_PER_GROUP = 150   # per (session, area, label)
WINDOW, STEP = 20.0, 5.0     # sliding window and step (cm)
BINNINGS = ["fixed", "sliding"]
R_BINS = cfg.RESPONSE_BINS
REDUNDANCY_MEASURE = "imin"
PID_SHUFFLES = 1
ALPHA = 0.05
MIN_PAIRS_PER_UNIT = 5
RANDOM_STATE = 0
BINNING_STRATEGIES = list(DEFAULT_STRATEGIES)
INCLUDE_CHOICE_TARGET = True
JOBLIB_VERBOSE = 5
TERMS = ["redundancy", "unique1", "unique2", "synergy"]
TERM_COLORS = {"redundancy": "tab:purple", "unique1": "tab:blue", "unique2": "tab:cyan", "synergy": "tab:red"}
VALUES = TERMS + ["I12"]
STORE_VALUES = VALUES + ["I1", "I2"]

parser = argparse.ArgumentParser()
add_scope_args(parser)
parser.add_argument("--strategies", nargs="+", default=None, choices=sorted(STIM_BINNING_STRATEGIES))
parser.add_argument("--max-pairs", type=int, default=MAX_PAIRS_PER_GROUP, help="pairs per (session, area, label)")
parser.add_argument("--binnings", nargs="+", default=BINNINGS, choices=BINNINGS)
parser.add_argument("--window", type=float, default=WINDOW, help="sliding window (cm)")
parser.add_argument("--step", type=float, default=STEP, help="sliding step (cm)")
parser.add_argument("--pid-shuffles", type=int, default=PID_SHUFFLES)
parser.add_argument("--redundancy", default=REDUNDANCY_MEASURE, choices=sorted(REDUNDANCY_MEASURES))
args = parser.parse_args()
scope = scope_from_args(args)
AGG = scope.aggregate
UNIT = unit_noun(AGG)
AGG_SUFFIX = "" if AGG == "pooled" else f"__{AGG}"
STRATEGIES = args.strategies or BINNING_STRATEGIES
TARGETS = STRATEGIES + ([CHOICE_TARGET] if INCLUDE_CHOICE_TARGET else [])
BINS_RUN = list(args.binnings)
if abs(args.window / args.step - round(args.window / args.step)) > 1e-9:
    raise SystemExit("the sliding window must be a multiple of the step")

paths = get_pipeline_paths(__file__)
table_dir, figdir = scoped_dirs(paths, scope)
clear_figures(figdir)
for _sub in figdir.glob("*/"):
    clear_figures(_sub)
set_style()
print(scope.describe())
print("Targets (stimulus binning strategies + choice):\n" + describe_strategies(TARGETS))
inp = ptt.session_inputs(paths, scope, PROTOCOL)


def process_session(ses, seed):
    sid = ses.session_id
    cells = inp.cells_by_session.get(sid)
    if not cells or ses.celldata is None or ses.calciumdata is None or ses.behaviordata is None:
        return []
    pairs = ptt.sample_group_pairs(ses, cells, args.max_pairs, "pairs3f", RANDOM_STATE)
    if pairs is None:
        return []
    use = sorted(set(pairs["cell_i"]) | set(pairs["cell_j"]))
    sw = ptt.SessionWindows(ses, inp.mu_sigma_lookup, use, TARGETS, S_PRE, S_POST, BINSIZE, args.window, args.step, BINS_RUN)
    if not sw.ok:
        return []
    I = pairs["cell_i"].map(sw.col_of).to_numpy()
    J = pairs["cell_j"].map(sw.col_of).to_numpy()
    keep = ~(sw.bad[I] | sw.bad[J])
    I, J, gcode = I[keep], J[keep], pairs["group"].map(ptt.GCODE).to_numpy()[keep]
    rng = np.random.default_rng(seed)
    recs = []
    for binning, s_rel, TM, cnt in sw.windows():
        for target in TARGETS:
            lab = sw.lab[target]
            ok = ptt.valid_trials(lab, cnt > 0)
            if ok is None:
                continue
            X, y = TM[ok], lab[ok]
            res = [pid_from_binned_responses(X[:, i], X[:, j], y, bins=R_BINS, redundancy_measure=args.redundancy,
                                             n_shuffles=args.pid_shuffles, rng=rng) for i, j in zip(I, J)]
            vals = {v: np.array([getattr(r, v) for r in res]) for v in STORE_VALUES}
            recs += ptt.moment_records(dict(session_id=sid, animal_id=ses.animal_id, binning=binning, target=target,
                                            s_rel=s_rel, n_trials=int(ok.sum())), ptt.group_moments(vals, gcode))
    print(f"  [{sid}] {len(pairs)} pairs, {len(recs)} rows")
    return recs


records = run_sessions(
    inp.session_ids, process_session, [PROTOCOL], paths.store, "pid_temporal",
    key_params=dict(s_pre=S_PRE, s_post=S_POST, binsize=BINSIZE, window=args.window, step=args.step, binnings=BINS_RUN,
                    r_bins=R_BINS, pid_shuffles=args.pid_shuffles, redundancy=args.redundancy, max_pairs=args.max_pairs,
                    targets=target_signature(TARGETS), min_trials=cfg.MIN_TRIALS_ANALYSIS,
                    min_per_bin=cfg.MIN_TRIALS_PER_TARGET_BIN, version="3f-v4", fits_mtime=int(inp.fits_path.stat().st_mtime)),
    load_kwargs=dict(load_behaviordata=True, load_videodata=True, load_celldata=True, load_calciumdata=True),
    cells_by_session=inp.cells_by_session, base_seed=RANDOM_STATE, n_jobs=scope.n_jobs, recompute=args.recompute,
    verbose=JOBLIB_VERBOSE, extra_key_by_session=inp.stamps, flatten=True)
if not records:
    raise SystemExit("No pair had enough valid trials in any position bin.")
mom = pd.DataFrame(records)
mom.to_csv(table_dir / "pair_pid_temporal_moments.csv", index=False)
print(f"\n{len(mom)} moment rows")

curves = ptt.moment_curves(mom, STORE_VALUES, ["binning", "target", "group", "s_rel"], AGG, MIN_PAIRS_PER_UNIT)
curves.to_csv(table_dir / f"pair_pid_temporal_by_group{AGG_SUFFIX}.csv", index=False)
agg_note = {"pooled": "mean +- SEM over all pairs", "session": "mean +- SEM over sessions",
            "mouse": "mean +- SEM over mice"}[AGG]

for binning in BINS_RUN:
    curves_b = curves[curves["binning"] == binning]
    bn_note = f"fixed {BINSIZE:g} cm bins" if binning == "fixed" else f"sliding {args.window:g} cm window, step {args.step:g}"
    for target in TARGETS:
        cur = curves_b[curves_b["target"] == target]
        if cur.empty:
            continue
        tag = f"{target} [{' | '.join(target_bin_names(target))}]"
        tdir = figdir / binning / target
        tdir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), squeeze=False, sharex=True)       # components, one panel per group
        for ax, g in zip(axes.flat, SIX_GROUP_ORDER):
            c = cur[cur["group"] == group_key(g)]
            if c.empty:
                ax.set_title(f"{SIX_GROUP_LABELS[g]} (no pairs)", color=SIX_GROUP_COLORS[g])
                continue
            for term in TERMS:
                ptt.draw_lines(ax, c, term, TERM_COLORS[term], label=term, aggregate=AGG)
            ptt.draw_lines(ax, c, "I12", "k", label="I12", ls="--", lw=1.5, band=False)
            ax.axhline(0, color="grey", lw=0.7)
            ax.axvline(0, color="black", ls=":", lw=1)
            ax.set_xlabel("position rel. to stimulus onset (cm)")
            ax.set_ylabel("information (bits)")
            ax.set_title(f"{SIX_GROUP_LABELS[g]} (n_pairs={int(c['n_pairs'].max())})", color=SIX_GROUP_COLORS[g])
        axes[0][0].legend(fontsize=7, frameon=False)
        fig.suptitle(f"Pairwise PID over position ({bn_note}) -- {tag}; {agg_note}", y=1.02, fontsize=9)
        fig.tight_layout()
        save_fig(fig, tdir, "1_pid_vs_position")
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))                                    # groups compared per quantity
        for ax, val in zip(axes, ["I12", "redundancy", "synergy"]):
            for g in SIX_GROUP_ORDER:
                c = cur[cur["group"] == group_key(g)]
                ptt.draw_lines(ax, c, val, SIX_GROUP_COLORS[g], label=SIX_GROUP_LABELS[g], aggregate=AGG)
            ax.axhline(0, color="grey", lw=0.7)
            ax.axvline(0, color="black", ls=":", lw=1)
            ax.set_xlabel("position rel. to stimulus onset (cm)")
            ax.set_ylabel(f"{val} (bits)")
            ax.legend(fontsize=6, frameon=False)
        fig.suptitle(f"Groups compared over position ({bn_note}) -- {tag}; {agg_note}", y=1.03, fontsize=9)
        fig.tight_layout()
        save_fig(fig, tdir, "2_group_comparison")
        plt.close(fig)

    for val in ("I12", "synergy"):
        fig, axes = plt.subplots(len(TARGETS), 1, figsize=(7, 3.4 * len(TARGETS)), squeeze=False)
        for ax, target in zip(axes[:, 0], TARGETS):
            cur = curves_b[curves_b["target"] == target]
            for g in SIX_GROUP_ORDER:
                ptt.draw_lines(ax, cur[cur["group"] == group_key(g)], val, SIX_GROUP_COLORS[g], label=SIX_GROUP_LABELS[g],
                               aggregate=AGG)
            ax.axhline(0, color="grey", lw=0.7)
            ax.axvline(0, color="black", ls=":", lw=1)
            ax.set_title(f"{val} -- {target}", fontsize=9)
            ax.set_xlabel("position rel. to stimulus onset (cm)")
            ax.legend(fontsize=6, frameon=False)
        fig.suptitle(f"{val} over position ({bn_note}), by target; {agg_note}", y=1.0, fontsize=9)
        fig.tight_layout()
        save_fig(fig, figdir / binning, f"compare_targets_{val}")
        plt.close(fig)
print(f"\nWrote tables to {table_dir}, figures to {figdir}")
