# -*- coding: utf-8 -*-
"""
3d_pair_activity_statistics_temporal
====================================
Position-resolved version of 3a: signal / noise / raw correlation of each pair at every position bin (relative to
stimulus onset) instead of once per coarse window, so a relationship that is only transient is not averaged away.
Definitions are 3a's (signal = correlation of the target-bin means across trials, noise = correlation of the residuals after
removing each cell's target-bin mean, raw = plain correlation); no shuffle test here (see 3a for the conditioned shuffles).
Each window = the trial's mean activity over the frames of the engaged trial inside the window, correlated across trials (the
`trial_mean` method of 3h; 3h also has the frame-pooled method, the time axis and more).

BINS: `fixed` = non-overlapping bins of BINSIZE cm; `sliding` = windows of WINDOW cm moved by STEP cm (neighbouring windows
share data: the curve is smoother and less noisy, its points are not independent). A frame is used only if it lies in ITS OWN
engaged trial's [tStart, tEnd].
PAIRS: MAX_PAIRS_PER_GROUP = 150 per (session, area, label), sampled here from the 2a + 2b included cells (as 3g / 3h), no
longer 3a's list of 8: the pairs therefore differ from 3a's.
TARGETS: stim__<strategy> (zero_to80_above80, pct_0-20_20-80_80-100, zero_vs_nonzero), `choice`, and `all_levels` (every stimulus
value its own bin: the best-defined signal correlation, 6 points). signal_corr needs >= 3 target bins, it is empty for the
2-bin targets (the correlation of 2 points is always +-1).

Project-wide rules (infotheory/project_config.py) -- what applies here
    * selection : --animals / --sessions restrict the sessions; a restricted run writes to out/selection/<tag>/.
    * aggregate : --aggregate pooled|session|mouse. Curves = per group, binning, target and position bin, the mean over
                  units of the unit's mean over its pairs (units with < MIN_PAIRS_PER_UNIT pairs at that bin left out),
                  band = SEM over units; pooled: mean over all pairs, band = SEM over pairs (pairs treated as independent).
                  Always the six groups.
    * memory    : one session in RAM per worker (--n-jobs); correlations of all pairs of a window are computed together and
                  reduced to per-(session, group) sums, no per-pair table is written (150 pairs x windows x targets would be
                  millions of rows).
    * binning   : the same targets, trial rule (finite values, target bins with < MIN_TRIALS_PER_TARGET_BIN trials dropped,
                  >= MIN_TRIALS_ANALYSIS trials, 2 labels) as 3a-3c. Helpers: infotheory.pair_temporal.
    * shading   : the position plots shade 3a's windows (-30..-10, 0..20, 25..45 cm).
    CHANGES vs the previous 3d: 150 pairs sampled here (before: 3a's list of 8); fixed bins + sliding windows; all_levels
    target; own-trial frame filter; per-pair table replaced by moments; figures in figures/<binning>/<target>/.

Outputs
-------
2_pipeline/3d_pair_activity_statistics_temporal/
    out/    pair_correlations_temporal_moments.csv    per (session, binning, target, bin, group): n_trials and, for each
                                                      correlation, n / sum / sum of squares over the pairs
            pair_correlations_temporal_by_group[__<aggregate>].csv   curves per group x binning x target x s_rel
            figures/<binning>/<target>/1_correlation_vs_position.png, figures/<binning>/compare_targets_*.png
    out/selection/<tag>/...                           the same for a restricted run
    store/  temporal_corr_<hash>.pkl                  per-session caches
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from infotheory import project_config as cfg
from infotheory.pipeline import get_pipeline_paths
from infotheory.celldata_utils import SIX_GROUP_ORDER, SIX_GROUP_LABELS, SIX_GROUP_COLORS
from infotheory.groups import group_key
from infotheory.plotting import set_style, save_fig, clear_figures
from infotheory.scope import add_scope_args, scope_from_args, scoped_dirs, unit_noun
from infotheory.session_jobs import run_sessions
from infotheory.stim_binning import (STIM_BINNING_STRATEGIES, DEFAULT_STRATEGIES, CHOICE_TARGET, target_signature,
                                     target_bin_names, describe_strategies)
from infotheory import pair_temporal as ptt

PROTOCOL = "DN"
S_PRE, S_POST, BINSIZE = -60.0, 80.0, 15.0
MAX_PAIRS_PER_GROUP = 150   # per (session, area, label)
WINDOW, STEP = 20.0, 5.0     # sliding window and step (cm)
BINNINGS = ["fixed", "sliding"]
WINDOWS = {"pre_stim": (-30.0, -10.0), "stim": (0.0, 20.0), "reward": (25.0, 45.0)}   # 3a's windows (cm), shaded
MIN_PAIRS_PER_UNIT = 5
RANDOM_STATE = 0
BINNING_STRATEGIES = list(DEFAULT_STRATEGIES) + ["all_levels"]
INCLUDE_CHOICE_TARGET = True
JOBLIB_VERBOSE = 5
METRICS = ["raw_corr", "noise_corr", "signal_corr"]      # raw first: its pair count defines n_pairs / n_units
PLOT_ORDER = ["signal_corr", "noise_corr", "raw_corr"]
SHARE_Y = True               # panels of one metric share the y range

parser = argparse.ArgumentParser()
add_scope_args(parser)
parser.add_argument("--strategies", nargs="+", default=None, choices=sorted(STIM_BINNING_STRATEGIES))
parser.add_argument("--max-pairs", type=int, default=MAX_PAIRS_PER_GROUP, help="pairs per (session, area, label)")
parser.add_argument("--binnings", nargs="+", default=BINNINGS, choices=BINNINGS)
parser.add_argument("--window", type=float, default=WINDOW, help="sliding window (cm)")
parser.add_argument("--step", type=float, default=STEP, help="sliding step (cm)")
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
    pairs = ptt.sample_group_pairs(ses, cells, args.max_pairs, "pairs3d", RANDOM_STATE)
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
    recs = []
    for binning, s_rel, TM, cnt in sw.windows():
        for target in TARGETS:
            lab = sw.lab[target]
            ok = ptt.valid_trials(lab, cnt > 0)
            if ok is None:
                continue
            _, y = np.unique(lab[ok], return_inverse=True)
            vals = ptt.corr_metrics(TM[ok], y, I, J)
            recs += ptt.moment_records(dict(session_id=sid, animal_id=ses.animal_id, binning=binning, target=target,
                                            s_rel=s_rel, n_trials=int(ok.sum())), ptt.group_moments(vals, gcode))
    print(f"  [{sid}] {len(pairs)} pairs, {len(recs)} rows")
    return recs


records = run_sessions(
    inp.session_ids, process_session, [PROTOCOL], paths.store, "temporal_corr",
    key_params=dict(s_pre=S_PRE, s_post=S_POST, binsize=BINSIZE, window=args.window, step=args.step, binnings=BINS_RUN,
                    max_pairs=args.max_pairs, targets=target_signature(TARGETS), min_trials=cfg.MIN_TRIALS_ANALYSIS,
                    min_per_bin=cfg.MIN_TRIALS_PER_TARGET_BIN, version="3d-v5", fits_mtime=int(inp.fits_path.stat().st_mtime)),
    load_kwargs=dict(load_behaviordata=True, load_videodata=True, load_celldata=True, load_calciumdata=True),
    cells_by_session=inp.cells_by_session, base_seed=RANDOM_STATE, n_jobs=scope.n_jobs, recompute=args.recompute,
    verbose=JOBLIB_VERBOSE, extra_key_by_session=inp.stamps, flatten=True)
if not records:
    raise SystemExit("No pair had enough valid trials in any position bin.")
mom = pd.DataFrame(records)
mom.to_csv(table_dir / "pair_correlations_temporal_moments.csv", index=False)
print(f"\n{len(mom)} moment rows")

curves = ptt.moment_curves(mom, METRICS, ["binning", "target", "group", "s_rel"], AGG, MIN_PAIRS_PER_UNIT)
curves.to_csv(table_dir / f"pair_correlations_temporal_by_group{AGG_SUFFIX}.csv", index=False)
agg_note = {"pooled": "mean +- SEM over all pairs", "session": "mean +- SEM over sessions",
            "mouse": "mean +- SEM over mice"}[AGG]


def group_lines(ax, cur, metric):
    for g in SIX_GROUP_ORDER:
        c = cur[cur["group"] == group_key(g)].dropna(subset=[metric])
        ptt.draw_lines(ax, c, metric, SIX_GROUP_COLORS[g],
                       label=f"{SIX_GROUP_LABELS[g]} (n={int(c['n_pairs'].max()) if len(c) else 0})", aggregate=AGG)
    for lo, hi in WINDOWS.values():
        ax.axvspan(lo, hi, color="grey", alpha=0.15, lw=0, zorder=0)
    ax.axvline(0, color="black", ls=":", lw=1)
    ax.axhline(0, color="grey", lw=0.7)
    ax.set_xlabel("position rel. to stimulus onset (cm)")
    ax.set_ylabel(metric)
    ax.legend(fontsize=6, frameon=False)


for binning in BINS_RUN:
    curves_b = curves[curves["binning"] == binning]
    bn_note = f"fixed {BINSIZE:g} cm bins" if binning == "fixed" else f"sliding {args.window:g} cm window, step {args.step:g}"
    for target in TARGETS:
        cur = curves_b[curves_b["target"] == target]
        if cur.empty:
            continue
        tdir = figdir / binning / target
        tdir.mkdir(parents=True, exist_ok=True)
        metrics = [m for m in PLOT_ORDER if cur[m].notna().any()]
        fig, axes = plt.subplots(1, len(metrics), figsize=(5.0 * len(metrics), 4.5), squeeze=False,
                                 sharey=False)
        for ax, metric in zip(axes[0], metrics):
            group_lines(ax, cur, metric)
        fig.suptitle(f"Pairwise correlation over position ({bn_note}) -- {target} [{' | '.join(target_bin_names(target))}]; "
                     f"{agg_note}", y=1.03, fontsize=9)
        fig.tight_layout()
        save_fig(fig, tdir, "1_correlation_vs_position")
        plt.close(fig)

    for metric in PLOT_ORDER:
        targets_m = [t for t in TARGETS if curves_b[(curves_b["target"] == t)][metric].notna().any()]
        if not targets_m:
            continue
        fig, axes = plt.subplots(len(targets_m), 1, figsize=(7, 3.4 * len(targets_m)), squeeze=False,
                                 sharey=SHARE_Y)
        for ax, target in zip(axes[:, 0], targets_m):
            group_lines(ax, curves_b[curves_b["target"] == target], metric)
            ax.set_title(f"{metric} -- {target}", fontsize=9)
        fig.suptitle(f"{metric} over position ({bn_note}), by target; {agg_note}", y=1.0, fontsize=9)
        fig.tight_layout()
        save_fig(fig, figdir / binning, f"compare_targets_{metric}")
        plt.close(fig)
print(f"\nWrote tables to {table_dir}, figures to {figdir}")
