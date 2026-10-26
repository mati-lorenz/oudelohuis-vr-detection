# -*- coding: utf-8 -*-
"""
behavior_pair_runner.py
=======================
The shared skeleton of 3i / 3j / 3k / 3l (pair analyses with a BEHAVIORAL target): arguments, per-session loop, moments,
curves, figures. A script supplies only what is specific to it (`compute`, the value names and the figure layout), so the four
scripts cannot drift apart in pairs, windows, trial rules or aggregation (project rule 4: they share ONE binning).

Per session (one in RAM per worker, cached in store/):
    1. pairs = `max_pairs` per (area, label) of the 2a + 2b included cells, the SAME sample in all four scripts (tag
       "pairs_behavior"), the project group of each pair;
    2. for every window (axis x binning, behavior_pairs.BehaviorWindows) and every behavior: the target = quantile bins of the
       trial-mean behavior in that window; trials by the project rule (pair_temporal.valid_trials); `compute(X, y, I, J, rng,
       args)` -> {value: array over pairs};
    3. per-pair values are reduced to per-(session, group) n / sum / sum of squares (no per-pair table is written).
Then the curves (mean +- SEM per group, aggregate = pooled | session | mouse as everywhere) and the figures:
    figures/<axis>/<binning>/<behavior>/1_<stem>_vs_<axis>.png   components, one panel per group
    figures/<axis>/<binning>/<behavior>/2_group_comparison.png   one panel per compared value, six groups
    figures/<axis>/<binning>/compare_behaviors_<value>.png       behaviors stacked (shared y)
    figures/position/windows/<value>_by_window.png               bars: 3a-3c's three windows, groups side by side
The rng of a session is seeded from the session id, so 3j and 3k (same pairs, windows, behaviors, same order of MI calls)
give identical I_total / I_lin and the numbers of 3i, 3j, 3k, 3l refer to the same trials.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import project_config as cfg
from . import pair_temporal as ptt
from .pipeline import get_pipeline_paths
from .celldata_utils import SIX_GROUP_ORDER, SIX_GROUP_LABELS, SIX_GROUP_COLORS
from .groups import group_key
from .plotting import set_style, save_fig, clear_figures
from .scope import add_scope_args, scope_from_args, scoped_dirs
from .session_jobs import run_sessions
from .behavior_pairs import BehaviorWindows, behavior_labels, BEHAVIOR_VARS, AXES_DEF, WINDOWS, BINNINGS

PROTOCOL = "DN"
MAX_PAIRS_PER_GROUP = 150
MIN_PAIRS_PER_UNIT = 5
RANDOM_STATE = 0
PAIR_TAG = "pairs_behavior"
JOBLIB_VERBOSE = 5
SHARE_Y = True
KEYS = ["axis", "binning", "window", "behavior", "group", "s_rel"]


def run(script_file, *, stem, store_name, version, what, components, compare, ylabel, compute, store_values,
        total=None, add_args=None, key_extra=None):
    """components: [(value, color)] drawn per group panel; total: value drawn dashed black; compare: values compared across
    groups; store_values: every value `compute` returns (the FIRST one must be defined for every pair that has any value: it
    sets n_pairs / n_units); compute(X, y, I, J, rng, args) -> dict value -> array over pairs."""
    parser = argparse.ArgumentParser()
    add_scope_args(parser)
    parser.add_argument("--behaviors", nargs="+", default=BEHAVIOR_VARS, choices=BEHAVIOR_VARS)
    parser.add_argument("--max-pairs", type=int, default=MAX_PAIRS_PER_GROUP, help="pairs per (session, area, label)")
    parser.add_argument("--axes", nargs="+", default=list(AXES_DEF), choices=list(AXES_DEF))
    parser.add_argument("--binnings", nargs="+", default=BINNINGS, choices=BINNINGS,
                        help="windows = 3a-3c's three position windows (position axis only)")
    for ax_ in AXES_DEF:
        parser.add_argument(f"--{ax_}-bin", type=float, default=AXES_DEF[ax_]["binsize"], help=f"fixed {ax_} bin")
        parser.add_argument(f"--{ax_}-window", type=float, default=AXES_DEF[ax_]["window"], help=f"sliding {ax_} window")
        parser.add_argument(f"--{ax_}-step", type=float, default=AXES_DEF[ax_]["step"], help=f"sliding {ax_} step")
    if add_args:
        add_args(parser)
    args = parser.parse_args()
    axes_def = {k: dict(v) for k, v in AXES_DEF.items()}
    for ax_ in axes_def:
        axes_def[ax_]["binsize"] = getattr(args, f"{ax_}_bin")
        axes_def[ax_]["window"] = getattr(args, f"{ax_}_window")
        axes_def[ax_]["step"] = getattr(args, f"{ax_}_step")
        r_ = axes_def[ax_]["window"] / axes_def[ax_]["step"]
        if abs(r_ - round(r_)) > 1e-9:
            raise SystemExit(f"{ax_}: the sliding window must be a multiple of the step")
    scope = scope_from_args(args)
    agg = scope.aggregate
    agg_suffix = "" if agg == "pooled" else f"__{agg}"
    agg_note = {"pooled": "mean +- SEM over all pairs", "session": "mean +- SEM over sessions",
                "mouse": "mean +- SEM over mice"}[agg]
    behaviors, axes, binnings = list(args.behaviors), list(args.axes), list(args.binnings)

    paths = get_pipeline_paths(script_file)
    table_dir, figdir = scoped_dirs(paths, scope)
    clear_figures(figdir)
    for sub in figdir.glob("*/*/"):
        clear_figures(sub)
        for sub2 in sub.glob("*/"):
            clear_figures(sub2)
    set_style()
    print(scope.describe())
    print(f"Targets: trial-mean {', '.join(behaviors)} in each window, {cfg.RESPONSE_BINS} quantile bins; axes {axes}; "
          f"binnings {binnings}")
    inp = ptt.session_inputs(paths, scope, PROTOCOL)

    def process_session(ses, seed):
        sid = ses.session_id
        cells = inp.cells_by_session.get(sid)
        if not cells or ses.celldata is None or ses.calciumdata is None or ses.behaviordata is None:
            return []
        pairs = ptt.sample_group_pairs(ses, cells, args.max_pairs, PAIR_TAG, RANDOM_STATE)
        if pairs is None:
            return []
        use = sorted(set(pairs["cell_i"]) | set(pairs["cell_j"]))
        bw = BehaviorWindows(ses, use, behaviors, axes_def, axes, binnings)
        if not bw.ok:
            return []
        I = pairs["cell_i"].map(bw.col_of).to_numpy()
        J = pairs["cell_j"].map(bw.col_of).to_numpy()
        keep = ~(bw.bad[I] | bw.bad[J])
        I, J, gcode = I[keep], J[keep], pairs["group"].map(ptt.GCODE).to_numpy()[keep]
        rng = np.random.default_rng(seed)
        recs = []
        for w in bw.windows():
            for k, beh in enumerate(bw.behaviors):
                lab = behavior_labels(w["TMB"][:, k], w["cnt"] > 0)
                ok = ptt.valid_trials(lab, w["cnt"] > 0)
                if ok is None:
                    continue
                vals = compute(w["TM"][ok], lab[ok], I, J, rng, args)
                recs += ptt.moment_records(dict(session_id=sid, animal_id=ses.animal_id, axis=w["axis"], binning=w["binning"],
                                                window=w["window"], behavior=beh, s_rel=w["s_rel"], n_trials=int(ok.sum())),
                                           ptt.group_moments(vals, gcode))
        print(f"  [{sid}] {len(pairs)} pairs, {len(recs)} rows")
        return recs

    key = dict(axes_def={a: axes_def[a] for a in axes}, axes=axes, binnings=binnings, behaviors=behaviors,
               r_bins=cfg.RESPONSE_BINS, max_pairs=args.max_pairs, min_trials=cfg.MIN_TRIALS_ANALYSIS,
               min_per_bin=cfg.MIN_TRIALS_PER_TARGET_BIN, windows=WINDOWS, version=version,
               fits_mtime=int(inp.fits_path.stat().st_mtime))
    if key_extra:
        key.update(key_extra(args))
    records = run_sessions(
        inp.session_ids, process_session, [PROTOCOL], paths.store, store_name, key_params=key,
        load_kwargs=dict(load_behaviordata=True, load_videodata=True, load_celldata=True, load_calciumdata=True),
        cells_by_session=inp.cells_by_session, base_seed=RANDOM_STATE, n_jobs=scope.n_jobs, recompute=args.recompute,
        verbose=JOBLIB_VERBOSE, extra_key_by_session=inp.stamps, flatten=True)
    if not records:
        raise SystemExit("No pair had enough valid trials in any window.")
    mom = pd.DataFrame(records)
    mom.to_csv(table_dir / f"{stem}_moments.csv", index=False)
    print(f"\n{len(mom)} moment rows")
    curves = ptt.moment_curves(mom, store_values, KEYS, agg, MIN_PAIRS_PER_UNIT)
    curves.to_csv(table_dir / f"{stem}_by_group{agg_suffix}.csv", index=False)

    # ----------------------------------------------------------------------------------------------------------- figures
    def decorate(ax, axis):
        if axis == "position":
            for lo, hi in WINDOWS.values():
                ax.axvspan(lo, hi, color="grey", alpha=0.15, lw=0, zorder=0)
        ax.axvline(0, color="black", ls=":", lw=1)
        ax.axhline(0, color="grey", lw=0.7)
        ax.set_xlabel(axes_def[axis]["label"])

    def bn_note(axis, binning):
        ad = axes_def[axis]
        u = ad["unit"]
        return (f"fixed {ad['binsize']:g} {u} bins" if binning == "fixed"
                else f"sliding {ad['window']:g} {u} window, step {ad['step']:g}")

    for axis in axes:
        for binning in binnings:
            cb = curves[(curves["axis"] == axis) & (curves["binning"] == binning)]
            if cb.empty or binning == "windows":
                continue
            note = bn_note(axis, binning)
            for beh in behaviors:
                cur = cb[cb["behavior"] == beh]
                if cur.empty:
                    continue
                tdir = figdir / axis / binning / beh
                tdir.mkdir(parents=True, exist_ok=True)
                fig, axs = plt.subplots(2, 3, figsize=(14, 8), squeeze=False, sharex=True, sharey=SHARE_Y)
                for ax, g in zip(axs.flat, SIX_GROUP_ORDER):
                    c = cur[cur["group"] == group_key(g)]
                    if c.empty:
                        ax.set_title(f"{SIX_GROUP_LABELS[g]} (no pairs)", color=SIX_GROUP_COLORS[g])
                        continue
                    for val, col in components:
                        ptt.draw_lines(ax, c.dropna(subset=[val]), val, col, label=val, aggregate=agg)
                    if total:
                        ptt.draw_lines(ax, c.dropna(subset=[total]), total, "k", label=total, ls="--", band=False)
                    decorate(ax, axis)
                    ax.set_ylabel(ylabel)
                    ax.set_title(f"{SIX_GROUP_LABELS[g]} (n_pairs={int(c['n_pairs'].max())})", color=SIX_GROUP_COLORS[g])
                axs[0][0].legend(fontsize=7, frameon=False)
                fig.suptitle(f"{what} over {axis} ({note}) -- target: {beh} (trial mean, {cfg.RESPONSE_BINS} bins); {agg_note}",
                             y=1.02, fontsize=9)
                fig.tight_layout()
                save_fig(fig, tdir, f"1_{stem}_vs_{axis}")
                plt.close(fig)

                fig, axs = plt.subplots(1, len(compare), figsize=(5.0 * len(compare), 4.5), squeeze=False)
                for ax, val in zip(axs[0], compare):
                    for g in SIX_GROUP_ORDER:
                        c = cur[cur["group"] == group_key(g)].dropna(subset=[val])
                        ptt.draw_lines(ax, c, val, SIX_GROUP_COLORS[g], label=SIX_GROUP_LABELS[g], aggregate=agg)
                    decorate(ax, axis)
                    ax.set_ylabel(val)
                    ax.legend(fontsize=6, frameon=False)
                fig.suptitle(f"Groups compared over {axis} ({note}) -- target: {beh}; {agg_note}", y=1.03, fontsize=9)
                fig.tight_layout()
                save_fig(fig, tdir, "2_group_comparison")
                plt.close(fig)

            for val in compare:
                behs = [b for b in behaviors if cb[cb["behavior"] == b][val].notna().any()]
                if not behs:
                    continue
                fig, axs = plt.subplots(len(behs), 1, figsize=(7, 3.4 * len(behs)), squeeze=False, sharex=True,
                                        sharey=SHARE_Y)
                for ax, beh in zip(axs[:, 0], behs):
                    cur = cb[cb["behavior"] == beh]
                    for g in SIX_GROUP_ORDER:
                        c = cur[cur["group"] == group_key(g)].dropna(subset=[val])
                        ptt.draw_lines(ax, c, val, SIX_GROUP_COLORS[g], label=SIX_GROUP_LABELS[g], aggregate=agg)
                    decorate(ax, axis)
                    ax.set_title(f"{val} -- target: {beh}", fontsize=9)
                    ax.set_ylabel(val)
                    ax.legend(fontsize=6, frameon=False)
                fig.suptitle(f"{val} over {axis} ({note}), by behavior; {agg_note}", y=1.0, fontsize=9)
                fig.tight_layout()
                save_fig(fig, figdir / axis / binning, f"compare_behaviors_{val}")
                plt.close(fig)

    # bars for 3a-3c's three windows
    cw = curves[(curves["axis"] == "position") & (curves["binning"] == "windows")]
    if not cw.empty:
        wdir = figdir / "position" / "windows"
        wdir.mkdir(parents=True, exist_ok=True)
        names = list(WINDOWS)
        width = 0.8 / len(SIX_GROUP_ORDER)
        for val in compare:
            behs = [b for b in behaviors if cw[cw["behavior"] == b][val].notna().any()]
            if not behs:
                continue
            fig, axs = plt.subplots(1, len(behs), figsize=(4.6 * len(behs), 4.3), squeeze=False, sharey=SHARE_Y)
            for ax, beh in zip(axs[0], behs):
                cur = cw[cw["behavior"] == beh]
                for gi, g in enumerate(SIX_GROUP_ORDER):
                    c = cur[cur["group"] == group_key(g)].set_index("window")
                    xs = [wi - 0.4 + width * (gi + 0.5) for wi in range(len(names))]
                    ys = [c[val].get(n, np.nan) for n in names]
                    es = [c[f"{val}_sem"].get(n, 0.0) for n in names]
                    ax.bar(xs, ys, width=width * 0.92, color=SIX_GROUP_COLORS[g], alpha=0.85, yerr=es,
                           error_kw=dict(lw=0.8, capsize=1.5), label=SIX_GROUP_LABELS[g])
                ax.axhline(0, color="grey", lw=0.7)
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels([f"{n}\n{WINDOWS[n][0]:g}..{WINDOWS[n][1]:g} cm" for n in names], fontsize=7)
                ax.set_title(f"target: {beh}", fontsize=9)
                ax.set_ylabel(val)
            axs[0][0].legend(fontsize=6, frameon=False)
            fig.suptitle(f"{val} in 3a-3c's position windows; {agg_note}", y=1.02, fontsize=9)
            fig.tight_layout()
            save_fig(fig, wdir, f"{val}_by_window")
            plt.close(fig)
    print(f"\nWrote tables to {table_dir}, figures to {figdir}")
