# -*- coding: utf-8 -*-
"""
1h_pupil_speed_vs_task
=======================
The REVERSE direction of 1f/1g and of 1d's section 7: there, pupil area and running speed are the PREDICTOR
(does pre-stimulus arousal predict accuracy / tuning of other behavior?). Here they are the OUTCOME: how does
pupil area and running speed change with (A) the STRENGTH of the stimulus on that trial and (B) whether the
trial was answered CORRECTLY, in each of the three trial windows (pre_stim / stim / reward, same as 1f/1g/1d)?
Across all three protocols (DM, DP, DN).

y  = pupil area (z-scored WITHIN session over the whole continuous trace, before windowing -- the project's
     standing rule: only FACE quantities are z-scored, never running speed) and running speed (left in cm/s).
     WINDOWS below are the window means of each, same three windows and same cm ranges as 1f's D_WINDOWS /
     1g's TUNING_WINDOWS / 1d's PRESTIM_WINDOW, so the numbers are directly comparable across scripts.

x, two independent views of "what was happening on the trial":
  A. STIMULUS STRENGTH
     A1. a continuous, per-session quantile view (N_QUANTILE_BINS, same logic as 1f/1g's own quantile bins):
         the RAW signal, cut into equal-occupancy bins within each session. Works for every protocol, no
         psychometric fit needed.
     A2. the SHARED stimulus-binning strategies of infotheory/stim_binning.py (STIM_STRATEGIES below -- the
         same strategies 1d uses as predictors of accuracy, and the project's rule 4: binning must be shared
         across scripts). These need stim_pct (position on the 1b psychometric curve), so curve-based
         strategies are only defined for protocols with a usable 1b fit (DN); a strategy with fewer than 2
         bins for a protocol is simply skipped for it, same as 1d.
  B. PERFORMANCE: correct / incorrect, generalized to every signal level exactly as in 1d/1f --
         correct = (signal > 0 AND lickResponse == 1) OR (signal == 0 AND lickResponse == 0).
     A unit's correct and incorrect trials are two disjoint, non-overlapping subsets of its OWN trials, so this
     is a paired comparison (correct mean vs error mean per unit), not a fitted model.

Per-unit statistics, all DESCRIPTIVE -- see "Why no cross-validation" below:
  A1/A2. trend: per-unit Spearman rho of y against the bin index; group test = Wilcoxon signed-rank across
         units for rho != 0, Holm-corrected across windows within (protocol, y).
  B.     per-unit mean of y on correct trials minus mean of y on error trials; group test = Wilcoxon signed-rank
         across units (paired by construction: one correct mean and one error mean per unit), Holm-corrected
         across windows within (protocol, y). Shown as paired lines per unit (figure 3), the same style as
         1f's "two halves" check.
  D.     bonus interaction view (figure 4): the A1 tuning curve, split by B (correct vs error) -- does the
         pupil/speed-vs-stimulus-strength relationship differ between correct and error trials?

Why NO cross-validation (unlike 1f/2j/2k): every headline number here is a bin mean, a per-unit Spearman rho, or
a paired difference of two means on genuinely disjoint trial subsets -- never an R2/eta2-style quantity whose
effect size is read off the SAME data a multi-parameter model was fit to (the scoping rule this project settled
on for 1f/2j/2k: cross-validate a FIT-AND-SCORE-ON-THE-SAME-DATA effect size, not every statistic in the
pipeline). A single correlation coefficient or a mean-of-a-fixed-subset has no such bias to correct, exactly
like 1f's two-halves check, 1g (nothing to cross-validate there either), or the MI-vs-linear-r2 comparisons used
throughout the project.

Project-wide rules (infotheory/project_config.py)
    * selection : --animals / --sessions restrict the sessions (always intersected with 1b's included list); a
                  restricted run writes to out/selection/<tag>/ and never overwrites the full-dataset out/.
    * aggregate : --aggregate pooled|session|mouse = the unit of analysis of every figure and table.
    * memory    : each session is loaded, reduced to its small per-trial table, cached in store/ and freed.
    * binning   : the stimulus-strategy bins (A2) are the SAME ones 1d uses as predictors of accuracy.
    * z-scoring : only pupil area (a face quantity) is z-scored within session; running speed stays in cm/s.

Reads: 2_pipeline/1b_psychometric/out/psychometric_fits.csv (included sessions + per-session mu/sigma).
Loads: raw behaviordata/videodata, one session at a time.

Outputs
-------
2_pipeline/1h_pupil_speed_vs_task/
    out/    signal_quantile_tuning[__<aggregate>].csv        mean +- SEM of y per (protocol, window, y, signal quantile bin)
            signal_quantile_group_tests[__<aggregate>].csv   per-unit Spearman rho, Wilcoxon across units, Holm across windows
            stim_strategy_tuning[__<aggregate>].csv          mean +- SEM of y per (protocol, window, y, strategy, bin)
            stim_strategy_group_tests[__<aggregate>].csv     same trend test as above, for the shared strategy bins
            performance_comparison_per_unit[__<aggregate>].csv   per-unit mean y on correct / error trials, per window
            performance_comparison_group_tests[__<aggregate>].csv  paired Wilcoxon across units, Holm across windows
            binning_used.csv   which shared stimulus strategies were usable, per protocol
            figures/*.png
    out/selection/<tag>/...  the same for a restricted run
    store/  task_trials_<hash>.pkl   per-session trial table (reused unless --recompute)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sps

from infotheory.pipeline import get_pipeline_paths
from infotheory.session import discover_sessions, PROTOCOLS
from infotheory.continuous import merge_behavior_video, restrict_to_engaged, remove_pupil_outliers, \
    zscore_per_session, compute_trial_position_window_means
from infotheory.plotting import set_style, save_fig, clear_figures, PROTOCOL_COLORS
from infotheory import project_config as cfg
from infotheory.scope import add_scope_args, scope_from_args, select_sessions, scoped_dirs, data_stamp, unit_noun
from infotheory.session_jobs import run_sessions
from infotheory.unit_stats import unit_curve
from infotheory.unit_plots import mouse_palette, mouse_legend
from infotheory.stim_binning import STIM_BINNING_STRATEGIES, DEFAULT_STRATEGIES, psychometric_pct, \
    bin_stimulus, target_signature

# --- what is analysed (edit freely) ---------------------------------------------------------------------------
Y_VARS = ["pupil_area", "runspeed"]
Y_LABELS = {"pupil_area": "pupil area (z)", "runspeed": "speed (cm/s)"}
WINDOWS = {"pre_stim": (-30.0, -10.0), "stim": (0.0, 20.0), "reward": (25.0, 45.0)}   # cm rel. stimStart; same as 1f/1g/1d
STIM_STRATEGIES = list(DEFAULT_STRATEGIES)      # shared strategies (infotheory/stim_binning.py); edit THERE
STRATEGY_PREFERENCE = ["pct_0-20_20-80_80-100", "zero_to80_above80", "zero_vs_nonzero"]   # which usable strategy to PLOT per protocol

# --- shared with the other scripts (project_config.py): change them THERE, not here ----------------------
MIN_TRIALS_ANALYSIS = cfg.MIN_TRIALS_ANALYSIS

# --- local to this script ---------------------------------------------------------------------------------
N_QUANTILE_BINS = 8            # signal quantile bins within session (same logic as 1f/1g's own x variables)
MIN_TRIALS_PER_CLASS = 10      # correct / error trials needed in a unit's window for the B comparison
ALPHA = 0.05
MIN_UNITS = 4                  # units needed for a group test
JOBLIB_VERBOSE = 5

parser = argparse.ArgumentParser()
add_scope_args(parser)
args = parser.parse_args()
scope = scope_from_args(args)
AGG = scope.aggregate
UNIT = unit_noun(AGG)
N_JOBS = scope.n_jobs
AGG_SUFFIX = "" if AGG == "pooled" else f"__{AGG}"
unit_col = {"pooled": None, "session": "session_id", "mouse": "animal_id"}[AGG]
UNIT_KEYS = {"pooled": [], "session": ["animal_id", "session_id"], "mouse": ["animal_id"]}[AGG]

paths = get_pipeline_paths(__file__)
table_dir, figdir = scoped_dirs(paths, scope)
clear_figures(figdir)
set_style()
print(scope.describe())

unknown = [s for s in STIM_STRATEGIES if s not in STIM_BINNING_STRATEGIES]
if unknown:
    raise SystemExit(f"STIM_STRATEGIES has undefined strategies {unknown}; defined: {list(STIM_BINNING_STRATEGIES)}")

# --------------------------------------------------------------------- #
# 1. Included sessions (1b out/) intersected with the selection, plus each session's mu / sigma
# --------------------------------------------------------------------- #
fits_path = paths.out_from("1b_psychometric") / "psychometric_fits.csv"
if not fits_path.exists():
    raise SystemExit(f"Run 1b_psychometric.py first -- expected {fits_path}")
fits_df = pd.read_csv(fits_path)
included = fits_df[fits_df["included"]]
all_triples = discover_sessions(PROTOCOLS)
selected_ids = {f"{a}_{d}" for _, a, d in select_sessions(all_triples, scope)}
included_ids = [sid for sid in included["session_id"].tolist() if sid in selected_ids]
if not included_ids:
    raise SystemExit("No selected session passed 1b_psychometric's inclusion criteria -- nothing to analyze.")
print(f"Using {len(included_ids)} included sessions "
      f"({included[included['session_id'].isin(included_ids)]['protocol'].value_counts().to_dict()})")
triple_of = {f"{a}_{d}": (p, a, d) for p, a, d in all_triples}
mouse_colors = mouse_palette(a for _, a, _ in all_triples)
mu_sigma_lookup = fits_df.set_index("session_id")[["mu", "sigma"]].to_dict("index")


# --------------------------------------------------------------------- #
# 2. Per-session trial tables (one session in RAM at a time), cached
# --------------------------------------------------------------------- #
def _session_key(sid):
    ms = mu_sigma_lookup.get(sid, {})
    return f"{data_stamp(*triple_of[sid])}|{ms.get('mu')}|{ms.get('sigma')}"


def process_session(ses, seed):
    """One row per engaged trial: signal, stim_pct, correct, and the window means of Y_VARS in every WINDOWS."""
    if ses.trialdata is None or ses.behaviordata is None:
        return None
    merged_full = zscore_per_session(remove_pupil_outliers(merge_behavior_video(ses.behaviordata, ses.videodata)))
    if merged_full.empty or not all(v in merged_full.columns for v in Y_VARS):
        return None
    merged = restrict_to_engaged(merged_full, ses.trialdata)
    if merged.empty:
        return None
    trials = (ses.trialdata[ses.trialdata["engaged"] == 1]
              if "engaged" in ses.trialdata.columns else ses.trialdata.copy())
    trials = trials.assign(correct=(((trials["signal"] > 0) & (trials["lickResponse"] == 1))
                                     | ((trials["signal"] == 0) & (trials["lickResponse"] == 0))).astype(int))
    ms = mu_sigma_lookup.get(ses.session_id, {})
    trials["stim_pct"] = psychometric_pct(trials["signal"], ms.get("mu"), ms.get("sigma"))
    t = trials[["trialNumber", "signal", "stim_pct", "lickResponse", "correct"]].copy()
    for wname, (lo, hi) in WINDOWS.items():
        pw = compute_trial_position_window_means(trials, merged, columns=Y_VARS, s_start=lo, s_end=hi)
        pw = pw.rename(columns={v: f"{v}@{wname}" for v in Y_VARS})
        t = t.merge(pw[["trialNumber"] + [f"{v}@{wname}" for v in Y_VARS]], on="trialNumber", how="left")
    t = t.sort_values("trialNumber").reset_index(drop=True)
    t["session_id"], t["animal_id"], t["protocol"] = ses.session_id, ses.animal_id, ses.protocol
    return t


session_tables = run_sessions(
    included_ids, process_session, ['DN'], paths.store, "task_trials",
    key_params=dict(version="1h-v1", y_vars=Y_VARS, windows=WINDOWS),
    load_kwargs=dict(load_behaviordata=True, load_videodata=True),
    n_jobs=N_JOBS, recompute=args.recompute, verbose=JOBLIB_VERBOSE,
    extra_key_by_session={sid: _session_key(sid) for sid in included_ids}, flatten=False)
session_tables = [t for t in session_tables if t is not None and len(t)]
if not session_tables:
    raise SystemExit("No selected session had usable behavior + video data -- nothing to analyze.")
trials_all = pd.concat(session_tables, ignore_index=True)
del session_tables
protocols_present = [p for p in PROTOCOLS if (trials_all["protocol"] == p).any()]
print(f"{len(trials_all)} engaged trials across {trials_all['session_id'].nunique()} sessions "
      f"({'/'.join(protocols_present)})")


def unit_groups(df):
    if unit_col is None:
        return [("all", df)]
    return list(df.groupby(unit_col))


def _holm(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    order = np.argsort(p[ok])
    adj = np.maximum.accumulate((ok.sum() - np.arange(ok.sum())) * p[ok][order])
    res = np.empty(ok.sum())
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def _wilcoxon(v, alternative="two-sided"):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < MIN_UNITS or np.all(v == 0):
        return np.nan
    try:
        return float(sps.wilcoxon(v, alternative=alternative).pvalue)
    except ValueError:
        return np.nan


# --------------------------------------------------------------------- #
# 3. Stimulus strategy bins (shared, project rule 4): applied per protocol, undefined strategies skipped
# --------------------------------------------------------------------- #
stim_strats_of = {p: [] for p in PROTOCOLS}
for strat in STIM_STRATEGIES:
    trials_all[f"stim__{strat}"] = np.nan
    rule = STIM_BINNING_STRATEGIES[strat]
    for protocol in PROTOCOLS:
        m = (trials_all["protocol"] == protocol).to_numpy()
        if not m.any():
            continue
        sub = trials_all.loc[m]
        cur = bin_stimulus(sub["signal"], sub["stim_pct"], rule)
        if np.unique(cur[np.isfinite(cur)]).size < 2:
            continue   # undefined (no 1b fit) or a single bin for this protocol: not usable
        trials_all.loc[m, f"stim__{strat}"] = cur
        stim_strats_of[protocol].append(strat)
plot_strategy_of = {p: next((s for s in STRATEGY_PREFERENCE if s in stim_strats_of[p]), None) for p in PROTOCOLS}
for p in protocols_present:
    print(f"  {p}: stimulus strategies available {stim_strats_of[p]}; plotted strategy = {plot_strategy_of[p]}")
binning_used = pd.DataFrame([{"protocol": p, "stim_strategies": ", ".join(stim_strats_of[p]),
                              "plotted_strategy": plot_strategy_of[p],
                              "strategy_definitions": target_signature(stim_strats_of[p])}
                             for p in protocols_present])
binning_used.to_csv(table_dir / "binning_used.csv", index=False)


# --------------------------------------------------------------------- #
# 4. A1. Signal-strength quantile tuning curves (continuous view, every protocol)
# --------------------------------------------------------------------- #
def add_signal_quantile(df):
    d = df.dropna(subset=["signal"]).copy()
    d["signal_bin"] = d.groupby("session_id")["signal"].transform(
        lambda v: pd.qcut(v.rank(method="first"), N_QUANTILE_BINS, labels=False) + 1
        if len(v) >= 2 * N_QUANTILE_BINS else np.nan)
    return d.dropna(subset=["signal_bin"])


trials_q = add_signal_quantile(trials_all)


def trend_test(df, xcol, ycol):
    """Per-unit Spearman rho of ycol against xcol (an ordinal bin index). Descriptive -- no CV needed (see
    module docstring): a single correlation coefficient is not biased by in-sample fitting."""
    rows = []
    for unit, usub in unit_groups(df):
        d = usub[[xcol, ycol]].dropna()
        if len(d) < MIN_TRIALS_ANALYSIS or d[xcol].nunique() < 2 or d[ycol].std() == 0:
            continue
        rho, _ = sps.spearmanr(d[xcol], d[ycol])
        rows.append({"unit": unit, "animal_id": usub["animal_id"].iloc[0], "n_trials": len(d), "rho": rho})
    return pd.DataFrame(rows)


sq_curve_rows, sq_test_rows = [], []
for protocol in protocols_present:
    for wname in WINDOWS:
        for y in Y_VARS:
            ycol = f"{y}@{wname}"
            d = trials_q[(trials_q["protocol"] == protocol)].dropna(subset=[ycol])
            if d.empty:
                continue
            m, sem = unit_curve(d, "signal_bin", ycol, AGG)
            for b, v in m.items():
                sq_curve_rows.append({"protocol": protocol, "window": wname, "y": y, "signal_bin": int(b),
                                      "mean": v, "sem": sem.get(b, 0.0)})
            tt = trend_test(d, "signal_bin", ycol)
            v = tt["rho"].dropna()
            sq_test_rows.append({"protocol": protocol, "window": wname, "y": y, "n_units": len(v),
                                 "mean_rho": v.mean() if len(v) else np.nan, "p": _wilcoxon(v)})
sq_curves = pd.DataFrame(sq_curve_rows)
sq_curves.to_csv(table_dir / f"signal_quantile_tuning{AGG_SUFFIX}.csv", index=False)
sq_tests = pd.DataFrame(sq_test_rows)
if len(sq_tests):
    for (protocol, y), idx in sq_tests.groupby(["protocol", "y"]).groups.items():
        sq_tests.loc[idx, "p_holm"] = _holm(sq_tests.loc[idx, "p"].to_numpy())
sq_tests.to_csv(table_dir / f"signal_quantile_group_tests{AGG_SUFFIX}.csv", index=False)

fig, axes = plt.subplots(len(Y_VARS), len(WINDOWS), figsize=(4.2 * len(WINDOWS), 3.4 * len(Y_VARS)), squeeze=False)
for ri, y in enumerate(Y_VARS):
    for ci, wname in enumerate(WINDOWS):
        ax = axes[ri][ci]
        for protocol in protocols_present:
            d = sq_curves[(sq_curves["protocol"] == protocol) & (sq_curves["window"] == wname) & (sq_curves["y"] == y)]
            if d.empty:
                continue
            d = d.sort_values("signal_bin")
            ax.errorbar(d["signal_bin"], d["mean"], yerr=d["sem"] if AGG != "pooled" else None,
                        color=PROTOCOL_COLORS[protocol], marker="o", markersize=4, capsize=3, label=protocol)
            m = sq_tests[(sq_tests["protocol"] == protocol) & (sq_tests["window"] == wname) & (sq_tests["y"] == y)]
            if len(m) and np.isfinite(m["p_holm"].iloc[0]) and m["p_holm"].iloc[0] < ALPHA:
                ax.text(0.02, 0.95 - 0.08 * protocols_present.index(protocol), f"{protocol}*",
                        transform=ax.transAxes, color=PROTOCOL_COLORS[protocol], fontsize=8, va="top")
        if ri == 0:
            ax.set_title(wname, fontsize=9)
        if ci == 0:
            ax.set_ylabel(Y_LABELS.get(y, y), fontsize=9)
        if ri == len(Y_VARS) - 1:
            ax.set_xlabel("signal quantile (within session; 1 = weakest)", fontsize=8)
        ax.set_xticks(range(1, N_QUANTILE_BINS + 1))
axes[0][0].legend(fontsize=7, frameon=False)
fig.suptitle(f"Pupil area and speed vs signal strength (unit = {UNIT}; * = Holm-corrected Wilcoxon on "
             f"per-unit Spearman rho, p < {ALPHA})", y=1.0)
fig.tight_layout()
save_fig(fig, figdir, "1_pupil_speed_vs_signal_quantile")
plt.close(fig)


# --------------------------------------------------------------------- #
# 5. A2. Shared stimulus-strategy bins (one plotted strategy per protocol; all usable strategies in the CSV)
# --------------------------------------------------------------------- #
ss_rows, ss_test_rows = [], []
for protocol in protocols_present:
    for strat in stim_strats_of[protocol]:
        xcol = f"stim__{strat}"
        for wname in WINDOWS:
            for y in Y_VARS:
                ycol = f"{y}@{wname}"
                d = trials_all[(trials_all["protocol"] == protocol)].dropna(subset=[xcol, ycol])
                if d.empty:
                    continue
                m, sem = unit_curve(d, xcol, ycol, AGG)
                for b, v in m.items():
                    ss_rows.append({"protocol": protocol, "strategy": strat, "window": wname, "y": y,
                                    "bin": int(b), "mean": v, "sem": sem.get(b, 0.0)})
                tt = trend_test(d, xcol, ycol)
                v = tt["rho"].dropna()
                ss_test_rows.append({"protocol": protocol, "strategy": strat, "window": wname, "y": y,
                                     "n_units": len(v), "mean_rho": v.mean() if len(v) else np.nan, "p": _wilcoxon(v)})
ss_curves = pd.DataFrame(ss_rows)
ss_curves.to_csv(table_dir / f"stim_strategy_tuning{AGG_SUFFIX}.csv", index=False)
ss_tests = pd.DataFrame(ss_test_rows)
if len(ss_tests):
    for (protocol, strat, y), idx in ss_tests.groupby(["protocol", "strategy", "y"]).groups.items():
        ss_tests.loc[idx, "p_holm"] = _holm(ss_tests.loc[idx, "p"].to_numpy())
ss_tests.to_csv(table_dir / f"stim_strategy_group_tests{AGG_SUFFIX}.csv", index=False)

plottable = [p for p in protocols_present if plot_strategy_of[p]]
if plottable:
    fig, axes = plt.subplots(len(Y_VARS), len(WINDOWS), figsize=(4.2 * len(WINDOWS), 3.4 * len(Y_VARS)), squeeze=False)
    for ri, y in enumerate(Y_VARS):
        for ci, wname in enumerate(WINDOWS):
            ax = axes[ri][ci]
            width = 0.8 / max(len(plottable), 1)
            panel_names = []            # longest bin-name list among the protocols actually drawn in THIS panel
            for pi, protocol in enumerate(plottable):
                strat = plot_strategy_of[protocol]
                names = STIM_BINNING_STRATEGIES[strat].get("names") or []
                d = ss_curves[(ss_curves["protocol"] == protocol) & (ss_curves["strategy"] == strat) &
                              (ss_curves["window"] == wname) & (ss_curves["y"] == y)].sort_values("bin")
                if d.empty:
                    continue
                xpos = d["bin"].to_numpy() + (pi - (len(plottable) - 1) / 2) * width
                ax.bar(xpos, d["mean"], width=width * 0.9, color=PROTOCOL_COLORS[protocol], alpha=0.6,
                       edgecolor="black", linewidth=0.4,
                       yerr=d["sem"] if AGG != "pooled" else None, error_kw=dict(lw=0.8, capsize=2),
                       label=protocol if ri == 0 and ci == 0 else None)
                if len(names) > len(panel_names):
                    panel_names = names
            if panel_names:
                ax.set_xticks(range(len(panel_names)))
                ax.set_xticklabels(panel_names, fontsize=7)
            if ri == 0:
                ax.set_title(wname, fontsize=9)
            if ci == 0:
                ax.set_ylabel(Y_LABELS.get(y, y), fontsize=9)
            if ri == len(Y_VARS) - 1:
                ax.set_xlabel("stimulus-strategy bin (preferred strategy per protocol)", fontsize=8)
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Pupil area and speed vs the shared stimulus-strategy bins (unit = {UNIT}; preferred strategy "
                 f"per protocol: {plot_strategy_of})", y=1.0, fontsize=9)
    fig.tight_layout()
    save_fig(fig, figdir, "2_pupil_speed_vs_stim_strategy")
    plt.close(fig)


# --------------------------------------------------------------------- #
# 6. B. Correct vs incorrect: paired per-unit comparison, one line per unit (mirrors 1f's "two halves" check)
# --------------------------------------------------------------------- #
pc_rows = []
for protocol in protocols_present:
    for wname in WINDOWS:
        for y in Y_VARS:
            ycol = f"{y}@{wname}"
            for unit, usub in unit_groups(trials_all[trials_all["protocol"] == protocol]):
                d = usub[["correct", ycol]].dropna()
                dc, de = d[d["correct"] == 1][ycol], d[d["correct"] == 0][ycol]
                if len(dc) < MIN_TRIALS_PER_CLASS or len(de) < MIN_TRIALS_PER_CLASS:
                    continue
                pc_rows.append({"protocol": protocol, "window": wname, "y": y, "unit": unit,
                                "animal_id": usub["animal_id"].iloc[0], "n_correct": len(dc), "n_error": len(de),
                                "mean_correct": dc.mean(), "mean_error": de.mean(), "diff": dc.mean() - de.mean()})
pc = pd.DataFrame(pc_rows)
if AGG == "pooled" and len(pc):
    pc = pc.drop(columns=["unit", "animal_id"])
pc.to_csv(table_dir / f"performance_comparison_per_unit{AGG_SUFFIX}.csv", index=False)

pc_test_rows = []
if len(pc):
    for (protocol, wname, y), g in pc.groupby(["protocol", "window", "y"]):
        pc_test_rows.append({"protocol": protocol, "window": wname, "y": y, "n_units": len(g),
                             "mean_diff": g["diff"].mean(), "p": _wilcoxon(g["diff"])})
pc_tests = pd.DataFrame(pc_test_rows)
if len(pc_tests):
    for (protocol, y), idx in pc_tests.groupby(["protocol", "y"]).groups.items():
        pc_tests.loc[idx, "p_holm"] = _holm(pc_tests.loc[idx, "p"].to_numpy())
pc_tests.to_csv(table_dir / f"performance_comparison_group_tests{AGG_SUFFIX}.csv", index=False)

fig, axes = plt.subplots(len(Y_VARS), len(WINDOWS), figsize=(3.6 * len(WINDOWS), 3.4 * len(Y_VARS)), squeeze=False, sharey="row")
for ri, y in enumerate(Y_VARS):
    for ci, wname in enumerate(WINDOWS):
        ax = axes[ri][ci]
        for pi, protocol in enumerate(protocols_present):
            d = pc[(pc["protocol"] == protocol) & (pc["window"] == wname) & (pc["y"] == y)]
            if d.empty:
                continue
            xo = pi * 0.12
            for mc, me in zip(d["mean_correct"], d["mean_error"]):
                ax.plot([0 + xo, 1 + xo], [me, mc], color=PROTOCOL_COLORS[protocol], alpha=0.35, lw=0.8)
            ax.scatter(np.full(len(d), 0 + xo), d["mean_error"], s=10, color=PROTOCOL_COLORS[protocol])
            ax.scatter(np.full(len(d), 1 + xo), d["mean_correct"], s=10, color=PROTOCOL_COLORS[protocol])
            ax.plot([0 + xo, 1 + xo], [d["mean_error"].mean(), d["mean_correct"].mean()],
                    color=PROTOCOL_COLORS[protocol], marker="D", lw=2.2, markeredgecolor="black",
                    label=protocol if ri == 0 and ci == 0 else None)
            m = pc_tests[(pc_tests["protocol"] == protocol) & (pc_tests["window"] == wname) & (pc_tests["y"] == y)]
            if len(m) and np.isfinite(m["p_holm"].iloc[0]):
                ax.text(0.5 + xo, 0.97 - 0.09 * pi, f"{protocol} p={m['p_holm'].iloc[0]:.3f}", transform=ax.transAxes,
                        fontsize=6.5, color=PROTOCOL_COLORS[protocol], ha="center", va="top")
        if ri == 0:
            ax.set_title(wname, fontsize=9)
        if ci == 0:
            ax.set_ylabel(Y_LABELS.get(y, y), fontsize=9)
        ax.set_xticks([0.12, 1.12])
        ax.set_xticklabels(["error", "correct"], fontsize=8)
        ax.set_xlim(-0.4, 1.6)
axes[0][0].legend(fontsize=7, frameon=False, loc="lower left")
fig.suptitle(f"Pupil area and speed on correct vs error trials (unit = {UNIT}s; lines = {UNIT}s; "
             f"p = Holm-corrected paired Wilcoxon across windows)", y=1.0)
fig.tight_layout()
save_fig(fig, figdir, "3_pupil_speed_vs_performance")
plt.close(fig)


# --------------------------------------------------------------------- #
# 7. D. Interaction view: the signal-quantile tuning curve, split by correct / error
# --------------------------------------------------------------------- #
fig, axes = plt.subplots(len(Y_VARS), len(WINDOWS), figsize=(4.2 * len(WINDOWS), 3.4 * len(Y_VARS)), squeeze=False)
for ri, y in enumerate(Y_VARS):
    for ci, wname in enumerate(WINDOWS):
        ax = axes[ri][ci]
        ycol = f"{y}@{wname}"
        for protocol in protocols_present:
            for outcome, ls, lab in ((1, "-", "correct"), (0, "--", "error")):
                d = trials_q[(trials_q["protocol"] == protocol) & (trials_q["correct"] == outcome)].dropna(subset=[ycol])
                if d.empty:
                    continue
                m, sem = unit_curve(d, "signal_bin", ycol, AGG)
                if not len(m):
                    continue
                m = m.sort_index()
                ax.errorbar(m.index, m.values, yerr=sem.reindex(m.index).fillna(0.0) if AGG != "pooled" else None,
                            color=PROTOCOL_COLORS[protocol], linestyle=ls, marker="o", markersize=3, capsize=2,
                            label=f"{protocol} {lab}" if ri == 0 and ci == 0 else None)
        if ri == 0:
            ax.set_title(wname, fontsize=9)
        if ci == 0:
            ax.set_ylabel(Y_LABELS.get(y, y), fontsize=9)
        if ri == len(Y_VARS) - 1:
            ax.set_xlabel("signal quantile (within session; 1 = weakest)", fontsize=8)
        ax.set_xticks(range(1, N_QUANTILE_BINS + 1))
axes[0][0].legend(fontsize=6.5, frameon=False, ncol=2)
fig.suptitle(f"Signal-strength tuning, split by trial outcome (unit = {UNIT}; solid = correct, dashed = error)", y=1.0)
fig.tight_layout()
save_fig(fig, figdir, "4_pupil_speed_vs_signal_by_outcome")
plt.close(fig)

print(f"\nWrote tables to {table_dir}, figures to {figdir}")
