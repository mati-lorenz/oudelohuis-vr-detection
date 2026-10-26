# -*- coding: utf-8 -*-
"""
1d_performance_predictors
==========================
Relationships between the animal's trial-by-trial PERFORMANCE and behavioral variables -- both from the
CURRENT trial and from the PREVIOUS trial (classic "history effects": does the previous trial's state or
outcome predict this trial's accuracy?). Across all three protocols (DM, DP, DN).

Performance is CORRECTNESS (0/1), generalized to EVERY signal level, not just the 0%/100% catch trials:
    correct = (signal > 0 AND lickResponse == 1) OR (signal == 0 AND lickResponse == 0)
(this subsumes the catch-trial-only HIT/CR = 1, MISS/FA = 0 definition: those are the signal == 100 /
signal == 0 special cases of the same rule).

Predictors
----------
* Behavioral state (speed, pupil area, motion energy, lick rate) from the PRE-STIMULUS window only
  ([-30 cm, -10 cm] relative to stimStart; a window after stimulus onset would be circular, since the
  response that determines "correct" happens there), for the current and the previous trial.
* Previous trial's correctness and choice (win-stay / lose-shift).
* STIMULUS strength, current and previous trial, binned with the SHARED strategies of
  infotheory/stim_binning.py (STIM_STRATEGIES below; the bin INDEX is the predictor value).
  Strategies defined on the psychometric curve need a 1b fit (DN only); DM/DP get the strategies that do
  not (zero_vs_nonzero). Strategies undefined for a protocol are skipped for it.

Project-wide rules (infotheory/project_config.py) -- what applies to 1d
    * selection : --animals / --sessions restrict the sessions; a restricted run writes to
                  out/selection/<tag>/ and NEVER overwrites the full-dataset out/ that later steps read.
    * aggregate : --aggregate pooled|session|mouse = the unit of analysis for every plot AND for the MI /
                  linear / multivariate tables. pooled: one analysis per protocol, no error bars.
                  session: one analysis per session; bars = mean over sessions +- SEM, dots = sessions
                  (coloured by mouse). mouse: a mouse's sessions pooled first, mean +- SEM over mice.
    * memory    : each session is loaded, reduced to its small trial table, cached in store/ and freed;
                  raw behavior/video data of several sessions is never in RAM together
                  (--n-jobs = sessions in parallel).
    * binning   : the histogram MI uses the SHARED project_config.RESPONSE_BINS (quantile bins) and the
                  shared trial filters (MIN_TRIALS_PER_TARGET_BIN / MIN_TRIALS_ANALYSIS), and the stimulus
                  is binned with the shared strategies -- so the numbers here are comparable with 2c / 3a-3c
                  BEFORE bias correction. (The old private MI_BINS = 20 and 4 quantile stimulus bins are gone.)

Reads: 2_pipeline/1b_psychometric/out/psychometric_fits.csv (included sessions + per-session mu/sigma).
Loads: raw behaviordata/videodata, one session at a time.

Outputs
-------
2_pipeline/1d_performance_predictors/
    out/    performance_mi_vs_linear[__<aggregate>].csv              (protocol, [unit], predictor): r, r2,
                                                MI raw/corrected, MI implied by r2, linear sufficient?
            performance_multivariate_mi_vs_linear[__<aggregate>].csv (protocol, [unit]): all predictors jointly
            performance_history_kernel[__<aggregate>].csv            (protocol, lag, [unit]): accuracy delta
            performance_overview[__<aggregate>].csv                  accuracy per (protocol, [unit])
            performance_binning_used.csv                              the shared settings + stimulus
                                                strategies actually used, per protocol
            figures/*.png
    out/selection/<tag>/...  the same for a restricted run
    store/  trial_performance_<hash>.pkl      per-session trial table (reused unless --recompute)
(pooled tables keep their original names and columns; session / mouse tables carry `unit` and `animal_id`.)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from infotheory.pipeline import get_pipeline_paths
from infotheory.session import discover_sessions, PROTOCOLS
from infotheory.continuous import merge_behavior_video, restrict_to_engaged, remove_pupil_outliers, \
    compute_trial_position_window_means
from infotheory.info_theory import mutual_information_shuffle, mutual_information_shuffle_ksg, \
    compare_mi_to_linear, linear_fit, multivariate_linear_fit, quantile_bin_edges
from infotheory.plotting import set_style, save_fig, clear_figures, PROTOCOL_COLORS
from infotheory import project_config as cfg
from infotheory.scope import add_scope_args, scope_from_args, select_sessions, scoped_dirs, data_stamp, \
    unit_columns, unit_noun
from infotheory.session_jobs import run_sessions
from infotheory.unit_stats import unit_curve, stable_seed
from infotheory.unit_plots import unit_bars, hbar_units, mouse_palette, mouse_legend
from infotheory.stim_binning import STIM_BINNING_STRATEGIES, DEFAULT_STRATEGIES, psychometric_pct, \
    bin_stimulus, strategy_names, target_signature

BEHAVIOR_VARS = ["runspeed", "pupil_area", "motionenergy", "lick"]
RATE_VARIABLES = ["lick"]  # 0/1 event channel -- binned as counts/width (a rate), never averaged
VAR_LABELS = {"runspeed": "speed (cm/s)", "pupil_area": "pupil area (a.u.)",
              "motionenergy": "motion energy (a.u.)", "lick": "lick rate", "stim": "stimulus bin"}
PRESTIM_WINDOW = (-30.0, -10.0)   # cm relative to stimStart -- BEFORE the stimulus (see docstring)

# --- stimulus binning (shared strategies, infotheory/stim_binning.py; edit them THERE) ----------------
STIM_STRATEGIES = list(DEFAULT_STRATEGIES)   # which strategies are used as pairwise predictors
MULTIVARIATE_STIM_PREFERENCE = ["pct_0-20_20-80_80-100", "zero_vs_nonzero"]   # the one strategy that goes into the
                                             # multivariate model of a protocol: first one defined for it

# --- shared with the other scripts (project_config.py): change them THERE, not here ----------------------
MI_BINS = cfg.RESPONSE_BINS                  # histogram-MI bins per variable (quantile, equal occupancy)
MIN_TRIALS_PER_TARGET_BIN = cfg.MIN_TRIALS_PER_TARGET_BIN   # each outcome class (correct / error) needs this many
MIN_TRIALS_ANALYSIS = cfg.MIN_TRIALS_ANALYSIS               # trials needed to analyse a (protocol, unit, predictor)

# --- local to this script -------------------------------------------------------------------------------
N_QUANTILE_BINS = 5      # exploratory tuning curves: accuracy per quantile bin of a predictor
MAX_LAG_KERNEL = 5       # history kernel: how many trials back
KERNEL_MIN_TRIALS = 5    # per class, for one kernel point
N_SHUFFLES = 100
KSG_K = 5                # neighbors of the multivariate KSG estimator (section 5)
N_SHUFFLES_MULTIVARIATE = 20
LINEAR_SUFFICIENT_FRAC = 0.8
JOBLIB_VERBOSE = 5

parser = argparse.ArgumentParser()
add_scope_args(parser)
args = parser.parse_args()
scope = scope_from_args(args)
AGG = scope.aggregate
UNIT = unit_noun(AGG)
N_JOBS = scope.n_jobs
AGG_SUFFIX = "" if AGG == "pooled" else f"__{AGG}"   # pooled tables keep their original file names
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
# 1. Included sessions from 1b (intersected with the selection), plus each session's mu / sigma
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
print(f"Using {len(included_ids)} included sessions from 1b_psychometric "
      f"({included[included['session_id'].isin(included_ids)]['protocol'].value_counts().to_dict()})")
triple_of = {f"{a}_{d}": (p, a, d) for p, a, d in all_triples}
mouse_colors = mouse_palette(a for _, a, _ in all_triples)
mu_sigma_lookup = fits_df.set_index("session_id")[["mu", "sigma"]].to_dict("index")


# --------------------------------------------------------------------- #
# 2. Per-session trial tables (one session in RAM at a time) -- cached per session
# --------------------------------------------------------------------- #
def _session_key(sid):
    ms = mu_sigma_lookup.get(sid, {})
    return f"{data_stamp(*triple_of[sid])}|{ms.get('mu')}|{ms.get('sigma')}"


def process_session(ses, seed):
    """One DataFrame (one row per engaged trial: current + previous-trial features, outcome) or None.
    Stimulus strategies are applied later (section 3) so editing them does not invalidate the cache."""
    if ses.trialdata is None or ses.behaviordata is None:
        return None
    merged_full = merge_behavior_video(ses.behaviordata, ses.videodata)
    merged_full = remove_pupil_outliers(merged_full)
    if merged_full.empty:
        return None
    merged = restrict_to_engaged(merged_full, ses.trialdata)
    if merged.empty:
        return None

    available_vars = [v for v in BEHAVIOR_VARS if v in merged.columns]
    available_rate_vars = [v for v in RATE_VARIABLES if v in available_vars]
    engaged_trials = (ses.trialdata[ses.trialdata["engaged"] == 1]
                      if "engaged" in ses.trialdata.columns else ses.trialdata.copy())
    engaged_trials = engaged_trials.assign(
        correct=(((engaged_trials["signal"] > 0) & (engaged_trials["lickResponse"] == 1))
                 | ((engaged_trials["signal"] == 0) & (engaged_trials["lickResponse"] == 0))).astype(int))

    ms = mu_sigma_lookup.get(ses.session_id, {})
    engaged_trials["stim_pct"] = psychometric_pct(engaged_trials["signal"], ms.get("mu"), ms.get("sigma"))

    pre = compute_trial_position_window_means(engaged_trials, merged, columns=available_vars,
                                              s_start=PRESTIM_WINDOW[0], s_end=PRESTIM_WINDOW[1],
                                              rate_columns=available_rate_vars)
    pre = pre.rename(columns={v: f"cur_{v}" for v in available_vars})
    trial = engaged_trials[["trialNumber", "signal", "stim_pct", "lickResponse", "correct"]].merge(
        pre, on="trialNumber", how="left")
    trial = trial.sort_values("trialNumber").reset_index(drop=True)

    # "previous" = the previous ENGAGED trial that is analysed (a gap in trialNumber is not special-cased)
    for col in [f"cur_{v}" for v in available_vars] + ["signal", "stim_pct", "correct", "lickResponse"]:
        trial[f"prev_{col[4:]}" if col.startswith("cur_") else f"prev_{col}"] = trial[col].shift(1)
    for lag in range(1, MAX_LAG_KERNEL + 1):
        trial[f"correct_lag{lag}"] = trial["correct"].shift(lag)
    trial["session_id"] = ses.session_id
    trial["animal_id"] = ses.animal_id
    trial["protocol"] = ses.protocol
    return trial


session_tables = run_sessions(
    included_ids, process_session, list(PROTOCOLS), paths.store, "trial_performance",
    key_params=dict(version="1d-v2", variables=BEHAVIOR_VARS, rate_variables=RATE_VARIABLES,
                    window=PRESTIM_WINDOW, max_lag=MAX_LAG_KERNEL),
    load_kwargs=dict(load_behaviordata=True, load_videodata=True),
    n_jobs=N_JOBS, recompute=args.recompute, verbose=JOBLIB_VERBOSE,
    extra_key_by_session={sid: _session_key(sid) for sid in included_ids}, flatten=False)
session_tables = [t for t in session_tables if t is not None and len(t)]
if not session_tables:
    raise SystemExit("No selected session had usable behaviordata -- nothing to analyze.")
trial_performance = pd.concat(session_tables, ignore_index=True)   # small: one row per engaged trial
del session_tables
print(f"Trials with a defined outcome: {len(trial_performance)} across "
      f"{trial_performance['session_id'].nunique()} sessions")


# --------------------------------------------------------------------- #
# 3. Stimulus predictors from the SHARED strategies. Applied per protocol (curve-based strategies are
#    NaN without a 1b fit -> that strategy is simply not available for the protocol). The bin index is
#    the numeric predictor value.
# --------------------------------------------------------------------- #
stim_strats_of = {p: [] for p in PROTOCOLS}   # strategies usable per protocol
for strat in STIM_STRATEGIES:
    trial_performance[f"cur_stim__{strat}"] = np.nan
    trial_performance[f"prev_stim__{strat}"] = np.nan
    rule = STIM_BINNING_STRATEGIES[strat]
    for protocol in PROTOCOLS:
        m = (trial_performance["protocol"] == protocol).to_numpy()
        if not m.any():
            continue
        sub = trial_performance.loc[m]
        cur = bin_stimulus(sub["signal"], sub["stim_pct"], rule)
        prev = bin_stimulus(sub["prev_signal"], sub["prev_stim_pct"], rule)
        if np.unique(cur[np.isfinite(cur)]).size < 2:
            continue   # undefined (no fit) or a single bin for this protocol
        trial_performance.loc[m, f"cur_stim__{strat}"] = cur
        trial_performance.loc[m, f"prev_stim__{strat}"] = prev
        stim_strats_of[protocol].append(strat)

mv_strategy = {p: next((s for s in MULTIVARIATE_STIM_PREFERENCE if s in stim_strats_of[p]), None)
               for p in PROTOCOLS}
for p in PROTOCOLS:
    if (trial_performance["protocol"] == p).any():
        print(f"  {p}: stimulus strategies available {stim_strats_of[p]}; multivariate model uses {mv_strategy[p]}")

current_vars = [v for v in BEHAVIOR_VARS if f"cur_{v}" in trial_performance.columns]
previous_vars = [v for v in BEHAVIOR_VARS if f"prev_{v}" in trial_performance.columns]
protocols_present = [p for p in PROTOCOLS if (trial_performance["protocol"] == p).any()]


def predictors_for(protocol, sub, multivariate=False):
    """Predictor columns usable for one protocol (not all-NaN there)."""
    if multivariate:
        s = mv_strategy[protocol]
        stim = [f"cur_stim__{s}", f"prev_stim__{s}"] if s else []
    else:
        stim = [f"{w}_stim__{s}" for s in stim_strats_of[protocol] for w in ("cur", "prev")]
    cols = (stim + [f"cur_{v}" for v in current_vars] + [f"prev_{v}" for v in previous_vars]
            + ["prev_correct", "prev_lickResponse"])
    return [c for c in cols if c in sub.columns and sub[c].notna().any()]


binning_used = pd.DataFrame([{
    "protocol": p, "response_bins": MI_BINS, "response_bin_method": cfg.RESPONSE_BIN_METHOD,
    "min_trials_per_target_bin": MIN_TRIALS_PER_TARGET_BIN, "min_trials_analysis": MIN_TRIALS_ANALYSIS,
    "pairwise_stim_strategies": ", ".join(stim_strats_of[p]), "multivariate_stim_strategy": mv_strategy[p],
    "strategy_definitions": target_signature(stim_strats_of[p] + ([mv_strategy[p]] if mv_strategy[p] else []))
} for p in protocols_present])
binning_used.to_csv(table_dir / "performance_binning_used.csv", index=False)


# --------------------------------------------------------------------- #
# 4. Exploratory plots (at the chosen unit of analysis)
# --------------------------------------------------------------------- #
def tag_unit(records, unit, sub):
    """Add unit columns for the non-pooled tables (pooled keeps its original schema)."""
    if unit_col is None:
        return records
    for r in records:
        r["unit"] = unit
        r["animal_id"] = sub["animal_id"].iloc[0]
    return records


def unit_groups(df):
    """(unit label, sub-table) pairs for df at the chosen aggregate."""
    if unit_col is None:
        return [("all", df)]
    return list(df.groupby(unit_col))


def per_unit_mean(df, by, y):
    """One row per (by..., unit): mean of y. Pooled -> one row per `by` combination."""
    out = df.groupby(by + UNIT_KEYS, observed=True)[y].mean().reset_index()
    return out


# 4a. overview -----------------------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
acc = per_unit_mean(trial_performance, ["protocol"], "correct")
unit_bars(axes[0], acc, "protocol", "correct", protocols_present, AGG, colors=PROTOCOL_COLORS,
          mouse_colors=mouse_colors, ylabel="accuracy (fraction correct)")
axes[0].set_title("Accuracy by protocol")
axes[0].set_ylim(0, 1)
if AGG != "pooled":
    mouse_legend(axes[0], mouse_colors, animals=trial_performance["animal_id"].unique(), loc="lower left")

for ax, col, mapping, title in [
        (axes[1], "prev_correct", {1: "prev correct", 0: "prev error"}, "Post-error vs. post-correct"),
        (axes[2], "prev_lickResponse", {1: "prev licked", 0: "prev no lick"}, "Accuracy by previous choice")]:
    d = trial_performance.dropna(subset=[col]).assign(cond=lambda x: x[col].map(mapping))
    u = per_unit_mean(d, ["protocol", "cond"], "correct")
    u["label"] = u["protocol"] + "\n" + u["cond"]
    order = [f"{p}\n{c}" for p in protocols_present for c in mapping.values()]
    unit_bars(ax, u, "label", "correct", order, AGG, colors={f"{p}\n{c}": PROTOCOL_COLORS[p]
                                                              for p in protocols_present for c in mapping.values()},
              mouse_colors=mouse_colors, ylabel="accuracy")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", labelsize=7)
fig.suptitle(f"Performance overview (correctness at every signal level; unit = {UNIT})", y=1.03)
fig.tight_layout()
save_fig(fig, figdir, "1_performance_overview")

overview = per_unit_mean(trial_performance, ["protocol"], "correct").rename(columns={"correct": "accuracy"})
n_tr = trial_performance.groupby(["protocol"] + UNIT_KEYS)["correct"].size().rename("n_trials").reset_index()
overview = overview.merge(n_tr, on=["protocol"] + UNIT_KEYS)
for col, name in [("prev_correct", "prev_correct"), ("prev_lickResponse", "prev_lick")]:
    for val in (1, 0):
        sub = trial_performance[trial_performance[col] == val]
        m = sub.groupby(["protocol"] + UNIT_KEYS)["correct"].mean().rename(f"accuracy_{name}{val}").reset_index()
        overview = overview.merge(m, on=["protocol"] + UNIT_KEYS, how="left")
overview.to_csv(table_dir / f"performance_overview{AGG_SUFFIX}.csv", index=False)


# 4b/4c. tuning curves: accuracy vs. quantile-binned predictor (shared bin edges per protocol) ------------
def plot_tuning_grid(columns, title, filename):
    """columns: [(label, column, is_stim_bin)]. Per protocol the quantile edges come from that protocol's
    selected trials; mean +- SEM across units (pooled: no error); faint lines = single units."""
    if not columns:
        return
    fig, axes = plt.subplots(1, len(columns), figsize=(4.2 * len(columns), 3.6), squeeze=False)
    for ax, (label, col, is_stim) in zip(axes[0], columns):
        for protocol in protocols_present:
            sub = trial_performance[trial_performance["protocol"] == protocol].dropna(subset=[col, "correct"])
            if len(sub) < N_QUANTILE_BINS * 2:
                continue
            sub = sub.copy()
            if is_stim:
                sub["_x"] = sub[col]
            else:
                edges = quantile_bin_edges(sub[col].to_numpy(), N_QUANTILE_BINS)
                if len(edges) < 2:
                    continue
                cut = pd.cut(sub[col], edges, include_lowest=True)
                sub["_x"] = cut.apply(lambda iv: iv.mid if pd.notna(iv) else np.nan).astype(float)
                sub = sub.dropna(subset=["_x"])
            m, sem = unit_curve(sub, "_x", "correct", AGG)
            color = PROTOCOL_COLORS[protocol]
            if AGG != "pooled":
                for _, u in sub.groupby(unit_col):
                    um = u.groupby("_x")["correct"].mean()
                    ax.plot(um.index, um.values, color=color, alpha=0.18, linewidth=0.8)
            ax.errorbar(m.index, m.values, yerr=sem.values, color=color, marker="o", markersize=4,
                        capsize=3, label=protocol)
        ax.set_xlabel(label)
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1)
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"{title} (unit = {UNIT})", y=1.03)
    fig.tight_layout()
    save_fig(fig, figdir, filename)


stim_note = "; ".join(f"{p}: {mv_strategy[p]}" for p in protocols_present if mv_strategy[p])
for window, pre_vars in [("cur", current_vars), ("prev", previous_vars)]:
    cols = [(VAR_LABELS[v], f"{window}_{v}", False) for v in pre_vars]
    # one stimulus panel per protocol-specific strategy would be cluttered: show the multivariate strategy
    # of each protocol on a shared "bin index" axis (the strategy is named in the label)
    by_col = {}
    for p in protocols_present:
        if mv_strategy[p]:
            by_col.setdefault(f"{window}_stim__{mv_strategy[p]}", []).append(p)
    uniq = list(cols)
    for col, prots in by_col.items():
        strat = col.split("__", 1)[1]
        uniq.append((f"stimulus bin index [{'/'.join(prots)}: {strat}]", col, True))
    plot_tuning_grid(uniq, f"Accuracy vs. {'CURRENT' if window == 'cur' else 'PREVIOUS'}-trial pre-stimulus state "
                           f"& stimulus", "2_tuning_current_trial" if window == "cur" else "3_tuning_previous_trial")


# 4d. history kernel --------------------------------------------------------------------------------------
kernel_rows = []
for protocol in protocols_present:
    psub = trial_performance[trial_performance["protocol"] == protocol]
    for unit, usub in unit_groups(psub):
        for lag in range(1, MAX_LAG_KERNEL + 1):
            s = usub.dropna(subset=[f"correct_lag{lag}", "correct"])
            same = s[s[f"correct_lag{lag}"] == 1]["correct"]
            diff = s[s[f"correct_lag{lag}"] == 0]["correct"]
            if len(same) < KERNEL_MIN_TRIALS or len(diff) < KERNEL_MIN_TRIALS:
                continue
            row = {"protocol": protocol, "lag": lag, "delta_accuracy": same.mean() - diff.mean(),
                   "n_prev_correct": len(same), "n_prev_error": len(diff)}
            kernel_rows.append(tag_unit([row], unit, usub)[0])
kernel = pd.DataFrame(kernel_rows)
kernel.to_csv(table_dir / f"performance_history_kernel{AGG_SUFFIX}.csv", index=False)
if len(kernel):
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for protocol in protocols_present:
        k = kernel[kernel["protocol"] == protocol]
        if k.empty:
            continue
        color = PROTOCOL_COLORS[protocol]
        if AGG != "pooled":
            for _, u in k.groupby("unit"):
                ax.plot(u["lag"], u["delta_accuracy"], color=color, alpha=0.18, linewidth=0.8)
        g = k.groupby("lag")["delta_accuracy"]
        sem = g.sem().fillna(0.0) if AGG != "pooled" else g.mean() * 0.0
        ax.errorbar(g.mean().index, g.mean().values, yerr=sem.values, color=color, marker="o", markersize=4,
                    capsize=3, label=protocol)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("trials back")
    ax.set_ylabel("accuracy(prev correct) - accuracy(prev error)")
    ax.set_title(f"History kernel (unit = {UNIT})")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save_fig(fig, figdir, "4_history_kernel")


# --------------------------------------------------------------------- #
# 5. Mutual information vs. linear fit, per predictor, per protocol, per unit
#    Histogram MI with the SHARED MI_BINS (quantile bins; a discrete variable gets one bin per value).
# --------------------------------------------------------------------- #
def class_counts_ok(y):
    c = pd.Series(y).value_counts()
    return all(c.get(k, 0) >= MIN_TRIALS_PER_TARGET_BIN for k in (0, 1))


def pairwise_for_group(protocol, unit, sub, predictors, mi_bins, n_shuffles, seed):
    rng_local = np.random.default_rng(seed)
    records = []
    for predictor in predictors:
        pair_data = sub[[predictor, "correct"]].dropna()
        if len(pair_data) < MIN_TRIALS_ANALYSIS or not class_counts_ok(pair_data["correct"]) \
                or pair_data[predictor].nunique() < 2:
            continue
        mi = mutual_information_shuffle(pair_data[predictor], pair_data["correct"], bins=mi_bins,
                                        n_shuffles=n_shuffles, rng=rng_local)
        lin = linear_fit(pair_data[predictor], pair_data["correct"])
        cmp = compare_mi_to_linear(mi.bits_corrected, lin.r2)
        records.append({
            "protocol": protocol, "predictor": predictor, "n_trials": mi.n,
            "r": lin.r, "r2": lin.r2, "p_linear": lin.p_value,
            "mi_bits_raw": mi.bits_raw, "mi_bits_shuffle_mean": mi.bits_shuffle_mean,
            "mi_bits_corrected": mi.bits_corrected, "mi_p_value": mi.p_value,
            "mi_bits_from_r2": cmp.mi_from_r2_bits, "mi_reference_bits": cmp.mi_reference_bits,
            "frac_explained_linearly": cmp.frac_explained_linearly,
            "linear_sufficient": (cmp.frac_explained_linearly >= LINEAR_SUFFICIENT_FRAC)
            if not np.isnan(cmp.frac_explained_linearly) else None,
        })
    return tag_unit(records, unit, sub)


groups = [(protocol, unit, usub) for protocol in protocols_present
          for unit, usub in unit_groups(trial_performance[trial_performance["protocol"] == protocol])]
print(f"\nPairwise MI/linear fits: {len(groups)} (protocol, {UNIT}) groups, {MI_BINS} quantile bins "
      f"(n_jobs={N_JOBS}) ...")
nested = Parallel(n_jobs=N_JOBS, backend="loky", verbose=JOBLIB_VERBOSE)(
    delayed(pairwise_for_group)(protocol, unit, usub, predictors_for(protocol, usub), MI_BINS, N_SHUFFLES,
                                seed=stable_seed("1d-pairwise", protocol, unit))
    for protocol, unit, usub in groups)
mi_table = pd.DataFrame([r for g in nested for r in g])
mi_table.to_csv(table_dir / f"performance_mi_vs_linear{AGG_SUFFIX}.csv", index=False)

if len(mi_table):
    top = (mi_table.groupby(["protocol", "predictor"])[["n_trials", "r2", "mi_bits_corrected"]].mean()
           .sort_values("mi_bits_corrected", ascending=False).head(10))
    print(f"\nStrongest predictors of performance (mean over {UNIT}s, bias-corrected MI, top 10):")
    print(top.to_string())
    n_insuff = (mi_table["linear_sufficient"] == False).sum()  # noqa: E712 (None must not match)
    print(f"\n{n_insuff}/{len(mi_table)} rows show a linear fit missing >{100 * (1 - LINEAR_SUFFICIENT_FRAC):.0f}% "
          f"of the (bias-corrected) mutual information.")


# --------------------------------------------------------------------- #
# 5b. MULTIVARIATE: one model per (protocol, unit) with every predictor at once; joint MI (KSG)
#     for comparison. Predictors that are constant within the unit are dropped from that unit's model.
# --------------------------------------------------------------------- #
def multivariate_for_group(protocol, unit, sub, predictors, ksg_k, n_shuffles, seed):
    data = sub[predictors + ["correct"]].dropna()
    predictors = [p for p in predictors if data[p].nunique() > 1]
    if len(predictors) < 2 or len(data) < max(MIN_TRIALS_ANALYSIS, len(predictors) + 10) \
            or not class_counts_ok(data["correct"]):
        return []
    X, y = data[predictors].to_numpy(), data["correct"].to_numpy()
    lin = multivariate_linear_fit(X, y, predictor_names=predictors)
    mi = mutual_information_shuffle_ksg(X, y, k=ksg_k, n_shuffles=n_shuffles, rng=np.random.default_rng(seed))
    cmp = compare_mi_to_linear(mi.bits_corrected, lin.r2)
    return tag_unit([{
        "protocol": protocol, "predictors": ", ".join(predictors), "n_predictors": len(predictors),
        "n_trials": lin.n, "r2": lin.r2, "adj_r2": lin.adj_r2, "p_linear": lin.p_value,
        "mi_bits_raw": mi.bits_raw, "mi_bits_shuffle_mean": mi.bits_shuffle_mean,
        "mi_bits_corrected": mi.bits_corrected, "mi_p_value": mi.p_value,
        "mi_bits_from_r2": cmp.mi_from_r2_bits, "mi_reference_bits": cmp.mi_reference_bits,
        "frac_explained_linearly": cmp.frac_explained_linearly,
        "linear_sufficient": (cmp.frac_explained_linearly >= LINEAR_SUFFICIENT_FRAC)
        if not np.isnan(cmp.frac_explained_linearly) else None,
    }], unit, sub)


print(f"\nMultivariate (all predictors jointly): {len(groups)} groups (n_jobs={N_JOBS}) ...")
nested_mv = Parallel(n_jobs=N_JOBS, backend="loky", verbose=JOBLIB_VERBOSE)(
    delayed(multivariate_for_group)(protocol, unit, usub, predictors_for(protocol, usub, multivariate=True),
                                    KSG_K, N_SHUFFLES_MULTIVARIATE,
                                    seed=stable_seed("1d-multivariate", protocol, unit))
    for protocol, unit, usub in groups)
multivariate_table = pd.DataFrame([r for g in nested_mv for r in g])
multivariate_table.to_csv(table_dir / f"performance_multivariate_mi_vs_linear{AGG_SUFFIX}.csv", index=False)
if len(multivariate_table):
    print("\nMultivariate fit of performance from ALL predictors jointly (mean over units):")
    print(multivariate_table.groupby("protocol")[["n_trials", "r2", "adj_r2", "mi_bits_corrected"]].mean()
          .to_string())


# --------------------------------------------------------------------- #
# 6. Figures: does a linear model capture what predicts performance?
#    small translucent points = single units, large markers = mean over units (pooled: the pooled value)
# --------------------------------------------------------------------- #
all_predictors = list(dict.fromkeys(mi_table["predictor"])) if len(mi_table) else []
palette = list(plt.cm.tab20(np.linspace(0, 1, 20)))
predictor_color = {p: palette[i % 20] for i, p in enumerate(all_predictors)}


def scatter_units(ax, df, color, label=None):
    """Single units small, their mean large."""
    d = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["mi_reference_bits", "mi_bits_from_r2"])
    if d.empty:
        return None
    if AGG != "pooled":
        ax.scatter(d["mi_reference_bits"], d["mi_bits_from_r2"], color=color, s=12, alpha=0.35, linewidth=0)
    return ax.scatter(d["mi_reference_bits"].mean(), d["mi_bits_from_r2"].mean(), color=color, s=75,
                      edgecolor="black", linewidth=0.5, label=label, zorder=3)


if len(mi_table) and np.isfinite(mi_table["mi_reference_bits"]).any():
    finite = mi_table.replace([np.inf, -np.inf], np.nan).dropna(subset=["mi_reference_bits", "mi_bits_from_r2"])
    top_lim = max(finite["mi_reference_bits"].max(), finite["mi_bits_from_r2"].max()) * 1.1 + 1e-6 if len(finite) else 1
    prots = [p for p in protocols_present if p in finite["protocol"].unique()]
    fig, axes = plt.subplots(1, len(prots), figsize=(4.3 * len(prots), 4.3), squeeze=False)
    handles = {}
    for ax, protocol in zip(axes[0], prots):
        for pred, g in finite[finite["protocol"] == protocol].groupby("predictor"):
            h = scatter_units(ax, g, predictor_color[pred])
            if h is not None:
                handles.setdefault(pred, h)
        ax.plot([0, top_lim], [0, top_lim], color="grey", linestyle="--", linewidth=1)
        ax.set_xlim(0, top_lim)
        ax.set_ylim(0, top_lim)
        ax.set_title(protocol)
        ax.set_xlabel("MI, best available estimate (bits)")
    axes[0][0].set_ylabel("MI implied by linear r² (bits)")
    fig.suptitle(f"Does a linear model capture what predicts performance? (unit = {UNIT}; dashed = exact)", y=1.06)
    fig.legend(handles.values(), handles.keys(), loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=7,
               title="predictor", frameon=False)
    fig.tight_layout()
    save_fig(fig, figdir, "5_mi_vs_linear_summary")

    ff = mi_table.dropna(subset=["frac_explained_linearly"])
    prots = [p for p in protocols_present if p in ff["protocol"].unique()]
    if prots:
        n_lab = max(ff[ff["protocol"] == p]["predictor"].nunique() for p in prots)
        fig, axes = plt.subplots(1, len(prots), figsize=(5 * len(prots), max(3.5, 0.32 * n_lab + 1.8)), squeeze=False)
        for ax, protocol in zip(axes[0], prots):
            sub = ff[ff["protocol"] == protocol]
            hbar_units(ax, sub, "predictor", "frac_explained_linearly", AGG, colors=predictor_color,
                       mouse_colors=mouse_colors)
            ax.axvline(1.0, color="grey", linestyle="--", linewidth=1)
            ax.set_xlabel("MI(linear) / MI(full)")
            ax.set_title(protocol)
            ax.tick_params(axis="y", labelsize=7)
        fig.suptitle(f"Fraction of the mutual information a linear fit captures (unit = {UNIT})", y=1.02)
        fig.tight_layout()
        save_fig(fig, figdir, "5a_frac_explained_linearly")

if len(multivariate_table) and np.isfinite(multivariate_table["mi_reference_bits"]).any():
    fm = multivariate_table.replace([np.inf, -np.inf], np.nan).dropna(subset=["mi_reference_bits", "mi_bits_from_r2"])
    if len(fm):
        lim = max(fm["mi_reference_bits"].max(), fm["mi_bits_from_r2"].max()) * 1.1 + 1e-6
        fig, ax = plt.subplots(figsize=(4.8, 4.5))
        for protocol, g in fm.groupby("protocol"):
            scatter_units(ax, g, PROTOCOL_COLORS[protocol], label=protocol)
        ax.plot([0, lim], [0, lim], color="grey", linestyle="--", linewidth=1)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel("joint MI, best available estimate (bits)")
        ax.set_ylabel("MI implied by multivariate r² (bits)")
        ax.set_title(f"Multivariate: all predictors vs. performance (unit = {UNIT})")
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        save_fig(fig, figdir, "6_multivariate_mi_vs_linear_summary")

    fmf = multivariate_table.dropna(subset=["frac_explained_linearly"])
    if len(fmf):
        order = [p for p in protocols_present if p in fmf["protocol"].unique()]
        fig, ax = plt.subplots(figsize=(4.5, 4))
        unit_bars(ax, fmf, "protocol", "frac_explained_linearly", order, AGG, colors=PROTOCOL_COLORS,
                  mouse_colors=mouse_colors, ylabel="MI(linear) / MI(full)", hline=1.0)
        ax.set_title(f"Fraction of the joint MI a multivariate\nlinear fit captures (unit = {UNIT})")
        fig.tight_layout()
        save_fig(fig, figdir, "6a_multivariate_frac_explained_linearly")

print(f"\nWrote tables to {table_dir}, figures to {figdir}")
