# -*- coding: utf-8 -*-
"""
1c_behavior
===========
Behavioral variables analyzed in relation to themselves: running speed,
position, pupil size, and video motion energy. Uses the sessions that
passed 1b's inclusion criteria, engaged trials only (Rule #1: reads 1b's
out/, doesn't recompute its exclusions).

What this does
--------------
1. Loads behaviordata+videodata for included sessions only, merges them
   onto one timebase, restricts to engaged-trial windows, and strips
   pupil blink artifacts (|z|>5).
2. Exploratory plots: per-protocol distributions, position ("corridor")
   tuning of each variable, and a pairwise view of all four variables
   together with linear fits overlaid.
3. Per-trial features (mean speed/pupil/motion in a window around
   stimulus onset) as the unit for MI and regression -- raw ~50Hz
   samples are too autocorrelated to treat as independent draws, which
   would badly overstate significance.
4. For every pair of variables, per protocol: mutual information
   (histogram estimate, bias-corrected via shuffling) AND a linear fit
   (r^2). Compares the MI implied by r^2 under a linear/Gaussian
   assumption against the actual (corrected) MI -- a big gap means the
   relationship has a nonlinear component a linear model would miss
   (the same logic the project's a_docs describes for comparing
   2c_information against 2d_linear_encod/2e_nonlinear_encod later on,
   applied here at the behavioral level first).

Reads: 2_pipeline/1b_psychometric/out/psychometric_fits.csv for the
included-session list (Rule #1: read an earlier script's out/, don't
recompute its exclusions).
Loads: raw behaviordata/videodata for those sessions -- 1b never loaded
these (only trialdata), so there's no cache to reuse here.

Outputs
-------
2_pipeline/1c_behavior/
    out/    behavior_mi_vs_linear.csv   one row per (protocol, variable
                                         pair): r, r2, MI raw/corrected,
                                         MI implied by r2, whether linear
                                         looks sufficient
            figures/*.png
    store/  trial_features.pkl          per-trial mean speed/pupil/motion
            position_binned.pkl         corridor-tuning aggregates (whole-session, not trial-locked)
            pooled_downsampled.pkl      pooled ~2Hz samples for the pairwise exploratory plot
            psth_time.pkl               trial-aligned time-domain traces (onset = stimulus)
            psth_position.pkl           trial-aligned position-domain traces (onset = stimulus)
            (all reused unless --recompute is passed)

Schema note: `stimStart` (and rewardZoneStart/rewardZoneEnd) are
POSITIONS, on the same scale as `zpos` -- confirmed against the real
pipeline's tensor_utils.py/behavior_signals.py, which always compare
them directly (`zpos_F - z_T[k]`). Position-domain alignment
(`align_trials_position`) uses that subtraction directly; time-domain
alignment (`align_trials_time`, and the per-trial window features
below) needs an actual timestamp, so `psth.derive_onset_time` finds the
wall-clock time at which each trial's own position trace first reaches
`stimStart` and uses that as the anchor instead.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from infotheory.pipeline import get_pipeline_paths
from infotheory.session import load_sessions, PROTOCOLS
from infotheory.continuous import (
    merge_behavior_video, check_zpos_consistency, restrict_to_engaged, remove_pupil_outliers,
    compute_trial_window_means, bin_by_position,
)
from infotheory.psth import align_trials_time, align_trials_position, derive_onset_time
from infotheory.behavior import add_trial_outcome
from infotheory.info_theory import mutual_information_shuffle_ksg, compare_mi_to_linear, linear_fit
from infotheory.plotting import set_style, save_fig, clear_figures, PROTOCOL_COLORS, OUTCOME_COLORS

# All behavior variables considered "in relation to themselves". Checked
# for availability per-session (`if c in merged.columns`), so this list
# can be longer than what any given dataset actually has -- extend it
# freely (e.g. more videoPC_k's) without breaking sessions that lack them.
VARIABLES = ["zpos", "runspeed", "lick", "pupil_area", "pupil_xpos", "pupil_ypos",
             "motionenergy", "videoPC_0", "videoPC_1"]
VAR_LABELS = {"zpos": "position (cm)", "runspeed": "speed (cm/s)", "lick": "lick rate",
              "pupil_area": "pupil area (a.u.)", "pupil_xpos": "pupil x (a.u.)",
              "pupil_ypos": "pupil y (a.u.)", "motionenergy": "motion energy (a.u.)",
              "videoPC_0": "video PC 1", "videoPC_1": "video PC 2"}
RATE_VARIABLES = ["lick"]  # 0/1 event channels -- binned as counts/width (a rate), never averaged
PRE_WINDOW_S = 2.0    # per-trial feature window: [stimStart - PRE_WINDOW_S, stimStart + POST_WINDOW_S]
POST_WINDOW_S = 1.0
N_POSITION_BINS = 24
KSG_K = 5             # neighbors for the KSG mutual-information estimator (doesn't need a bin count,
                      # and doesn't share histogram MI's downward bias on strong relationships)
N_SHUFFLES = 20
DOWNSAMPLE_HZ = 2.0   # pooled continuous samples (pairwise plots only) are downsampled to this rate
                      # per session to limit autocorrelation; the MI/regression numbers themselves
                      # use per-trial means (see compute_trial_window_means), not these pooled samples
LINEAR_SUFFICIENT_FRAC = 0.8  # linear MI / corrected MI >= this -> "linear looks sufficient"
PSTH_VARS = ["runspeed", "lick", "pupil_area", "motionenergy"]
PSTH_T_PRE, PSTH_T_POST, PSTH_T_BINSIZE = -3.0, 3.0, 0.15      # trial-aligned time window/bin (s)
PSTH_S_PRE, PSTH_S_POST, PSTH_S_BINSIZE = -80.0, 60.0, 5.0     # trial-aligned position window/bin (cm)

parser = argparse.ArgumentParser()
parser.add_argument("--recompute", action="store_true",
                     help="Reload sessions from 0_data/ instead of reusing the cached store/ tables")
args = parser.parse_args()

paths = get_pipeline_paths(__file__)
figdir = paths.out / "figures"
clear_figures(figdir)
set_style()
rng = np.random.default_rng(0)


# --------------------------------------------------------------------- #
# 1. Included sessions from 1b
# --------------------------------------------------------------------- #
fits_path = paths.out_from("1b_psychometric") / "psychometric_fits.csv"
if not fits_path.exists():
    raise SystemExit(f"Run 1b_psychometric.py first -- expected {fits_path}")

fits_df = pd.read_csv(fits_path)
included = fits_df[fits_df["included"]]
included_ids = included["session_id"].tolist()
if not included_ids:
    raise SystemExit("No sessions passed 1b_psychometric's inclusion criteria -- nothing to analyze.")
print(f"Using {len(included_ids)} included sessions from 1b_psychometric "
      f"({included['protocol'].value_counts().to_dict()})")

# mu/sigma per session (only ever non-NaN for DN, since that's the only
# protocol 1b actually fits) -- used below to z-score intermediate signal
# levels into signal_psy, matching the old noise_to_psy convention.
mu_sigma_lookup = fits_df.set_index("session_id")[["mu", "sigma"]].to_dict("index")


# --------------------------------------------------------------------- #
# 2. Per-trial features + position-binned corridor stats -- or cache
# --------------------------------------------------------------------- #
trial_features_cache = paths.store / "trial_features.pkl"
position_binned_cache = paths.store / "position_binned.pkl"
time_course_cache = paths.store / "time_course.pkl"
pooled_cache = paths.store / "pooled_downsampled.pkl"
psth_time_cache = paths.store / "psth_time.pkl"
psth_position_cache = paths.store / "psth_position.pkl"

if (not args.recompute and trial_features_cache.exists() and position_binned_cache.exists()
        and time_course_cache.exists() and pooled_cache.exists()
        and psth_time_cache.exists() and psth_position_cache.exists()):
    print(f"Reusing cached behavior tables from {paths.store} (pass --recompute to reload 0_data/)")
    trial_features = pd.read_pickle(trial_features_cache)
    position_binned = pd.read_pickle(position_binned_cache)
    time_course = pd.read_pickle(time_course_cache)
    pooled = pd.read_pickle(pooled_cache)
    psth_time = pd.read_pickle(psth_time_cache)
    psth_position = pd.read_pickle(psth_position_cache)
else:
    sessions = load_sessions(protocols=PROTOCOLS, load_behaviordata=True, load_videodata=True,
                              only_session_ids=included_ids)

    trial_rows, position_rows, pooled_rows = [], [], []
    time_course_rows = []
    psth_time_rows, psth_position_rows = [], []
    global_bins = None
    zpos_diag_printed = False

    for ses in sessions:
        if ses.trialdata is None or ses.behaviordata is None:
            continue

        if not zpos_diag_printed:
            diag = check_zpos_consistency(ses.behaviordata, ses.videodata)
            if diag is not None:
                print(f"\nzpos QC ({ses.session_id}): behaviordata vs videodata position after "
                      f"nearest-timestamp match -- median |diff|={diag['median_abs_diff']:.2f}, "
                      f"95th pct={diag['p95_abs_diff']:.2f}, max={diag['max_abs_diff']:.2f} "
                      f"(n={diag['n_compared']}). merge_behavior_video keeps behaviordata's copy; "
                      "large numbers here would mean that choice matters.")
            zpos_diag_printed = True

        merged_full = merge_behavior_video(ses.behaviordata, ses.videodata)
        merged_full = remove_pupil_outliers(merged_full)
        if merged_full.empty:
            continue
        # Engaged-only view: for whole-session/pooled stats and the
        # per-trial-mean features, which SHOULD exclude disengaged
        # periods. NOT used for the PSTH alignment below -- that needs
        # the full trace so a window around trial k's onset can still
        # reach into the tail of trial k-1 / the ITI (see psth.py's
        # module docstring for why restricting first is wrong here).
        merged = restrict_to_engaged(merged_full, ses.trialdata)
        if merged.empty:
            continue

        available_vars = [c for c in VARIABLES if c in merged.columns]
        available_rate_vars = [c for c in RATE_VARIABLES if c in available_vars]

        engaged_trials = (ses.trialdata[ses.trialdata["engaged"] == 1]
                           if "engaged" in ses.trialdata.columns else ses.trialdata)
        # stimStart is a POSITION (see psth.py's module docstring), so
        # anything that needs an actual TIME to anchor a window on
        # (per-trial mean features here, time-domain PSTH below) uses a
        # derived onset time instead of stimStart directly. Derived once
        # per session from the FULL trace (merged_full), since deriving
        # it from the engaged-only `merged` could itself run into the
        # same truncation issue align_trials_time/_position guard against.
        engaged_trials = derive_onset_time(engaged_trials, merged_full)

        tf = compute_trial_window_means(engaged_trials, merged, columns=available_vars,
                                         pre_s=PRE_WINDOW_S, post_s=POST_WINDOW_S,
                                         rate_columns=available_rate_vars)
        tf["session_id"] = ses.session_id
        tf["protocol"] = ses.protocol
        trial_rows.append(tf)

        if "zpos" in merged.columns:
            if global_bins is None:
                global_bins = np.linspace(merged["zpos"].min(), merged["zpos"].max(), N_POSITION_BINS + 1)
            corridor_vars = [c for c in available_vars if c != "zpos"]
            pb = bin_by_position(merged, columns=corridor_vars, bins=global_bins,
                                  rate_columns=available_rate_vars)
            pb["session_id"] = ses.session_id
            pb["protocol"] = ses.protocol
            position_rows.append(pb)

        # Elapsed-session-time binning, as a whole-session view that
        # stays meaningful regardless of whether `zpos` resets every lap
        # or (as turned out to be the case here) accumulates across the
        # whole session -- see the print diagnostic after this loop.
        elapsed = merged.assign(elapsed_s=merged["ts"] - merged["ts"].iloc[0])
        time_bins = np.linspace(0, elapsed["elapsed_s"].max(), N_POSITION_BINS + 1)
        tc = bin_by_position(elapsed, columns=[c for c in available_vars if c != "zpos"],
                              bins=time_bins, position_col="elapsed_s", rate_columns=available_rate_vars)
        tc["session_id"] = ses.session_id
        tc["protocol"] = ses.protocol
        time_course_rows.append(tc)

        median_dt = np.median(np.diff(merged["ts"].to_numpy()))
        step = 1 if not np.isfinite(median_dt) or median_dt <= 0 else max(1, int(round(1 / (DOWNSAMPLE_HZ * median_dt))))
        down = merged.iloc[::step][available_vars].copy()
        down["session_id"] = ses.session_id
        down["protocol"] = ses.protocol
        pooled_rows.append(down)

        # Per-trial metadata to merge onto the PSTH tables below: outcome
        # (HIT/MISS/FA/CR) and, where a fit exists (DN only), the
        # z-scored intermediate-stimulus regressor signal_psy = (signal - mu) / sigma.
        trial_meta = add_trial_outcome(engaged_trials)
        mu_sigma = mu_sigma_lookup.get(ses.session_id, {})
        if mu_sigma and np.isfinite(mu_sigma.get("mu", np.nan)) and np.isfinite(mu_sigma.get("sigma", np.nan)):
            trial_meta = trial_meta.assign(
                signal_psy=(trial_meta["signal"] - mu_sigma["mu"]) / mu_sigma["sigma"])
        else:
            trial_meta = trial_meta.assign(signal_psy=np.nan)
        trial_meta = trial_meta[["trialNumber", "trialOutcome", "signal", "signal_psy"]]

        psth_vars_avail = [c for c in PSTH_VARS if c in merged_full.columns]
        psth_rate_vars = [c for c in RATE_VARIABLES if c in psth_vars_avail]
        if psth_vars_avail:
            pt = align_trials_time(engaged_trials, merged_full, columns=psth_vars_avail,
                                    t_pre=PSTH_T_PRE, t_post=PSTH_T_POST, binsize=PSTH_T_BINSIZE,
                                    rate_columns=psth_rate_vars)
            if len(pt):
                pt = pt.merge(trial_meta, on="trialNumber", how="left")
                pt["session_id"], pt["protocol"] = ses.session_id, ses.protocol
                psth_time_rows.append(pt)

            pp = align_trials_position(engaged_trials, merged_full, columns=psth_vars_avail,
                                        s_pre=PSTH_S_PRE, s_post=PSTH_S_POST, binsize=PSTH_S_BINSIZE,
                                        rate_columns=psth_rate_vars)
            if len(pp):
                pp = pp.merge(trial_meta, on="trialNumber", how="left")
                pp["session_id"], pp["protocol"] = ses.session_id, ses.protocol
                psth_position_rows.append(pp)

    trial_features = pd.concat(trial_rows, ignore_index=True) if trial_rows else pd.DataFrame()
    position_binned = pd.concat(position_rows, ignore_index=True) if position_rows else pd.DataFrame()
    time_course = pd.concat(time_course_rows, ignore_index=True) if time_course_rows else pd.DataFrame()
    pooled = pd.concat(pooled_rows, ignore_index=True) if pooled_rows else pd.DataFrame()
    psth_time = pd.concat(psth_time_rows, ignore_index=True) if psth_time_rows else pd.DataFrame()
    psth_position = pd.concat(psth_position_rows, ignore_index=True) if psth_position_rows else pd.DataFrame()

    trial_features.to_pickle(trial_features_cache)
    position_binned.to_pickle(position_binned_cache)
    time_course.to_pickle(time_course_cache)
    pooled.to_pickle(pooled_cache)
    psth_time.to_pickle(psth_time_cache)
    psth_position.to_pickle(psth_position_cache)

if not position_binned.empty:
    zpos_range = position_binned["bin_center"].max() - position_binned["bin_center"].min()
    if zpos_range > 300:
        print(f"\nNote: observed zpos range is {zpos_range:,.0f} cm -- this looks like a cumulative "
              "odometer (total distance run), not a coordinate that resets every lap/trial. The "
              "whole-session '2_corridor_position_tuning' plot below is really just the session's time "
              "course reparametrized by distance in that case, not a repeating-corridor tuning curve; "
              "see '2b_session_time_course' for the same whole-session view in elapsed time instead. "
              "The trial-aligned PSTH plots (5_/6_) are unaffected -- they use each trial's own "
              "position-at-onset as a reference point regardless of whether zpos resets or accumulates.")

n_sessions_used = trial_features["session_id"].nunique() if len(trial_features) else 0
print(f"Trial-level features: {len(trial_features)} engaged trials across {n_sessions_used} sessions")
print(f"Pooled downsampled samples: {len(pooled)} (~{DOWNSAMPLE_HZ} Hz/session)")


# --------------------------------------------------------------------- #
# 3. Exploratory plots
# --------------------------------------------------------------------- #
# 3a. Distributions per protocol
plot_vars = [v for v in VARIABLES if v not in ("zpos", *RATE_VARIABLES) and v in pooled.columns]
if plot_vars:
    ncols = min(4, len(plot_vars))
    nrows = int(np.ceil(len(plot_vars) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), squeeze=False)
    flat_axes = axes.flatten()
    for ax, var in zip(flat_axes, plot_vars):
        sns.violinplot(data=pooled, x="protocol", y=var, order=PROTOCOLS, ax=ax,
                        hue="protocol", palette=PROTOCOL_COLORS, legend=False)
        ax.set_title(VAR_LABELS.get(var, var))
        ax.set_ylabel("")
    for ax in flat_axes[len(plot_vars):]:
        ax.set_visible(False)
    fig.suptitle("Behavior variable distributions (engaged trials)", y=1.01)
    fig.tight_layout()
    save_fig(fig, figdir, "1_variable_distributions")

# 3b. Corridor (position) tuning -- whole session, not trial-locked.
# Caveat: only a meaningful "spatial tuning" curve if zpos resets each
# lap/trial. If it's a cumulative odometer instead (see the diagnostic
# printed above), this is really the session's time course reparametrized
# by distance -- '2b_session_time_course' below is the distance-agnostic
# version of the same whole-session view.
corridor_vars = [v for v in VARIABLES if v != "zpos"
                  and not position_binned.empty and f"{v}_mean" in position_binned.columns]
if corridor_vars:
    ncols = min(4, len(corridor_vars))
    nrows = int(np.ceil(len(corridor_vars) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)
    flat_axes = axes.flatten()
    for ax, var in zip(flat_axes, corridor_vars):
        for protocol in PROTOCOLS:
            sub = position_binned[position_binned["protocol"] == protocol]
            if sub.empty:
                continue
            agg = sub.groupby("bin_center")[f"{var}_mean"].mean()
            sem = sub.groupby("bin_center")[f"{var}_mean"].sem().fillna(0)
            ax.plot(agg.index, agg.to_numpy(), color=PROTOCOL_COLORS[protocol], label=protocol)
            ax.fill_between(agg.index, (agg - sem).to_numpy(), (agg + sem).to_numpy(),
                             color=PROTOCOL_COLORS[protocol], alpha=0.2)
        ax.set_xlabel("position (cm)")
        ax.set_ylabel(VAR_LABELS.get(var, var))
    for ax in flat_axes[len(corridor_vars):]:
        ax.set_visible(False)
    flat_axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Behavior variables vs. absolute position, whole session (engaged trials)", y=1.01)
    fig.tight_layout()
    save_fig(fig, figdir, "2_corridor_position_tuning")

# 2b. Same whole-session view, but by elapsed session time instead of
# distance -- meaningful regardless of whether zpos resets or accumulates.
time_vars = [v for v in VARIABLES if v != "zpos"
             and not time_course.empty and f"{v}_mean" in time_course.columns]
if time_vars:
    ncols = min(4, len(time_vars))
    nrows = int(np.ceil(len(time_vars) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)
    flat_axes = axes.flatten()
    for ax, var in zip(flat_axes, time_vars):
        for protocol in PROTOCOLS:
            sub = time_course[time_course["protocol"] == protocol]
            if sub.empty:
                continue
            agg = sub.groupby("bin_center")[f"{var}_mean"].mean()
            sem = sub.groupby("bin_center")[f"{var}_mean"].sem().fillna(0)
            ax.plot(agg.index / 60, agg.to_numpy(), color=PROTOCOL_COLORS[protocol], label=protocol)
            ax.fill_between(agg.index / 60, (agg - sem).to_numpy(), (agg + sem).to_numpy(),
                             color=PROTOCOL_COLORS[protocol], alpha=0.2)
        ax.set_xlabel("elapsed session time (min)")
        ax.set_ylabel(VAR_LABELS.get(var, var))
    for ax in flat_axes[len(time_vars):]:
        ax.set_visible(False)
    flat_axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Behavior variables vs. elapsed session time (engaged trials)", y=1.01)
    fig.tight_layout()
    save_fig(fig, figdir, "2b_session_time_course")

# 3c. Pairwise relationships (linear fit overlaid, by protocol)
pairplot_vars = [v for v in VARIABLES if v not in RATE_VARIABLES and v in pooled.columns]
if len(pairplot_vars) >= 2 and not pooled.empty:
    plot_df = pooled[pairplot_vars + ["protocol"]].dropna()
    if len(plot_df) > 6000:
        plot_df = pd.concat([
            d.sample(min(len(d), 2000), random_state=0) for _, d in plot_df.groupby("protocol")
        ], ignore_index=True)
    g = sns.pairplot(plot_df, hue="protocol", palette=PROTOCOL_COLORS, corner=True, kind="reg",
                      plot_kws=dict(scatter_kws=dict(alpha=0.25, s=8), line_kws=dict(linewidth=1.5)),
                      diag_kind="hist")
    g.figure.suptitle("Pairwise behavior variables (engaged trials, downsampled)", y=1.02)
    save_fig(g.figure, figdir, "3_pairwise_relationships")


# --------------------------------------------------------------------- #
# 3d. Trial-aligned ("PSTH"-style) traces, onset = stimulus presentation
#     (t=0 / relative position=0). These are the main plots of interest:
#     one row per variable, one column per grouping (trial outcome
#     always; the z-scored intermediate-stimulus level `signal_psy` too,
#     for DN where a psychometric fit exists -- DP falls back to raw
#     `signal` since it has intermediate levels but no fit; DM has
#     neither, so outcome is its only grouping). The whole-session,
#     non-trial-locked view is the "2_corridor_position_tuning" plot
#     above -- these are the trial-locked complement.
# --------------------------------------------------------------------- #
SIGNAL_GROUP_BINS = 10  # continuous groupings (signal / signal_psy) get binned into this many levels
                         # before averaging -- pooling sessions by *exact* z-scored value (each session
                         # has its own fitted mu/sigma) would otherwise produce dozens of near-unique
                         # levels instead of a handful of representative ones


def _group_order(values, group_col):
    if group_col == "trialOutcome":
        order = ["HIT", "MISS", "FA", "CR"]
        return [v for v in order if v in values]
    return sorted(values)


def plot_trial_aligned(psth_long: pd.DataFrame, x_col: str, x_label: str, prefix: str):
    if psth_long.empty:
        return
    cmap = plt.get_cmap("plasma")

    for protocol in PROTOCOLS:
        sub_protocol = psth_long[psth_long["protocol"] == protocol]
        if sub_protocol.empty:
            continue

        groupings = [("trialOutcome", "by outcome", OUTCOME_COLORS)]
        signal_col = "signal_psy" if protocol == "DN" else ("signal" if protocol == "DP" else None)
        if signal_col is not None and sub_protocol[signal_col].notna().any():
            label = "by z-scored signal (signal_psy)" if signal_col == "signal_psy" else "by signal (%)"
            groupings.append((signal_col, label, None))

        variables = [v for v in PSTH_VARS if v in sub_protocol["variable"].unique()]
        if not variables:
            continue

        fig, axes = plt.subplots(len(variables), len(groupings),
                                  figsize=(5.5 * len(groupings), 3 * len(variables)), squeeze=False)
        for i, var in enumerate(variables):
            var_data = sub_protocol[sub_protocol["variable"] == var]
            for j, (group_col, group_label, palette) in enumerate(groupings):
                ax = axes[i][j]
                gdata_all = var_data.dropna(subset=[group_col]).copy()

                if palette is None:
                    # Continuous grouping (signal / signal_psy): bin into a
                    # handful of representative levels rather than pooling
                    # by exact value across sessions.
                    edges = np.unique(np.quantile(gdata_all[group_col], np.linspace(0, 1, SIGNAL_GROUP_BINS + 1)))
                    if len(edges) < 2:
                        continue
                    gdata_all["_level"] = pd.cut(gdata_all[group_col], bins=edges, include_lowest=True)
                    level_values = {b: b.mid for b in gdata_all["_level"].cat.categories}
                    levels = sorted(level_values, key=lambda b: b.mid)
                    norm = plt.Normalize(min(level_values.values()), max(level_values.values()))
                    group_key_col = "_level"
                else:
                    levels = _group_order(gdata_all[group_col].unique(), group_col)
                    group_key_col = group_col
                    norm = None

                for level in levels:
                    gdata = gdata_all[gdata_all[group_key_col] == level]
                    if gdata.empty:
                        continue
                    agg = gdata.groupby(x_col)["value"].mean()
                    sem = gdata.groupby(x_col)["value"].sem().fillna(0)
                    if palette is not None:
                        color, label_str = palette[level], str(level)
                    else:
                        color, label_str = cmap(norm(level_values[level])), f"{level_values[level]:.2g}"
                    ax.plot(agg.index, agg.to_numpy(), color=color, label=label_str, linewidth=1.3)
                    ax.fill_between(agg.index, (agg - sem).to_numpy(), (agg + sem).to_numpy(),
                                     color=color, alpha=0.15)

                ax.axvline(0, color="black", linestyle=":", linewidth=1)
                if i == 0:
                    ax.set_title(group_label)
                if j == 0:
                    ax.set_ylabel(VAR_LABELS.get(var, var))
                ax.set_xlabel(x_label)
                ax.legend(fontsize=6, frameon=False, ncol=2 if len(levels) > 6 else 1)

        fig.suptitle(f"Trial-aligned {prefix.split('_')[-1]} -- {protocol} (0 = stimulus onset)", y=1.02)
        fig.tight_layout()
        save_fig(fig, figdir, f"{prefix}_{protocol}")


plot_trial_aligned(psth_time, "t_rel", "time rel. to stimulus onset (s)", "5_psth_time")
plot_trial_aligned(psth_position, "s_rel", "position rel. to stimulus onset (cm)", "6_psth_position")


# --------------------------------------------------------------------- #
# 4. Mutual information vs. linear regression, per pair, per protocol.
#    "All interesting combinations" = every pair among the variables
#    that actually have per-trial features (VARIABLES intersected with
#    trial_features' columns) -- zpos and lick (as a rate) included.
# --------------------------------------------------------------------- #
records = []
mi_vars = [v for v in VARIABLES if v in trial_features.columns]
pairs = list(itertools.combinations(mi_vars, 2))

for protocol in PROTOCOLS:
    sub = trial_features[trial_features["protocol"] == protocol] if len(trial_features) else pd.DataFrame()
    if sub.empty:
        continue
    for var_x, var_y in pairs:
        if var_x not in sub.columns or var_y not in sub.columns:
            continue
        pair_data = sub[[var_x, var_y]].dropna()
        if len(pair_data) < 10:
            continue

        mi = mutual_information_shuffle_ksg(pair_data[var_x], pair_data[var_y], k=KSG_K,
                                             n_shuffles=N_SHUFFLES, rng=rng)
        lin = linear_fit(pair_data[var_x], pair_data[var_y])
        cmp = compare_mi_to_linear(mi.bits_corrected, lin.r2)

        records.append({
            "protocol": protocol, "var_x": var_x, "var_y": var_y, "n_trials": mi.n,
            "r": lin.r, "r2": lin.r2, "p_linear": lin.p_value,
            "mi_bits_raw": mi.bits_raw, "mi_bits_shuffle_mean": mi.bits_shuffle_mean,
            "mi_bits_corrected": mi.bits_corrected, "mi_p_value": mi.p_value,
            "mi_bits_from_r2": cmp.mi_from_r2_bits, "mi_reference_bits": cmp.mi_reference_bits,
            "frac_explained_linearly": cmp.frac_explained_linearly,
            "linear_sufficient": (cmp.frac_explained_linearly >= LINEAR_SUFFICIENT_FRAC)
            if not np.isnan(cmp.frac_explained_linearly) else None,
        })

mi_table = pd.DataFrame(records)
mi_table.to_csv(paths.out / "behavior_mi_vs_linear.csv", index=False)

if len(mi_table):
    n_insufficient = (mi_table["linear_sufficient"] == False).sum()  # noqa: E712 (None must not match)
    print(f"\n{n_insufficient}/{len(mi_table)} (protocol, pair) combinations show a linear fit "
          f"missing >{100 * (1 - LINEAR_SUFFICIENT_FRAC):.0f}% of the (bias-corrected) mutual information:")
    if n_insufficient:
        print(mi_table.loc[mi_table["linear_sufficient"] == False,  # noqa: E712
                            ["protocol", "var_x", "var_y", "r2", "mi_bits_corrected", "mi_bits_from_r2"]]
              .to_string(index=False))


# --------------------------------------------------------------------- #
# 5. Figure: MI (best estimate) vs. MI-implied-by-r2, one point per
#    (protocol, pair). Marker SHAPE encodes var_x (the first variable of
#    the pair), color encodes var_y (the second) -- two small legends
#    instead of one legend entry per pair, so it stays readable even
#    with dozens of pairs (a pair's identity is "read off" as
#    shape+color rather than needing to scan a long combined list).
# --------------------------------------------------------------------- #
if len(mi_table) and np.isfinite(mi_table["mi_reference_bits"]).any():
    finite = mi_table.replace([np.inf, -np.inf], np.nan).dropna(subset=["mi_reference_bits", "mi_bits_from_r2"])
    lims = [0, finite["mi_reference_bits"].max() * 1.1 + 1e-6] if len(finite) else [0, 1]

    all_vars_in_pairs = sorted(set(v for pair in pairs for v in pair))
    marker_shapes = ["o", "s", "^", "v", "D", "P", "X", "*", "h", "p", "<", ">"]
    var_marker = {v: marker_shapes[i % len(marker_shapes)] for i, v in enumerate(all_vars_in_pairs)}
    color_palette = list(plt.cm.tab10(np.linspace(0, 1, 10))) + list(plt.cm.Set3(np.linspace(0, 1, 12)))
    var_color = {v: color_palette[i % len(color_palette)] for i, v in enumerate(all_vars_in_pairs)}

    protocols_present = [p for p in PROTOCOLS if p in mi_table["protocol"].unique()]
    fig, axes = plt.subplots(1, len(protocols_present), figsize=(4.3 * len(protocols_present), 4.3), squeeze=False)
    axes = axes[0]
    for ax, protocol in zip(axes, protocols_present):
        sub = finite[finite["protocol"] == protocol]
        for _, row in sub.iterrows():
            ax.scatter(row["mi_reference_bits"], row["mi_bits_from_r2"],
                       marker=var_marker[row["var_x"]], color=var_color[row["var_y"]],
                       s=70, edgecolor="black", linewidth=0.4)
        ax.plot(lims, lims, color="grey", linestyle="--", linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(protocol)
        ax.set_xlabel("MI, best available estimate (bits)")
    axes[0].set_ylabel("MI implied by linear r\u00b2 (bits)")
    fig.suptitle("Is the linear fit capturing the full relationship?\n"
                 "(shape = 1st variable, color = 2nd variable; dashed = linear matches MI exactly)", y=1.08)

    shape_handles = [plt.Line2D([0], [0], marker=var_marker[v], linestyle="none", markerfacecolor="white",
                                 markeredgecolor="black", markersize=8, label=v) for v in all_vars_in_pairs]
    color_handles = [plt.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=var_color[v],
                                 markeredgecolor="black", markersize=8, label=v) for v in all_vars_in_pairs]
    leg1 = fig.legend(handles=shape_handles, loc="center left", bbox_to_anchor=(1.0, 0.7),
                       fontsize=7, title="shape = 1st var", frameon=False)
    fig.add_artist(leg1)
    fig.legend(handles=color_handles, loc="center left", bbox_to_anchor=(1.0, 0.3),
               fontsize=7, title="color = 2nd var", frameon=False)
    fig.tight_layout()
    save_fig(fig, figdir, "4_mi_vs_linear_summary")

print(f"\nWrote tables to {paths.out}, figures to {figdir}")
print("Run `python b_progress/make_progress_md.py 1c_behavior` to build a markdown summary.")
