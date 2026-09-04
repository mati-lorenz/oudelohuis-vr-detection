# -*- coding: utf-8 -*-
"""
1a_performance
==============
First-pass explorative survey of the raw behavioral dataset. No trial or
session exclusions are applied here (that starts in 1b/1d) -- the goal is
purely descriptive: how much data do we have, how engaged were the
animals, and what does baseline detection performance (d') look like.

Protocols: DM, DP, DN (see a_docs/project_proposal.md).

Caching
-------
Loading and aggregating every session's raw CSVs is the expensive part
of this script; the plots below it are cheap and get iterated on a lot.
So by default this script reuses `out/session_summary.csv` and
`store/all_trialdata.pkl` from a previous run instead of re-reading
0_data/ -- pass `--recompute` to force a fresh pass over the raw data
(e.g. after new sessions have been added).

Outputs
-------
2_pipeline/1a_performance/
    out/    session_summary.csv     one row per session (Rule #1: later
                                     steps -- 1b, 1c, 1d -- read from here
                                     instead of re-touching 0_data/)
            flagged_sessions.csv    sessions worth a second look (too few
                                     trials, no video/pupil, mostly
                                     disengaged, ...)
            figures/*.png           figures worth reporting on (picked up
                                     by b_progress/make_progress_md.py)
    store/  all_trialdata.pkl       concatenated, trial-outcome-labeled
                                     trialdata -- this script's own cache,
                                     reused on the next run unless
                                     --recompute is passed
    tmp/    (nothing by default -- use for your own ad hoc/debug plots
             that aren't meant to go in a report)
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from infotheory.pipeline import get_pipeline_paths
from infotheory.session import load_sessions, PROTOCOLS
from infotheory.behavior import (
    add_trial_outcome,
    build_session_summary_table,
    rolling_performance,
)
from infotheory.plotting import set_style, save_fig, clear_figures, OUTCOME_COLORS, PROTOCOL_COLORS
from infotheory.criteria import MIN_TRIALS, MIN_FRAC_ENGAGED

MAX_PANELS_PER_FIG = 24  # engagement-timeline grid: paginate protocols with more sessions than this
GRID_NCOLS = 4

parser = argparse.ArgumentParser()
parser.add_argument("--recompute", action="store_true",
                     help="Reload every session from 0_data/ instead of reusing the cached "
                          "out/session_summary.csv + store/all_trialdata.pkl")
args = parser.parse_args()

paths = get_pipeline_paths(__file__)
figdir = paths.out / "figures"
clear_figures(figdir)
set_style()


# --------------------------------------------------------------------- #
# 1. Load every session, no exclusions -- or reuse the cache
# --------------------------------------------------------------------- #
summary_cache = paths.out / "session_summary.csv"
trialdata_cache = paths.store / "all_trialdata.pkl"

if not args.recompute and summary_cache.exists() and trialdata_cache.exists():
    print(f"Reusing cached results from {paths.store} / {paths.out} (pass --recompute to reload 0_data/)")
    summary = pd.read_csv(summary_cache)
    all_trialdata = pd.read_pickle(trialdata_cache)
else:
    sessions = load_sessions(protocols=PROTOCOLS, load_behaviordata=True, load_videodata=True)
    if len(sessions) == 0:
        sys.exit("No sessions found under 0_data/ -- nothing to survey.")

    summary = build_session_summary_table(sessions)
    summary.to_csv(summary_cache, index=False)

    all_trialdata = pd.concat(
        [add_trial_outcome(ses.trialdata).assign(session_id=ses.session_id, protocol=ses.protocol)
         for ses in sessions if ses.trialdata is not None],
        ignore_index=True,
    )
    all_trialdata.to_pickle(trialdata_cache)


# --------------------------------------------------------------------- #
# 2. Dataset overview: sessions, trials, duration, per protocol/animal
# --------------------------------------------------------------------- #
overview = (summary.groupby("protocol")
            .agg(n_sessions=("session_id", "nunique"),
                 n_animals=("animal_id", "nunique"),
                 n_trials_total=("n_trials", "sum"),
                 n_trials_median=("n_trials", "median"),
                 duration_min_median=("duration_s", lambda x: np.nanmedian(x) / 60))
            .reset_index())
print("\n=== Dataset overview ===")
print(overview.to_string(index=False))

print(f"\nTotal sessions: {len(summary)}  |  animals: {summary['animal_id'].nunique()}"
      f"  |  trials: {int(summary['n_trials'].sum())}")

fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
sns.countplot(data=summary, x="protocol", hue="animal_id", ax=axes[0], legend=False,
              order=PROTOCOLS)
axes[0].set_title("Sessions per animal")
axes[0].set_ylabel("# sessions")

sns.boxplot(data=summary, x="protocol", y="n_trials", ax=axes[1], order=PROTOCOLS,
            hue="protocol", palette=PROTOCOL_COLORS, legend=False)
sns.stripplot(data=summary, x="protocol", y="n_trials", ax=axes[1], order=PROTOCOLS,
              color="black", alpha=0.5, size=3)
axes[1].axhline(MIN_TRIALS, color="grey", linestyle="--", linewidth=1)
axes[1].set_title("Trials per session")

sns.boxplot(data=summary, x="protocol", y=summary["duration_s"] / 60, ax=axes[2], order=PROTOCOLS,
            hue="protocol", palette=PROTOCOL_COLORS, legend=False)
sns.stripplot(data=summary, x="protocol", y=summary["duration_s"] / 60, ax=axes[2], order=PROTOCOLS,
              color="black", alpha=0.5, size=3)
axes[2].set_ylabel("duration (min)")
axes[2].set_title("Session duration")

fig.tight_layout()
save_fig(fig, figdir, "1_dataset_overview")


# --------------------------------------------------------------------- #
# 3. Engagement: how much, and how it's distributed
# --------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

sns.stripplot(data=summary, x="protocol", y="eng_frac_engaged", order=PROTOCOLS, ax=axes[0],
              hue="protocol", palette=PROTOCOL_COLORS, size=5, legend=False)
sns.pointplot(data=summary, x="protocol", y="eng_frac_engaged", order=PROTOCOLS, ax=axes[0],
              color="black", linestyle="none", errorbar="sd", capsize=0.15)
axes[0].axhline(MIN_FRAC_ENGAGED, color="grey", linestyle="--", linewidth=1)
axes[0].set_ylim(0, 1.02)
axes[0].set_ylabel("fraction of trials engaged")
axes[0].set_title("Engagement per session")

frac_position = summary["eng_first_disengaged_trial"] / summary["n_trials"]
sns.histplot(frac_position.dropna(), bins=20, ax=axes[1], color="#555555")
axes[1].set_xlabel("trial position (fraction of session) of first\nsustained disengagement")
axes[1].set_title("When engagement drops")
axes[1].set_xlim(0, 1)

fig.tight_layout()
save_fig(fig, figdir, "2_engagement_overview")


# --------------------------------------------------------------------- #
# 2b. Raw engagement timeline, one small subplot per session, grouped by
#     protocol. Unlike the trajectory plot below, this shows the actual
#     engaged/disengaged trace (no smoothing) so individual lapses and
#     mislabeled trials are visible. Paginated so a protocol with many
#     sessions still renders as a readable grid.
# --------------------------------------------------------------------- #
for protocol in PROTOCOLS:
    subset = all_trialdata[all_trialdata["protocol"] == protocol]
    if subset.empty or "engaged" not in subset.columns:
        continue

    session_ids = subset["session_id"].unique()
    n_sessions = len(session_ids)
    n_pages = int(np.ceil(n_sessions / MAX_PANELS_PER_FIG))

    for page in range(n_pages):
        page_sessions = session_ids[page * MAX_PANELS_PER_FIG:(page + 1) * MAX_PANELS_PER_FIG]
        n_panels = len(page_sessions)
        ncols = min(GRID_NCOLS, n_panels)
        nrows = int(np.ceil(n_panels / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 1.7),
                                  squeeze=False, sharey=True)
        for i, session_id in enumerate(page_sessions):
            ax = axes[i // ncols][i % ncols]
            sdata = subset[subset["session_id"] == session_id].sort_values("trialNumber")
            ax.step(sdata["trialNumber"], sdata["engaged"], where="post", color="black", linewidth=1)
            ax.set_title(session_id, fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks([0, 1])
            ax.tick_params(labelsize=7)
        for j in range(n_panels, nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)

        fig.supxlabel("trial number", fontsize=9)
        fig.supylabel("engaged (0/1)", fontsize=9)
        page_suffix = f"_page{page + 1}" if n_pages > 1 else ""
        fig.suptitle(f"Engagement across the session -- {protocol}"
                     + (f" (page {page + 1}/{n_pages})" if n_pages > 1 else ""), y=1.02)
        fig.tight_layout()
        save_fig(fig, figdir, f"3_engagement_timeline_{protocol}{page_suffix}")


# --------------------------------------------------------------------- #
# 4. Population-level rolling hit-rate/d' trajectory across the session,
#    as a smoothed complement to the raw timeline above (computed from
#    the cached all_trialdata -- no Session objects needed).
# --------------------------------------------------------------------- #
traj_rows = []
for session_id, sdata in all_trialdata.groupby("session_id"):
    if len(sdata) < 20:
        continue
    roll = rolling_performance(sdata.sort_values("trialNumber"))
    roll["frac_session"] = roll["trialNumber"] / roll["trialNumber"].max()
    roll["protocol"] = sdata["protocol"].iloc[0]
    roll["session_id"] = session_id
    traj_rows.append(roll)
trajectories = pd.concat(traj_rows, ignore_index=True)
trajectories["frac_bin"] = pd.cut(trajectories["frac_session"], bins=20, labels=False)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharex=True)
for ax, col, ylabel in zip(axes, ["rolling_hitrate", "rolling_dprime"],
                           ["rolling hit rate", "rolling d'"]):
    sns.lineplot(data=trajectories, x="frac_bin", y=col, hue="protocol",
                 hue_order=PROTOCOLS, palette=PROTOCOL_COLORS, ax=ax, errorbar="se")
    ax.set_xlabel("session progress (binned)")
    ax.set_ylabel(ylabel)
axes[0].legend(title=None, fontsize=8)
axes[1].get_legend().remove()
fig.suptitle("Smoothed performance trajectory across the session", y=1.02)
fig.tight_layout()
save_fig(fig, figdir, "4_performance_trajectory")


# --------------------------------------------------------------------- #
# 5. Trial-outcome distribution (HIT/MISS/FA/CR) per protocol
# --------------------------------------------------------------------- #
catch = all_trialdata.dropna(subset=["trialOutcome"])
outcome_counts = catch.groupby(["protocol", "trialOutcome"]).size().unstack(fill_value=0)
pivot = outcome_counts.div(outcome_counts.sum(axis=1), axis=0).reindex(
    index=PROTOCOLS, columns=["HIT", "CR", "MISS", "FA"])

fig, ax = plt.subplots(figsize=(5, 3.2))
pivot.plot(kind="bar", stacked=True, ax=ax, color=[OUTCOME_COLORS[c] for c in pivot.columns])
ax.set_ylabel("fraction of catch trials")
ax.set_title("Trial outcomes (0%/100% signal trials)")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
fig.tight_layout()
save_fig(fig, figdir, "5_trial_outcomes")


# --------------------------------------------------------------------- #
# 6. d-prime and criterion distributions (all trials vs engaged-only)
# --------------------------------------------------------------------- #
dprime_long = summary.melt(id_vars=["protocol", "session_id"],
                            value_vars=["dprime_all", "dprime_engaged"],
                            var_name="trial_set", value_name="dprime")
dprime_long["trial_set"] = dprime_long["trial_set"].str.replace("dprime_", "")

fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
sns.violinplot(data=dprime_long, x="protocol", y="dprime", hue="trial_set", order=PROTOCOLS,
               inner=None, ax=axes[0])
sns.swarmplot(data=dprime_long, x="protocol", y="dprime", hue="trial_set", order=PROTOCOLS,
              dodge=True, size=3, palette={"all": "black", "engaged": "black"}, alpha=0.6,
              ax=axes[0], legend=False)
axes[0].axhline(0, color="grey", linewidth=0.8)
axes[0].set_title("d' by protocol (all vs engaged trials)")
axes[0].legend(title=None, fontsize=8)

sns.scatterplot(data=summary, x="dprime_all", y="criterion_all", hue="protocol",
                 hue_order=PROTOCOLS, palette=PROTOCOL_COLORS, ax=axes[1])
axes[1].axhline(0, color="grey", linewidth=0.8)
axes[1].axvline(1, color="grey", linewidth=0.8)
axes[1].set_title("Bias vs. sensitivity (all trials)")
axes[1].legend(fontsize=8)

sns.scatterplot(data=summary, x="dprime_engaged", y="criterion_engaged", hue="protocol",
                 hue_order=PROTOCOLS, palette=PROTOCOL_COLORS, ax=axes[2])
axes[2].axhline(0, color="grey", linewidth=0.8)
axes[2].axvline(1, color="grey", linewidth=0.8)
axes[2].set_title("Bias vs. sensitivity (eng trials)")
axes[2].legend(fontsize=8)
fig.tight_layout()
save_fig(fig, figdir, "6_dprime_criterion")


# --------------------------------------------------------------------- #
# 7. Signal-strength trial counts: are conditions balanced?
# --------------------------------------------------------------------- #
fig, axes = plt.subplots(1, len(PROTOCOLS), figsize=(11, 3), sharey=False)
for ax, protocol in zip(axes, PROTOCOLS):
    subset = all_trialdata[all_trialdata["protocol"] == protocol]
    if subset.empty:
        ax.set_visible(False)
        continue
    counts = subset.groupby(["session_id", "signal"]).size().reset_index(name="n_trials")
    sns.boxplot(data=counts, x="signal", y="n_trials", ax=ax, color=PROTOCOL_COLORS[protocol])
    sns.stripplot(data=counts, x="signal", y="n_trials", ax=ax, color="black", alpha=0.4, size=3)
    ax.set_title(protocol)
    ax.set_xlabel("signal strength (%)")
fig.suptitle("Trials per signal level, per session", y=1.03)
fig.tight_layout()
save_fig(fig, figdir, "7_signal_level_balance")


# --------------------------------------------------------------------- #
# 8. Data-quality flags for downstream steps
# --------------------------------------------------------------------- #
flags = []
for _, row in summary.iterrows():
    reasons = []
    if row["n_trials"] < MIN_TRIALS:
        reasons.append(f"only {row['n_trials']} trials (< {MIN_TRIALS})")
    if not row["has_behaviordata"]:
        reasons.append("missing behaviordata")
    if not row["has_videodata"]:
        reasons.append("missing videodata")
    if not np.isnan(row.get("eng_frac_engaged", np.nan)) and row["eng_frac_engaged"] < MIN_FRAC_ENGAGED:
        reasons.append(f"only {row['eng_frac_engaged']:.0%} of trials engaged")
    if reasons:
        flags.append({"session_id": row["session_id"], "protocol": row["protocol"],
                       "reasons": "; ".join(reasons)})

flagged = pd.DataFrame(flags)
flagged.to_csv(paths.out / "flagged_sessions.csv", index=False)
print(f"\nFlagged {len(flagged)}/{len(summary)} sessions for a closer look "
      f"(see {paths.out / 'flagged_sessions.csv'}):")
if len(flagged):
    print(flagged.to_string(index=False))

print(f"\nWrote tables to {paths.out}, figures to {figdir}")
print(f"Run `python b_progress/make_progress_md.py 1a_performance` to build a markdown summary.")
