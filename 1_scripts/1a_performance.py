# -*- coding: utf-8 -*-
"""
1a_performance
==============
First-pass explorative survey of the raw behavioral dataset. No trial or session exclusions are applied
here (that starts in 1b/1d) -- the goal is purely descriptive: how much data do we have, how engaged were
the animals, and what does baseline detection performance (d') look like.

Protocols: DM, DP, DN (see a_docs/project_proposal.md).

Project-wide rules implemented here (see infotheory/project_config.py; every script follows them)
-------------------------------------------------------------------------------------------------
1. SELECTION      --animals DNL00 DNL01  /  --sessions DNL00_2024_01_09   (or set ANIMALS / SESSIONS in
                  project_config.py). A restricted run writes to out/selection/<tag>/ and NEVER touches
                  the full-dataset out/ files that later steps (1b, 1c, ...) read.
2. AGGREGATION    --aggregate pooled|session|mouse   (or AGGREGATE_BY in project_config.py). The unit of
                  analysis for every plot and reported table:
                     pooled : everything pooled (counts summed, rates / d' recomputed from pooled counts)
                     session: one unit per session; bars = mean over sessions +- SEM, dots = sessions
                     mouse  : one unit per mouse (its sessions pooled first); bars = mean over mice +- SEM
3. MEMORY         The dataset is never loaded at once. Each worker loads ONE session (behavior + video),
                  reduces it to a summary row plus a slim trial table (6 columns), caches that in store/
                  and frees it. Peak RAM ~ n_jobs x one session (--n-jobs / N_JOBS_SESSIONS). Only the
                  slim derived tables of all sessions are held together afterwards.
4. BINNING        Not used in 1a (no neural/information measures). Shared binning lives in
                  project_config.py for the steps that need it.

Caching
-------
Per-session results are cached in store/ (file name = hash of session id, settings and a fingerprint of
the session's raw files), so re-runs only compute new/changed sessions. --recompute ignores the caches.

Outputs
-------
2_pipeline/1a_performance/
    out/    session_summary.csv     one row per session (Rule #1: later steps read this)
            flagged_sessions.csv    sessions worth a second look
            unit_summary_<aggregate>.csv   the table behind the figures, at the chosen unit of analysis
            figures/*.png
    out/selection/<tag>/...         same files for a restricted run (never overwrites the above)
    store/  session_behavior_<hash>.pkl   per-session cache
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from infotheory.pipeline import get_pipeline_paths
from infotheory.session import discover_sessions, PROTOCOLS
from infotheory.behavior import add_trial_outcome, summarize_session, rolling_performance
from infotheory.plotting import set_style, save_fig, clear_figures, OUTCOME_COLORS, PROTOCOL_COLORS
from infotheory.criteria import MIN_TRIALS, MIN_FRAC_ENGAGED
from infotheory.scope import (add_scope_args, scope_from_args, select_sessions, scoped_dirs, data_stamp,
                              unit_columns, unit_noun)
from infotheory.session_jobs import run_sessions
from infotheory.unit_stats import unit_summary
from infotheory.unit_plots import unit_bars, mouse_palette, mouse_legend

SIGNAL_PRESENT, SIGNAL_ABSENT = 100, 0
TRIAL_COLUMNS = ["trialNumber", "signal", "lickResponse", "engaged", "trialOutcome"]
MAX_PANELS_PER_FIG = 24  # engagement-timeline grid: paginate protocols with more panels than this
GRID_NCOLS = 4
N_FRAC_BINS = 20         # session-progress bins of the performance trajectory

parser = argparse.ArgumentParser()
add_scope_args(parser)
args = parser.parse_args()
scope = scope_from_args(args)
AGG = scope.aggregate
UNIT = unit_noun(AGG)

paths = get_pipeline_paths(__file__)
table_dir, figdir = scoped_dirs(paths, scope)
clear_figures(figdir)
set_style()
print(scope.describe())


# --------------------------------------------------------------------- #
# 1. Select sessions (metadata only), then process ONE session at a time
# --------------------------------------------------------------------- #
all_triples = discover_sessions(PROTOCOLS)
if not all_triples:
    sys.exit("No sessions found under 0_data/ -- nothing to survey.")
triples = select_sessions(all_triples, scope)
session_ids = [f"{a}_{d}" for _, a, d in triples]
stamps = {f"{a}_{d}": data_stamp(p, a, d) for p, a, d in triples}
mouse_colors = mouse_palette(a for _, a, _ in all_triples)   # colours stable even when a subset is shown
print(f"{len(triples)}/{len(all_triples)} sessions selected "
      f"({len({a for _, a, _ in triples})} mice)")


def process_session(ses, seed):
    """Reduce one loaded session to (summary row, slim trial table). Nothing else is kept."""
    summary_row = summarize_session(ses, SIGNAL_PRESENT, SIGNAL_ABSENT)
    trials = None
    if ses.trialdata is not None and len(ses.trialdata):
        t = add_trial_outcome(ses.trialdata, SIGNAL_PRESENT, SIGNAL_ABSENT)
        t = t[[c for c in TRIAL_COLUMNS if c in t.columns]].copy()
        t.insert(0, "session_id", ses.session_id)
        t.insert(1, "animal_id", ses.animal_id)
        t.insert(2, "protocol", ses.protocol)
        trials = t
    return {"summary": summary_row, "trials": trials}


results = run_sessions(
    session_ids, process_session, list(PROTOCOLS), paths.store, "session_behavior",
    key_params=dict(version="1a-v2", present=SIGNAL_PRESENT, absent=SIGNAL_ABSENT, trial_columns=TRIAL_COLUMNS),
    load_kwargs=dict(load_behaviordata=True, load_videodata=True),
    n_jobs=scope.n_jobs, recompute=args.recompute, extra_key_by_session=stamps, flatten=False)

results = [r for r in results if r is not None]
if not results:
    sys.exit("None of the selected sessions could be loaded.")
summary = pd.DataFrame([r["summary"] for r in results])
trials = pd.concat([r["trials"] for r in results if r["trials"] is not None], ignore_index=True)
del results

summary.to_csv(table_dir / "session_summary.csv", index=False)
units = unit_summary(summary, AGG, by=["protocol"])
units.to_csv(table_dir / f"unit_summary_{AGG}.csv", index=False)

protocols_present = [p for p in PROTOCOLS if p in summary["protocol"].unique()]
unit_cols = unit_columns(AGG)


# --------------------------------------------------------------------- #
# 2. Dataset overview
# --------------------------------------------------------------------- #
print(f"\n=== Dataset overview (unit of analysis: {UNIT}) ===")
overview = (summary.groupby("protocol")
            .agg(n_sessions=("session_id", "nunique"), n_animals=("animal_id", "nunique"),
                 n_trials_total=("n_trials", "sum"), n_trials_median=("n_trials", "median"),
                 duration_min_median=("duration_s", lambda x: np.nanmedian(x) / 60))
            .reindex(protocols_present).reset_index())
print(overview.to_string(index=False))
print(f"\nTotal sessions: {len(summary)}  |  animals: {summary['animal_id'].nunique()}"
      f"  |  trials: {int(summary['n_trials'].sum())}")

fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
# sessions per mouse (always per mouse -- this panel is about the dataset itself)
cnt = summary.groupby(["animal_id", "protocol"]).size().unstack(fill_value=0).reindex(columns=protocols_present)
cnt.plot(kind="bar", stacked=True, ax=axes[0], color=[PROTOCOL_COLORS[p] for p in cnt.columns], width=0.8)
axes[0].set_title("Sessions per mouse")
axes[0].set_ylabel("# sessions")
axes[0].set_xlabel("")
axes[0].tick_params(axis="x", labelsize=7, rotation=60)
axes[0].legend(title=None, fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
pcol = {p: PROTOCOL_COLORS[p] for p in protocols_present}
unit_bars(axes[1], units, "protocol", "n_trials", protocols_present, AGG, colors=pcol, mouse_colors=mouse_colors,
          ylabel=f"trials per {UNIT}", hline=MIN_TRIALS if AGG == "session" else None)
axes[1].set_title(f"Trials per {UNIT}")
unit_bars(axes[2], units, "protocol", "duration_min", protocols_present, AGG, colors=pcol, mouse_colors=mouse_colors,
          ylabel=f"duration per {UNIT} (min)")
axes[2].set_title(f"Recording time per {UNIT}")
mouse_legend(axes[2], mouse_colors, animals=summary["animal_id"].unique(), loc="upper left", bbox_to_anchor=(1.0, 1.0))
fig.suptitle(f"Dataset overview -- unit: {UNIT}", y=1.03)
fig.tight_layout()
save_fig(fig, figdir, "1_dataset_overview")
plt.close(fig)


# --------------------------------------------------------------------- #
# 3. Engagement
# --------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
unit_bars(axes[0], units, "protocol", "frac_engaged", protocols_present, AGG, colors=pcol, mouse_colors=mouse_colors,
          ylabel=f"fraction of trials engaged (per {UNIT})", hline=MIN_FRAC_ENGAGED)
axes[0].set_ylim(0, 1.02)
axes[0].set_title(f"Engagement per {UNIT}")
if AGG == "pooled":   # pooled: one value per protocol -> show the distribution over sessions instead
    first_dis = (summary["eng_first_disengaged_trial"] / summary["n_trials"]).dropna()
    note = "sessions"
else:
    first_dis = units["first_disengaged_frac"].dropna()
    note = f"{UNIT}s (mean over a unit's sessions)"
axes[1].hist(first_dis, bins=np.linspace(0, 1, 21), color="#555555")
axes[1].set_xlabel("trial position (fraction of session) of first\nsustained disengagement")
axes[1].set_ylabel(f"# {note}", fontsize=8)
axes[1].set_title("When engagement drops")
axes[1].set_xlim(0, 1)
fig.tight_layout()
save_fig(fig, figdir, "2_engagement_overview")
plt.close(fig)


# --------------------------------------------------------------------- #
# 3b. Raw engagement timeline. One panel per analysis unit:
#     session -> one panel per session; mouse -> one per mouse (its sessions as lines);
#     pooled  -> one per protocol (all sessions as lines).
# --------------------------------------------------------------------- #
for protocol in protocols_present:
    sub = trials[trials["protocol"] == protocol]
    if sub.empty or "engaged" not in sub.columns:
        continue
    if AGG == "session":
        panels = [(sid, [sid]) for sid in sub["session_id"].unique()]
    elif AGG == "mouse":
        panels = [(a, list(g["session_id"].unique())) for a, g in sub.groupby("animal_id")]
    else:
        panels = [(f"{protocol} (all sessions pooled)", list(sub["session_id"].unique()))]
    n_pages = int(np.ceil(len(panels) / MAX_PANELS_PER_FIG))
    for page in range(n_pages):
        page_panels = panels[page * MAX_PANELS_PER_FIG:(page + 1) * MAX_PANELS_PER_FIG]
        ncols = min(GRID_NCOLS, len(page_panels))
        nrows = int(np.ceil(len(page_panels) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 1.7), squeeze=False, sharey=True)
        for i, (title, sids) in enumerate(page_panels):
            ax = axes[i // ncols][i % ncols]
            for sid in sids:
                sdata = sub[sub["session_id"] == sid].sort_values("trialNumber")
                ax.step(sdata["trialNumber"], sdata["engaged"], where="post", color="black",
                        linewidth=1 if len(sids) == 1 else 0.6, alpha=1 if len(sids) == 1 else 0.35)
            ax.set_title(title if len(sids) == 1 else f"{title} (n={len(sids)} sessions)", fontsize=8)
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks([0, 1])
            ax.tick_params(labelsize=7)
        for j in range(len(page_panels), nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)
        fig.supxlabel("trial number", fontsize=9)
        fig.supylabel("engaged (0/1)", fontsize=9)
        suffix = f"_page{page + 1}" if n_pages > 1 else ""
        fig.suptitle(f"Engagement across the session -- {protocol}, unit: {UNIT}"
                     + (f" (page {page + 1}/{n_pages})" if n_pages > 1 else ""), y=1.02)
        fig.tight_layout()
        save_fig(fig, figdir, f"3_engagement_timeline_{protocol}{suffix}")
        plt.close(fig)


# --------------------------------------------------------------------- #
# 4. Rolling hit-rate / d' trajectory across the session, one curve per unit (thin) and the mean over
#    units (thick, +- SEM over units; pooled: mean over all trials, no error band)
# --------------------------------------------------------------------- #
traj_rows = []
for sid, sdata in trials.groupby("session_id"):
    if len(sdata) < 20:
        continue
    roll = rolling_performance(sdata.sort_values("trialNumber"), signal_present_value=SIGNAL_PRESENT,
                               signal_absent_value=SIGNAL_ABSENT)
    roll["frac_session"] = roll["trialNumber"] / roll["trialNumber"].max()
    roll["protocol"], roll["session_id"], roll["animal_id"] = sdata["protocol"].iloc[0], sid, sdata["animal_id"].iloc[0]
    traj_rows.append(roll)
if traj_rows:
    traj = pd.concat(traj_rows, ignore_index=True)
    traj["frac_bin"] = pd.cut(traj["frac_session"], bins=N_FRAC_BINS, labels=False)
    ukeys = ["protocol"] + {"pooled": [], "session": ["animal_id", "session_id"], "mouse": ["animal_id"]}[AGG]
    metrics = ["rolling_hitrate", "rolling_dprime"]
    curves = traj.groupby(ukeys + ["frac_bin"])[metrics].mean().reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharex=True)
    for ax, col, ylabel in zip(axes, metrics, ["rolling hit rate", "rolling d'"]):
        for protocol in protocols_present:
            c = curves[curves["protocol"] == protocol]
            if c.empty:
                continue
            if AGG != "pooled":
                for _, u in c.groupby(ukeys[1:] if len(ukeys) > 1 else ["protocol"]):
                    ax.plot(u["frac_bin"], u[col], color=PROTOCOL_COLORS[protocol], alpha=0.15, linewidth=0.7)
            g = c.groupby("frac_bin")[col]
            m = g.mean()
            ax.plot(m.index, m.values, color=PROTOCOL_COLORS[protocol], linewidth=2, label=protocol)
            if AGG != "pooled":
                sem = g.sem().fillna(0)
                ax.fill_between(m.index, m - sem, m + sem, color=PROTOCOL_COLORS[protocol], alpha=0.25)
        ax.set_xlabel("session progress (binned)")
        ax.set_ylabel(ylabel)
    axes[0].legend(title=None, fontsize=8, frameon=False)
    fig.suptitle(f"Smoothed performance trajectory across the session -- unit: {UNIT}", y=1.02)
    fig.tight_layout()
    save_fig(fig, figdir, "4_performance_trajectory")
    plt.close(fig)


# --------------------------------------------------------------------- #
# 5. Trial outcomes (HIT/MISS/FA/CR) on the 0%/100% catch trials, one stacked bar per unit
# --------------------------------------------------------------------- #
ucat = units.copy()
cnt_cols = [f"n_{o}_all" for o in ("HIT", "MISS", "FA", "CR")]
tot = ucat[cnt_cols].sum(axis=1).replace(0, np.nan)
for o, c in zip(("HIT", "MISS", "FA", "CR"), cnt_cols):
    ucat[o] = ucat[c] / tot
ucat["protocol"] = pd.Categorical(ucat["protocol"], protocols_present, ordered=True)
ucat = ucat.sort_values(["protocol"] + [c for c in ("animal_id", "session_id") if c in ucat.columns])
labels = (ucat["protocol"].astype(str) if AGG == "pooled" else
          ucat["protocol"].astype(str) + " " + (ucat["session_id"] if AGG == "session" else ucat["animal_id"]))
fig, ax = plt.subplots(figsize=(max(5, 0.3 * len(ucat) + 2), 3.6))
bottom = np.zeros(len(ucat))
for o in ("HIT", "MISS", "FA", "CR"):
    v = ucat[o].fillna(0).to_numpy()
    ax.bar(range(len(ucat)), v, bottom=bottom, color=OUTCOME_COLORS[o], label=o, width=0.85)
    bottom += v
ax.set_xticks(range(len(ucat)))
ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=6 if len(ucat) > 12 else 8)
ax.set_ylabel("fraction of catch trials")
ax.set_title(f"Trial outcomes (0%/100% signal trials) -- one bar per {UNIT}")
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
fig.tight_layout()
save_fig(fig, figdir, "5_trial_outcomes")
plt.close(fig)


# --------------------------------------------------------------------- #
# 6. d-prime and criterion (all trials vs engaged-only), computed from the unit's pooled counts
# --------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
long = pd.concat([units.assign(cat=units["protocol"] + "\n" + lab, dprime=units[f"dprime_{lab}"])
                  for lab in ("all", "engaged")], ignore_index=True)
order = [f"{p}\n{lab}" for p in protocols_present for lab in ("all", "engaged")]
unit_bars(axes[0], long, "cat", "dprime", order, AGG, colors={o: PROTOCOL_COLORS[o.split("\n")[0]] for o in order},
          mouse_colors=mouse_colors, ylabel="d'", hline=0)
axes[0].tick_params(axis="x", labelsize=7)
axes[0].set_title(f"d' per {UNIT} (all vs engaged)", fontsize=9)
for ax, lab in zip(axes[1:], ("all", "engaged")):
    for protocol in protocols_present:
        u = units[units["protocol"] == protocol]
        ax.scatter(u[f"dprime_{lab}"], u[f"criterion_{lab}"], s=28, color=PROTOCOL_COLORS[protocol],
                   edgecolor="black", linewidth=0.3, label=protocol)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("d'")
    ax.set_ylabel("criterion")
    ax.set_title(f"Bias vs. sensitivity ({lab}), per {UNIT}", fontsize=9)
axes[1].legend(fontsize=8, frameon=False)
fig.tight_layout()
save_fig(fig, figdir, "6_dprime_criterion")
plt.close(fig)


# --------------------------------------------------------------------- #
# 7. Signal-strength trial counts per unit: are conditions balanced?
# --------------------------------------------------------------------- #
ckeys = ["protocol"] + {"pooled": [], "session": ["animal_id", "session_id"], "mouse": ["animal_id"]}[AGG]
counts = trials.groupby(ckeys + ["signal"]).size().reset_index(name="n_trials")
fig, axes = plt.subplots(1, len(protocols_present), figsize=(3.8 * len(protocols_present), 3.2), squeeze=False)
for ax, protocol in zip(axes[0], protocols_present):
    c = counts[counts["protocol"] == protocol]
    levels = sorted(c["signal"].unique())
    unit_bars(ax, c, "signal", "n_trials", levels, AGG, colors=PROTOCOL_COLORS[protocol], mouse_colors=mouse_colors,
              ylabel=f"trials per {UNIT}")
    ax.set_title(protocol)
    ax.set_xlabel("signal strength (%)")
fig.suptitle(f"Trials per signal level, per {UNIT}", y=1.03)
fig.tight_layout()
save_fig(fig, figdir, "7_signal_level_balance")
plt.close(fig)


# --------------------------------------------------------------------- #
# 8. Data-quality flags for downstream steps (always per session)
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
        flags.append({"session_id": row["session_id"], "protocol": row["protocol"], "reasons": "; ".join(reasons)})
flagged = pd.DataFrame(flags, columns=["session_id", "protocol", "reasons"])
flagged.to_csv(table_dir / "flagged_sessions.csv", index=False)
print(f"\nFlagged {len(flagged)}/{len(summary)} sessions for a closer look (see {table_dir / 'flagged_sessions.csv'}):")
if len(flagged):
    print(flagged.to_string(index=False))

print(f"\nWrote tables to {table_dir}, figures to {figdir}")
print("Run `python b_progress/make_progress_md.py 1a_performance` to build a markdown summary.")
