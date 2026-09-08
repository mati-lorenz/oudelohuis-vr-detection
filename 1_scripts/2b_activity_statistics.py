# -*- coding: utf-8 -*-
"""
2b_activity_statistics
========================
Single-cell statistical analysis of the deconvolved (\"spike\") calcium
activity -- DN sessions exclusively, same reason as 2a_cell_distribution
(cell/calcium recording only happens for DN). This is a SANITY-CHECK /
threshold-picking step, not a filtering step: every analysis here runs
on the FULL, unfiltered population from celldata.csv/deconvdata.csv --
no anatomical, proximity, or QC exclusion is ever applied to what gets
plotted. The one exception is the informational qc_per_cell.csv output
(see section 4): it computes and reports which cells WOULD fail under
the default thresholds, purely so later 2_ steps have a QC reference to
opt into -- it doesn't remove anything here.

Ported from the lab's plot_spike_exploratory.py + qc_lib.py, adapted to
this project's schema and to what's actually computable without a
trial-response-matrix pipeline (which doesn't exist yet in this
project -- population coupling and split-half reliability, which need
one, are deferred to whichever future 2_ step builds it; everything
here operates on the continuous, session-long deconvolved trace, or on
approximate frame timestamps for the runspeed check, per section 2's
note on that approximation).

What this does
--------------
1. Loads celldata + deconvdata for every DN session that has both, no
   session-level exclusion of any kind (deliberately NOT reading 1b's
   included-session list -- 1b's criteria are about task performance,
   irrelevant to whether a session's recording quality is sane).
2. Per-neuron activity statistics (infotheory.spike_stats): event rate
   (magnitude- and count-based), sparsity, inter-event-interval (IEI)
   median + CV, Fano factor (magnitude- and count-based), autocorrelogram,
   and a running-speed correlation (movement-artifact check).
3. QC metrics (infotheory.qc_lib): per-neuron rate/std/skew/NaN-fraction/
   Fano-based flags against tunable thresholds (see QC_THRESHOLDS below),
   combined with celldata's own 'noise_level' column.
4. Every plot splits by area AND label (unlabeled vs labeled V1/PM;
   matching 2a_cell_distribution.py's convention) -- never pooled across
   areas or label status, per the project's standing convention that
   V1/PM unlabeled and labeled populations are conceptually different
   groups worth seeing separately.

Outputs
-------
2_pipeline/2b_activity_statistics/
    out/    activity_per_neuron_summary.csv   one row per neuron: area,
                                                label, session_id, cell_id,
                                                every stat computed here
            qc_per_cell.csv                    session_id, cell_id,
                                                qc_pass, qc_fail_reason,
                                                per-reason flags -- same
                                                format as the lab's own
                                                qc_per_cell.csv, so
                                                qc_lib.load_qc_pass-style
                                                consumption works if a
                                                later step wants it (this
                                                step itself never filters
                                                on it)
            qc_summary_by_group.csv            pass/fail counts by
                                                session x area x label
            figures/*.png
    store/  activity_combined.pkl              cached per-neuron table +
                                                pooled IEI/autocorrelogram
                                                arrays (reused unless
                                                --recompute)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from infotheory.pipeline import get_pipeline_paths
from infotheory.session import load_sessions
from infotheory.celldata_utils import (
    get_area_label, ordered_groups, bar_by_group, AREA_COLORS, LABEL_LINESTYLES, DEFAULT_LABEL_ORDER,
    LABEL_SPLIT_AREAS,
)
from infotheory import spike_stats as ss
from infotheory import qc_lib
from infotheory.plotting import set_style, save_fig, clear_figures

PROTOCOL = "DN"

FANO_WINDOW_SEC = 1.0
MAX_LAG_SEC_AUTOCORR = 2.0
MAX_NEURONS_FOR_IEI_AUTOCORR = 300  # per session -- IEI/autocorrelogram loop per-neuron in
                                     # plain Python (see spike_stats.py), so cap the cost;
                                     # tracked explicitly so the subsample's area/label
                                     # identity stays known (see process_session below)
RANDOM_STATE = 0

FCHAN2_ARTIFACT_Z = 3.0  # |Fchan2| above this = a candidate motion/z-drift artifact frame
                          # (Fchan2 is already z-scored -- see session.py's module docstring)

# QC thresholds -- tunable starting points from the lab's own qc_lib.py;
# this script's whole purpose is to help you decide whether these are
# right for THIS dataset (see the QC histograms below, with these drawn
# as reference lines) rather than to enforce them.
QC_THRESHOLDS = dict(
    rate_thr=qc_lib.RATE_THR, noise_thr=qc_lib.NOISE_THR, fano_thr=qc_lib.FANO_THR,
    skew_min=qc_lib.SKEW_MIN, flat_std_thr=qc_lib.FLAT_STD_THR, nan_frac_thr=qc_lib.NAN_FRAC_THR,
)

N_JOBS_SESSIONS = -1
N_JOBS_NEURONS = 1   # keep at 1 when N_JOBS_SESSIONS parallelizes across sessions already,
                     # to avoid oversubscribing CPUs with nested parallel pools
JOBLIB_VERBOSE = 5

parser = argparse.ArgumentParser()
parser.add_argument("--recompute", action="store_true",
                     help="Reload sessions from 0_data/ instead of reusing the cached "
                          "store/activity_combined.pkl")
args = parser.parse_args()

paths = get_pipeline_paths(__file__)
figdir = paths.out / "figures"
clear_figures(figdir)
set_style()


# --------------------------------------------------------------------- #
# 1. Load celldata + deconvdata for every DN session -- or reuse cache.
#    Deliberately no upstream inclusion filter (see module docstring):
#    every DN session with both files present is used.
# --------------------------------------------------------------------- #
combined_cache = paths.store / "activity_combined.pkl"
REQUIRED_COLUMNS = ["session_id", "roi_name", "labeled", "qc_rate", "qc_fano", "active_frame_rate_hz",
                    "fchan2_corr"]

cached_data = None
if not args.recompute and combined_cache.exists():
    try:
        candidate = pd.read_pickle(combined_cache)
        missing = [c for c in REQUIRED_COLUMNS if c not in candidate["per_neuron"].columns]
        if missing:
            print(f"Cached activity table is missing columns {missing} -- looks like it was built "
                  f"by an older version of this script. Recomputing from 0_data/ instead of using "
                  f"{combined_cache} (safe to delete that file to silence this check in the future).")
        else:
            cached_data = candidate
            print(f"Reusing cached activity tables from {paths.store} (pass --recompute to reload 0_data/)")
    except Exception as e:
        print(f"Could not load cached activity tables ({e}); recomputing from 0_data/ ...")

if cached_data is None:
    sessions = load_sessions(protocols=[PROTOCOL], load_behaviordata=True, load_videodata=False,
                              load_celldata=True, load_calciumdata=True)
    sessions = [ses for ses in sessions if ses.celldata is not None and ses.calciumdata is not None]
    print(f"{len(sessions)} {PROTOCOL} sessions have both celldata and deconvdata")

    def process_session(ses, session_index):
        """All per-session computation, returned as a small, fully
        picklable dict -- dispatched via joblib.Parallel across
        sessions."""
        rng = np.random.default_rng(RANDOM_STATE + session_index)
        fs = ss.get_frame_rate(ses, fallback=8.0)
        calciumdata = ses.calciumdata.to_numpy()
        celldata = ses.celldata.reset_index(drop=True)
        area, label, _ = get_area_label(celldata, label_split_areas=LABEL_SPLIT_AREAS)
        T, N = calciumdata.shape

        rate = ss.event_rate(calciumdata, fs)
        active_rate = ss.active_frame_rate(calciumdata, fs)
        spars = ss.sparsity(calciumdata)
        fano = ss.fano_factor(calciumdata, fs, window_sec=FANO_WINDOW_SEC)
        active_fano = ss.fano_factor(calciumdata, fs, window_sec=FANO_WINDOW_SEC, threshold=0.0)

        qc = qc_lib.compute_qc_metrics(calciumdata, fs, fano_window_sec=FANO_WINDOW_SEC)

        # Movement-artifact check: use REAL per-frame imaging timestamps
        # (ses.ts_F, from Ftsdata.csv) when available; fall back to
        # approximating them as evenly spaced across the session only if
        # that file is missing (e.g. older/synthetic data without it).
        runspeed_corr = np.full(N, np.nan)
        if ses.behaviordata is not None and "runspeed" in ses.behaviordata.columns and T > 1:
            ts_bhv = ses.behaviordata["ts"].to_numpy()
            if ses.ts_F is not None and len(ses.ts_F) == T:
                ts_F = ses.ts_F
            else:
                duration = ts_bhv[-1] - ts_bhv[0] if len(ts_bhv) > 1 else T / fs
                ts_F = np.linspace(0, duration, T) + ts_bhv[0]
            runspeed_F = np.interp(ts_F, ts_bhv, ses.behaviordata["runspeed"].to_numpy())
            runspeed_corr = ss.runspeed_correlation(calciumdata, runspeed_F)

        # Motion/z-drift-artifact check: correlation between each cell's
        # activity and the session-wide Fchan2 signal (ses.fchan2, from
        # Fchan2data.csv -- a z-scored red-channel/structural-marker
        # trace; abrupt changes flag likely FOV motion, see session.py's
        # module docstring). Reuses runspeed_correlation's logic directly
        # -- it's just "correlate calciumdata against a 1D reference
        # trace", regardless of what that reference physically is.
        fchan2_corr = np.full(N, np.nan)
        fchan2_artifact_frac = np.nan
        if ses.fchan2 is not None and len(ses.fchan2) == T:
            fchan2_corr = ss.runspeed_correlation(calciumdata, ses.fchan2)
            fchan2_artifact_frac = float(np.mean(np.abs(ses.fchan2) > FCHAN2_ARTIFACT_Z))

        # Subsampled, per-neuron-loop stats (IEI + autocorrelogram)
        subsample_idx = np.arange(N)
        if N > MAX_NEURONS_FOR_IEI_AUTOCORR:
            subsample_idx = rng.choice(N, size=MAX_NEURONS_FOR_IEI_AUTOCORR, replace=False)
        calciumdata_sub = calciumdata[:, subsample_idx]
        area_sub, label_sub = area[subsample_idx], label[subsample_idx]

        median_iei, cv_iei, _ = ss.iei_stats(calciumdata_sub, fs, n_jobs=N_JOBS_NEURONS)
        lags, ac_sub = ss.autocorrelogram(calciumdata_sub, fs, max_lag_sec=MAX_LAG_SEC_AUTOCORR,
                                           n_jobs=N_JOBS_NEURONS)

        per_neuron = pd.DataFrame({
            "session_id": ses.session_id, "cell_id": celldata["cell_id"].to_numpy(),
            "roi_name": area, "labeled": label,
            "mean_activity_per_sec": rate, "active_frame_rate_hz": active_rate, "sparsity": spars,
            "fano_factor_magnitude": fano, "fano_factor_active_frames": active_fano,
            "runspeed_corr": runspeed_corr, "fchan2_corr": fchan2_corr,
            "noise_level": celldata["noise_level"].to_numpy(),
        })
        per_neuron = pd.concat([per_neuron, qc.reset_index(drop=True)], axis=1)

        iei_sub = pd.DataFrame({"session_id": ses.session_id, "roi_name": area_sub, "labeled": label_sub,
                                 "median_iei": median_iei, "cv_iei": cv_iei})

        session_summary = {"session_id": ses.session_id, "fchan2_artifact_frac": fchan2_artifact_frac}

        print(f"  [{ses.session_id}] fs={fs:.2f} Hz, {N} neurons, "
              f"{len(subsample_idx)} subsampled for IEI/autocorr")

        return {"per_neuron": per_neuron, "iei_sub": iei_sub, "lags": lags, "ac_sub": ac_sub,
                "area_sub": area_sub, "label_sub": label_sub, "session_summary": session_summary}

    print(f"Processing {len(sessions)} sessions (n_jobs={N_JOBS_SESSIONS}) ...")
    session_results = Parallel(n_jobs=N_JOBS_SESSIONS, backend="loky", verbose=JOBLIB_VERBOSE)(
        delayed(process_session)(ses, i) for i, ses in enumerate(sessions)
    )

    per_neuron = pd.concat([r["per_neuron"] for r in session_results], ignore_index=True)
    iei_sub_all = pd.concat([r["iei_sub"] for r in session_results], ignore_index=True)
    session_summary = pd.DataFrame([r["session_summary"] for r in session_results])

    reference_lags = session_results[0]["lags"]
    ac_rows = []
    for r in session_results:
        ac = r["ac_sub"]
        if len(r["lags"]) != len(reference_lags) or not np.allclose(r["lags"], reference_lags):
            ac = np.vstack([np.interp(reference_lags, r["lags"], row) for row in ac])
        for i in range(ac.shape[0]):
            ac_rows.append({"roi_name": r["area_sub"][i], "labeled": r["label_sub"][i],
                             **{f"lag_{j}": ac[i, j] for j in range(ac.shape[1])}})
    ac_df = pd.DataFrame(ac_rows)

    cached_data = {"per_neuron": per_neuron, "iei_sub": iei_sub_all, "ac": ac_df, "lags": reference_lags,
                   "session_summary": session_summary}
    pd.to_pickle(cached_data, combined_cache)

per_neuron = cached_data["per_neuron"]
iei_sub = cached_data["iei_sub"]
ac_df = cached_data["ac"]
reference_lags = cached_data["lags"]
session_summary = cached_data.get("session_summary", pd.DataFrame(columns=["session_id", "fchan2_artifact_frac"]))

n_sessions = per_neuron["session_id"].nunique()
print(f"\n{len(per_neuron)} neurons across {n_sessions} {PROTOCOL} sessions "
      f"(no exclusions applied -- full population)")


# --------------------------------------------------------------------- #
# 2. QC flagging -- computed and reported, never used to filter
#    anything in this script (see module docstring).
# --------------------------------------------------------------------- #
per_neuron_qc = qc_lib.flag_bad_cells(per_neuron, **QC_THRESHOLDS)
per_neuron_qc.to_csv(paths.out / "activity_per_neuron_summary.csv", index=False)

qc_summary = qc_lib.summarize_qc(per_neuron_qc)
qc_summary.to_csv(paths.out / "qc_summary_by_group.csv", index=False)
print(f"\nOverall: {int(per_neuron_qc['qc_pass'].sum())}/{len(per_neuron_qc)} neurons would pass "
      f"under the current QC_THRESHOLDS (informational only -- nothing is excluded by this script)")
print("\nWorst 10 (session, area, label) groups by pass fraction:")
print(qc_summary[["session_id", "roi_name", "labeled", "n_total", "n_pass", "frac_pass"]]
      .head(10).to_string(index=False))

qc_per_cell = per_neuron_qc[["session_id", "cell_id", "qc_pass", "qc_fail_reason"]
                             + [f"qc_flag_{r}" for r in qc_lib.REASON_COLS]]
qc_per_cell.to_csv(paths.out / "qc_per_cell.csv", index=False)


# --------------------------------------------------------------------- #
# 3. Figure 1: basic activity statistics (event rate, sparsity, IEI,
#    Fano factor), split by area/label -- count-based versions (the
#    ones directly comparable to "events/sec"/"~1 for Poisson"
#    intuitions; the magnitude-weighted versions are in the CSV but not
#    plotted here for that reason, see spike_stats.py's docstrings).
# --------------------------------------------------------------------- #
active_rate_by_group = {(a, l): sub["active_frame_rate_hz"].to_numpy()
                         for (a, l), sub in per_neuron.groupby(["roi_name", "labeled"])}
sparsity_by_group = {(a, l): sub["sparsity"].to_numpy()
                      for (a, l), sub in per_neuron.groupby(["roi_name", "labeled"])}
active_fano_by_group = {(a, l): sub["fano_factor_active_frames"].to_numpy()
                         for (a, l), sub in per_neuron.groupby(["roi_name", "labeled"])}
median_iei_by_group = {(a, l): sub["median_iei"].dropna().to_numpy()
                        for (a, l), sub in iei_sub.groupby(["roi_name", "labeled"])}

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
bar_by_group(axes[0, 0], active_rate_by_group, ylabel="active-frame rate (events/s)")
bar_by_group(axes[0, 1], sparsity_by_group, ylabel="sparsity (fraction of frames active)")
bar_by_group(axes[1, 0], median_iei_by_group, ylabel="median inter-event interval (s)", log_y=True)
bar_by_group(axes[1, 1], active_fano_by_group,
             ylabel=f"Fano factor of active-frame counts ({FANO_WINDOW_SEC:.0f}s windows)", zero_line=False)
axes[1, 1].axhline(1.0, color="k", linestyle=":", linewidth=1)
fig.suptitle(f"Basic activity statistics -- {PROTOCOL}, {n_sessions} sessions "
             "(full population, no exclusions)")
fig.tight_layout()
save_fig(fig, figdir, "1_basic_activity_stats")


# --------------------------------------------------------------------- #
# 4. Figure 2: autocorrelogram, mean +/- SEM per group
# --------------------------------------------------------------------- #
lag_cols = [c for c in ac_df.columns if c.startswith("lag_")]
fig, ax = plt.subplots(figsize=(7, 5))
for a, l in ordered_groups(sorted(ac_df["roi_name"].unique())):
    sub = ac_df[(ac_df["roi_name"] == a) & (ac_df["labeled"] == l)]
    if sub.empty:
        continue
    vals = sub[lag_cols].to_numpy()
    mean_ac = np.nanmean(vals, axis=0)
    sem_ac = np.nanstd(vals, axis=0) / np.sqrt(max(vals.shape[0], 1))
    color, ls = AREA_COLORS.get(a, "gray"), LABEL_LINESTYLES.get(l, "-")
    ax.plot(reference_lags, mean_ac, color=color, linestyle=ls, label=f"{a} ({l}, n={vals.shape[0]})")
    ax.fill_between(reference_lags, mean_ac - sem_ac, mean_ac + sem_ac, color=color, alpha=0.15)
ax.axhline(0, color="gray", linewidth=0.5)
ax.set_xlabel("lag (s)")
ax.set_ylabel("autocorrelation")
ax.set_title(f"Deconvolved-trace autocorrelogram -- {PROTOCOL}")
ax.legend(fontsize=8)
fig.tight_layout()
save_fig(fig, figdir, "2_autocorrelogram")


# --------------------------------------------------------------------- #
# 5. Figure 3: QC metric distributions, split by (area, label) group AND
#    colored by pass/fail -- the "which thresholds should I use" figure.
#    Reference lines mark QC_THRESHOLDS.
# --------------------------------------------------------------------- #
qc_metrics = [("qc_rate", "activity rate (qc_rate)", QC_THRESHOLDS["rate_thr"]),
              ("noise_level", "noise_level", QC_THRESHOLDS["noise_thr"]),
              ("qc_skew", "trace skewness", QC_THRESHOLDS["skew_min"]),
              ("qc_fano", "Fano factor (count-based)", QC_THRESHOLDS["fano_thr"]),
              ("qc_nan_frac", "fraction NaN", QC_THRESHOLDS["nan_frac_thr"])]

groups = list(per_neuron_qc.groupby(["roi_name", "labeled"]))
groups.sort(key=lambda g: (ordered_groups(sorted(per_neuron_qc["roi_name"].unique())).index(g[0])
                            if g[0] in ordered_groups(sorted(per_neuron_qc["roi_name"].unique())) else 99))
n_groups, n_metrics = len(groups), len(qc_metrics)

fig, axes = plt.subplots(n_groups, n_metrics, figsize=(3.6 * n_metrics, 2.4 * n_groups), squeeze=False)
for i, (g, sub) in enumerate(groups):
    n_total, n_pass = len(sub), int(sub["qc_pass"].sum())
    glabel = f"{g[0]} / {g[1]}"
    for j, (metric, mlabel, thr) in enumerate(qc_metrics):
        ax = axes[i, j]
        all_vals = sub[metric].dropna().to_numpy()
        if len(all_vals) == 0:
            continue
        lo, hi = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        if lo == hi:
            lo, hi = lo - 1, hi + 1
        bins = np.linspace(lo, hi, 31)
        ax.hist(sub.loc[sub["qc_pass"], metric].dropna(), bins=bins, color="tab:green", alpha=0.6, label="pass")
        ax.hist(sub.loc[~sub["qc_pass"], metric].dropna(), bins=bins, color="tab:red", alpha=0.6, label="flagged")
        ax.set_xlim(lo, hi)
        if lo <= thr <= hi:
            ax.axvline(thr, color="black", linestyle="--", linewidth=1)
        else:
            # Threshold falls outside this group's observed range for this
            # metric -- drawing it would force the axis to stretch out and
            # squash the actual data into an invisible sliver (this metric
            # is what originally happened here). Note it instead of hiding
            # it: an off-axis threshold means the CURRENT default doesn't
            # constrain this group's data at all, which is itself useful
            # information for picking a better one.
            ax.annotate(f"thr={thr:g}\n(off-axis)", xy=(0.97, 0.92), xycoords="axes fraction",
                        fontsize=5, ha="right", va="top", color="black")
        if j == 0:
            ax.set_ylabel(f"{glabel}\n(n={n_total}, pass={n_pass})", fontsize=7)
        if i == 0:
            ax.set_title(mlabel, fontsize=8)
        if i == n_groups - 1:
            ax.set_xlabel(mlabel, fontsize=7)
axes[0, 0].legend(fontsize=6, loc="upper right")
fig.suptitle(f"QC metric distributions by area/label (dashed = current threshold) -- {PROTOCOL}", y=1.01)
fig.tight_layout()
save_fig(fig, figdir, "3_qc_metrics_by_group")


# --------------------------------------------------------------------- #
# 6. Figure 4: QC pass fraction per session (sorted) -- flags bad
#    sessions at a glance.
# --------------------------------------------------------------------- #
frac_pass_session = per_neuron_qc.groupby("session_id")["qc_pass"].mean().sort_values()
fig, ax = plt.subplots(figsize=(6, max(3, 0.25 * len(frac_pass_session))))
ax.barh(np.arange(len(frac_pass_session)), frac_pass_session.to_numpy(), color="grey")
ax.set_yticks(np.arange(len(frac_pass_session)))
ax.set_yticklabels(frac_pass_session.index, fontsize=6)
ax.set_xlabel("fraction of neurons passing QC")
ax.set_xlim(0, 1.02)
ax.set_title(f"QC pass rate per session -- {PROTOCOL}")
fig.tight_layout()
save_fig(fig, figdir, "4_qc_pass_rate_per_session")


# --------------------------------------------------------------------- #
# 7. Figure 5: QC failure reasons, stacked by area/label
# --------------------------------------------------------------------- #
grouped = per_neuron_qc.groupby(["roi_name", "labeled"])[[f"qc_flag_{r}" for r in qc_lib.REASON_COLS]].sum()
group_order = [g for g in ordered_groups(sorted(per_neuron_qc["roi_name"].unique())) if g in grouped.index]
grouped = grouped.loc[group_order]
group_labels = [f"{a}\n{l}" for a, l in group_order]

fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(grouped)), 5))
x = np.arange(len(grouped))
bottom = np.zeros(len(grouped))
for r in qc_lib.REASON_COLS:
    vals = grouped[f"qc_flag_{r}"].to_numpy()
    ax.bar(x, vals, bottom=bottom, color=qc_lib.REASON_COLORS[r], label=qc_lib.REASON_LABELS[r])
    bottom += vals
ax.set_xticks(x)
ax.set_xticklabels(group_labels, fontsize=8)
ax.set_ylabel("number of neurons flagged")
ax.set_title(f"QC failure reasons by area/label -- {PROTOCOL}")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
save_fig(fig, figdir, "5_qc_failure_reasons")


# --------------------------------------------------------------------- #
# 8. Figure 6: running-speed correlation -- movement-artifact /
#    locomotion-modulation check. Uses real per-frame timestamps
#    (ses.ts_F) when available, else an evenly-spaced approximation
#    (see process_session above).
# --------------------------------------------------------------------- #
runspeed_by_group = {(a, l): sub["runspeed_corr"].dropna().to_numpy()
                      for (a, l), sub in per_neuron.groupby(["roi_name", "labeled"])
                      if sub["runspeed_corr"].notna().any()}
if runspeed_by_group:
    fig, ax = plt.subplots(figsize=(7, 5))
    bar_by_group(ax, runspeed_by_group, ylabel="correlation with running speed (r)")
    ax.set_title(f"Locomotion-related activity -- {PROTOCOL}")
    fig.tight_layout()
    save_fig(fig, figdir, "6_runspeed_correlation")
else:
    print("\nNo behaviordata/runspeed available -- skipped Figure 6.")


# --------------------------------------------------------------------- #
# 9. Fchan2 (red-channel / motion-artifact) battery. Fchan2 is a
#    SESSION-WIDE signal, not per-cell (see session.py's module
#    docstring: a zscored, absolute red-channel fluorescence CHANGE
#    trace -- tdTomato is a static structural marker, so abrupt changes
#    flag likely z-motion/refocusing events, not real activity). Three
#    angles: what the raw signal looks like and how much of it crosses
#    the artifact threshold, whether some sessions have more of these
#    events than others, and whether any cell's activity suspiciously
#    tracks this session-wide artifact signal (the same "movement
#    artifact" logic as the runspeed check above, applied to a
#    different reference trace).
# --------------------------------------------------------------------- #
example_session_id = per_neuron.loc[per_neuron["fchan2_corr"].notna(), "session_id"]
example_session_id = example_session_id.iloc[0] if len(example_session_id) else None

if example_session_id is not None:
    example_ses = load_sessions(protocols=[PROTOCOL], load_behaviordata=False, load_videodata=False,
                                 load_celldata=False, load_calciumdata=True,
                                 only_session_ids=[example_session_id], verbose=False)[0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if example_ses.ts_F is not None:
        axes[0].plot(example_ses.ts_F, example_ses.fchan2, color="firebrick", linewidth=0.5)
        axes[0].set_xlabel("session time (s)")
    else:
        axes[0].plot(example_ses.fchan2, color="firebrick", linewidth=0.5)
        axes[0].set_xlabel("frame")
    axes[0].axhline(FCHAN2_ARTIFACT_Z, color="black", linestyle="--", linewidth=1)
    axes[0].axhline(-FCHAN2_ARTIFACT_Z, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Fchan2 (z-scored)")
    axes[0].set_title(f"Example session: {example_session_id}")

    axes[1].hist(example_ses.fchan2, bins=60, color="firebrick", alpha=0.7)
    axes[1].axvline(FCHAN2_ARTIFACT_Z, color="black", linestyle="--", linewidth=1,
                     label=f"|z| > {FCHAN2_ARTIFACT_Z:g}")
    axes[1].axvline(-FCHAN2_ARTIFACT_Z, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Fchan2 (z-scored)")
    axes[1].set_ylabel("frames")
    axes[1].legend(fontsize=8, frameon=False)
    axes[1].set_title("Distribution (example session)")

    fig.suptitle(f"Fchan2 (red-channel motion-artifact signal) -- {PROTOCOL}", y=1.03)
    fig.tight_layout()
    save_fig(fig, figdir, "7_fchan2_example")

if session_summary["fchan2_artifact_frac"].notna().any():
    sess_frac = session_summary.dropna(subset=["fchan2_artifact_frac"]).sort_values("fchan2_artifact_frac")
    fig, ax = plt.subplots(figsize=(6, max(3, 0.25 * len(sess_frac))))
    ax.barh(np.arange(len(sess_frac)), sess_frac["fchan2_artifact_frac"].to_numpy(), color="firebrick")
    ax.set_yticks(np.arange(len(sess_frac)))
    ax.set_yticklabels(sess_frac["session_id"], fontsize=6)
    ax.set_xlabel(f"fraction of frames with |Fchan2| > {FCHAN2_ARTIFACT_Z:g}")
    ax.set_title(f"Motion-artifact frame rate per session -- {PROTOCOL}")
    fig.tight_layout()
    save_fig(fig, figdir, "8_fchan2_artifact_rate_per_session")

fchan2_corr_by_group = {(a, l): sub["fchan2_corr"].dropna().to_numpy()
                         for (a, l), sub in per_neuron.groupby(["roi_name", "labeled"])
                         if sub["fchan2_corr"].notna().any()}
if fchan2_corr_by_group:
    fig, ax = plt.subplots(figsize=(7, 5))
    bar_by_group(ax, fchan2_corr_by_group, ylabel="correlation with Fchan2 (r)")
    ax.set_title(f"Does activity track the session-wide motion-artifact signal? -- {PROTOCOL}")
    fig.tight_layout()
    save_fig(fig, figdir, "9_fchan2_correlation")
else:
    print("\nNo Fchan2 data available -- skipped Figures 7-9.")

print(f"\nWrote tables to {paths.out}, figures to {figdir}")
print("Run `python b_progress/make_progress_md.py 2b_activity_statistics` to build a markdown summary.")
