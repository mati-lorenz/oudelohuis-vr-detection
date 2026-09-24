# -*- coding: utf-8 -*-
"""
4a_population_pca
====================
Population-level PCA within each (area, label) group: how many
effective dimensions does the population's trial-to-trial activity
occupy, and how much of that structure survives destroying correlated
variability at a fixed stimulus, a fixed choice, or both?

CONDITIONED SHUFFLES (see infotheory.shuffling /
infotheory.population.build_standard_condition_specs): for each
population, PCA is run on the REAL data once, and on N_SHUFFLES
realizations of each of:
    global      : every cell independently, fully randomized -- destroys
                  ALL trial-to-trial structure, including each cell's
                  own stimulus/choice tuning
    stim        : each cell independently shuffled WITHIN each stimulus
                  bin -- destroys cross-cell correlations, keeps each
                  cell's own stimulus tuning
    choice      : same, within choice
    stim_choice : same, within the joint (stimulus, choice)
If the population's low-dimensional structure survives the "stim"
shuffle (participation ratio barely changes), that structure is mostly
explained by shared TUNING, not genuine noise correlations; if it
collapses toward the "global" shuffle's dimensionality, the low-D
structure depends on real correlated trial-to-trial variability. This
is the same logic as 3a/3b/3c's pairwise conditioned-shuffle tests,
scaled up to the population.

MODULARITY: which conditions to shuffle within is entirely defined by
infotheory.population.build_standard_condition_specs (stim/choice/
stim_choice right now) -- to add a behavioral variable later, build an
equivalent dict with one more named entry and pass it in; nothing else
in this script, or in infotheory.population/shuffling, needs to change.

Uses scikit-learn's PCA directly (this is exactly the kind of
well-established numerical method not worth reimplementing).

Dataset, sessions/cells: same filters as 3a/3b/3c (1b's included
sessions, engaged trials, cells passing 2a's proximity filter AND 2b's
QC). PCA is run within one (area, label) population at a time (never
mixing areas).

Outputs
-------
2_pipeline/4a_population_pca/
    out/    pca_shuffle_results.csv   one row per (session, window, area,
                                       label, condition, rep): participation
                                       ratio, top-3 variance explained
            pca_real_summary.csv       one row per (session, window,
                                       area, label): population size,
                                       real (unshuffled) metrics, plus
                                       full per-PC variance (as a
                                       comma-joined string -- for a later
                                       script to reload and re-plot
                                       without recomputing PCA)
            trajectory_scores.csv      PART 2 (trial-averaged PCA) --
                                       one row per (session, area, label,
                                       position bin): PC1..PCk scores
            trajectory_variance.csv    PART 2 -- one row per (session,
                                       area, label, PC): variance explained
            figures/*.png
    store/  pca.pkl   cached (reused unless --recompute; PART 2 does not
                       cache separately, see that section's own comments)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.decomposition import PCA

from infotheory.pipeline import get_pipeline_paths
from infotheory.session import load_sessions
from infotheory.neural_encoding import load_cell_inclusion
from infotheory.celldata_utils import get_area_label, AREA_COLORS, LABEL_SPLIT_AREAS, DEFAULT_AREA_ORDER, \
    SIX_GROUP_ORDER, SIX_GROUP_LABELS, SIX_GROUP_COLORS
from infotheory.population import (
    zscore_population, build_standard_condition_specs, run_across_shuffle_conditions,
    plot_shuffle_condition_comparison, plot_metric_vs_population_size, iterate_population_groups,
)
from infotheory.plotting import set_style, save_fig, clear_figures

PROTOCOL = "DN"
WINDOWS = {"pre_stim": (-30.0, -10.0), "stim": (0.0, 20.0), "reward": (25.0, 45.0)}

MIN_CELLS_PER_GROUP = 10
N_SHUFFLES = 20
RANDOM_STATE = 0

N_JOBS_SESSIONS = -1
JOBLIB_VERBOSE = 5

parser = argparse.ArgumentParser()
parser.add_argument("--recompute", action="store_true",
                     help="Reload sessions from 0_data/ instead of reusing the cached "
                          "store/pca.pkl")
args = parser.parse_args()

paths = get_pipeline_paths(__file__)
figdir = paths.out / "figures"
clear_figures(figdir)
set_style()


# --------------------------------------------------------------------- #
# 1. Included sessions/cells -- same filters as 3a/3b/3c
# --------------------------------------------------------------------- #
fits_path = paths.out_from("1b_psychometric") / "psychometric_fits.csv"
fits_df = pd.read_csv(fits_path)
included = fits_df[fits_df["included"] & (fits_df["protocol"] == PROTOCOL)]
included_ids = included["session_id"].tolist()
if not included_ids:
    raise SystemExit(f"No {PROTOCOL} sessions passed 1b_psychometric's inclusion criteria.")
mu_sigma_lookup = fits_df.set_index("session_id")[["mu", "sigma"]].to_dict("index")

cell_dist_out = paths.out_from("2a_cell_distribution")
activity_out = paths.out_from("2b_activity_statistics")
cell_inclusion = load_cell_inclusion(cell_dist_out, activity_out)
included_cell_ids_by_session = {
    sid: set(sub.loc[sub["include"], "cell_id"]) for sid, sub in cell_inclusion.groupby("session_id")
}
celldata_combined = pd.read_csv(cell_dist_out / "celldata_combined.csv")
print(f"Using {len(included_ids)} included {PROTOCOL} sessions, "
      f"{int(cell_inclusion['include'].sum())}/{len(cell_inclusion)} cells included")


def pca_metrics(matrix: np.ndarray) -> dict:
    """Analysis function for run_across_shuffle_conditions: z-score,
    fit PCA, return scalar summary metrics. Participation ratio
    (Gao & Ganguli-style effective dimensionality: (sum lambda)^2 /
    sum(lambda^2)) is the headline metric -- a single population with
    lambda concentrated in few PCs has a LOW participation ratio (very
    low-dimensional); one with evenly spread variance has a HIGH one
    (close to the number of cells)."""
    z = zscore_population(matrix)
    n_components = min(z.shape)
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pca.fit(z)
    eigenvalues = pca.explained_variance_
    pr = float((eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum())
    var_ratio = pca.explained_variance_ratio_
    return {"participation_ratio": pr, "var_pc1": float(var_ratio[0]),
            "var_top3": float(var_ratio[:min(3, len(var_ratio))].sum())}


# --------------------------------------------------------------------- #
# 2. Run PCA (real + shuffle conditions) for every (session, window,
#    area, label) population -- or reuse the cache
# --------------------------------------------------------------------- #
pca_cache = paths.store / "pca.pkl"
REQUIRED_COLUMNS = ["session_id", "window", "roi_name", "labeled_grouped", "condition", "rep",
                    "participation_ratio"]

shuffle_results = real_summary = None
if not args.recompute and pca_cache.exists():
    try:
        candidate = pd.read_pickle(pca_cache)
        missing = [c for c in REQUIRED_COLUMNS if c not in candidate["shuffle_results"].columns]
        if missing:
            print(f"Cached PCA results are missing columns {missing} -- recomputing from 0_data/ "
                  f"instead of using {pca_cache} (safe to delete that file to silence this check).")
        else:
            shuffle_results = candidate["shuffle_results"]
            real_summary = candidate["real_summary"]
            print(f"Reusing cached PCA results from {paths.store} (pass --recompute to reload 0_data/)")
    except Exception as e:
        print(f"Could not load cached PCA results ({e}); recomputing from 0_data/ ...")

if shuffle_results is None:
    sessions = load_sessions(protocols=[PROTOCOL], load_behaviordata=True, load_videodata=True,
                              load_celldata=True, load_calciumdata=True, only_session_ids=included_ids)

    area_label_by_session = {}
    for ses in sessions:
        included_cells = included_cell_ids_by_session.get(ses.session_id)
        if not included_cells:
            continue
        cd = ses.celldata[ses.celldata["cell_id"].isin(included_cells)].reset_index(drop=True)
        area, label, _ = get_area_label(cd, label_split_areas=LABEL_SPLIT_AREAS)
        area_label_by_session[ses.session_id] = dict(zip(cd["cell_id"], zip(area, label)))

    def process_one_group(session_id, window_name, area, label, matrix, stim_bin, choice, cell_ids, seed):
        rng_local = np.random.default_rng(seed)
        specs = build_standard_condition_specs(stim_bin, choice)
        df = run_across_shuffle_conditions(matrix, specs, pca_metrics, n_shuffles=N_SHUFFLES, rng=rng_local)
        df.insert(0, "session_id", session_id)
        df.insert(1, "window", window_name)
        df.insert(2, "roi_name", area)
        df.insert(3, "labeled_grouped", label)
        df.insert(4, "n_cells", len(cell_ids))

        z = zscore_population(matrix)
        pca = PCA(n_components=min(z.shape), random_state=seed)
        pca.fit(z)
        real_row = {"session_id": session_id, "window": window_name, "roi_name": area,
                    "labeled_grouped": label, "n_cells": len(cell_ids), "n_trials": matrix.shape[0],
                    "var_ratio_per_pc": ",".join(f"{v:.6f}" for v in pca.explained_variance_ratio_)}
        return df, real_row

    print(f"Running PCA across shuffle conditions for {len(included_ids)} sessions "
          f"(n_jobs={N_JOBS_SESSIONS}) ...")
    groups = list(iterate_population_groups(sessions, mu_sigma_lookup, WINDOWS,
                                              included_cell_ids_by_session, area_label_by_session,
                                              min_cells=MIN_CELLS_PER_GROUP))
    print(f"{len(groups)} (session, window, area, label) populations with >={MIN_CELLS_PER_GROUP} cells")
    nested = Parallel(n_jobs=N_JOBS_SESSIONS, backend="loky", verbose=JOBLIB_VERBOSE)(
        delayed(process_one_group)(sid, wname, area, label, matrix, stim_bin, choice, cell_ids,
                                    RANDOM_STATE + i)
        for i, (sid, wname, area, label, matrix, stim_bin, choice, cell_ids) in enumerate(groups)
    )
    shuffle_results = pd.concat([r[0] for r in nested], ignore_index=True)
    real_summary = pd.DataFrame([r[1] for r in nested])
    pd.to_pickle({"shuffle_results": shuffle_results, "real_summary": real_summary}, pca_cache)

shuffle_results.to_csv(paths.out / "pca_shuffle_results.csv", index=False)
real_summary.to_csv(paths.out / "pca_real_summary.csv", index=False)
print(f"\n{len(real_summary)} populations analyzed, {len(shuffle_results)} total (condition, rep) rows")


# --------------------------------------------------------------------- #
# 3. Summary
# --------------------------------------------------------------------- #
if len(shuffle_results):
    print("\nMean participation ratio by (window, area, condition):")
    print(shuffle_results.groupby(["window", "roi_name", "condition"])["participation_ratio"]
          .mean().round(2).to_string())


# --------------------------------------------------------------------- #
# 4. Figures
# --------------------------------------------------------------------- #
if len(shuffle_results):
    # 4a. COMMON PLOT 1: shuffle condition comparison (participation ratio)
    # -- ALWAYS the six canonical groups (V1-lab, V1-unl, PM-lab, PM-unl,
    # AL, RSP), one panel each, never collapsed to just 4 areas (which
    # would silently show only whichever of lab/unl for V1/PM happened to
    # be picked up first).
    for window_name in WINDOWS:
        sub = shuffle_results[shuffle_results["window"] == window_name]
        if sub.empty:
            continue
        fig, axes = plt.subplots(1, len(SIX_GROUP_ORDER), figsize=(4.5 * len(SIX_GROUP_ORDER), 4.5),
                                  squeeze=False)
        for ax, (area, label) in zip(axes[0], SIX_GROUP_ORDER):
            gsub = sub[(sub["roi_name"] == area) & (sub["labeled_grouped"] == label)]
            if gsub.empty:
                ax.set_title(f"{SIX_GROUP_LABELS[(area, label)]} (no data)", fontsize=9)
                continue
            plot_shuffle_condition_comparison(ax, gsub, "participation_ratio",
                                               "participation ratio (effective dimensionality)")
            ax.set_title(SIX_GROUP_LABELS[(area, label)])
        fig.suptitle(f"PCA: participation ratio across shuffle conditions -- {window_name}, {PROTOCOL}",
                     y=1.03)
        fig.tight_layout()
        save_fig(fig, figdir, f"1_shuffle_comparison_{window_name}")

    # 4b. COMMON PLOT 2: metric vs population size (real data) -- one
    # combined scatter per window, colored by the six groups (not just area)
    real_pr = shuffle_results[shuffle_results["condition"] == "real"].copy()
    real_pr["group_label"] = [SIX_GROUP_LABELS.get((a, l), f"{a}-{l}")
                               for a, l in zip(real_pr["roi_name"], real_pr["labeled_grouped"])]
    group_colors_by_label = {SIX_GROUP_LABELS[g]: SIX_GROUP_COLORS[g] for g in SIX_GROUP_ORDER}
    fig, axes = plt.subplots(1, len(WINDOWS), figsize=(5 * len(WINDOWS), 4.5), squeeze=False)
    for ax, window_name in zip(axes[0], WINDOWS):
        sub = real_pr[real_pr["window"] == window_name]
        if sub.empty:
            continue
        plot_metric_vs_population_size(ax, sub, "n_cells", "participation_ratio", "group_label",
                                        "participation ratio (real data)", group_colors_by_label)
        ax.set_title(window_name)
    fig.suptitle(f"PCA: dimensionality vs. population size -- {PROTOCOL}", y=1.03)
    fig.tight_layout()
    save_fig(fig, figdir, "2_dimensionality_vs_size")

    # 4c. UNIQUE TO PCA: scree plots, real vs conditions -- ALWAYS the six
    # canonical groups, one example population per group
    for window_name in ["stim"]:
        sub_real = real_summary[real_summary["window"] == window_name]
        if sub_real.empty:
            continue
        fig, axes = plt.subplots(1, len(SIX_GROUP_ORDER), figsize=(4.5 * len(SIX_GROUP_ORDER), 4.5),
                                  squeeze=False)
        for ax, (area, label) in zip(axes[0], SIX_GROUP_ORDER):
            gsub = sub_real[(sub_real["roi_name"] == area) & (sub_real["labeled_grouped"] == label)]
            group_label = SIX_GROUP_LABELS[(area, label)]
            if gsub.empty:
                ax.set_title(f"{group_label} (no data)", fontsize=9)
                continue
            example = gsub.iloc[0]
            var_ratio = np.array([float(v) for v in example["var_ratio_per_pc"].split(",")])
            ax.plot(range(1, len(var_ratio) + 1), np.cumsum(var_ratio), marker="o", markersize=3,
                    color=SIX_GROUP_COLORS[(area, label)])
            ax.set_xlabel("PC")
            ax.set_ylabel("cumulative variance explained")
            ax.set_title(f"{group_label} (example: {example['session_id']}, n={example['n_cells']} cells)",
                         fontsize=9)
            ax.set_ylim(0, 1.02)
        fig.suptitle(f"PCA scree (cumulative variance) -- {window_name}, {PROTOCOL}", y=1.03)
        fig.tight_layout()
        save_fig(fig, figdir, f"3_scree_example_{window_name}")

print(f"\nWrote tables to {paths.out}, figures to {figdir}")


# ======================================================================= #
# PART 2: PCA on TRIAL-AVERAGED activity -- a different question from
# everything above. Part 1 asks "how many dimensions does trial-to-trial
# VARIABILITY occupy, and how much is real correlation vs shared
# tuning" (hence the conditioned-shuffle framework -- shuffling trials
# only makes sense when trials are still distinct samples). Averaging
# across trials FIRST removes trial-to-trial variability on purpose --
# there's no "noise" left to shuffle -- and asks instead: "what does the
# population's AVERAGE response look like as a trajectory through
# activity space over the course of a trial, and how many dimensions
# does THAT occupy." Same technique (PCA), different input, different
# question, so no shuffle-condition test is run here.
#
# TENSOR: same position-binned-trial-tensor convention as 3d/3e/3f_
# pair_*_temporal.py (infotheory.psth.build_position_binned_trial_tensor,
# one value per trial per narrow position bin) -- but instead of keeping
# each trial as its own sample, this AVERAGES across trials at each bin,
# giving one (position bins, cells) matrix per (session, area, label)
# population. Fitting PCA on THAT -- treating each position bin as a
# sample -- is the standard "neural trajectory" analysis: the PC SCORES
# at each bin, plotted against position, show how far the population's
# average activity pattern has moved through state space by that point
# in the trial (PART 2B below is exactly this -- "how it changes across
# the whole trial duration/length").
#
# MEMORY: same per-session on-demand loading + explicit cleanup as
# 3d/3e/3f_pair_*_temporal.py (see 3d's module docstring) -- sessions
# load one at a time, N_JOBS_SESSIONS caps how many are in memory at
# once.
#
# CROSS-SESSION CAVEAT: each session has a different population of
# cells, and PCA's components have an arbitrary sign/rotation that
# isn't guaranteed to line up across independently-fit sessions -- so
# trajectories are shown PER SESSION (a small-multiples grid), not
# pooled/averaged into one trajectory per area. A shared low-D space
# across sessions (e.g. via CCA alignment) would be a reasonable next
# step but is out of scope here.
#
# Outputs (added to the same out/store dirs as PART 1 above):
#   trajectory_scores.csv     one row per (session, area, label,
#                              s_rel): PC1..PCk scores of the trial-
#                              averaged population state at that
#                              position bin
#   trajectory_variance.csv   one row per (session, area, label, PC):
#                              variance explained
# ======================================================================= #
from infotheory.session import Session, discover_sessions
from infotheory.psth import build_position_binned_trial_tensor
from infotheory.continuous import merge_behavior_video, restrict_to_engaged, remove_pupil_outliers, build_calcium_continuous

TRAJ_S_PRE, TRAJ_S_POST, TRAJ_BINSIZE = -60.0, 80.0, 10.0
TRAJ_N_COMPONENTS = 5
MIN_CELLS_FOR_TRAJECTORY = 10
MIN_TRIALS_PER_TRAJ_BIN = 15
SMOOTH_SIGMA_BINS = 1.0  # Gaussian smoothing along position, in units of BINS, applied to the
                          # scaled trial-averaged matrix before PCA -- see the fix comment
                          # below for why this and the raw-std scaling were both needed.
                          # Set to 0/None to disable and see the raw (jagged) trajectories.

session_lookup = {f"{animal_id}_{sessiondate}": (protocol, animal_id, sessiondate)
                   for protocol, animal_id, sessiondate in discover_sessions([PROTOCOL])
                   if f"{animal_id}_{sessiondate}" in included_ids}


def process_one_session_trajectory(session_id, seed):
    protocol, animal_id, sessiondate = session_lookup[session_id]
    included_cells = included_cell_ids_by_session.get(session_id)
    if not included_cells:
        return [], []

    ses = Session(protocol, animal_id, sessiondate)
    ses.load(load_behaviordata=True, load_videodata=True, load_celldata=True, load_calciumdata=True)

    celldata = ses.celldata[ses.celldata["cell_id"].isin(included_cells)].reset_index(drop=True)
    if celldata.empty:
        return [], []
    area, label, _ = get_area_label(celldata, label_split_areas=LABEL_SPLIT_AREAS)
    cell_ids = celldata["cell_id"].tolist()

    merged_full = merge_behavior_video(ses.behaviordata, ses.videodata)
    merged_full = remove_pupil_outliers(merged_full)
    engaged = restrict_to_engaged(merged_full, ses.trialdata)
    score_rows, var_rows = [], []
    if not engaged.empty:
        engaged_trials = ses.trialdata[ses.trialdata.get("engaged", 1) == 1] \
            if "engaged" in ses.trialdata.columns else ses.trialdata

        for (a, l) in sorted(set(zip(area, label))):
            group_cell_ids = [c for c, ga, gl in zip(cell_ids, area, label) if ga == a and gl == l]
            if len(group_cell_ids) < MIN_CELLS_FOR_TRAJECTORY:
                continue

            calcium_continuous = build_calcium_continuous(ses.calciumdata[group_cell_ids], ses.ts_F,
                                                            ses.behaviordata)
            tensor_wide = build_position_binned_trial_tensor(
                engaged_trials, calcium_continuous, cell_cols=group_cell_ids,
                s_pre=TRAJ_S_PRE, s_post=TRAJ_S_POST, binsize=TRAJ_BINSIZE)

            trial_avg = (tensor_wide.groupby("s_rel")[group_cell_ids]
                         .agg(lambda s: s.mean() if s.notna().sum() >= MIN_TRIALS_PER_TRAJ_BIN else np.nan)
                         .dropna())
            if len(trial_avg) < TRAJ_N_COMPONENTS + 1:
                continue

            # Center using the trial-averaged data's own per-bin mean, but SCALE using each
            # cell's std across ALL individual trials (pooled, from tensor_wide directly) --
            # not the std across just the handful of BIN AVERAGES (2-14 samples here, an
            # unstable estimate that a single noisy bin can dominate). This was the source of
            # the "scree looks too smooth" symptom: with the old bin-averaged-std scaling, a
            # group left with only 2-3 valid bins would show PC1 trivially explaining most of
            # the variance, since with that few samples almost any shared direction dominates
            # by chance -- confirmed directly on real V1/PM examples that had only 2-3 valid
            # bins after the MIN_TRIALS_PER_TRAJ_BIN filter. Scaling by the pooled-trial std
            # (hundreds of samples, not single digits) is a far more stable estimate of each
            # cell's actual noise scale, and correctly downweights cells whose trial-averaged
            # wiggle is mostly noise rather than real position tuning (a cell with small
            # trial-averaged variance AND large raw per-trial variance is weakly tuned and
            # should contribute little to PCA, not get amplified to unit variance).
            raw_std = tensor_wide[group_cell_ids].std(skipna=True)
            raw_std_safe = raw_std.replace(0, 1.0)
            centered = trial_avg - trial_avg.mean(axis=0)
            z = (centered / raw_std_safe).to_numpy()

            # Optional position-domain smoothing, applied AFTER scaling so every cell is
            # already on a comparable scale first -- reduces bin-to-bin sampling noise (each
            # bin's trial average still has residual variance ~ sigma^2/n_trials_in_that_bin,
            # which the scaling fix above does not remove) directly in the jagged PC-vs-
            # position trajectories. Assumes bins are contiguous (no gaps from dropped bins);
            # with large gaps this smooths across positions that aren't really adjacent --
            # acceptable for a first pass but worth checking bin counts (n_bins in
            # trajectory_variance.csv) before trusting heavily-smoothed groups.
            if SMOOTH_SIGMA_BINS and SMOOTH_SIGMA_BINS > 0 and z.shape[0] > 2:
                from scipy.ndimage import gaussian_filter1d
                z = gaussian_filter1d(z, sigma=SMOOTH_SIGMA_BINS, axis=0, mode="nearest")

            n_components = min(TRAJ_N_COMPONENTS, z.shape[0], z.shape[1])
            pca = PCA(n_components=n_components, random_state=seed)
            scores = pca.fit_transform(z)

            for s_rel, score_row in zip(trial_avg.index, scores):
                score_rows.append({"session_id": session_id, "roi_name": a, "labeled_grouped": l,
                                    "s_rel": s_rel,
                                    **{f"PC{k + 1}": score_row[k] for k in range(n_components)}})
            for k in range(n_components):
                var_rows.append({"session_id": session_id, "roi_name": a, "labeled_grouped": l,
                                  "PC": k + 1, "var_ratio": float(pca.explained_variance_ratio_[k]),
                                  "n_cells": len(group_cell_ids), "n_bins": len(trial_avg)})

    for attr in ("calciumdata", "videodata", "behaviordata"):
        if hasattr(ses, attr):
            delattr(ses, attr)
    print(f"  [{session_id}] trial-averaged trajectory done ({len(score_rows)} bin-rows)")
    return score_rows, var_rows


print(f"\nComputing trial-averaged PCA trajectories for {len(included_ids)} sessions "
      f"(n_jobs={N_JOBS_SESSIONS}) ...")
traj_nested = Parallel(n_jobs=N_JOBS_SESSIONS, backend="loky", verbose=JOBLIB_VERBOSE)(
    delayed(process_one_session_trajectory)(sid, RANDOM_STATE + i) for i, sid in enumerate(included_ids)
)
trajectory_scores = pd.DataFrame([row for scores, _ in traj_nested for row in scores])
trajectory_variance = pd.DataFrame([row for _, var in traj_nested for row in var])
trajectory_scores.to_csv(paths.out / "trajectory_scores.csv", index=False)
trajectory_variance.to_csv(paths.out / "trajectory_variance.csv", index=False)
print(f"{len(trajectory_scores)} (session, area, label, position bin) trajectory rows, "
      f"{trajectory_scores.groupby(['session_id', 'roi_name', 'labeled_grouped']).ngroups if len(trajectory_scores) else 0} "
      f"populations")

if len(trajectory_scores):
    # 5a. Variance explained (trial-averaged PCA) -- ALWAYS the six
    #     canonical groups, one example (session) per group
    fig, axes = plt.subplots(1, len(SIX_GROUP_ORDER), figsize=(4 * len(SIX_GROUP_ORDER), 4), squeeze=False)
    for ax, (area, label) in zip(axes[0], SIX_GROUP_ORDER):
        group_label = SIX_GROUP_LABELS[(area, label)]
        gsub = trajectory_variance[(trajectory_variance["roi_name"] == area)
                                    & (trajectory_variance["labeled_grouped"] == label)]
        if gsub.empty:
            ax.set_title(f"{group_label} (no data)", fontsize=9)
            continue
        example_sid = gsub["session_id"].iloc[0]
        ex = gsub[gsub["session_id"] == example_sid].sort_values("PC")
        ax.plot(ex["PC"], np.cumsum(ex["var_ratio"]), marker="o", markersize=4, color=SIX_GROUP_COLORS[(area, label)])
        ax.set_xlabel("PC")
        ax.set_ylabel("cumulative variance explained")
        ax.set_ylim(0, 1.02)
        ax.set_title(f"{group_label} (example: {example_sid}, n={int(ex['n_cells'].iloc[0])} cells, "
                     f"{int(ex['n_bins'].iloc[0])} bins)", fontsize=8)
    fig.suptitle(f"Trial-averaged PCA: variance explained -- {PROTOCOL}", y=1.08)
    fig.tight_layout()
    save_fig(fig, figdir, "4_trajectory_variance_explained")

    # 5b. PC scores vs position -- "how it changes across the whole trial
    #     duration/length" -- ALWAYS the six canonical groups
    fig, axes = plt.subplots(1, len(SIX_GROUP_ORDER), figsize=(4.5 * len(SIX_GROUP_ORDER), 4.5), squeeze=False)
    for ax, (area, label) in zip(axes[0], SIX_GROUP_ORDER):
        group_label = SIX_GROUP_LABELS[(area, label)]
        gsub = trajectory_scores[(trajectory_scores["roi_name"] == area)
                                  & (trajectory_scores["labeled_grouped"] == label)]
        if gsub.empty:
            ax.set_title(f"{group_label} (no data)", fontsize=9)
            continue
        example_sid = gsub["session_id"].iloc[0]
        ex = gsub[gsub["session_id"] == example_sid].sort_values("s_rel")
        pc_cols = [c for c in ex.columns if c.startswith("PC")]
        for pc in pc_cols[:3]:
            ax.plot(ex["s_rel"], ex[pc], marker="o", markersize=3, label=pc)
        ax.axvline(0, color="black", linestyle=":", linewidth=1)
        ax.axhline(0, color="grey", linewidth=0.7)
        ax.set_xlabel("position rel. to stimulus onset (cm)")
        ax.set_ylabel("PC score (trial-averaged)")
        ax.set_title(f"{group_label} (example: {example_sid})", fontsize=9)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"Trial-averaged population trajectory, PC scores vs position -- {PROTOCOL}", y=1.03)
    fig.tight_layout()
    save_fig(fig, figdir, "5_trajectory_pc_vs_position")

    # 5c. PC1 vs PC2, 2D trajectory colored by position -- ALWAYS the six
    #     canonical groups
    fig, axes = plt.subplots(1, len(SIX_GROUP_ORDER), figsize=(4.5 * len(SIX_GROUP_ORDER), 4.5), squeeze=False)
    for ax, (area, label) in zip(axes[0], SIX_GROUP_ORDER):
        group_label = SIX_GROUP_LABELS[(area, label)]
        gsub = trajectory_scores[(trajectory_scores["roi_name"] == area)
                                  & (trajectory_scores["labeled_grouped"] == label)]
        if gsub.empty or "PC1" not in gsub.columns or "PC2" not in gsub.columns:
            ax.set_title(f"{group_label} (no data)", fontsize=9)
            continue
        example_sid = gsub["session_id"].iloc[0]
        ex = gsub[gsub["session_id"] == example_sid].sort_values("s_rel")
        sc = ax.scatter(ex["PC1"], ex["PC2"], c=ex["s_rel"], cmap="viridis", s=40, zorder=2)
        ax.plot(ex["PC1"], ex["PC2"], color="grey", alpha=0.5, linewidth=1, zorder=1)
        fig.colorbar(sc, ax=ax, label="position (cm)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"{group_label} (example: {example_sid})", fontsize=9)
    fig.suptitle(f"Trial-averaged population trajectory in PC space -- {PROTOCOL}", y=1.03)
    fig.tight_layout()
    save_fig(fig, figdir, "6_trajectory_pc1_pc2")

print(f"\nWrote trial-averaged trajectory tables to {paths.out}, figures to {figdir}")
print("Run `python b_progress/make_progress_md.py 4a_population_pca` to build a markdown summary.")
