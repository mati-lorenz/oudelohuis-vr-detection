# -*- coding: utf-8 -*-
"""
4c_population_svm
====================
Population-level decoding within each (area, label) group: how
accurately does population activity predict the animal's CHOICE, using
a linear vs. an RBF-kernel SVM -- and how much of that accuracy
survives destroying correlated variability at a fixed stimulus, choice,
or both?

Primary decoding target: CHOICE (binary). A secondary, smaller analysis
decodes stimulus bin (multi-class) on the real data only, for the
linear-vs-RBF comparison specifically (see section 5) -- kept out of
the full shuffle-condition treatment to keep compute bounded (this
script already runs 2 kernels x several shuffle conditions x repeats
x cross-validation folds for choice decoding alone).

CONDITIONED SHUFFLES: identical framework to 4a/4b (see those scripts'
module docstrings and infotheory.population/shuffling) -- global/stim/
choice/stim_choice, applied to the NEURAL matrix only (shuffling within
a group never changes which trial has which choice/stimulus label,
only which neural activity pattern is paired with it). If choice-
decoding accuracy collapses under the "stim" shuffle (which preserves
each cell's own tuning to both stimulus AND choice, since choice-
tuning survives within-stimulus shuffling too -- only cross-cell
correlation is destroyed), that tells you decoding depended on genuine
correlated structure, not just each cell's own choice-related firing
rate difference.

LINEAR VS. RBF: if RBF meaningfully outperforms linear, the population
code has real nonlinear structure a hyperplane can't capture; if they're
close, a linear readout is already sufficient -- the same logic
2e_nonlinear_encod.py applied to single cells, here applied to the
whole population's joint code.

IMPORTANT CAVEAT (found while validating this script -- read before
trusting any RBF result): RBF-SVM accuracy is confounded by population
size in a way linear accuracy is NOT. Checked directly on this
project's own synthetic data (zero true neural-choice relationship by
construction): RBF accuracy correlated with population size (r=0.28)
while linear did not (r=-0.13), and the largest populations (84-136
cells, ~100-300 trials) showed RBF "accuracy" of 0.80-0.85 on the SAME
rows where linear sat at chance (~0.50). This is curse-of-dimensionality
overfitting -- with enough feature dimensions relative to trial count,
an unregularized-enough RBF kernel can carve out an almost arbitrary
partition of the training folds that happens to correlate with the
target, and default C=1.0/gamma='scale' don't prevent this at these
cell-to-trial ratios. It is NOT evidence of real nonlinear population
coding. This script prints each group's cells-to-trials ratio and flags
(HIGH_DIM_RATIO_WARN) any group where it's large enough that RBF
results are least trustworthy; treat RBF numbers for those groups, and
any RBF-vs-linear gap in general, with real skepticism until proper
nested-cross-validated hyperparameter tuning (not implemented here) is
run to rule this out. Linear SVM does not show this pattern and can be
read normally.

MODULARITY: same as 4a/4b -- condition specs come from
infotheory.population.build_standard_condition_specs.

Uses scikit-learn's SVC and StratifiedKFold directly.

Dataset, sessions/cells: same filters as 3a/3b/3c/4a/4b.

Outputs
-------
2_pipeline/4c_population_svm/
    out/    svm_shuffle_results.csv   one row per (session, window,
                                       area, label, kernel, condition,
                                       rep): cross-validated choice-
                                       decoding accuracy
            svm_stim_decoding.csv      one row per (session, window,
                                       area, label, kernel): real-data
                                       stimulus-bin decoding accuracy
            figures/*.png
    store/  svm.pkl   cached (reused unless --recompute)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score

from infotheory.pipeline import get_pipeline_paths
from infotheory.session import load_sessions
from infotheory.neural_encoding import load_cell_inclusion
from infotheory.celldata_utils import get_area_label, AREA_COLORS, LABEL_SPLIT_AREAS, DEFAULT_AREA_ORDER
from infotheory.population import (
    zscore_population, build_standard_condition_specs, run_across_shuffle_conditions,
    plot_shuffle_condition_comparison, plot_metric_vs_population_size, iterate_population_groups,
)
from infotheory.plotting import set_style, save_fig, clear_figures

PROTOCOL = "DN"
WINDOWS = {"pre_stim": (-30.0, -10.0), "stim": (0.0, 20.0), "reward": (25.0, 45.0)}

MIN_CELLS_PER_GROUP = 10
MIN_TRIALS_PER_CLASS = 8   # StratifiedKFold needs at least N_SPLITS members per class
N_SPLITS = 3
N_SHUFFLES = 8              # reduced relative to 4a/4b -- SVM cross-validation is far more
                             # expensive per call than a single PCA/FA fit
KERNELS = ("linear", "rbf")
RANDOM_STATE = 0

N_JOBS_SESSIONS = 13
JOBLIB_VERBOSE = 5

parser = argparse.ArgumentParser()
parser.add_argument("--recompute", action="store_true",
                     help="Reload sessions from 0_data/ instead of reusing the cached "
                          "store/svm.pkl")
args = parser.parse_args()

paths = get_pipeline_paths(__file__)
figdir = paths.out / "figures"
clear_figures(figdir)
set_style()


# --------------------------------------------------------------------- #
# 1. Included sessions/cells -- same filters as 3a/3b/3c/4a/4b
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
print(f"Using {len(included_ids)} included {PROTOCOL} sessions, "
      f"{int(cell_inclusion['include'].sum())}/{len(cell_inclusion)} cells included")


def decode_accuracy(matrix: np.ndarray, target: np.ndarray, kernel: str) -> float:
    """Cross-validated decoding accuracy, or NaN if any class has too
    few members for StratifiedKFold."""
    z = zscore_population(matrix)
    _, counts = np.unique(target, return_counts=True)
    if len(counts) < 2 or counts.min() < MIN_TRIALS_PER_CLASS:
        return np.nan
    clf = SVC(kernel=kernel, C=1.0)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    try:
        scores = cross_val_score(clf, z, target, cv=cv)
    except ValueError:
        return np.nan
    return float(scores.mean())


def make_choice_analysis_fn(choice: np.ndarray, kernel: str):
    def analysis_fn(matrix):
        return {"accuracy": decode_accuracy(matrix, choice, kernel)}
    return analysis_fn


# --------------------------------------------------------------------- #
# 2. Run SVM choice-decoding (real + shuffle conditions), both kernels
#    -- or reuse the cache. Stim-bin decoding (real data only) is
#    computed alongside, cheaply, in the same pass.
# --------------------------------------------------------------------- #
svm_cache = paths.store / "svm.pkl"
REQUIRED_COLUMNS = ["session_id", "window", "roi_name", "labeled_grouped", "n_trials", "kernel",
                    "condition", "rep", "accuracy"]

shuffle_results = stim_decoding = None
if not args.recompute and svm_cache.exists():
    try:
        candidate = pd.read_pickle(svm_cache)
        missing = [c for c in REQUIRED_COLUMNS if c not in candidate["shuffle_results"].columns]
        if missing:
            print(f"Cached SVM results are missing columns {missing} -- recomputing from 0_data/ "
                  f"instead of using {svm_cache} (safe to delete that file to silence this check).")
        else:
            shuffle_results = candidate["shuffle_results"]
            stim_decoding = candidate["stim_decoding"]
            print(f"Reusing cached SVM results from {paths.store} (pass --recompute to reload 0_data/)")
    except Exception as e:
        print(f"Could not load cached SVM results ({e}); recomputing from 0_data/ ...")

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
        dfs = []
        for kernel in KERNELS:
            fn = make_choice_analysis_fn(choice, kernel)
            df = run_across_shuffle_conditions(matrix, specs, fn, n_shuffles=N_SHUFFLES, rng=rng_local)
            df.insert(0, "kernel", kernel)
            dfs.append(df)
        combined = pd.concat(dfs, ignore_index=True)
        combined.insert(0, "session_id", session_id)
        combined.insert(1, "window", window_name)
        combined.insert(2, "roi_name", area)
        combined.insert(3, "labeled_grouped", label)
        combined.insert(4, "n_cells", len(cell_ids))
        combined.insert(5, "n_trials", matrix.shape[0])

        stim_rows = []
        for kernel in KERNELS:
            acc = decode_accuracy(matrix, stim_bin, kernel)
            stim_rows.append({"session_id": session_id, "window": window_name, "roi_name": area,
                               "labeled_grouped": label, "n_cells": len(cell_ids), "kernel": kernel,
                               "accuracy": acc, "n_classes": len(np.unique(stim_bin))})
        return combined, stim_rows

    print(f"Running SVM decoding across shuffle conditions for {len(included_ids)} sessions "
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
    stim_decoding = pd.DataFrame([row for r in nested for row in r[1]])
    pd.to_pickle({"shuffle_results": shuffle_results, "stim_decoding": stim_decoding}, svm_cache)

shuffle_results.to_csv(paths.out / "svm_shuffle_results.csv", index=False)
stim_decoding.to_csv(paths.out / "svm_stim_decoding.csv", index=False)
print(f"\n{shuffle_results['session_id'].nunique()} sessions, {len(shuffle_results)} total rows "
      f"(choice decoding), {len(stim_decoding)} rows (stim decoding, real data only)")

# --------------------------------------------------------------------- #
# 2b. Curse-of-dimensionality diagnostic (see module docstring). Flags
#    groups where n_cells is large relative to n_trials -- exactly
#    where RBF accuracy was found to become untrustworthy on this
#    project's own synthetic (zero-true-signal) data.
# --------------------------------------------------------------------- #
HIGH_DIM_RATIO_WARN = 0.15  # n_cells / n_trials above this -- flagged as high overfitting risk
real_rows = shuffle_results[shuffle_results["condition"] == "real"].dropna(subset=["accuracy"])
if len(real_rows):
    for kernel in KERNELS:
        ksub = real_rows[real_rows["kernel"] == kernel]
        if ksub["n_cells"].nunique() > 2 and ksub["accuracy"].notna().sum() > 3:
            corr = np.corrcoef(ksub["n_cells"], ksub["accuracy"])[0, 1]
            flag = " <-- check for dimensionality-driven inflation, see module docstring" \
                if kernel == "rbf" and corr > 0.15 else ""
            print(f"corr(n_cells, real accuracy), kernel={kernel}: {corr:+.3f}{flag}")

    group_cols = ["session_id", "window", "roi_name", "labeled_grouped", "n_cells", "n_trials"]
    ratio_table = real_rows[group_cols].drop_duplicates().copy()
    ratio_table["cells_to_trials_ratio"] = ratio_table["n_cells"] / ratio_table["n_trials"]
    ratio_table.to_csv(paths.out / "svm_dimensionality_risk.csv", index=False)
    n_flagged = int((ratio_table["cells_to_trials_ratio"] > HIGH_DIM_RATIO_WARN).sum())
    print(f"\n{n_flagged}/{len(ratio_table)} groups have cells:trials ratio > {HIGH_DIM_RATIO_WARN} "
          f"(saved in svm_dimensionality_risk.csv) -- treat RBF results for these with real "
          f"skepticism, per this script's module docstring.")



# --------------------------------------------------------------------- #
# 3. Summary
# --------------------------------------------------------------------- #
if len(shuffle_results):
    print("\nMean choice-decoding accuracy by (window, area, kernel, condition):")
    print(shuffle_results.dropna(subset=["accuracy"])
          .groupby(["window", "roi_name", "kernel", "condition"])["accuracy"]
          .mean().round(3).to_string())


# --------------------------------------------------------------------- #
# 4. Figures
# --------------------------------------------------------------------- #
if len(shuffle_results):
    # 4a. COMMON PLOT 1: shuffle condition comparison, per kernel
    for window_name in WINDOWS:
        for kernel in KERNELS:
            sub = shuffle_results[(shuffle_results["window"] == window_name)
                                   & (shuffle_results["kernel"] == kernel)].dropna(subset=["accuracy"])
            if sub.empty:
                continue
            areas = [a for a in DEFAULT_AREA_ORDER if a in sub["roi_name"].unique()]
            fig, axes = plt.subplots(1, len(areas), figsize=(4.5 * len(areas), 4.5), squeeze=False)
            for ax, area in zip(axes[0], areas):
                asub = sub[sub["roi_name"] == area]
                plot_shuffle_condition_comparison(ax, asub, "accuracy", "choice-decoding accuracy")
                ax.axhline(0.5, color="grey", linestyle=":", linewidth=1)
                ax.set_title(area)
            fig.suptitle(f"SVM ({kernel}): choice accuracy across shuffle conditions -- "
                         f"{window_name}, {PROTOCOL}", y=1.03)
            fig.tight_layout()
            save_fig(fig, figdir, f"1_shuffle_comparison_{kernel}_{window_name}")

    # 4b. COMMON PLOT 2: metric vs population size
    real_acc = shuffle_results[shuffle_results["condition"] == "real"].dropna(subset=["accuracy"])
    for kernel in KERNELS:
        fig, axes = plt.subplots(1, len(WINDOWS), figsize=(5 * len(WINDOWS), 4.5), squeeze=False)
        for ax, window_name in zip(axes[0], WINDOWS):
            sub = real_acc[(real_acc["window"] == window_name) & (real_acc["kernel"] == kernel)]
            if sub.empty:
                continue
            plot_metric_vs_population_size(ax, sub, "n_cells", "accuracy", "roi_name",
                                            "choice accuracy (real data)", AREA_COLORS)
            ax.axhline(0.5, color="grey", linestyle=":", linewidth=1)
            ax.set_title(window_name)
        fig.suptitle(f"SVM ({kernel}): accuracy vs. population size -- {PROTOCOL}", y=1.03)
        fig.tight_layout()
        save_fig(fig, figdir, f"2_accuracy_vs_size_{kernel}")

    # 4c. UNIQUE TO SVM: linear vs RBF, per pair (choice decoding, real data)
    pivot = real_acc.pivot_table(index=["session_id", "window", "roi_name", "labeled_grouped"],
                                  columns="kernel", values="accuracy").dropna()
    if not pivot.empty and set(KERNELS).issubset(pivot.columns):
        fig, ax = plt.subplots(figsize=(5, 5))
        for area in [a for a in DEFAULT_AREA_ORDER if a in pivot.index.get_level_values("roi_name")]:
            asub = pivot.xs(area, level="roi_name")
            ax.scatter(asub["linear"], asub["rbf"], s=20, alpha=0.6, color=AREA_COLORS.get(area, "gray"),
                       label=area)
        lims = [0.4, 1.0]
        ax.plot(lims, lims, color="grey", linestyle="--", linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("linear SVM accuracy")
        ax.set_ylabel("RBF SVM accuracy")
        ax.set_title(f"Does a nonlinear boundary help? -- {PROTOCOL}")
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        save_fig(fig, figdir, "3_linear_vs_rbf")

# 4d. UNIQUE TO SVM: stimulus-bin decoding, linear vs RBF (real data only)
if len(stim_decoding):
    stim_pivot = stim_decoding.dropna(subset=["accuracy"]).pivot_table(
        index=["session_id", "window", "roi_name"], columns="kernel", values="accuracy")
    if not stim_pivot.empty and set(KERNELS).issubset(stim_pivot.columns):
        fig, ax = plt.subplots(figsize=(5, 5))
        for area in [a for a in DEFAULT_AREA_ORDER if a in stim_pivot.index.get_level_values("roi_name")]:
            asub = stim_pivot.xs(area, level="roi_name")
            ax.scatter(asub["linear"], asub["rbf"], s=20, alpha=0.6, color=AREA_COLORS.get(area, "gray"),
                       label=area)
        chance = 1 / stim_decoding["n_classes"].mean()
        lims = [0, 1.0]
        ax.plot(lims, lims, color="grey", linestyle="--", linewidth=1)
        ax.axhline(chance, color="grey", linestyle=":", linewidth=1)
        ax.axvline(chance, color="grey", linestyle=":", linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("linear SVM accuracy")
        ax.set_ylabel("RBF SVM accuracy")
        ax.set_title(f"Stimulus-bin decoding, linear vs RBF -- {PROTOCOL}\n(dotted = chance level)")
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        save_fig(fig, figdir, "4_stim_decoding_linear_vs_rbf")

print(f"\nWrote tables to {paths.out}, figures to {figdir}")
print("Run `python b_progress/make_progress_md.py 4c_population_svm` to build a markdown summary.")
