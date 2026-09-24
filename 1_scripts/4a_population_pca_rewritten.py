# -*- coding: utf-8 -*-
"""
4a_population_pca.py
====================

Population-level PCA within each (area, label) group.

PCA is performed after z-scoring each cell across trials. Dimensionality
is summarized by participation ratio (PR) and by the number of PCs needed
for 50/80/90% cumulative variance. The normalized PCA spectrum is also
stored for real and shuffle populations.

Shuffle conditions:
    real        : original data
    global      : shuffle each cell across all trials
    stim        : shuffle each cell within stimulus
    choice      : shuffle each cell within choice
    stim_choice : shuffle each cell within stimulus + choice

Figures:
    1_shuffle_comparison_<window>.png
        PR across shuffle conditions.
    2_dimensionality_vs_size.png
        PR and rank-normalized PR versus population size.
    3_pca_spectrum_<window>.png
        Representative population per area: scree + cumulative variance,
        real versus conditioned shuffles.
    4_pca_shuffle_effects_<window>.png
        Population-level paired effects of shuffling, after collapsing
        the 20 shuffle realizations within each population.
    5_pca_spectrum_population_<window>.png
        Aggregate normalized PCA spectrum across populations.
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
from infotheory.celldata_utils import (
    get_area_label,
    AREA_COLORS,
    LABEL_SPLIT_AREAS,
    DEFAULT_AREA_ORDER,
)
from infotheory.population import (
    zscore_population,
    build_standard_condition_specs,
    run_across_shuffle_conditions,
    plot_shuffle_condition_comparison,
    iterate_population_groups,
)
from infotheory.plotting import set_style, save_fig, clear_figures


PROTOCOL = "DN"
WINDOWS = {
    "pre_stim": (-30.0, -10.0),
    "stim": (0.0, 20.0),
    "reward": (25.0, 45.0),
    "whole_trial": (-80.0, 60.0),
}

MIN_CELLS_PER_GROUP = 10
N_SHUFFLES = 20
RANDOM_STATE = 0

N_JOBS_SESSIONS = 12
JOBLIB_VERBOSE = 5

# Plotting controls.
SCREE_N_PC = 50
CUMULATIVE_N_PC = 50
AGGREGATE_N_PC = 50
CI_LOW = 2.5
CI_HIGH = 97.5

CONDITION_ORDER = ("real", "global", "stim", "choice", "stim_choice")
SHUFFLE_CONDITIONS = ("global", "stim", "choice", "stim_choice")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--recompute",
    action="store_true",
    help="Reload sessions from 0_data/ instead of reusing store/pca.pkl",
)
args = parser.parse_args()

paths = get_pipeline_paths(__file__)
figdir = paths.out / "figures"
clear_figures(figdir)
set_style()


# --------------------------------------------------------------------- #
# 1. Included sessions/cells
# --------------------------------------------------------------------- #
fits_path = paths.out_from("1b_psychometric") / "psychometric_fits.csv"
fits_df = pd.read_csv(fits_path)

included = fits_df[fits_df["included"] & (fits_df["protocol"] == PROTOCOL)]
included_ids = included["session_id"].tolist()

if not included_ids:
    raise SystemExit(
        f"No {PROTOCOL} sessions passed 1b_psychometric's inclusion criteria."
    )

mu_sigma_lookup = fits_df.set_index("session_id")[["mu", "sigma"]].to_dict("index")

cell_dist_out = paths.out_from("2a_cell_distribution")
activity_out = paths.out_from("2b_activity_statistics")
cell_inclusion = load_cell_inclusion(cell_dist_out, activity_out)

included_cell_ids_by_session = {
    sid: set(sub.loc[sub["include"], "cell_id"])
    for sid, sub in cell_inclusion.groupby("session_id")
}

print(
    f"Using {len(included_ids)} included {PROTOCOL} sessions, "
    f"{int(cell_inclusion['include'].sum())}/{len(cell_inclusion)} cells included"
)


# --------------------------------------------------------------------- #
# 2. PCA helpers
# --------------------------------------------------------------------- #
def fit_pca(matrix: np.ndarray, random_state: int = RANDOM_STATE) -> PCA:
    """Z-score cells and fit PCA up to the maximum non-zero covariance rank."""
    z = zscore_population(matrix)

    if z.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix, got shape {z.shape}.")

    n_trials, n_cells = z.shape
    n_components = min(n_cells, n_trials - 1)

    if n_components < 1:
        raise ValueError(f"Need at least two trials; got shape {z.shape}.")

    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(z)
    return pca


def pcs_for_threshold(cumulative: np.ndarray, threshold: float) -> int:
    """Minimum PC count required to reach a cumulative variance threshold."""
    return int(np.searchsorted(cumulative, threshold, side="left") + 1)


def pca_metrics(matrix: np.ndarray) -> dict:
    """Return dimensionality metrics and the first SCREE_N_PC PCs."""
    pca = fit_pca(matrix, random_state=RANDOM_STATE)
    eigenvalues = pca.explained_variance_
    var_ratio = pca.explained_variance_ratio_
    cumulative = np.cumsum(var_ratio)

    denom = np.sum(eigenvalues ** 2)
    pr = float(np.sum(eigenvalues) ** 2 / denom) if denom > 0 else np.nan

    n_show = min(SCREE_N_PC, len(var_ratio))
    spectrum = ",".join(f"{v:.8g}" for v in var_ratio[:n_show])

    return {
        "participation_ratio": pr,
        "pr_fraction_of_rank": (
            float(pr / len(var_ratio)) if np.isfinite(pr) else np.nan
        ),
        "var_pc1": float(var_ratio[0]),
        "var_top3": float(var_ratio[:min(3, len(var_ratio))].sum()),
        "n_pc_50": pcs_for_threshold(cumulative, 0.50),
        "n_pc_80": pcs_for_threshold(cumulative, 0.80),
        "n_pc_90": pcs_for_threshold(cumulative, 0.90),
        "pca_rank": len(var_ratio),
        "var_ratio_top": spectrum,
    }


def spectrum_from_string(value) -> np.ndarray:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return np.array([], dtype=float)
    text = str(value).strip()
    if not text:
        return np.array([], dtype=float)
    return np.array([float(v) for v in text.split(",")], dtype=float)


def percentile_band(spectra: list[np.ndarray]):
    """Mean and 2.5--97.5 percentile interval for common-length spectra."""
    spectra = [s for s in spectra if len(s)]
    if not spectra:
        return None

    n = min(len(s) for s in spectra)
    arr = np.vstack([s[:n] for s in spectra])
    return (
        arr.mean(axis=0),
        np.percentile(arr, CI_LOW, axis=0),
        np.percentile(arr, CI_HIGH, axis=0),
    )


def add_cumulative_thresholds(ax, cumulative: np.ndarray, max_pc: int):
    for threshold in (0.50, 0.80, 0.90):
        ax.axhline(
            threshold, color="0.65", linestyle=":", linewidth=0.8, zorder=0
        )
        n_pc = pcs_for_threshold(cumulative, threshold)
        if n_pc <= max_pc:
            ax.axvline(
                n_pc, color="0.65", linestyle=":", linewidth=0.8, zorder=0
            )
            ax.text(
                n_pc, threshold + 0.025, f"{n_pc}",
                fontsize=7, ha="left", va="bottom"
            )


# --------------------------------------------------------------------- #
# 3. Run PCA for every population and shuffle condition
# --------------------------------------------------------------------- #
pca_cache = paths.store / "pca.pkl"

REQUIRED_SHUFFLE_COLUMNS = [
    "session_id", "window", "roi_name", "labeled_grouped",
    "condition", "rep", "n_cells",
    "participation_ratio", "pr_fraction_of_rank",
    "n_pc_50", "n_pc_80", "n_pc_90", "pca_rank", "var_ratio_top",
]

REQUIRED_REAL_COLUMNS = [
    "session_id", "window", "roi_name", "labeled_grouped",
    "n_cells", "n_trials",
    "participation_ratio", "pr_fraction_of_rank",
    "n_pc_50", "n_pc_80", "n_pc_90", "pca_rank",
    "var_pc1", "var_top3", "var_ratio_per_pc",
]

shuffle_results = None
real_summary = None

if not args.recompute and pca_cache.exists():
    try:
        candidate = pd.read_pickle(pca_cache)
        cached_shuffle = candidate["shuffle_results"]
        cached_real = candidate["real_summary"]

        missing_shuffle = [
            c for c in REQUIRED_SHUFFLE_COLUMNS
            if c not in cached_shuffle.columns
        ]
        missing_real = [
            c for c in REQUIRED_REAL_COLUMNS
            if c not in cached_real.columns
        ]

        if not missing_shuffle and not missing_real:
            shuffle_results = cached_shuffle
            real_summary = cached_real
            print(
                f"Reusing cached PCA results from {paths.store} "
                "(pass --recompute to reload 0_data/)"
            )
        else:
            print(
                "Cached PCA results are from an older analysis version; "
                "recomputing."
            )
    except Exception as exc:
        print(f"Could not load cached PCA results ({exc}); recomputing.")


if shuffle_results is None:
    sessions = load_sessions(
        protocols=[PROTOCOL],
        load_behaviordata=True,
        load_videodata=True,
        load_celldata=True,
        load_calciumdata=True,
        only_session_ids=included_ids,
    )

    area_label_by_session = {}

    for ses in sessions:
        included_cells = included_cell_ids_by_session.get(ses.session_id)
        if not included_cells:
            continue

        cd = ses.celldata[
            ses.celldata["cell_id"].isin(included_cells)
        ].reset_index(drop=True)

        area, label, _ = get_area_label(
            cd,
            label_split_areas=LABEL_SPLIT_AREAS,
        )
        area_label_by_session[ses.session_id] = dict(
            zip(cd["cell_id"], zip(area, label))
        )

    def process_one_group(
        session_id,
        window_name,
        area,
        label,
        matrix,
        stim_bin,
        choice,
        cell_ids,
        seed,
    ):
        rng_local = np.random.default_rng(seed)
        specs = build_standard_condition_specs(stim_bin, choice)

        df = run_across_shuffle_conditions(
            matrix,
            specs,
            pca_metrics,
            n_shuffles=N_SHUFFLES,
            rng=rng_local,
        )

        df.insert(0, "session_id", session_id)
        df.insert(1, "window", window_name)
        df.insert(2, "roi_name", area)
        df.insert(3, "labeled_grouped", label)
        df.insert(4, "n_cells", len(cell_ids))

        # Store the full real spectrum. Shuffle rows store only the first
        # SCREE_N_PC PCs to keep the cache/CSV reasonably sized.
        pca_real = fit_pca(matrix, random_state=seed)
        eigenvalues = pca_real.explained_variance_
        var_ratio = pca_real.explained_variance_ratio_
        cumulative = np.cumsum(var_ratio)

        denom = np.sum(eigenvalues ** 2)
        real_pr = (
            float(np.sum(eigenvalues) ** 2 / denom)
            if denom > 0 else np.nan
        )

        real_row = {
            "session_id": session_id,
            "window": window_name,
            "roi_name": area,
            "labeled_grouped": label,
            "n_cells": len(cell_ids),
            "n_trials": matrix.shape[0],
            "participation_ratio": real_pr,
            "pr_fraction_of_rank": (
                real_pr / len(var_ratio)
                if np.isfinite(real_pr) else np.nan
            ),
            "n_pc_50": pcs_for_threshold(cumulative, 0.50),
            "n_pc_80": pcs_for_threshold(cumulative, 0.80),
            "n_pc_90": pcs_for_threshold(cumulative, 0.90),
            "pca_rank": len(var_ratio),
            "var_pc1": float(var_ratio[0]),
            "var_top3": float(var_ratio[:min(3, len(var_ratio))].sum()),
            "var_ratio_per_pc": ",".join(f"{v:.8g}" for v in var_ratio),
        }

        return df, real_row

    print(
        f"Running PCA across shuffle conditions for {len(included_ids)} "
        f"sessions (n_jobs={N_JOBS_SESSIONS}) ..."
    )

    groups = list(
        iterate_population_groups(
            sessions,
            mu_sigma_lookup,
            WINDOWS,
            included_cell_ids_by_session,
            area_label_by_session,
            min_cells=MIN_CELLS_PER_GROUP,
        )
    )

    print(
        f"{len(groups)} populations with "
        f">={MIN_CELLS_PER_GROUP} cells"
    )

    nested = Parallel(
        n_jobs=N_JOBS_SESSIONS,
        backend="loky",
        verbose=JOBLIB_VERBOSE,
    )(
        delayed(process_one_group)(
            sid, wname, area, label, matrix, stim_bin, choice,
            cell_ids, RANDOM_STATE + i
        )
        for i, (
            sid, wname, area, label, matrix, stim_bin, choice, cell_ids
        ) in enumerate(groups)
    )

    shuffle_results = pd.concat(
        [r[0] for r in nested], ignore_index=True
    )
    real_summary = pd.DataFrame([r[1] for r in nested])

    pd.to_pickle(
        {"shuffle_results": shuffle_results, "real_summary": real_summary},
        pca_cache,
    )


shuffle_results.to_csv(paths.out / "pca_shuffle_results.csv", index=False)
real_summary.to_csv(paths.out / "pca_real_summary.csv", index=False)

print(
    f"\n{len(real_summary)} populations analyzed, "
    f"{len(shuffle_results)} total (condition, rep) rows"
)


# --------------------------------------------------------------------- #
# 4. Text summaries
# --------------------------------------------------------------------- #
if len(shuffle_results):
    print("\nMean participation ratio by (window, area, condition):")
    print(
        shuffle_results
        .groupby(["window", "roi_name", "condition"])["participation_ratio"]
        .mean()
        .round(2)
        .to_string()
    )

    print("\nReal-data dimensionality by (window, area):")
    print(
        real_summary
        .groupby(["window", "roi_name"])[
            ["participation_ratio", "pr_fraction_of_rank", "n_pc_80", "n_pc_90"]
        ]
        .mean()
        .round(2)
        .to_string()
    )


# --------------------------------------------------------------------- #
# 5. Figure 1 -- PR across shuffle conditions
# --------------------------------------------------------------------- #
if len(shuffle_results):
    for window_name in WINDOWS:
        sub = shuffle_results[shuffle_results["window"] == window_name]
        if sub.empty:
            continue

        areas = [
            a for a in DEFAULT_AREA_ORDER
            if a in sub["roi_name"].unique()
        ]

        fig, axes = plt.subplots(
            1, len(areas),
            figsize=(4.5 * len(areas), 4.5),
            squeeze=False,
        )

        for ax, area in zip(axes[0], areas):
            asub = sub[sub["roi_name"] == area]
            plot_shuffle_condition_comparison(
                ax,
                asub,
                "participation_ratio",
                "participation ratio (effective dimensionality)",
            )
            ax.set_title(area)

        fig.suptitle(
            f"PCA: participation ratio across shuffle conditions -- "
            f"{window_name}, {PROTOCOL}",
            y=1.03,
        )
        fig.tight_layout()
        save_fig(fig, figdir, f"1_shuffle_comparison_{window_name}")


# --------------------------------------------------------------------- #
# 6. Figure 2 -- dimensionality versus population size
#
# PR is expected to depend partly on the number of available dimensions.
# The lower row therefore shows PR normalized by the PCA rank.
# --------------------------------------------------------------------- #
if len(real_summary):
    fig, axes = plt.subplots(
        2, len(WINDOWS),
        figsize=(5 * len(WINDOWS), 7.0),
        squeeze=False,
    )

    for col, window_name in enumerate(WINDOWS):
        sub = real_summary[real_summary["window"] == window_name]
        if sub.empty:
            continue

        for area in DEFAULT_AREA_ORDER:
            asub = sub[sub["roi_name"] == area]
            if asub.empty:
                continue

            axes[0, col].scatter(
                asub["n_cells"],
                asub["participation_ratio"],
                s=18,
                alpha=0.65,
                color=AREA_COLORS.get(area, "gray"),
                label=area,
            )
            axes[1, col].scatter(
                asub["n_cells"],
                asub["pr_fraction_of_rank"],
                s=18,
                alpha=0.65,
                color=AREA_COLORS.get(area, "gray"),
                label=area,
            )

        axes[0, col].set_title(window_name)
        axes[0, col].set_xlabel("population size (cells)")
        axes[0, col].set_ylabel("participation ratio")
        axes[1, col].set_xlabel("population size (cells)")
        axes[1, col].set_ylabel("PR / PCA rank")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 0].legend(handles, labels, fontsize=8, frameon=False)

    fig.suptitle(
        f"PCA: dimensionality versus population size -- {PROTOCOL}",
        y=1.01,
    )
    fig.tight_layout()
    save_fig(fig, figdir, "2_dimensionality_vs_size")


# --------------------------------------------------------------------- #
# 7. Figure 3 -- representative real-vs-shuffle spectra
#
# The representative population is closest to the median population
# size within each area/window, rather than simply the first population.
# --------------------------------------------------------------------- #
if len(real_summary) and len(shuffle_results):
    for window_name in WINDOWS:
        sub_real = real_summary[
            real_summary["window"] == window_name
        ]
        if sub_real.empty:
            continue

        areas = [
            a for a in DEFAULT_AREA_ORDER
            if a in sub_real["roi_name"].unique()
        ]

        fig, axes = plt.subplots(
            2, len(areas),
            figsize=(5.0 * len(areas), 7.0),
            squeeze=False,
        )

        for col, area in enumerate(areas):
            area_real = sub_real[sub_real["roi_name"] == area].copy()
            if area_real.empty:
                continue

            median_size = area_real["n_cells"].median()
            distance = np.abs(
                area_real["n_cells"].to_numpy() - median_size
            )
            example = area_real.iloc[np.argmin(distance)]

            example_id = example["session_id"]
            example_label = example["labeled_grouped"]
            real_spectrum = spectrum_from_string(
                example["var_ratio_per_pc"]
            )

            example_shuffle = shuffle_results[
                (shuffle_results["window"] == window_name)
                & (shuffle_results["roi_name"] == area)
                & (shuffle_results["session_id"] == example_id)
                & (shuffle_results["labeled_grouped"] == example_label)
                & (shuffle_results["condition"] != "real")
            ]

            # ---- conventional scree plot -----------------------------
            ax = axes[0, col]
            n_real = min(SCREE_N_PC, len(real_spectrum))
            x_real = np.arange(1, n_real + 1)

            ax.plot(
                x_real,
                real_spectrum[:n_real],
                marker="o",
                markersize=3,
                linewidth=1.5,
                color=AREA_COLORS.get(area, "gray"),
                label="real",
                zorder=4,
            )

            for condition in SHUFFLE_CONDITIONS:
                cond = example_shuffle[
                    example_shuffle["condition"] == condition
                ]
                band = percentile_band(
                    [
                        spectrum_from_string(v)
                        for v in cond["var_ratio_top"].dropna()
                    ]
                )
                if band is None:
                    continue

                mean, low, high = band
                n = min(SCREE_N_PC, len(mean))
                x = np.arange(1, n + 1)

                line = ax.plot(
                    x, mean[:n], linewidth=1.2, label=condition
                )[0]
                ax.fill_between(
                    x, low[:n], high[:n],
                    alpha=0.15,
                    color=line.get_color(),
                    linewidth=0,
                )

            ax.set_xlabel("PC")
            ax.set_ylabel("variance explained")
            ax.set_xlim(1, min(SCREE_N_PC, len(real_spectrum)))
            ax.set_title(
                f"{area}\n{example_id}, {example_label}, "
                f"n={example['n_cells']} cells"
            )
            if col == 0:
                ax.legend(fontsize=7, frameon=False)

            # ---- cumulative variance ---------------------------------
            ax = axes[1, col]
            n_real = min(CUMULATIVE_N_PC, len(real_spectrum))
            real_cum = np.cumsum(real_spectrum[:n_real])
            x_real = np.arange(1, n_real + 1)

            ax.plot(
                x_real,
                real_cum,
                linewidth=1.8,
                color=AREA_COLORS.get(area, "gray"),
                label="real",
                zorder=4,
            )

            add_cumulative_thresholds(
                ax,
                np.cumsum(real_spectrum),
                min(CUMULATIVE_N_PC, len(real_spectrum)),
            )

            for condition in SHUFFLE_CONDITIONS:
                cond = example_shuffle[
                    example_shuffle["condition"] == condition
                ]
                spectra = [
                    spectrum_from_string(v)
                    for v in cond["var_ratio_top"].dropna()
                ]
                spectra = [s for s in spectra if len(s)]
                if not spectra:
                    continue

                band = percentile_band([np.cumsum(s) for s in spectra])
                mean, low, high = band
                n = min(CUMULATIVE_N_PC, len(mean))
                x = np.arange(1, n + 1)

                line = ax.plot(
                    x, mean[:n], linewidth=1.2, label=condition
                )[0]
                ax.fill_between(
                    x, low[:n], high[:n],
                    alpha=0.15,
                    color=line.get_color(),
                    linewidth=0,
                )

            ax.set_xlabel("PC")
            ax.set_ylabel("cumulative variance explained")
            ax.set_xlim(
                1, min(CUMULATIVE_N_PC, len(real_spectrum))
            )
            ax.set_ylim(0, 1.02)

            if col == 0:
                ax.legend(
                    fontsize=7,
                    frameon=False,
                    loc="lower right",
                )

        fig.suptitle(
            f"PCA spectrum: representative population, real vs shuffles -- "
            f"{window_name}, {PROTOCOL}",
            y=1.01,
        )
        fig.tight_layout()
        save_fig(fig, figdir, f"3_pca_spectrum_{window_name}")


# --------------------------------------------------------------------- #
# 8. Figure 4 -- paired population-level shuffle effects
#
# The 20 shuffle realizations are collapsed within each population first.
# Thus each population contributes exactly one value per condition.
#
# Positive delta_PR means shuffling increased dimensionality.
# Positive delta_PC80 means shuffling required more PCs for 80% variance.
# --------------------------------------------------------------------- #
if len(shuffle_results):
    for window_name in WINDOWS:
        sub = shuffle_results[
            shuffle_results["window"] == window_name
        ]
        if sub.empty:
            continue

        real = sub[sub["condition"] == "real"][
            [
                "session_id", "roi_name", "labeled_grouped",
                "participation_ratio", "n_pc_80",
            ]
        ].rename(
            columns={
                "participation_ratio": "real_pr",
                "n_pc_80": "real_n_pc_80",
            }
        )

        shuf = sub[sub["condition"].isin(SHUFFLE_CONDITIONS)]

        shuf_summary = (
            shuf.groupby(
                ["session_id", "roi_name", "labeled_grouped", "condition"],
                as_index=False,
            )
            .agg(
                shuffle_pr=("participation_ratio", "median"),
                shuffle_n_pc_80=("n_pc_80", "median"),
            )
        )

        paired = shuf_summary.merge(
            real,
            on=["session_id", "roi_name", "labeled_grouped"],
            how="inner",
        )
        paired["delta_pr"] = paired["shuffle_pr"] - paired["real_pr"]
        paired["delta_n_pc_80"] = (
            paired["shuffle_n_pc_80"] - paired["real_n_pc_80"]
        )

        areas = [
            a for a in DEFAULT_AREA_ORDER
            if a in paired["roi_name"].unique()
        ]

        fig, axes = plt.subplots(
            2, len(areas),
            figsize=(5.0 * len(areas), 7.0),
            squeeze=False,
        )

        for col, area in enumerate(areas):
            asub = paired[paired["roi_name"] == area]

            pr_data = []
            pr_labels = []
            pc_data = []
            pc_labels = []

            for condition in SHUFFLE_CONDITIONS:
                vals_pr = asub.loc[
                    asub["condition"] == condition, "delta_pr"
                ].dropna()
                vals_pc = asub.loc[
                    asub["condition"] == condition, "delta_n_pc_80"
                ].dropna()

                if len(vals_pr):
                    pr_data.append(vals_pr)
                    pr_labels.append(condition)
                if len(vals_pc):
                    pc_data.append(vals_pc)
                    pc_labels.append(condition)

            ax = axes[0, col]
            if pr_data:
                ax.boxplot(
                    pr_data,
                    tick_labels=pr_labels,
                    showfliers=False,
                )
            ax.axhline(0, color="0.35", linestyle="--", linewidth=1)
            ax.set_ylabel("shuffle PR - real PR")
            ax.set_title(area)
            ax.tick_params(axis="x", rotation=35)

            ax = axes[1, col]
            if pc_data:
                ax.boxplot(
                    pc_data,
                    tick_labels=pc_labels,
                    showfliers=False,
                )
            ax.axhline(0, color="0.35", linestyle="--", linewidth=1)
            ax.set_ylabel("shuffle PC80 - real PC80")
            ax.set_xlabel("shuffle condition")
            ax.tick_params(axis="x", rotation=35)

        fig.suptitle(
            f"PCA: paired dimensionality change caused by shuffling -- "
            f"{window_name}, {PROTOCOL}",
            y=1.01,
        )
        fig.tight_layout()
        save_fig(fig, figdir, f"4_pca_shuffle_effects_{window_name}")


# --------------------------------------------------------------------- #
# 9. Figure 5 -- aggregate normalized spectrum across populations
#
# The shaded region is a 2.5--97.5 percentile interval, not a formal
# confidence interval. It is descriptive across populations/realizations.
# --------------------------------------------------------------------- #
if len(shuffle_results):
    for window_name in WINDOWS:
        sub = shuffle_results[
            shuffle_results["window"] == window_name
        ]
        if sub.empty:
            continue

        areas = [
            a for a in DEFAULT_AREA_ORDER
            if a in sub["roi_name"].unique()
        ]

        fig, axes = plt.subplots(
            1, len(areas),
            figsize=(5.0 * len(areas), 4.5),
            squeeze=False,
        )

        for ax, area in zip(axes[0], areas):
            asub = sub[sub["roi_name"] == area]

            for condition in CONDITION_ORDER:
                cond = asub[asub["condition"] == condition]
                if cond.empty:
                    continue

                spectra = [
                    spectrum_from_string(v)
                    for v in cond["var_ratio_top"].dropna()
                ]
                band = percentile_band(spectra)
                if band is None:
                    continue

                mean, low, high = band
                n = min(AGGREGATE_N_PC, len(mean))
                x = np.arange(1, n + 1)

                line = ax.plot(
                    x, mean[:n], linewidth=1.5, label=condition
                )[0]
                ax.fill_between(
                    x, low[:n], high[:n],
                    alpha=0.12,
                    color=line.get_color(),
                    linewidth=0,
                )

            ax.set_xlabel("PC")
            ax.set_ylabel("variance explained")
            ax.set_title(area)
            ax.set_xlim(1, AGGREGATE_N_PC)

        axes[0, 0].legend(fontsize=7, frameon=False)

        fig.suptitle(
            f"PCA spectrum across populations -- {window_name}, {PROTOCOL}",
            y=1.03,
        )
        fig.tight_layout()
        save_fig(
            fig,
            figdir,
            f"5_pca_spectrum_population_{window_name}",
        )


print(f"\nWrote tables to {paths.out}, figures to {figdir}")
print(
    "Run `python b_progress/make_progress_md.py "
    "4a_population_pca` to build a markdown summary."
)
