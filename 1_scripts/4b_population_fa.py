# -*- coding: utf-8 -*-
"""
4b_population_fa
===================
Population-level Factor Analysis within each (area, label) group.
Unlike PCA (4a_population_pca.py), FA explicitly separates each cell's
variance into a SHARED part (explained by a small number of common
factors) and a PRIVATE/unique part (that cell's own idiosyncratic
noise) -- PCA's components can absorb single-cell noise into a
"shared" component if that noise happens to be large, since PCA has no
concept of per-cell noise at all. FA's `noise_variance_` is exactly
this project's answer to "how much of this cell's variance is actually
shared with the rest of the population" -- the headline metric here
(shared_variance_fraction) is 1 minus the mean of that, computed on
z-scored data so each cell's total variance is exactly 1 and
noise_variance_ is already a fraction.

CONDITIONED SHUFFLES: identical logic and conditions (global/stim/
choice/stim_choice) as 4a_population_pca.py -- see that script's module
docstring, and infotheory.population/shuffling. If shared_variance_
fraction collapses under the "stim" shuffle (which preserves each
cell's own tuning but destroys cross-cell correlations), the "shared"
variance FA is finding is mostly a byproduct of shared stimulus tuning,
not genuine correlated trial-to-trial variability.

MODULARITY: same as 4a -- condition specs come from
infotheory.population.build_standard_condition_specs; add a behavioral
variable there (or build an equivalent dict) with no other changes
needed anywhere in this script.

Uses scikit-learn's FactorAnalysis directly.

Dataset, sessions/cells: same filters as 3a/3b/3c/4a.

Outputs
-------
2_pipeline/4b_population_fa/
    out/    fa_shuffle_results.csv   one row per (session, window, area,
                                      label, condition, rep): shared
                                      variance fraction
            fa_real_communality.csv   one row per (session, window,
                                      area, label, cell_id): that
                                      cell's communality (1 -
                                      noise_variance) on the real data
            figures/*.png
    store/  fa.pkl   cached (reused unless --recompute)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.decomposition import FactorAnalysis

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
N_FACTORS = 5   # fixed, modest factor count -- a cross-validated choice (comparing held-out
                 # log-likelihood across candidate factor counts) would be more principled but
                 # much more expensive at this scale; fine as a first pass given the goal here
                 # is the shared/private SPLIT, not the exact optimal factor count
N_SHUFFLES = 20
RANDOM_STATE = 0

N_JOBS_SESSIONS = -1
JOBLIB_VERBOSE = 5

parser = argparse.ArgumentParser()
parser.add_argument("--recompute", action="store_true",
                     help="Reload sessions from 0_data/ instead of reusing the cached "
                          "store/fa.pkl")
args = parser.parse_args()

paths = get_pipeline_paths(__file__)
figdir = paths.out / "figures"
clear_figures(figdir)
set_style()


# --------------------------------------------------------------------- #
# 1. Included sessions/cells -- same filters as 3a/3b/3c/4a
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
      f"{int(cell_inclusion['include'].sum())}/{len(cell_inclusion)} cells included, "
      f"N_FACTORS={N_FACTORS}")


def n_factors_for(n_cells: int) -> int:
    return max(1, min(N_FACTORS, n_cells - 1))


def fa_metrics(matrix: np.ndarray) -> dict:
    """Analysis function for run_across_shuffle_conditions."""
    z = zscore_population(matrix)
    k = n_factors_for(z.shape[1])
    fa = FactorAnalysis(n_components=k, random_state=RANDOM_STATE)
    fa.fit(z)
    shared_fraction = float(1 - np.mean(fa.noise_variance_))
    return {"shared_variance_fraction": shared_fraction}


# --------------------------------------------------------------------- #
# 2. Run FA (real + shuffle conditions) -- or reuse the cache
# --------------------------------------------------------------------- #
fa_cache = paths.store / "fa.pkl"
REQUIRED_COLUMNS = ["session_id", "window", "roi_name", "labeled_grouped", "condition", "rep",
                    "shared_variance_fraction"]

shuffle_results = communality_table = None
if not args.recompute and fa_cache.exists():
    try:
        candidate = pd.read_pickle(fa_cache)
        missing = [c for c in REQUIRED_COLUMNS if c not in candidate["shuffle_results"].columns]
        if missing:
            print(f"Cached FA results are missing columns {missing} -- recomputing from 0_data/ "
                  f"instead of using {fa_cache} (safe to delete that file to silence this check).")
        else:
            shuffle_results = candidate["shuffle_results"]
            communality_table = candidate["communality"]
            print(f"Reusing cached FA results from {paths.store} (pass --recompute to reload 0_data/)")
    except Exception as e:
        print(f"Could not load cached FA results ({e}); recomputing from 0_data/ ...")

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
        df = run_across_shuffle_conditions(matrix, specs, fa_metrics, n_shuffles=N_SHUFFLES, rng=rng_local)
        df.insert(0, "session_id", session_id)
        df.insert(1, "window", window_name)
        df.insert(2, "roi_name", area)
        df.insert(3, "labeled_grouped", label)
        df.insert(4, "n_cells", len(cell_ids))

        z = zscore_population(matrix)
        fa = FactorAnalysis(n_components=n_factors_for(z.shape[1]), random_state=seed)
        fa.fit(z)
        communality = 1 - fa.noise_variance_
        comm_rows = [{"session_id": session_id, "window": window_name, "roi_name": area,
                       "labeled_grouped": label, "cell_id": cid, "communality": float(c)}
                     for cid, c in zip(cell_ids, communality)]
        return df, comm_rows

    print(f"Running FA across shuffle conditions for {len(included_ids)} sessions "
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
    communality_table = pd.DataFrame([row for r in nested for row in r[1]])
    pd.to_pickle({"shuffle_results": shuffle_results, "communality": communality_table}, fa_cache)

shuffle_results.to_csv(paths.out / "fa_shuffle_results.csv", index=False)
communality_table.to_csv(paths.out / "fa_real_communality.csv", index=False)
print(f"\n{shuffle_results['session_id'].nunique()} sessions, "
      f"{len(shuffle_results)} total (condition, rep) rows, {len(communality_table)} per-cell communalities")


# --------------------------------------------------------------------- #
# 3. Summary
# --------------------------------------------------------------------- #
if len(shuffle_results):
    print("\nMean shared variance fraction by (window, area, condition):")
    print(shuffle_results.groupby(["window", "roi_name", "condition"])["shared_variance_fraction"]
          .mean().round(3).to_string())


# --------------------------------------------------------------------- #
# 4. Figures
# --------------------------------------------------------------------- #
if len(shuffle_results):
    # 4a. COMMON PLOT 1: shuffle condition comparison
    for window_name in WINDOWS:
        sub = shuffle_results[shuffle_results["window"] == window_name]
        if sub.empty:
            continue
        areas = [a for a in DEFAULT_AREA_ORDER if a in sub["roi_name"].unique()]
        fig, axes = plt.subplots(1, len(areas), figsize=(4.5 * len(areas), 4.5), squeeze=False)
        for ax, area in zip(axes[0], areas):
            asub = sub[sub["roi_name"] == area]
            plot_shuffle_condition_comparison(ax, asub, "shared_variance_fraction",
                                               "shared variance fraction")
            ax.set_title(area)
        fig.suptitle(f"FA: shared variance fraction across shuffle conditions -- {window_name}, {PROTOCOL}",
                     y=1.03)
        fig.tight_layout()
        save_fig(fig, figdir, f"1_shuffle_comparison_{window_name}")

    # 4b. COMMON PLOT 2: metric vs population size
    real_sv = shuffle_results[shuffle_results["condition"] == "real"]
    fig, axes = plt.subplots(1, len(WINDOWS), figsize=(5 * len(WINDOWS), 4.5), squeeze=False)
    for ax, window_name in zip(axes[0], WINDOWS):
        sub = real_sv[real_sv["window"] == window_name]
        if sub.empty:
            continue
        plot_metric_vs_population_size(ax, sub, "n_cells", "shared_variance_fraction", "roi_name",
                                        "shared variance fraction (real data)", AREA_COLORS)
        ax.set_title(window_name)
    fig.suptitle(f"FA: shared variance vs. population size -- {PROTOCOL}", y=1.03)
    fig.tight_layout()
    save_fig(fig, figdir, "2_shared_variance_vs_size")

    # 4c. UNIQUE TO FA: per-cell communality distribution by area/label --
    #    the direct visualization of FA's distinguishing feature (which
    #    PCA has no equivalent of): how much of EACH cell's variance is
    #    shared with the population vs private to that cell.
    if len(communality_table):
        merged = communality_table.merge(
            pd.read_csv(cell_dist_out / "celldata_combined.csv")[["session_id", "cell_id", "labeled"]],
            on=["session_id", "cell_id"], how="left")
        for window_name in WINDOWS:
            sub = merged[merged["window"] == window_name]
            if sub.empty:
                continue
            area_order = [a for a in DEFAULT_AREA_ORDER if a in sub["roi_name"].unique()]
            fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(area_order)), 5))
            sns.boxplot(showfliers=False, data=sub, x="roi_name", y="communality", order=area_order, hue="roi_name",
                            palette=AREA_COLORS, legend=False, ax=ax)
            ax.set_ylim(0, 1)
            ax.set_xlabel("")
            ax.set_ylabel("communality (fraction of this cell's variance that's shared)")
            ax.set_title(f"Per-cell communality -- {window_name}, {PROTOCOL}")
            fig.tight_layout()
            save_fig(fig, figdir, f"3_communality_by_area_{window_name}")

print(f"\nWrote tables to {paths.out}, figures to {figdir}")
print("Run `python b_progress/make_progress_md.py 4b_population_fa` to build a markdown summary.")
