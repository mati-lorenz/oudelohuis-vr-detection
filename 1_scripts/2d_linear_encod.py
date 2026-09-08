# -*- coding: utf-8 -*-
"""
2d_linear_encod
=================
Linear regressions relating each recorded cell's trial-by-trial
activity to the same behavioral/performance predictors 2c_information.py
computed mutual information for -- then compares the two: how much of
each cell's information about a predictor does a LINEAR fit actually
capture? (mi_from_r2 / frac_explained_linearly, same logic as
1c_behavior.py's pairwise comparison, applied here per cell instead of
per behavioral-variable-pair.)

Same dataset assembly as 2c_information.py (see infotheory.
neural_encoding and that script's module docstring for the session/
trial/cell filtering and the position-window convention) -- this script
independently rebuilds the same per-session tables rather than reading
2c's raw data (Rule #1: read another step's out/, not its store/), and
reads ONLY 2c's cell_predictor_mi.csv (the actual MI values) to merge
against.

Also computes, per cell per window, a MULTIVARIATE fit (that cell's
activity ~ every predictor at once), matching 1c_behavior.py's leave-
one-out multivariate section -- answering "how well can this cell's
activity be explained by everything we measured, taken together" in
addition to the pairwise picture.

Same important caveat as 2c_information.py applies here: choice/correct
r^2 values can be inflated by the SAME sampling-precision confound
(choice-dependent running speed changes trial-window frame counts) --
see that script's module docstring for the validated details, and this
script's own confound-check print.

Outputs
-------
2_pipeline/2d_linear_encod/
    out/    cell_predictor_linear.csv    one row per (session, window,
                                          cell, predictor): r, r2,
                                          p_linear, merged with 2c's MI
                                          -> mi_bits_from_r2,
                                          frac_explained_linearly
            cell_multivariate_linear.csv one row per (session, window,
                                          cell): r2/adj_r2 from ALL
                                          predictors at once
            figures/*.png
    store/  linear_combined.pkl           cached tables (reused unless
                                          --recompute)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

from infotheory.pipeline import get_pipeline_paths
from infotheory.session import load_sessions
from infotheory.neural_encoding import load_cell_inclusion, build_session_tables, BEHAVIOR_PREDICTORS
from infotheory.info_theory import linear_fit, multivariate_linear_fit, compare_mi_to_linear
from infotheory.celldata_utils import AREA_COLORS, LABEL_SPLIT_AREAS, get_area_label
from infotheory.plotting import set_style, save_fig, clear_figures

PROTOCOL = "DN"
WINDOWS = {"pre_stim": (-30.0, -10.0), "stim": (0.0, 20.0), "reward": (25.0, 45.0)}
PREDICTORS = [f"cur_{v}" for v in BEHAVIOR_PREDICTORS] + ["stim_bin", "choice", "correct"]
MAX_CELLS_PER_SESSION = None
LINEAR_SUFFICIENT_FRAC = 0.8
RANDOM_STATE = 0

N_JOBS_SESSIONS = 4
N_JOBS_GROUPS = 2
JOBLIB_VERBOSE = 5

parser = argparse.ArgumentParser()
parser.add_argument("--recompute", action="store_true",
                     help="Reload sessions from 0_data/ instead of reusing the cached "
                          "store/linear_combined.pkl")
args = parser.parse_args()

paths = get_pipeline_paths(__file__)
figdir = paths.out / "figures"
clear_figures(figdir)
set_style()


# --------------------------------------------------------------------- #
# 1. Included sessions (1b), mu/sigma, cell inclusion (2a+2b) -- same as
#    2c_information.py
# --------------------------------------------------------------------- #
fits_path = paths.out_from("1b_psychometric") / "psychometric_fits.csv"
if not fits_path.exists():
    raise SystemExit(f"Run 1b_psychometric.py first -- expected {fits_path}")
fits_df = pd.read_csv(fits_path)
included = fits_df[fits_df["included"] & (fits_df["protocol"] == PROTOCOL)]
included_ids = included["session_id"].tolist()
if not included_ids:
    raise SystemExit(f"No {PROTOCOL} sessions passed 1b_psychometric's inclusion criteria.")
mu_sigma_lookup = fits_df.set_index("session_id")[["mu", "sigma"]].to_dict("index")

cell_dist_out = paths.out_from("2a_cell_distribution")
activity_out = paths.out_from("2b_activity_statistics")
information_out = paths.out_from("2c_information")
for req_path, req_file in [(cell_dist_out, "celldata_combined.csv"), (activity_out, "qc_per_cell.csv"),
                            (information_out, "cell_predictor_mi.csv")]:
    if not (req_path / req_file).exists():
        raise SystemExit(f"Missing {req_path / req_file} -- run the earlier pipeline steps first.")

cell_inclusion = load_cell_inclusion(cell_dist_out, activity_out)
included_cell_ids_by_session = {
    sid: set(sub.loc[sub["include"], "cell_id"]) for sid, sub in cell_inclusion.groupby("session_id")
}
print(f"Using {len(included_ids)} included {PROTOCOL} sessions, "
      f"{int(cell_inclusion['include'].sum())}/{len(cell_inclusion)} cells included")


# --------------------------------------------------------------------- #
# 2. Build tables and compute pairwise + multivariate linear fits -- or
#    reuse the cache.
# --------------------------------------------------------------------- #
combined_cache = paths.store / "linear_combined.pkl"
REQUIRED_COLUMNS = ["session_id", "window", "cell_id", "predictor", "r2", "p_linear"]

pairwise_table = multivariate_table = None
if not args.recompute and combined_cache.exists():
    try:
        candidate = pd.read_pickle(combined_cache)
        missing = [c for c in REQUIRED_COLUMNS if c not in candidate["pairwise"].columns]
        if missing:
            print(f"Cached linear tables are missing columns {missing} -- looks like they were built "
                  f"by an older version of this script. Recomputing from 0_data/ instead of using "
                  f"{combined_cache} (safe to delete that file to silence this check in the future).")
        else:
            pairwise_table = candidate["pairwise"]
            multivariate_table = candidate["multivariate"]
            print(f"Reusing cached linear tables from {paths.store} (pass --recompute to reload 0_data/)")
    except Exception as e:
        print(f"Could not load cached linear tables ({e}); recomputing from 0_data/ ...")

if pairwise_table is None:
    sessions = load_sessions(protocols=[PROTOCOL], load_behaviordata=True, load_videodata=True,
                              load_celldata=True, load_calciumdata=True, only_session_ids=included_ids)

    def build_one_session(ses, rng_seed):
        rng_local = np.random.default_rng(rng_seed)
        included_cells = included_cell_ids_by_session.get(ses.session_id)
        if not included_cells:
            return {}
        if MAX_CELLS_PER_SESSION is not None and len(included_cells) > MAX_CELLS_PER_SESSION:
            included_cells = set(rng_local.choice(list(included_cells), size=MAX_CELLS_PER_SESSION,
                                                    replace=False))
        return build_session_tables(ses, mu_sigma_lookup, WINDOWS, included_cell_ids=included_cells)

    print(f"Building per-session tables for {len(sessions)} sessions (n_jobs={N_JOBS_SESSIONS}) ...")
    session_tables = Parallel(n_jobs=N_JOBS_SESSIONS, backend="loky", verbose=JOBLIB_VERBOSE)(
        delayed(build_one_session)(ses, RANDOM_STATE + i) for i, ses in enumerate(sessions)
    )

    known_non_cell_cols = {"trialNumber", "stim_bin", "choice", "correct"} | {
        f"cur_{v}" for v in BEHAVIOR_PREDICTORS}
    groups = []
    for ses, tables in zip(sessions, session_tables):
        for window_name, table in tables.items():
            if table.empty:
                continue
            cell_cols = [c for c in table.columns if c not in known_non_cell_cols]
            groups.append((ses.session_id, window_name, table, cell_cols))

    def compute_linear_for_group(session_id, window_name, table, cell_cols):
        pairwise_records, multivariate_records = [], []
        available_predictors = [p for p in PREDICTORS if p in table.columns]
        for cell_id in cell_cols:
            for predictor in available_predictors:
                pair_data = table[[predictor, cell_id]].dropna()
                if len(pair_data) < 10:
                    continue
                lin = linear_fit(pair_data[predictor], pair_data[cell_id])
                pairwise_records.append({
                    "session_id": session_id, "window": window_name, "cell_id": cell_id,
                    "predictor": predictor, "n_trials": lin.n, "r": lin.r, "r2": lin.r2,
                    "p_linear": lin.p_value,
                })

            mv_data = table[available_predictors + [cell_id]].dropna()
            if len(mv_data) >= len(available_predictors) + 10:
                mv = multivariate_linear_fit(mv_data[available_predictors].to_numpy(), mv_data[cell_id],
                                              predictor_names=available_predictors)
                multivariate_records.append({
                    "session_id": session_id, "window": window_name, "cell_id": cell_id,
                    "n_predictors": len(available_predictors), "n_trials": mv.n,
                    "r2": mv.r2, "adj_r2": mv.adj_r2, "p_linear": mv.p_value,
                })
        return pairwise_records, multivariate_records

    print(f"Computing linear fits for {len(groups)} (session, window) groups (n_jobs={N_JOBS_GROUPS}) ...")
    nested = Parallel(n_jobs=N_JOBS_GROUPS, backend="loky", verbose=JOBLIB_VERBOSE)(
        delayed(compute_linear_for_group)(sid, wname, table, cell_cols)
        for sid, wname, table, cell_cols in groups
    )
    pairwise_table = pd.DataFrame([rec for pw, _ in nested for rec in pw])
    multivariate_table = pd.DataFrame([rec for _, mv in nested for rec in mv])
    pd.to_pickle({"pairwise": pairwise_table, "multivariate": multivariate_table}, combined_cache)

pairwise_table.to_csv(paths.out / "cell_predictor_linear_raw.csv", index=False)
multivariate_table.to_csv(paths.out / "cell_multivariate_linear.csv", index=False)
print(f"\n{len(pairwise_table)} pairwise linear fits, {len(multivariate_table)} multivariate fits")


# --------------------------------------------------------------------- #
# 3. Merge with 2c's MI table -> the actual MI-vs-linear comparison
# --------------------------------------------------------------------- #
mi_table = pd.read_csv(information_out / "cell_predictor_mi.csv")
merge_cols = ["session_id", "window", "cell_id", "predictor"]
comparison = pairwise_table.merge(mi_table[merge_cols + ["mi_bits_corrected", "mi_p_value"]],
                                   on=merge_cols, how="inner")

frac_list, mi_from_r2_list, mi_ref_list, suff_list = [], [], [], []
for _, row in comparison.iterrows():
    cmp = compare_mi_to_linear(row["mi_bits_corrected"], row["r2"])
    frac_list.append(cmp.frac_explained_linearly)
    mi_from_r2_list.append(cmp.mi_from_r2_bits)
    mi_ref_list.append(cmp.mi_reference_bits)
    suff_list.append((cmp.frac_explained_linearly >= LINEAR_SUFFICIENT_FRAC)
                      if not np.isnan(cmp.frac_explained_linearly) else None)
comparison["mi_bits_from_r2"] = mi_from_r2_list
comparison["mi_reference_bits"] = mi_ref_list
comparison["frac_explained_linearly"] = frac_list
comparison["linear_sufficient"] = suff_list

celldata_combined = pd.read_csv(cell_dist_out / "celldata_combined.csv")[
    ["session_id", "cell_id", "roi_name", "labeled"]]
comparison = comparison.merge(celldata_combined, on=["session_id", "cell_id"], how="left")
_, comparison["labeled_grouped"], _ = get_area_label(comparison, label_split_areas=LABEL_SPLIT_AREAS)
comparison.to_csv(paths.out / "cell_predictor_mi_vs_linear.csv", index=False)

if len(comparison):
    finite = comparison.dropna(subset=["frac_explained_linearly"])
    n_insufficient = (finite["linear_sufficient"] == False).sum()  # noqa: E712
    print(f"\n{n_insufficient}/{len(finite)} (cell, predictor, window) combinations show a linear fit "
          f"missing >{100 * (1 - LINEAR_SUFFICIENT_FRAC):.0f}% of the (bias-corrected) mutual information "
          f"(candidates for 2e_nonlinear_encod.py):")
    by_predictor = (finite.groupby("predictor")["linear_sufficient"]
                     .apply(lambda s: (s == False).mean()).sort_values(ascending=False))  # noqa: E712
    print(by_predictor.to_string())


# --------------------------------------------------------------------- #
# 4. Figures
# --------------------------------------------------------------------- #
if len(comparison):
    for window_name in WINDOWS:
        sub = comparison[comparison["window"] == window_name].dropna(
            subset=["mi_reference_bits", "mi_bits_from_r2"])
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=["mi_reference_bits", "mi_bits_from_r2"])
        if sub.empty:
            continue
        lims = [0, sub["mi_reference_bits"].max() * 1.1 + 1e-9]
        fig, ax = plt.subplots(figsize=(5, 5))
        pred_order = [p for p in PREDICTORS if p in sub["predictor"].unique()]
        palette = dict(zip(pred_order, sns.color_palette("tab10", len(pred_order))))
        for pred in pred_order:
            psub = sub[sub["predictor"] == pred]
            ax.scatter(psub["mi_reference_bits"], psub["mi_bits_from_r2"], s=10, alpha=0.5,
                       color=palette[pred], label=pred)
        ax.plot(lims, lims, color="grey", linestyle="--", linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("MI, best available estimate (bits)")
        ax.set_ylabel("MI implied by linear r\u00b2 (bits)")
        ax.set_title(f"Per-cell: does linear capture the full relationship? -- {window_name}")
        ax.legend(fontsize=7, markerscale=2, frameon=False)
        fig.tight_layout()
        save_fig(fig, figdir, f"1_mi_vs_linear_{window_name}")

    # Fraction explained linearly, by predictor, per window (box plot,
    # split by area -- same house style as 1c_behavior.py/2c)
    for window_name in WINDOWS:
        sub = comparison[(comparison["window"] == window_name)].dropna(subset=["frac_explained_linearly"])
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(PREDICTORS)), 5))
        pred_order = [p for p in PREDICTORS if p in sub["predictor"].unique()]
        sns.boxplot(data=sub, x="predictor", y="frac_explained_linearly", hue="roi_name",
                    palette=AREA_COLORS, order=pred_order, ax=ax, showfliers=False)
        ax.axhline(1.0, color="black", linestyle=":", linewidth=1)
        ax.set_ylabel("MI(linear) / MI(full)")
        ax.set_xlabel("")
        ax.set_title(f"Fraction of MI a linear fit captures, per cell -- {window_name}")
        ax.legend(fontsize=8, title="area", frameon=False)
        fig.tight_layout()
        save_fig(fig, figdir, f"2_frac_explained_linearly_{window_name}")

if len(multivariate_table):
    fig, ax = plt.subplots(figsize=(6, 5))
    window_order = [w for w in WINDOWS if w in multivariate_table["window"].unique()]
    sns.violinplot(data=multivariate_table, x="window", y="adj_r2", order=window_order, ax=ax,
                    hue="window", palette="Set2", legend=False)
    ax.axhline(0, color="grey", linewidth=0.7)
    ax.set_ylabel("adjusted R\u00b2 (all predictors jointly)")
    ax.set_title(f"Multivariate fit of each cell's activity -- {PROTOCOL}")
    fig.tight_layout()
    save_fig(fig, figdir, "3_multivariate_r2")

print(f"\nWrote tables to {paths.out}, figures to {figdir}")
print("Run `python b_progress/make_progress_md.py 2d_linear_encod` to build a markdown summary.")
