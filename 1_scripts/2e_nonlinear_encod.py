# -*- coding: utf-8 -*-
"""
2e_nonlinear_encod
=====================
For cell-predictor relationships where 2d_linear_encod.py found a big
gap between the total mutual information and what a linear fit
captures, refit using a polynomial (degree 2 and 3) to see whether
allowing curvature closes the gap.

Only refits CONTINUOUS predictors (cur_runspeed, cur_pupil_area,
cur_motionenergy, cur_lick, stim_bin) -- NOT choice/correct. Those are
strictly binary (0/1), and a plain linear fit on a binary predictor
already recovers the exact conditional mean in each of the only two
possible groups, which is the most ANY function of that variable could
ever explain about the MEAN relationship -- there is no curvature left
for a polynomial to find. If 2c/2d showed a linear-vs-MI gap for
choice/correct anyway, a polynomial refit confirming zero improvement
is not a null result -- it's a positive confirmation that the gap
reflects something other than curvature in the mean (see 2c_information.py's
module docstring: choice-dependent running speed changes trial-window
sample sizes, which MI can pick up as a genuine but non-encoding-related
statistical dependency; 2e can't fix that, only a linear/polynomial fit
of a DIFFERENT variable, or controlling for it directly, could).

"Big gap" selection (which cell-predictor-window combinations actually
get refit here): mi_p_value < 0.05 (a real relationship, not noise) AND
frac_explained_linearly < NONLINEAR_GAP_THR (2d's own linear-sufficiency
threshold) -- reusing 2d's own significance/threshold logic rather than
introducing a third, inconsistent one.

Same dataset assembly as 2c/2d (see infotheory.neural_encoding) --
independently rebuilds the per-session tables (Rule #1) and reads ONLY
2d's cell_predictor_mi_vs_linear.csv to decide what to refit.

Outputs
-------
2_pipeline/2e_nonlinear_encod/
    out/    nonlinear_refits.csv   one row per refit candidate: linear
                                    r2, polynomial (deg 2, deg 3) r2,
                                    MI reference, whether the polynomial
                                    closes most of the remaining gap
            figures/*.png
    store/  refits.pkl              cached table (reused unless --recompute)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from infotheory.pipeline import get_pipeline_paths
from infotheory.session import load_sessions
from infotheory.neural_encoding import load_cell_inclusion, build_session_tables
from infotheory.info_theory import multivariate_linear_fit, compare_mi_to_linear
from infotheory.plotting import set_style, save_fig, clear_figures

PROTOCOL = "DN"
WINDOWS = {"pre_stim": (-30.0, -10.0), "stim": (0.0, 20.0), "reward": (25.0, 45.0)}
CONTINUOUS_PREDICTORS = ["cur_runspeed", "cur_pupil_area", "cur_motionenergy", "cur_lick", "stim_bin"]
NONLINEAR_GAP_THR = 0.8   # same threshold 2d_linear_encod.py uses for "linear_sufficient"
MAX_POLY_DEGREE = 3
RANDOM_STATE = 0

N_JOBS_SESSIONS = 4
N_JOBS_GROUPS = 2
JOBLIB_VERBOSE = 5

parser = argparse.ArgumentParser()
parser.add_argument("--recompute", action="store_true",
                     help="Reload sessions from 0_data/ instead of reusing the cached "
                          "store/refits.pkl")
args = parser.parse_args()

paths = get_pipeline_paths(__file__)
figdir = paths.out / "figures"
clear_figures(figdir)
set_style()


# --------------------------------------------------------------------- #
# 1. Read 2d's comparison table, select refit candidates
# --------------------------------------------------------------------- #
linear_out = paths.out_from("2d_linear_encod")
comparison_path = linear_out / "cell_predictor_mi_vs_linear.csv"
if not comparison_path.exists():
    raise SystemExit(f"Run 2d_linear_encod.py first -- expected {comparison_path}")

comparison = pd.read_csv(comparison_path)
candidates = comparison[
    (comparison["predictor"].isin(CONTINUOUS_PREDICTORS))
    & (comparison["mi_p_value"] < 0.05)
    & (comparison["frac_explained_linearly"] < NONLINEAR_GAP_THR)
].copy()
print(f"{len(candidates)}/{len(comparison)} (cell, predictor, window) combinations are refit candidates: "
      f"significant MI (p<0.05), continuous predictor, and linear captures <{NONLINEAR_GAP_THR:.0%} of it")

if candidates.empty:
    print("\nNothing to refit -- either every significant relationship is already well captured "
          "linearly, or none of the continuous-predictor relationships were significant. "
          "No nonlinear analysis needed for this dataset/threshold.")
    print(f"\nWrote nothing new; see {linear_out} for the underlying comparison.")
    raise SystemExit(0)


# --------------------------------------------------------------------- #
# 2. Included sessions/cells (same as 2c/2d) -- only load what's needed
#    to rebuild tables for the sessions candidates actually came from.
# --------------------------------------------------------------------- #
fits_path = paths.out_from("1b_psychometric") / "psychometric_fits.csv"
fits_df = pd.read_csv(fits_path)
mu_sigma_lookup = fits_df.set_index("session_id")[["mu", "sigma"]].to_dict("index")

cell_dist_out = paths.out_from("2a_cell_distribution")
activity_out = paths.out_from("2b_activity_statistics")
cell_inclusion = load_cell_inclusion(cell_dist_out, activity_out)
included_cell_ids_by_session = {
    sid: set(sub.loc[sub["include"], "cell_id"]) for sid, sub in cell_inclusion.groupby("session_id")
}
candidate_session_ids = sorted(candidates["session_id"].unique())


# --------------------------------------------------------------------- #
# 3. Refit -- or reuse the cache
# --------------------------------------------------------------------- #
refit_cache = paths.store / "refits.pkl"
REQUIRED_COLUMNS = ["session_id", "window", "cell_id", "predictor", "r2_linear", "r2_poly2", "r2_poly3"]

refit_table = None
if not args.recompute and refit_cache.exists():
    try:
        candidate_cache = pd.read_pickle(refit_cache)
        missing = [c for c in REQUIRED_COLUMNS if c not in candidate_cache.columns]
        if missing or set(candidate_cache["session_id"]) != set(candidate_session_ids):
            print("Cached refits are stale (missing columns or a different candidate-session set) "
                  "-- recomputing from 0_data/.")
        else:
            refit_table = candidate_cache
            print(f"Reusing cached refits from {paths.store} (pass --recompute to reload 0_data/)")
    except Exception as e:
        print(f"Could not load cached refits ({e}); recomputing from 0_data/ ...")

if refit_table is None:
    sessions = load_sessions(protocols=[PROTOCOL], load_behaviordata=True, load_videodata=True,
                              load_celldata=True, load_calciumdata=True,
                              only_session_ids=candidate_session_ids)

    def build_one_session(ses):
        included_cells = included_cell_ids_by_session.get(ses.session_id)
        if not included_cells:
            return {}
        return build_session_tables(ses, mu_sigma_lookup, WINDOWS, included_cell_ids=included_cells)

    print(f"Rebuilding tables for {len(sessions)} candidate sessions (n_jobs={N_JOBS_SESSIONS}) ...")
    session_tables = Parallel(n_jobs=N_JOBS_SESSIONS, backend="loky", verbose=JOBLIB_VERBOSE)(
        delayed(build_one_session)(ses) for ses in sessions
    )
    tables_by_session_window = {}
    for ses, tables in zip(sessions, session_tables):
        for window_name, table in tables.items():
            tables_by_session_window[(ses.session_id, window_name)] = table

    def refit_one(row):
        table = tables_by_session_window.get((row["session_id"], row["window"]))
        if table is None or row["predictor"] not in table.columns or row["cell_id"] not in table.columns:
            return None
        pair_data = table[[row["predictor"], row["cell_id"]]].dropna()
        if len(pair_data) < 15:
            return None
        x = pair_data[row["predictor"]].to_numpy()
        y = pair_data[row["cell_id"]].to_numpy()
        x_std = x.std()
        if x_std == 0:
            return None
        x_z = (x - x.mean()) / x_std  # standardize before building powers, so x^2/x^3 don't
                                        # numerically dwarf x^1 and destabilize the fit

        results = {"session_id": row["session_id"], "window": row["window"], "cell_id": row["cell_id"],
                   "predictor": row["predictor"], "n_trials": len(pair_data), "r2_linear": row["r2"],
                   "mi_reference_bits": row["mi_reference_bits"]}
        design = x_z[:, None]
        for degree in range(2, MAX_POLY_DEGREE + 1):
            design = np.column_stack([design, x_z ** degree])
            mv = multivariate_linear_fit(design, y, predictor_names=[f"x^{d}" for d in range(1, degree + 1)])
            results[f"r2_poly{degree}"] = mv.adj_r2  # adjusted, not raw -- raw R^2 is mathematically
                                                        # guaranteed to be >= the linear fit's just from
                                                        # having more free parameters (2-3 vs 1), even
                                                        # under pure noise; adjusted R^2 penalizes for
                                                        # that so "improvement" reflects real curvature
        return results

    print(f"Refitting {len(candidates)} candidates with polynomial regression (n_jobs={N_JOBS_GROUPS}) ...")
    refit_rows = Parallel(n_jobs=N_JOBS_GROUPS, backend="loky", verbose=JOBLIB_VERBOSE)(
        delayed(refit_one)(row) for _, row in candidates.iterrows()
    )
    refit_table = pd.DataFrame([r for r in refit_rows if r is not None])
    refit_table.to_pickle(refit_cache)

if refit_table.empty:
    print("\nNo candidate could actually be refit (too few valid trials or a constant predictor "
          "after filtering) -- nothing further to report.")
    raise SystemExit(0)

best_degree_col = f"r2_poly{MAX_POLY_DEGREE}"
refit_table["mi_bits_from_poly"] = [
    compare_mi_to_linear(row["mi_reference_bits"], row[best_degree_col]).mi_from_r2_bits
    for _, row in refit_table.iterrows()
]
refit_table["frac_explained_poly"] = [
    compare_mi_to_linear(row["mi_reference_bits"], row[best_degree_col]).frac_explained_linearly
    for _, row in refit_table.iterrows()
]
refit_table["poly_improvement"] = refit_table[best_degree_col] - refit_table["r2_linear"]
refit_table.to_csv(paths.out / "nonlinear_refits.csv", index=False)

print(f"\n{len(refit_table)} candidates refit. Polynomial (degree {MAX_POLY_DEGREE}) vs linear R\u00b2:")
print(f"  median improvement: {refit_table['poly_improvement'].median():.4f}")
print(f"  candidates where polynomial reaches >={NONLINEAR_GAP_THR:.0%} of MI "
      f"(gap closed): {(refit_table['frac_explained_poly'] >= NONLINEAR_GAP_THR).sum()}/{len(refit_table)}")
print("\nBy predictor, median improvement (r2_poly - r2_linear):")
print(refit_table.groupby("predictor")["poly_improvement"].median().sort_values(ascending=False).to_string())


# --------------------------------------------------------------------- #
# 4. Figures
# --------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(refit_table["r2_linear"], refit_table[best_degree_col], s=15, alpha=0.5, color="tab:purple")
lo = min(0, refit_table["r2_linear"].min(), refit_table[best_degree_col].min())
hi = max(refit_table["r2_linear"].max(), refit_table[best_degree_col].max()) * 1.1 + 1e-6
lims = [lo, hi]
ax.plot(lims, lims, color="grey", linestyle="--", linewidth=1)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("linear r\u00b2")
ax.set_ylabel(f"polynomial (degree {MAX_POLY_DEGREE}) r\u00b2")
ax.set_title(f"Does allowing curvature help? -- {len(refit_table)} candidates, {PROTOCOL}")
fig.tight_layout()
save_fig(fig, figdir, "1_linear_vs_polynomial_r2")

fig, ax = plt.subplots(figsize=(7, 5))
order = refit_table.groupby("predictor")["poly_improvement"].median().sort_values(ascending=False).index
ax.boxplot([refit_table.loc[refit_table["predictor"] == p, "poly_improvement"] for p in order],
           tick_labels=order, showfliers=False)
ax.axhline(0, color="grey", linewidth=0.8)
ax.set_ylabel("r2_poly - r2_linear")
ax.set_title(f"Improvement from allowing curvature, by predictor -- {PROTOCOL}")
fig.tight_layout()
save_fig(fig, figdir, "2_improvement_by_predictor")

print(f"\nWrote tables to {paths.out}, figures to {figdir}")
print("Run `python b_progress/make_progress_md.py 2e_nonlinear_encod` to build a markdown summary.")
