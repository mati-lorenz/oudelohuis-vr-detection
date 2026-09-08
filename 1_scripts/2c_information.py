# -*- coding: utf-8 -*-
"""
2c_information
================
Mutual information between each recorded cell's trial-by-trial activity
and (a) ongoing behavioral state (running speed, pupil area, motion
energy, lick rate) and (b) task/performance variables (stimulus
strength, choice, correctness) -- DN sessions exclusively (the only
protocol with cell recordings).

Dataset: sessions filtered to 1b_psychometric's inclusion criteria,
trials restricted to engaged==1, cells restricted to those passing
BOTH 2a_cell_distribution's proximity filter and 2b_activity_statistics's
QC (see infotheory.neural_encoding.load_cell_inclusion) -- i.e. every
standard this project has established gets applied here, for the first
time actually excluding anything (2a/2b compute and report their
filters but never apply them; this is the step that finally uses them).

Response windows mirror 1c_behavior.py's MI_WINDOWS (pre_stim/stim/
reward, positions in cm relative to stimStart) -- a cell's activity is
averaged in each window per trial, exactly like the behavioral
per-trial features elsewhere in this project (see infotheory.
neural_encoding.build_session_tables, which does both the behavioral
AND neural extraction consistently in one place).

Behavioral predictors are taken from the SAME window as the neural
response (not restricted to pre-stimulus, unlike 1d_performance_
predictors.py) -- relating a cell's response to same-epoch behavior
(e.g. pupil size during the response window) is a standard, non-
circular encoding question, unlike 1d's performance-PREDICTION setup
where using post-stimulus behavior to predict the response itself
would be circular.

MI estimator: histogram-based (mutual_information_shuffle), not the
KSG estimator 1c_behavior.py uses for its (much smaller) behavior-only
pairwise table. This is a deliberate compute-budget choice: with
potentially hundreds to thousands of cells x ~7 predictors x 3 windows
per session, KSG's O(n log n) k-d tree search per pair becomes
prohibitive at this scale, while histogram MI (with this project's
`quantile_bin_edges`, robust to the low-cardinality choice/correct
predictors) is fast enough -- known to somewhat UNDERESTIMATE MI for
strong smooth relationships (see info_theory.py's docstrings), which
2d_linear_encod.py's mi_from_r2 comparison is aware of and accounts for.

IMPORTANT CAVEAT (found while validating this script -- read before
trusting choice/correct/stim_bin results): position-locked trial
windows give each trial a DIFFERENT NUMBER OF FRAMES depending on how
fast the animal moved through that window, and running speed is itself
choice-dependent (slower on trials where the animal licks -- an
anticipatory slowdown established throughout this project's earlier
steps). Slower trials average over more frames, giving a systematically
MORE PRECISE (lower-variance) per-trial activity estimate -- verified
directly on this project's own synthetic data: choice==1 trials showed
~20% lower within-group variance than choice==0 for the large majority
of cells (92% of 100 cells checked), with ZERO deliberate choice-
activity relationship in the generator. Histogram MI correctly detects
this variance difference as "information" (MI captures ANY statistical
dependency, not just mean shifts) -- it is real information in a
narrow technical sense, but it reflects measurement PRECISION varying
with choice, not neural encoding of choice. This inflated a real
validation run's "fraction of cells significantly encoding choice" to
~40% in the stim window with no genuine relationship present. Before
trusting a choice/correct finding here: check whether cur_runspeed
itself shows similarly elevated significance in the same window (a
sign the same confound is at play), and treat results in windows where
choice-dependent running-speed differences are largest (per 1c_behavior.py's
PSTH plots) with the most caution.

Outputs
-------
2_pipeline/2c_information/
    out/    cell_predictor_mi.csv   one row per (session, window, cell,
                                     predictor): MI raw/corrected, p-value
            figures/*.png
    store/  mi_combined.pkl         cached full table (reused unless
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
from infotheory.info_theory import mutual_information_shuffle
from infotheory.celldata_utils import AREA_COLORS, LABEL_SPLIT_AREAS, get_area_label, ordered_groups
from infotheory.plotting import set_style, save_fig, clear_figures

PROTOCOL = "DN"

# Response windows: [start_cm, end_cm] relative to stimStart (a position
# -- see psth.py's module docstring), matching 1c_behavior.py's
# MI_WINDOWS exactly, for direct comparability with the behavioral
# analysis.
WINDOWS = {"pre_stim": (-30.0, -10.0), "stim": (0.0, 20.0), "reward": (25.0, 45.0)}

PREDICTORS = [f"cur_{v}" for v in BEHAVIOR_PREDICTORS] + ["stim_bin", "choice", "correct"]

MI_BINS = 12             # fewer than 1c_behavior.py's 20 -- per-session trial counts here
                          # (one session at a time, not pooled) are typically smaller
N_SHUFFLES = 20           # reduced from 1d's 100 for compute-budget reasons (see module docstring)
MAX_CELLS_PER_SESSION = None  # optional subsampling cap per session; None = every included cell
RANDOM_STATE = 0

N_JOBS_SESSIONS = 4
N_JOBS_GROUPS = 2        # parallelizes the (session, window) MI computation loop
JOBLIB_VERBOSE = 5

parser = argparse.ArgumentParser()
parser.add_argument("--recompute", action="store_true",
                     help="Reload sessions from 0_data/ instead of reusing the cached "
                          "store/mi_combined.pkl")
args = parser.parse_args()

paths = get_pipeline_paths(__file__)
figdir = paths.out / "figures"
clear_figures(figdir)
set_style()


# --------------------------------------------------------------------- #
# 1. Included sessions (1b), mu/sigma lookup, cell inclusion (2a+2b)
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
if not (cell_dist_out / "celldata_combined.csv").exists():
    raise SystemExit(f"Run 2a_cell_distribution.py first -- expected {cell_dist_out / 'celldata_combined.csv'}")
if not (activity_out / "qc_per_cell.csv").exists():
    raise SystemExit(f"Run 2b_activity_statistics.py first -- expected {activity_out / 'qc_per_cell.csv'}")
cell_inclusion = load_cell_inclusion(cell_dist_out, activity_out)
included_cell_ids_by_session = {
    sid: set(sub.loc[sub["include"], "cell_id"]) for sid, sub in cell_inclusion.groupby("session_id")
}
n_cells_included = int(cell_inclusion["include"].sum())
print(f"Using {len(included_ids)} included {PROTOCOL} sessions, {n_cells_included}/{len(cell_inclusion)} "
      f"cells passing both the proximity filter (2a) and QC (2b)")


# --------------------------------------------------------------------- #
# 2. Build per-session, per-window (predictor + neural response) tables
#    -- or reuse the cache.
# --------------------------------------------------------------------- #
combined_cache = paths.store / "mi_combined.pkl"
REQUIRED_COLUMNS = ["session_id", "window", "cell_id", "predictor", "mi_bits_corrected", "mi_p_value"]

mi_table = None
if not args.recompute and combined_cache.exists():
    try:
        candidate = pd.read_pickle(combined_cache)
        missing = [c for c in REQUIRED_COLUMNS if c not in candidate.columns]
        if missing:
            print(f"Cached MI table is missing columns {missing} -- looks like it was built by an "
                  f"older version of this script. Recomputing from 0_data/ instead of using "
                  f"{combined_cache} (safe to delete that file to silence this check in the future).")
        else:
            mi_table = candidate
            print(f"Reusing cached MI table from {paths.store} (pass --recompute to reload 0_data/)")
    except Exception as e:
        print(f"Could not load cached MI table ({e}); recomputing from 0_data/ ...")

if mi_table is None:
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

    groups = []  # (session_id, window_name, table, cell_ids) -- flattened for parallel MI computation
    known_non_cell_cols = {"trialNumber", "stim_bin", "choice", "correct"} | {
        f"cur_{v}" for v in BEHAVIOR_PREDICTORS}
    for ses, tables in zip(sessions, session_tables):
        for window_name, table in tables.items():
            if table.empty:
                continue
            cell_cols = [c for c in table.columns if c not in known_non_cell_cols]
            groups.append((ses.session_id, window_name, table, cell_cols))

    def compute_mi_for_group(session_id, window_name, table, cell_cols, seed):
        rng_local = np.random.default_rng(seed)
        records = []
        for cell_id in cell_cols:
            for predictor in PREDICTORS:
                if predictor not in table.columns:
                    continue
                pair_data = table[[predictor, cell_id]].dropna()
                if len(pair_data) < 10:
                    continue
                mi = mutual_information_shuffle(pair_data[predictor], pair_data[cell_id], bins=MI_BINS,
                                                 n_shuffles=N_SHUFFLES, rng=rng_local)
                records.append({
                    "session_id": session_id, "window": window_name, "cell_id": cell_id,
                    "predictor": predictor, "n_trials": mi.n,
                    "mi_bits_raw": mi.bits_raw, "mi_bits_shuffle_mean": mi.bits_shuffle_mean,
                    "mi_bits_corrected": mi.bits_corrected, "mi_p_value": mi.p_value,
                })
        return records

    print(f"Computing MI for {len(groups)} (session, window) groups "
          f"(n_jobs={N_JOBS_GROUPS}) ...")
    nested = Parallel(n_jobs=N_JOBS_GROUPS, backend="loky", verbose=JOBLIB_VERBOSE)(
        delayed(compute_mi_for_group)(sid, wname, table, cell_cols, seed=1000 + i)
        for i, (sid, wname, table, cell_cols) in enumerate(groups)
    )
    mi_table = pd.DataFrame([rec for group in nested for rec in group])
    mi_table.to_pickle(combined_cache)

mi_table.to_csv(paths.out / "cell_predictor_mi.csv", index=False)
print(f"\n{len(mi_table)} (session, window, cell, predictor) MI values computed")


# --------------------------------------------------------------------- #
# 3. Attach area/label (for grouped plots) from 2a's celldata
# --------------------------------------------------------------------- #
celldata_combined = pd.read_csv(cell_dist_out / "celldata_combined.csv")[
    ["session_id", "cell_id", "roi_name", "labeled"]]
mi_table = mi_table.merge(celldata_combined, on=["session_id", "cell_id"], how="left")
_, mi_table["labeled_grouped"], _ = get_area_label(mi_table, label_split_areas=LABEL_SPLIT_AREAS)

if len(mi_table):
    print("\nStrongest (predictor, window) combinations by fraction of significant cells (p<0.05):")
    sig_frac = (mi_table.assign(sig=mi_table["mi_p_value"] < 0.05)
                .groupby(["window", "predictor"])["sig"].mean().sort_values(ascending=False))
    print(sig_frac.head(10).to_string())

    # Confound check (see module docstring): if choice/correct look
    # "significant" mainly because choice-dependent running speed
    # changes how many frames land in each trial's window (not because
    # of genuine encoding), cur_runspeed itself should show similarly
    # elevated significance in the SAME window -- print both side by
    # side so this is visible without digging through the CSV.
    print("\nConfound check: cur_runspeed's own significant fraction, vs choice/correct, per window "
          "(similar magnitudes suggest the speed-dependent sampling-precision confound described in "
          "this script's module docstring, not necessarily genuine choice/correct encoding):")
    compare_cols = [p for p in ["cur_runspeed", "choice", "correct"] if p in sig_frac.index.get_level_values(1)]
    if compare_cols:
        print(sig_frac.loc[(slice(None), compare_cols)].unstack("predictor").to_string())


# --------------------------------------------------------------------- #
# 4. Figure: MI distribution per predictor, per window, split by area/
#    label (color=area, x=predictor within each window panel)
# --------------------------------------------------------------------- #
if len(mi_table):
    for window_name in WINDOWS:
        sub = mi_table[mi_table["window"] == window_name]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(PREDICTORS)), 5))
        sns.boxplot(data=sub, x="predictor", y="mi_bits_corrected", hue="roi_name",
                    palette=AREA_COLORS, order=[p for p in PREDICTORS if p in sub["predictor"].unique()],
                    ax=ax, showfliers=False)
        ax.set_ylabel("MI (bits, bias-corrected)")
        ax.set_xlabel("")
        ax.set_title(f"Per-cell MI with each predictor -- {window_name} window ({PROTOCOL})")
        ax.legend(fontsize=8, title="area", frameon=False)
        fig.tight_layout()
        save_fig(fig, figdir, f"1_mi_by_predictor_{window_name}")

    # Fraction of cells with significant MI (p<0.05), per predictor/window
    frac_sig = (mi_table.assign(sig=mi_table["mi_p_value"] < 0.05)
                .groupby(["window", "predictor"])["sig"].mean().reset_index())
    window_order = list(WINDOWS.keys())
    pred_order = [p for p in PREDICTORS if p in frac_sig["predictor"].unique()]
    pivot = frac_sig.pivot_table(index="predictor", columns="window", values="sig").reindex(
        index=pred_order, columns=[w for w in window_order if w in frac_sig["window"].unique()])
    fig, ax = plt.subplots(figsize=(6, max(3, 0.5 * len(pivot))))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.to_numpy()[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.0%}", ha="center", va="center",
                        color="white" if val < 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="fraction of cells with p<0.05")
    ax.set_title(f"Fraction of cells significantly encoding each predictor -- {PROTOCOL}")
    fig.tight_layout()
    save_fig(fig, figdir, "2_frac_significant_heatmap")

print(f"\nWrote tables to {paths.out}, figures to {figdir}")
print("Run `python b_progress/make_progress_md.py 2c_information` to build a markdown summary.")
