# -*- coding: utf-8 -*-
"""
3c_pair_information_pid
==========================
Partial Information Decomposition (PID, Williams & Beer 2010) for
pairs of simultaneously recorded cells: splits the pair's joint
information about the stimulus into REDUNDANT (both cells carry it),
UNIQUE to each cell, and SYNERGISTIC (only available from the pair
jointly) components. See infotheory/pid.py's module docstring for the
underlying linear system and the redundancy measure this reduces to
computing.

REDUNDANCY MEASURE: I_min (Williams & Beer 2010) by design -- the
simplest PID redundancy measure, with a known tendency to sometimes
overstate redundancy relative to more recent measures. Set
REDUNDANCY_MEASURE below to switch; `infotheory.pid.REDUNDANCY_MEASURES`
is a registry precisely so a more robust measure (BROJA-2PID --
Bertschinger, Rauh, Olbrich, Jost & Ay 2014, robustly estimated by
Makkeh, Theis & Vicente 2018's cone-programming solver) can be dropped
in later as a one-line change once that package is installed -- see
infotheory/pid.py's `redundancy_broja_STUB` for the integration sketch
(not wired in yet: the package isn't on PyPI and needs the ECOS cone
solver). NOTHING else in this script needs to change to switch measures.

Validated against the two textbook PID cases (a fully-redundant "copy"
distribution and an XOR synergy gate) before being used here -- see
infotheory/pid.py's module docstring for the exact expected values and
confirmation this implementation reproduces them.

Dataset, pairs, response binning: identical to 3a/3b (same session/
trial/cell filters, same sample_pairs_within_groups call, same R_BINS
quantile-binning convention -- see 3b_pair_information_breakdown.py's
module docstring for why R_BINS=2-3 is the right choice for calcium
data specifically).

Outputs
-------
2_pipeline/3c_pair_information_pid/
    out/    pair_pid.csv    one row per (session, window, area, label,
                             cell_i, cell_j): I1, I2, I12, redundancy,
                             unique1, unique2, synergy (+ shuffle-
                             corrected redundancy and its p-value)
            figures/*.png
    store/  pid.pkl         cached (reused unless --recompute)
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
from infotheory.neural_encoding import load_cell_inclusion, build_session_tables
from infotheory.celldata_utils import (
    get_area_label, sample_pairs_within_groups, AREA_COLORS, LABEL_SPLIT_AREAS, DEFAULT_AREA_ORDER,
)
from infotheory.pid import pid_from_binned_responses, REDUNDANCY_MEASURES
from infotheory.plotting import set_style, save_fig, clear_figures

PROTOCOL = "DN"
WINDOWS = {"pre_stim": (-30.0, -10.0), "stim": (0.0, 20.0), "reward": (25.0, 45.0)}

REDUNDANCY_MEASURE = "imin"  # see module docstring -- change to "broja" once wired in
MAX_PAIRS_PER_GROUP = 100
R_BINS = 3
PID_SHUFFLES = 30    # bias-correction shuffles for the redundancy estimate (cheap here -- PID
                      # from an already-binned joint histogram is a handful of array ops, not
                      # an iterative estimator, so this can afford more than 3b's shuffle counts)
MIN_TRIALS_PER_STIM_BIN = 3
RANDOM_STATE = 0

N_JOBS_SESSIONS = 10
JOBLIB_VERBOSE = 5

parser = argparse.ArgumentParser()
parser.add_argument("--recompute", action="store_true",
                     help="Reload sessions from 0_data/ instead of reusing the cached "
                          "store/pid.pkl")
args = parser.parse_args()

paths = get_pipeline_paths(__file__)
figdir = paths.out / "figures"
clear_figures(figdir)
set_style()

if REDUNDANCY_MEASURE not in REDUNDANCY_MEASURES:
    raise SystemExit(f"Unknown REDUNDANCY_MEASURE {REDUNDANCY_MEASURE!r}; "
                      f"must be one of {list(REDUNDANCY_MEASURES)}")


# --------------------------------------------------------------------- #
# 1. Included sessions/cells -- same filters as 3a/3b
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
      f"redundancy measure = {REDUNDANCY_MEASURE!r}")


# --------------------------------------------------------------------- #
# 2. Per-session PID -- or reuse the cache
# --------------------------------------------------------------------- #
pid_cache = paths.store / "pid.pkl"
REQUIRED_COLUMNS = ["session_id", "window", "roi_name", "labeled_grouped", "cell_i", "cell_j",
                    "I1", "I2", "I12", "redundancy", "unique1", "unique2", "synergy"]

pid_table = None
if not args.recompute and pid_cache.exists():
    try:
        candidate = pd.read_pickle(pid_cache)
        missing = [c for c in REQUIRED_COLUMNS if c not in candidate.columns]
        if missing or candidate.attrs.get("redundancy_measure") != REDUNDANCY_MEASURE:
            print(f"Cached PID table is stale (missing columns, or a different redundancy "
                  f"measure than the current {REDUNDANCY_MEASURE!r}) -- recomputing from 0_data/ "
                  f"instead of using {pid_cache}.")
        else:
            pid_table = candidate
            print(f"Reusing cached PID table from {paths.store} (pass --recompute to reload 0_data/)")
    except Exception as e:
        print(f"Could not load cached PID table ({e}); recomputing from 0_data/ ...")

if pid_table is None:
    sessions = load_sessions(protocols=[PROTOCOL], load_behaviordata=True, load_videodata=True,
                              load_celldata=True, load_calciumdata=True, only_session_ids=included_ids)

    def process_session(ses, seed):
        rng_local = np.random.default_rng(seed)
        included_cells = included_cell_ids_by_session.get(ses.session_id)
        if not included_cells:
            return []
        celldata = ses.celldata[ses.celldata["cell_id"].isin(included_cells)].reset_index(drop=True)
        if celldata.empty:
            return []
        area, label, _ = get_area_label(celldata, label_split_areas=LABEL_SPLIT_AREAS)

        tables = build_session_tables(ses, mu_sigma_lookup, WINDOWS, included_cell_ids=included_cells)
        records = []
        for window_name, table in tables.items():
            if table.empty or "stim_bin" not in table.columns:
                continue
            valid_bins = table["stim_bin"].value_counts()
            valid_bins = valid_bins[valid_bins >= MIN_TRIALS_PER_STIM_BIN].index
            table = table[table["stim_bin"].isin(valid_bins)].reset_index(drop=True)
            if table["stim_bin"].nunique() < 2 or len(table) < 20:
                continue
            stim_bin = table["stim_bin"].to_numpy()

            pairs, pair_groups = sample_pairs_within_groups(area, label, max_pairs_per_group=MAX_PAIRS_PER_GROUP,
                                                              rng=rng_local)
            for (i, j), (a, l) in zip(pairs, pair_groups):
                cell_i, cell_j = celldata["cell_id"].iloc[i], celldata["cell_id"].iloc[j]
                if cell_i not in table.columns or cell_j not in table.columns:
                    continue
                x1 = table[cell_i].to_numpy()
                x2 = table[cell_j].to_numpy()
                valid = np.isfinite(x1) & np.isfinite(x2)
                if valid.sum() < 20:
                    continue
                x1, x2, sb = x1[valid], x2[valid], stim_bin[valid]
                if len(np.unique(sb)) < 2:
                    continue

                result = pid_from_binned_responses(x1, x2, sb, bins=R_BINS,
                                                    redundancy_measure=REDUNDANCY_MEASURE,
                                                    n_shuffles=PID_SHUFFLES, rng=rng_local)
                records.append({
                    "session_id": ses.session_id, "window": window_name, "roi_name": a,
                    "labeled_grouped": l, "cell_i": cell_i, "cell_j": cell_j, "n_trials": result.n,
                    "I1": result.I1, "I2": result.I2, "I12": result.I12,
                    "redundancy": result.redundancy, "unique1": result.unique1,
                    "unique2": result.unique2, "synergy": result.synergy,
                    "redundancy_p_value": result.redundancy_p_value,
                })
        print(f"  [{ses.session_id}] PID done")
        return records

    print(f"Computing PID for {len(sessions)} sessions (n_jobs={N_JOBS_SESSIONS}) ...")
    nested = Parallel(n_jobs=N_JOBS_SESSIONS, backend="loky", verbose=JOBLIB_VERBOSE)(
        delayed(process_session)(ses, RANDOM_STATE + i) for i, ses in enumerate(sessions)
    )
    pid_table = pd.DataFrame([rec for group in nested for rec in group])
    pid_table.attrs["redundancy_measure"] = REDUNDANCY_MEASURE
    pd.to_pickle(pid_table, pid_cache)

pid_table.to_csv(paths.out / "pair_pid.csv", index=False)
print(f"\n{len(pid_table)} pairs analyzed")


# --------------------------------------------------------------------- #
# 3. Summary
# --------------------------------------------------------------------- #
if len(pid_table):
    print("\nMean redundancy / unique1 / unique2 / synergy by (window, area):")
    print(pid_table.groupby(["window", "roi_name"])
          [["redundancy", "unique1", "unique2", "synergy"]].mean().round(4).to_string())

    frac_sig_redundant = (pid_table["redundancy_p_value"] < 0.05).mean()
    print(f"\n{frac_sig_redundant:.1%} of pairs show significant (p<0.05) redundancy beyond "
          f"the global-shuffle null.")


# --------------------------------------------------------------------- #
# 4. Figures
# --------------------------------------------------------------------- #
if len(pid_table):
    # 4a. Stacked I1/I2-independent decomposition: redundancy, unique1,
    #    unique2, synergy (sum to I12), per area
    fig, axes = plt.subplots(1, len(WINDOWS), figsize=(5 * len(WINDOWS), 4.5), squeeze=False)
    for ax, window_name in zip(axes[0], WINDOWS):
        sub = pid_table[pid_table["window"] == window_name]
        if sub.empty:
            continue
        area_order = [a for a in DEFAULT_AREA_ORDER if a in sub["roi_name"].unique()]
        means = sub.groupby("roi_name")[["redundancy", "unique1", "unique2", "synergy"]].mean().reindex(area_order)
        bottom = np.zeros(len(means))
        for comp, color in zip(["redundancy", "unique1", "unique2", "synergy"],
                                ["#4c72b0", "#dd8452", "#c44e52", "#55a868"]):
            vals = means[comp].to_numpy()
            ax.bar(range(len(means)), vals, bottom=bottom, label=comp, color=color,
                   edgecolor="black", linewidth=0.3)
            bottom += vals
        ax.axhline(0, color="grey", linewidth=0.7)
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(means.index)
        ax.set_ylabel("bits")
        ax.set_title(window_name)
        if window_name == list(WINDOWS)[0]:
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"PID components (stack to I12) -- {REDUNDANCY_MEASURE}, {PROTOCOL}", y=1.03)
    fig.tight_layout()
    save_fig(fig, figdir, "1_pid_components_by_area")

    # 4b. Redundancy vs synergy scatter, per pair
    fig, axes = plt.subplots(1, len(WINDOWS), figsize=(5 * len(WINDOWS), 5), squeeze=False)
    for ax, window_name in zip(axes[0], WINDOWS):
        sub = pid_table[pid_table["window"] == window_name]
        if sub.empty:
            continue
        for area in [a for a in DEFAULT_AREA_ORDER if a in sub["roi_name"].unique()]:
            asub = sub[sub["roi_name"] == area]
            ax.scatter(asub["redundancy"], asub["synergy"], s=10, alpha=0.4,
                       color=AREA_COLORS.get(area, "gray"), label=area)
        ax.axhline(0, color="grey", linewidth=0.7)
        ax.axvline(0, color="grey", linewidth=0.7)
        ax.set_xlabel("redundancy")
        ax.set_ylabel("synergy")
        ax.set_title(window_name)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"Redundancy vs. synergy, per pair -- {REDUNDANCY_MEASURE}, {PROTOCOL}", y=1.03)
    fig.tight_layout()
    save_fig(fig, figdir, "2_redundancy_vs_synergy")

print(f"\nWrote tables to {paths.out}, figures to {figdir}")
print("Run `python b_progress/make_progress_md.py 3c_pair_information_pid` to build a markdown summary.")
