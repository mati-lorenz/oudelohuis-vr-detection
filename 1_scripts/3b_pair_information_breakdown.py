# -*- coding: utf-8 -*-
"""
3b_pair_information_breakdown
================================
The Pola, Thiele, Hoffmann & Panzeri (2003) "exact" information
breakdown, applied to pairs of simultaneously recorded cells: how much
of a pair's joint information about the stimulus is explained by each
cell's own tuning, how much is lost/gained from the two cells having
similar/different tuning, and how much comes from genuine trial-to-
trial correlated activity.

TERMINOLOGY AND EXACTNESS -- please read before interpreting the
output. Reconstructed from Pola et al. 2003 and Montani, Kohn, Smith &
Schultz (2007, J Neurosci 27:2338, which restates the same formalism in
modern notation) since the original paper's equations weren't
extractable as text; the core terms below are stated unambiguously in
both sources and cross-checked against this project's own validated
shuffle-core (infotheory/shuffling.py):

    I_total  : mutual information between the pair's JOINT response
               (r1, r2) and the stimulus -- the direct-method MI,
               computed from the actual joint response probabilities.
    I_lin    : the information if the two cells were fully independent
               -- literally I_1 + I_2, the sum of each cell's own
               single-cell information.
    I_ind    : information computed from the "independent" surrogate
               P_ind(r1,r2|s) = P(r1|s) P(r2|s) -- i.e. each cell's own
               tuning is preserved exactly, but trial-to-trial pairing
               between cells is scrambled WITHIN each stimulus value.
               This is EXACTLY this project's stim-conditioned shuffle
               (infotheory.shuffling, mode="stim") applied to cell 2's
               response and averaged over many shuffle realizations.
    I_sigsim = I_ind - I_lin : the "signal similarity" term -- how much
               having similar (I_sigsim<0, redundant) or different
               (I_sigsim>0, mildly synergistic) trial-averaged tuning
               curves alone changes the joint information, with NO
               correlated trial-to-trial variability at all.
    I_cor    = I_total - I_ind : everything attributable to the ACTUAL
               observed trial-to-trial correlation structure (both its
               average level and any stimulus-dependence).
    RSI      = I_total - I_lin = I_sigsim + I_cor  (Redundancy-Synergy
               Index; >0 synergy, <0 redundancy overall)

All of the above are EXACT (up to the usual finite-sample bias, which
is shuffle-corrected the same way as every other MI estimate in this
project). Pola et al. 2003 further splits I_cor into a stimulus-
INDEPENDENT component (I_corind, the average correlation level) and a
stimulus-DEPENDENT component (I_cordep, correlation whose STRENGTH
varies by stimulus) -- but that split needs an additional KL-divergence
correction term (called chi/Delta in the source papers) that isn't
reconstructable from the available text without risking a subtly wrong
formula. Rather than silently implement a guess, this script computes
an EXPLICITLY APPROXIMATE version (I_corind_approx, I_cordep_approx,
see build_avgcor_surrogate below) using an extra surrogate that
reattaches the OVERALL (stimulus-pooled) bivariate residual
distribution to each stimulus's own tuning -- conceptually the right
idea (removes stimulus-specific modulation of the residual correlation
while keeping its average strength/sign), cross-validated on synthetic
ground truth below, but NOT claimed to exactly match Pola's analytic
formula (the missing correction term is described as "small" in
Montani et al. 2007, so the approximation should be close).

Dataset, pairs, and the underlying shuffle machinery: identical to
3a_pair_activity_statistics.py (same session/trial/cell filters, same
sample_pairs_within_groups call) -- see that script's module docstring.
Response binning: R_BINS quantile bins per cell (2-3, matching Pola et
al.'s own recommendation for calcium/fluorescence data specifically --
see the MINT toolbox paper, Lorenz/Engel et al. 2025, Fig 2 legend:
"R = 2 equi-populated bins (appropriate for ... non-deconvolved calcium
fluorescent traces)").

Outputs
-------
2_pipeline/3b_pair_information_breakdown/
    out/    pair_information_breakdown.csv   one row per (session,
                                              window, area, label,
                                              cell_i, cell_j): I_total,
                                              I_lin, I_sigsim, I_cor,
                                              RSI, I_corind_approx,
                                              I_cordep_approx
            figures/*.png
    store/  breakdown.pkl   cached (reused unless --recompute)
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
from infotheory.pair_information import compute_pair_breakdown
from infotheory.plotting import set_style, save_fig, clear_figures

PROTOCOL = "DN"
WINDOWS = {"pre_stim": (-30.0, -10.0), "stim": (0.0, 20.0), "reward": (25.0, 45.0)}

MAX_PAIRS_PER_GROUP = 40   # reduced from 3a's 150 -- this script's per-pair cost is ~40x
                            # higher (many nested MI-shuffle calls: I_total, I_1, I_2, N_IND_REPS
                            # realizations of I_ind, N_IND_REPS more of the avgcor surrogate),
                            # so the pair budget had to come down to stay tractable
R_BINS = 3          # quantile bins per cell's response -- see module docstring
MI_SHUFFLES = 15     # bias-correction shuffles for every direct-method MI below (reduced from
                      # an initial 30 after timing this on real data)
N_IND_REPS = 8       # independent realizations of the stim-shuffled surrogate to average over,
                      # for a stable I_ind (and I_avgcor) estimate -- reduced from an initial 20
MIN_TRIALS_PER_STIM_BIN = 3
RANDOM_STATE = 0

N_JOBS_SESSIONS = -1
JOBLIB_VERBOSE = 5

parser = argparse.ArgumentParser()
parser.add_argument("--recompute", action="store_true",
                     help="Reload sessions from 0_data/ instead of reusing the cached "
                          "store/breakdown.pkl")
args = parser.parse_args()

paths = get_pipeline_paths(__file__)
figdir = paths.out / "figures"
clear_figures(figdir)
set_style()


# --------------------------------------------------------------------- #
# 1. Included sessions/cells -- same filters as 3a
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


# --------------------------------------------------------------------- #
# 2. Per-session information breakdown -- or reuse the cache
# --------------------------------------------------------------------- #
breakdown_cache = paths.store / "breakdown.pkl"
REQUIRED_COLUMNS = ["session_id", "window", "roi_name", "labeled_grouped", "cell_i", "cell_j",
                    "I_total", "I_lin", "I_sigsim", "I_cor", "RSI"]

breakdown_table = None
if not args.recompute and breakdown_cache.exists():
    try:
        candidate = pd.read_pickle(breakdown_cache)
        missing = [c for c in REQUIRED_COLUMNS if c not in candidate.columns]
        if missing:
            print(f"Cached breakdown is missing columns {missing} -- looks like it was built by "
                  f"an older version of this script. Recomputing from 0_data/ instead of using "
                  f"{breakdown_cache} (safe to delete that file to silence this check in the future).")
        else:
            breakdown_table = candidate
            print(f"Reusing cached breakdown from {paths.store} (pass --recompute to reload 0_data/)")
    except Exception as e:
        print(f"Could not load cached breakdown ({e}); recomputing from 0_data/ ...")

if breakdown_table is None:
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

                result = compute_pair_breakdown(x1, x2, sb, r_bins=R_BINS, mi_shuffles=MI_SHUFFLES,
                                                 n_ind_reps=N_IND_REPS, rng=rng_local)
                records.append({
                    "session_id": ses.session_id, "window": window_name, "roi_name": a,
                    "labeled_grouped": l, "cell_i": cell_i, "cell_j": cell_j, **result,
                })
        print(f"  [{ses.session_id}] information breakdown done")
        return records

    print(f"Computing information breakdown for {len(sessions)} sessions (n_jobs={N_JOBS_SESSIONS}) ...")
    nested = Parallel(n_jobs=N_JOBS_SESSIONS, backend="loky", verbose=JOBLIB_VERBOSE)(
        delayed(process_session)(ses, RANDOM_STATE + i) for i, ses in enumerate(sessions)
    )
    breakdown_table = pd.DataFrame([rec for group in nested for rec in group])
    breakdown_table.to_pickle(breakdown_cache)

breakdown_table.to_csv(paths.out / "pair_information_breakdown.csv", index=False)
print(f"\n{len(breakdown_table)} pairs analyzed")


# --------------------------------------------------------------------- #
# 3. Summary
# --------------------------------------------------------------------- #
if len(breakdown_table):
    print("\nMean I_total / I_lin / I_sigsim / I_cor / RSI by (window, area):")
    print(breakdown_table.groupby(["window", "roi_name"])
          [["I_total", "I_lin", "I_sigsim", "I_cor", "RSI"]].mean().round(4).to_string())
    n_synergy = (breakdown_table["RSI"] > 0).sum()
    print(f"\n{n_synergy}/{len(breakdown_table)} pairs show net synergy (RSI>0); "
          f"the rest show net redundancy.")


# --------------------------------------------------------------------- #
# 4. Figures
# --------------------------------------------------------------------- #
if len(breakdown_table):
    # 4a. RSI distribution by area
    fig, axes = plt.subplots(1, len(WINDOWS), figsize=(5 * len(WINDOWS), 4.5), squeeze=False)
    for ax, window_name in zip(axes[0], WINDOWS):
        sub = breakdown_table[breakdown_table["window"] == window_name]
        if sub.empty:
            continue
        area_order = [a for a in DEFAULT_AREA_ORDER if a in sub["roi_name"].unique()]
        sns.violinplot(data=sub, x="roi_name", y="RSI", order=area_order, hue="roi_name",
                        palette=AREA_COLORS, legend=False, ax=ax)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("")
        ax.set_title(window_name)
    fig.suptitle(f"Redundancy-Synergy Index by area (>0 synergy, <0 redundancy) -- {PROTOCOL}", y=1.03)
    fig.tight_layout()
    save_fig(fig, figdir, "1_rsi_by_area")

    # 4b. Component breakdown: mean I_lin, I_sigsim, I_cor stacked (summing to I_total), per window
    fig, axes = plt.subplots(1, len(WINDOWS), figsize=(5 * len(WINDOWS), 4.5), squeeze=False)
    for ax, window_name in zip(axes[0], WINDOWS):
        sub = breakdown_table[breakdown_table["window"] == window_name]
        if sub.empty:
            continue
        area_order = [a for a in DEFAULT_AREA_ORDER if a in sub["roi_name"].unique()]
        means = sub.groupby("roi_name")[["I_lin", "I_sigsim", "I_cor"]].mean().reindex(area_order)
        bottom = np.zeros(len(means))
        for comp, color in zip(["I_lin", "I_sigsim", "I_cor"], ["#4c72b0", "#dd8452", "#55a868"]):
            vals = means[comp].to_numpy()
            ax.bar(range(len(means)), vals, bottom=bottom, label=comp, color=color, edgecolor="black",
                   linewidth=0.3)
            bottom += vals
        ax.axhline(0, color="grey", linewidth=0.7)
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(means.index)
        ax.set_ylabel("bits")
        ax.set_title(window_name)
        if window_name == list(WINDOWS)[0]:
            ax.legend(fontsize=8, frameon=False)
    fig.suptitle(f"Information breakdown components (stack to I_total) -- {PROTOCOL}", y=1.03)
    fig.tight_layout()
    save_fig(fig, figdir, "2_component_breakdown")

    # 4c. Approximate stim-independent vs stim-dependent correlation split
    fig, axes = plt.subplots(1, len(WINDOWS), figsize=(5 * len(WINDOWS), 4.5), squeeze=False)
    for ax, window_name in zip(axes[0], WINDOWS):
        sub = breakdown_table[breakdown_table["window"] == window_name]
        if sub.empty:
            continue
        ax.scatter(sub["I_corind_approx"], sub["I_cordep_approx"], s=10, alpha=0.4, color="tab:green")
        ax.axhline(0, color="grey", linewidth=0.7)
        ax.axvline(0, color="grey", linewidth=0.7)
        ax.set_xlabel("I_corind (approx.)")
        ax.set_ylabel("I_cordep (approx.)")
        ax.set_title(window_name)
    fig.suptitle(f"Approximate split of I_cor -- {PROTOCOL} (see module docstring: approximate, not exact)",
                 y=1.03)
    fig.tight_layout()
    save_fig(fig, figdir, "3_corind_vs_cordep_approx")

print(f"\nWrote tables to {paths.out}, figures to {figdir}")
print("Run `python b_progress/make_progress_md.py 3b_pair_information_breakdown` to build a markdown summary.")
