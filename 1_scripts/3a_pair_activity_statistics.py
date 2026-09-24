# -*- coding: utf-8 -*-
"""
3a_pair_activity_statistics
==============================
Pairwise correlation statistics between simultaneously recorded cells
-- the "activity statistics" step of the pairwise/population analyses
(3a/3b/3c), preceding the information-theoretic Information Breakdown
(3b, Pola et al. 2003) and Partial Information Decomposition (3c) that
build on the same pairs and the same conditioned-shuffle machinery
introduced here.

For each sampled pair of cells (within an area/label group -- see
infotheory.celldata_utils.sample_pairs_within_groups), per trial window
(same pre_stim/stim/reward convention as the rest of this project):
    signal correlation : correlation between the two cells' trial-
                          averaged tuning curves across stim_bin --
                          how similarly the pair responds to stimulus
                          ON AVERAGE, ignoring trial-to-trial variability
    noise correlation   : correlation between the two cells' RESIDUALS
                          (activity minus that trial's stim_bin mean)
                          -- how similarly the pair's trial-to-trial
                          fluctuations co-vary AROUND their own tuning
    raw correlation      : plain trial-by-trial correlation, no
                          conditioning -- signal + noise combined

Also runs the new condition-conditioned shuffle test
(infotheory.shuffling.conditioned_shuffle_test) for each pair, in all
four modes:
    global      : is there ANY relationship at all (destroys everything,
                  including shared tuning)
    stim        : is there a relationship BEYOND shared stimulus tuning
                  (the shuffle-based analogue of the noise correlation
                  above -- computed independently, as a cross-check;
                  see this script's validation notes)
    choice      : is there a relationship BEYOND shared choice
    stim_choice : is there a relationship BEYOND both jointly

This demonstrates the shuffle-core framework end to end (built and
validated against known ground truth before this script was written --
see infotheory/shuffling.py's docstring and the paper it's based on,
Lorenz/Engel et al. 2025's MINT toolbox / Pola et al. 2003) and produces
the pairwise correlation tables 3b/3c will both need.

Dataset: same session/trial/cell filtering as 2c/2d/2e/2f (see
infotheory.neural_encoding.load_cell_inclusion) -- 1b's included
sessions, engaged trials, cells passing 2a's proximity filter AND 2b's
QC. Pairs are sampled WITHIN each (area, label) group only (never
across groups -- see sample_pairs_within_groups's own docstring for why).

Outputs
-------
2_pipeline/3a_pair_activity_statistics/
    out/    pair_correlations.csv   one row per (session, window, area,
                                      label, cell_i, cell_j): signal/
                                      noise/raw correlation, plus each
                                      shuffle mode's corrected statistic
                                      and p-value
            figures/*.png
    store/  pair_correlations.pkl   cached (reused unless --recompute)
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
from infotheory.celldata_utils import (
    get_area_label, sample_pairs_within_groups, AREA_COLORS, LABEL_SPLIT_AREAS, DEFAULT_AREA_ORDER,
)
from infotheory.shuffling import conditioned_shuffle_test, SHUFFLE_MODES
from infotheory.plotting import set_style, save_fig, clear_figures

PROTOCOL = "DN"
WINDOWS = {"pre_stim": (-30.0, -10.0), "stim": (0.0, 20.0), "reward": (25.0, 45.0)}

MAX_PAIRS_PER_GROUP = 150  # per (session, window, area, label) -- matches
                            # sample_pairs_within_groups's own default
N_SHUFFLES = 50  # reduced from an initial 200 after timing this on real data -- 4 shuffle modes x
                  # up to 150 pairs x 3 windows already means thousands of shuffle-tests per
                  # session; 200 shuffles each made a single session take several minutes
MIN_TRIALS_PER_STIM_BIN = 3  # need at least this many trials per stim_bin to treat that bin's
                              # mean as a real per-condition tuning value (for signal correlation
                              # and residualization), not noise from 1-2 trials
RANDOM_STATE = 0

N_JOBS_SESSIONS = 14
JOBLIB_VERBOSE = 5

parser = argparse.ArgumentParser()
parser.add_argument("--recompute", action="store_true",
                     help="Reload sessions from 0_data/ instead of reusing the cached "
                          "store/pair_correlations.pkl")
args = parser.parse_args()

paths = get_pipeline_paths(__file__)
figdir = paths.out / "figures"
clear_figures(figdir)
set_style()


# --------------------------------------------------------------------- #
# 1. Included sessions/cells -- same filters as 2c/2d/2e/2f
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


def corr_fn(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------- #
# 2. Per-session pairwise correlations -- or reuse the cache
# --------------------------------------------------------------------- #
pair_cache = paths.store / "pair_correlations.pkl"
REQUIRED_COLUMNS = ["session_id", "window", "roi_name", "labeled_grouped", "cell_i", "cell_j",
                    "signal_corr", "noise_corr", "raw_corr"] + [f"shuffle_{m}_p" for m in SHUFFLE_MODES]

pair_table = None
if not args.recompute and pair_cache.exists():
    try:
        candidate = pd.read_pickle(pair_cache)
        missing = [c for c in REQUIRED_COLUMNS if c not in candidate.columns]
        if missing:
            print(f"Cached pair correlations are missing columns {missing} -- looks like they were "
                  f"built by an older version of this script. Recomputing from 0_data/ instead of "
                  f"using {pair_cache} (safe to delete that file to silence this check in the future).")
        else:
            pair_table = candidate
            print(f"Reusing cached pair correlations from {paths.store} (pass --recompute to reload 0_data/)")
    except Exception as e:
        print(f"Could not load cached pair correlations ({e}); recomputing from 0_data/ ...")

if pair_table is None:
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
            # Only keep stim_bin values with enough trials to trust as a per-condition mean
            valid_bins = table["stim_bin"].value_counts()
            valid_bins = valid_bins[valid_bins >= MIN_TRIALS_PER_STIM_BIN].index
            table = table[table["stim_bin"].isin(valid_bins)]
            if table["stim_bin"].nunique() < 2 or len(table) < 20:
                continue

            stim_bin = table["stim_bin"].to_numpy()
            choice = table["choice"].to_numpy() if "choice" in table.columns else None
            stim_means = table.groupby("stim_bin")

            pairs, pair_groups = sample_pairs_within_groups(area, label, max_pairs_per_group=MAX_PAIRS_PER_GROUP,
                                                              rng=rng_local)
            for (i, j), (a, l) in zip(pairs, pair_groups):
                cell_i, cell_j = celldata["cell_id"].iloc[i], celldata["cell_id"].iloc[j]
                if cell_i not in table.columns or cell_j not in table.columns:
                    continue
                xi = table[cell_i].to_numpy()
                xj = table[cell_j].to_numpy()
                valid = np.isfinite(xi) & np.isfinite(xj)
                if valid.sum() < 20:
                    continue
                xi, xj, sb = xi[valid], xj[valid], stim_bin[valid]
                ch = choice[valid] if choice is not None else None

                tuning_i = table.loc[valid, cell_i].groupby(sb).transform("mean").to_numpy()
                tuning_j = table.loc[valid, cell_j].groupby(sb).transform("mean").to_numpy()
                bin_means_i = pd.Series(xi).groupby(sb).mean()
                bin_means_j = pd.Series(xj).groupby(sb).mean()
                signal_corr = corr_fn(bin_means_i.to_numpy(), bin_means_j.to_numpy())

                resid_i = xi - tuning_i
                resid_j = xj - tuning_j
                noise_corr = corr_fn(resid_i, resid_j)
                raw_corr = corr_fn(xi, xj)

                row = {"session_id": ses.session_id, "window": window_name, "roi_name": a,
                       "labeled_grouped": l, "cell_i": cell_i, "cell_j": cell_j,
                       "n_trials": len(xi), "signal_corr": signal_corr, "noise_corr": noise_corr,
                       "raw_corr": raw_corr}
                for mode in SHUFFLE_MODES:
                    res = conditioned_shuffle_test(corr_fn, xi, xj, mode=mode, stim=sb, choice=ch,
                                                    n_shuffles=N_SHUFFLES, rng=rng_local)
                    row[f"shuffle_{mode}_corrected"] = res.stat_corrected
                    row[f"shuffle_{mode}_p"] = res.p_value
                records.append(row)
        print(f"  [{ses.session_id}] pairwise correlations done")
        return records

    print(f"Computing pairwise correlations for {len(sessions)} sessions (n_jobs={N_JOBS_SESSIONS}) ...")
    nested = Parallel(n_jobs=N_JOBS_SESSIONS, backend="loky", verbose=JOBLIB_VERBOSE)(
        delayed(process_session)(ses, RANDOM_STATE + i) for i, ses in enumerate(sessions)
    )
    pair_table = pd.DataFrame([rec for group in nested for rec in group])
    pair_table.to_pickle(pair_cache)

pair_table.to_csv(paths.out / "pair_correlations.csv", index=False)
print(f"\n{len(pair_table)} pairs analyzed")


# --------------------------------------------------------------------- #
# 3. Summary prints
# --------------------------------------------------------------------- #
if len(pair_table):
    print("\nMean signal/noise/raw correlation by (window, area):")
    print(pair_table.groupby(["window", "roi_name"])[["signal_corr", "noise_corr", "raw_corr"]]
          .mean().round(3).to_string())

    print("\nFraction of pairs with a significant (p<0.05) relationship, by shuffle mode and window "
          "(global should be the largest -- it's the least conservative test; stim_choice the "
          "smallest -- it controls for the most):")
    sig_frac = pd.DataFrame({
        mode: pair_table.groupby("window").apply(lambda d, m=mode: (d[f"shuffle_{m}_p"] < 0.05).mean())
        for mode in SHUFFLE_MODES})
    print(sig_frac.round(3).to_string())


# --------------------------------------------------------------------- #
# 4. Figures
# --------------------------------------------------------------------- #
if len(pair_table):
    # 4a. Signal vs noise correlation scatter, one panel per window
    fig, axes = plt.subplots(1, len(WINDOWS), figsize=(5 * len(WINDOWS), 5), squeeze=False)
    for ax, window_name in zip(axes[0], WINDOWS):
        sub = pair_table[pair_table["window"] == window_name].dropna(subset=["signal_corr", "noise_corr"])
        for area in [a for a in DEFAULT_AREA_ORDER if a in sub["roi_name"].unique()]:
            asub = sub[sub["roi_name"] == area]
            ax.scatter(asub["signal_corr"], asub["noise_corr"], s=10, alpha=0.4,
                       color=AREA_COLORS.get(area, "gray"), label=area)
        ax.axhline(0, color="grey", linewidth=0.7)
        ax.axvline(0, color="grey", linewidth=0.7)
        ax.set_xlabel("signal correlation")
        ax.set_ylabel("noise correlation")
        ax.set_title(window_name)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"Signal vs. noise correlation, per pair -- {PROTOCOL}", y=1.03)
    fig.tight_layout()
    save_fig(fig, figdir, "1_signal_vs_noise_correlation")

    # 4b. Noise correlation distribution by (area, label)
    fig, axes = plt.subplots(1, len(WINDOWS), figsize=(5 * len(WINDOWS), 4.5), squeeze=False)
    for ax, window_name in zip(axes[0], WINDOWS):
        sub = pair_table[pair_table["window"] == window_name].dropna(subset=["noise_corr"])
        if sub.empty:
            continue
        area_order = [a for a in DEFAULT_AREA_ORDER if a in sub["roi_name"].unique()]
        sns.violinplot(data=sub, x="roi_name", y="noise_corr", order=area_order, hue="roi_name",
                        palette=AREA_COLORS, legend=False, ax=ax)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("")
        ax.set_ylabel("noise correlation")
        ax.set_title(window_name)
    fig.suptitle(f"Noise correlation by area -- {PROTOCOL}", y=1.03)
    fig.tight_layout()
    save_fig(fig, figdir, "2_noise_correlation_by_area")

    # 4c. What survives each shuffle mode -- shows how much of the raw
    #     correlation each conditioning level "explains away", the
    #     direct visual of the shuffle-core's decomposition.
    fig, axes = plt.subplots(1, len(WINDOWS), figsize=(5 * len(WINDOWS), 4.5), squeeze=False)
    for ax, window_name in zip(axes[0], WINDOWS):
        sub = pair_table[pair_table["window"] == window_name]
        if sub.empty:
            continue
        vals = [sub[f"shuffle_{m}_corrected"].dropna() for m in SHUFFLE_MODES]
        ax.boxplot(vals, tick_labels=SHUFFLE_MODES, showfliers=False)
        ax.axhline(0, color="grey", linewidth=0.7)
        ax.set_ylabel("shuffle-corrected statistic (correlation)")
        ax.set_title(window_name)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle(f"What survives each conditioned shuffle -- {PROTOCOL}", y=1.03)
    fig.tight_layout()
    save_fig(fig, figdir, "3_shuffle_mode_comparison")

print(f"\nWrote tables to {paths.out}, figures to {figdir}")
print("Run `python b_progress/make_progress_md.py 3a_pair_activity_statistics` to build a markdown summary.")
