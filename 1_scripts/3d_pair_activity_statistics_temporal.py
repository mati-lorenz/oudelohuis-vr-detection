# -*- coding: utf-8 -*-
"""
3d_pair_activity_statistics_temporal
========================================
Position-resolved version of 3a_pair_activity_statistics.py: signal/
noise/raw correlation for each sampled cell pair, computed at every
bin of a (trials, cells, position bins) tensor instead of once per the
three coarse pre_stim/stim/reward windows -- so a relationship that
only shows up transiently isn't averaged away.

TENSOR CONSTRUCTION: matches the real project's own convention
(tensor_utils.compute_tensor_for_session's "binmean" tensor, K trials x
N neurons x T bins) -- each (trial, position bin) entry is that
trial's OWN mean activity in that narrow bin, built here by reusing
infotheory.psth.build_position_binned_trial_tensor (itself a thin
wrapper around align_trials_position, already validated via
1c_behavior.py's position PSTHs). This is DIFFERENT from
2f_lag_information.py's approach (pooling raw frames across trials at
each position) -- that script needed raw frames specifically to search
across temporal lags; this one is asking a different question (does
the pairwise relationship itself vary with position) and one value per
trial per bin is both the standard convention for that question and
much cheaper.

MEMORY: this (and 3e/3f) is the heaviest kind of script in this
project to run -- the position-binned tensor construction scales with
(trials x bins x cells), and the naive version of this script (an
earlier version of this file) loaded every included session's FULL
calciumdata/behaviordata/videodata into memory up front via
load_sessions(...), then handed complete Session objects to as many
parallel workers as there are CPU cores. Changes here, matching the
lab's own plot_temporal_*.py reference scripts:
  1. Each session is loaded ON DEMAND, one per worker (via
     infotheory.session.Session directly, not the load_sessions batch
     helper) -- so only ONE session's data is ever in memory per active
     worker, not every included session simultaneously.
  2. Pairs are sampled from celldata FIRST (cheap -- no calcium data
     needed), and only the calcium columns for cells that actually end
     up in a sampled pair are carried through the (expensive) position-
     binning step -- deconvdata.csv itself still has to be read in full
     from disk (build_session_tables, which this reuses for the trial-
     level stim_bin/choice metadata, requires calciumdata already
     loaded), but the large intermediate long-format DataFrame inside
     build_position_binned_trial_tensor only ever holds the cells that
     matter, not the whole population -- the dominant cost when a
     group has far more cells than end up in MAX_PAIRS_PER_GROUP pairs.
  3. Each worker explicitly deletes its session's large attributes
     (calciumdata, videodata, behaviordata) before returning.
  REDUCING N_SHUFFLES OR PID_SHUFFLES (in 3e/3f) DOES NOT MEANINGFULLY
  REDUCE MEMORY -- a shuffle-corrected MI estimate stores only
  n_shuffles FLOATS at a time (tens of bytes), not n_shuffles copies of
  the data; shuffle count trades off compute TIME, not RAM. The actual
  memory lever is N_JOBS_SESSIONS below: it controls how many sessions'
  data are held in memory SIMULTANEOUSLY (one Session per active
  worker) -- lower it (e.g. to 2-4, or 1 to process strictly
  sequentially) to trade wall-clock time for lower peak memory.

CACHING IS NOW PER SESSION (infotheory.cache.cache_path/load_or_compute
-- see that module's docstring), not one big store/*.pkl for the whole
script. This means: run this script on however many sessions fit
comfortably in memory today; run it again another day (same settings)
and already-cached sessions are loaded instantly from disk while any
NEW sessions get computed and added -- results accumulate across runs
rather than needing a full recompute every time. Changing any
parameter that affects the result (bin size, pair cap, shuffle counts,
...) automatically computes fresh, separately-cached results for that
new parameter combination rather than silently mixing with old ones;
delete the specific stale cache files in store/temporal_corr_cache/ if
you want to reclaim disk space, or leave them -- they don't affect
correctness either way.

Bin edges/response "alphabet" for the underlying MI-adjacent quantities
elsewhere in this project are computed once by pooling across all bins
(see 3e/3f); this script computes plain Pearson correlation, which
needs no such binning.

Same session/trial/cell filters and pair-sampling as 3a (same
(area, label) grouping, same sample_pairs_within_groups call).

Outputs
-------
2_pipeline/3d_pair_activity_statistics_temporal/
    out/    pair_correlations_temporal.csv   one row per (session,
                                              area, label, cell_i,
                                              cell_j, s_rel): signal/
                                              noise/raw correlation
            figures/*.png
    store/  temporal_corr_cache/<session>__<hash>.pkl   one cache file
                                              per (session, parameter
                                              combination) -- see
                                              infotheory/cache.py
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from infotheory.pipeline import get_pipeline_paths
from infotheory.session import Session, discover_sessions
from infotheory.neural_encoding import load_cell_inclusion, build_session_tables
from infotheory.celldata_utils import (
    get_area_label, sample_pairs_within_groups, AREA_COLORS, LABEL_SPLIT_AREAS, DEFAULT_AREA_ORDER,
)
from infotheory.continuous import merge_behavior_video, restrict_to_engaged, remove_pupil_outliers, build_calcium_continuous
from infotheory.psth import build_position_binned_trial_tensor
from infotheory.cache import cache_path, load_or_compute
from infotheory.plotting import set_style, save_fig, clear_figures

PROTOCOL = "DN"
S_PRE, S_POST, BINSIZE = -80.0, 60.0, 15.0
MAX_PAIRS_PER_GROUP = 100
MIN_TRIALS_PER_BIN = 20
MIN_TRIALS_PER_STIM_BIN = 3
RANDOM_STATE = 0

N_JOBS_SESSIONS = 8   # THE memory lever -- see module docstring. Lower this (e.g. 2-4, or 1
                        # for strictly sequential) if RAM is tight; it does NOT need to match
                        # your CPU count, and unlike shuffle counts it directly controls how
                        # many sessions' full data are held in memory at once.
JOBLIB_VERBOSE = 5

parser = argparse.ArgumentParser()
parser.add_argument("--recompute", action="store_true",
                     help="Ignore existing per-session cache files and recompute everything "
                          "(only affects sessions whose cache already exists -- see module docstring)")
args = parser.parse_args()

paths = get_pipeline_paths(__file__)
figdir = paths.out / "figures"
clear_figures(figdir)
set_style()
cache_dir = paths.store / "temporal_corr_cache"


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
qc_per_cell_path = activity_out / "qc_per_cell.csv"
cell_inclusion = load_cell_inclusion(cell_dist_out, activity_out)
included_cell_ids_by_session = {
    sid: set(sub.loc[sub["include"], "cell_id"]) for sid, sub in cell_inclusion.groupby("session_id")
}
print(f"Using {len(included_ids)} included {PROTOCOL} sessions, "
      f"{int(cell_inclusion['include'].sum())}/{len(cell_inclusion)} cells included")

# Map session_id -> (protocol, animal_id, sessiondate) so each worker can load ONE session
# on demand, rather than loading every session up front via load_sessions(...).
session_lookup = {f"{animal_id}_{sessiondate}": (protocol, animal_id, sessiondate)
                   for protocol, animal_id, sessiondate in discover_sessions([PROTOCOL])
                   if f"{animal_id}_{sessiondate}" in included_ids}


def corr_fn(a, b):
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------- #
# 2. Per-session, position-resolved pairwise correlations, cached
#    individually -- pairs are sampled from celldata FIRST (cheap, no
#    calcium data needed), so only the calcium traces for cells that
#    end up in a sampled pair are ever loaded/kept.
# --------------------------------------------------------------------- #
def process_one_session(session_id, seed):
    protocol, animal_id, sessiondate = session_lookup[session_id]
    included_cells = included_cell_ids_by_session.get(session_id)
    if not included_cells:
        return []

    key_params = dict(
        s_pre=S_PRE, s_post=S_POST, binsize=BINSIZE, max_pairs_per_group=MAX_PAIRS_PER_GROUP,
        min_trials_per_bin=MIN_TRIALS_PER_BIN, min_trials_per_stim_bin=MIN_TRIALS_PER_STIM_BIN,
        random_state=seed,
        qc_mtime=qc_per_cell_path.stat().st_mtime if qc_per_cell_path.exists() else None,
    )
    path = cache_path(cache_dir, session_id, **key_params)

    def _compute():
        rng_local = np.random.default_rng(seed)
        ses = Session(protocol, animal_id, sessiondate)
        # Load everything at once -- build_session_tables needs calciumdata/ts_F already
        # present (it returns {} immediately otherwise), so the pairs-sampled-before-loading-
        # calcium-data optimization tried in an earlier version of this fix isn't compatible
        # with reusing that function as-is. The memory win that MATTERS most (one session's
        # data in memory at a time, not every included session simultaneously, plus explicit
        # cleanup below) is preserved regardless.
        ses.load(load_behaviordata=True, load_videodata=True, load_celldata=True, load_calciumdata=True)

        celldata = ses.celldata[ses.celldata["cell_id"].isin(included_cells)].reset_index(drop=True)
        if celldata.empty:
            return []
        area, label, _ = get_area_label(celldata, label_split_areas=LABEL_SPLIT_AREAS)
        cell_ids = celldata["cell_id"].tolist()

        pairs, pair_groups = sample_pairs_within_groups(area, label, max_pairs_per_group=MAX_PAIRS_PER_GROUP,
                                                          rng=rng_local)
        needed_cell_ids = sorted({cell_ids[i] for i, _ in pairs} | {cell_ids[j] for _, j in pairs})
        if not needed_cell_ids:
            return []

        tables = build_session_tables(ses, mu_sigma_lookup, {"stim": (0.0, 20.0)},
                                       included_cell_ids=set(needed_cell_ids))
        if "stim" not in tables or tables["stim"].empty or "stim_bin" not in tables["stim"].columns:
            return []
        trial_meta = tables["stim"][["trialNumber", "stim_bin", "choice"]]

        merged_full = merge_behavior_video(ses.behaviordata, ses.videodata)
        merged_full = remove_pupil_outliers(merged_full)
        engaged = restrict_to_engaged(merged_full, ses.trialdata)
        if engaged.empty:
            return []
        engaged_trials = ses.trialdata[ses.trialdata.get("engaged", 1) == 1] \
            if "engaged" in ses.trialdata.columns else ses.trialdata

        # Only the columns actually needed go into the position-binning step, even though
        # the full deconvdata.csv had to be read from disk above (build_session_tables's
        # requirements don't allow skipping that part -- see comment above).
        calcium_continuous = build_calcium_continuous(ses.calciumdata[needed_cell_ids], ses.ts_F,
                                                        ses.behaviordata)
        tensor_wide = build_position_binned_trial_tensor(
            engaged_trials, calcium_continuous,
            cell_cols=needed_cell_ids, s_pre=S_PRE, s_post=S_POST, binsize=BINSIZE)
        tensor_wide = tensor_wide.merge(trial_meta, on="trialNumber", how="inner")

        valid_bins = tensor_wide.groupby("trialNumber")["stim_bin"].first().pipe(
            lambda s: s.value_counts())
        valid_bins = valid_bins[valid_bins >= MIN_TRIALS_PER_STIM_BIN].index
        tensor_wide = tensor_wide[tensor_wide["stim_bin"].isin(valid_bins)]

        records = []
        for s_rel, bin_table in tensor_wide.groupby("s_rel"):
            if len(bin_table) < MIN_TRIALS_PER_BIN or bin_table["stim_bin"].nunique() < 2:
                continue
            sb = bin_table["stim_bin"].to_numpy()
            for (i, j), (a, l) in zip(pairs, pair_groups):
                cell_i, cell_j = cell_ids[i], cell_ids[j]
                if cell_i not in bin_table.columns or cell_j not in bin_table.columns:
                    continue
                xi = bin_table[cell_i].to_numpy()
                xj = bin_table[cell_j].to_numpy()
                valid = np.isfinite(xi) & np.isfinite(xj)
                if valid.sum() < MIN_TRIALS_PER_BIN:
                    continue
                xi_v, xj_v, sb_v = xi[valid], xj[valid], sb[valid]

                tuning_i = pd.Series(xi_v).groupby(sb_v).transform("mean").to_numpy()
                tuning_j = pd.Series(xj_v).groupby(sb_v).transform("mean").to_numpy()
                bin_means_i = pd.Series(xi_v).groupby(sb_v).mean()
                bin_means_j = pd.Series(xj_v).groupby(sb_v).mean()

                records.append({
                    "session_id": session_id, "roi_name": a, "labeled_grouped": l,
                    "cell_i": cell_i, "cell_j": cell_j, "s_rel": s_rel, "n_trials": len(xi_v),
                    "signal_corr": corr_fn(bin_means_i.to_numpy(), bin_means_j.to_numpy()),
                    "noise_corr": corr_fn(xi_v - tuning_i, xj_v - tuning_j),
                    "raw_corr": corr_fn(xi_v, xj_v),
                })

        # Explicit cleanup -- matches the reference scripts' own pattern -- so the worker
        # process doesn't hold onto the full traces any longer than necessary.
        for attr in ("calciumdata", "videodata", "behaviordata"):
            if hasattr(ses, attr):
                delattr(ses, attr)

        print(f"  [{session_id}] temporal pairwise correlations done "
              f"({len(pairs)} pairs x {tensor_wide['s_rel'].nunique()} bins, "
              f"{len(needed_cell_ids)} cells loaded)")
        return records

    return load_or_compute(path, _compute, force_recompute=args.recompute)


print(f"Computing position-resolved pairwise correlations for {len(included_ids)} sessions "
      f"(n_jobs={N_JOBS_SESSIONS}, cache: {cache_dir}) ...")
nested = Parallel(n_jobs=N_JOBS_SESSIONS, backend="loky", verbose=JOBLIB_VERBOSE)(
    delayed(process_one_session)(sid, RANDOM_STATE + i) for i, sid in enumerate(included_ids)
)
corr_table = pd.DataFrame([rec for group in nested for rec in group])
corr_table.to_csv(paths.out / "pair_correlations_temporal.csv", index=False)
print(f"\n{len(corr_table)} (pair, position bin) rows computed "
      f"({len(list(cache_dir.glob('*.pkl')))} session cache files on disk in total)")



# --------------------------------------------------------------------- #
# 3. Figure: noise correlation vs position, mean +/- SEM across pairs,
#    one line per area (matching the real project's plot_temporal_
#    information.py style)
# --------------------------------------------------------------------- #
if len(corr_table):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, metric in zip(axes, ["signal_corr", "noise_corr", "raw_corr"]):
        for area in [a for a in DEFAULT_AREA_ORDER if a in corr_table["roi_name"].unique()]:
            sub = corr_table[corr_table["roi_name"] == area].dropna(subset=[metric])
            if sub.empty:
                continue
            grouped = sub.groupby("s_rel")[metric].agg(["mean", "sem", "count"]).reset_index()
            ax.plot(grouped["s_rel"], grouped["mean"], color=AREA_COLORS.get(area, "gray"),
                    label=f"{area} (n_pairs~{int(grouped['count'].median())})")
            ax.fill_between(grouped["s_rel"], grouped["mean"] - grouped["sem"],
                             grouped["mean"] + grouped["sem"], color=AREA_COLORS.get(area, "gray"),
                             alpha=0.15)
        ax.axvline(0, color="black", linestyle=":", linewidth=1)
        ax.axhline(0, color="grey", linewidth=0.7)
        ax.set_xlabel("position rel. to stimulus onset (cm)")
        ax.set_ylabel(metric)
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"Pairwise correlation over position -- {PROTOCOL}", y=1.03)
    fig.tight_layout()
    save_fig(fig, figdir, "1_correlation_vs_position")

print(f"\nWrote tables to {paths.out}, figures to {figdir}")
print("Run `python b_progress/make_progress_md.py 3d_pair_activity_statistics_temporal` to build a markdown summary.")
