# -*- coding: utf-8 -*-
"""
3f_pair_information_pid_temporal
====================================
Position-resolved version of 3c_pair_information_pid.py: the same
Partial Information Decomposition (redundant/unique/synergistic, I_min
by default -- see 3c's and infotheory/pid.py's module docstrings for
the redundancy-measure registry and swap-in path for BROJA-2PID later),
computed at every bin of a position-binned trial tensor instead of once
per coarse window. Reuses infotheory.pid.pid_from_binned_responses
directly -- the exact same, already-validated (against the textbook
copy/XOR cases) function 3c calls, just once per (pair, position bin).

TENSOR CONSTRUCTION: same convention as 3d/3e_..._temporal.py (one
value per trial per narrow position bin) -- see 3d's module docstring.

This is the cheapest of the three temporal scripts to run (PID from an
already-binned joint histogram is a handful of array operations, same
as 3c itself), so it can afford closer-to-3c's own pair/bin settings
than 3e needed to cut down to.

MEMORY AND CACHING: same fix as 3d/3e_..._temporal.py -- see 3d's
module docstring for the full explanation. Summary: sessions load ONE
AT A TIME (not the load_sessions batch helper), so N_JOBS_SESSIONS caps
how many sessions' data are in memory at once (THE memory lever here,
not PID_SHUFFLES -- that trades off compute time only); pairs are
sampled from celldata first so only the needed cell columns get
carried through the tensor-building step; each worker frees
calciumdata/videodata/behaviordata before returning; results are
cached PER SESSION (infotheory.cache), so running this again later
reuses whatever's already cached.

Same session/trial/cell filters and pair-sampling as 3a/3b/3c.

Outputs
-------
2_pipeline/3f_pair_information_pid_temporal/
    out/    pid_temporal.csv    one row per (session, area, label,
                                 cell_i, cell_j, s_rel): I1, I2, I12,
                                 redundancy, unique1, unique2, synergy
            figures/*.png
    store/  temporal_pid_cache/<session>__<hash>.pkl   one cache file
                                 per (session, parameter combination)
                                 -- see infotheory/cache.py
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
from infotheory.pid import pid_from_binned_responses, REDUNDANCY_MEASURES
from infotheory.cache import cache_path, load_or_compute
from infotheory.plotting import set_style, save_fig, clear_figures

PROTOCOL = "DN"
S_PRE, S_POST, BINSIZE = -60.0, 80.0, 20.0
REDUNDANCY_MEASURE = "imin"  # see infotheory/pid.py -- change to "broja" once wired in
MAX_PAIRS_PER_GROUP = 60
R_BINS = 3
PID_SHUFFLES = 20             # affects TIME, not memory -- see module docstring
MIN_TRIALS_PER_BIN = 20
MIN_TRIALS_PER_STIM_BIN = 3
RANDOM_STATE = 0

N_JOBS_SESSIONS = 8   # THE memory lever -- see module docstring. Lower this (e.g. 2-4, or 1
                        # for strictly sequential) if RAM is tight.
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
cache_dir = paths.store / "temporal_pid_cache"

if REDUNDANCY_MEASURE not in REDUNDANCY_MEASURES:
    raise SystemExit(f"Unknown REDUNDANCY_MEASURE {REDUNDANCY_MEASURE!r}; "
                      f"must be one of {list(REDUNDANCY_MEASURES)}")


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
      f"{int(cell_inclusion['include'].sum())}/{len(cell_inclusion)} cells included, "
      f"redundancy measure = {REDUNDANCY_MEASURE!r}")

session_lookup = {f"{animal_id}_{sessiondate}": (protocol, animal_id, sessiondate)
                   for protocol, animal_id, sessiondate in discover_sessions([PROTOCOL])
                   if f"{animal_id}_{sessiondate}" in included_ids}


# --------------------------------------------------------------------- #
# 2. Per-session, position-resolved PID, cached individually
# --------------------------------------------------------------------- #
def process_one_session(session_id, seed):
    protocol, animal_id, sessiondate = session_lookup[session_id]
    included_cells = included_cell_ids_by_session.get(session_id)
    if not included_cells:
        return []

    key_params = dict(
        s_pre=S_PRE, s_post=S_POST, binsize=BINSIZE, redundancy_measure=REDUNDANCY_MEASURE,
        max_pairs_per_group=MAX_PAIRS_PER_GROUP, r_bins=R_BINS, pid_shuffles=PID_SHUFFLES,
        min_trials_per_bin=MIN_TRIALS_PER_BIN, min_trials_per_stim_bin=MIN_TRIALS_PER_STIM_BIN,
        random_state=seed,
        qc_mtime=qc_per_cell_path.stat().st_mtime if qc_per_cell_path.exists() else None,
    )
    path = cache_path(cache_dir, session_id, **key_params)

    def _compute():
        rng_local = np.random.default_rng(seed)
        ses = Session(protocol, animal_id, sessiondate)
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

        calcium_continuous = build_calcium_continuous(ses.calciumdata[needed_cell_ids], ses.ts_F,
                                                        ses.behaviordata)
        tensor_wide = build_position_binned_trial_tensor(
            engaged_trials, calcium_continuous,
            cell_cols=needed_cell_ids, s_pre=S_PRE, s_post=S_POST, binsize=BINSIZE)
        tensor_wide = tensor_wide.merge(trial_meta, on="trialNumber", how="inner")

        valid_stims = tensor_wide.groupby("trialNumber")["stim_bin"].first().value_counts()
        valid_stims = valid_stims[valid_stims >= MIN_TRIALS_PER_STIM_BIN].index
        tensor_wide = tensor_wide[tensor_wide["stim_bin"].isin(valid_stims)]

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
                if len(np.unique(sb_v)) < 2:
                    continue

                result = pid_from_binned_responses(xi_v, xj_v, sb_v, bins=R_BINS,
                                                    redundancy_measure=REDUNDANCY_MEASURE,
                                                    n_shuffles=PID_SHUFFLES, rng=rng_local)
                records.append({"session_id": session_id, "roi_name": a, "labeled_grouped": l,
                                 "cell_i": cell_i, "cell_j": cell_j, "s_rel": s_rel, "n_trials": result.n,
                                 "I1": result.I1, "I2": result.I2, "I12": result.I12,
                                 "redundancy": result.redundancy, "unique1": result.unique1,
                                 "unique2": result.unique2, "synergy": result.synergy,
                                 "redundancy_p_value": result.redundancy_p_value})

        for attr in ("calciumdata", "videodata", "behaviordata"):
            if hasattr(ses, attr):
                delattr(ses, attr)

        print(f"  [{session_id}] temporal PID done "
              f"({len(pairs)} pairs x {tensor_wide['s_rel'].nunique()} bins, "
              f"{len(needed_cell_ids)} cells loaded)")
        return records

    return load_or_compute(path, _compute, force_recompute=args.recompute)


print(f"Computing position-resolved PID for {len(included_ids)} sessions "
      f"(n_jobs={N_JOBS_SESSIONS}, cache: {cache_dir}) ...")
nested = Parallel(n_jobs=N_JOBS_SESSIONS, backend="loky", verbose=JOBLIB_VERBOSE)(
    delayed(process_one_session)(sid, RANDOM_STATE + i) for i, sid in enumerate(included_ids)
)
pid_table = pd.DataFrame([rec for group in nested for rec in group])
pid_table.to_csv(paths.out / "pid_temporal.csv", index=False)
print(f"\n{len(pid_table)} (pair, position bin) rows computed "
      f"({len(list(cache_dir.glob('*.pkl')))} session cache files on disk in total)")


# --------------------------------------------------------------------- #
# 3. Figure: redundancy/unique1/unique2/synergy vs position (mean
#    across pairs), one panel per area -- matching the real project's
#    plot_temporal_pid.py style
# --------------------------------------------------------------------- #
TERM_COLORS = {"redundancy": "tab:blue", "unique1": "tab:purple", "unique2": "tab:orange",
               "synergy": "tab:green"}

if len(pid_table):
    areas_present = [a for a in DEFAULT_AREA_ORDER if a in pid_table["roi_name"].unique()]
    fig, axes = plt.subplots(1, len(areas_present), figsize=(4.5 * len(areas_present), 4.5), squeeze=False)
    for ax, area in zip(axes[0], areas_present):
        sub = pid_table[pid_table["roi_name"] == area]
        grouped = sub.groupby("s_rel")[["I12", "redundancy", "unique1", "unique2", "synergy"]].mean().reset_index()
        n_pairs = sub.groupby("s_rel")["cell_i"].count().median()
        for term, color in TERM_COLORS.items():
            ax.plot(grouped["s_rel"], grouped[term], color=color, label=term)
        ax.plot(grouped["s_rel"], grouped["I12"], color="k", linestyle="--", linewidth=1.5, label="I12")
        ax.axhline(0, color="grey", linewidth=0.7)
        ax.axvline(0, color="black", linestyle=":", linewidth=1)
        ax.set_xlabel("position rel. to stimulus onset (cm)")
        ax.set_ylabel("information (bits)")
        ax.set_title(f"{area} (n_pairs~{int(n_pairs)})")
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Pairwise PID over position -- {REDUNDANCY_MEASURE}, {PROTOCOL}", y=1.03)
    fig.tight_layout()
    save_fig(fig, figdir, "1_pid_vs_position")

print(f"\nWrote tables to {paths.out}, figures to {figdir}")
print("Run `python b_progress/make_progress_md.py 3f_pair_information_pid_temporal` to build a markdown summary.")
