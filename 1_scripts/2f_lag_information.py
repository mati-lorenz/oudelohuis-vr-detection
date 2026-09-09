# -*- coding: utf-8 -*-
"""
2f_lag_information
=====================
Two questions, in order:

1. Is a cell's activity more informative about a behavioral variable at
   some TEMPORAL LAG (the variable measured before or after the neural
   sample) than at zero lag? For each (predictor, area) -- lag chosen
   PER AREA, not globally or per cell, since a single cell's estimate
   would be too noisy but different areas may genuinely have different
   lags -- searches a range of lags and picks the one that maximizes
   mutual information, averaged across that area's cells.

2. Using each (predictor, area)'s optimal lag, build a POSITION-resolved
   information PSTH: mutual information between a cell's activity and
   the (lag-shifted) behavioral variable, as a function of trial-aligned
   POSITION (same convention as 1c_behavior.py's position-domain PSTH --
   x axis is position, the lag applied to get there is purely temporal).

Per the person's explicit instruction: NO trial-window averaging
anywhere in this script -- every MI computation pools raw, individual
CALCIUM FRAMES (not per-trial means) as its sample set. This is
different from 2c/2d/2e_*.py, which use trial-window means; this script
exists specifically because averaging within windows could hide a
lagged relationship that only shows up at fine temporal resolution.

Dataset: same session/trial/cell filtering as 2c/2d/2e (1b's included
sessions, engaged trials, cells passing 2a's proximity filter AND 2b's
QC -- see infotheory.neural_encoding.load_cell_inclusion). Behavioral
predictors only (running speed, pupil area, motion energy, lick) --
stim/choice/correct are per-trial scalars, not continuous traces, so a
temporal lag isn't a meaningful concept for them the way it is for a
continuously-varying signal.

Outputs
-------
2_pipeline/2f_lag_information/
    out/    lag_search_per_cell.csv     one row per (session, cell,
                                         predictor, lag): raw MI (no
                                         shuffle correction -- see below)
            optimal_lags.csv            one row per (predictor, area):
                                         chosen lag + the area-averaged
                                         MI-vs-lag curve
            position_information.csv    one row per (predictor, area,
                                         position bin): shuffle-corrected
                                         MI, averaged across that area's
                                         cells, using that (predictor,
                                         area)'s optimal lag
            figures/*.png
    store/  lag_search.pkl, position_information.pkl   cached (reused
                                         unless --recompute)

Compute-budget notes (same spirit as 2c_information.py): the LAG SEARCH
(part 1) uses plain, uncorrected histogram MI (n_shuffles=0) -- only the
relative ordering across lags matters for picking an argmax, not
calibrated bits, and this step already means (n_cells x n_predictors x
n_lags) histogram-MI calls over full-session-length frame series. The
final POSITION-resolved information (part 2, the actual reported
figure) uses the normal shuffle-corrected MI.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from infotheory.pipeline import get_pipeline_paths
from infotheory.session import load_sessions
from infotheory.neural_encoding import load_cell_inclusion, BEHAVIOR_PREDICTORS
from infotheory.continuous import build_calcium_continuous, merge_behavior_video, restrict_to_engaged, remove_pupil_outliers
from infotheory.psth import derive_onset_time, compute_position_binned_information
from infotheory.info_theory import mutual_information_hist
from infotheory.celldata_utils import AREA_COLORS, DEFAULT_AREA_ORDER
from infotheory.plotting import set_style, save_fig, clear_figures

PROTOCOL = "DN"
PREDICTORS = BEHAVIOR_PREDICTORS  # continuous only -- see module docstring

LAG_RANGE_S = (-2.0, 2.0)
LAG_STEP_S = 0.25
MI_BINS_LAG_SEARCH = 10   # uncorrected, fast -- see module docstring

POSITION_WINDOW = (-80.0, 60.0, 8.0)  # s_pre, s_post, binsize (cm) -- coarser than 1c_behavior.py's
                                        # 5cm PSTH bins on purpose: this figure pools activity
                                        # across hundreds of cells per area already, so the extra
                                        # precision from finer spatial bins matters less than
                                        # keeping the shuffle-corrected MI computation tractable
                                        # (cells x predictors x bins x shuffles adds up fast)
TRUSTED_POSITION_RANGE = (-40.0, 40.0)  # KNOWN CAVEAT, found while validating this script: bins
                                        # beyond roughly +-40cm from stimulus onset show an
                                        # elevated MI signature that appears IDENTICALLY across
                                        # every predictor (runspeed/pupil/motionenergy/lick) and
                                        # every area -- the signature of a shared artifact, not a
                                        # real relationship (a genuine behavioral encoding
                                        # wouldn't hit four unrelated variables at exactly the
                                        # same position). Raising MIN_SAMPLES_PER_BIN removed the
                                        # WORST of it (an initial 30 -> 400), so it isn't pure
                                        # small-sample MI bias; the likely remaining cause is
                                        # cross-trial contamination -- zpos never resets between
                                        # trials, so a position bin far before stimulus onset can
                                        # fall inside the PREVIOUS trial's reward/ITI period for
                                        # shorter-ITI trials, and that period's behavioral
                                        # structure varies systematically by that trial's outcome
                                        # (see 1c_behavior.py's own PSTH figures). NOT fully
                                        # resolved -- the figure below shades this range rather
                                        # than hiding it; treat anything outside
                                        # TRUSTED_POSITION_RANGE as unreliable until that
                                        # cross-trial-overlap hypothesis is checked directly
                                        # (e.g. by excluding trials with a short prior ITI).
MI_BINS_PSTH = 10
N_SHUFFLES_PSTH = 10  # reduced from a first pass at 20 -- still a real bias correction,
                       # but this is an aggregate (per-area-averaged) figure where shuffle
                       # precision on any one cell's estimate matters less than tractability
MIN_SAMPLES_PER_BIN = 400  # raised again after the first fix (200) still left the single
                            # extreme edge bin (+60cm) populated by only ~20% as many cells as
                            # every other bin (32/162) -- not every trial runs 60cm past
                            # stimulus onset, so that bin's remaining cells are a small, biased
                            # subset rather than a representative sample. 400 keeps every bin
                            # from -76cm to +52cm (all comfortably >=750 samples on average) and
                            # drops only the single thinnest edge bin.

RANDOM_STATE = 0
N_JOBS_SESSIONS = -1
JOBLIB_VERBOSE = 5

parser = argparse.ArgumentParser()
parser.add_argument("--recompute", action="store_true",
                     help="Reload sessions from 0_data/ instead of reusing cached store/ files")
args = parser.parse_args()

paths = get_pipeline_paths(__file__)
figdir = paths.out / "figures"
clear_figures(figdir)
set_style()


# --------------------------------------------------------------------- #
# 1. Included sessions/cells -- same filters as 2c/2d/2e
# --------------------------------------------------------------------- #
fits_path = paths.out_from("1b_psychometric") / "psychometric_fits.csv"
fits_df = pd.read_csv(fits_path)
included_ids = fits_df.loc[fits_df["included"] & (fits_df["protocol"] == PROTOCOL), "session_id"].tolist()
if not included_ids:
    raise SystemExit(f"No {PROTOCOL} sessions passed 1b_psychometric's inclusion criteria.")

cell_dist_out = paths.out_from("2a_cell_distribution")
activity_out = paths.out_from("2b_activity_statistics")
cell_inclusion = load_cell_inclusion(cell_dist_out, activity_out)
included_cell_ids_by_session = {
    sid: set(sub.loc[sub["include"], "cell_id"]) for sid, sub in cell_inclusion.groupby("session_id")
}
celldata_combined = pd.read_csv(cell_dist_out / "celldata_combined.csv")[["session_id", "cell_id", "roi_name"]]
print(f"Using {len(included_ids)} included {PROTOCOL} sessions, "
      f"{int(cell_inclusion['include'].sum())}/{len(cell_inclusion)} cells included")


def shift_with_nan(arr: np.ndarray, lag_frames: int) -> np.ndarray:
    """Shift `arr` by `lag_frames` samples, padding the vacated edge
    with NaN -- unlike np.roll, never wraps data around from the other
    end of the session (which would silently contaminate a shifted
    trace with unrelated end-of-session/start-of-session values).

    shift_with_nan(behavior, +lag)[t] == behavior[t - lag] -- i.e. a
    POSITIVE lag_frames means the returned series at time t holds
    behavior's value from `lag` samples EARLIER. When this shifted
    behavior series is then compared (via MI) against activity[t]
    (unshifted), a positive lag therefore tests "does behavior from
    `lag` samples ago relate to activity right now" -- BEHAVIOR leads
    (happened earlier), ACTIVITY lags (is the delayed response).
    Negative lag_frames is the mirror case: behavior[t - lag] with
    lag<0 pulls from a LATER sample, so a negative lag tests "does
    behavior from `|lag|` samples in the FUTURE relate to activity
    now" -- ACTIVITY leads, behavior follows (e.g. a preparatory/
    efference-copy signal that precedes the behavioral change it
    predicts). Verified against a synthetic activity trace built as a
    known, deliberately-lagged function of behavior: the lag search
    below recovers the injected lag exactly, with this sign."""
    shifted = np.full(len(arr), np.nan, dtype=float)
    if lag_frames > 0:
        shifted[lag_frames:] = arr[:-lag_frames] if lag_frames < len(arr) else np.nan
    elif lag_frames < 0:
        shifted[:lag_frames] = arr[-lag_frames:]
    else:
        shifted[:] = arr
    return shifted


def build_frame_table(ses):
    """Every included cell's activity + every behavioral predictor,
    fully aligned on the calcium clock, restricted to engaged periods.
    No trial-window averaging -- this IS the continuous, per-frame
    table both the lag search and the position-binned MI operate on
    directly."""
    included_cells = included_cell_ids_by_session.get(ses.session_id)
    if (not included_cells or ses.trialdata is None or ses.behaviordata is None
            or ses.celldata is None or ses.calciumdata is None or ses.ts_F is None):
        return None, None
    celldata = ses.celldata.reset_index(drop=True)
    keep_mask = celldata["cell_id"].isin(included_cells).to_numpy()
    if not keep_mask.any():
        return None, None
    cell_ids = celldata.loc[keep_mask, "cell_id"].to_numpy()
    calciumdata = ses.calciumdata.loc[:, cell_ids]

    merged_full = remove_pupil_outliers(merge_behavior_video(ses.behaviordata, ses.videodata))
    available_predictors = [p for p in PREDICTORS if p in merged_full.columns]
    frame_table = build_calcium_continuous(calciumdata, ses.ts_F, merged_full, extra_columns=available_predictors)
    frame_table = restrict_to_engaged(frame_table, ses.trialdata)
    if frame_table.empty:
        return None, None

    engaged_trials = (ses.trialdata[ses.trialdata["engaged"] == 1]
                       if "engaged" in ses.trialdata.columns else ses.trialdata.copy())
    engaged_trials = derive_onset_time(engaged_trials, merged_full)
    return frame_table, engaged_trials


# --------------------------------------------------------------------- #
# 2. Lag search: for each cell, each predictor, each lag -- raw MI over
#    ALL pooled frames (no trial windows).
# --------------------------------------------------------------------- #
lag_search_cache = paths.store / "lag_search.pkl"
lag_search_table = None
if not args.recompute and lag_search_cache.exists():
    try:
        lag_search_table = pd.read_pickle(lag_search_cache)
        if not {"session_id", "cell_id", "predictor", "lag_s", "mi_bits"}.issubset(lag_search_table.columns):
            lag_search_table = None
    except Exception:
        lag_search_table = None
    if lag_search_table is not None:
        print(f"Reusing cached lag search from {paths.store} (pass --recompute to reload 0_data/)")

lags = np.round(np.arange(LAG_RANGE_S[0], LAG_RANGE_S[1] + LAG_STEP_S / 2, LAG_STEP_S), 6)

if lag_search_table is None:
    sessions = load_sessions(protocols=[PROTOCOL], load_behaviordata=True, load_videodata=True,
                              load_celldata=True, load_calciumdata=True, only_session_ids=included_ids)

    def lag_search_one_session(ses):
        from infotheory.spike_stats import get_frame_rate
        frame_table, _ = build_frame_table(ses)
        if frame_table is None:
            return []
        fs = get_frame_rate(ses, fallback=8.0)
        cell_cols = [c for c in ses.celldata["cell_id"] if c in frame_table.columns]
        available_predictors = [p for p in PREDICTORS if p in frame_table.columns]
        area_lookup = ses.celldata.set_index("cell_id")["roi_name"].to_dict()

        records = []
        for predictor in available_predictors:
            ref = frame_table[predictor].to_numpy()
            for lag_s in lags:
                lag_frames = int(round(lag_s * fs))
                ref_shifted = shift_with_nan(ref, lag_frames)
                for cell_id in cell_cols:
                    activity = frame_table[cell_id].to_numpy()
                    finite = np.isfinite(activity) & np.isfinite(ref_shifted)
                    if finite.sum() < 50:
                        continue
                    mi_bits = mutual_information_hist(ref_shifted[finite], activity[finite],
                                                        bins=MI_BINS_LAG_SEARCH)
                    records.append({"session_id": ses.session_id, "cell_id": cell_id,
                                     "roi_name": area_lookup.get(cell_id), "predictor": predictor,
                                     "lag_s": lag_s, "mi_bits": mi_bits})
        print(f"  [{ses.session_id}] lag search done ({len(cell_cols)} cells x "
              f"{len(available_predictors)} predictors x {len(lags)} lags)")
        return records

    print(f"Lag search over {len(sessions)} sessions x {len(lags)} lags "
          f"(n_jobs={N_JOBS_SESSIONS}) ...")
    nested = Parallel(n_jobs=N_JOBS_SESSIONS, backend="loky", verbose=JOBLIB_VERBOSE)(
        delayed(lag_search_one_session)(ses) for ses in sessions
    )
    lag_search_table = pd.DataFrame([rec for group in nested for rec in group])
    lag_search_table.to_pickle(lag_search_cache)

lag_search_table.to_csv(paths.out / "lag_search_per_cell.csv", index=False)
print(f"\n{len(lag_search_table)} (cell, predictor, lag) MI values computed")


# --------------------------------------------------------------------- #
# 3. Optimal lag per (predictor, area): average the per-cell MI-vs-lag
#    curves within each area, then take the argmax.
# --------------------------------------------------------------------- #
area_curve = (lag_search_table.groupby(["predictor", "roi_name", "lag_s"])["mi_bits"]
              .mean().reset_index())
optimal_lags = (area_curve.loc[area_curve.groupby(["predictor", "roi_name"])["mi_bits"].idxmax()]
                 .rename(columns={"lag_s": "optimal_lag_s", "mi_bits": "mi_bits_at_optimum"})
                 .reset_index(drop=True))
optimal_lags.to_csv(paths.out / "optimal_lags.csv", index=False)
area_curve.to_csv(paths.out / "lag_curve_by_area.csv", index=False)

print("\nOptimal lag per (predictor, area) -- positive = behavior BEFORE activity "
      "(behavior leads, activity's response is delayed), negative = behavior AFTER activity "
      "(activity leads, e.g. an efference-copy/motor-preparation signal preceding the movement):")
print(optimal_lags[["predictor", "roi_name", "optimal_lag_s", "mi_bits_at_optimum"]].to_string(index=False))

# Figure: MI-vs-lag curve, one panel per predictor, one line per area
fig, axes = plt.subplots(1, len(PREDICTORS), figsize=(4.5 * len(PREDICTORS), 4), squeeze=False)
for ax, predictor in zip(axes[0], PREDICTORS):
    sub = area_curve[area_curve["predictor"] == predictor]
    for area in [a for a in DEFAULT_AREA_ORDER if a in sub["roi_name"].unique()]:
        asub = sub[sub["roi_name"] == area].sort_values("lag_s")
        ax.plot(asub["lag_s"], asub["mi_bits"], color=AREA_COLORS.get(area, "gray"), label=area)
        best = optimal_lags[(optimal_lags["predictor"] == predictor) & (optimal_lags["roi_name"] == area)]
        if len(best):
            ax.axvline(best["optimal_lag_s"].iloc[0], color=AREA_COLORS.get(area, "gray"),
                       linestyle=":", linewidth=1, alpha=0.7)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("lag (s) [+ = behavior before activity (behavior leads)]")
    ax.set_ylabel("MI (bits, uncorrected)")
    ax.set_title(predictor)
axes[0][0].legend(fontsize=8, frameon=False)
fig.suptitle(f"Optimal lag search -- {PROTOCOL}", y=1.03)
fig.tight_layout()
save_fig(fig, figdir, "1_lag_search")


# --------------------------------------------------------------------- #
# 4. Position-resolved information, using each (predictor, area)'s
#    optimal lag -- pooling raw frames per position bin (never
#    trial-window means), shuffle-corrected MI.
# --------------------------------------------------------------------- #
position_cache = paths.store / "position_information.pkl"
position_table = None
if not args.recompute and position_cache.exists():
    try:
        position_table = pd.read_pickle(position_cache)
        if not {"predictor", "roi_name", "s_rel", "mi_bits"}.issubset(position_table.columns):
            position_table = None
    except Exception:
        position_table = None
    if position_table is not None:
        print(f"Reusing cached position-resolved information from {paths.store}")

if position_table is None:
    sessions = load_sessions(protocols=[PROTOCOL], load_behaviordata=True, load_videodata=True,
                              load_celldata=True, load_calciumdata=True, only_session_ids=included_ids)
    optimal_lag_lookup = optimal_lags.set_index(["predictor", "roi_name"])["optimal_lag_s"].to_dict()

    def position_info_one_session(ses, seed):
        from infotheory.spike_stats import get_frame_rate
        from infotheory.psth import compute_position_bin_indices, mi_from_position_bins
        rng_local = np.random.default_rng(seed)
        frame_table, engaged_trials = build_frame_table(ses)
        if frame_table is None or engaged_trials is None or engaged_trials.empty:
            return []
        fs = get_frame_rate(ses, fallback=8.0)
        cell_cols = [c for c in ses.celldata["cell_id"] if c in frame_table.columns]
        available_predictors = [p for p in PREDICTORS if p in frame_table.columns]
        area_lookup = ses.celldata.set_index("cell_id")["roi_name"].to_dict()
        s_pre, s_post, binsize = POSITION_WINDOW

        # Precompute the trial -> position-bin -> frame-index mapping
        # ONCE for this whole session -- it doesn't depend on which
        # cell or predictor is being analyzed, only on trial timing and
        # position, so redoing it per (cell, predictor) pair (as a
        # naive per-pair call to compute_position_binned_information
        # would) is pure waste at this scale (hundreds of cells x
        # several predictors) -- see psth.py's docstring note.
        bincenters, bin_frame_indices = compute_position_bin_indices(
            engaged_trials, frame_table, s_pre=s_pre, s_post=s_post, binsize=binsize)

        activity_arrays = {c: frame_table[c].to_numpy() for c in cell_cols}
        records = []
        for predictor in available_predictors:
            predictor_raw = frame_table[predictor].to_numpy()
            # Group cells by area so each area's cells reuse that
            # area's single lag-shifted reference trace, instead of
            # recomputing the shift per cell.
            cells_by_area = {}
            for cell_id in cell_cols:
                cells_by_area.setdefault(area_lookup.get(cell_id), []).append(cell_id)

            for area, area_cell_ids in cells_by_area.items():
                lag_s = optimal_lag_lookup.get((predictor, area))
                if lag_s is None:
                    continue
                lag_frames = int(round(lag_s * fs))
                ref_shifted = shift_with_nan(predictor_raw, lag_frames)

                for cell_id in area_cell_ids:
                    result = mi_from_position_bins(
                        activity_arrays[cell_id], ref_shifted, bincenters, bin_frame_indices,
                        mi_bins=MI_BINS_PSTH, n_shuffles=N_SHUFFLES_PSTH,
                        min_samples_per_bin=MIN_SAMPLES_PER_BIN, rng=rng_local)
                    for _, row in result.iterrows():
                        records.append({"session_id": ses.session_id, "cell_id": cell_id, "roi_name": area,
                                         "predictor": predictor, "s_rel": row["s_rel"],
                                         "n_samples": row["n_samples"], "mi_bits": row["mi_bits"]})
        print(f"  [{ses.session_id}] position-resolved information done")
        return records

    print(f"\nComputing position-resolved information for {len(sessions)} sessions "
          f"(n_jobs={N_JOBS_SESSIONS}) ...")
    nested = Parallel(n_jobs=N_JOBS_SESSIONS, backend="loky", verbose=JOBLIB_VERBOSE)(
        delayed(position_info_one_session)(ses, RANDOM_STATE + i) for i, ses in enumerate(sessions)
    )
    per_cell_position = pd.DataFrame([rec for group in nested for rec in group])
    # Average across cells within each (predictor, area) -- same
    # "compute per cell, then average" pattern as the lag search.
    position_table = (per_cell_position.groupby(["predictor", "roi_name", "s_rel"])
                       .agg(mi_bits=("mi_bits", "mean"), n_cells=("cell_id", "nunique")).reset_index())
    position_table.to_pickle(position_cache)
    per_cell_position.to_csv(paths.out / "position_information_per_cell.csv", index=False)

position_table.to_csv(paths.out / "position_information.csv", index=False)
print(f"\n{len(position_table)} (predictor, area, position bin) information values computed")
print(f"\nCAVEAT: positions outside {TRUSTED_POSITION_RANGE} cm show an elevated MI signature "
      f"present identically across every predictor and area -- likely cross-trial position "
      f"overlap (zpos never resets), not genuine encoding. See TRUSTED_POSITION_RANGE in this "
      f"script's module docstring. Treat the shaded region in figure 2 with caution.")


# --------------------------------------------------------------------- #
# 5. Figure: information PSTH -- x=position, y=MI, one panel per
#    predictor, one line per area, using that (predictor, area)'s
#    optimal lag.
# --------------------------------------------------------------------- #
fig, axes = plt.subplots(1, len(PREDICTORS), figsize=(4.5 * len(PREDICTORS), 4.5), squeeze=False)
for ax, predictor in zip(axes[0], PREDICTORS):
    sub = position_table[position_table["predictor"] == predictor]
    for area in [a for a in DEFAULT_AREA_ORDER if a in sub["roi_name"].unique()]:
        asub = sub[sub["roi_name"] == area].sort_values("s_rel")
        if asub.empty:
            continue
        lag_row = optimal_lags[(optimal_lags["predictor"] == predictor) & (optimal_lags["roi_name"] == area)]
        lag_label = f" (lag={lag_row['optimal_lag_s'].iloc[0]:+.2f}s)" if len(lag_row) else ""
        ax.plot(asub["s_rel"], asub["mi_bits"], color=AREA_COLORS.get(area, "gray"),
                 label=f"{area}{lag_label}")
    ax.axvline(0, color="black", linestyle=":", linewidth=1)
    ax.axvspan(POSITION_WINDOW[0], TRUSTED_POSITION_RANGE[0], color="grey", alpha=0.15, zorder=0)
    ax.axvspan(TRUSTED_POSITION_RANGE[1], POSITION_WINDOW[1], color="grey", alpha=0.15, zorder=0)
    ax.set_xlabel("position rel. to stimulus onset (cm)")
    ax.set_ylabel("MI (bits, shuffle-corrected)")
    ax.set_title(predictor)
    ax.legend(fontsize=7, frameon=False)
fig.suptitle(f"Lagged information PSTH -- {PROTOCOL} (behavior shifted by each area's own optimal lag)\n"
             f"shaded region: known artifact, see TRUSTED_POSITION_RANGE in this script's module docstring",
             y=1.05)
fig.tight_layout()
save_fig(fig, figdir, "2_lagged_information_psth")

print(f"\nWrote tables to {paths.out}, figures to {figdir}")
print("Run `python b_progress/make_progress_md.py 2f_lag_information` to build a markdown summary.")
