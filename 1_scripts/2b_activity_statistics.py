# -*- coding: utf-8 -*-
"""
2b_activity_statistics
========================
Single-cell statistics of the deconvolved ("spike") calcium activity -- DN sessions exclusively -- plus the
trial-averaged activity RASTER (formerly 2i_activity_raster, now part 2 of this script).

PART 1 -- statistics and QC (sections 1-9)
    A SANITY-CHECK / threshold-picking step, not a filtering step: every statistic runs on the FULL, unfiltered
    population (no anatomical, proximity or QC exclusion, and 1b's session list is deliberately NOT used: 1b judges
    task performance, not recording quality). The one product that later steps consume is qc_per_cell.csv: which
    cells WOULD fail under QC_THRESHOLDS.
    Per cell: event rate, active-frame rate, sparsity, inter-event-interval (IEI) median + CV, Fano factor,
    autocorrelogram, running-speed correlation and Fchan2 (motion artifact) correlation; QC flags (infotheory.qc_lib).

PART 2 -- activity raster (sections 10-13; was 2i)
    Trial-averaged activity of every cell as a function of position relative to the stimulus onset (0 cm =
    stimStart), one row per cell, sorted -- after Chen et al. 2013. This part follows the same filters as 2c-4c:
    1b-included DN sessions, engaged trials, and the cells that pass 2a's proximity filter AND this script's QC.
    Needs 2a's and 1b's out/ (skipped with a message if they are missing, or with --no-raster); the optional
    behavior-class strip reads 2g's out/ if it exists (a later step: on a first run it is simply absent).
    Deconvolved activity is averaged per (trial, cell, position bin) -- BINSIZE cm bins from -S_PRE to +S_POST -- and
    then over trials; bins with fewer than MIN_TRIALS_PER_BIN trials stay blank.
    ZSCORE_MODE: "profile" (z-score each cell's profile across bins, default) | "frames" (z-score each cell over
    the session's engaged frames before averaging) | "minmax" ([0,1] per profile, Chen et al. Supp. Fig. 6b) |
    "none". SORT_BY (plot time, no recompute): see infotheory.raster.SORTERS. CV_SORT=True: sort on odd trials and
    display even trials (a diagonal then only survives if tuning is reproducible; `reliability` = odd-vs-even profile
    correlation is saved per cell). Caveat: the position axis is NOT time -- slow stretches contribute more frames per
    cm; bins are narrow, which keeps this small.

PART 3 -- outcome analysis (sections 14-20), after van Vugt et al. 2018 (Science 360:537) and Rowland et al. 2023
    Same cells / sessions / engaged trials as the raster. Outcomes: HIT (signal>0, lick), MISS (signal>0, no lick),
    FA (signal==0, lick), CR (signal==0, no lick); stimulus trials split into difficult / intermediate / easy by the
    psychometric position of the contrast (stim_binning rule `pct_0-20_20-80_80-100`). Activity is z-scored per cell over
    the session's engaged frames. Hits and misses are CONTRAST-MATCHED within a category (hits re-weighted to the miss
    contrast distribution on the levels both have). Details in infotheory.outcome_analysis.
      14 traces   hit / miss (+ CR / FA dashed) along the corridor, per group and contrast category
      15 miss fraction   100 x mean response(miss) / mean response(hit), response = stim (0-20 cm) minus pre_stim (-30..-10 cm)
                  window mean, per group and category. Ratio of unit means (not mean of cell ratios); NaN if the hit
                  response is <= 0. Unstable when hit responses are small -- check the unit counts / activity columns.
      16 false alarm vs correct rejection   pre-stimulus activity and AUROC (FA vs CR) per cell, per group
      17-18 choice probability   per-cell AUROC hit vs miss, conditioned on contrast level (so contrast cannot drive it),
                  before (pre_stim) and after (stim) onset, and in every 5 cm position bin; intermediate contrasts (van Vugt)
                  and all contrasts; permutation test (--n-perm-outcome)
      19 prestimulus markers   AUROC of single pre_stim markers for hit-vs-miss and FA-vs-CR: population mean and variance
                  across cells of each group (Rowland: variance), run speed, pupil, motion energy, lick rate, time since
                  the previous trial. Session-level (the markers are per trial); mouse / pooled = pair-weighted mean
                  over the sessions
      20 hit probability by prestimulus population variance / mean (terciles within session; intermediate contrasts)
    Not done (needs simultaneous V1 and PM cells in one session = pair level, fits 3a-3c): van Vugt's V1->PM coupling.
    Caveats: misses/FA are few in well-trained sessions (see outcome_counts_by_session.csv; conditions with fewer than
    MIN_TRIALS_COND trials are blank); choice probability and miss fraction mix stimulus-evoked and lick/movement-related
    activity (the lick itself follows stimulus onset on hits). --no-outcome skips this part.

Project-wide rules (infotheory/project_config.py) -- what applies here
    * selection : --animals / --sessions restrict the sessions; a restricted run writes to out/selection/<tag>/ and
                  NEVER overwrites qc_per_cell.csv in out/ (read by every later step). QC is per cell, so selecting
                  sessions does not change any cell's result.
    * aggregate : --aggregate pooled|session|mouse = the unit for group comparisons: pooled = all cells of a group
                  together, no error bars; session / mouse = one value per session / mouse (MIN_CELLS_PER_UNIT cells
                  needed), bars = mean +- SEM over units, dots = units coloured by mouse. Always the six groups
                  (V1-lab, V1-unl, PM-lab, PM-unl, AL, RSP). Autocorrelogram and population-mean profiles:
                  mean +- SEM over units (pooled: over cells, shaded band omitted). Rasters: one pooled raster per
                  group always; in session / mouse mode additionally one raster figure per session / mouse.
                  Not affected (session-level or cell-level by nature): the QC histograms (which need the cell
                  distribution to pick thresholds), QC pass rate and Fchan2 artifact rate per session, the
                  failure-reason counts. No violin plots.
    * memory    : each session is loaded once per part (statistics; raster), reduced to small tables, cached per
                  session in store/ and freed. Peak RAM ~ n_jobs x one session.
    * binning   : not used (no discrete information measures here; the raster is binned in position only).

Outputs
-------
2_pipeline/2b_activity_statistics/
    out/    activity_per_neuron_summary.csv    one row per neuron: area, label, session_id, cell_id, animal_id, every stat
            qc_per_cell.csv                    session_id, cell_id, qc_pass, qc_fail_reason, per-reason flags (read by 2c...)
            qc_summary_by_group.csv            pass/fail counts by session x area x label
            activity_by_unit[__<aggregate>].csv  per group (and unit): n_cells + mean of each plotted statistic
            raster_cells.csv                   one row per raster cell: session, group, peak positions, reliability
            raster_profiles.npz                profiles (cells x bins) all / odd / even; row order = raster_cells.csv
            raster_reliability_by_unit[__<aggregate>].csv
            outcome_counts_by_session.csv, outcome_traces_by_group[__<aggregate>].csv, miss_fraction[__<aggregate>].csv,
            outcome_scalars_per_cell.csv, false_alarm_vs_cr_per_cell.csv, choice_probability_per_cell.csv,
            choice_probability_by_group[__<aggregate>].csv, choice_probability_trace[__<aggregate>].csv,
            prestim_marker_auroc_per_session.csv, prestim_marker_auroc[__<aggregate>].csv,
            hit_probability_by_prestim_marker[__<aggregate>].csv     (part 3)
            figures/*.png                      1-9 statistics/QC; 10-13 raster; 14-20 outcome analysis
    out/selection/<tag>/...                    the same for a restricted run
    store/  activity_session_<hash>.pkl, raster_session_<hash>.pkl, outcome_session_<hash>.pkl   per-session caches
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt

from infotheory.pipeline import get_pipeline_paths
from infotheory.session import discover_sessions, load_sessions
from infotheory.celldata_utils import (
    get_area_label, LABEL_SPLIT_AREAS, SIX_GROUP_ORDER, SIX_GROUP_LABELS, SIX_GROUP_COLORS,
)
from infotheory.groups import add_six_group_column, six_panel_figure, group_key
from infotheory import spike_stats as ss
from infotheory import qc_lib
from infotheory import raster as rs
from infotheory import tensors as tn
from infotheory.plotting import set_style, save_fig, clear_figures
from infotheory.scope import add_scope_args, scope_from_args, select_sessions, scoped_dirs, data_stamp, unit_noun
from infotheory.session_jobs import run_sessions
from infotheory.unit_stats import group_unit_table, stable_seed
from infotheory.unit_plots import (group_unit_bars, grouped_series_bars, grouped_unit_value_bars, unit_bars,
                                   mouse_palette, mouse_legend)
from infotheory.stim_binning import STIM_BINNING_STRATEGIES, target_signature
from infotheory import outcome_analysis as oa

PROTOCOL = "DN"

# ---- part 1: statistics / QC ----------------------------------------------------------------------------
FANO_WINDOW_SEC = 1.0
MAX_LAG_SEC_AUTOCORR = 2.0
MAX_NEURONS_FOR_IEI_AUTOCORR = 300   # per session (IEI/autocorrelogram loop per neuron in Python)
RANDOM_STATE = 0
FCHAN2_ARTIFACT_Z = 3.0              # |Fchan2| above this = candidate motion/z-drift artifact frame
QC_THRESHOLDS = dict(
    rate_thr=qc_lib.RATE_THR, noise_thr=qc_lib.NOISE_THR, fano_thr=qc_lib.FANO_THR,
    skew_min=qc_lib.SKEW_MIN, flat_std_thr=qc_lib.FLAT_STD_THR, nan_frac_thr=qc_lib.NAN_FRAC_THR,
)
MIN_CELLS_PER_UNIT = 5               # session / mouse aggregate: cells needed in a group for a unit's value to be used
N_JOBS_NEURONS = 1                   # keep 1: sessions are already parallel

# ---- part 2: raster -------------------------------------------------------------------------------------
RUN_RASTER = True
S_PRE, S_POST = 80.0, 60.0           # cm before / after the stimulus onset
BINSIZE = 5.0                        # cm
MIN_TRIALS_PER_BIN = 10              # bins with fewer trials are left blank
ZSCORE_MODE = "profile"              # "profile" | "frames" | "minmax" | "none"
SORT_BY = "peak_position"            # key of infotheory.raster.SORTERS
CV_SORT = True                       # sort on odd trials, display even trials
RELIABILITY_MIN = 0.3                # cells below this are "unreliable" in the peak histograms
VMIN, VMAX = (0.0, 1.0) if ZSCORE_MODE == "minmax" else (-1.5, 3.0)
CLASS_STRIP = True                   # strip with the 2g behavior class beside each raster row (if 2g has run)
CMAP = "viridis"
MAX_CELLS_PER_SESSION = None
JOBLIB_VERBOSE = 5

# ---- part 3: outcome analysis (van Vugt et al. 2018) ----------------------------------------------------
RUN_OUTCOME = True
OUTCOME_STRATEGY = "pct_0-20_20-80_80-100"   # contrast categories: difficult / intermediate / easy (3 bins)
OUTCOME_WINDOWS = {"pre_stim": (-30.0, -10.0), "stim": (0.0, 20.0)}   # cm relative to stimStart (as 2c/2d)
EQUALIZE_QUANTILE = 0.1              # position-matched frame sampling for the window means (as 2d's ROC)
MIN_TRIALS_COND = 5                  # trials needed in an outcome condition (per cell) for a trace / value
MIN_PER_CLASS = 5                    # trials per class (and contrast level) for AUROC
MIN_POP_CELLS = 5                    # cells of a group in a session for its population markers
ALPHA = 0.05
N_MARKER_BINS = 3                    # prestimulus-variance bins for hit probability (per session)
MIN_TRIALS_PER_MARKER_BIN = 5
OUTCOME_COLORS = {"HIT": "#2ca02c", "MISS": "#d62728", "CR": "black", "FA": "#1f77b4"}

parser = argparse.ArgumentParser()
add_scope_args(parser)
parser.add_argument("--sort-by", default=SORT_BY, choices=sorted(rs.SORTERS), help="raster sorting (no recompute)")
parser.add_argument("--no-raster", action="store_true", help="skip part 2 (raster)")
parser.add_argument("--no-outcome", action="store_true", help="skip part 3 (outcome analysis)")
parser.add_argument("--n-perm-outcome", type=int, default=200, help="permutations for the per-cell choice probability")
args = parser.parse_args()
scope = scope_from_args(args)
AGG = scope.aggregate
UNIT = unit_noun(AGG)
UNITS = "mice" if UNIT == "mouse" else f"{UNIT}s"   # plural for labels
N_JOBS = scope.n_jobs
AGG_SUFFIX = "" if AGG == "pooled" else f"__{AGG}"
UNIT_KEYS = {"pooled": [], "session": ["animal_id", "session_id"], "mouse": ["animal_id"]}[AGG]
unit_col = {"pooled": None, "session": "session_id", "mouse": "animal_id"}[AGG]
sort_by = args.sort_by
display_z = ZSCORE_MODE if ZSCORE_MODE in ("profile", "minmax") else "none"

paths = get_pipeline_paths(__file__)
table_dir, figdir = scoped_dirs(paths, scope)
clear_figures(figdir)
set_style()
print(scope.describe())

GROUP_KEYS = [group_key(g) for g in SIX_GROUP_ORDER]
GROUP_COLOR = {group_key(g): SIX_GROUP_COLORS[g] for g in SIX_GROUP_ORDER}

all_triples = discover_sessions([PROTOCOL])
triples = select_sessions(all_triples, scope)
session_ids = [f"{a}_{d}" for _, a, d in triples]
stamps = {f"{a}_{d}": data_stamp(p, a, d) for p, a, d in triples}
mouse_colors = mouse_palette(a for _, a, _ in all_triples)


# --------------------------------------------------------------------- #
# 1. Per-session statistics (one session in RAM at a time, cached per session).
#    No upstream inclusion filter: every selected DN session with celldata + deconvdata is used.
# --------------------------------------------------------------------- #
def process_session(ses, seed):
    """All per-session computation, returned as small picklable tables."""
    if ses.celldata is None or ses.calciumdata is None:
        return None
    rng = np.random.default_rng(RANDOM_STATE + seed)
    fs = ss.get_frame_rate(ses, fallback=8.0)
    calciumdata = ses.calciumdata.to_numpy()
    celldata = ses.celldata.reset_index(drop=True)
    area, label, _ = get_area_label(celldata, label_split_areas=LABEL_SPLIT_AREAS)
    T, N = calciumdata.shape

    rate = ss.event_rate(calciumdata, fs)
    active_rate = ss.active_frame_rate(calciumdata, fs)
    spars = ss.sparsity(calciumdata)
    fano = ss.fano_factor(calciumdata, fs, window_sec=FANO_WINDOW_SEC)
    active_fano = ss.fano_factor(calciumdata, fs, window_sec=FANO_WINDOW_SEC, threshold=0.0)
    qc = qc_lib.compute_qc_metrics(calciumdata, fs, fano_window_sec=FANO_WINDOW_SEC)

    # movement-artifact check: real per-frame timestamps (ses.ts_F) when available
    runspeed_corr = np.full(N, np.nan)
    if ses.behaviordata is not None and "runspeed" in ses.behaviordata.columns and T > 1:
        ts_bhv = ses.behaviordata["ts"].to_numpy()
        if ses.ts_F is not None and len(ses.ts_F) == T:
            ts_F = ses.ts_F
        else:
            duration = ts_bhv[-1] - ts_bhv[0] if len(ts_bhv) > 1 else T / fs
            ts_F = np.linspace(0, duration, T) + ts_bhv[0]
        runspeed_F = np.interp(ts_F, ts_bhv, ses.behaviordata["runspeed"].to_numpy())
        runspeed_corr = ss.runspeed_correlation(calciumdata, runspeed_F)

    fchan2_corr = np.full(N, np.nan)
    fchan2_artifact_frac = np.nan
    if ses.fchan2 is not None and len(ses.fchan2) == T:
        fchan2_corr = ss.runspeed_correlation(calciumdata, ses.fchan2)
        fchan2_artifact_frac = float(np.mean(np.abs(ses.fchan2) > FCHAN2_ARTIFACT_Z))

    subsample_idx = np.arange(N)
    if N > MAX_NEURONS_FOR_IEI_AUTOCORR:
        subsample_idx = np.sort(rng.choice(N, size=MAX_NEURONS_FOR_IEI_AUTOCORR, replace=False))
    calciumdata_sub = calciumdata[:, subsample_idx]
    median_iei, cv_iei, _ = ss.iei_stats(calciumdata_sub, fs, n_jobs=N_JOBS_NEURONS)
    lags, ac_sub = ss.autocorrelogram(calciumdata_sub, fs, max_lag_sec=MAX_LAG_SEC_AUTOCORR, n_jobs=N_JOBS_NEURONS)

    per_neuron = pd.DataFrame({
        "session_id": ses.session_id, "animal_id": ses.animal_id, "cell_id": celldata["cell_id"].to_numpy(),
        "roi_name": area, "labeled": label,
        "mean_activity_per_sec": rate, "active_frame_rate_hz": active_rate, "sparsity": spars,
        "fano_factor_magnitude": fano, "fano_factor_active_frames": active_fano,
        "runspeed_corr": runspeed_corr, "fchan2_corr": fchan2_corr,
        "noise_level": celldata["noise_level"].to_numpy(),
    })
    per_neuron = pd.concat([per_neuron, qc.reset_index(drop=True)], axis=1)
    sub_id = {"session_id": ses.session_id, "animal_id": ses.animal_id,
              "cell_id": celldata["cell_id"].to_numpy()[subsample_idx],
              "roi_name": area[subsample_idx], "labeled": label[subsample_idx]}
    iei_sub = pd.DataFrame({**sub_id, "median_iei": median_iei, "cv_iei": cv_iei})
    ac = pd.concat([pd.DataFrame(sub_id), pd.DataFrame(ac_sub, columns=[f"lag_{j}" for j in range(ac_sub.shape[1])])],
                   axis=1)
    print(f"  [{ses.session_id}] fs={fs:.2f} Hz, {N} neurons, {len(subsample_idx)} subsampled for IEI/autocorr")
    return {"per_neuron": per_neuron, "iei_sub": iei_sub, "ac": ac, "lags": lags,
            "session_summary": {"session_id": ses.session_id, "animal_id": ses.animal_id,
                                "fchan2_artifact_frac": fchan2_artifact_frac}}


results = run_sessions(
    session_ids, process_session, [PROTOCOL], paths.store, "activity_session",
    key_params=dict(version="2b-v2", fano=FANO_WINDOW_SEC, max_lag=MAX_LAG_SEC_AUTOCORR,
                    max_iei=MAX_NEURONS_FOR_IEI_AUTOCORR, rs=RANDOM_STATE, fchan2_z=FCHAN2_ARTIFACT_Z),
    load_kwargs=dict(load_behaviordata=True, load_videodata=False, load_celldata=True, load_calciumdata=True),
    n_jobs=N_JOBS, recompute=args.recompute, verbose=JOBLIB_VERBOSE, extra_key_by_session=stamps, flatten=False)
results = [r for r in results if r is not None]
if not results:
    raise SystemExit(f"No selected {PROTOCOL} session has both celldata and deconvdata -- nothing to analyze.")

per_neuron = pd.concat([r["per_neuron"] for r in results], ignore_index=True)
iei_sub = pd.concat([r["iei_sub"] for r in results], ignore_index=True)
session_summary = pd.DataFrame([r["session_summary"] for r in results])
reference_lags = results[0]["lags"]
ac_parts = []
for r in results:
    ac = r["ac"]
    if len(r["lags"]) != len(reference_lags) or not np.allclose(r["lags"], reference_lags):
        lag_cols_r = [c for c in ac.columns if c.startswith("lag_")]
        vals = np.vstack([np.interp(reference_lags, r["lags"], row) for row in ac[lag_cols_r].to_numpy()])
        ac = pd.concat([ac.drop(columns=lag_cols_r),
                        pd.DataFrame(vals, columns=[f"lag_{j}" for j in range(vals.shape[1])])], axis=1)
    ac_parts.append(ac)
ac_df = pd.concat(ac_parts, ignore_index=True)
del results, ac_parts

per_neuron = add_six_group_column(per_neuron)
iei_sub = add_six_group_column(iei_sub)
ac_df = add_six_group_column(ac_df)

n_sessions = per_neuron["session_id"].nunique()
print(f"\n{len(per_neuron)} neurons across {n_sessions} {PROTOCOL} sessions (no exclusions applied -- full population)")


# --------------------------------------------------------------------- #
# 2. QC flagging -- computed and reported, never used to filter anything in part 1
# --------------------------------------------------------------------- #
per_neuron_qc = qc_lib.flag_bad_cells(per_neuron, **QC_THRESHOLDS)
per_neuron_qc.drop(columns=["area", "label", "group", "group_name"]).to_csv(
    table_dir / "activity_per_neuron_summary.csv", index=False)

qc_summary = qc_lib.summarize_qc(per_neuron_qc)
qc_summary.to_csv(table_dir / "qc_summary_by_group.csv", index=False)
print(f"\nOverall: {int(per_neuron_qc['qc_pass'].sum())}/{len(per_neuron_qc)} neurons would pass under the current "
      f"QC_THRESHOLDS (informational only -- nothing is excluded by part 1)")
print("\nWorst 10 (session, area, label) groups by pass fraction:")
print(qc_summary[["session_id", "roi_name", "labeled", "n_total", "n_pass", "frac_pass"]].head(10).to_string(index=False))

qc_per_cell = per_neuron_qc[["session_id", "cell_id", "qc_pass", "qc_fail_reason"]
                            + [f"qc_flag_{r}" for r in qc_lib.REASON_COLS]]
qc_per_cell.to_csv(table_dir / "qc_per_cell.csv", index=False)

# report: every plotted per-cell statistic by group and unit
stat_cols = ["active_frame_rate_hz", "sparsity", "fano_factor_active_frames", "runspeed_corr", "fchan2_corr",
             "mean_activity_per_sec", "qc_pass"]
act_tab = group_unit_table(per_neuron_qc.assign(qc_pass=per_neuron_qc["qc_pass"].astype(float)), stat_cols, AGG)
iei_tab = group_unit_table(iei_sub, ["median_iei", "cv_iei"], AGG).rename(columns={"n_cells": "n_cells_iei"})
merge_keys = ["group"] + (["unit", "animal_id"] if AGG != "pooled" else [])
act_tab.merge(iei_tab, on=merge_keys, how="left").rename(columns={"qc_pass": "frac_qc_pass"}).to_csv(
    table_dir / f"activity_by_unit{AGG_SUFFIX}.csv", index=False)


# --------------------------------------------------------------------- #
# 3. Figure 1: basic activity statistics by group, at the unit of analysis
# --------------------------------------------------------------------- #
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
group_unit_bars(axes[0, 0], per_neuron, "active_frame_rate_hz", AGG, mouse_colors, MIN_CELLS_PER_UNIT,
                ylabel="active-frame rate (events/s)")
group_unit_bars(axes[0, 1], per_neuron, "sparsity", AGG, mouse_colors, MIN_CELLS_PER_UNIT,
                ylabel="sparsity (fraction of frames active)")
group_unit_bars(axes[1, 0], iei_sub.dropna(subset=["median_iei"]), "median_iei", AGG, mouse_colors,
                MIN_CELLS_PER_UNIT, log_y=True, ylabel="median inter-event interval (s)")
group_unit_bars(axes[1, 1], per_neuron, "fano_factor_active_frames", AGG, mouse_colors, MIN_CELLS_PER_UNIT,
                ylabel=f"Fano factor of active-frame counts ({FANO_WINDOW_SEC:.0f}s windows)", hline=1.0)
if AGG != "pooled":
    mouse_legend(axes[0, 0], mouse_colors, animals=per_neuron["animal_id"].unique(), loc="upper right")
fig.suptitle(f"Basic activity statistics -- {PROTOCOL}, {n_sessions} sessions (full population; unit = {UNIT})")
fig.tight_layout()
save_fig(fig, figdir, "1_basic_activity_stats")


# --------------------------------------------------------------------- #
# 4. Figure 2: autocorrelogram, mean +- SEM over units per group (pooled: mean over cells, no band)
# --------------------------------------------------------------------- #
lag_cols = [c for c in ac_df.columns if c.startswith("lag_")]
fig, ax = plt.subplots(figsize=(7.5, 5))
for g in SIX_GROUP_ORDER:
    sub = ac_df[ac_df["group"] == group_key(g)]
    if sub.empty:
        continue
    color = SIX_GROUP_COLORS[g]
    if AGG == "pooled":
        mean_ac = np.nanmean(sub[lag_cols].to_numpy(), axis=0)
        ax.plot(reference_lags, mean_ac, color=color, label=f"{SIX_GROUP_LABELS[g]} (n={len(sub)} cells)")
        continue
    cnt = sub.groupby(UNIT_KEYS).size()
    unit_means = sub.groupby(UNIT_KEYS)[lag_cols].mean()[cnt >= MIN_CELLS_PER_UNIT]
    if unit_means.empty:
        continue
    for _, row in unit_means.iterrows():
        ax.plot(reference_lags, row.to_numpy(), color=color, alpha=0.15, linewidth=0.7)
    m = unit_means.mean().to_numpy()
    sem = unit_means.sem().fillna(0.0).to_numpy() if len(unit_means) > 1 else np.zeros_like(m)
    ax.plot(reference_lags, m, color=color, label=f"{SIX_GROUP_LABELS[g]} (n={len(unit_means)} {UNITS})")
    ax.fill_between(reference_lags, m - sem, m + sem, color=color, alpha=0.2)
ax.axhline(0, color="gray", linewidth=0.5)
ax.set_xlabel("lag (s)")
ax.set_ylabel("autocorrelation")
ax.set_title(f"Deconvolved-trace autocorrelogram -- {PROTOCOL} (unit = {UNIT})")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
save_fig(fig, figdir, "2_autocorrelogram")


# --------------------------------------------------------------------- #
# 5. Figure 3: QC metric distributions per group (cell-level histograms: they are what the thresholds are
#    picked from), coloured pass / flagged; dashed = QC_THRESHOLDS
# --------------------------------------------------------------------- #
qc_metrics = [("qc_rate", "activity rate (qc_rate)", QC_THRESHOLDS["rate_thr"]),
              ("noise_level", "noise_level", QC_THRESHOLDS["noise_thr"]),
              ("qc_skew", "trace skewness", QC_THRESHOLDS["skew_min"]),
              ("qc_fano", "Fano factor (count-based)", QC_THRESHOLDS["fano_thr"]),
              ("qc_nan_frac", "fraction NaN", QC_THRESHOLDS["nan_frac_thr"])]
n_groups, n_metrics = len(SIX_GROUP_ORDER), len(qc_metrics)
fig, axes = plt.subplots(n_groups, n_metrics, figsize=(3.6 * n_metrics, 2.4 * n_groups), squeeze=False)
for i, g in enumerate(SIX_GROUP_ORDER):
    sub = per_neuron_qc[per_neuron_qc["group"] == group_key(g)]
    for j, (metric, mlabel, thr) in enumerate(qc_metrics):
        ax = axes[i, j]
        if j == 0:
            ax.set_ylabel(f"{SIX_GROUP_LABELS[g]}\n(n={len(sub)}, pass={int(sub['qc_pass'].sum())})", fontsize=7,
                          color=SIX_GROUP_COLORS[g])
        if i == 0:
            ax.set_title(mlabel, fontsize=8)
        if i == n_groups - 1:
            ax.set_xlabel(mlabel, fontsize=7)
        all_vals = sub[metric].dropna().to_numpy()
        if len(all_vals) == 0:
            ax.text(0.5, 0.5, "no cells", transform=ax.transAxes, ha="center", fontsize=7)
            continue
        lo, hi = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        if lo == hi:
            lo, hi = lo - 1, hi + 1
        bins = np.linspace(lo, hi, 31)
        ax.hist(sub.loc[sub["qc_pass"], metric].dropna(), bins=bins, color="tab:green", alpha=0.6, label="pass")
        ax.hist(sub.loc[~sub["qc_pass"], metric].dropna(), bins=bins, color="tab:red", alpha=0.6, label="flagged")
        ax.set_xlim(lo, hi)
        if lo <= thr <= hi:
            ax.axvline(thr, color="black", linestyle="--", linewidth=1)
        else:   # threshold outside this group's range: note it instead of squashing the data
            ax.annotate(f"thr={thr:g}\n(off-axis)", xy=(0.97, 0.92), xycoords="axes fraction", fontsize=5,
                        ha="right", va="top", color="black")
axes[0, 0].legend(fontsize=6, loc="upper right")
fig.suptitle(f"QC metric distributions by group (dashed = current threshold) -- {PROTOCOL}", y=1.01)
fig.tight_layout()
save_fig(fig, figdir, "3_qc_metrics_by_group")


# --------------------------------------------------------------------- #
# 6. Figure 4: QC pass fraction per session (session-level diagnostic)
# --------------------------------------------------------------------- #
frac_pass_session = per_neuron_qc.groupby("session_id")["qc_pass"].mean().sort_values()
fig, ax = plt.subplots(figsize=(6, max(3, 0.25 * len(frac_pass_session))))
ax.barh(np.arange(len(frac_pass_session)), frac_pass_session.to_numpy(), color="grey")
ax.set_yticks(np.arange(len(frac_pass_session)))
ax.set_yticklabels(frac_pass_session.index, fontsize=6)
ax.set_xlabel("fraction of neurons passing QC")
ax.set_xlim(0, 1.02)
ax.set_title(f"QC pass rate per session -- {PROTOCOL}")
fig.tight_layout()
save_fig(fig, figdir, "4_qc_pass_rate_per_session")


# --------------------------------------------------------------------- #
# 7. Figure 5: QC failure reasons (counts of flagged neurons), stacked, per group
# --------------------------------------------------------------------- #
flag_cols = [f"qc_flag_{r}" for r in qc_lib.REASON_COLS]
grouped = per_neuron_qc.groupby("group")[flag_cols].sum().reindex(GROUP_KEYS).fillna(0)
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(grouped))
bottom = np.zeros(len(grouped))
for r in qc_lib.REASON_COLS:
    vals = grouped[f"qc_flag_{r}"].to_numpy()
    ax.bar(x, vals, bottom=bottom, color=qc_lib.REASON_COLORS[r], label=qc_lib.REASON_LABELS[r])
    bottom += vals
ax.set_xticks(x)
ax.set_xticklabels([SIX_GROUP_LABELS[g] for g in SIX_GROUP_ORDER], fontsize=8)
ax.set_ylabel("number of neurons flagged")
ax.set_title(f"QC failure reasons by group -- {PROTOCOL}")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
save_fig(fig, figdir, "5_qc_failure_reasons")


# --------------------------------------------------------------------- #
# 8. Figure 6: running-speed correlation (movement-artifact / locomotion check)
# --------------------------------------------------------------------- #
if per_neuron["runspeed_corr"].notna().any():
    fig, ax = plt.subplots(figsize=(7.5, 5))
    group_unit_bars(ax, per_neuron, "runspeed_corr", AGG, mouse_colors, MIN_CELLS_PER_UNIT,
                    ylabel="correlation with running speed (r)", hline=0.0)
    ax.set_title(f"Locomotion-related activity -- {PROTOCOL} (unit = {UNIT})")
    fig.tight_layout()
    save_fig(fig, figdir, "6_runspeed_correlation")
else:
    print("\nNo behaviordata/runspeed available -- skipped Figure 6.")


# --------------------------------------------------------------------- #
# 9. Fchan2 (red-channel / motion-artifact) battery. Fchan2 is a SESSION-WIDE signal (z-scored red-channel
#    change; abrupt changes flag likely z-motion/refocusing), so: an example trace, the artifact rate per session,
#    and whether any cell's activity tracks it.
# --------------------------------------------------------------------- #
example_session_id = per_neuron.loc[per_neuron["fchan2_corr"].notna(), "session_id"]
example_session_id = example_session_id.iloc[0] if len(example_session_id) else None
if example_session_id is not None:
    example_ses = load_sessions(protocols=[PROTOCOL], load_behaviordata=False, load_videodata=False,
                                load_celldata=False, load_calciumdata=True,
                                only_session_ids=[example_session_id], verbose=False)[0]   # ONE session
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if example_ses.ts_F is not None:
        axes[0].plot(example_ses.ts_F, example_ses.fchan2, color="firebrick", linewidth=0.5)
        axes[0].set_xlabel("session time (s)")
    else:
        axes[0].plot(example_ses.fchan2, color="firebrick", linewidth=0.5)
        axes[0].set_xlabel("frame")
    axes[0].axhline(FCHAN2_ARTIFACT_Z, color="black", linestyle="--", linewidth=1)
    axes[0].axhline(-FCHAN2_ARTIFACT_Z, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Fchan2 (z-scored)")
    axes[0].set_title(f"Example session: {example_session_id}")
    axes[1].hist(example_ses.fchan2, bins=60, color="firebrick", alpha=0.7)
    axes[1].axvline(FCHAN2_ARTIFACT_Z, color="black", linestyle="--", linewidth=1, label=f"|z| > {FCHAN2_ARTIFACT_Z:g}")
    axes[1].axvline(-FCHAN2_ARTIFACT_Z, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Fchan2 (z-scored)")
    axes[1].set_ylabel("frames")
    axes[1].legend(fontsize=8, frameon=False)
    axes[1].set_title("Distribution (example session)")
    fig.suptitle(f"Fchan2 (red-channel motion-artifact signal) -- {PROTOCOL}", y=1.03)
    fig.tight_layout()
    save_fig(fig, figdir, "7_fchan2_example")
    del example_ses

if session_summary["fchan2_artifact_frac"].notna().any():
    sess_frac = session_summary.dropna(subset=["fchan2_artifact_frac"]).sort_values("fchan2_artifact_frac")
    fig, ax = plt.subplots(figsize=(6, max(3, 0.25 * len(sess_frac))))
    ax.barh(np.arange(len(sess_frac)), sess_frac["fchan2_artifact_frac"].to_numpy(), color="firebrick")
    ax.set_yticks(np.arange(len(sess_frac)))
    ax.set_yticklabels(sess_frac["session_id"], fontsize=6)
    ax.set_xlabel(f"fraction of frames with |Fchan2| > {FCHAN2_ARTIFACT_Z:g}")
    ax.set_title(f"Motion-artifact frame rate per session -- {PROTOCOL}")
    fig.tight_layout()
    save_fig(fig, figdir, "8_fchan2_artifact_rate_per_session")

if per_neuron["fchan2_corr"].notna().any():
    fig, ax = plt.subplots(figsize=(7.5, 5))
    group_unit_bars(ax, per_neuron, "fchan2_corr", AGG, mouse_colors, MIN_CELLS_PER_UNIT,
                    ylabel="correlation with Fchan2 (r)", hline=0.0)
    ax.set_title(f"Does activity track the session-wide motion-artifact signal? -- {PROTOCOL} (unit = {UNIT})")
    fig.tight_layout()
    save_fig(fig, figdir, "9_fchan2_correlation")
else:
    print("\nNo Fchan2 data available -- skipped Figures 7-9.")


# =====================================================================================================
# PART 2 -- ACTIVITY RASTER (was 2i_activity_raster)
# =====================================================================================================
def raster_inputs():
    """(raster session ids, {session: cell ids}, 2a celldata table) or None (with a message) if inputs are missing."""
    fits_path = paths.out_from("1b_psychometric") / "psychometric_fits.csv"
    cd_path = paths.out_from("2a_cell_distribution") / "celldata_combined.csv"
    for f in (fits_path, cd_path):
        if not f.exists():
            print(f"\nRaster skipped: run the step that writes {f} first.")
            return None
    fits_df = pd.read_csv(fits_path)
    inc = fits_df[fits_df["included"] & (fits_df["protocol"] == PROTOCOL)]["session_id"].tolist()
    ids = [sid for sid in session_ids if sid in set(inc)]
    if not ids:
        print(f"\nRaster skipped: no selected {PROTOCOL} session passed 1b_psychometric's inclusion criteria.")
        return None
    cd = pd.read_csv(cd_path)
    # same rule as neural_encoding.load_cell_inclusion: 2a's proximity filter AND this script's QC
    incl = cd[["session_id", "cell_id", "passes_proximity_filter"]].merge(
        qc_per_cell[["session_id", "cell_id", "qc_pass"]], on=["session_id", "cell_id"], how="inner")
    incl["include"] = incl["passes_proximity_filter"] & incl["qc_pass"]
    cells_by = {sid: sorted(sub.loc[sub["include"], "cell_id"]) for sid, sub in incl.groupby("session_id")}
    ids = [sid for sid in ids if cells_by.get(sid)]
    print(f"\nRaster: {len(ids)} included {PROTOCOL} sessions, {int(incl['include'].sum())}/{len(incl)} cells pass 2a + QC")
    for sid in ids:
        if MAX_CELLS_PER_SESSION is not None and len(cells_by[sid]) > MAX_CELLS_PER_SESSION:
            r = np.random.default_rng(stable_seed("raster-cells", sid))
            cells_by[sid] = sorted(r.choice(cells_by[sid], MAX_CELLS_PER_SESSION, replace=False))
    return ids, {sid: cells_by[sid] for sid in ids}, cd, fits_df.set_index("session_id")[["mu", "sigma"]].to_dict("index")


want_raster = not args.no_raster and RUN_RASTER
want_outcome = not args.no_outcome and RUN_OUTCOME
ri = raster_inputs() if (want_raster or want_outcome) else None

if ri is not None and want_raster:
    raster_ids, cells_by_session, cd_table, _ = ri

    def process_raster(ses, seed):
        cell_ids = cells_by_session[ses.session_id]
        if ses.calciumdata is None or ses.ts_F is None or ses.trialdata is None:
            return None
        A = ses.calciumdata.loc[:, cell_ids].to_numpy(dtype=float)
        trials = ses.trialdata.reset_index(drop=True)
        trials = trials[trials["engaged"] == 1].reset_index(drop=True)
        if len(trials) < 2 * MIN_TRIALS_PER_BIN:
            return None
        frame_ok = tn.engaged_frame_mask(ses.ts_F, ses.trialdata)
        zpos_F = tn.interp_to_imaging_frames(ses.behaviordata["ts"], ses.behaviordata["zpos"], ses.ts_F)
        k, rel = tn.position_relative_to_stim(zpos_F, trials["stimStart"].to_numpy(), S_PRE, S_POST)
        if ZSCORE_MODE == "frames":
            A = tn.zscore_frames(A, frame_ok)
        tensor, counts, centers = tn.build_position_tensor(A, k, rel, frame_ok, len(trials), S_PRE, S_POST, BINSIZE)
        hp = rs.half_profiles(tensor, min_trials=MIN_TRIALS_PER_BIN)
        return dict(session_id=ses.session_id, animal_id=ses.animal_id, cell_ids=list(cell_ids), centers=centers,
                    all=hp["all"], odd=hp["odd"], even=hp["even"], n_per_bin=hp["n_all"], n_trials=len(trials))

    rasters = run_sessions(
        raster_ids, process_raster, [PROTOCOL], paths.store, "raster_session",
        key_params=dict(version="2b-raster-v1", s_pre=S_PRE, s_post=S_POST, binsize=BINSIZE,
                        min_trials=MIN_TRIALS_PER_BIN, zscore=ZSCORE_MODE),
        load_kwargs=dict(load_behaviordata=True, load_videodata=False, load_celldata=True, load_calciumdata=True),
        cells_by_session=cells_by_session, n_jobs=N_JOBS, recompute=args.recompute, verbose=JOBLIB_VERBOSE,
        extra_key_by_session={sid: stamps[sid] for sid in raster_ids}, flatten=False)
    sessions = [r for r in rasters if r is not None]
    del rasters
    if not sessions:
        print("Raster skipped: no session with enough engaged trials / cells.")
        want_raster = False

if ri is not None and want_raster:
    centers = sessions[0]["centers"]

    # ----------------------------------------------------------------- #
    # 10. Stack all cells; attach groups; per-cell summary table
    # ----------------------------------------------------------------- #
    cells = pd.concat([pd.DataFrame({"session_id": s["session_id"], "animal_id": s["animal_id"],
                                     "cell_id": s["cell_ids"]}) for s in sessions], ignore_index=True)
    profiles = {name: np.vstack([s[name] for s in sessions]) for name in ("all", "odd", "even")}
    meta = cd_table[["session_id", "cell_id", "roi_name", "labeled"]]
    cells = add_six_group_column(cells.merge(meta, on=["session_id", "cell_id"], how="left"))
    assert len(cells) == len(profiles["all"])

    class_path = paths.out_from("2g_behavior_correlation") / "behavior_class.csv"
    strip_on = CLASS_STRIP and class_path.exists()
    if strip_on:
        cells = cells.merge(pd.read_csv(class_path)[["session_id", "cell_id", "class"]],
                            on=["session_id", "cell_id"], how="left")
        cells["class"] = cells["class"].fillna("unclassified")
        assert len(cells) == len(profiles["all"])
    else:
        cells["class"] = "unclassified"
        if CLASS_STRIP:
            print("(no 2g behavior_class.csv yet -- raster drawn without the class strip)")

    for name in ("all", "odd", "even"):
        _, key = rs.sort_order(rs.zscore_profiles(profiles[name], "profile"), centers, "peak_position")
        cells[f"peak_position_{name}"] = key
    cells["peak_z_all"] = np.nanmax(rs.zscore_profiles(profiles["all"], "profile"), axis=1)
    cells["reliability"] = rs.reliability(profiles["odd"], profiles["even"])
    cells.to_csv(table_dir / "raster_cells.csv", index=False)
    np.savez_compressed(table_dir / "raster_profiles.npz", centers=centers, **profiles)
    print(f"{len(cells)} raster cells in {len(sessions)} sessions; saved profiles {profiles['all'].shape}")

    cells["reliable"] = (cells["reliability"] >= RELIABILITY_MIN).astype(float)
    group_unit_table(cells, ["reliability", "peak_z_all", "reliable"], AGG).rename(
        columns={"reliable": "frac_reliable"}).to_csv(table_dir / f"raster_reliability_by_unit{AGG_SUFFIX}.csv",
                                                     index=False)
    print(f"\nMedian reliability (odd vs even trial profile correlation) per group (all cells):")
    print(cells.groupby("group")["reliability"].median().reindex(GROUP_KEYS).round(2).to_string())

    # ----------------------------------------------------------------- #
    # 11. Rasters (always the six groups)
    # ----------------------------------------------------------------- #
    def make_raster(idx):
        sub = {n: profiles[n][idx] for n in ("all", "odd", "even")}
        return rs.build_raster(sub, centers, method=sort_by, cv=CV_SORT, zscore=display_z)

    CLASS_ORDER = ["runspeed", "pupil_area", "lick", "mixed", "unclassified"]
    CLASS_CODES = {c: i for i, c in enumerate(CLASS_ORDER)}
    CLASS_CMAP = matplotlib.colors.ListedColormap(
        [plt.get_cmap("tab10")(0), plt.get_cmap("tab10")(1), plt.get_cmap("tab10")(2), "#8c564b", "#dddddd"])

    def draw_group_rasters(row_mask, title, fname):
        fig, axd = six_panel_figure(figsize_per_panel=(3.6, 3.6), sharex=True, sharey=False)
        cmap = plt.get_cmap(CMAP).with_extremes(bad="#dddddd")
        im = None
        for g, ax in axd.items():
            idx = np.where(row_mask & (cells["group"] == group_key(g)).to_numpy())[0]
            if len(idx) == 0:
                ax.text(0.5, 0.5, "no cells", transform=ax.transAxes, ha="center")
                continue
            out = make_raster(idx)
            im = ax.imshow(out["display"], aspect="auto", cmap=cmap, vmin=VMIN, vmax=VMAX,
                           extent=[centers[0] - BINSIZE / 2, centers[-1] + BINSIZE / 2, out["n_cells"], 0],
                           interpolation="nearest")
            ax.axvline(0, color="w", lw=0.8, ls="--")
            if strip_on:
                strip = ax.inset_axes([-0.07, 0, 0.04, 1])
                codes = np.array([CLASS_CODES[c] for c in cells["class"].to_numpy()[idx][out["order"]]])
                strip.imshow(codes[:, None], aspect="auto", cmap=CLASS_CMAP, vmin=0, vmax=len(CLASS_CODES) - 1,
                             interpolation="nearest")
                strip.axis("off")
                ax.tick_params(axis="y", pad=14)
            ax.set_title(f"{SIX_GROUP_LABELS[g]}  (n={out['n_cells']})", color=SIX_GROUP_COLORS[g], fontsize=10)
            ax.set_xlabel("position from stimulus onset (cm)", fontsize=8)
            ax.set_ylabel("cells", fontsize=8)
        if im is not None:
            fig.colorbar(im, ax=list(axd.values()), shrink=0.6,
                         label="normalized activity" if ZSCORE_MODE == "minmax" else "z-scored activity")
        if strip_on:
            handles = [plt.Line2D([], [], marker="s", ls="", color=CLASS_CMAP(i), label=c) for i, c in enumerate(CLASS_ORDER)]
            fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7, frameon=False,
                       title="behavior class (strip left of each raster, from 2g)", title_fontsize=7)
        sort_note = (f"sorted by {sort_by} on odd trials, shown on even" if CV_SORT else f"sorted by {sort_by} (same data)")
        fig.suptitle(f"{title}\n{sort_note}; blank = <{MIN_TRIALS_PER_BIN} trials", fontsize=10)
        save_fig(fig, figdir, fname)
        plt.close(fig)

    draw_group_rasters(np.ones(len(cells), bool), f"Trial-averaged activity, all selected sessions pooled ({PROTOCOL})",
                       "10_raster_pooled")
    if AGG != "pooled":   # one raster figure per session / per mouse
        for u in sorted(cells[unit_col].unique()):
            n_ses = cells.loc[cells[unit_col] == u, "session_id"].nunique()
            draw_group_rasters((cells[unit_col] == u).to_numpy(),
                               f"{UNIT.capitalize()} {u} ({n_ses} session{'s' if n_ses != 1 else ''})",
                               f"11_raster_{UNIT}_{u}")

    # ----------------------------------------------------------------- #
    # 12. Population-mean profile per group: mean +- SEM over units (pooled: over cells); thin lines = units
    # ----------------------------------------------------------------- #
    Zall = rs.zscore_profiles(profiles["all"], "profile")
    fig, axd = six_panel_figure(figsize_per_panel=(3.4, 2.6), sharex=True, sharey=True)
    for g, ax in axd.items():
        m = (cells["group"] == group_key(g)).to_numpy()
        if not m.any():
            continue
        color = SIX_GROUP_COLORS[g]
        if AGG == "pooled":
            mean = np.nanmean(Zall[m], 0)
            sem = np.nanstd(Zall[m], 0) / np.sqrt(np.isfinite(Zall[m]).sum(0).clip(1))
            ax.plot(centers, mean, color=color, lw=2)
            ax.fill_between(centers, mean - sem, mean + sem, color=color, alpha=0.25)   # cell-level SEM of the pooled mean
        else:
            profs = []
            for u in cells.loc[m, unit_col].unique():
                mu_ = m & (cells[unit_col] == u).to_numpy()
                if mu_.sum() >= MIN_CELLS_PER_UNIT:
                    p = np.nanmean(Zall[mu_], 0)
                    profs.append(p)
                    ax.plot(centers, p, color=color, lw=0.6, alpha=0.4)
            if profs:
                P = np.vstack(profs)
                mean = np.nanmean(P, 0)
                sem = (np.nanstd(P, 0, ddof=1) / np.sqrt(len(profs))) if len(profs) > 1 else np.zeros_like(mean)
                ax.plot(centers, mean, color=color, lw=2)
                ax.fill_between(centers, mean - sem, mean + sem, color=color, alpha=0.25)
        ax.axvline(0, color="k", ls="--", lw=0.6)
        ax.axhline(0, color="k", lw=0.4)
        ax.set_xlabel("position from stimulus onset (cm)", fontsize=8)
        ax.set_ylabel("mean z (profile)", fontsize=8)
    fig.suptitle("Population-mean profile per group (thick: " + ("pooled cells +- SEM over cells)" if AGG == "pooled" else
                 f"mean +- SEM over {UNITS}; thin: {UNITS} with >={MIN_CELLS_PER_UNIT} cells)"))
    fig.tight_layout()
    save_fig(fig, figdir, "12_population_mean_profile")
    plt.close(fig)

    # ----------------------------------------------------------------- #
    # 13. Where do cells peak? Pooled: cell counts; session / mouse: fraction of each unit's cells (mean +- SEM
    #     over units, thin lines = units). Reliable = odd/even profile r >= RELIABILITY_MIN.
    # ----------------------------------------------------------------- #
    edges = np.concatenate([centers - BINSIZE / 2, [centers[-1] + BINSIZE / 2]])
    fig, axd = six_panel_figure(figsize_per_panel=(3.4, 2.6), sharex=True, sharey=False)
    for g, ax in axd.items():
        sub = cells[cells["group"] == group_key(g)]
        if sub.empty:
            continue
        rel_ok = sub["reliability"] >= RELIABILITY_MIN
        color = SIX_GROUP_COLORS[g]
        if AGG == "pooled":
            ax.hist(sub["peak_position_all"].dropna(), bins=edges, color="#bbbbbb", label="all cells")
            ax.hist(sub.loc[rel_ok, "peak_position_all"].dropna(), bins=edges, color=color,
                    label=f"reliable (r>={RELIABILITY_MIN}): {rel_ok.mean():.0%}")
            ax.set_ylabel("cells", fontsize=8)
        else:
            fr = []
            for u, usub in sub.groupby(unit_col):
                if len(usub) < MIN_CELLS_PER_UNIT:
                    continue
                h, _ = np.histogram(usub["peak_position_all"].dropna(), bins=edges)
                fr.append(h / max(len(usub), 1))
                ax.plot(centers, fr[-1], color=color, lw=0.6, alpha=0.4)
            if fr:
                F = np.vstack(fr)
                mean = F.mean(0)
                sem = F.std(0, ddof=1) / np.sqrt(len(F)) if len(F) > 1 else np.zeros_like(mean)
                ax.plot(centers, mean, color=color, lw=2, label=f"{len(F)} {UNITS}; reliable {rel_ok.mean():.0%}")
                ax.fill_between(centers, mean - sem, mean + sem, color=color, alpha=0.25)
            ax.set_ylabel(f"fraction of {UNIT}'s cells", fontsize=8)
        ax.axvline(0, color="k", ls="--", lw=0.6)
        ax.legend(fontsize=6, frameon=False)
        ax.set_xlabel("peak position (cm)", fontsize=8)
    fig.suptitle(f"Position of peak activity (all trials; unit = {UNIT})")
    fig.tight_layout()
    save_fig(fig, figdir, "13_peak_position_hist")
    plt.close(fig)

    # ----------------------------------------------------------------- #
    # 13b. Do projection-labelled cells peak at different positions than their unlabelled neighbours in the same
    #      area? (Safaai et al. 2025, Fig. 1d-f.) V1-lab vs V1-unl, PM-lab vs PM-unl. Two views:
    #      * cell level (CDFs + test): the test shuffles the lab/unl label among the area's cells WITHIN each
    #        session (keeps the session structure), statistic = difference of median peak positions
    #      * unit level (session / mouse aggregate): per-unit median peak of each class (>= MIN_CELLS_PER_UNIT
    #        cells in both), paired difference lab - unl, Wilcoxon signed-rank over units
    # ----------------------------------------------------------------- #
    from scipy.stats import wilcoxon
    N_PERM_PEAK = 5000
    prng = np.random.default_rng(RANDOM_STATE)
    AREA_PAIRS = [("V1", "V1_lab", "V1_unl"), ("PM", "PM_lab", "PM_unl")]
    AREA_PAIRS = [p for p in AREA_PAIRS if {p[1], p[2]} <= set(cells["group"])]
    test_rows, unit_rows = [], []
    for area, g_lab, g_unl in AREA_PAIRS:
        for subset, mask_fn in (("all cells", lambda d: np.ones(len(d), bool)),
                                (f"reliable (r>={RELIABILITY_MIN})", lambda d: (d["reliability"] >= RELIABILITY_MIN).to_numpy())):
            d = cells[cells["group"].isin([g_lab, g_unl])].dropna(subset=["peak_position_all"])
            d = d[mask_fn(d)]
            is_lab = (d["group"] == g_lab).to_numpy()
            pos = d["peak_position_all"].to_numpy(float)
            if is_lab.sum() < 5 or (~is_lab).sum() < 5:
                continue
            obs = np.median(pos[is_lab]) - np.median(pos[~is_lab])
            sess_idx = [np.where(d["session_id"].to_numpy() == s)[0] for s in d["session_id"].unique()]
            null = np.empty(N_PERM_PEAK)
            for k in range(N_PERM_PEAK):
                perm = is_lab.copy()
                for ix in sess_idx:
                    perm[ix] = prng.permutation(is_lab[ix])
                null[k] = np.median(pos[perm]) - np.median(pos[~perm])
            ks = __import__("scipy.stats", fromlist=["ks_2samp"]).ks_2samp(pos[is_lab], pos[~is_lab])
            test_rows.append(dict(area=area, subset=subset, n_lab=int(is_lab.sum()), n_unl=int((~is_lab).sum()),
                                  median_lab=np.median(pos[is_lab]), median_unl=np.median(pos[~is_lab]),
                                  median_diff=obs, p_perm_within_session=(np.sum(np.abs(null) >= abs(obs)) + 1) / (N_PERM_PEAK + 1),
                                  ks_stat=ks.statistic, p_ks_cells_independent=ks.pvalue))
            if AGG != "pooled" and subset == "all cells":
                for u, du in d.groupby(unit_col):
                    a_, b_ = du.loc[du["group"] == g_lab, "peak_position_all"], du.loc[du["group"] == g_unl, "peak_position_all"]
                    if len(a_) >= MIN_CELLS_PER_UNIT and len(b_) >= MIN_CELLS_PER_UNIT:
                        unit_rows.append(dict(area=area, unit=u, animal_id=du["animal_id"].iloc[0], n_lab=len(a_), n_unl=len(b_),
                                              median_lab=a_.median(), median_unl=b_.median(), diff=a_.median() - b_.median()))
    peak_tests = pd.DataFrame(test_rows)
    unit_peaks = pd.DataFrame(unit_rows)
    if len(peak_tests):
        peak_tests.to_csv(table_dir / "peak_position_lab_vs_unl_test.csv", index=False)
        print("\nPeak position, labelled vs unlabelled cells of the same area (cm; the p-value shuffles labels within sessions):")
        print(peak_tests.round(3).to_string(index=False))
    if len(unit_peaks):
        unit_peaks.to_csv(table_dir / f"peak_position_lab_vs_unl_by_unit{AGG_SUFFIX}.csv", index=False)
        for area, d in unit_peaks.groupby("area"):
            if len(d) >= 5 and (d["diff"] != 0).any():
                print(f"  {area}: median(lab) - median(unl) per {UNIT}: mean {d['diff'].mean():.1f} cm over {len(d)} {UNITS}, "
                      f"Wilcoxon p = {wilcoxon(d['diff']).pvalue:.3f}")
    if AREA_PAIRS and len(peak_tests):
        two_rows = AGG != "pooled" and len(unit_peaks)
        fig, axes = plt.subplots(2 if two_rows else 1, len(AREA_PAIRS), figsize=(4.4 * len(AREA_PAIRS), 6.4 if two_rows else 3.4),
                                 squeeze=False)
        for j, (area, g_lab, g_unl) in enumerate(AREA_PAIRS):
            ax = axes[0, j]
            for g_, ls_ in ((g_lab, "-"), (g_unl, "--")):
                gg = [g for g in SIX_GROUP_ORDER if group_key(g) == g_][0]
                v = np.sort(cells.loc[cells["group"] == g_, "peak_position_all"].dropna().to_numpy())
                ax.plot(v, np.arange(1, len(v) + 1) / max(len(v), 1), color=SIX_GROUP_COLORS[gg], ls=ls_, lw=1.8,
                        label=f"{SIX_GROUP_LABELS[gg]} (n={len(v)})")
            ax.axvline(0, color="k", ls=":", lw=0.6)
            t = peak_tests[(peak_tests["area"] == area) & (peak_tests["subset"] == "all cells")]
            if len(t):
                ax.set_title(f"{area}: median diff {t['median_diff'].iloc[0]:+.1f} cm, p = {t['p_perm_within_session'].iloc[0]:.3f}",
                             fontsize=9)
            ax.set_xlabel("peak position (cm)")
            ax.set_ylabel("cumulative fraction of cells")
            ax.legend(fontsize=7, frameon=False)
            if two_rows:
                ax = axes[1, j]
                d = unit_peaks[unit_peaks["area"] == area]
                if len(d):
                    ax.axhline(0, color="k", lw=0.6)
                    xj = prng.uniform(-0.08, 0.08, len(d))
                    ax.bar(0, d["diff"].mean(), 0.5, color="#bbbbbb")
                    if len(d) > 1:
                        ax.errorbar(0, d["diff"].mean(), d["diff"].std(ddof=1) / np.sqrt(len(d)), color="k", capsize=3)
                    ax.scatter(xj, d["diff"], c=[mouse_colors.get(a, "k") for a in d["animal_id"]], edgecolor="k", s=22, zorder=3)
                ax.set_xticks([])
                ax.set_xlim(-1.2, 1.2)
                ax.set_ylabel(f"median peak lab - unl (cm), per {UNIT}")
                ax.set_title(f"{area}: {len(d)} {UNITS} with >= {MIN_CELLS_PER_UNIT} cells of each class", fontsize=8)
        if two_rows:
            mouse_legend(axes[1, -1], mouse_colors, animals=unit_peaks["animal_id"].unique(), loc="upper right")
        fig.suptitle("Peak position of projection-labelled vs unlabelled cells (0 = stimulus onset)")
        fig.tight_layout()
        save_fig(fig, figdir, "13b_peak_position_lab_vs_unl")
        plt.close(fig)


# =====================================================================================================
# PART 3 -- OUTCOME ANALYSIS (after van Vugt et al. 2018): hit / miss / false alarm / correct rejection
# =====================================================================================================
if ri is not None and want_outcome:
    o_ids, o_cells, o_cd, o_mu_sigma = ri
    rule = STIM_BINNING_STRATEGIES[OUTCOME_STRATEGY]
    cd_g = add_six_group_column(o_cd[["session_id", "cell_id", "roi_name", "labeled"]])
    group_of = dict(zip(zip(cd_g["session_id"], cd_g["cell_id"]), cd_g["group"]))
    fits_mtime = int((paths.out_from("1b_psychometric") / "psychometric_fits.csv").stat().st_mtime)

    def process_outcome(ses, seed):
        cells = o_cells[ses.session_id]
        cg = {c: group_of.get((ses.session_id, c), "") for c in cells}
        rng = np.random.default_rng(stable_seed(ses.session_id, "outcome", base=seed))
        return oa.analyze_session(ses, cells, cg, o_mu_sigma.get(ses.session_id, {}), rule, OUTCOME_WINDOWS,
                                  S_PRE, S_POST, BINSIZE, EQUALIZE_QUANTILE, MIN_TRIALS_COND, MIN_PER_CLASS,
                                  args.n_perm_outcome, ALPHA, MIN_POP_CELLS, rng)

    outs = run_sessions(
        o_ids, process_outcome, [PROTOCOL], paths.store, "outcome_session",
        key_params=dict(version="2b-outcome-v1", rule=target_signature([OUTCOME_STRATEGY]), windows=OUTCOME_WINDOWS,
                        s_pre=S_PRE, s_post=S_POST, binsize=BINSIZE, eq=EQUALIZE_QUANTILE, min_cond=MIN_TRIALS_COND,
                        min_class=MIN_PER_CLASS, n_perm=args.n_perm_outcome, alpha=ALPHA, min_pop=MIN_POP_CELLS,
                        fits=fits_mtime),
        load_kwargs=dict(load_behaviordata=True, load_videodata=True, load_celldata=True, load_calciumdata=True),
        cells_by_session=o_cells, n_jobs=N_JOBS, recompute=args.recompute, verbose=JOBLIB_VERBOSE,
        extra_key_by_session={sid: stamps[sid] for sid in o_ids}, flatten=False)
    outs = [o for o in outs if o is not None]
    if not outs:
        print("Outcome analysis skipped: no session with a psychometric fit, enough engaged trials and stimulus trials.")
    else:
        # ------------------------------------------------------------------------------------------- #
        # 14. Stack cells; unit helpers
        # ------------------------------------------------------------------------------------------- #
        centers_o = outs[0]["centers"]
        animal_of = {o["session_id"]: o["animal_id"] for o in outs}
        cat_names = list(oa.CATEGORY_NAMES)

        def add_unit(df):
            """animal_id, and `unit` ('all' | session id | mouse id) columns of the chosen aggregate."""
            df = df.copy()
            if "animal_id" not in df:
                df["animal_id"] = df["session_id"].map(animal_of)
            df["unit"] = {"pooled": "all", "session": df.get("session_id"), "mouse": df["animal_id"]}[AGG]
            return df

        def attach_group(df):
            return add_unit(add_six_group_column(df.merge(
                o_cd[["session_id", "cell_id", "roi_name", "labeled"]], on=["session_id", "cell_id"], how="left")))

        def out_cols(df, keep_unit=True):
            """Drop helper columns for CSV: pooled keeps original names, others get unit + animal_id."""
            return df if AGG != "pooled" else df.drop(columns=[c for c in ("unit", "animal_id", "session_id")
                                                                if c in df.columns])

        cells_o = attach_group(pd.concat([pd.DataFrame({"session_id": o["session_id"], "cell_id": o["cell_ids"]})
                                          for o in outs], ignore_index=True))
        conds = [f"{o_}|{c}" for o_ in ("HIT", "MISS") for c in cat_names] + ["CR", "FA"]
        n_bins = len(centers_o)
        trace_arr = {c: np.vstack([o["traces"][c] if o["traces"].get(c) is not None
                                   else np.full((len(o["cell_ids"]), n_bins), np.nan) for o in outs]) for c in conds}
        print(f"\nOutcome analysis: {len(outs)} sessions, {len(cells_o)} cells; "
              f"minimum {MIN_TRIALS_COND} trials per outcome condition")
        counts = pd.concat([o["counts"] for o in outs], ignore_index=True)
        counts.to_csv(table_dir / "outcome_counts_by_session.csv", index=False)
        tot = counts.groupby(["category", "outcome"])["n"].sum().unstack("outcome")
        print("Trials per contrast category and outcome (all selected sessions):")
        print(tot.to_string())

        def unit_trace_table(arr_by_name, name_col):
            """Per (group, unit, name): mean over the unit's cells at each position bin (bins with fewer than
            MIN_CELLS_PER_UNIT cells -- 1 when pooled -- are NaN). Returns DataFrame + dict -> array."""
            min_c = 1 if AGG == "pooled" else MIN_CELLS_PER_UNIT
            rows = []
            for name, arr in arr_by_name.items():
                for (g, u), idx in cells_o.groupby(["group", "unit"]).indices.items():
                    sub = arr[idx]
                    nf = np.isfinite(sub).sum(0)
                    with np.errstate(invalid="ignore"):
                        m = np.where(nf >= min_c, np.nanmean(np.where(np.isfinite(sub), sub, np.nan), 0), np.nan)
                    rows.append(dict(group=g, unit=u, animal_id=cells_o["animal_id"].iloc[idx[0]],
                                     **{name_col: name}, mean=m, n_cells=int(len(idx))))
            return rows

        def mean_over_units(rows, group, name, name_col):
            """mean and SEM (over units; none when pooled) of the unit traces of one (group, name)."""
            M = np.vstack([r["mean"] for r in rows if r["group"] == group and r[name_col] == name]
                          or [np.full(n_bins, np.nan)])
            with np.errstate(invalid="ignore"):
                m = np.nanmean(M, 0)
                n = np.isfinite(M).sum(0)
                sem = np.where(n > 1, np.nanstd(M, 0, ddof=1) / np.sqrt(np.maximum(n, 1)), 0.0)
            return m, (sem if AGG != "pooled" else np.zeros_like(m)), M

        def rows_to_long(rows, name_col, extra=None):
            recs = []
            for r in rows:
                for pos, v in zip(centers_o, r["mean"]):
                    recs.append({"group": r["group"], **({"unit": r["unit"], "animal_id": r["animal_id"]}
                                                         if AGG != "pooled" else {}),
                                 name_col: r[name_col], "position_cm": pos, "mean": v, "n_cells": r["n_cells"]})
            return pd.DataFrame(recs)

        # ------------------------------------------------------------------------------------------- #
        # 15. Figure 14: hit / miss / CR / FA traces per group and contrast category (van Vugt Fig. 2A)
        # ------------------------------------------------------------------------------------------- #
        trows = unit_trace_table(trace_arr, "condition")
        rows_to_long(trows, "condition").assign(
            category=lambda d: d["condition"].str.split("|").str[1].fillna("no stimulus"),
            outcome=lambda d: d["condition"].str.split("|").str[0]).drop(columns="condition").to_csv(
            table_dir / f"outcome_traces_by_group{AGG_SUFFIX}.csv", index=False)
        fig, axes = plt.subplots(len(SIX_GROUP_ORDER), len(cat_names), figsize=(4.2 * len(cat_names), 2.3 * 6),
                                 sharex=True, squeeze=False)
        for gi, g in enumerate(SIX_GROUP_ORDER):
            for ci, cname in enumerate(cat_names):
                ax = axes[gi, ci]
                for cond, label in ((f"HIT|{cname}", "hit"), (f"MISS|{cname}", "miss"), ("CR", "CR"), ("FA", "FA")):
                    m, sem, M = mean_over_units(trows, group_key(g), cond, "condition")
                    if not np.isfinite(m).any():
                        continue
                    col = OUTCOME_COLORS[cond.split("|")[0]]
                    ax.plot(centers_o, m, color=col, lw=1.6 if "|" in cond else 1.0,
                            ls="-" if "|" in cond else "--", label=label)
                    ax.fill_between(centers_o, m - sem, m + sem, color=col, alpha=0.2, lw=0)
                ax.axvline(0, color="grey", lw=0.6, ls=":")
                ax.axhline(0, color="grey", lw=0.4)
                ax.axvspan(*OUTCOME_WINDOWS["stim"], color="grey", alpha=0.08, lw=0)
                if gi == 0:
                    ax.set_title(f"{cname} contrasts", fontsize=9)
                if ci == 0:
                    ax.set_ylabel(f"{SIX_GROUP_LABELS[g]}\nactivity (z)", color=SIX_GROUP_COLORS[g], fontsize=8)
                if gi == len(SIX_GROUP_ORDER) - 1:
                    ax.set_xlabel("position relative to stimulus onset (cm)", fontsize=8)
        axes[0, -1].legend(fontsize=7, frameon=False, loc="upper right")
        fig.suptitle(f"Activity on hits / misses (contrast-matched) and, dashed, correct rejections / false alarms; "
                     f"{'mean over cells' if AGG == 'pooled' else f'mean +- SEM over {UNITS}'}; shaded = stim window", y=1.0)
        fig.tight_layout()
        save_fig(fig, figdir, "14_outcome_traces")
        plt.close(fig)

        # ------------------------------------------------------------------------------------------- #
        # 16. Figure 15: miss fraction = 100 x activity(miss) / activity(hit) (van Vugt Fig. 2B)
        # ------------------------------------------------------------------------------------------- #
        sc = pd.concat([o["scalars"] for o in outs if len(o["scalars"])], ignore_index=True)
        if len(sc):
            sc = attach_group(sc)
            sc.to_csv(table_dir / "outcome_scalars_per_cell.csv", index=False)
            both = sc.dropna(subset=["resp_hit", "resp_miss"])
            min_c = 1 if AGG == "pooled" else MIN_CELLS_PER_UNIT
            mf = both.groupby(["group", "category", "unit"]).agg(
                animal_id=("animal_id", "first"), n_cells=("cell_id", "size"), activity_hit=("resp_hit", "mean"),
                activity_miss=("resp_miss", "mean")).reset_index()
            mf = mf[mf["n_cells"] >= min_c]
            mf["miss_fraction_pct"] = np.where(mf["activity_hit"] > 0, 100 * mf["activity_miss"] / mf["activity_hit"],
                                               np.nan)
            out_cols(mf).to_csv(table_dir / f"miss_fraction{AGG_SUFFIX}.csv", index=False)
            piv = mf.groupby(["category", "group"])["miss_fraction_pct"].mean().unstack("group") \
                .reindex(index=cat_names, columns=GROUP_KEYS)
            print(f"\nMiss fraction (%) = 100 x mean response on misses / on hits (stim - pre_stim, contrast-matched), "
                  f"{'pooled' if AGG == 'pooled' else f'mean over {UNITS}'}:")
            print(piv.round(0).to_string())
            fig, ax = plt.subplots(figsize=(9, 4.2))
            handles = grouped_unit_value_bars(ax, mf, "miss_fraction_pct", "category", cat_names, AGG,
                                              mouse_colors=mouse_colors, hline=100,
                                              ylabel="miss fraction (% of hit activity)", tick_size=8)
            ax.legend(*handles, fontsize=8, frameon=False, title="contrast")
            ax.set_title(f"Miss fraction: activity on misses relative to hits (dashed = 100%); "
                         f"unit = {UNIT}, {MIN_CELLS_PER_UNIT if AGG != 'pooled' else 1}+ cells per unit", fontsize=9)
            fig.tight_layout()
            save_fig(fig, figdir, "15_miss_fraction")
            plt.close(fig)
        else:
            print("\nMiss fraction skipped: no session has enough hit AND miss trials in a contrast category.")

        # ------------------------------------------------------------------------------------------- #
        # 17. Figure 16: false alarms vs correct rejections, BEFORE the stimulus (van Vugt Fig. 2A / 3A)
        # ------------------------------------------------------------------------------------------- #
        fc = pd.concat([o["fa_cr"] for o in outs if len(o["fa_cr"])], ignore_index=True) if any(
            len(o["fa_cr"]) for o in outs) else pd.DataFrame()
        cp_all = pd.concat([o["cp"] for o in outs if len(o["cp"])], ignore_index=True) if any(
            len(o["cp"]) for o in outs) else pd.DataFrame()
        if len(cp_all):
            cp_all = attach_group(cp_all)
            cp_all.to_csv(table_dir / "choice_probability_per_cell.csv", index=False)
            cp_f = cp_all[np.isfinite(cp_all["auc"])].copy()
            cp_f["sig"] = cp_f["significant"].astype(float)
            cp_f["auc_dev"] = cp_f["auc"]
            cpt = cp_f.groupby(["group", "analysis", "window"] + (["unit", "animal_id"] if AGG != "pooled" else [])) \
                .agg(n_cells=("auc", "size"), mean_auc=("auc", "mean"), frac_sig=("sig", "mean")).reset_index()
            cpt.to_csv(table_dir / f"choice_probability_by_group{AGG_SUFFIX}.csv", index=False)
            p = cp_f.groupby(["analysis", "window", "unit"]).agg(auc=("auc", "mean"), sig=("sig", "mean")) \
                .reset_index().groupby(["analysis", "window"])[["auc", "sig"]].mean()
            print(f"\nChoice probability (AUROC; > 0.5 = more activity on the positive class: hit, or FA), mean over cells"
                  f"{'' if AGG == 'pooled' else f' of each {UNIT}, then over {UNITS}'}:")
            print(p.round(3).to_string())
        if len(fc):
            fc = attach_group(fc)
            fc.to_csv(table_dir / "false_alarm_vs_cr_per_cell.csv", index=False)
            long_fc = pd.concat([fc.assign(trials="CR", value=fc["pre_CR"]), fc.assign(trials="FA", value=fc["pre_FA"])])
        panels_fc = []
        if len(fc):
            panels_fc.append("pre")
        if len(cp_all) and (cp_all["analysis"] == "FA_vs_CR").any():
            panels_fc.append("auc")
        if panels_fc:
            fig, axes = plt.subplots(1, len(panels_fc), figsize=(7.5 * len(panels_fc), 4.2), squeeze=False)
            for ax, kind in zip(axes[0], panels_fc):
                if kind == "pre":
                    handles = grouped_series_bars(ax, long_fc, "value", "trials", ["CR", "FA"], AGG,
                                                  series_colors={"CR": "black", "FA": OUTCOME_COLORS["FA"]},
                                                  mouse_colors=mouse_colors, min_cells_per_unit=MIN_CELLS_PER_UNIT,
                                                  hline=0, ylabel="pre-stimulus activity (z)", tick_size=8)
                    ax.legend(*handles, fontsize=8, frameon=False)
                    ax.set_title("Pre-stimulus activity before no-stimulus trials", fontsize=9)
                else:
                    s_ = cp_f[(cp_f["analysis"] == "FA_vs_CR")]
                    handles = grouped_series_bars(ax, s_, "auc", "window", [w for w in OUTCOME_WINDOWS], AGG,
                                                  mouse_colors=mouse_colors, min_cells_per_unit=MIN_CELLS_PER_UNIT,
                                                  hline=0.5, ylabel="AUROC (FA vs CR)", tick_size=8)
                    ax.legend(*handles, fontsize=8, frameon=False, title="window")
                    ax.set_title("Can a cell's activity predict a false alarm? (0.5 = no)", fontsize=9)
            fig.suptitle(f"False alarms vs correct rejections; unit = {UNIT}", y=1.02)
            fig.tight_layout()
            save_fig(fig, figdir, "16_false_alarm_vs_correct_rejection")
            plt.close(fig)

        # ------------------------------------------------------------------------------------------- #
        # 18. Figures 17-18: choice probability (hit vs miss, contrast-conditioned) before / after onset
        # ------------------------------------------------------------------------------------------- #
        if len(cp_all):
            hm = [a for a in ("hit_vs_miss_intermediate", "hit_vs_miss_all") if (cp_f["analysis"] == a).any()]
            if hm:
                fig, axes = plt.subplots(2, len(hm), figsize=(7.5 * len(hm), 7.5), squeeze=False)
                for ci, a in enumerate(hm):
                    s_ = cp_f[cp_f["analysis"] == a]
                    for ri_, (val, yl, hl) in enumerate((("auc", "AUROC hit vs miss", 0.5),
                                                         ("sig", f"fraction of cells outside 95% of null", ALPHA))):
                        ax = axes[ri_, ci]
                        handles = grouped_series_bars(ax, s_, val, "window", list(OUTCOME_WINDOWS), AGG,
                                                      mouse_colors=mouse_colors, min_cells_per_unit=MIN_CELLS_PER_UNIT,
                                                      hline=hl, ylabel=yl, tick_size=8)
                        if ri_ == 0:
                            ax.set_title({"hit_vs_miss_intermediate": "intermediate contrasts (van Vugt)",
                                          "hit_vs_miss_all": "all stimulus contrasts"}[a], fontsize=9)
                            ax.legend(*handles, fontsize=8, frameon=False, title="window")
                fig.suptitle(f"Choice probability before (pre_stim) and after (stim) stimulus onset; AUROC conditioned on "
                             f"contrast level; unit = {UNIT}", y=1.0)
                fig.tight_layout()
                save_fig(fig, figdir, "17_choice_probability")
                plt.close(fig)

            cptr = {}
            for name in ("hit_vs_miss_intermediate", "hit_vs_miss_all"):
                cptr[name] = np.vstack([o["cp_trace"].get(name, np.full((len(o["cell_ids"]), n_bins), np.nan))
                                        for o in outs])
            crows = unit_trace_table(cptr, "analysis")
            rows_to_long(crows, "analysis").to_csv(table_dir / f"choice_probability_trace{AGG_SUFFIX}.csv", index=False)
            fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True, sharey=True)
            for ax, g in zip(axes.flat, SIX_GROUP_ORDER):
                for name, col in (("hit_vs_miss_intermediate", "#1f77b4"), ("hit_vs_miss_all", "#ff7f0e")):
                    m, sem, _ = mean_over_units(crows, group_key(g), name, "analysis")
                    if np.isfinite(m).any():
                        ax.plot(centers_o, m, color=col, label=name.replace("hit_vs_miss_", ""))
                        ax.fill_between(centers_o, m - sem, m + sem, color=col, alpha=0.2, lw=0)
                ax.axhline(0.5, color="grey", ls="--", lw=0.8)
                ax.axvline(0, color="grey", ls=":", lw=0.6)
                ax.set_title(SIX_GROUP_LABELS[g], color=SIX_GROUP_COLORS[g], fontsize=10)
                ax.set_xlabel("position relative to stimulus (cm)", fontsize=8)
                ax.set_ylabel("AUROC hit vs miss", fontsize=8)
            axes[0, 0].legend(fontsize=7, frameon=False, title="contrasts")
            fig.suptitle(f"Choice probability along the corridor (per position bin, conditioned on contrast level); "
                         f"{'mean over cells' if AGG == 'pooled' else f'mean +- SEM over {UNITS}'}")
            fig.tight_layout()
            save_fig(fig, figdir, "18_choice_probability_trace")
            plt.close(fig)

        # ------------------------------------------------------------------------------------------- #
        # 19. Figure 19: outcome predicted from single prestimulus markers (van Vugt Fig. 3A; Rowland 2023)
        # ------------------------------------------------------------------------------------------- #
        mk = pd.concat([o["markers"] for o in outs if len(o["markers"])], ignore_index=True) if any(
            len(o["markers"]) for o in outs) else pd.DataFrame()
        if len(mk):
            mk = add_unit(mk)
            mk["label"] = np.where(mk["group"] != "", mk["marker"] + " " + mk["group"].str.replace("_", "-"), mk["marker"])
            mk["pairs"] = mk["n_pos"] * mk["n_neg"]
            mk.to_csv(table_dir / "prestim_marker_auroc_per_session.csv", index=False)

            def wavg(d):
                return pd.Series({"auc": np.average(d["auc"], weights=d["pairs"]) if d["pairs"].sum() > 0 else np.nan,
                                  "n_sessions": d["session_id"].nunique(), "pairs": d["pairs"].sum()})
            mku = mk.groupby(["comparison", "label", "marker", "group", "unit"]).apply(wavg).reset_index()
            mku["animal_id"] = mku["unit"].map(lambda u: animal_of.get(u, u) if AGG == "session" else u) \
                if AGG != "pooled" else "all"
            out_cols(mku).to_csv(table_dir / f"prestim_marker_auroc{AGG_SUFFIX}.csv", index=False)
            pop_order = [f"{kind} {g.replace('_', '-')}" for kind in ("popmean", "popvar") for g in GROUP_KEYS]
            beh_order = [m for m in oa.BEHAVIOR_MARKERS + ["iti"] if (mku["label"] == m).any()]
            order = [o_ for o_ in pop_order if (mku["label"] == o_).any()] + beh_order
            colors = {f"{kind} {g.replace('_', '-')}": GROUP_COLOR[g] for kind in ("popmean", "popvar") for g in GROUP_KEYS}
            fig, axes = plt.subplots(2, 1, figsize=(max(9, 0.45 * len(order) + 2), 8.5), squeeze=False)
            for ax, cmp_name, ttl in zip(axes[:, 0], ("hit_vs_miss", "FA_vs_CR"),
                                         ("hits vs misses (stimulus trials, conditioned on contrast level)",
                                          "false alarms vs correct rejections (no-stimulus trials)")):
                s_ = mku[mku["comparison"] == cmp_name]
                unit_bars(ax, s_, "label", "auc", order, AGG, colors=colors, mouse_colors=mouse_colors,
                          ylabel="AUROC", hline=0.5)
                ax.set_xticks(range(len(order)))
                ax.set_xticklabels(order, rotation=60, ha="right", fontsize=7)
                ax.set_title(f"Outcome predicted from a single prestimulus marker: {ttl}", fontsize=9)
            if AGG != "pooled":
                mouse_legend(axes[0, 0], mouse_colors, animals=mku["animal_id"].unique(), loc="upper right")
            fig.suptitle("Prestimulus markers (pre_stim window): population mean / variance of a group's cells, behavior, "
                         f"time since the previous trial; unit = {UNIT} (pair-weighted over its sessions)", y=1.0)
            fig.tight_layout()
            save_fig(fig, figdir, "19_prestim_marker_auroc")
            plt.close(fig)

        # ------------------------------------------------------------------------------------------- #
        # 20. Figure 20: hit probability vs prestimulus population variance / mean (Rowland 2023 Fig. 4c)
        # ------------------------------------------------------------------------------------------- #
        mtr = []
        for o in outs:
            m_ = o["marker_trials"]
            m_ = m_[(m_["category"] == 1) & m_["outcome"].isin(["HIT", "MISS"])].copy()
            m_["hit"] = (m_["outcome"] == "HIT").astype(float)
            for col in [c for c in m_.columns if c.startswith(("popvar__", "popmean__"))]:
                x = m_[col]
                ok = x.notna()
                if ok.sum() < N_MARKER_BINS * MIN_TRIALS_PER_MARKER_BIN:
                    continue
                bins = pd.qcut(x[ok], N_MARKER_BINS, labels=False, duplicates="drop")
                if bins.nunique() < N_MARKER_BINS or bins.value_counts().min() < MIN_TRIALS_PER_MARKER_BIN:
                    continue
                mtr.append(pd.DataFrame({"session_id": o["session_id"], "marker": col.split("__")[0],
                                         "group": col.split("__")[1], "bin": bins.to_numpy() + 1,
                                         "hit": m_.loc[ok, "hit"].to_numpy()}))
        if mtr:
            mtr = add_unit(pd.concat(mtr, ignore_index=True))
            hp = mtr.groupby(["marker", "group", "unit", "bin"]).agg(hit_probability=("hit", "mean"), n_trials=("hit", "size"),
                                                                     animal_id=("animal_id", "first")).reset_index()
            out_cols(hp).to_csv(table_dir / f"hit_probability_by_prestim_marker{AGG_SUFFIX}.csv", index=False)
            fig, axes = plt.subplots(2, 6, figsize=(17, 6), sharey=True, squeeze=False)
            for ri_, marker in enumerate(("popvar", "popmean")):
                for ci, g in enumerate(SIX_GROUP_ORDER):
                    ax = axes[ri_, ci]
                    s_ = hp[(hp["marker"] == marker) & (hp["group"] == group_key(g))]
                    if len(s_):
                        for u, su in s_.groupby("unit"):
                            if AGG != "pooled":
                                ax.plot(su["bin"], su["hit_probability"], color=mouse_colors.get(su["animal_id"].iloc[0], "k"),
                                        lw=0.8, alpha=0.6)
                        mm = s_.groupby("bin")["hit_probability"]
                        m, sem = mm.mean(), mm.sem().fillna(0)
                        ax.errorbar(m.index, m.values, yerr=None if AGG == "pooled" else sem.values, color="k", lw=2,
                                    marker="o", ms=4, capsize=2)
                    ax.set_xticks(range(1, N_MARKER_BINS + 1))
                    ax.set_title(f"{SIX_GROUP_LABELS[g]} (n={s_['unit'].nunique()})", color=SIX_GROUP_COLORS[g], fontsize=9)
                    ax.set_xlabel(f"prestim. {'variance' if marker == 'popvar' else 'mean'} (low -> high)", fontsize=8)
                    if ci == 0:
                        ax.set_ylabel("hit probability\n(intermediate contrasts)", fontsize=8)
            fig.suptitle("Hit probability by prestimulus population variance (top) and mean (bottom) of the group's cells; "
                         f"bins = {N_MARKER_BINS} per session; thin = {UNIT}s, thick = mean +- SEM", y=1.0)
            fig.tight_layout()
            save_fig(fig, figdir, "20_hit_probability_by_prestim_marker")
            plt.close(fig)


print(f"\nWrote tables to {table_dir}, figures to {figdir}")
