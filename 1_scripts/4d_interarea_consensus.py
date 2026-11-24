# -*- coding: utf-8 -*-
"""
4d_interarea_consensus
=======================
MODEL-FREE TEST of the "dynamic consensus" claim in Javadzadeh, Schimel, Hofer, Ahmadian & Hennequin (2026, Nat. Neurosci.),
"Reciprocal connections dynamically build consensus between neocortical areas": in paired V1-LM recordings, the trial-averaged
activity patterns on which the two areas AGREE decay more slowly (longer autocorrelation) than the patterns on which they
DISAGREE -- a prediction of their reciprocal-excitation line-attractor model that the paper also verifies independently of the
fitted circuit model (their Fig. 4h/5a). This script reimplements that model-free check (their eqs. 37-41) on our own V1-PM
data (DN protocol), with PM standing in for their LM (both are higher visual areas one synapse downstream of V1; see the
project's own citation note on this paper: "whether labelled V1->PM and PM->V1 neurons load more on [the slow mode] is
testable in the DN data").

WHAT IS, AND ISN'T, THE SAME AS THE PAPER
    * Their data: extracellular ephys (spikes, 5 ms bins), head-fixed STATIONARY mice, go/no-go on a static grating, optogenetic
      silencing trials excluded from the model-free test. Our data: calcium imaging ("deconvolved activity", ~5-30 Hz, so bins
      here are whole imaging frames, 30-200 ms -- much coarser than their 5 ms bins), mice RUNNING through a virtual corridor
      (no optogenetic perturbations), with a graded "noise" signal strength (not just two discrete stimuli), so most trials
      have no well-defined HIT/MISS/FA/CR label (trialOutcome is only defined at 0 %/100 % signal, see infotheory.behavior).
      CORRECT_ONLY therefore defaults to False here (restricting to the catch trials alone would throw out most of the data);
      flip it on to replicate their "only correct trials" choice on just the 0 %/100 % trials.
    * Their windows are TIME (ms) relative to the actual stimulus-onset timestamp; our project's other scripts mostly use
      POSITION (cm) windows (stimStart is a position, see psth.py). This script uses the project's own time-domain machinery
      (psth.derive_onset_time + psth.align_trials_time) to stay faithful to the paper's literal ms-relative-to-onset windows.
    * Their eqs. 37-41, reimplemented almost verbatim (see `trial_autocorr` below):
          a_k(t) = mean_{i in V1} s_k^i(t) + mean_{i in PM} s_k^i(t)      ("agree" / balanced-sum mode)
          d_k(t) = mean_{i in V1} s_k^i(t) - mean_{i in PM} s_k^i(t)      ("disagree" / balanced-difference mode)
          ab_k(t) = a_k(t) - mean_k'[a_k'(t)]                            (mean-subtract ACROSS TRIALS at each time bin)
          C_k(tau) = <ab_k(t) ab_k(t+tau)>_t / <ab_k(t) ab_k(t)>_t       (per-trial normalized autocorrelation, averaged over
                                                                          t such that both t, t+tau fall in the window)
      We report lag>=0 only (a real-valued signal's autocorrelation is symmetric by construction; the paper plots both sides
      purely for display). s_k^i is their SPIKE COUNT per neuron per bin; here it is the cell's deconvolved-activity value per
      imaging frame (not a literal spike count -- see spike_stats.py's module docstring on what calcium "events" do and don't
      support). No z-scoring is applied (matching the paper: an unweighted per-area mean, not a per-cell-normalized one).
    * Their Delta_max / Delta_slow summary statistics (how MUCH slower the agree mode decays) are defined in their Supplementary
      Information section S3, which is not available to us. As an honest, clearly-labelled STAND-IN we instead use the area
      under the autocorrelation curve for lag > 0 (a monotonic proxy for "how slowly it decays": a flatter curve has a larger
      area), one number per session and window and mode, compared with a paired two-sided Wilcoxon signed-rank test across
      sessions (agree AUC vs disagree AUC) -- this is NOT a reproduction of their exact statistic, only a comparable one.
    * POOLING ACROSS TRIALS, NOT PER-TRIAL RATIOS (deviation from a literal eq. 41, added after the first real-data run): the
      paper's spike data has ~80-100 5 ms bins per 400-500 ms window, so a per-trial ratio C_k(tau) = <x(t)x(t+tau)>_t/<x(t)^2>_t
      averages over many within-trial (t, t+tau) pairs before the across-trial average is taken. Our calcium frames are far
      coarser (binsize_ms is the session's own imaging frame period, often > 150 ms -- see binsize_ms in the summary table),
      so a 400-500 ms window has only 3-4 bins total: at lag=1 that is only 2-3 within-trial (t, t+1) pairs, at lag=2 only 1-2.
      A genuine per-trial ratio is then either undefined or a single near-noise ratio (and unstable: dividing by one near-zero,
      mean-subtracted value can blow up). We therefore POOL the numerator and denominator over ALL (trial, t) pairs in the
      session before taking the ratio -- C(tau) = sum_{k,t valid} xbar_k(t) xbar_k(t+tau) / sum_{k,t valid} xbar_k(t)^2 -- the
      same quantity, just averaged over trials before, not after, the division. This is far more numerically stable given our
      bin counts; the within-trial sum in the paper's ratio becomes, in effect, the cross-trial sum here. Error bars on the
      resulting single per-session curve come from bootstrap resampling of trials (`bootstrap_autocorr_sem`), not from a
      per-trial standard error (there is no longer a population of "per-trial C(tau) values" to take a standard error of).

EXTENSION BEYOND THE PAPER: a REWARD window (time-locked to rewardZoneStart, like the project's own WINDOWS={"prestim",
"stim","reward"} convention in 1f/1g/2l) is included by default (INCLUDE_REWARD_WINDOW) because the project's own earlier
analyses (1g/1f) found a reward-window-specific disengagement signature (criterion rising sharply with running speed / pupil
area, specifically in the reward window) -- worth checking whether inter-area consensus also looks different there. This has
no counterpart in the paper itself.

SECOND EXTENSION: SHALLOW vs. DEEP PM, AND LABELED vs. UNLABELED V1/PM. The project's group system (celldata_utils.py)
splits BOTH V1 and PM into "lab" (deliberately, retrogradely labeled projection neurons -- V1 cells labeled from a PM
injection, PM cells labeled from a V1 injection) and "unl" (unlabeled), and additionally splits PM by depth (PM_lab/PM_unl
= shallow, <= MAX_DEPTH_UM; PMdeep_lab/PMdeep_unl = deep, DEEP_PM_DEPTH_UM). This is exactly the project's own citation note
on the paper -- "whether labelled V1->PM and PM->V1 neurons load more on [the slow mode] is testable in the DN data" -- so
rather than pooling V1_lab+V1_unl and PM_lab+PM_unl into single "V1"/"PM" means (as the paper's own LM population is, by
necessity, unlabeled), the test is run on every V1-label x PM-layer x PM-label combination: V1_LABELS = {lab, unl} paired
against PM_SUBGROUPS = {shallow-lab, shallow-unl, deep-lab, deep-unl} (8 combinations total when INCLUDE_DEEP_PM; 4 when
not). Each combination is gated independently by MIN_CELLS_PER_AREA (a session can qualify for some combinations and not
others -- labeled populations in particular are often small), and every curve/summary row carries v1_label, pm_layer and
pm_label columns. Figures are split one file per V1 label (V1-lab cells vs. V1-unl cells), each showing all 4 PM subgroups
as rows x windows as columns, so "all relevant combinations" are visible without a single unreadably large grid.

Project-wide rules
    * selection : --animals / --sessions restrict the sessions (a restricted run writes to out/selection/<tag>/)
    * aggregate : --aggregate pooled|session|mouse = the unit of the summary plot and of the mean curves
    * memory    : one session in RAM per worker (session_jobs.run_sessions); per-session results cached in store/
    * cells     : 1b-included sessions; cells passing 2a (proximity, depth rule) and 2b QC; V1 = V1_lab+V1_unl, PM =
                  PM_lab+PM_unl(+PMdeep_lab+PMdeep_unl if INCLUDE_DEEP_PM, matching the project's deep-PM feedback-loop
                  rationale in project_config.py)

Reads: 1b psychometric_fits.csv, 2a celldata_combined.csv, 2b qc_per_cell.csv. Loads: raw behavior / calcium, one session at a
time (DN only -- deconvolved activity exists only for DN, see session.py).

Outputs
-------
2_pipeline/4d_interarea_consensus/
    out/    interarea_autocorr_curves.csv   per (session, window, v1_label, pm_layer, pm_label, mode, lag_bin):
                                            pooled-across-trials autocorrelation (mean_autocorr), its trial-bootstrap SEM,
                                            and n_pairs (pooled valid (trial, t) pairs at that lag -- shrinks at the
                                            largest lags; see "SECOND EXTENSION" in the module docstring)
            interarea_autocorr_summary.csv  per (session, window, v1_label, pm_layer, pm_label, mode): AUC (lag>0) of
                                            that session's pooled curve, n_trials, n_v1_cells, n_pm_cells (that
                                            combination's cell counts)
            figures/1_autocorr_by_window_V1-<label>.png  mean +- SEM autocorrelation vs lag (ms), agree vs disagree, one
                                                         file per V1 label, PM subgroup (shallow/deep x lab/unl) rows x
                                                         window columns
                    2_auc_summary_V1-<label>.png          paired per-session AUC (agree vs disagree), same file/row/col
                                                         layout, Wilcoxon p
    store/  interarea_consensus_session_<hash>.pkl
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

from infotheory.pipeline import get_pipeline_paths
from infotheory.session import discover_sessions
from infotheory.neural_encoding import load_cell_inclusion
from infotheory.continuous import build_calcium_continuous
from infotheory.psth import derive_onset_time, align_trials_time
from infotheory.spike_stats import get_frame_rate
from infotheory.plotting import set_style, save_fig, clear_figures
from infotheory.scope import add_scope_args, scope_from_args, select_sessions, scoped_dirs, data_stamp, unit_noun
from infotheory.session_jobs import run_sessions
from infotheory.unit_plots import mouse_palette

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom")

# ======================================================================= #
# SETTINGS -- edit here, not in the body below
# ======================================================================= #
PROTOCOL = "DN"                        # deconvolved activity only exists for DN (session.py)
INCLUDE_DEEP_PM = True                 # PM = PM_lab+PM_unl(+PMdeep_lab+PMdeep_unl); the deep-PM groups matter for the
                                       # V1<->PM feedback loop (project_config.py's own rationale)
MIN_CELLS_PER_AREA = 5                 # a session needs at least this many included cells in EACH area to be used

ENGAGED_ONLY = True                    # only engaged trials (continuous.restrict_to_engaged's convention)
CORRECT_ONLY = False                   # paper: only correct go/no-go trials. Our task has graded "noise" trials with no
                                       # HIT/MISS/FA/CR label (only defined at 0 %/100 %, infotheory.behavior.add_trial_outcome)
                                       # -- restricting to correct-only would drop most trials, so default False (uses all
                                       # engaged trials). Set True to replicate the paper's choice on the 0 %/100 % subset only.

# Time windows, ms relative to the relevant onset (psth.derive_onset_time). "prestim"/"stim" replicate the paper's Fig. 4h/5a
# windows exactly ([-400,0] and [0,500] ms rel. to stimulus onset). "reward" is a project-specific EXTENSION (see docstring),
# time-locked to rewardZoneStart instead of stimStart.
WINDOWS_MS = {"prestim": (-400.0, 0.0, "stimStart"), "stim": (0.0, 500.0, "stimStart")}
INCLUDE_REWARD_WINDOW = True
REWARD_WINDOW_MS = (0.0, 500.0, "rewardZoneStart")     # (t_pre, t_post, position_value_col used by derive_onset_time)

BINSIZE_MODE = "frame"                 # "frame": one imaging frame per bin (finest possible, session's own fs); "fixed": use
                                       # FIXED_BINSIZE_MS for every session (coarser, but comparable in ms across sessions of
                                       # different fs)
FIXED_BINSIZE_MS = 100.0               # only used if BINSIZE_MODE == "fixed"
MAX_LAG_MS = None                      # None = use the whole window (every lag up to the window length); otherwise cap the
                                       # lags computed/plotted to this many ms
MIN_TRIALS_WINDOW = 15                 # a (session, window) with fewer usable trials is skipped
MIN_VALID_BINS_PER_TRIAL = 2           # a trial needs at least this many non-NaN bins in the window to contribute at all
MIN_PAIRS_PER_LAG = 20                 # a lag with fewer than this many valid (trial, t) pairs POOLED across the whole
                                       # session is left NaN rather than reported from too little data (see "POOLING ACROSS
                                       # TRIALS" in the module docstring -- this is a pooled, not a within-trial, count)
N_BOOT = 200                           # trial-resampling bootstrap draws for the curve's SEM

AGG_COLORS = {"agree": "#2ca02c", "disagree": "#7b3294"}      # green/purple, matching the paper's Fig. 4g/4h agree/disagree
AREA_LINESTYLE = "-"
JOBLIB_VERBOSE = 5
ENGINE_VERSION = "4d-v4"               # bumped: V1 lab/unl x PM shallow/deep x lab/unl breakdown added (see "SECOND
                                       # EXTENSION" in the module docstring) -- forces a cache recompute

LAYERS = ["shallow", "deep"] if INCLUDE_DEEP_PM else ["shallow"]      # PM depth -- see "SECOND EXTENSION" above
V1_LABELS = ["lab", "unl"]             # V1 projection-labeling -- see "SECOND EXTENSION" above
PM_LABELS = ["lab", "unl"]             # PM projection-labeling -- see "SECOND EXTENSION" above
PM_SUBGROUPS = [(layer, label) for layer in LAYERS for label in PM_LABELS]     # e.g. [("shallow","lab"), ("shallow","unl"),
                                                                               # ("deep","lab"), ("deep","unl")]

parser = argparse.ArgumentParser()
add_scope_args(parser)
args = parser.parse_args()
scope = scope_from_args(args)
AGG = scope.aggregate
UNIT = unit_noun(AGG)
UNIT_KEYS = {"pooled": [], "session": ["animal_id", "session_id"], "mouse": ["animal_id"]}[AGG]

WINDOW_NAMES = list(WINDOWS_MS) + (["reward"] if INCLUDE_REWARD_WINDOW else [])
ALL_WINDOWS_MS = dict(WINDOWS_MS, **({"reward": REWARD_WINDOW_MS} if INCLUDE_REWARD_WINDOW else {}))

paths = get_pipeline_paths(__file__)
table_dir, figdir = scoped_dirs(paths, scope)
clear_figures(figdir)
set_style()
print(scope.describe())

# --------------------------------------------------------------------- #
# 1. Included sessions (1b) and cells (2a + 2b), split into V1 / PM
# --------------------------------------------------------------------- #
fits_path = paths.out_from("1b_psychometric") / "psychometric_fits.csv"
cell_dist_out = paths.out_from("2a_cell_distribution")
activity_out = paths.out_from("2b_activity_statistics")
for f, hint in [(fits_path, "1b_psychometric.py"), (cell_dist_out / "celldata_combined.csv", "2a_cell_distribution.py"),
                (activity_out / "qc_per_cell.csv", "2b_activity_statistics.py")]:
    if not f.exists():
        raise SystemExit(f"Run {hint} first -- expected {f}")
fits_df = pd.read_csv(fits_path)
included_all = set(fits_df[fits_df["included"] & (fits_df["protocol"] == PROTOCOL)]["session_id"])
all_triples = discover_sessions([PROTOCOL])
triples = select_sessions(all_triples, scope)
selected = {f"{a}_{d}": (p, a, d) for p, a, d in triples}
session_ids = [s for s in selected if s in included_all]
if not session_ids:
    raise SystemExit("None of the selected sessions passed 1b_psychometric's inclusion criteria.")
mouse_colors = mouse_palette(a for _, a, _ in all_triples)

cell_inclusion = load_cell_inclusion(cell_dist_out, activity_out)


def parse_group(group: str) -> tuple[str | None, str | None, str | None]:
    """group is the project's own group key (e.g. "V1_lab", "PMdeep_unl", see celldata_utils.py). Returns
    (area, pm_layer, label): area in {"V1", "PM", None}; pm_layer is None for V1, "shallow"/"deep" for PM; label is
    "lab"/"unl" for either area (see "SECOND EXTENSION" in the module docstring)."""
    if group in ("V1_lab", "V1_unl"):
        return "V1", None, group.split("_", 1)[1]
    if group in ("PM_lab", "PM_unl"):
        return "PM", "shallow", group.split("_", 1)[1]
    if INCLUDE_DEEP_PM and group in ("PMdeep_lab", "PMdeep_unl"):
        return "PM", "deep", group.split("_", 1)[1]
    return None, None, None


cell_inclusion = cell_inclusion[cell_inclusion["include"]].copy()
parsed = cell_inclusion["group"].map(parse_group)
cell_inclusion["area"] = [p[0] for p in parsed]
cell_inclusion["pm_layer"] = [p[1] for p in parsed]
cell_inclusion["cell_label"] = [p[2] for p in parsed]
cell_inclusion = cell_inclusion[cell_inclusion["area"].notna()]
cells_by_session = {}
for sid, sub in cell_inclusion.groupby("session_id"):
    v1_by_label = {label: sorted(sub.loc[(sub["area"] == "V1") & (sub["cell_label"] == label), "cell_id"])
                  for label in V1_LABELS}
    pm_by_subgroup = {(layer, label): sorted(sub.loc[(sub["area"] == "PM") & (sub["pm_layer"] == layer) &
                                                      (sub["cell_label"] == label), "cell_id"])
                      for (layer, label) in PM_SUBGROUPS}
    # a session qualifies if at least one V1 label AND at least one PM subgroup clear the bar -- a specific combination
    # that doesn't is simply skipped for that session inside process_session, not dropped from the whole run
    v1_ok = any(len(c) >= MIN_CELLS_PER_AREA for c in v1_by_label.values())
    pm_ok = any(len(c) >= MIN_CELLS_PER_AREA for c in pm_by_subgroup.values())
    if sid in session_ids and v1_ok and pm_ok:
        cells_by_session[sid] = {"V1": v1_by_label, "PM": pm_by_subgroup}
session_ids = [s for s in session_ids if s in cells_by_session]
if not session_ids:
    raise SystemExit(f"No session has >= {MIN_CELLS_PER_AREA} included cells in at least one V1 label and at least one "
                     f"PM subgroup.")
animal_of = {s: selected[s][1] for s in session_ids}
stamps = {s: data_stamp(*selected[s]) for s in session_ids}
n_by_v1_label = {label: sum(len(v["V1"][label]) for v in cells_by_session.values()) for label in V1_LABELS}
n_by_pm_subgroup = {sg: sum(len(v["PM"][sg]) for v in cells_by_session.values()) for sg in PM_SUBGROUPS}
print(f"{len(session_ids)} sessions; V1 cells by label: {n_by_v1_label}; PM cells by (layer, label): {n_by_pm_subgroup}")


# --------------------------------------------------------------------- #
# 2. Per-trial normalized autocorrelation (eqs. 37-41 of the paper)
# --------------------------------------------------------------------- #
def pooled_autocorr(mat: np.ndarray, max_lag_bins: int, min_pairs: int = MIN_PAIRS_PER_LAG) -> tuple[np.ndarray, np.ndarray]:
    """mat: (n_trials, n_bins), already mean-subtracted ACROSS TRIALS at each bin (eq. 39/40). For every lag in
    0..max_lag_bins (bins), C(lag) = sum_{k,t valid}[x_k(t)*x_k(t+lag)] / sum_{k,t valid}[x_k(t)**2], pooling the (trial, t)
    pairs across the WHOLE session before dividing (see "POOLING ACROSS TRIALS" in the module docstring) -- C(0) == 1 by
    construction. NaN at lags with fewer than `min_pairs` pooled-valid (trial, t) pairs. Returns (C, n_pairs)."""
    n_trials, n_bins = mat.shape
    max_lag_bins = int(min(max_lag_bins, n_bins - 1))
    C = np.full(max_lag_bins + 1, np.nan)
    n_pairs = np.zeros(max_lag_bins + 1, dtype=int)
    for lag in range(max_lag_bins + 1):
        if lag == 0:
            valid = np.isfinite(mat)
            n_pairs[0] = int(valid.sum())
            if n_pairs[0] >= min_pairs:
                C[0] = 1.0
            continue
        a0, a1 = mat[:, :-lag], mat[:, lag:]
        valid = np.isfinite(a0) & np.isfinite(a1)
        n_pairs[lag] = int(valid.sum())
        if n_pairs[lag] < min_pairs:
            continue
        denom = np.mean(a0[valid] ** 2)
        if denom > 0:
            C[lag] = np.mean(a0[valid] * a1[valid]) / denom
    return C, n_pairs


def bootstrap_autocorr_sem(mat: np.ndarray, max_lag_bins: int, min_pairs: int, n_boot: int, rng) -> np.ndarray:
    """SEM of `pooled_autocorr`'s C(lag) via resampling TRIALS (rows of `mat`) with replacement -- the error bar a per-trial
    standard error can no longer supply once the ratio is taken pooled rather than per-trial (see module docstring)."""
    n_trials, n_bins = mat.shape
    n_lags = int(min(max_lag_bins, n_bins - 1)) + 1
    boots = np.full((n_boot, n_lags), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n_trials, n_trials)
        C_b, _ = pooled_autocorr(mat[idx], max_lag_bins, min_pairs=min_pairs)
        boots[b, :len(C_b)] = C_b
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sem = np.nanstd(boots, axis=0, ddof=1)
    return sem


def area_under_curve(C: np.ndarray) -> float:
    """Sum of C(lag) for lag > 0 (lag 0 == 1 by construction, excluded) -- a monotonic, bin-count-robust proxy for "how
    slowly this session's autocorrelation decays" (NOT the paper's Delta_slow, see module docstring)."""
    vals = C[1:]
    return float(np.nansum(vals)) if np.isfinite(vals).any() else np.nan


def process_session(ses, seed):
    if ses.trialdata is None or ses.behaviordata is None or ses.calciumdata is None or ses.ts_F is None:
        return []
    v1_cells_by_label = {label: [c for c in cells_by_session[ses.session_id]["V1"][label] if c in ses.calciumdata.columns]
                        for label in V1_LABELS}
    usable_v1_labels = [l for l in V1_LABELS if len(v1_cells_by_label[l]) >= MIN_CELLS_PER_AREA]
    pm_cells_by_subgroup = {sg: [c for c in cells_by_session[ses.session_id]["PM"][sg] if c in ses.calciumdata.columns]
                            for sg in PM_SUBGROUPS}
    usable_pm_subgroups = [sg for sg in PM_SUBGROUPS if len(pm_cells_by_subgroup[sg]) >= MIN_CELLS_PER_AREA]
    if not usable_v1_labels or not usable_pm_subgroups:
        return []

    fs = get_frame_rate(ses, fallback=None)
    binsize_s = (1.0 / fs) if BINSIZE_MODE == "frame" else (FIXED_BINSIZE_MS / 1000.0)

    all_v1_cells = sorted(set().union(*(v1_cells_by_label[l] for l in usable_v1_labels)))
    all_pm_cells = sorted(set().union(*(pm_cells_by_subgroup[sg] for sg in usable_pm_subgroups)))
    frame_table = build_calcium_continuous(ses.calciumdata.loc[:, all_v1_cells + all_pm_cells], ses.ts_F, ses.behaviordata)
    small = pd.DataFrame({"ts": frame_table["ts"]})
    for label in usable_v1_labels:
        small[f"v1_mean_{label}"] = frame_table[v1_cells_by_label[label]].mean(axis=1, skipna=True)
    for (layer, label) in usable_pm_subgroups:
        small[f"pm_mean_{layer}_{label}"] = frame_table[pm_cells_by_subgroup[(layer, label)]].mean(axis=1, skipna=True)

    trials = ses.trialdata
    if ENGAGED_ONLY and "engaged" in trials.columns:
        trials = trials[trials["engaged"] == 1]
    if CORRECT_ONLY and "trialOutcome" in trials.columns:
        trials = trials[trials["trialOutcome"].isin(["HIT", "CR"])]
    elif CORRECT_ONLY:
        from infotheory.behavior import add_trial_outcome
        trials = add_trial_outcome(trials)
        trials = trials[trials["trialOutcome"].isin(["HIT", "CR"])]
    if trials.empty:
        return []

    records = []
    for window, (t_pre_ms, t_post_ms, onset_col) in ALL_WINDOWS_MS.items():
        if onset_col not in trials.columns:
            continue
        trials_on = derive_onset_time(trials, ses.behaviordata, position_value_col=onset_col)
        window_len_ms = t_post_ms - t_pre_ms
        max_lag_ms = window_len_ms if MAX_LAG_MS is None else min(MAX_LAG_MS, window_len_ms)
        max_lag_bins = int(round(max_lag_ms / (binsize_s * 1000.0)))
        if max_lag_bins < 1:
            continue

        v1_cols = [f"v1_mean_{l}" for l in usable_v1_labels]
        pm_cols = [f"pm_mean_{layer}_{label}" for (layer, label) in usable_pm_subgroups]
        long_df = align_trials_time(trials_on, small, columns=v1_cols + pm_cols,
                                    t_pre=t_pre_ms / 1000.0, t_post=t_post_ms / 1000.0, binsize=binsize_s)
        if long_df.empty:
            continue
        v1_wide_by_label = {l: long_df[long_df["variable"] == f"v1_mean_{l}"].pivot_table(
            index="trialNumber", columns="t_rel", values="value") for l in usable_v1_labels}
        pm_wide_by_subgroup = {sg: long_df[long_df["variable"] == f"pm_mean_{sg[0]}_{sg[1]}"].pivot_table(
            index="trialNumber", columns="t_rel", values="value") for sg in usable_pm_subgroups}
        rng = np.random.default_rng(seed if seed is not None else 0)

        for v1_label in usable_v1_labels:
            v1_wide_full = v1_wide_by_label[v1_label]
            for (layer, pm_label) in usable_pm_subgroups:
                pm_wide_full = pm_wide_by_subgroup[(layer, pm_label)]
                v1_wide, pm_wide = v1_wide_full.align(pm_wide_full, join="inner", axis=None)
                v1_wide, pm_wide = v1_wide.sort_index(axis=1), pm_wide.sort_index(axis=1)
                usable = (v1_wide.notna().sum(axis=1) >= MIN_VALID_BINS_PER_TRIAL) & \
                         (pm_wide.notna().sum(axis=1) >= MIN_VALID_BINS_PER_TRIAL)
                v1_w, pm_w = v1_wide.loc[usable], pm_wide.loc[usable]
                n_trials = len(v1_w)
                if n_trials < MIN_TRIALS_WINDOW:
                    continue

                for mode, mat in (("agree", v1_w.to_numpy() + pm_w.to_numpy()),
                                  ("disagree", v1_w.to_numpy() - pm_w.to_numpy())):
                    mat_bar = mat - np.nanmean(mat, axis=0, keepdims=True)   # mean-subtract ACROSS TRIALS (eq. 39/40)
                    C, n_pairs = pooled_autocorr(mat_bar, max_lag_bins, min_pairs=MIN_PAIRS_PER_LAG)
                    sem_C = bootstrap_autocorr_sem(mat_bar, max_lag_bins, MIN_PAIRS_PER_LAG, N_BOOT, rng)
                    lag_bins = np.arange(0, len(C))
                    lag_ms = lag_bins * binsize_s * 1000.0
                    for b, lg, mc, sc, npz in zip(lag_bins, lag_ms, C, sem_C, n_pairs):
                        records.append(dict(kind="curve", session_id=ses.session_id, animal_id=ses.animal_id,
                                            protocol=PROTOCOL, window=window, v1_label=v1_label, pm_layer=layer,
                                            pm_label=pm_label, mode=mode, lag_bin=int(b), lag_ms=float(lg),
                                            mean_autocorr=float(mc) if np.isfinite(mc) else np.nan,
                                            sem_autocorr=float(sc) if np.isfinite(sc) else np.nan,
                                            n_trials=n_trials, n_pairs=int(npz)))
                    auc = area_under_curve(C)
                    records.append(dict(kind="summary", session_id=ses.session_id, animal_id=ses.animal_id,
                                        protocol=PROTOCOL, window=window, v1_label=v1_label, pm_layer=layer,
                                        pm_label=pm_label, mode=mode, mean_auc=float(auc) if np.isfinite(auc) else np.nan,
                                        n_trials=n_trials, n_v1_cells=len(v1_cells_by_label[v1_label]),
                                        n_pm_cells=len(pm_cells_by_subgroup[(layer, pm_label)]),
                                        binsize_ms=binsize_s * 1000.0, max_lag_bins=len(C) - 1))
    return records


key_params = dict(version=ENGINE_VERSION, areas=f"v1-pm{'+deep' if INCLUDE_DEEP_PM else ''}", v1_labels=repr(V1_LABELS),
                  pm_subgroups=repr(PM_SUBGROUPS), min_cells=MIN_CELLS_PER_AREA, engaged_only=ENGAGED_ONLY,
                  correct_only=CORRECT_ONLY, windows=repr(sorted(ALL_WINDOWS_MS.items())), binsize_mode=BINSIZE_MODE,
                  fixed_binsize_ms=FIXED_BINSIZE_MS, max_lag_ms=MAX_LAG_MS, min_trials_window=MIN_TRIALS_WINDOW,
                  min_valid_bins=MIN_VALID_BINS_PER_TRIAL, min_pairs_per_lag=MIN_PAIRS_PER_LAG, n_boot=N_BOOT)
records = run_sessions(session_ids, process_session, PROTOCOL, paths.store, "interarea_consensus", key_params,
                       load_kwargs=dict(load_behaviordata=True, load_videodata=False, load_celldata=True,
                                        load_calciumdata=True),
                       n_jobs=scope.n_jobs, recompute=args.recompute, verbose=JOBLIB_VERBOSE, extra_key_by_session=stamps)
if not records:
    raise SystemExit("No session produced usable trials in any window -- check MIN_CELLS_PER_AREA / MIN_TRIALS_WINDOW "
                     "or whether stimStart/rewardZoneStart exist in trialdata.")

all_df = pd.DataFrame(records)
CURVE_COLS = ["session_id", "animal_id", "protocol", "window", "v1_label", "pm_layer", "pm_label", "mode", "lag_bin",
             "lag_ms", "mean_autocorr", "sem_autocorr", "n_trials", "n_pairs"]
SUMMARY_COLS = ["session_id", "animal_id", "protocol", "window", "v1_label", "pm_layer", "pm_label", "mode", "mean_auc",
                "n_trials", "n_v1_cells", "n_pm_cells", "binsize_ms", "max_lag_bins"]
curve_df = all_df[all_df["kind"] == "curve"][CURVE_COLS].reset_index(drop=True)
summary_df = all_df[all_df["kind"] == "summary"][SUMMARY_COLS].reset_index(drop=True)
curve_df.to_csv(table_dir / "interarea_autocorr_curves.csv", index=False)
summary_df.to_csv(table_dir / "interarea_autocorr_summary.csv", index=False)
print(f"{summary_df['session_id'].nunique()} sessions with >= {MIN_TRIALS_WINDOW} usable trials in at least one "
     f"window/combination; V1 labels={V1_LABELS}, PM subgroups={PM_SUBGROUPS}")


# --------------------------------------------------------------------- #
# 3. Figures
# --------------------------------------------------------------------- #
def agg_curve(df, window, mode):
    sub = df[(df["window"] == window) & (df["mode"] == mode)]
    if sub.empty:
        return None
    if AGG == "pooled":
        g = sub.groupby("lag_ms")["mean_autocorr"].mean().reset_index()
        return g["lag_ms"].to_numpy(), g["mean_autocorr"].to_numpy(), None
    if AGG == "mouse":
        per_unit = sub.groupby(["animal_id", "lag_ms"])["mean_autocorr"].mean().reset_index()
    else:
        per_unit = sub.rename(columns={"mean_autocorr": "mean_autocorr"})
    g = per_unit.groupby("lag_ms")["mean_autocorr"]
    mean, n = g.mean(), g.count()
    sem = g.std(ddof=1) / np.sqrt(n.clip(lower=1))
    return mean.index.to_numpy(), mean.to_numpy(), sem.to_numpy()


def agg_curve_combo(df, window, v1_label, pm_layer, pm_label, mode):
    sub = df[(df["v1_label"] == v1_label) & (df["pm_layer"] == pm_layer) & (df["pm_label"] == pm_label)]
    return agg_curve(sub, window, mode)


def pm_subgroup_label(layer, label):
    return f"PM {layer}-{label}"


# Shared y-axis across BOTH V1-label figure files, per the project's shared-axis convention
ylim = (0.0, 1.05)
if not curve_df.empty:
    ylim = (min(ylim[0], curve_df["mean_autocorr"].min() - 0.05), ylim[1])
auc_ylim = (0.0, 1.0)
if not summary_df.empty:
    auc_ylim = (min(0.0, summary_df["mean_auc"].min() - 0.1), summary_df["mean_auc"].max() + 0.1)

# Figure 1 (one file per V1 label): mean autocorrelation curves, one row per PM subgroup (shallow/deep x lab/unl), one
# column per window, agree vs disagree -- "all relevant combinations" of labeled/unlabeled/shallow/deep (see "SECOND
# EXTENSION" in the module docstring)
for v1_label in V1_LABELS:
    fig, axes = plt.subplots(len(PM_SUBGROUPS), len(WINDOW_NAMES), figsize=(4.2 * len(WINDOW_NAMES), 3.2 * len(PM_SUBGROUPS)),
                             sharey=True, sharex=True, squeeze=False)
    for i, (layer, pm_label) in enumerate(PM_SUBGROUPS):
        for j, window in enumerate(WINDOW_NAMES):
            ax = axes[i, j]
            for mode in ("agree", "disagree"):
                out = agg_curve_combo(curve_df, window, v1_label, layer, pm_label, mode)
                if out is None:
                    continue
                x, y, sem = out
                order = np.argsort(x)
                x, y = x[order], y[order]
                ax.plot(x, y, color=AGG_COLORS[mode], lw=1.6, label=mode, marker="o", markersize=3)
                if sem is not None:
                    sem = sem[order]
                    ax.fill_between(x, y - sem, y + sem, color=AGG_COLORS[mode], alpha=0.2, linewidth=0)
            ax.axhline(0, color="grey", lw=0.6, linestyle="--")
            if i == 0:
                t_pre_ms, t_post_ms, onset_col = ALL_WINDOWS_MS[window]
                ax.set_title(f"{window} ({t_pre_ms:g}..{t_post_ms:g} ms rel. {onset_col})", fontsize=9)
            if i == len(PM_SUBGROUPS) - 1:
                ax.set_xlabel("lag (ms)")
            if j == 0:
                ax.set_ylabel(f"{pm_subgroup_label(layer, pm_label)}\nautocorr., mean +- SEM over {UNIT}s", fontsize=8)
            ax.set_ylim(*ylim)
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"V1-PM inter-area consensus (V1 {v1_label} cells): agree (V1+PM) vs disagree (V1-PM) autocorrelation -- "
                f"{scope.describe()}", fontsize=9)
    fig.tight_layout(h_pad=2.0, rect=(0, 0, 1, 0.95))
    save_fig(fig, figdir, f"1_autocorr_by_window_V1-{v1_label}")
    plt.close(fig)

# Figure 2 (one file per V1 label): paired per-session AUC, agree vs disagree, one row per PM subgroup, one column per
# window, Wilcoxon p across SESSIONS within that combination (the paired unit the paper itself uses -- "two-sided
# Wilcoxon signed-rank test" across animals/sessions -- irrespective of --aggregate, which only controls figure 1's
# curve averaging).
for v1_label in V1_LABELS:
    fig, axes = plt.subplots(len(PM_SUBGROUPS), len(WINDOW_NAMES), figsize=(3.6 * len(WINDOW_NAMES), 3.4 * len(PM_SUBGROUPS)),
                             sharey=True, squeeze=False)
    for i, (layer, pm_label) in enumerate(PM_SUBGROUPS):
        for j, window in enumerate(WINDOW_NAMES):
            ax = axes[i, j]
            sub = summary_df[(summary_df["window"] == window) & (summary_df["v1_label"] == v1_label) &
                             (summary_df["pm_layer"] == layer) & (summary_df["pm_label"] == pm_label)]
            piv = sub.pivot_table(index="session_id", columns="mode", values="mean_auc")
            piv = piv.dropna(subset=["agree", "disagree"]) if {"agree", "disagree"}.issubset(piv.columns) else piv.iloc[0:0]
            if piv.empty:
                ax.set_title(f"{window}\n(no sessions)", fontsize=9)
            else:
                for sid, r in piv.iterrows():
                    c = mouse_colors.get(animal_of.get(sid), "black")
                    ax.plot([0, 1], [r["agree"], r["disagree"]], color=c, alpha=0.6, lw=1.0, marker="o", markersize=4)
                n = len(piv)
                if n >= 2:
                    stat, p = wilcoxon(piv["agree"], piv["disagree"])
                    ax.set_title(f"{window}\nn={n} sessions, Wilcoxon p={p:.3g}", fontsize=9)
                else:
                    ax.set_title(f"{window}\nn={n} sessions (too few for a test)", fontsize=9)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["agree", "disagree"])
            ax.set_ylim(*auc_ylim)
            if j == 0:
                ax.set_ylabel(f"{pm_subgroup_label(layer, pm_label)}\nAUC (lag>0), per session", fontsize=8)
    fig.suptitle(f"V1-PM AUC summary (V1 {v1_label} cells) -- {scope.describe()}", fontsize=9)
    fig.tight_layout(h_pad=3.0, rect=(0, 0, 1, 0.96))
    save_fig(fig, figdir, f"2_auc_summary_V1-{v1_label}")
    plt.close(fig)

print(f"Done. Tables -> {table_dir}; figures -> {figdir}")
