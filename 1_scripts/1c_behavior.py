# -*- coding: utf-8 -*-
"""
1c_behavior
===========
Behavioral variables analyzed in relation to themselves: running speed,
position, pupil size, and video motion energy. Uses the sessions that
passed 1b's inclusion criteria, engaged trials only (Rule #1: reads 1b's
out/, doesn't recompute its exclusions).

What this does
--------------
1. Loads behaviordata+videodata for included sessions only, merges them
   onto one timebase, restricts to engaged-trial windows, and strips
   pupil blink artifacts (|z|>5).
2. Exploratory plots: per-protocol distributions, position ("corridor")
   tuning of each variable, and a pairwise view of all four variables
   together with linear fits overlaid.
3. Per-trial features (mean speed/pupil/motion) in each of three
   POSITION windows relative to stimulus onset -- pre_stim, stim,
   reward (see MI_WINDOWS below, given as [start_cm, end_cm] since
   stimStart is itself a position) -- as the unit for MI and
   regression; raw ~50Hz samples are too autocorrelated to treat as
   independent draws, which would badly overstate significance.
4. For every pair of variables, per protocol, per window: mutual
   information (KSG estimate, bias-corrected via shuffling) AND a
   linear fit (r^2). Compares the MI implied by r^2 under a
   linear/Gaussian assumption against the actual (corrected) MI -- a
   big gap means the relationship has a nonlinear component a linear
   model would miss (the same logic the project's a_docs describes for
   comparing 2c_information against 2d_linear_encod/2e_nonlinear_encod
   later on, applied here at the behavioral level first).
5. Same comparison, but MULTIVARIATE (not just pairwise): for each
   variable, a linear fit and a joint MI against ALL other variables
   AT ONCE (leave-one-out) -- a variable can be poorly predicted by any
   single other variable pairwise while still being well predicted by
   several of them together (see section 4b/5b below).
6. Forward-stepwise selection (section 4c/8): the leave-one-out R^2
   alone can't tell "many variables genuinely add independent
   information" apart from "one dominant predictor does nearly all the
   work, the rest just ride along on their correlation with it" (e.g. a
   pupil x/y tracking pair). Adds predictors one at a time, whichever
   gives the biggest R^2 gain at each step, so step 1's R^2 is the
   single best predictor's own pairwise R^2 -- directly showing how
   much (if anything) the rest of the predictor set adds beyond it.

Reads: 2_pipeline/1b_psychometric/out/psychometric_fits.csv for the
included-session list (Rule #1: read an earlier script's out/, don't
recompute its exclusions).
Loads: raw behaviordata/videodata for those sessions -- 1b never loaded
these (only trialdata), so there's no cache to reuse here.

Outputs
-------
2_pipeline/1c_behavior/
    out/    behavior_mi_vs_linear.csv   one row per (protocol, window,
                                         variable pair): r, r2, MI
                                         raw/corrected, MI implied by r2,
                                         whether linear looks sufficient.
                                         window is one of MI_WINDOWS'
                                         keys (pre_stim/stim/reward by
                                         default, each a [start_cm,
                                         end_cm] position range -- see
                                         MI_WINDOWS below)
            behavior_multivariate_mi_vs_linear.csv   one row per
                                         (protocol, window, target
                                         variable): the same comparison,
                                         but for a multivariate fit of
                                         that target from every OTHER
                                         variable at once (see section
                                         4b) -- includes adj_r2 and the
                                         predictor list used
            behavior_multivariate_stepwise.csv   one row per (protocol,
                                         window, target, step): forward-
                                         stepwise predictor selection
                                         (section 4c) -- which predictor
                                         got added at each step, and the
                                         cumulative/incremental r2 --
                                         tells apart "many predictors
                                         genuinely add information" from
                                         "one dominant predictor does
                                         nearly all the work"
            figures/*.png                includes one
                                         4_mi_vs_linear_summary_<window>.png
                                         and one
                                         7_multivariate_mi_vs_linear_summary_<window>.png
                                         and one
                                         8_stepwise_dominant_predictor_<window>.png
                                         per MI_WINDOWS entry
    store/  trial_features.pkl          per-trial mean speed/pupil/motion, one row per
                                         (trial, window) -- see MI_WINDOWS below
            position_binned.pkl         corridor-tuning aggregates (whole-session, not trial-locked)
            pooled_downsampled.pkl      pooled continuous samples for the exploratory
                                         distribution/pairwise plots (full resolution by
                                         default -- see DOWNSAMPLE_HZ below)
            psth_time.pkl               trial-aligned time-domain traces (onset = stimulus)
            psth_position.pkl           trial-aligned position-domain traces (onset = stimulus)
            (all reused unless --recompute is passed)

Schema note: `stimStart` (and rewardZoneStart/rewardZoneEnd) are
POSITIONS, on the same scale as `zpos` -- confirmed against the real
pipeline's tensor_utils.py/behavior_signals.py, which always compare
them directly (`zpos_F - z_T[k]`). Position-domain alignment
(`align_trials_position`) uses that subtraction directly; time-domain
alignment (`align_trials_time`, and the per-trial window features
below) needs an actual timestamp, so `psth.derive_onset_time` finds the
wall-clock time at which each trial's own position trace first reaches
`stimStart` and uses that as the anchor instead.
"""
from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

from infotheory.pipeline import get_pipeline_paths
from infotheory.session import load_sessions, PROTOCOLS
from infotheory.continuous import (
    merge_behavior_video, check_zpos_consistency, restrict_to_engaged, remove_pupil_outliers,
    compute_trial_position_window_means, bin_by_position,
)
from infotheory.psth import align_trials_time, align_trials_position, derive_onset_time
from infotheory.behavior import add_trial_outcome
from infotheory.info_theory import mutual_information_shuffle_ksg, compare_mi_to_linear, linear_fit, \
    multivariate_linear_fit, quantile_bin_edges, forward_stepwise_selection
from infotheory.plotting import set_style, save_fig, clear_figures, PROTOCOL_COLORS, OUTCOME_COLORS

# All behavior variables considered "in relation to themselves". Checked
# for availability per-session (`if c in merged.columns`), so this list
# can be longer than what any given dataset actually has -- extend it
# freely (e.g. more videoPC_k's) without breaking sessions that lack them.
VARIABLES = ["zpos", "runspeed", "lick", "pupil_area", "pupil_xpos", "pupil_ypos",
             "motionenergy", "videoPC_0", "videoPC_1"]
VAR_LABELS = {"zpos": "position (cm)", "runspeed": "speed (cm/s)", "lick": "lick rate",
              "pupil_area": "pupil area (a.u.)", "pupil_xpos": "pupil x (a.u.)",
              "pupil_ypos": "pupil y (a.u.)", "motionenergy": "motion energy (a.u.)",
              "videoPC_0": "video PC 1", "videoPC_1": "video PC 2"}
RATE_VARIABLES = ["lick"]  # 0/1 event channels -- binned as counts/width (a rate), never averaged

# MI/regression are computed separately in each of these windows,
# defined as [start_cm, end_cm] relative to `stimStart` -- a POSITION
# (cm), directly comparable to `zpos` (see psth.py's module docstring:
# `stimStart` is on the same scale as `zpos`, not a timestamp). Given
# task geometry: stimulus zone is 20cm starting at onset, then a 5cm
# gap, then a 20cm reward zone -- adjust freely for your own geometry.
MI_WINDOWS = {
    "pre_stim": (-30.0, -10.0),
    "stim": (0.0, 20.0),
    "reward": (25.0, 45.0),
}

N_POSITION_BINS = 24
KSG_K = 5             # neighbors for the KSG mutual-information estimator (doesn't need a bin count,
                      # and doesn't share histogram MI's downward bias on strong relationships)
N_SHUFFLES = 20

# Pooled continuous samples (variable-distributions and pairwise-relationship
# plots only -- NOT the MI/regression numbers, which use the per-trial,
# per-window means above) can optionally be downsampled per session to limit
# autocorrelation in those two EXPLORATORY plots. None (default) = use every
# sample, no downsampling. Set to a number (e.g. 2.0) to downsample to that
# many Hz per session if the plots get too dense/slow to render.
DOWNSAMPLE_HZ = None

# Separate, purely-for-rendering-speed cap on the pairwise plot specifically
# (sns.pairplot with kind="reg" fits a regression per hue group, which gets
# slow with very large N): None (default) = plot every pooled sample, however
# many that is. Set an int (e.g. 6000) to randomly subsample to about that
# many total rows (split evenly across protocols) before rendering ONLY the
# pairwise-relationships figure -- doesn't touch the saved data or any other
# plot/table.
PAIRPLOT_MAX_ROWS = None

LINEAR_SUFFICIENT_FRAC = 0.8  # linear MI / corrected MI >= this -> "linear looks sufficient"
PSTH_VARS = ["runspeed", "lick", "pupil_area", "motionenergy"]
PSTH_T_PRE, PSTH_T_POST, PSTH_T_BINSIZE = -3.0, 3.0, 0.15      # trial-aligned time window/bin (s)
PSTH_S_PRE, PSTH_S_POST, PSTH_S_BINSIZE = -80.0, 60.0, 5.0     # trial-aligned position window/bin (cm)

# Parallelism (joblib, matching the project's convention -- see e.g. the
# uploaded plot_mi_behavior.py/plot_temporal_information.py, which
# parallelize the same way): -1 = use all available cores. Session
# loading/feature-extraction is embarrassingly parallel (one session's
# work never depends on another's); the MI computation loop is
# parallelized separately, coarse-grained over (protocol, window) groups
# -- see section 4 -- since KSG with shuffling is the other real cost.
N_JOBS_SESSIONS = -1
N_JOBS_MI = -1
JOBLIB_VERBOSE = 5

parser = argparse.ArgumentParser()
parser.add_argument("--recompute", action="store_true",
                     help="Reload sessions from 0_data/ instead of reusing the cached store/ tables")
args = parser.parse_args()

paths = get_pipeline_paths(__file__)
figdir = paths.out / "figures"
clear_figures(figdir)
set_style()
rng = np.random.default_rng(0)


# --------------------------------------------------------------------- #
# 1. Included sessions from 1b
# --------------------------------------------------------------------- #
fits_path = paths.out_from("1b_psychometric") / "psychometric_fits.csv"
if not fits_path.exists():
    raise SystemExit(f"Run 1b_psychometric.py first -- expected {fits_path}")

fits_df = pd.read_csv(fits_path)
included = fits_df[fits_df["included"]]
included_ids = included["session_id"].tolist()
if not included_ids:
    raise SystemExit("No sessions passed 1b_psychometric's inclusion criteria -- nothing to analyze.")
print(f"Using {len(included_ids)} included sessions from 1b_psychometric "
      f"({included['protocol'].value_counts().to_dict()})")

# mu/sigma per session (only ever non-NaN for DN, since that's the only
# protocol 1b actually fits) -- used below to z-score intermediate signal
# levels into signal_psy, matching the old noise_to_psy convention.
mu_sigma_lookup = fits_df.set_index("session_id")[["mu", "sigma"]].to_dict("index")


# --------------------------------------------------------------------- #
# 2. Per-trial features + position-binned corridor stats -- or cache
# --------------------------------------------------------------------- #
trial_features_cache = paths.store / "trial_features.pkl"
position_binned_cache = paths.store / "position_binned.pkl"
time_course_cache = paths.store / "time_course.pkl"
pooled_cache = paths.store / "pooled_downsampled.pkl"
psth_time_cache = paths.store / "psth_time.pkl"
psth_position_cache = paths.store / "psth_position.pkl"

# Schema required in each cached table -- checked after loading so a
# cache built by an OLDER version of this script (before some column
# existed, e.g. frac_position) is treated as a miss and recomputed,
# rather than crashing with a bare KeyError deep in the plotting code.
# Bump/extend this whenever a cached table's columns change.
REQUIRED_CACHE_COLUMNS = {
    "trial_features": ["window", "session_id", "protocol"],
    "position_binned": ["frac_position", "bin_center", "session_id", "protocol"],
    "time_course": ["bin_center", "session_id", "protocol"],
    "pooled": ["session_id", "protocol"],
    "psth_time": ["t_rel", "variable", "value", "trialOutcome"],
    "psth_position": ["s_rel", "variable", "value", "trialOutcome"],
}


def _load_cache_if_valid():
    """Load all six cached tables and check their columns against
    REQUIRED_CACHE_COLUMNS. Returns the tables on success, or None (and
    prints why) if anything is missing/stale -- the caller should then
    fall through to recomputing from 0_data/."""
    cache_files = {
        "trial_features": trial_features_cache, "position_binned": position_binned_cache,
        "time_course": time_course_cache, "pooled": pooled_cache,
        "psth_time": psth_time_cache, "psth_position": psth_position_cache,
    }
    if args.recompute or not all(p.exists() for p in cache_files.values()):
        return None
    try:
        loaded = {name: pd.read_pickle(path) for name, path in cache_files.items()}
    except Exception as e:
        print(f"Could not load cached tables from {paths.store} ({e}); recomputing from 0_data/ ...")
        return None

    for name, required_cols in REQUIRED_CACHE_COLUMNS.items():
        df = loaded[name]
        missing = [c for c in required_cols if not df.empty and c not in df.columns]
        if missing:
            print(f"Cached '{name}' is missing columns {missing} -- looks like it was built by an "
                  f"older version of this script. Recomputing from 0_data/ instead of using {paths.store} "
                  "(safe to delete that folder's *.pkl files to silence this check in the future).")
            return None
    return loaded


cached = _load_cache_if_valid()
if cached is not None:
    print(f"Reusing cached behavior tables from {paths.store} (pass --recompute to reload 0_data/)")
    trial_features = cached["trial_features"]
    position_binned = cached["position_binned"]
    time_course = cached["time_course"]
    pooled = cached["pooled"]
    psth_time = cached["psth_time"]
    psth_position = cached["psth_position"]
else:
    sessions = load_sessions(protocols=PROTOCOLS, load_behaviordata=True, load_videodata=True,
                              only_session_ids=included_ids)

    def process_session(ses, mu_sigma_lookup, compute_zpos_diag=False):
        """All per-session work for one session, returned as a dict of
        the (up to) six sub-tables this step builds -- None for any
        that don't apply (e.g. no zpos column). Kept as a single,
        self-contained function (rather than appending to shared outer-
        scope lists) specifically so it can be dispatched via
        joblib.Parallel across sessions: each worker returns its own
        results independently, then the caller concatenates them --
        no shared mutable state between sessions.
        """
        result = {"session_id": ses.session_id, "zpos_diag": None, "trial_features": None,
                   "position_binned": None, "time_course": None, "pooled": None,
                   "psth_time": None, "psth_position": None}
        if ses.trialdata is None or ses.behaviordata is None:
            return result

        if compute_zpos_diag:
            result["zpos_diag"] = check_zpos_consistency(ses.behaviordata, ses.videodata)

        merged_full = merge_behavior_video(ses.behaviordata, ses.videodata)
        merged_full = remove_pupil_outliers(merged_full)
        if merged_full.empty:
            return result
        # Engaged-only view: for whole-session/pooled stats and the
        # per-trial-mean features, which SHOULD exclude disengaged
        # periods. NOT used for the PSTH alignment below -- that needs
        # the full trace so a window around trial k's onset can still
        # reach into the tail of trial k-1 / the ITI (see psth.py's
        # module docstring for why restricting first is wrong here).
        merged = restrict_to_engaged(merged_full, ses.trialdata)
        if merged.empty:
            return result

        available_vars = [c for c in VARIABLES if c in merged.columns]
        available_rate_vars = [c for c in RATE_VARIABLES if c in available_vars]

        engaged_trials = (ses.trialdata[ses.trialdata["engaged"] == 1]
                           if "engaged" in ses.trialdata.columns else ses.trialdata)
        # stimStart is a POSITION (see psth.py's module docstring), so the
        # per-trial window features below subtract it directly against
        # zpos -- no time derivation needed for that. A derived onset
        # TIME is still needed further down for the time-domain PSTH
        # (align_trials_time), so compute it once here regardless.
        engaged_trials = derive_onset_time(engaged_trials, merged_full)

        # One set of per-trial mean features per MI_WINDOWS entry
        # (pre_stim/stim/reward), stacked into one table with a "window"
        # column -- each window is [start_cm, end_cm] relative to
        # stimStart (a position), see MI_WINDOWS above to adjust the
        # bounds. Uses `merged` (engaged-only) since these features
        # should reflect engaged behavior only -- unlike the PSTH
        # alignment below, these are small, fixed-size windows expected
        # to sit entirely within one trial, so there's no need for the
        # "search the full trace" treatment align_trials_position uses.
        tf_windows = []
        for window_name, (s_start, s_end) in MI_WINDOWS.items():
            tf_w = compute_trial_position_window_means(engaged_trials, merged, columns=available_vars,
                                                         s_start=s_start, s_end=s_end,
                                                         rate_columns=available_rate_vars)
            tf_w["window"] = window_name
            tf_windows.append(tf_w)
        tf = pd.concat(tf_windows, ignore_index=True)
        tf["session_id"] = ses.session_id
        tf["protocol"] = ses.protocol
        result["trial_features"] = tf

        if "zpos" in merged.columns:
            # Session-local bins (not a shared global range -- each
            # session can span a different absolute zpos range,
            # especially since zpos may accumulate across the whole
            # session rather than reset; see the zpos QC diagnostic
            # above). Pooling across sessions therefore uses
            # frac_position (0-1, this session's own range) rather than
            # bin_center directly -- see the plotting section below.
            session_bins = np.linspace(merged["zpos"].min(), merged["zpos"].max(), N_POSITION_BINS + 1)
            corridor_vars = [c for c in available_vars if c != "zpos"]
            pb = bin_by_position(merged, columns=corridor_vars, bins=session_bins,
                                  rate_columns=available_rate_vars)
            span = session_bins[-1] - session_bins[0]
            pb["frac_position"] = (pb["bin_center"] - session_bins[0]) / span if span > 0 else np.nan
            pb["session_id"] = ses.session_id
            pb["protocol"] = ses.protocol
            result["position_binned"] = pb

        # Elapsed-session-time binning, as a whole-session view that
        # stays meaningful regardless of whether `zpos` resets every lap
        # or (as turned out to be the case here) accumulates across the
        # whole session -- see the print diagnostic after this loop.
        elapsed = merged.assign(elapsed_s=merged["ts"] - merged["ts"].iloc[0])
        time_bins = np.linspace(0, elapsed["elapsed_s"].max(), N_POSITION_BINS + 1)
        tc = bin_by_position(elapsed, columns=[c for c in available_vars if c != "zpos"],
                              bins=time_bins, position_col="elapsed_s", rate_columns=available_rate_vars)
        tc["session_id"] = ses.session_id
        tc["protocol"] = ses.protocol
        result["time_course"] = tc

        if DOWNSAMPLE_HZ is not None:
            median_dt = np.median(np.diff(merged["ts"].to_numpy()))
            step = 1 if not np.isfinite(median_dt) or median_dt <= 0 else max(1, int(round(1 / (DOWNSAMPLE_HZ * median_dt))))
        else:
            step = 1
        down = merged.iloc[::step][available_vars].copy()
        down["session_id"] = ses.session_id
        down["protocol"] = ses.protocol
        result["pooled"] = down

        # Per-trial metadata to merge onto the PSTH tables below: outcome
        # (HIT/MISS/FA/CR) and, where a fit exists (DN only), the
        # z-scored intermediate-stimulus regressor signal_psy = (signal - mu) / sigma.
        trial_meta = add_trial_outcome(engaged_trials)
        mu_sigma = mu_sigma_lookup.get(ses.session_id, {})
        if mu_sigma and np.isfinite(mu_sigma.get("mu", np.nan)) and np.isfinite(mu_sigma.get("sigma", np.nan)):
            trial_meta = trial_meta.assign(
                signal_psy=(trial_meta["signal"] - mu_sigma["mu"]) / mu_sigma["sigma"])
        else:
            trial_meta = trial_meta.assign(signal_psy=np.nan)
        trial_meta = trial_meta[["trialNumber", "trialOutcome", "signal", "signal_psy"]]

        psth_vars_avail = [c for c in PSTH_VARS if c in merged_full.columns]
        psth_rate_vars = [c for c in RATE_VARIABLES if c in psth_vars_avail]
        if psth_vars_avail:
            pt = align_trials_time(engaged_trials, merged_full, columns=psth_vars_avail,
                                    t_pre=PSTH_T_PRE, t_post=PSTH_T_POST, binsize=PSTH_T_BINSIZE,
                                    rate_columns=psth_rate_vars)
            if len(pt):
                pt = pt.merge(trial_meta, on="trialNumber", how="left")
                pt["session_id"], pt["protocol"] = ses.session_id, ses.protocol
                result["psth_time"] = pt

            pp = align_trials_position(engaged_trials, merged_full, columns=psth_vars_avail,
                                        s_pre=PSTH_S_PRE, s_post=PSTH_S_POST, binsize=PSTH_S_BINSIZE,
                                        rate_columns=psth_rate_vars)
            if len(pp):
                pp = pp.merge(trial_meta, on="trialNumber", how="left")
                pp["session_id"], pp["protocol"] = ses.session_id, ses.protocol
                result["psth_position"] = pp

        return result

    print(f"Processing {len(sessions)} sessions (n_jobs={N_JOBS_SESSIONS}) ...")
    session_results = Parallel(n_jobs=N_JOBS_SESSIONS, backend="loky", verbose=JOBLIB_VERBOSE)(
        delayed(process_session)(ses, mu_sigma_lookup, compute_zpos_diag=(i == 0))
        for i, ses in enumerate(sessions)
    )

    for r in session_results:
        if r["zpos_diag"] is not None:
            diag = r["zpos_diag"]
            print(f"\nzpos QC ({r['session_id']}): behaviordata vs videodata position after "
                  f"nearest-timestamp match -- median |diff|={diag['median_abs_diff']:.2f}, "
                  f"95th pct={diag['p95_abs_diff']:.2f}, max={diag['max_abs_diff']:.2f} "
                  f"(n={diag['n_compared']}). merge_behavior_video keeps behaviordata's copy; "
                  "large numbers here would mean that choice matters.")
            break

    trial_rows = [r["trial_features"] for r in session_results if r["trial_features"] is not None]
    position_rows = [r["position_binned"] for r in session_results if r["position_binned"] is not None]
    time_course_rows = [r["time_course"] for r in session_results if r["time_course"] is not None]
    pooled_rows = [r["pooled"] for r in session_results if r["pooled"] is not None]
    psth_time_rows = [r["psth_time"] for r in session_results if r["psth_time"] is not None]
    psth_position_rows = [r["psth_position"] for r in session_results if r["psth_position"] is not None]

    trial_features = pd.concat(trial_rows, ignore_index=True) if trial_rows else pd.DataFrame()
    position_binned = pd.concat(position_rows, ignore_index=True) if position_rows else pd.DataFrame()
    time_course = pd.concat(time_course_rows, ignore_index=True) if time_course_rows else pd.DataFrame()
    pooled = pd.concat(pooled_rows, ignore_index=True) if pooled_rows else pd.DataFrame()
    psth_time = pd.concat(psth_time_rows, ignore_index=True) if psth_time_rows else pd.DataFrame()
    psth_position = pd.concat(psth_position_rows, ignore_index=True) if psth_position_rows else pd.DataFrame()

    trial_features.to_pickle(trial_features_cache)
    position_binned.to_pickle(position_binned_cache)
    time_course.to_pickle(time_course_cache)
    pooled.to_pickle(pooled_cache)
    psth_time.to_pickle(psth_time_cache)
    psth_position.to_pickle(psth_position_cache)

if not pooled.empty and "zpos" in pooled.columns:
    zpos_range = pooled["zpos"].max() - pooled["zpos"].min()
    if zpos_range > 300:
        print(f"\nNote: observed zpos range is {zpos_range:,.0f} cm -- this looks like a cumulative "
              "odometer (total distance run), not a coordinate that resets every lap/trial. The "
              "whole-session '2_corridor_position_tuning' plot below uses each session's OWN "
              "fractional position range (frac_position, 0-1) rather than a shared absolute-cm axis "
              "for exactly this reason; see '2b_session_time_course' for the same whole-session view "
              "in elapsed time instead. The trial-aligned PSTH plots (5_/6_) are unaffected -- they use "
              "each trial's own position-at-onset as a reference point regardless of whether zpos "
              "resets or accumulates.")

n_sessions_used = trial_features["session_id"].nunique() if len(trial_features) else 0
n_trial_rows = len(trial_features) // len(MI_WINDOWS) if len(trial_features) else 0
print(f"Trial-level features: ~{n_trial_rows} engaged trials x {len(MI_WINDOWS)} windows "
      f"across {n_sessions_used} sessions")
downsample_note = f"downsampled to ~{DOWNSAMPLE_HZ} Hz/session" if DOWNSAMPLE_HZ is not None else "no downsampling"
print(f"Pooled continuous samples: {len(pooled)} ({downsample_note})")


# --------------------------------------------------------------------- #
# 3. Exploratory plots
# --------------------------------------------------------------------- #
# 3a. Distributions per protocol
plot_vars = [v for v in VARIABLES if v not in ("zpos", *RATE_VARIABLES) and v in pooled.columns]
if plot_vars:
    ncols = min(4, len(plot_vars))
    nrows = int(np.ceil(len(plot_vars) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows), squeeze=False)
    flat_axes = axes.flatten()
    for ax, var in zip(flat_axes, plot_vars):
        sns.violinplot(data=pooled, x="protocol", y=var, order=PROTOCOLS, ax=ax,
                        hue="protocol", palette=PROTOCOL_COLORS, legend=False)
        ax.set_title(VAR_LABELS.get(var, var))
        ax.set_ylabel("")
    for ax in flat_axes[len(plot_vars):]:
        ax.set_visible(False)
    fig.suptitle("Behavior variable distributions (engaged trials)", y=1.01)
    fig.tight_layout()
    save_fig(fig, figdir, "1_variable_distributions")

# 3b. Corridor (position) tuning -- whole session, not trial-locked.
# Uses each session's own FRACTIONAL position range (frac_position,
# 0-1) rather than a shared absolute-cm axis, since sessions can span
# different absolute zpos ranges (especially if zpos accumulates across
# the whole session rather than resetting -- see the diagnostic printed
# above); '2b_session_time_course' below is the same whole-session view
# in elapsed time instead of distance.
corridor_vars = [v for v in VARIABLES if v != "zpos"
                  and not position_binned.empty and f"{v}_mean" in position_binned.columns]
if corridor_vars:
    ncols = min(4, len(corridor_vars))
    nrows = int(np.ceil(len(corridor_vars) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)
    flat_axes = axes.flatten()
    for ax, var in zip(flat_axes, corridor_vars):
        for protocol in PROTOCOLS:
            sub = position_binned[position_binned["protocol"] == protocol]
            if sub.empty:
                continue
            frac_rounded = sub["frac_position"].round(4)
            agg = sub.groupby(frac_rounded)[f"{var}_mean"].mean()
            sem = sub.groupby(frac_rounded)[f"{var}_mean"].sem().fillna(0)
            ax.plot(agg.index, agg.to_numpy(), color=PROTOCOL_COLORS[protocol], label=protocol)
            ax.fill_between(agg.index, (agg - sem).to_numpy(), (agg + sem).to_numpy(),
                             color=PROTOCOL_COLORS[protocol], alpha=0.2)
        ax.set_xlabel("position (fraction of session range)")
        ax.set_ylabel(VAR_LABELS.get(var, var))
    for ax in flat_axes[len(corridor_vars):]:
        ax.set_visible(False)
    flat_axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Behavior variables vs. position, whole session (engaged trials)", y=1.01)
    fig.tight_layout()
    save_fig(fig, figdir, "2_corridor_position_tuning")

# 2b. Same whole-session view, but by elapsed session time instead of
# distance -- meaningful regardless of whether zpos resets or accumulates.
time_vars = [v for v in VARIABLES if v != "zpos"
             and not time_course.empty and f"{v}_mean" in time_course.columns]
if time_vars:
    ncols = min(4, len(time_vars))
    nrows = int(np.ceil(len(time_vars) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)
    flat_axes = axes.flatten()
    for ax, var in zip(flat_axes, time_vars):
        for protocol in PROTOCOLS:
            sub = time_course[time_course["protocol"] == protocol]
            if sub.empty:
                continue
            agg = sub.groupby("bin_center")[f"{var}_mean"].mean()
            sem = sub.groupby("bin_center")[f"{var}_mean"].sem().fillna(0)
            ax.plot(agg.index / 60, agg.to_numpy(), color=PROTOCOL_COLORS[protocol], label=protocol)
            ax.fill_between(agg.index / 60, (agg - sem).to_numpy(), (agg + sem).to_numpy(),
                             color=PROTOCOL_COLORS[protocol], alpha=0.2)
        ax.set_xlabel("elapsed session time (min)")
        ax.set_ylabel(VAR_LABELS.get(var, var))
    for ax in flat_axes[len(time_vars):]:
        ax.set_visible(False)
    flat_axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Behavior variables vs. elapsed session time (engaged trials)", y=1.01)
    fig.tight_layout()
    save_fig(fig, figdir, "2b_session_time_course")

# 3c. Pairwise relationships (linear fit overlaid, by protocol)
pairplot_vars = [v for v in VARIABLES if v not in RATE_VARIABLES and v in pooled.columns]
if len(pairplot_vars) >= 2 and not pooled.empty:
    plot_df = pooled[pairplot_vars + ["protocol"]].dropna()
    if PAIRPLOT_MAX_ROWS is not None and len(plot_df) > PAIRPLOT_MAX_ROWS:
        per_protocol_cap = PAIRPLOT_MAX_ROWS // max(plot_df["protocol"].nunique(), 1)
        plot_df = pd.concat([
            d.sample(min(len(d), per_protocol_cap), random_state=0) for _, d in plot_df.groupby("protocol")
        ], ignore_index=True)
    g = sns.pairplot(plot_df, hue="protocol", palette=PROTOCOL_COLORS, corner=True, kind="reg",
                      plot_kws=dict(scatter_kws=dict(alpha=0.25, s=8), line_kws=dict(linewidth=1.5)),
                      diag_kind="hist")
    title_suffix = "" if PAIRPLOT_MAX_ROWS is None or len(pooled) <= PAIRPLOT_MAX_ROWS else " (subsampled for rendering)"
    g.figure.suptitle(f"Pairwise behavior variables (engaged trials{title_suffix})", y=1.02)
    save_fig(g.figure, figdir, "3_pairwise_relationships")


# --------------------------------------------------------------------- #
# 3d. Trial-aligned ("PSTH"-style) traces, onset = stimulus presentation
#     (t=0 / relative position=0). These are the main plots of interest:
#     one row per variable, one column per grouping (trial outcome
#     always; the z-scored intermediate-stimulus level `signal_psy` too,
#     for DN where a psychometric fit exists -- DP falls back to raw
#     `signal` since it has intermediate levels but no fit; DM has
#     neither, so outcome is its only grouping). The whole-session,
#     non-trial-locked view is the "2_corridor_position_tuning" plot
#     above -- these are the trial-locked complement.
# --------------------------------------------------------------------- #
SIGNAL_GROUP_BINS = 10  # continuous groupings (signal / signal_psy) get binned into this many levels
                         # before averaging -- pooling sessions by *exact* z-scored value (each session
                         # has its own fitted mu/sigma) would otherwise produce dozens of near-unique
                         # levels instead of a handful of representative ones


def _group_order(values, group_col):
    if group_col == "trialOutcome":
        order = ["HIT", "MISS", "FA", "CR"]
        return [v for v in order if v in values]
    return sorted(values)


def plot_trial_aligned(psth_long: pd.DataFrame, x_col: str, x_label: str, prefix: str):
    if psth_long.empty:
        return
    cmap = plt.get_cmap("plasma")

    for protocol in PROTOCOLS:
        sub_protocol = psth_long[psth_long["protocol"] == protocol]
        if sub_protocol.empty:
            continue

        groupings = [("trialOutcome", "by outcome", OUTCOME_COLORS)]
        signal_col = "signal_psy" if protocol == "DN" else ("signal" if protocol == "DP" else None)
        if signal_col is not None and sub_protocol[signal_col].notna().any():
            label = "by z-scored signal (signal_psy)" if signal_col == "signal_psy" else "by signal (%)"
            groupings.append((signal_col, label, None))

        variables = [v for v in PSTH_VARS if v in sub_protocol["variable"].unique()]
        if not variables:
            continue

        fig, axes = plt.subplots(len(variables), len(groupings),
                                  figsize=(5.5 * len(groupings), 3 * len(variables)), squeeze=False)
        for i, var in enumerate(variables):
            var_data = sub_protocol[sub_protocol["variable"] == var]
            for j, (group_col, group_label, palette) in enumerate(groupings):
                ax = axes[i][j]
                gdata_all = var_data.dropna(subset=[group_col]).copy()

                if palette is None:
                    # Continuous grouping (signal / signal_psy): bin into a
                    # handful of representative levels rather than pooling
                    # by exact value across sessions. quantile_bin_edges
                    # (not a naive np.quantile+unique) avoids collapsing to
                    # far fewer bins than requested when the data is
                    # concentrated on a few repeated values.
                    edges = quantile_bin_edges(gdata_all[group_col].to_numpy(dtype=float), SIGNAL_GROUP_BINS)
                    if len(edges) < 2:
                        continue
                    gdata_all["_level"] = pd.cut(gdata_all[group_col], bins=edges, include_lowest=True)
                    level_values = {b: b.mid for b in gdata_all["_level"].cat.categories}
                    levels = sorted(level_values, key=lambda b: b.mid)
                    norm = plt.Normalize(min(level_values.values()), max(level_values.values()))
                    group_key_col = "_level"
                else:
                    levels = _group_order(gdata_all[group_col].unique(), group_col)
                    group_key_col = group_col
                    norm = None

                for level in levels:
                    gdata = gdata_all[gdata_all[group_key_col] == level]
                    if gdata.empty:
                        continue
                    agg = gdata.groupby(x_col)["value"].mean()
                    sem = gdata.groupby(x_col)["value"].sem().fillna(0)
                    if palette is not None:
                        color, label_str = palette[level], str(level)
                    else:
                        color, label_str = cmap(norm(level_values[level])), f"{level_values[level]:.2g}"
                    ax.plot(agg.index, agg.to_numpy(), color=color, label=label_str, linewidth=1.3)
                    ax.fill_between(agg.index, (agg - sem).to_numpy(), (agg + sem).to_numpy(),
                                     color=color, alpha=0.15)

                ax.axvline(0, color="black", linestyle=":", linewidth=1)
                if i == 0:
                    ax.set_title(group_label)
                if j == 0:
                    ax.set_ylabel(VAR_LABELS.get(var, var))
                ax.set_xlabel(x_label)
                ax.legend(fontsize=6, frameon=False, ncol=2 if len(levels) > 6 else 1)

        fig.suptitle(f"Trial-aligned {prefix.split('_')[-1]} -- {protocol} (0 = stimulus onset)", y=1.02)
        fig.tight_layout()
        save_fig(fig, figdir, f"{prefix}_{protocol}")


plot_trial_aligned(psth_time, "t_rel", "time rel. to stimulus onset (s)", "5_psth_time")
plot_trial_aligned(psth_position, "s_rel", "position rel. to stimulus onset (cm)", "6_psth_position")


# --------------------------------------------------------------------- #
# 4. Mutual information vs. linear regression, per pair, per protocol,
#    per window (pre_stim/stim/reward -- see MI_WINDOWS above).
#    "All interesting combinations" = every pair among the variables
#    that actually have per-trial features (VARIABLES intersected with
#    trial_features' columns) -- zpos and lick (as a rate) included.
#    Parallelized over (protocol, window) groups -- each task computes
#    every pair for that group sequentially.
# --------------------------------------------------------------------- #
mi_vars = [v for v in VARIABLES if v in trial_features.columns]
pairs = list(itertools.combinations(mi_vars, 2))

mi_groups = []
for protocol in PROTOCOLS:
    for window_name in MI_WINDOWS:
        sub = (trial_features[(trial_features["protocol"] == protocol)
                               & (trial_features["window"] == window_name)]
               if len(trial_features) else pd.DataFrame())
        if not sub.empty:
            mi_groups.append((protocol, window_name, sub))


def compute_pairwise_mi_for_group(protocol, window_name, sub, pairs, ksg_k, n_shuffles, seed):
    rng_local = np.random.default_rng(seed)
    records = []
    for var_x, var_y in pairs:
        if var_x not in sub.columns or var_y not in sub.columns:
            continue
        pair_data = sub[[var_x, var_y]].dropna()
        if len(pair_data) < 10:
            continue

        mi = mutual_information_shuffle_ksg(pair_data[var_x], pair_data[var_y], k=ksg_k,
                                             n_shuffles=n_shuffles, rng=rng_local)
        lin = linear_fit(pair_data[var_x], pair_data[var_y])
        cmp = compare_mi_to_linear(mi.bits_corrected, lin.r2)

        records.append({
            "protocol": protocol, "window": window_name, "var_x": var_x, "var_y": var_y,
            "n_trials": mi.n, "r": lin.r, "r2": lin.r2, "p_linear": lin.p_value,
            "mi_bits_raw": mi.bits_raw, "mi_bits_shuffle_mean": mi.bits_shuffle_mean,
            "mi_bits_corrected": mi.bits_corrected, "mi_p_value": mi.p_value,
            "mi_bits_from_r2": cmp.mi_from_r2_bits, "mi_reference_bits": cmp.mi_reference_bits,
            "frac_explained_linearly": cmp.frac_explained_linearly,
            "linear_sufficient": (cmp.frac_explained_linearly >= LINEAR_SUFFICIENT_FRAC)
            if not np.isnan(cmp.frac_explained_linearly) else None,
        })
    return records


print(f"\nComputing pairwise MI/linear fits for {len(mi_groups)} (protocol, window) groups "
      f"x {len(pairs)} pairs (n_jobs={N_JOBS_MI}) ...")
nested_pairwise = Parallel(n_jobs=N_JOBS_MI, backend="loky", verbose=JOBLIB_VERBOSE)(
    delayed(compute_pairwise_mi_for_group)(protocol, window_name, sub, pairs, KSG_K, N_SHUFFLES, seed=i)
    for i, (protocol, window_name, sub) in enumerate(mi_groups)
)
records = [rec for group in nested_pairwise for rec in group]

mi_table = pd.DataFrame(records)
mi_table.to_csv(paths.out / "behavior_mi_vs_linear.csv", index=False)

if len(mi_table):
    n_insufficient = (mi_table["linear_sufficient"] == False).sum()  # noqa: E712 (None must not match)
    print(f"\n{n_insufficient}/{len(mi_table)} (protocol, window, pair) combinations show a linear fit "
          f"missing >{100 * (1 - LINEAR_SUFFICIENT_FRAC):.0f}% of the (bias-corrected) mutual information:")
    if n_insufficient:
        print(mi_table.loc[mi_table["linear_sufficient"] == False,  # noqa: E712
                            ["protocol", "window", "var_x", "var_y", "r2", "mi_bits_corrected", "mi_bits_from_r2"]]
              .to_string(index=False))


# --------------------------------------------------------------------- #
# 4b. MULTIVARIATE regression + joint MI: not just pairwise -- for each
#    variable (target), fit it from ALL OTHER variables at once
#    (leave-one-out), and compute the JOINT mutual information between
#    that whole predictor SET and the target (mutual_information_ksg
#    accepts a multivariate X, see info_theory.py). This is the
#    "how well is speed explained by everything else together, not just
#    one thing at a time" question -- a multivariate model can fit much
#    better than any single pairwise one even when no individual pair is
#    that strong (e.g. speed might be only weakly related to pupil area
#    alone AND weakly to lick rate alone, but strongly predictable from
#    the two TOGETHER).
# --------------------------------------------------------------------- #
def compute_multivariate_for_group(protocol, window_name, sub, mi_vars, ksg_k, n_shuffles, seed):
    rng_local = np.random.default_rng(seed)
    available = [v for v in mi_vars if v in sub.columns]
    records = []
    for target in available:
        predictor_names = [v for v in available if v != target]
        if len(predictor_names) < 2:
            continue  # "multivariate" needs at least 2 predictors; otherwise it's just the pairwise case above
        data = sub[[target] + predictor_names].dropna()
        if len(data) < len(predictor_names) + 10:
            continue

        X = data[predictor_names].to_numpy()
        y = data[target].to_numpy()
        lin = multivariate_linear_fit(X, y, predictor_names=predictor_names)
        mi = mutual_information_shuffle_ksg(X, y, k=ksg_k, n_shuffles=n_shuffles, rng=rng_local)
        cmp = compare_mi_to_linear(mi.bits_corrected, lin.r2)

        records.append({
            "protocol": protocol, "window": window_name, "target": target,
            "predictors": ", ".join(predictor_names), "n_predictors": len(predictor_names),
            "n_trials": lin.n, "r2": lin.r2, "adj_r2": lin.adj_r2, "p_linear": lin.p_value,
            "mi_bits_raw": mi.bits_raw, "mi_bits_shuffle_mean": mi.bits_shuffle_mean,
            "mi_bits_corrected": mi.bits_corrected, "mi_p_value": mi.p_value,
            "mi_bits_from_r2": cmp.mi_from_r2_bits, "mi_reference_bits": cmp.mi_reference_bits,
            "frac_explained_linearly": cmp.frac_explained_linearly,
            "linear_sufficient": (cmp.frac_explained_linearly >= LINEAR_SUFFICIENT_FRAC)
            if not np.isnan(cmp.frac_explained_linearly) else None,
        })
    return records


print(f"\nComputing multivariate (leave-one-out) MI/linear fits for {len(mi_groups)} "
      f"(protocol, window) groups x {len(mi_vars)} targets (n_jobs={N_JOBS_MI}) ...")
nested_multivariate = Parallel(n_jobs=N_JOBS_MI, backend="loky", verbose=JOBLIB_VERBOSE)(
    delayed(compute_multivariate_for_group)(protocol, window_name, sub, mi_vars, KSG_K, N_SHUFFLES, seed=1000 + i)
    for i, (protocol, window_name, sub) in enumerate(mi_groups)
)
multivariate_records = [rec for group in nested_multivariate for rec in group]

multivariate_table = pd.DataFrame(multivariate_records)
multivariate_table.to_csv(paths.out / "behavior_multivariate_mi_vs_linear.csv", index=False)

if len(multivariate_table):
    top = multivariate_table.sort_values("r2", ascending=False).head(10)
    print("\nStrongest multivariate fits (target explained by all other variables jointly, top 10 by r2):")
    print(top[["protocol", "window", "target", "predictors", "r2", "adj_r2", "mi_bits_corrected"]]
          .to_string(index=False))


# --------------------------------------------------------------------- #
# 4c. FORWARD STEPWISE selection: the leave-one-out R^2 above answers
#    "how well is this target explained by everything else combined",
#    but can't tell "many variables genuinely add independent
#    information" apart from "one dominant predictor does nearly all
#    the work, the rest just ride along on their correlation with it"
#    -- e.g. two variables that are themselves strongly correlated with
#    each other (like a pupil x/y tracking pair) can make a combined
#    fit look like broad multivariate structure when it's really one
#    relationship. Forward-selects predictors one at a time (whichever
#    gives the biggest R^2 gain at each step -- see
#    info_theory.forward_stepwise_selection), so step 1's R^2 IS the
#    single best predictor's pairwise R^2, and how much the LATER steps
#    still add tells you whether the rest of the predictor set is
#    actually contributing anything beyond that first variable.
# --------------------------------------------------------------------- #
def compute_stepwise_for_group(protocol, window_name, sub, mi_vars, seed):
    available = [v for v in mi_vars if v in sub.columns]
    records = []
    for target in available:
        predictor_names = [v for v in available if v != target]
        if len(predictor_names) < 2:
            continue
        data = sub[[target] + predictor_names].dropna()
        if len(data) < len(predictor_names) + 10:
            continue

        X = data[predictor_names].to_numpy()
        y = data[target].to_numpy()
        steps = forward_stepwise_selection(X, y, predictor_names)
        final_r2 = steps[-1].cumulative_r2 if steps else np.nan
        for s in steps:
            records.append({
                "protocol": protocol, "window": window_name, "target": target,
                "step": s.step, "predictor_added": s.predictor_added,
                "cumulative_r2": s.cumulative_r2, "cumulative_adj_r2": s.cumulative_adj_r2,
                "delta_r2": s.delta_r2,
                "frac_of_full_r2": (s.cumulative_r2 / final_r2) if final_r2 else np.nan,
            })
    return records


print(f"\nComputing forward-stepwise selection for {len(mi_groups)} (protocol, window) groups "
      f"x {len(mi_vars)} targets (n_jobs={N_JOBS_MI}) ...")
nested_stepwise = Parallel(n_jobs=N_JOBS_MI, backend="loky", verbose=JOBLIB_VERBOSE)(
    delayed(compute_stepwise_for_group)(protocol, window_name, sub, mi_vars, seed=2000 + i)
    for i, (protocol, window_name, sub) in enumerate(mi_groups)
)
stepwise_records = [rec for group in nested_stepwise for rec in group]
stepwise_table = pd.DataFrame(stepwise_records)
stepwise_table.to_csv(paths.out / "behavior_multivariate_stepwise.csv", index=False)

if len(stepwise_table):
    first_steps = stepwise_table[stepwise_table["step"] == 1].sort_values("frac_of_full_r2", ascending=False)
    print("\nHow much of each target's full multivariate R^2 comes from its single best predictor alone "
          "(top 10 -- close to 1.0 means the rest of the predictor set adds little beyond that one variable):")
    print(first_steps[["protocol", "window", "target", "predictor_added", "cumulative_r2", "frac_of_full_r2"]]
          .head(10).to_string(index=False))


# --------------------------------------------------------------------- #
# 5. Figure: MI (best estimate) vs. MI-implied-by-r2, one point per
#    (protocol, pair) -- one figure PER WINDOW (pre_stim/stim/response),
#    since a pair's relationship can differ a lot between them (e.g.
#    lick-pupil_area is probably much stronger in the reward window
#    than pre-stimulus). Within each figure: marker SHAPE encodes var_x
#    (the first variable of the pair), color encodes var_y (the second)
#    -- two small legends instead of one legend entry per pair, so it
#    stays readable even with dozens of pairs.
# --------------------------------------------------------------------- #
if len(mi_table) and np.isfinite(mi_table["mi_reference_bits"]).any():
    all_vars_in_pairs = sorted(set(v for pair in pairs for v in pair))
    marker_shapes = ["o", "s", "^", "v", "D", "P", "X", "*", "h", "p", "<", ">"]
    var_marker = {v: marker_shapes[i % len(marker_shapes)] for i, v in enumerate(all_vars_in_pairs)}
    color_palette = list(plt.cm.tab10(np.linspace(0, 1, 10))) + list(plt.cm.Set3(np.linspace(0, 1, 12)))
    var_color = {v: color_palette[i % len(color_palette)] for i, v in enumerate(all_vars_in_pairs)}

    finite_all = mi_table.replace([np.inf, -np.inf], np.nan).dropna(subset=["mi_reference_bits", "mi_bits_from_r2"])
    # Shared axis limits across all three window figures, so the same
    # pair's dot at the same position is directly comparable window to window.
    lims = [0, finite_all["mi_reference_bits"].max() * 1.1 + 1e-6] if len(finite_all) else [0, 1]

    for window_name in MI_WINDOWS:
        finite = finite_all[finite_all["window"] == window_name]
        if finite.empty:
            continue
        protocols_present = [p for p in PROTOCOLS if p in finite["protocol"].unique()]
        fig, axes = plt.subplots(1, len(protocols_present), figsize=(4.3 * len(protocols_present), 4.3), squeeze=False)
        axes = axes[0]
        for ax, protocol in zip(axes, protocols_present):
            sub = finite[finite["protocol"] == protocol]
            for _, row in sub.iterrows():
                ax.scatter(row["mi_reference_bits"], row["mi_bits_from_r2"],
                           marker=var_marker[row["var_x"]], color=var_color[row["var_y"]],
                           s=70, edgecolor="black", linewidth=0.4)
            ax.plot(lims, lims, color="grey", linestyle="--", linewidth=1)
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            ax.set_title(protocol)
            ax.set_xlabel("MI, best available estimate (bits)")
        axes[0].set_ylabel("MI implied by linear r\u00b2 (bits)")
        fig.suptitle(f"Is the linear fit capturing the full relationship? -- {window_name} window\n"
                     "(shape = 1st variable, color = 2nd variable; dashed = linear matches MI exactly)", y=1.08)

        shape_handles = [plt.Line2D([0], [0], marker=var_marker[v], linestyle="none", markerfacecolor="white",
                                     markeredgecolor="black", markersize=8, label=v) for v in all_vars_in_pairs]
        color_handles = [plt.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=var_color[v],
                                     markeredgecolor="black", markersize=8, label=v) for v in all_vars_in_pairs]
        leg1 = fig.legend(handles=shape_handles, loc="center left", bbox_to_anchor=(1.0, 0.7),
                           fontsize=7, title="shape = 1st var", frameon=False)
        fig.add_artist(leg1)
        fig.legend(handles=color_handles, loc="center left", bbox_to_anchor=(1.0, 0.3),
                   fontsize=7, title="color = 2nd var", frameon=False)
        fig.tight_layout()
        save_fig(fig, figdir, f"4_mi_vs_linear_summary_{window_name}")

    # 5a. Same comparison, but as the FRACTION explained linearly
    # directly (frac_explained_linearly = mi_from_r2 / max(mi_corrected,
    # mi_from_r2), already in [0,1] by construction) rather than reading
    # it off the scatter's distance from the diagonal. One horizontal
    # bar per pair, sorted, one panel per protocol, one figure per window.
    for window_name in MI_WINDOWS:
        finite = finite_all[finite_all["window"] == window_name].dropna(subset=["frac_explained_linearly"])
        if finite.empty:
            continue
        finite = finite.assign(pair_label=finite["var_x"] + " - " + finite["var_y"])
        protocols_present = [p for p in PROTOCOLS if p in finite["protocol"].unique()]
        n_pairs_max = finite.groupby("protocol").size().max()
        fig, axes = plt.subplots(1, len(protocols_present),
                                  figsize=(4.8 * len(protocols_present), max(4, 0.22 * n_pairs_max)),
                                  squeeze=False)
        for ax, protocol in zip(axes[0], protocols_present):
            sub = finite[finite["protocol"] == protocol].sort_values("frac_explained_linearly")
            ax.barh(sub["pair_label"], sub["frac_explained_linearly"], color=PROTOCOL_COLORS[protocol],
                    edgecolor="black", linewidth=0.4)
            ax.axvline(1.0, color="grey", linestyle="--", linewidth=1)
            ax.set_xlim(0, max(1.05, sub["frac_explained_linearly"].max() * 1.05))
            ax.set_xlabel("MI(linear) / MI(full)")
            ax.set_title(protocol)
            ax.tick_params(axis="y", labelsize=6)
        fig.suptitle(f"Fraction of mutual information a linear fit captures, per pair -- {window_name} window",
                     y=1.005)
        fig.tight_layout()
        save_fig(fig, figdir, f"4a_frac_explained_linearly_{window_name}")


# --------------------------------------------------------------------- #
# 5b. Figure: same MI-vs-linear comparison as above, but for the
#    MULTIVARIATE (leave-one-out) fits from section 4b -- one point per
#    (protocol, target), color-coded by target, one figure per window.
#    A point far below the diagonal here means "this variable's
#    relationship to everything else jointly has a real nonlinear
#    component a multivariate LINEAR model is missing" -- the
#    multivariate analogue of the pairwise plot above.
# --------------------------------------------------------------------- #
if len(multivariate_table) and np.isfinite(multivariate_table["mi_reference_bits"]).any():
    target_color = {v: color_palette[i % len(color_palette)] for i, v in enumerate(mi_vars)}
    finite_mv_all = multivariate_table.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["mi_reference_bits", "mi_bits_from_r2"])
    lims_mv = [0, finite_mv_all["mi_reference_bits"].max() * 1.1 + 1e-6] if len(finite_mv_all) else [0, 1]

    for window_name in MI_WINDOWS:
        finite_mv = finite_mv_all[finite_mv_all["window"] == window_name]
        if finite_mv.empty:
            continue
        protocols_present = [p for p in PROTOCOLS if p in finite_mv["protocol"].unique()]
        fig, axes = plt.subplots(1, len(protocols_present), figsize=(4.3 * len(protocols_present), 4.3), squeeze=False)
        axes = axes[0]
        handles = {}
        for ax, protocol in zip(axes, protocols_present):
            sub = finite_mv[finite_mv["protocol"] == protocol]
            for _, row in sub.iterrows():
                h = ax.scatter(row["mi_reference_bits"], row["mi_bits_from_r2"],
                                color=target_color[row["target"]], s=90, edgecolor="black", linewidth=0.5)
                handles.setdefault(row["target"], h)
            ax.plot(lims_mv, lims_mv, color="grey", linestyle="--", linewidth=1)
            ax.set_xlim(lims_mv)
            ax.set_ylim(lims_mv)
            ax.set_title(protocol)
            ax.set_xlabel("joint MI, best available estimate (bits)")
        axes[0].set_ylabel("MI implied by multivariate r\u00b2 (bits)")
        fig.suptitle(f"Multivariate: does a linear model of ALL other variables capture the full "
                     f"relationship? -- {window_name} window\n(color = target variable, predicted from "
                     "every other variable jointly; dashed = linear matches MI exactly)", y=1.1)
        fig.legend(handles.values(), handles.keys(), loc="center left", bbox_to_anchor=(1.0, 0.5),
                   fontsize=7, title="target", frameon=False)
        fig.tight_layout()
        save_fig(fig, figdir, f"7_multivariate_mi_vs_linear_summary_{window_name}")

    # 7a. Same, as a fraction: one bar per target, sorted, per protocol,
    # per window.
    for window_name in MI_WINDOWS:
        finite_mv = finite_mv_all[finite_mv_all["window"] == window_name].dropna(subset=["frac_explained_linearly"])
        if finite_mv.empty:
            continue
        protocols_present = [p for p in PROTOCOLS if p in finite_mv["protocol"].unique()]
        fig, axes = plt.subplots(1, len(protocols_present), figsize=(4.5 * len(protocols_present), 4), squeeze=False)
        for ax, protocol in zip(axes[0], protocols_present):
            sub = finite_mv[finite_mv["protocol"] == protocol].sort_values("frac_explained_linearly")
            colors = [target_color[t] for t in sub["target"]]
            ax.barh(sub["target"], sub["frac_explained_linearly"], color=colors, edgecolor="black", linewidth=0.5)
            ax.axvline(1.0, color="grey", linestyle="--", linewidth=1)
            ax.set_xlim(0, max(1.05, sub["frac_explained_linearly"].max() * 1.05))
            ax.set_xlabel("MI(linear) / MI(full)")
            ax.set_title(protocol)
            ax.tick_params(axis="y", labelsize=7)
        fig.suptitle(f"Multivariate: fraction of joint MI a linear fit captures, per target -- "
                     f"{window_name} window", y=1.02)
        fig.tight_layout()
        save_fig(fig, figdir, f"7a_multivariate_frac_explained_linearly_{window_name}")


# --------------------------------------------------------------------- #
# 8. Figure: how much of each target's full multivariate R^2 comes from
#    its single best predictor alone (frac_of_full_r2 at step 1 of the
#    forward-stepwise selection, section 4c) -- close to 1.0 means the
#    "multivariate" fit in sections 4b/7 for that target is really just
#    one dominant pairwise relationship wearing a multivariate costume;
#    much less than 1.0 means multiple predictors are genuinely adding
#    independent information. One bar per target, sorted, per protocol,
#    per window -- directly comparable to the 7a fraction-explained-
#    linearly plots, but asking a different question (breadth of the
#    predictor set that matters, not linearity of the relationship).
# --------------------------------------------------------------------- #
if len(stepwise_table):
    first_steps_all = stepwise_table[stepwise_table["step"] == 1].dropna(subset=["frac_of_full_r2"])
    for window_name in MI_WINDOWS:
        finite_step = first_steps_all[first_steps_all["window"] == window_name]
        if finite_step.empty:
            continue
        protocols_present = [p for p in PROTOCOLS if p in finite_step["protocol"].unique()]
        fig, axes = plt.subplots(1, len(protocols_present), figsize=(4.5 * len(protocols_present), 4), squeeze=False)
        for ax, protocol in zip(axes[0], protocols_present):
            sub = finite_step[finite_step["protocol"] == protocol].sort_values("frac_of_full_r2")
            colors = [target_color[t] for t in sub["target"]]
            labels = [f"{t} (+{p})" for t, p in zip(sub["target"], sub["predictor_added"])]
            ax.barh(labels, sub["frac_of_full_r2"], color=colors, edgecolor="black", linewidth=0.5)
            ax.axvline(1.0, color="grey", linestyle="--", linewidth=1)
            ax.set_xlim(0, 1.05)
            ax.set_xlabel("R\u00b2(best single predictor) / R\u00b2(all predictors)")
            ax.set_title(protocol)
            ax.tick_params(axis="y", labelsize=6)
        fig.suptitle(f"Is the multivariate fit real synergy, or one dominant predictor? -- {window_name} "
                     "window\n(label = target (+its single best predictor); close to 1.0 = the rest add "
                     "little beyond that one variable)", y=1.06)
        fig.tight_layout()
        save_fig(fig, figdir, f"8_stepwise_dominant_predictor_{window_name}")

print(f"\nWrote tables to {paths.out}, figures to {figdir}")
print("Run `python b_progress/make_progress_md.py 1c_behavior` to build a markdown summary.")
