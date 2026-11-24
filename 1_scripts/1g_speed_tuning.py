# -*- coding: utf-8 -*-
"""
1g_speed_tuning
===============
RUNNING SPEED against every other behavioral variable, in two views that replace the scatter plots of 1c.
TRIALS: only the PROTOCOLS_USED (DN) sessions and, with NOISE_TRIALS_ONLY = True, only the NOISE trials (0 < signal < 100 %) of the
engaged trials: the frames of the other trials are left out of every figure and table (both settings are at the top).

  A. TUNING CURVES (figures 1, 2, 3; MIRRORED by 1f for pupil area -- same y variables, bins, levels and figures, code in
     infotheory/tuning_common.py).  Speed is cut into QUANTILE bins within each session (the same logic as the pupil quantiles of
     1f): bin 0 = stationary (speed <= STOP_SPEED; running speed is zero-inflated, so the stops get a bin of their own instead of
     being smeared over several "equal-occupancy" bins), bins 1..N-1 = equal-occupancy quantiles of the MOVING samples. For
     every other variable (pupil area, pupil x / y, motion energy, lick rate, video PCs) the mean in each speed bin is drawn against
     the mean speed of the bin (x axis in cm/s, so a saturating or inverted-U curve looks like one).
         figure 1  whole trial, POOLED FRAMES (every time sample of the selected trials)   1_speed_tuning_whole_trial
         figure 2  per-trial window means (pre_stim / stim / reward, as 1c), plus P(lick)  2_speed_tuning_by_window  (y shared per column)
         figure 3  whole trial as SECTION means (non-overlapping SECTION_WIDTH cm sections of SECTION_RANGE; one sample per trial and
                   section): the independent-sample version of figure 1, to use whenever a test is needed (frames are autocorrelated;
                   they do not bias a mean curve but their p-values would be meaningless)                3_speed_tuning_sections
     Each curve is the mean over analysis units (--aggregate) of the unit's bin means, with SEM over units; pooled = one curve per
     protocol (sample-weighted over every session), no error bars (project convention). The Spearman rho between speed and the
     variable (mean over units) is written in each panel of figure 1.

     Each of figures 1-3 is also written with the QUANTILE (bin number; bin 0 = "stop") on the x axis instead of the mean speed of the bin
     (1q_, 2q_, 3q_ ..._by_quantile): equal spacing, no dependence on the speed values.
     Each of figures 1-3 is ALSO written with FIXED, EQUAL-WIDTH speed bins (1w_, 2w_, 3w_ ..._equal_width; EQUALWIDTH_RANGE, same cm/s
     range in every session, after the same stationary bin) instead of the default within-session quantiles: equipopulated (quantile)
     bins give every bin the same statistical power regardless of how skewed running speed is, but distort the x-axis spacing where the
     speed distribution isn't uniform; equal-width bins keep the real physical spacing but can leave a sparse bin (e.g. high speeds)
     noisy. Neither replaces the other -- both are kept so a curve's shape can be checked against the binning scheme that produced it.

  A2. PERFORMANCE by speed quantile (figures 7, 7b; the mirror of 1f's figure 2 / 2b): per window, hit rate, FA rate and balanced accuracy
     on ALL engaged trials; d' and criterion on the NOISE trials only (infotheory/noise_sdt.py: DPRIME_METHOD / CRITERION_METHOD; keep the
     settings equal to 1f's). PERF_BINS speed quantiles within session (no stationary bin: every bin needs enough trials).
         speed_performance_by_quantile_windows[__<aggregate>].csv

  B. DENSITY HEATMAPS (figures 4, 5, 6).  Figure 6 (6_speed_vs_each_variable_<protocol>) is the speed-centred view: running speed on the
     x axis against each other variable (position, lick rate, pupil, motion, video PCs) as one large density heatmap each, with the
     quantile-bin tuning curve drawn on top. Figures 4 / 5 are the full matrix of all pairs (speed is its first column).  The lower-triangle matrix of 1c's pairwise scatter, but every panel is a 2-D density
     (histogram on a shared grid, light Gaussian smoothing, one-hue sequential map; value space: sqrt scale, each panel scaled to its own
     maximum; rank space: linear scale shared by all panels, in units of the uniform density, so 1 = independent) and the diagonal is the marginal histogram. One figure per protocol and per DENSITY_SPACES entry:
         value   raw values (z-scored within session for pupil / motion / video PCs, speed in cm/s) on axes shared by all protocols
                 (DENSITY_RANGE_PCT percentiles of the pooled samples)
         rank    the within-session RANK (0-1) of each variable: independent of the marginal shapes and of session offsets, so the
                 dependence structure itself is shown (the zero-inflated speed becomes a vertical band at its mean rank)
     The density of a unit is normalised to sum 1; for --aggregate session / mouse the panel is the MEAN density of the units
     (every unit weighs the same); pooled = all (capped) samples together.
     Position = pos_rel, the position INSIDE the trial relative to the stimulus onset (zpos - stimStart, cm); the raw zpos is a cumulative
     odometer and is not used (add "zpos" to DENSITY_VARS to see it anyway). Lick rate = lick events smoothed with a LICK_SMOOTH_S boxcar (Hz).
     SAMPLING: the density panels and the Spearman values use FRAMES (every engaged time sample, ~one per frame); the density panels use
     POOLED_MAX_SAMPLES_PER_SESSION evenly spaced frames per session, the Spearman values all of them. Only figure 2 uses trial means.

CROSS-VALIDATION: unlike 1f/2j/2k, nothing here needs it. Every statistic in this script is a bin mean, a
Spearman correlation, or a 2-D density -- a descriptive summary of the samples, with no model fit or
data-dependent selection whose effect size is read off the SAME data it was chosen or fitted on. There is no
R2, eta2, or quadratic coefficient anywhere in 1g to be inflated by in-sample fitting, so there is no in-sample
statistic here that a held-out half could show to be biased (the scoping rule used throughout 1f/2j/2k: CV
is for a fit-and-score-on-the-same-data effect size, not for every number in the pipeline).
Checked against the four CV ideas used elsewhere in the project (fit-on-half/plot-against-other-half,
5-fold held-out loss, averaged curve across folds, odd-sort/even-display heatmaps): none apply here. Ideas
2/3 (K-fold loss / averaged fit) need a fitted model with a loss to average -- there is none. Idea 4
(sort on one half, display the other) is for PER-CELL/PER-UNIT heatmaps where a data-dependent SORT ORDER can
make noise look structured; 1g's heatmaps (figures 4-6) are pairwise sample densities with no per-row sort to
decontaminate. Idea 1 (fit on one half, overlay the other half's points) is the closest fit -- a bin-mean curve
IS a fit of sorts -- but it tests the same thing split_half_r / eta2_cv test elsewhere: does the curve
replicate on independent trials. Here that is already covered by the project's split-half machinery wherever
a bin-mean curve is tested (2j/2k's h1/h2 and split_half_r); 1g's curves are shown for description, not tested,
so adding an odd/even overlay would duplicate that reliability check without adding a new one (and would need
reworking infotheory/tuning_common.py, shared with 1f's own tuning curves, for a check this script does not use
numerically anywhere).

Project-wide rules (infotheory/project_config.py)
    * selection : --animals / --sessions restrict the sessions (always intersected with 1b's included list); a restricted run
                  writes to out/selection/<tag>/ and never overwrites the full-dataset out/.
    * aggregate : --aggregate pooled|session|mouse = the unit of analysis of every figure and table.
    * memory    : one session in RAM per worker (--n-jobs); the per-session cache keeps only bin sums, Spearman values and a
                  capped sample (POOLED_MAX_SAMPLES_PER_SESSION rows) -- the whole dataset is never loaded at once.
    * z-scoring : only the face quantities are z-scored (within session); running speed is left in cm/s.

Reads: 2_pipeline/1b_psychometric/out/psychometric_fits.csv (included sessions).
Loads: raw behaviordata / videodata, one session at a time.

Outputs
-------
2_pipeline/1g_speed_tuning/
    out/    speed_tuning_curves[__<aggregate>].csv     one row per (protocol, level, y, speed bin), level = whole_trial / pre_stim / stim / reward /
                                                       sections: mean speed, mean y, SEM over units, n units, n samples/trials
            speed_correlations[__<aggregate>].csv      Spearman rho (mean +- SEM over units) for every pair of DENSITY_VARS
            speed_bin_occupancy[__<aggregate>].csv     fraction of samples in the stationary bin, per unit
            figures/*.png
    out/selection/<tag>/...  the same for a restricted run
    store/  speed_tuning_session_<hash>.pkl           per-session cache (reused unless --recompute)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, Normalize
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata

from infotheory.pipeline import get_pipeline_paths
from infotheory.session import discover_sessions, PROTOCOLS
from infotheory.continuous import merge_behavior_video, restrict_to_trials, remove_pupil_outliers, \
    zscore_per_session, compute_trial_position_window_means
from infotheory.plotting import set_style, save_fig, clear_figures, PROTOCOL_COLORS
from infotheory.scope import add_scope_args, scope_from_args, select_sessions, scoped_dirs, data_stamp, unit_noun
from infotheory.session_jobs import run_sessions
from infotheory.unit_plots import mouse_palette
from infotheory import tuning_common as tc
from infotheory import performance_common as pc

# --- what is analysed (edit freely) ---------------------------------------------------------------------------
X_VAR = "runspeed"
PROTOCOLS_USED = ["DN"]                    # protocols analysed
NOISE_TRIALS_ONLY = True                   # True: only noise trials (0 < signal < 100 %); False: every engaged trial
Y_VARS = [v for v in tc.TUNING_VARS if v != X_VAR]      # the SAME y variables as 1f (everything vs pupil), minus the x variable itself
WINDOW_Y_EXCLUDE = tc.NO_WINDOW_Y          # no per-window version (the position inside a fixed window is its centre)
VAR_LABELS = tc.VAR_LABELS
LICK_SMOOTH_S = 1.0                        # lick events -> lick rate (Hz): boxcar of this width, for the density panels
RATE_VARIABLES = ["lick"]                  # 0/1 event channels: averaged per sample then converted to Hz (whole session) / per-window rate
TUNING_WINDOWS = {"pre_stim": (-30.0, -10.0), "stim": (0.0, 20.0), "reward": (25.0, 45.0)}   # cm relative to stimStart, as 1c / 1f
# the whole trial as SECTION means (independent-ish samples instead of autocorrelated frames; same settings as 1f's POS_RANGE / POS_WINDOW)
SECTION_RANGE = (-60.0, 80.0)
SECTION_WIDTH = 20.0

# speed bins (within session)
N_QUANTILE_BINS = 8                        # total bins, INCLUDING the stationary bin when STOP_BIN
STOP_BIN = True                            # True: bin 0 = stationary, the rest are quantiles of the moving samples
STOP_SPEED = 1.0                           # cm/s: speed <= this counts as stationary
MIN_PER_BIN_SAMPLES = 30                   # whole-session level: samples needed in a bin of a session
MIN_PER_BIN_TRIALS = 5                     # window / section level: trials (samples) needed in a bin of a session
EQUALWIDTH_RANGE = (STOP_SPEED, 60.0)      # cm/s: FIXED range for the equal-width tuning-curve bins (figures 1w-3w), AFTER the
                                            # stationary bin (same STOP_BIN floor as the default quantile scheme); samples above
                                            # the top edge are dropped from this view (edit the upper bound to taste)

# PERFORMANCE by speed quantile (figures 7, 7b; the mirror of 1f's figure 2): hit / FA / balanced accuracy on ALL engaged trials, d' and
# criterion on the NOISE trials only (see infotheory/noise_sdt.py). Keep these equal to the settings of 1f.
PERF_BINS = 6                              # speed quantile bins (within session, among ALL engaged trials; no stationary bin: every bin needs trials)
DPRIME_METHOD = "probit_shift"             # "probit_pred" | "probit_shift" | "split"
# "probit_shift" = the inverse-sigma sensitivity: dprime_probit_shift = b / b0 = (1/sigma of this bin's psychometric fit) /
# (1/sigma of the session's own 1b curve) -- sensitivity relative to the session's own baseline, noise trials only
# (0 < signal < 100 %; see infotheory/noise_sdt.py). Kept equal to 1f's setting.
CRITERION_METHOD = "probit_shift"
MIN_PER_CLASS_BIN = 3                      # 0 % and 100 % trials (hit / FA) or weak / strong trials (split) needed in a bin
MIN_NOISE_PER_BIN = 20                     # noise trials needed in one (session, speed bin) for its d' / criterion
PROBIT_PRIOR_A_SD = 1.0
PROBIT_PRIOR_B_FRAC = 0.75
RATE_CLIP = 0.005

# density heatmaps
DENSITY_VARS = ["runspeed", "pos_rel", "lick_rate", "pupil_area", "pupil_xpos", "pupil_ypos", "motionenergy", "videoPC_0", "videoPC_1"]
DENSITY_SPACES = ["value", "rank"]
DENSITY_BINS = 36
DENSITY_RANGE_PCT = (0.5, 99.5)            # value space: axis range = these percentiles of the pooled samples
DENSITY_SMOOTH_SIGMA = 1.0                 # bins; 0 = no smoothing
DENSITY_CMAP = "Blues"                     # one-hue sequential, light -> dark
RANK_VMAX = 3.0                            # rank space: colour scale 0 .. RANK_VMAX x the uniform density (1 = independent), shared by all panels
DENSITY_GAMMA = 0.5                        # colour scale: density ** gamma (0.5 = sqrt) so the tails stay visible
MIN_SAMPLES_UNIT = 200                     # a unit needs this many samples in a density panel
POOLED_MAX_SAMPLES_PER_SESSION = 5000      # evenly spaced samples kept per session for the density panels (the correlations use ALL samples)
TRIAL_NOTE = "noise trials (0 < signal < 100 %)" if NOISE_TRIALS_ONLY else "all engaged trials"
JOBLIB_VERBOSE = 5

parser = argparse.ArgumentParser()
add_scope_args(parser)
args = parser.parse_args()
scope = scope_from_args(args)
AGG = scope.aggregate
UNIT = unit_noun(AGG)
N_JOBS = scope.n_jobs
AGG_SUFFIX = "" if AGG == "pooled" else f"__{AGG}"
UNIT_KEYS = {"pooled": [], "session": ["animal_id", "session_id"], "mouse": ["animal_id"]}[AGG]

paths = get_pipeline_paths(__file__)
table_dir, figdir = scoped_dirs(paths, scope)
clear_figures(figdir)
set_style()
print(scope.describe())

# --------------------------------------------------------------------- #
# 1. Included sessions (1b out/) intersected with the selection
# --------------------------------------------------------------------- #
fits_path = paths.out_from("1b_psychometric") / "psychometric_fits.csv"
if not fits_path.exists():
    raise SystemExit(f"Run 1b_psychometric.py first -- expected {fits_path}")
fits_df = pd.read_csv(fits_path)
included = fits_df[fits_df["included"]]
all_triples = discover_sessions(PROTOCOLS)
selected_ids = {f"{a}_{d}" for _, a, d in select_sessions(all_triples, scope)}
protocol_of = {f"{a}_{d}": p for p, a, d in all_triples}
included_ids = [sid for sid in included["session_id"].tolist() if sid in selected_ids
                and protocol_of.get(sid) in PROTOCOLS_USED]
if not included_ids:
    raise SystemExit(f"No selected {'/'.join(PROTOCOLS_USED)} session passed 1b_psychometric's inclusion criteria -- nothing to analyze.")
print(f"Using {len(included_ids)} included sessions "
      f"({included[included['session_id'].isin(included_ids)]['protocol'].value_counts().to_dict()})")
triple_of = {f"{a}_{d}": (p, a, d) for p, a, d in all_triples}
mouse_colors = mouse_palette(a for _, a, _ in all_triples)

ALL_VARS = list(dict.fromkeys([X_VAR] + Y_VARS + DENSITY_VARS))


# --------------------------------------------------------------------- #
# 2. Per-session reduction (one session in RAM), cached
# --------------------------------------------------------------------- #
def rank01(v):
    """Within-session rank in (0, 1) (average ranks for ties); NaN stays NaN."""
    v = np.asarray(v, float)
    out = np.full(len(v), np.nan)
    ok = np.isfinite(v)
    if ok.sum():
        out[ok] = (rankdata(v[ok], method="average") - 0.5) / ok.sum()
    return out


def process_session(ses, seed):
    if ses.trialdata is None or ses.behaviordata is None:
        return None
    merged_full = zscore_per_session(remove_pupil_outliers(merge_behavior_video(ses.behaviordata, ses.videodata)))
    if merged_full.empty or X_VAR not in merged_full.columns:
        return None
    dt_full = np.median(np.diff(merged_full["ts"].to_numpy())) if "ts" in merged_full.columns and len(merged_full) > 2 else np.nan
    fs_full = 1.0 / dt_full if np.isfinite(dt_full) and dt_full > 0 else 1.0
    if "lick" in merged_full.columns:                      # smoothed lick rate (Hz) on the whole trace, before the engaged restriction
        k = max(1, int(round(LICK_SMOOTH_S * fs_full)))
        merged_full["lick_rate"] = np.convolve(merged_full["lick"].fillna(0).to_numpy(float), np.ones(k) / k,
                                               mode="same") * fs_full
    td = ses.trialdata
    if "signal" not in td.columns:
        return None
    eng = td[td["engaged"] == 1] if "engaged" in td.columns else td
    if NOISE_TRIALS_ONLY:                                  # noise trials: an intermediate signal, 0 < signal < 100 %
        eng = eng[(eng["signal"] > 0) & (eng["signal"] < 100)]
    if eng.empty:
        return None
    merged = restrict_to_trials(merged_full, td, eng["trialNumber"])
    if merged.empty:
        return None
    if "zpos" in merged.columns and "stimStart" in eng.columns:    # position inside the trial, relative to the stimulus onset (cm)
        ts_m, z_m = merged["ts"].to_numpy(), merged["zpos"].to_numpy(float)
        pos_rel = np.full(len(merged), np.nan)
        for _, tr in eng.iterrows():
            m = (ts_m >= tr["tStart"]) & (ts_m <= tr["tEnd"])
            pos_rel[m] = z_m[m] - tr["stimStart"]
        merged["pos_rel"] = pos_rel
    ys = [v for v in Y_VARS if v in merged.columns and merged[v].notna().any()]
    dvars = [v for v in DENSITY_VARS if v in merged.columns and merged[v].notna().any()]
    rate_vars = [v for v in RATE_VARIABLES if v in merged.columns]
    fs = fs_full

    # tuning table (shared with 1f, infotheory/tuning_common.py): whole trial (pooled frames), the windows, whole trial as sections
    x = merged[X_VAR].to_numpy(dtype=float)
    rows = tc.session_tuning(merged, eng, x_var=X_VAR, y_vars=ys, rate_vars=rate_vars, fs=fs, windows=TUNING_WINDOWS,
                             section_range=SECTION_RANGE, section_width=SECTION_WIDTH, n_bins=N_QUANTILE_BINS,
                             stop_bin=STOP_BIN, stop_value=STOP_SPEED, min_frames=MIN_PER_BIN_SAMPLES, min_trials=MIN_PER_BIN_TRIALS,
                             fixed_range=EQUALWIDTH_RANGE)
    bx = tc.x_bins(x, N_QUANTILE_BINS, STOP_BIN, STOP_SPEED)
    stationary_frac = float(np.mean(bx[bx >= 0] == 0)) if STOP_BIN and (bx >= 0).any() else np.nan

    # Spearman of every pair (ALL engaged samples; samples are autocorrelated -- descriptive only)
    cm = merged[dvars].corr(method="spearman")
    sp = [dict(var_a=a, var_b=b, rho=float(cm.loc[a, b])) for i, a in enumerate(dvars) for b in dvars[i + 1:]]

    # capped sample for the density panels: values + within-session ranks (ranks over ALL samples, then capped)
    smp = merged[dvars].copy()
    for v in dvars:
        smp[f"{v}__rank"] = rank01(merged[v].to_numpy(float))
    if POOLED_MAX_SAMPLES_PER_SESSION and len(smp) > POOLED_MAX_SAMPLES_PER_SESSION:
        smp = smp.iloc[np.linspace(0, len(smp) - 1, POOLED_MAX_SAMPLES_PER_SESSION).astype(int)]
    smp = smp.astype("float32").reset_index(drop=True)
    smp["session_id"], smp["animal_id"], smp["protocol"] = ses.session_id, ses.animal_id, ses.protocol

    tun = rows

    # performance table: ALL engaged trials (hit / FA / balanced accuracy need the 0 % / 100 % trials), the speed of each window
    eng_all = td[td["engaged"] == 1] if "engaged" in td.columns else td
    perf = pd.DataFrame()
    m_all = restrict_to_trials(merged_full, td, eng_all["trialNumber"]) if len(eng_all) else merged_full.iloc[0:0]
    if len(m_all) and "lickResponse" in eng_all.columns:
        perf = eng_all[["trialNumber", "signal", "lickResponse"]].copy()
        for wname, (lo, hi) in TUNING_WINDOWS.items():
            pw = compute_trial_position_window_means(eng_all, m_all, columns=[X_VAR], s_start=lo, s_end=hi)
            if len(pw) and X_VAR in pw.columns:
                perf = perf.merge(pw[["trialNumber", X_VAR]].rename(columns={X_VAR: f"{X_VAR}_{wname}"}), on="trialNumber", how="left")
        perf["session_id"], perf["animal_id"], perf["protocol"] = ses.session_id, ses.animal_id, ses.protocol
    for tbl in (tun, ):
        tbl["session_id"], tbl["animal_id"], tbl["protocol"] = ses.session_id, ses.animal_id, ses.protocol
    spd = pd.DataFrame(sp)
    spd["session_id"], spd["animal_id"], spd["protocol"] = ses.session_id, ses.animal_id, ses.protocol
    return dict(session_id=ses.session_id, animal_id=ses.animal_id, protocol=ses.protocol, tuning=tun, spearman=spd,
                samples=smp, perf=perf, stationary_frac=stationary_frac, n_samples=int(len(merged)))


session_results = run_sessions(
    included_ids, process_session, list(PROTOCOLS), paths.store, "speed_tuning_session",
    key_params=dict(version="1g-v6", perf_windows=True, sections=(SECTION_RANGE, SECTION_WIDTH), protocols=PROTOCOLS_USED, noise_only=NOISE_TRIALS_ONLY, lick_smooth=LICK_SMOOTH_S, x=X_VAR, ys=Y_VARS, dvars=DENSITY_VARS, rate=RATE_VARIABLES, windows=TUNING_WINDOWS, eqw=EQUALWIDTH_RANGE,
                    n_bins=N_QUANTILE_BINS, stop=(STOP_BIN, STOP_SPEED), min_n=(MIN_PER_BIN_SAMPLES, MIN_PER_BIN_TRIALS),
                    cap=POOLED_MAX_SAMPLES_PER_SESSION),
    load_kwargs=dict(load_behaviordata=True, load_videodata=True),
    n_jobs=N_JOBS, recompute=args.recompute, verbose=JOBLIB_VERBOSE,
    extra_key_by_session={sid: data_stamp(*triple_of[sid]) for sid in included_ids}, flatten=False)
session_results = [r for r in session_results if r is not None]
if not session_results:
    raise SystemExit("No selected session had behavior (+ video) data -- nothing to analyze.")


def cat(key):
    parts = [r[key] for r in session_results if r.get(key) is not None and len(r[key])]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


tuning_all, spearman_all, samples = cat("tuning"), cat("spearman"), cat("samples")
perf_trials = cat("perf")
occupancy = pd.DataFrame([dict(session_id=r["session_id"], animal_id=r["animal_id"], protocol=r["protocol"],
                               stationary_fraction=r["stationary_frac"], n_samples=r["n_samples"]) for r in session_results])
protocols_present = [p for p in PROTOCOLS if (occupancy["protocol"] == p).any()]
print(f"{len(session_results)} sessions; stationary fraction (speed <= {STOP_SPEED}): "
      f"{occupancy.groupby('protocol')['stationary_fraction'].mean().round(2).to_dict()}")
occ_keys = ["protocol"] + UNIT_KEYS
occ_tbl = (occupancy.groupby(occ_keys).apply(lambda d: pd.Series(dict(
    stationary_fraction=np.average(d["stationary_fraction"].fillna(0), weights=d["n_samples"]), n_samples=d["n_samples"].sum())),
    include_groups=False).reset_index())
occ_tbl.to_csv(table_dir / f"speed_bin_occupancy{AGG_SUFFIX}.csv", index=False)


# --------------------------------------------------------------------- #
# 3. Tuning curves: bin sums -> unit means -> mean +- SEM over units
# --------------------------------------------------------------------- #
curves = tc.unit_curves(tuning_all, UNIT_KEYS, AGG, "speed") if len(tuning_all) else pd.DataFrame()
curves.to_csv(table_dir / f"speed_tuning_curves{AGG_SUFFIX}.csv", index=False)


def unit_corr(sp):
    s = sp.copy()
    if AGG == "mouse":
        s = s.groupby(["protocol", "animal_id", "var_a", "var_b"])["rho"].mean().reset_index()
    g = s.groupby(["protocol", "var_a", "var_b"])["rho"]
    out = g.agg(rho=("mean"), sem=("sem"), n_units=("size")).reset_index()
    out["sem"] = out["sem"].fillna(0.0)
    return out


corr = unit_corr(spearman_all) if len(spearman_all) else pd.DataFrame()
corr.to_csv(table_dir / f"speed_correlations{AGG_SUFFIX}.csv", index=False)


def rho_lookup(protocol, y):
    if corr.empty:
        return np.nan
    y = "lick_rate" if y == "lick" else y              # the lick tuning is in Hz, its rho is with the smoothed rate
    m = corr[(corr["protocol"] == protocol) & (((corr["var_a"] == X_VAR) & (corr["var_b"] == y))
                                                | ((corr["var_b"] == X_VAR) & (corr["var_a"] == y)))]
    return float(m["rho"].iloc[0]) if len(m) else np.nan


ys_present = [y for y in Y_VARS if len(curves) and (curves["y"] == y).any()]
tc.plot_tuning_figures(
    curves, figdir=figdir, names=("1_speed_tuning_whole_trial", "2_speed_tuning_by_window", "3_speed_tuning_sections"),
    x_col="speed", x_label="speed (cm/s)", x_name="running speed", protocols=protocols_present, agg=AGG, unit=UNIT,
    ys=ys_present, windows=TUNING_WINDOWS, rho_lookup=rho_lookup,
    bins_note=f"Speed bins: stationary (<= {STOP_SPEED:g} cm/s) + {N_QUANTILE_BINS - 1} quantiles of the moving samples, within session",
    trial_note=TRIAL_NOTE, section_range=SECTION_RANGE, section_width=SECTION_WIDTH)
# the same figures with the QUANTILE (bin number) on the x axis instead of the mean speed of the bin
tc.plot_tuning_figures(
    curves, figdir=figdir, names=("1q_speed_tuning_whole_trial_by_quantile", "2q_speed_tuning_by_window_by_quantile",
                                  "3q_speed_tuning_sections_by_quantile"),
    x_col="speed", x_label="speed (cm/s)", x_name="speed", protocols=protocols_present, agg=AGG, unit=UNIT,
    ys=ys_present, windows=TUNING_WINDOWS, rho_lookup=rho_lookup,
    bins_note=f"Speed bins: stationary (<= {STOP_SPEED:g} cm/s) + {N_QUANTILE_BINS - 1} quantiles of the moving samples, within session",
    trial_note=TRIAL_NOTE, section_range=SECTION_RANGE, section_width=SECTION_WIDTH, x_mode="quantile", stop_bin=STOP_BIN)
# the same figures again with FIXED, EQUAL-WIDTH speed bins (same cm/s range in every session, after the stationary bin)
# instead of the default within-session quantiles -- does the curve's shape depend on which binning scheme is used?
# (older figures above are unchanged; this is an additional view, not a replacement)
tc.plot_tuning_figures(
    curves, figdir=figdir, names=("1w_speed_tuning_whole_trial_equal_width", "2w_speed_tuning_by_window_equal_width",
                                  "3w_speed_tuning_sections_equal_width"),
    x_col="speed", x_label="speed (cm/s)", x_name="running speed", protocols=protocols_present, agg=AGG, unit=UNIT,
    ys=ys_present, windows=TUNING_WINDOWS, rho_lookup=rho_lookup,
    bins_note=f"Speed bins: stationary (<= {STOP_SPEED:g} cm/s) + {N_QUANTILE_BINS - 1} FIXED equal-width bins over "
              f"[{EQUALWIDTH_RANGE[0]:g}, {EQUALWIDTH_RANGE[1]:g}) cm/s (same range in every session, unlike the "
              f"within-session quantiles of figures 1-3; speeds >= {EQUALWIDTH_RANGE[1]:g} cm/s are dropped from this view)",
    trial_note=TRIAL_NOTE, section_range=SECTION_RANGE, section_width=SECTION_WIDTH, scheme="equal_width")


# --------------------------------------------------------------------- #
# 3b. Performance by speed quantile (mirror of 1f figure 2): hit / FA / balanced accuracy (all engaged trials), d' / criterion (noise trials)
# --------------------------------------------------------------------- #
FIT_PARAMS = fits_df.set_index("session_id")[["mu", "sigma", "lapse_rate", "guess_rate"]].to_dict("index")
SPLIT_AT = {}
if len(perf_trials):
    _n = perf_trials[(perf_trials["signal"] > 0) & (perf_trials["signal"] < 100)]
    SPLIT_AT = _n.groupby("session_id")["signal"].median().to_dict()


def _speed_bin(v):
    """Speed quantile bin (1..PERF_BINS) of the trials of ONE session (equal occupancy, ties broken in time order)."""
    if len(v) < 4 * PERF_BINS:
        return np.full(len(v), np.nan)
    b = tc.x_bins(v, PERF_BINS, stop_bin=False).astype(float)
    b[b < 0] = np.nan
    return b + 1.0


perf_rows = []
for wname in TUNING_WINDOWS:
    xc = f"{X_VAR}_{wname}"
    if len(perf_trials) and xc in perf_trials.columns:
        perf_rows += [dict(r, window=wname) for r in pc.perf_by_bins(
            perf_trials[perf_trials["protocol"].isin(PROTOCOLS_USED)], xc, _speed_bin, protocols=PROTOCOLS_USED, unit_keys=UNIT_KEYS,
            fit_params=FIT_PARAMS, split_at=SPLIT_AT, dprime_method=DPRIME_METHOD, criterion_method=CRITERION_METHOD,
            min_per_class=MIN_PER_CLASS_BIN, min_noise=MIN_NOISE_PER_BIN, sd_a=PROBIT_PRIOR_A_SD, sd_b_frac=PROBIT_PRIOR_B_FRAC,
            rate_clip=RATE_CLIP, bin_col="speed_bin")]
perf_df = pd.DataFrame(perf_rows)
if len(perf_df):
    perf_df.to_csv(table_dir / f"speed_performance_by_quantile_windows{AGG_SUFFIX}.csv", index=False)
    pc.plot_performance_figures(perf_df, figdir=figdir, names=("7_speed_performance_windows", "7b_speed_dprime_criterion_methods"),
                                windows=TUNING_WINDOWS, protocols=PROTOCOLS_USED, unit_keys=UNIT_KEYS, unit=UNIT, n_bins=PERF_BINS,
                                x_name="speed", dprime_method=DPRIME_METHOD, criterion_method=CRITERION_METHOD, bin_col="speed_bin")
else:
    print("\nPerformance by speed quantile: no session had trials with a speed in the windows.")


# --------------------------------------------------------------------- #
# 4. Density heatmaps (lower-triangle matrix, one figure per protocol and space)
# --------------------------------------------------------------------- #
dvars_present = [v for v in DENSITY_VARS if v in samples.columns and samples[v].notna().any()]
value_edges = {}
for v in dvars_present:
    lo, hi = np.nanpercentile(samples[v].to_numpy(float), DENSITY_RANGE_PCT)
    if not np.isfinite(lo) or hi <= lo:
        lo, hi = float(np.nanmin(samples[v])), float(np.nanmax(samples[v])) + 1e-9
    value_edges[v] = np.linspace(lo, hi, DENSITY_BINS + 1)
rank_edges = np.linspace(0.0, 1.0, DENSITY_BINS + 1)


def _hist2(d, xc, yc, xe, ye):
    H, _, _ = np.histogram2d(d[xc].to_numpy(float), d[yc].to_numpy(float), bins=[xe, ye])
    return H / H.sum() if H.sum() > 0 else None


def density2d(df, xc, yc, xe, ye):
    """Normalised 2-D density; for session / mouse the mean of the units' densities. Returns array [nx, ny] or None."""
    sub = df[UNIT_KEYS + [xc, yc]].dropna()
    if UNIT_KEYS:
        Hs = [h for _, d in sub.groupby(UNIT_KEYS) if len(d) >= MIN_SAMPLES_UNIT
              for h in [_hist2(d, xc, yc, xe, ye)] if h is not None]
        H = np.mean(Hs, axis=0) if Hs else None
    else:
        H = _hist2(sub, xc, yc, xe, ye) if len(sub) >= MIN_SAMPLES_UNIT else None
    if H is not None and DENSITY_SMOOTH_SIGMA > 0:
        H = gaussian_filter(H, DENSITY_SMOOTH_SIGMA)
    return H


def density1d(df, xc, xe):
    sub = df[UNIT_KEYS + [xc]].dropna()
    hs = []
    groups = [d for _, d in sub.groupby(UNIT_KEYS)] if UNIT_KEYS else [sub]
    for d in groups:
        if len(d) >= MIN_SAMPLES_UNIT:
            h, _ = np.histogram(d[xc].to_numpy(float), bins=xe)
            hs.append(h / max(h.sum(), 1))
    return np.mean(hs, axis=0) if hs else None


def density_figure(protocol, space):
    df = samples[samples["protocol"] == protocol]
    n = len(dvars_present)
    if df.empty or n < 2:
        return
    col = (lambda v: v) if space == "value" else (lambda v: f"{v}__rank")
    edges = (lambda v: value_edges[v]) if space == "value" else (lambda v: rank_edges)
    fig, axes = plt.subplots(n, n, figsize=(2.15 * n, 2.15 * n), squeeze=False)
    im = None
    for i, yv in enumerate(dvars_present):            # rows = y
        for j, xv in enumerate(dvars_present):        # columns = x
            ax = axes[i][j]
            if j > i:
                ax.axis("off")
                continue
            xe, ye = edges(xv), edges(yv)
            if i == j:
                h = density1d(df, col(xv), xe)
                if h is not None:
                    ax.stairs(h, xe, color=PROTOCOL_COLORS[protocol], lw=1.3, fill=True, alpha=0.35)
                    ax.stairs(h, xe, color=PROTOCOL_COLORS[protocol], lw=1.3)
                ax.set_yticks([])
                ax.spines["left"].set_visible(False)
                ax.set_xlim(xe[0], xe[-1])
            else:
                H = density2d(df, col(xv), col(yv), xe, ye)
                if H is None:
                    ax.axis("off")
                    continue
                if space == "rank":                                # density relative to uniform: 1 = what independence would give
                    norm, Hp = Normalize(0.0, RANK_VMAX), H * H.size
                else:
                    norm, Hp = PowerNorm(DENSITY_GAMMA, vmin=0.0, vmax=float(H.max())), H
                im = ax.imshow(Hp.T, origin="lower", aspect="auto", extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap=DENSITY_CMAP,
                               norm=norm, interpolation="nearest")
                rho = corr[(corr["protocol"] == protocol) & (((corr["var_a"] == xv) & (corr["var_b"] == yv))
                                                             | ((corr["var_a"] == yv) & (corr["var_b"] == xv)))]["rho"] \
                    if len(corr) else []
                if len(rho):
                    ax.text(0.04, 0.95, f"ρ={float(rho.iloc[0]):+.2f}", transform=ax.transAxes, va="top", fontsize=7,
                            color="#222222", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
            ax.tick_params(labelsize=6, length=2)
            if i == n - 1:
                ax.set_xlabel(VAR_LABELS.get(xv, xv) if space == "value" else f"rank of {VAR_LABELS.get(xv, xv)}", fontsize=7)
            else:
                ax.set_xticklabels([])
            if j == 0 and i > 0:
                ax.set_ylabel(VAR_LABELS.get(yv, yv) if space == "value" else f"rank of {VAR_LABELS.get(yv, yv)}", fontsize=7)
            elif j > 0:
                ax.set_yticklabels([])
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.06, top=0.955, wspace=0.10, hspace=0.10)
    if im is not None:
        cax = fig.add_axes([0.70, 0.86, 0.22, 0.012])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        if space == "rank":
            cb.set_label("density relative to uniform (1 = independent)", fontsize=7)
            cb.set_ticks(range(int(RANK_VMAX) + 1))
            cb.ax.tick_params(labelsize=6)
        else:
            cb.set_label("relative density (panel max = 1, sqrt scale)", fontsize=7)
            cb.set_ticks([])
    what = "values" if space == "value" else "within-session ranks"
    fig.suptitle(f"{protocol}: pairwise density of behavior variables ({what}; frames of the {TRIAL_NOTE}; "
                 f"{'mean density over ' + UNIT + 's' if AGG != 'pooled' else 'pooled samples'}; ρ = Spearman, "
                 f"{'mean over sessions' if AGG != 'mouse' else 'mean over mice'})", y=0.985, fontsize=10)
    save_fig(fig, figdir, f"{4 if space == 'value' else 5}_pairwise_density_{space}_{protocol}")
    plt.close(fig)


def speed_density_figure(protocol, ncols=4):
    """Running speed (x) against every other variable (y): large 2-D density heatmaps (values, axes shared by the protocols) with
    the quantile-bin tuning curve of the whole-session level drawn on top (mean +- SEM over units of the bin means)."""
    df = samples[samples["protocol"] == protocol]
    yvars = [v for v in dvars_present if v != X_VAR]
    if df.empty or X_VAR not in dvars_present or not yvars:
        return
    nrows = int(np.ceil(len(yvars) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.9 * ncols, 3.4 * nrows), squeeze=False)
    xe = value_edges[X_VAR]
    im = None
    for ax, yv in zip(axes.ravel(), yvars):
        ye = value_edges[yv]
        H = density2d(df, X_VAR, yv, xe, ye)
        ax.set_title(VAR_LABELS.get(yv, yv), fontsize=9)
        if H is None:
            ax.axis("off")
            continue
        im = ax.imshow(H.T, origin="lower", aspect="auto", extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap=DENSITY_CMAP,
                       norm=PowerNorm(DENSITY_GAMMA, vmin=0.0, vmax=float(H.max())), interpolation="nearest")
        tname = "lick" if yv == "lick_rate" else yv                      # the tuning table has the same quantity as `lick` (Hz)
        c = curves[(curves["level"] == "whole_trial") & (curves["y"] == tname) & (curves["protocol"] == protocol)].sort_values("bin")
        if len(c):
            ax.errorbar(c["speed"], c["ymean"], yerr=c["sem"] if AGG != "pooled" else None, color=PROTOCOL_COLORS[protocol],
                        marker="o", markersize=4, markeredgecolor="white", markeredgewidth=0.6, lw=1.6, capsize=2,
                        elinewidth=0.9, label="mean per speed quantile bin")
        r_ = rho_lookup(protocol, yv)
        if np.isfinite(r_):
            ax.text(0.04, 0.95, f"\u03c1={r_:+.2f}", transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
        ax.set_xlabel("speed (cm/s)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(xe[0], xe[-1])
        ax.set_ylim(ye[0], ye[-1])
    for ax in axes.ravel()[len(yvars):]:
        ax.axis("off")
    axes[0][0].legend(frameon=True, framealpha=0.8, fontsize=7, loc="lower left")
    fig.suptitle(f"{protocol}: running speed against every other variable (frames of the {TRIAL_NOTE}; density = "
                 f"{'mean over ' + UNIT + 's' if AGG != 'pooled' else 'pooled samples'}; line = mean of the variable in each within-session "
                 f"speed quantile bin; \u03c1 = Spearman)", y=1.0, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    if im is not None:                                  # colourbar added after the layout, under the grid
        cax = fig.add_axes([0.78, 0.015, 0.19, 0.012])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label("relative density (panel max = 1, sqrt scale)", fontsize=7)
        cb.set_ticks([])
    save_fig(fig, figdir, f"6_speed_vs_each_variable_{protocol}")
    plt.close(fig)


for protocol in protocols_present:
    speed_density_figure(protocol)
for space in DENSITY_SPACES:
    for protocol in protocols_present:
        density_figure(protocol, space)

print(f"Done. Tables -> {table_dir}; figures -> {figdir}")
