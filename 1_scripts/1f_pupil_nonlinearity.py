# -*- coding: utf-8 -*-
"""
1f_pupil_nonlinearity
======================
Is the relationship between PUPIL AREA and other variables NON-LINEAR -- specifically an INVERTED U
(Yerkes-Dodson: best performance / most "engaged" behavior at intermediate arousal)? Across DP and DN (PROTOCOLS_USED, edit at the top).

PERFORMANCE vs PUPIL (figure 2, separate from everything above): ALL engaged trials. hit rate = P(lick | signal > 0) over every signal
strength, FA rate = P(lick | signal = 0) (the only no-signal trials), balanced accuracy = (hit + 1 - FA) / 2; d' and criterion are
signal-detection measures and use the 0 % / 100 % trials only. A bin needs >= MIN_PER_CLASS_BIN trials of each class, one subplot per window (D_WINDOWS: pre_stim,
stim, reward), x = the pupil of THAT window (not the whole-trial mean), DPRIME_BINS quantile bins within session.

TRIALS: only NOISE trials, 0 < signal < 100 % (NOISE_TRIALS_ONLY = True, edit at the top). On those trials every lick is a response to
a signal, so `correct`, `lick_signal` and `lick_any` are the same variable (kept once, as P(lick)); `lick_nosignal`, `hit_100`
and the d' curve (which need the 0 % / 100 % trials) are not part of the trial-level tests. Set NOISE_TRIALS_ONLY = False for the previous all-trials analysis.

x  = pre-stimulus pupil area ([-30, -10] cm before stimStart, the same window as 1d), z-scored within each
     session (after blink rejection). Optionally also the PREVIOUS trial's pupil (PUPIL_PREDICTORS).
y  = one trial-level variable at a time (Y_VARIABLES):
         correct        correctness at every signal level (as in 1d)
         lick_signal    P(lick | signal > 0)     ("hit-like")
         lick_nosignal  P(lick | signal = 0)     (false alarms)
         lick_any       P(lick), all trials      (overall response bias)
         hit_100        P(lick | signal = 100)
         runspeed, motionenergy, lick   the pre-stimulus behavioral state in the same window
     Every y is CENTERED WITHIN SESSION before it is fitted/plotted, so session baselines cannot create a
     curvature across sessions when sessions are pooled (pooled / mouse aggregate).
     d' (0% / 100% trials only) is shown as a curve per pupil bin (figure 1) but not tested: it needs many
     trials per bin and there is no trial-level version of it.

An inverted U is NOT a correlation: a perfect inverted U has r ~ 0. Four complementary checks, per unit
(session / mouse / pooled, --aggregate):
  A. QUADRATIC     y ~ 1 + z + z^2 (z standardized within the unit; y standardized too, so coefficients are
                   comparable between variables). beta_quadratic < 0 = inverted U, > 0 = U.
  B. LIND-MEHLUM   the proper "is it really a hump" test: the quadratic coefficient alone can be significant
                   for a curve that is merely saturating. Inverted U requires the fitted slope at the LOW end
                   of the pupil range to be significantly > 0 AND at the HIGH end significantly < 0
                   (one-sided, heteroscedasticity-robust HC3 SEs; intersection-union, so no extra correction),
                   with the turning point inside the range. Range = RANGE_PCT percentiles of z (not the
                   extremes, which are leverage points).
  C. TWO HALVES    linear slope in the lower and in the upper half of the session's pupil (split at the
                   within-session median -- fixed in advance, so no breakpoint hunting). Inverted U =
                   slope_low > 0 and slope_high < 0. Shown as paired lines per unit (figure 3).
  D. MODEL-FREE    bias-corrected histogram MI(pupil; y) with the SHARED project bins (RESPONSE_BINS) versus
                   the MI implied by the LINEAR r^2: the excess is information a straight line cannot see.
                   (Same estimator as 1d / 2c / 3a-3c, so the numbers are comparable before bias correction.)
  + CV gain        5-fold cross-validated gain in squared error of the quadratic over the linear model,
                   1 - MSE_quad / MSE_lin (> 0 = the curvature generalizes to held-out trials).
Group level (needs >= MIN_UNITS units): Wilcoxon signed-rank across units for beta_quadratic != 0, slope_low
> 0, slope_high < 0, CV gain > 0 and MI excess > 0, Holm-corrected across the y variables within a protocol
(and x). Pooled = a single unit per protocol, so only the per-unit (trial-level) tests are available there.

Project-wide rules (infotheory/project_config.py)
    * selection : --animals / --sessions restrict the sessions; a restricted run writes to
                  out/selection/<tag>/ and NEVER overwrites the full-dataset out/.
    * aggregate : --aggregate pooled|session|mouse = the unit of analysis (default session).
    * memory    : each session is loaded, reduced to its small trial table, cached in store/ and freed.
    * binning   : the histogram MI uses the shared RESPONSE_BINS and trial filters.

PUPIL POSITION: the eye position (pupil_xpos, pupil_ypos) is analysed next to pupil area -- as x (figures 1c / 1d, 3c / 3d, ...) and as y
against pupil area; pupil area is also a y against the position.
BEHAVIORAL PCA (section 9, figures 9a-9c, pupil_pca_*.csv): PCA of runspeed, motion energy, lick rate and pupil position (pre-stimulus
window, z-scored within session; PCA_VARIABLES), pupil AREA excluded; then each component's score is related to pupil area
(correlation, quadratic fit, group test across units).

POSITION IN THE CORRIDOR (section 10, figures 10a-10d, pupil_position_*.csv): the pupil-vs-y relation (quadratic coefficient, linear r,
inverted-U fraction; POS_Y) and the performance measures, recomputed in sliding POS_WINDOW cm windows along the corridor
(POS_RANGE, step POS_STEP, cm from the stimulus onset). In each window x = the pupil of THAT window and y is measured in the same window.
Windows outside TRUSTED_POSITION_RANGE are shaded; a window with fewer than MIN_UNITS units is not drawn.

Reads: 2_pipeline/1b_psychometric/out/psychometric_fits.csv (included sessions).
Loads: raw behaviordata/videodata, one session at a time.

Outputs
-------
2_pipeline/1f_pupil_nonlinearity/
    out/    pupil_nonlinearity_per_unit[__<aggregate>].csv   one row per (protocol, [unit], x, y): n, quadratic /
                         linear coefficients, p-values, slopes at the range ends, turning point, inverted-U flags,
                         two-halves slopes, CV gain, MI / MI-from-r2 / excess
            pupil_nonlinearity_group_tests[__<aggregate>].csv  Wilcoxon tests across units, Holm-corrected
            pupil_nonlinearity_curves[__<aggregate>].csv      mean +- SEM of each y per pupil quantile bin
            pupil_performance_by_quantile_windows[__<aggregate>].csv   hit / FA rate, balanced accuracy (all trials), d', criterion (0 % / 100 % trials) per pupil quantile, per window
            figures/*.png
    out/selection/<tag>/...  the same for a restricted run
    store/  pupil_trials_<hash>.pkl     per-session trial table (reused unless --recompute)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as sps
from joblib import Parallel, delayed

from infotheory.pipeline import get_pipeline_paths
from infotheory.session import discover_sessions, PROTOCOLS
from infotheory.continuous import merge_behavior_video, restrict_to_engaged, remove_pupil_outliers, \
    zscore_per_session, compute_trial_position_window_means
from infotheory.info_theory import mutual_information_shuffle, compare_mi_to_linear, linear_fit
from infotheory.behavior import compute_dprime
from infotheory.plotting import set_style, save_fig, clear_figures, PROTOCOL_COLORS
from infotheory import project_config as cfg
from infotheory.scope import add_scope_args, scope_from_args, select_sessions, scoped_dirs, data_stamp, \
    unit_noun
from infotheory.session_jobs import run_sessions
from infotheory.unit_stats import unit_curve, stable_seed
from infotheory.unit_plots import unit_bars, mouse_palette, mouse_legend

# --- what is analysed ------------------------------------------------------------------------------------
BEHAVIOR_VARS = ["runspeed", "pupil_area", "pupil_xpos", "pupil_ypos", "motionenergy", "lick"]
RATE_VARIABLES = ["lick"]
PRESTIM_WINDOW = (-30.0, -10.0)       # cm relative to stimStart (same as 1d)
# windows of the d' / criterion figures (cm relative to stimStart; the same as 2c). The pupil of EACH window is the x of its
# subplot -- never the mean over the whole trial.
D_WINDOWS = {"pre_stim": (-30.0, -10.0), "stim": (0.0, 20.0), "reward": (25.0, 45.0)}
# POSITION-RESOLVED analysis (section 10): the pupil-vs-y relation in sliding position windows along the corridor (cm relative to the
# stimulus onset; range and trusted range as in 2h). The x of each window is the pupil of THAT window, y is measured in the same window.
POS_RANGE = (-60.0, 80.0)
POS_WINDOW, POS_STEP = 20.0, 10.0
TRUSTED_POSITION_RANGE = (-40.0, 40.0)    # windows reaching outside are shaded (running / reward / ITI contaminate them)
POS_WINDOWS = [(float(lo), float(lo + POS_WINDOW)) for lo in np.arange(POS_RANGE[0], POS_RANGE[1] - POS_WINDOW + 1e-9, POS_STEP)]
POS_Y = ["cur_runspeed", "cur_motionenergy", "cur_lick", "cur_pupil_xpos", "cur_pupil_ypos", "lick_any"]   # lick_any = the trial's response
MIN_PER_CLASS_BIN = 3                 # 0 % and 100 % trials each needed in a pupil bin for its performance measures
DPRIME_BINS = 6                       # pupil quantile bins (within session, among ALL engaged trials) of the performance figure
# x: pupil AREA of this / the previous trial, and the pupil POSITION (eye position, horizontal / vertical; z-scored within session) of
# this trial, to compare the two. Figures are numbered 1a, 1b, ... in this order.
PUPIL_PREDICTORS = ["cur_pupil_area", "prev_pupil_area", "cur_pupil_xpos", "cur_pupil_ypos"]
X_LABELS = {"cur_pupil_area": "pupil area (this trial)", "prev_pupil_area": "pupil area (previous trial)",
            "cur_pupil_xpos": "pupil position x (this trial)", "cur_pupil_ypos": "pupil position y (this trial)"}
Y_VARIABLES = ["correct", "lick_signal", "lick_nosignal", "lick_any", "hit_100",
               "cur_runspeed", "cur_motionenergy", "cur_lick", "cur_pupil_area", "cur_pupil_xpos", "cur_pupil_ypos"]
Y_LABELS = {"correct": "accuracy", "lick_signal": "P(lick | signal>0)", "lick_nosignal": "P(lick | signal=0)",
            "lick_any": "P(lick)", "hit_100": "P(lick | 100%)", "cur_runspeed": "speed",
            "cur_motionenergy": "motion energy", "cur_lick": "lick rate",
            "cur_pupil_area": "pupil area", "cur_pupil_xpos": "pupil x position", "cur_pupil_ypos": "pupil y position"}
# PCA of the behavioral variables (section 9): pre-stimulus window means, z-scored within session; pupil AREA is left out and
# related to the components afterwards
PCA_VARIABLES = ["cur_runspeed", "cur_motionenergy", "cur_lick", "cur_pupil_xpos", "cur_pupil_ypos"]
PCA_RELATE_TO = "cur_pupil_area"
N_PCS_SHOWN = 3                       # components drawn in the curve figure (all are in the tables)
NOISE_TRIALS_ONLY = True              # True: only trials with 0 < signal < 100 (as in 1e); False: all engaged trials
PROTOCOLS_USED = ["DP", "DN"]         # protocols analysed (edit freely)
# on noise trials every lick is a response to a signal > 0: `correct` and `lick_signal` equal `lick_any`, and there are no
# 0 % / 100 % trials, so lick_nosignal, hit_100 and d' do not exist. They are left out instead of being repeated.
NOISE_UNDEFINED_OR_REDUNDANT = ["correct", "lick_signal", "lick_nosignal", "hit_100"]

# --- shared with the other scripts (project_config.py) ----------------------------------------------------
MI_BINS = cfg.RESPONSE_BINS
MIN_TRIALS_ANALYSIS = cfg.MIN_TRIALS_ANALYSIS

# --- local to this script ---------------------------------------------------------------------------------
N_QUANTILE_BINS = 6                   # pupil quantile bins for the curves (within session)
RANGE_PCT = (5, 95)                   # percentiles of z that define the "low" and "high" end of the range
ALPHA = 0.05
MIN_TRIALS_FIT = 40                   # trials needed for a quadratic fit in one unit
MIN_UNITS = 4                         # units needed for a group test
CV_FOLDS = 5
N_SHUFFLES = 100
JOBLIB_VERBOSE = 5

parser = argparse.ArgumentParser()
add_scope_args(parser)
args = parser.parse_args()
scope = scope_from_args(args)
AGG = scope.aggregate
UNIT = unit_noun(AGG)
N_JOBS = scope.n_jobs
AGG_SUFFIX = "" if AGG == "pooled" else f"__{AGG}"
unit_col = {"pooled": None, "session": "session_id", "mouse": "animal_id"}[AGG]
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


# --------------------------------------------------------------------- #
# 2. Per-session trial tables (one session in RAM at a time), cached
# --------------------------------------------------------------------- #
def process_session(ses, seed):
    """One row per engaged trial: pre-stimulus pupil (this and previous trial), the y variables, the signal."""
    if ses.trialdata is None or ses.behaviordata is None or ses.videodata is None:
        return None
    merged_full = zscore_per_session(remove_pupil_outliers(merge_behavior_video(ses.behaviordata, ses.videodata)))
    if merged_full.empty or "pupil_area" not in merged_full.columns:
        return None
    merged = restrict_to_engaged(merged_full, ses.trialdata)
    if merged.empty:
        return None
    available = [v for v in BEHAVIOR_VARS if v in merged.columns]
    trials = (ses.trialdata[ses.trialdata["engaged"] == 1]
              if "engaged" in ses.trialdata.columns else ses.trialdata.copy())
    pre = compute_trial_position_window_means(trials, merged, columns=available, s_start=PRESTIM_WINDOW[0],
                                              s_end=PRESTIM_WINDOW[1],
                                              rate_columns=[v for v in RATE_VARIABLES if v in available])
    pre = pre.rename(columns={v: f"cur_{v}" for v in available})
    t = trials[["trialNumber", "signal", "lickResponse"]].merge(pre, on="trialNumber", how="left")
    for w, (lo, hi) in D_WINDOWS.items():                      # pupil of each window (for the d' / criterion figure)
        pw = compute_trial_position_window_means(trials, merged, columns=["pupil_area"], s_start=lo, s_end=hi)
        t = t.merge(pw[["trialNumber", "pupil_area"]].rename(columns={"pupil_area": f"pupil_{w}"}), on="trialNumber", how="left")
    for k, (lo, hi) in enumerate(POS_WINDOWS):                 # all variables in each sliding position window (section 10)
        pw = compute_trial_position_window_means(trials, merged, columns=available, s_start=lo, s_end=hi,
                                                 rate_columns=[v for v in RATE_VARIABLES if v in available])
        pw = pw.rename(columns={v: f"{v}@{k}" for v in available})
        t = t.merge(pw[["trialNumber"] + [f"{v}@{k}" for v in available]], on="trialNumber", how="left")
    t = t.sort_values("trialNumber").reset_index(drop=True)
    t["prev_pupil_area"] = t["cur_pupil_area"].shift(1)
    t["correct"] = (((t["signal"] > 0) & (t["lickResponse"] == 1))
                    | ((t["signal"] == 0) & (t["lickResponse"] == 0))).astype(float)
    t["lick_any"] = t["lickResponse"].astype(float)
    t["lick_signal"] = np.where(t["signal"] > 0, t["lickResponse"], np.nan)
    t["lick_nosignal"] = np.where(t["signal"] == 0, t["lickResponse"], np.nan)
    t["hit_100"] = np.where(t["signal"] == 100, t["lickResponse"], np.nan)
    t["session_id"], t["animal_id"], t["protocol"] = ses.session_id, ses.animal_id, ses.protocol
    return t


session_tables = run_sessions(
    included_ids, process_session, list(PROTOCOLS), paths.store, "pupil_trials",
    key_params=dict(version="1f-v4", variables=BEHAVIOR_VARS, rate_variables=RATE_VARIABLES, window=PRESTIM_WINDOW,
                    d_windows=D_WINDOWS, pos_windows=POS_WINDOWS),
    load_kwargs=dict(load_behaviordata=True, load_videodata=True),
    n_jobs=N_JOBS, recompute=args.recompute, verbose=JOBLIB_VERBOSE,
    extra_key_by_session={sid: data_stamp(*triple_of[sid]) for sid in included_ids}, flatten=False)
session_tables = [t for t in session_tables if t is not None and len(t)]
if not session_tables:
    raise SystemExit("No selected session had behavior + video data -- nothing to analyze.")
trials = pd.concat(session_tables, ignore_index=True)
del session_tables
trials_all = trials                    # every engaged trial (the d' / criterion figure needs the 0 % / 100 % ones)
if NOISE_TRIALS_ONLY:
    trials = trials[(trials["signal"] > 0) & (trials["signal"] < 100)].copy()
protocols_present = [p for p in PROTOCOLS if (trials["protocol"] == p).any()]
ys = [y for y in Y_VARIABLES if y in trials.columns and trials[y].notna().any()
      and not (NOISE_TRIALS_ONLY and y in NOISE_UNDEFINED_OR_REDUNDANT)]
xs = [x for x in PUPIL_PREDICTORS if x in trials.columns and trials[x].notna().any()]
print(f"{len(trials)} engaged {'noise (0<signal<100) ' if NOISE_TRIALS_ONLY else ''}trials, {trials['session_id'].nunique()} sessions "
      f"({'/'.join(PROTOCOLS_USED)}); y: {ys}; x: {xs}")

# y centered WITHIN session (see docstring) -- raw values are kept in the *_raw columns for the CSVs
for y in ys:
    trials[f"{y}__c"] = trials[y] - trials.groupby("session_id")[y].transform("mean")


def unit_groups(df):
    if unit_col is None:
        return [("all", df)]
    return list(df.groupby(unit_col))


# --------------------------------------------------------------------- #
# 3. Per-unit tests
# --------------------------------------------------------------------- #
def _hc3_ols(X, y):
    """OLS with HC3 robust covariance. Returns beta, cov."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    w = (resid / np.clip(1 - h, 1e-6, None)) ** 2
    return beta, XtX_inv @ (X.T * w) @ X @ XtX_inv


def _slope_test(beta, cov, z0):
    """Slope of the quadratic at z0 = b1 + 2 b2 z0 and its robust SE."""
    g = np.array([0.0, 1.0, 2.0 * z0])
    return float(g @ beta), float(np.sqrt(max(g @ cov @ g, 1e-18)))


def _cv_gain(z, y, seed):
    n = len(z)
    folds = np.random.default_rng(seed).permutation(n) % CV_FOLDS
    se_lin = se_quad = 0.0
    for k in range(CV_FOLDS):
        tr, te = folds != k, folds == k
        for deg, acc in ((1, "lin"), (2, "quad")):
            X = np.vander(z, deg + 1, increasing=True)
            b = np.linalg.lstsq(X[tr], y[tr], rcond=None)[0]
            err = float(np.sum((y[te] - X[te] @ b) ** 2))
            if acc == "lin":
                se_lin += err
            else:
                se_quad += err
    return 1.0 - se_quad / se_lin if se_lin > 0 else np.nan


def fit_unit(protocol, unit, sub, x, y, seed):
    d = sub[[x, f"{y}__c", y]].dropna()
    if len(d) < MIN_TRIALS_FIT or d[x].std() == 0 or d[f"{y}__c"].std() == 0:
        return None
    z = d[x].to_numpy(float)
    z = (z - z.mean()) / z.std()
    yy = d[f"{y}__c"].to_numpy(float)
    yy = (yy - yy.mean()) / yy.std()
    X = np.column_stack([np.ones(len(z)), z, z ** 2])
    beta, cov = _hc3_ols(X, yy)
    se = np.sqrt(np.clip(np.diag(cov), 1e-18, None))
    p_quad = 2 * sps.norm.sf(abs(beta[2] / se[2]))
    lo, hi = np.percentile(z, RANGE_PCT)
    s_lo, se_lo = _slope_test(beta, cov, lo)
    s_hi, se_hi = _slope_test(beta, cov, hi)
    p_lo = sps.norm.sf(s_lo / se_lo)          # H1: slope at the low end > 0
    p_hi = sps.norm.cdf(s_hi / se_hi)         # H1: slope at the high end < 0
    turning = -beta[1] / (2 * beta[2]) if beta[2] != 0 else np.nan
    inv_u = bool(beta[2] < 0 and p_lo < ALPHA and p_hi < ALPHA and lo < turning < hi)
    u_shape = bool(beta[2] > 0 and sps.norm.cdf(s_lo / se_lo) < ALPHA and sps.norm.sf(s_hi / se_hi) < ALPHA
                   and lo < turning < hi)
    # two halves, split at the within-unit median of z
    med = np.median(z)
    lowm, highm = z <= med, z > med
    slope_low = np.polyfit(z[lowm], yy[lowm], 1)[0] if lowm.sum() >= 10 else np.nan
    slope_high = np.polyfit(z[highm], yy[highm], 1)[0] if highm.sum() >= 10 else np.nan
    # MI vs linear (shared bins, bias-corrected); only when enough trials
    mi_bits = mi_lin = excess = np.nan
    r = r2 = np.nan
    if len(d) >= MIN_TRIALS_ANALYSIS:
        rng = np.random.default_rng(seed)
        mi = mutual_information_shuffle(d[x], d[y], bins=MI_BINS, n_shuffles=N_SHUFFLES, rng=rng)
        lin = linear_fit(d[x], d[y])
        cmp = compare_mi_to_linear(mi.bits_corrected, lin.r2)
        mi_bits, mi_lin, r, r2 = mi.bits_corrected, cmp.mi_from_r2_bits, lin.r, lin.r2
        excess = mi_bits - mi_lin
    row = {"protocol": protocol, "x": x, "y": y, "n_trials": len(d), "beta_linear": beta[1],
           "beta_quadratic": beta[2], "p_quadratic": p_quad, "slope_low_end": s_lo, "slope_high_end": s_hi,
           "p_low_end_positive": p_lo, "p_high_end_negative": p_hi, "turning_point_z": turning,
           "range_low_z": lo, "range_high_z": hi, "inverted_U": inv_u, "U_shape": u_shape,
           "slope_lower_half": slope_low, "slope_upper_half": slope_high,
           "cv_gain": _cv_gain(z, yy, seed), "r": r, "r2": r2, "mi_bits_corrected": mi_bits,
           "mi_bits_from_r2": mi_lin, "mi_excess_bits": excess}
    row["unit"] = unit
    row["animal_id"] = sub["animal_id"].iloc[0]
    return row


jobs = [(protocol, unit, usub, x, y)
        for protocol in protocols_present
        for unit, usub in unit_groups(trials[trials["protocol"] == protocol])
        for x in xs for y in ys if y != x]
print(f"\nPer-unit fits: {len(jobs)} (protocol, {UNIT}, x, y) combinations (n_jobs={N_JOBS}) ...")
rows = Parallel(n_jobs=N_JOBS, backend="loky", verbose=JOBLIB_VERBOSE)(
    delayed(fit_unit)(p, u, s, x, y, stable_seed("1f", p, u, x, y)) for p, u, s, x, y in jobs)
per_unit = pd.DataFrame([r for r in rows if r is not None])
if per_unit.empty:
    raise SystemExit("No unit had enough trials for the fits.")
if AGG == "pooled":
    per_unit = per_unit.drop(columns=["unit", "animal_id"])
per_unit.to_csv(table_dir / f"pupil_nonlinearity_per_unit{AGG_SUFFIX}.csv", index=False)


# --------------------------------------------------------------------- #
# 4. Group tests across units (Holm across y within protocol and x)
# --------------------------------------------------------------------- #
def _holm(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return out
    order = np.argsort(p[ok])
    adj = np.maximum.accumulate((ok.sum() - np.arange(ok.sum())) * p[ok][order])
    res = np.empty(ok.sum())
    res[order] = np.minimum(adj, 1.0)
    out[ok] = res
    return out


def _wilcoxon(v, alternative):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < MIN_UNITS or np.all(v == 0):
        return np.nan
    try:
        return float(sps.wilcoxon(v, alternative=alternative).pvalue)
    except ValueError:
        return np.nan


group_rows = []
if AGG != "pooled":
    tests = [("beta_quadratic", "two-sided"), ("slope_lower_half", "greater"), ("slope_upper_half", "less"),
             ("cv_gain", "greater"), ("mi_excess_bits", "greater")]
    for (protocol, x), g in per_unit.groupby(["protocol", "x"]):
        for stat, alt in tests:
            recs = []
            for y, gy in g.groupby("y"):
                v = gy[stat].dropna()
                recs.append({"protocol": protocol, "x": x, "y": y, "statistic": stat, "alternative": alt,
                             "n_units": len(v), "mean": v.mean() if len(v) else np.nan,
                             "median": v.median() if len(v) else np.nan, "p": _wilcoxon(v, alt),
                             "frac_inverted_U": float(gy["inverted_U"].mean())})
            ps = _holm([r["p"] for r in recs])
            for r, pa in zip(recs, ps):
                r["p_holm"] = pa
            group_rows.extend(recs)
group_tests = pd.DataFrame(group_rows)
group_tests.to_csv(table_dir / f"pupil_nonlinearity_group_tests{AGG_SUFFIX}.csv", index=False)

print(f"\nPer-unit summary (mean over {UNIT}s; beta_quadratic < 0 = inverted U):")
summ = (per_unit.groupby(["protocol", "x", "y"])
        .agg(n_units=("n_trials", "size"), beta_quadratic=("beta_quadratic", "mean"),
             frac_inverted_U=("inverted_U", "mean"), cv_gain=("cv_gain", "mean"),
             mi_excess_bits=("mi_excess_bits", "mean")).round(4))
print(summ.to_string())


# --------------------------------------------------------------------- #
# 5. Curves: y (centered within session) per within-session pupil quantile bin
# --------------------------------------------------------------------- #
def add_bins(df, x):
    d = df.dropna(subset=[x]).copy()
    d["pupil_bin"] = d.groupby("session_id")[x].transform(
        lambda v: pd.qcut(v.rank(method="first"), N_QUANTILE_BINS, labels=False) + 1
        if len(v) >= 2 * N_QUANTILE_BINS else np.nan)
    return d.dropna(subset=["pupil_bin"])


def sig_star(p):
    return "" if not np.isfinite(p) else ("***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "")


def sig_lookup(protocol, x, y, stat):
    if group_tests.empty:
        return np.nan
    m = group_tests[(group_tests["protocol"] == protocol) & (group_tests["x"] == x) &
                    (group_tests["y"] == y) & (group_tests["statistic"] == stat)]
    return float(m["p_holm"].iloc[0]) if len(m) else np.nan


curve_rows = []
for xi, x in enumerate(xs):
    ncol = 4
    nrow = int(np.ceil(len(ys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.4 * nrow), squeeze=False, sharex=True)
    ylim_cache = {}
    for ax, y in zip(axes.ravel(), ys):
        if y == x:
            ax.axis("off")                      # y is the x variable itself: nothing to show
            continue
        for protocol in protocols_present:
            d = add_bins(trials[trials["protocol"] == protocol], x).dropna(subset=[f"{y}__c"])
            if d.empty:
                continue
            m, sem = unit_curve(d, "pupil_bin", f"{y}__c", AGG)
            ax.errorbar(m.index, m.values, yerr=sem.values, color=PROTOCOL_COLORS[protocol], marker="o",
                        markersize=4, capsize=3, label=protocol)
            for b, v in m.items():
                curve_rows.append({"protocol": protocol, "x": x, "y": y, "pupil_bin": int(b),
                                   "mean_centered": v, "sem": sem.get(b, 0.0)})
            star = sig_star(sig_lookup(protocol, x, y, "beta_quadratic"))
            if star:
                ax.text(0.02 + 0.2 * protocols_present.index(protocol), 0.96, f"{protocol}{star}",
                        transform=ax.transAxes, color=PROTOCOL_COLORS[protocol], fontsize=8, va="top")
        ax.axhline(0, color="grey", ls="--", lw=0.8)
        ax.set_title(Y_LABELS.get(y, y), fontsize=9)
        ax.set_xticks(range(1, N_QUANTILE_BINS + 1))
        ax.set_ylabel("change from session mean", fontsize=8)
    for ax in axes.ravel()[len(ys):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel(f"{X_LABELS.get(x, x)} quantile (within session; 1 = smallest)", fontsize=8)
    axes[0][0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle(f"{X_LABELS.get(x, x)}: y by pupil quantile (unit = {UNIT}; * = Holm-corrected Wilcoxon on "
                 f"quadratic coefficient)", y=1.0)
    fig.tight_layout()
    save_fig(fig, figdir, f"1{'abcd'[xi]}_pupil_curves_{x}")
    plt.close(fig)
pd.DataFrame(curve_rows).to_csv(table_dir / f"pupil_nonlinearity_curves{AGG_SUFFIX}.csv", index=False)

# d' and criterion per pupil quantile, 0 % and 100 % trials ONLY, one subplot per window (pre_stim / stim / reward); the x of each
# subplot is the pupil of THAT window (z-scored within session), quantile bins within session among all engaged trials
d01 = trials_all[trials_all["protocol"].isin(PROTOCOLS_USED)]          # ALL engaged trials (bins, hit rate, FA rate, balanced accuracy)


def perf_by_bins(d01, xcol):
    """Performance measures per (protocol, unit, pupil quantile bin of `xcol`); bins within session among all engaged trials."""
    out = []
    dw = d01.dropna(subset=[xcol]).copy()
    if dw.empty:
        return out
    dw["pupil_bin"] = dw.groupby("session_id")[xcol].transform(
        lambda v: pd.qcut(v.rank(method="first"), DPRIME_BINS, labels=False) + 1 if len(v) >= 4 * DPRIME_BINS else np.nan)
    dw = dw.dropna(subset=["pupil_bin"])
    for protocol in [q for q in PROTOCOLS_USED if (dw["protocol"] == q).any()]:
        d = dw[dw["protocol"] == protocol]
        for keys, g in d.groupby(UNIT_KEYS + ["pupil_bin"] if UNIT_KEYS else ["pupil_bin"]):
            keys = keys if isinstance(keys, tuple) else (keys,)
            n0, n100, npos = int((g["signal"] == 0).sum()), int((g["signal"] == 100).sum()), int((g["signal"] > 0).sum())
            ok_hf = min(npos, n0) >= MIN_PER_CLASS_BIN
            hit = float(g.loc[g["signal"] > 0, "lickResponse"].mean()) if ok_hf else np.nan        # any signal > 0
            fa = float(g.loc[g["signal"] == 0, "lickResponse"].mean()) if ok_hf else np.nan
            dp, crit = compute_dprime(g["signal"], g["lickResponse"]) if min(n100, n0) >= MIN_PER_CLASS_BIN else (np.nan, np.nan)
            rec = {"protocol": protocol, "pupil_bin": int(keys[-1]), "dprime": dp, "criterion": crit,
                   "hit_rate": hit, "fa_rate": fa, "balanced_accuracy": (hit + 1.0 - fa) / 2.0,
                   "n_trials_0": n0, "n_trials_100": n100, "n_trials_signal_pos": npos}
            rec.update(dict(zip(UNIT_KEYS, keys[:-1])))
            out.append(rec)
    return out


dp_rows = []
for w in D_WINDOWS:
    if f"pupil_{w}" in d01.columns:
        dp_rows += [dict(r, window=w) for r in perf_by_bins(d01, f"pupil_{w}")]
dp_df = pd.DataFrame(dp_rows)
if len(dp_df):
    dp_df.to_csv(table_dir / f"pupil_performance_by_quantile_windows{AGG_SUFFIX}.csv", index=False)
    wins = [w for w in D_WINDOWS if (dp_df["window"] == w).any()]
    MEASURES = [("hit_rate", "hit rate  P(lick | signal>0)\n(all signal strengths)"), ("fa_rate", "FA rate  P(lick | 0 %)"),
                ("balanced_accuracy", "balanced accuracy\n(hit[signal>0] + CR) / 2"), ("dprime", "d'"), ("criterion", "criterion")]
    fig, axes = plt.subplots(len(MEASURES), len(wins), figsize=(4.4 * len(wins), 2.9 * len(MEASURES)), squeeze=False, sharey="row")
    for r, (col, lab) in enumerate(MEASURES):
        for ax, w in zip(axes[r], wins):
            for protocol in [q for q in PROTOCOLS_USED if ((dp_df["window"] == w) & (dp_df["protocol"] == protocol)).any()]:
                dd = dp_df[(dp_df["window"] == w) & (dp_df["protocol"] == protocol)].dropna(subset=[col])
                if dd.empty:
                    continue
                if UNIT_KEYS:
                    per = dd.groupby(UNIT_KEYS + ["pupil_bin"])[col].mean().reset_index()
                    g = per.groupby("pupil_bin")[col]
                    m, sem = g.mean(), g.sem().fillna(0)
                else:
                    m, sem = dd.groupby("pupil_bin")[col].mean(), None
                ax.errorbar(m.index, m.values, yerr=None if sem is None else sem.values, color=PROTOCOL_COLORS[protocol],
                            marker="o", markersize=4, capsize=3, label=protocol)
            ax.set_xticks(range(1, DPRIME_BINS + 1))
            ax.set_ylabel(lab)
            if r == 0:
                lo, hi = D_WINDOWS[w]
                ax.set_title(f"{w} ({lo:g}..{hi:g} cm)", fontsize=10)
            if col == "balanced_accuracy":
                ax.axhline(0.5, color="grey", lw=0.8, ls="--")
            if r == len(MEASURES) - 1:
                ax.set_xlabel(f"pupil quantile in the {w} window\n(within session)")
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Performance by pupil quantile; hit / FA / balanced accuracy on ALL engaged trials, d' and criterion on the 0 % / 100 % trials (unit = {UNIT}; "
                 f"{'one value per bin, no error bars' if not UNIT_KEYS else 'mean +- SEM over units'})", y=1.0, fontsize=9)
    fig.tight_layout()
    save_fig(fig, figdir, "2_pupil_performance_windows")
    plt.close(fig)


# --------------------------------------------------------------------- #
# 6. Quadratic coefficient, inverted-U fraction and MI excess per (y, protocol)
# --------------------------------------------------------------------- #
for xi, x in enumerate(xs):
    px = per_unit[per_unit["x"] == x]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    width = 0.8 / max(len(protocols_present), 1)
    for ax, col, lab, hl in [(axes[0], "beta_quadratic", "standardized quadratic coefficient\n(< 0 = inverted U)", 0),
                              (axes[1], "inverted_U", f"fraction of {UNIT}s with a Lind-Mehlum inverted U", None),
                              (axes[2], "mi_excess_bits", "MI - MI implied by linear r$^2$ (bits)", 0)]:
        for pi, protocol in enumerate(protocols_present):
            for yi, y in enumerate(ys):
                v = px[(px["protocol"] == protocol) & (px["y"] == y)][col].astype(float).dropna()
                if v.empty:
                    continue
                xpos = yi + (pi - (len(protocols_present) - 1) / 2) * width
                ax.bar(xpos, v.mean(), width=width * 0.9, color=PROTOCOL_COLORS[protocol], alpha=0.5,
                       edgecolor="black", linewidth=0.4, label=protocol if yi == 0 else None)
                if AGG != "pooled" and col != "inverted_U" and len(v) > 1:
                    sub = px[(px["protocol"] == protocol) & (px["y"] == y)].dropna(subset=[col])
                    dot_c = [mouse_colors.get(a, "k") for a in sub["animal_id"]]
                    ax.scatter(xpos + np.random.default_rng(yi).uniform(-width / 3, width / 3, len(v)), v,
                               s=10, c=dot_c, edgecolor="black", linewidth=0.2, zorder=3)
                    ax.errorbar(xpos, v.mean(), yerr=v.sem(), color="black", capsize=2, lw=0.8)
        if hl is not None:
            ax.axhline(hl, color="grey", ls="--", lw=1)
        ax.set_xticks(range(len(ys)))
        ax.set_xticklabels([Y_LABELS.get(y, y) for y in ys], rotation=40, ha="right", fontsize=8)
        ax.set_ylabel(lab, fontsize=9)
        if col == "inverted_U":
            ax.set_ylim(0, 1)
    axes[0].legend(fontsize=7, frameon=False)
    if AGG != "pooled":
        mouse_legend(axes[2], mouse_colors, animals=px["animal_id"].unique(), loc="upper right")
    fig.suptitle(f"Is {X_LABELS.get(x, x)} non-linearly related to y? (unit = {UNIT}; dots = {UNIT}s)", y=1.02)
    fig.tight_layout()
    save_fig(fig, figdir, f"3{'abcd'[xi]}_pupil_quadratic_{x}")
    plt.close(fig)


# --------------------------------------------------------------------- #
# 7. Two halves: slope in the lower vs upper half of the pupil range, one line per unit
# --------------------------------------------------------------------- #
for xi, x in enumerate(xs):
    px = per_unit[per_unit["x"] == x]
    ncol = 4
    nrow = int(np.ceil(len(ys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.4 * nrow), squeeze=False, sharey=True)
    for ax, y in zip(axes.ravel(), ys):
        for pi, protocol in enumerate(protocols_present):
            d = px[(px["protocol"] == protocol) & (px["y"] == y)].dropna(subset=["slope_lower_half", "slope_upper_half"])
            if d.empty:
                continue
            xo = pi * 0.12
            for lo, hi in zip(d["slope_lower_half"], d["slope_upper_half"]):
                ax.plot([0 + xo, 1 + xo], [lo, hi], color=PROTOCOL_COLORS[protocol], alpha=0.35, lw=0.8)
            ax.scatter(np.full(len(d), 0 + xo), d["slope_lower_half"], s=10, color=PROTOCOL_COLORS[protocol])
            ax.scatter(np.full(len(d), 1 + xo), d["slope_upper_half"], s=10, color=PROTOCOL_COLORS[protocol])
            ax.plot([0 + xo, 1 + xo], [d["slope_lower_half"].mean(), d["slope_upper_half"].mean()],
                    color=PROTOCOL_COLORS[protocol], marker="D", lw=2.2, markeredgecolor="black",
                    label=protocol if y == ys[0] else None)
            if AGG != "pooled" and len(d) >= MIN_UNITS:
                try:
                    pp = sps.wilcoxon(d["slope_lower_half"], d["slope_upper_half"]).pvalue
                    ax.text(0.5 + xo, 0.97 - 0.07 * pi, f"{protocol} p={pp:.3f}", transform=ax.transAxes,
                            fontsize=7, color=PROTOCOL_COLORS[protocol], ha="center", va="top")
                except ValueError:
                    pass
        ax.axhline(0, color="grey", ls="--", lw=0.8)
        ax.set_xticks([0.12, 1.12])
        ax.set_xticklabels(["lower half", "upper half"], fontsize=8)
        ax.set_title(Y_LABELS.get(y, y), fontsize=9)
        ax.set_xlim(-0.4, 1.6)
    for ax in axes.ravel()[len(ys):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("slope (SD of y per SD of pupil)", fontsize=8)
    axes[0][0].legend(fontsize=7, frameon=False, loc="lower left")
    fig.suptitle(f"Slope of y vs {X_LABELS.get(x, x)} in the lower and upper half of the pupil range "
                 f"(inverted U: up then down; lines = {UNIT}s; paired Wilcoxon p)", y=1.0)
    fig.tight_layout()
    save_fig(fig, figdir, f"4{'abcd'[xi]}_pupil_two_halves_{x}")
    plt.close(fig)


# --------------------------------------------------------------------- #
# 8. Where is the peak? Turning points of the units that show an inverted U
# --------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(1.6 * max(len(ys), 4) + 2, 4))
tp = per_unit[(per_unit["x"] == "cur_pupil_area") & per_unit["inverted_U"]]
width = 0.8 / max(len(protocols_present), 1)
for pi, protocol in enumerate(protocols_present):
    for yi, y in enumerate(ys):
        v = tp[(tp["protocol"] == protocol) & (tp["y"] == y)]["turning_point_z"]
        if v.empty:
            continue
        xpos = yi + (pi - (len(protocols_present) - 1) / 2) * width
        ax.scatter(np.full(len(v), xpos) + np.random.default_rng(yi).uniform(-width / 4, width / 4, len(v)), v,
                   s=14, color=PROTOCOL_COLORS[protocol], edgecolor="black", linewidth=0.3,
                   label=protocol if yi == 0 else None)
        ax.plot([xpos - width / 2.2, xpos + width / 2.2], [v.median()] * 2, color="black", lw=1.5)
ax.axhline(0, color="grey", ls="--", lw=1)
ax.set_xticks(range(len(ys)))
ax.set_xticklabels([Y_LABELS.get(y, y) for y in ys], rotation=40, ha="right", fontsize=8)
ax.set_ylabel("peak location (pupil z within the unit)")
ax.set_title(f"Peak of the inverted U, {UNIT}s that pass the Lind-Mehlum test (black = median)", fontsize=9)
handles, labels = ax.get_legend_handles_labels()
if handles:
    ax.legend(dict(zip(labels, handles)).values(), dict(zip(labels, handles)).keys(), fontsize=7, frameon=False)
fig.tight_layout()
save_fig(fig, figdir, "5_pupil_peak_location")
plt.close(fig)


# --------------------------------------------------------------------- #
# 9. PCA of the behavioral variables and how its components relate to pupil area
# --------------------------------------------------------------------- #
# Per unit (--aggregate): the PCA_VARIABLES of the pre-stimulus window (z-scored within session, then pooled over the unit's
# sessions) -> principal components by SVD. Pupil AREA is not in the PCA; each component's trial score is related to it
# (correlation, and the same quadratic / inverted-U fit as everywhere above, via fit_unit with y = PCk). The sign of a component
# is arbitrary: each unit's components are aligned to the PCA of ALL the protocol's trials (reference), so r > 0 means the same
# thing in every unit. Components are matched by rank (PC1 with PC1, ...), which is exact for the pooled reference and
# approximate when two components of a unit have similar variance.
pca_vars = [v for v in PCA_VARIABLES if v in trials.columns and trials[v].notna().any()]
PC_NAMES = [f"PC{k + 1}" for k in range(len(pca_vars))]


def _zs(df, cols):
    return df.groupby("session_id")[cols].transform(lambda v: (v - v.mean()) / (v.std() if v.std() > 0 else np.nan))


def _pca(sub):
    """(loadings [var x PC], explained-variance ratio, scores [trials x PC], index of the trials used) or None."""
    z = _zs(sub, pca_vars).dropna()
    if len(z) < max(MIN_TRIALS_FIT, 5 * len(pca_vars)):
        return None
    X = z.to_numpy(float)
    X = X - X.mean(0)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt.T, S ** 2 / np.sum(S ** 2), X @ Vt.T, z.index


pca_rows, load_rows, score_frames, pca_fit_rows = [], [], [], []
if len(pca_vars) >= 3 and PCA_RELATE_TO in trials.columns:
    for protocol in protocols_present:
        tp = trials[trials["protocol"] == protocol]
        ref = _pca(tp)
        if ref is None:
            continue
        for unit, usub in unit_groups(tp):
            res = _pca(usub)
            if res is None:
                continue
            V, ev, sc, idx = res
            sg = np.sign(np.sum(V * ref[0], axis=0))
            sg[sg == 0] = 1.0
            V, sc = V * sg, sc * sg
            animal = usub["animal_id"].iloc[0]
            for k, nm in enumerate(PC_NAMES):
                pca_rows.append(dict(protocol=protocol, unit=unit, animal_id=animal, pc=nm, explained_variance=ev[k],
                                     n_trials=len(idx)))
                for vi, v in enumerate(pca_vars):
                    load_rows.append(dict(protocol=protocol, unit=unit, animal_id=animal, pc=nm, variable=v, loading=V[vi, k]))
            fr = usub.loc[idx, ["session_id", "animal_id", "protocol", PCA_RELATE_TO]].copy()
            for k, nm in enumerate(PC_NAMES):
                fr[nm] = sc[:, k]
                fr[f"{nm}__c"] = fr[nm] - fr.groupby("session_id")[nm].transform("mean")
            fr["unit"] = unit
            score_frames.append(fr)
            for k, nm in enumerate(PC_NAMES):
                r = fit_unit(protocol, unit, fr, PCA_RELATE_TO, nm, stable_seed("1f-pca", protocol, unit, nm))
                if r is not None:
                    pca_fit_rows.append(r)
if not pca_rows:
    print("\nPCA of the behavioral variables: no unit had enough complete trials.")
else:
    pca_var_df, pca_load_df = pd.DataFrame(pca_rows), pd.DataFrame(load_rows)
    pca_fit = pd.DataFrame(pca_fit_rows).drop(columns=["x"]).rename(columns={"y": "pc"})
    if AGG == "pooled":
        pca_var_df, pca_load_df = pca_var_df.drop(columns=["unit", "animal_id"]), pca_load_df.drop(columns=["unit", "animal_id"])
        pca_fit = pca_fit.drop(columns=["unit", "animal_id"])
    pca_var_df.to_csv(table_dir / f"pupil_pca_variance{AGG_SUFFIX}.csv", index=False)
    pca_load_df.to_csv(table_dir / f"pupil_pca_loadings{AGG_SUFFIX}.csv", index=False)
    pca_fit.to_csv(table_dir / f"pupil_pca_vs_pupil_area{AGG_SUFFIX}.csv", index=False)
    if AGG != "pooled":                       # across units: is the component correlated with pupil area? (Holm across PCs)
        trows = []
        for protocol, g in pca_fit.groupby("protocol"):
            recs = []
            for nm, gp in g.groupby("pc"):
                v = gp["r"].dropna()
                recs.append(dict(protocol=protocol, pc=nm, n_units=len(v), mean_r=v.mean() if len(v) else np.nan,
                                 frac_positive=float((v > 0).mean()) if len(v) else np.nan, p=_wilcoxon(v, "two-sided"),
                                 mean_beta_quadratic=gp["beta_quadratic"].mean(),
                                 p_quadratic=_wilcoxon(gp["beta_quadratic"].dropna(), "two-sided")))
            for r_, ph in zip(recs, _holm([r_["p"] for r_ in recs])):
                r_["p_holm"] = ph
            trows.extend(recs)
        pd.DataFrame(trows).to_csv(table_dir / f"pupil_pca_group_tests{AGG_SUFFIX}.csv", index=False)
    print(f"\nPCA of {pca_vars} (pre-stim window, within-session z): explained variance (mean over {UNIT}s) and r with {PCA_RELATE_TO}")
    print(pd.concat([pca_var_df.groupby(["protocol", "pc"])["explained_variance"].mean().round(3),
                     pca_fit.groupby(["protocol", "pc"])["r"].mean().round(3)], axis=1).to_string())

    # 9a. variance explained and loadings
    pcs_shown = PC_NAMES
    fig, axes = plt.subplots(2, len(protocols_present), figsize=(6.2 * len(protocols_present), 7.4), squeeze=False,
                             constrained_layout=True)
    for ci, protocol in enumerate(protocols_present):
        pv = pca_var_df[pca_var_df["protocol"] == protocol]
        if pv.empty:
            continue
        ax = axes[0][ci]
        g = pv.groupby("pc")["explained_variance"]
        ax.bar(range(len(pcs_shown)), [g.mean().get(n, np.nan) for n in pcs_shown], color=PROTOCOL_COLORS[protocol], alpha=0.6,
               yerr=[g.sem().fillna(0).get(n, 0) for n in pcs_shown] if AGG != "pooled" else None, capsize=3)
        ax.set_xticks(range(len(pcs_shown)))
        ax.set_xticklabels(pcs_shown)
        ax.set_ylabel("explained variance ratio")
        ax.set_title(f"{protocol}", color=PROTOCOL_COLORS[protocol])
        ax = axes[1][ci]
        L = pca_load_df[pca_load_df["protocol"] == protocol].groupby(["variable", "pc"])["loading"].mean().unstack()
        L = L.reindex(index=pca_vars, columns=pcs_shown)
        im = ax.imshow(L.to_numpy(float), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        for i in range(L.shape[0]):
            for j in range(L.shape[1]):
                ax.text(j, i, f"{L.iloc[i, j]:+.2f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(pcs_shown)))
        ax.set_xticklabels(pcs_shown)
        ax.set_yticks(range(len(pca_vars)))
        ax.set_yticklabels([Y_LABELS.get(v, v) for v in pca_vars], fontsize=8)
        ax.set_title("loadings (mean over units, sign-aligned)", fontsize=8)
    fig.colorbar(im, ax=axes[1].tolist(), shrink=0.7, label="loading")
    fig.suptitle(f"PCA of the behavioral variables, pre-stimulus window (unit = {UNIT}; pupil area is NOT in the PCA)", fontsize=10)
    save_fig(fig, figdir, "9a_behavior_pca_variance_loadings")
    plt.close(fig)

    # 9b. component score per pupil-area quantile (within session)
    sc_all = pd.concat(score_frames, ignore_index=True)
    ncomp = min(N_PCS_SHOWN, len(PC_NAMES))
    fig, axes = plt.subplots(ncomp, len(protocols_present), figsize=(4.6 * len(protocols_present), 3.0 * ncomp), squeeze=False,
                             sharex=True, sharey="row")
    for ci, protocol in enumerate(protocols_present):
        d_p = add_bins(sc_all[sc_all["protocol"] == protocol], PCA_RELATE_TO)
        for k in range(ncomp):
            ax = axes[k][ci]
            if d_p.empty:
                continue
            m, sem = unit_curve(d_p, "pupil_bin", f"{PC_NAMES[k]}__c", AGG)
            ax.errorbar(m.index, m.values, yerr=sem.values, color=PROTOCOL_COLORS[protocol], marker="o", markersize=4, capsize=3)
            ax.axhline(0, color="grey", lw=0.7)
            fr_ = pca_fit[(pca_fit["protocol"] == protocol) & (pca_fit["pc"] == PC_NAMES[k])]["r"]
            ax.set_title(f"{protocol} {PC_NAMES[k]}: mean r = {fr_.mean():+.2f}" if len(fr_) else f"{protocol} {PC_NAMES[k]}", fontsize=9,
                         color=PROTOCOL_COLORS[protocol])
            ax.set_ylabel("component score\n(centered in session)")
            if k == ncomp - 1:
                ax.set_xlabel(f"{X_LABELS.get(PCA_RELATE_TO, PCA_RELATE_TO)} quantile (within session)")
    fig.suptitle(f"Behavioral PCs vs pupil area (unit = {UNIT}; mean +- SEM over units)", y=1.0, fontsize=10)
    fig.tight_layout()
    save_fig(fig, figdir, "9b_behavior_pca_vs_pupil_curves")
    plt.close(fig)

    # 9c. correlation of each component with pupil area, one dot per unit
    fig, ax = plt.subplots(figsize=(1.6 * len(PC_NAMES) + 2.5, 4))
    width = 0.8 / max(len(protocols_present), 1)
    for pi, protocol in enumerate(protocols_present):
        for k, nm in enumerate(PC_NAMES):
            v = pca_fit[(pca_fit["protocol"] == protocol) & (pca_fit["pc"] == nm)]["r"].dropna()
            if v.empty:
                continue
            xp = k + (pi - (len(protocols_present) - 1) / 2) * width
            ax.bar(xp, v.mean(), width=width * 0.9, color=PROTOCOL_COLORS[protocol], alpha=0.5, edgecolor="black", linewidth=0.4,
                   label=protocol if k == 0 else None)
            if AGG != "pooled" and len(v) > 1:
                ax.scatter(xp + np.random.default_rng(k).uniform(-width / 3, width / 3, len(v)), v, s=10, color="k", zorder=3)
                ax.errorbar(xp, v.mean(), yerr=v.sem(), color="black", capsize=2, lw=0.8)
    ax.axhline(0, color="grey", ls="--", lw=1)
    ax.set_xticks(range(len(PC_NAMES)))
    ax.set_xticklabels(PC_NAMES)
    ax.set_ylabel(f"Pearson r with {X_LABELS.get(PCA_RELATE_TO, PCA_RELATE_TO)}")
    ax.legend(fontsize=7, frameon=False)
    ax.set_title(f"Which behavioral component tracks pupil area? (unit = {UNIT}; dots = {UNIT}s)", fontsize=9)
    fig.tight_layout()
    save_fig(fig, figdir, "9c_behavior_pca_pupil_correlation")
    plt.close(fig)


# --------------------------------------------------------------------- #
# 10. Position-resolved: the pupil-vs-y relation in sliding position windows along the corridor
# --------------------------------------------------------------------- #
# Window k: x = pupil area of the window, y = the variable in the SAME window (POS_Y; lick_any = the trial's response). Same fit as
# everywhere above (fit_unit: standardized quadratic, linear r, Lind-Mehlum inverted U, CV gain, MI excess), per unit and window; group
# level = Wilcoxon over units of the quadratic coefficient per window (Holm across windows within protocol and y). Performance
# (hit / FA / balanced accuracy / d' / criterion, all engaged trials as in figure 2): top-minus-bottom pupil bin and
# centre-minus-edges (mean of the middle bins - mean of the two outer bins, > 0 = inverted U) per window.
XCOL = "cur_pupil_area"
win_k = [k for k in range(len(POS_WINDOWS)) if f"pupil_area@{k}" in trials.columns]
pos_ys = [y for y in POS_Y if y == "lick_any" or f"{y[4:]}@{win_k[0]}" in trials.columns] if win_k else []
if win_k and pos_ys:
    pos_jobs = []
    for k in win_k:
        fw = trials[["session_id", "animal_id", "protocol", "lick_any"]].copy()
        fw[XCOL] = trials[f"pupil_area@{k}"]
        for y in pos_ys:
            if y != "lick_any":
                fw[y] = trials[f"{y[4:]}@{k}"]
            fw[f"{y}__c"] = fw[y] - fw.groupby("session_id")[y].transform("mean")
        for protocol in protocols_present:
            for unit, usub in unit_groups(fw[fw["protocol"] == protocol]):
                for y in pos_ys:
                    pos_jobs.append((protocol, unit, usub, k, y))
    print(f"\nPosition-resolved fits: {len(pos_jobs)} (protocol, {UNIT}, window, y) combinations ...")
    prow = Parallel(n_jobs=N_JOBS, backend="loky", verbose=JOBLIB_VERBOSE)(
        delayed(fit_unit)(p_, u_, s_, XCOL, y_, stable_seed("1f-pos", p_, u_, k_, y_)) for p_, u_, s_, k_, y_ in pos_jobs)
    pos_fit = []
    for (p_, u_, s_, k_, y_), r_ in zip(pos_jobs, prow):
        if r_ is not None:
            r_ = dict(r_, window_k=k_, win_lo=POS_WINDOWS[k_][0], win_hi=POS_WINDOWS[k_][1],
                      win_center=(POS_WINDOWS[k_][0] + POS_WINDOWS[k_][1]) / 2)
            pos_fit.append(r_)
    pos_fit = pd.DataFrame(pos_fit)
    del pos_jobs
else:
    pos_fit = pd.DataFrame()
if pos_fit.empty:
    print("\nPosition-resolved analysis: no unit had enough trials.")
else:
    if AGG == "pooled":
        pos_fit = pos_fit.drop(columns=["unit", "animal_id"])
    pos_fit.to_csv(table_dir / f"pupil_position_windows_per_unit{AGG_SUFFIX}.csv", index=False)
    grows = []
    for (protocol, y, k), g in pos_fit.groupby(["protocol", "y", "window_k"]):
        grows.append(dict(protocol=protocol, y=y, window_k=k, win_center=g["win_center"].iloc[0], n_units=len(g),
                          mean_beta_quadratic=g["beta_quadratic"].mean(),
                          sem_beta_quadratic=g["beta_quadratic"].sem() if len(g) > 1 else 0.0,
                          mean_r=g["r"].mean(), sem_r=g["r"].sem() if len(g) > 1 else 0.0,
                          frac_inverted_U=float(g["inverted_U"].mean()), mean_cv_gain=g["cv_gain"].mean(),
                          p_beta_quadratic=_wilcoxon(g["beta_quadratic"].dropna(), "two-sided") if AGG != "pooled" else np.nan,
                          p_r=_wilcoxon(g["r"].dropna(), "two-sided") if AGG != "pooled" else np.nan))
    pos_group = pd.DataFrame(grows)
    for col in ("p_beta_quadratic", "p_r"):
        pos_group[col + "_holm"] = np.nan
        for _, idx in pos_group.groupby(["protocol", "y"]).groups.items():
            pos_group.loc[idx, col + "_holm"] = _holm(pos_group.loc[idx, col].to_numpy())
    pos_group.to_csv(table_dir / f"pupil_position_windows_group{AGG_SUFFIX}.csv", index=False)

    def shade_untrusted(ax):
        lo_t, hi_t = TRUSTED_POSITION_RANGE
        ax.axvspan(-1e9, lo_t + POS_WINDOW / 2, color="grey", alpha=0.12, lw=0)
        ax.axvspan(hi_t - POS_WINDOW / 2, 1e9, color="grey", alpha=0.12, lw=0)
        ax.axvline(0, color="black", ls=":", lw=0.8)
        ax.set_xlim(POS_WINDOWS[0][0] + POS_WINDOW / 2 - POS_STEP / 2, POS_WINDOWS[-1][1] - POS_WINDOW / 2 + POS_STEP / 2)

    def line_by_position(ax, df, col, sem_col, color, label, p_col=None):
        if AGG != "pooled":
            df = df[df["n_units"] >= MIN_UNITS]            # windows where fewer than MIN_UNITS units have enough trials are not drawn
        df = df.sort_values("win_center")
        if df.empty:
            return
        ax.errorbar(df["win_center"], df[col], yerr=None if sem_col is None else df[sem_col], color=color, marker="o",
                    markersize=3.5, capsize=2, lw=1.4, label=label)
        if p_col is not None:
            sig = df[p_col] < ALPHA
            if sig.any():
                ax.scatter(df.loc[sig, "win_center"], df.loc[sig, col], s=60, facecolors="none", edgecolors=color, linewidths=1.4,
                           zorder=4)

    stat_figs = [("10a_pupil_effect_by_position_quadratic", "mean_beta_quadratic", "sem_beta_quadratic", "p_beta_quadratic_holm",
                  "standardized quadratic coefficient (< 0 = inverted U)", 0.0),
                 ("10b_pupil_effect_by_position_linear", "mean_r", "sem_r", "p_r_holm", "linear r with pupil area", 0.0),
                 ("10c_pupil_inverted_u_by_position", "frac_inverted_U", None, None,
                  f"fraction of {UNIT}s with a Lind-Mehlum inverted U", None)]
    for fname, col, sem_col, p_col, lab, hl in stat_figs:
        ncol = 3
        nrow = int(np.ceil(len(pos_ys) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.2 * nrow), squeeze=False, sharex=True)
        for ax, y in zip(axes.ravel(), pos_ys):
            shade_untrusted(ax)
            for protocol in protocols_present:
                line_by_position(ax, pos_group[(pos_group["protocol"] == protocol) & (pos_group["y"] == y)], col,
                                 sem_col if AGG != "pooled" else None, PROTOCOL_COLORS[protocol], protocol, p_col)
            if hl is not None:
                ax.axhline(hl, color="grey", lw=0.8, ls="--")
            ax.set_title(Y_LABELS.get(y, y), fontsize=9)
            ax.set_ylabel(lab, fontsize=7)
        for ax in axes.ravel()[len(pos_ys):]:
            ax.axis("off")
        for ax in axes[-1]:
            ax.set_xlabel("window centre (cm from stimulus onset)", fontsize=8)
        axes[0][0].legend(fontsize=7, frameon=False)
        fig.suptitle(f"Pupil area vs y in sliding {POS_WINDOW:g} cm position windows (unit = {UNIT}; "
                     f"{'mean +- SEM over units; circle = Holm-corrected Wilcoxon p < ' + str(ALPHA) if p_col else 'fraction of units'}; "
                     f"grey = outside {TRUSTED_POSITION_RANGE[0]:g}..{TRUSTED_POSITION_RANGE[1]:g} cm)", y=1.0, fontsize=9)
        fig.tight_layout()
        save_fig(fig, figdir, fname)
        plt.close(fig)

    # performance (all engaged trials) by position window
    pb = []
    for k in win_k:
        pb += [dict(r_, window_k=k, win_center=(POS_WINDOWS[k][0] + POS_WINDOWS[k][1]) / 2)
               for r_ in perf_by_bins(d01, f"pupil_area@{k}")]
    MEAS = [("hit_rate", "hit rate"), ("fa_rate", "FA rate"), ("balanced_accuracy", "balanced accuracy"), ("dprime", "d'"),
            ("criterion", "criterion")]
    prow_ = []
    if pb:
        pbd = pd.DataFrame(pb)
        for keys, g in pbd.groupby(["protocol"] + UNIT_KEYS + ["window_k"]):
            keys = keys if isinstance(keys, tuple) else (keys,)
            g = g.sort_values("pupil_bin").set_index("pupil_bin")
            for m_, _ in MEAS:
                v = g[m_].reindex(range(1, DPRIME_BINS + 1)).to_numpy(float)
                if not (np.isfinite(v[0]) and np.isfinite(v[-1])) or not np.isfinite(v[1:-1]).any():
                    continue
                rec = dict(zip(["protocol"] + UNIT_KEYS + ["window_k"], keys), measure=m_, top_minus_bottom=v[-1] - v[0],
                           center_minus_edges=np.nanmean(v[1:-1]) - (v[0] + v[-1]) / 2.0,
                           win_center=g["win_center"].iloc[0])
                prow_.append(rec)
    if prow_:
        pp = pd.DataFrame(prow_)
        pp.to_csv(table_dir / f"pupil_position_performance_per_unit{AGG_SUFFIX}.csv", index=False)
        stats_ = []
        for (protocol, m_, k), g in pp.groupby(["protocol", "measure", "window_k"]):
            rec = dict(protocol=protocol, measure=m_, window_k=k, win_center=g["win_center"].iloc[0], n_units=len(g))
            for c_ in ("top_minus_bottom", "center_minus_edges"):
                rec[f"mean_{c_}"] = g[c_].mean()
                rec[f"sem_{c_}"] = g[c_].sem() if len(g) > 1 else 0.0
                rec[f"p_{c_}"] = _wilcoxon(g[c_].dropna(), "two-sided") if AGG != "pooled" else np.nan
            stats_.append(rec)
        ppg = pd.DataFrame(stats_)
        for c_ in ("top_minus_bottom", "center_minus_edges"):
            ppg[f"p_{c_}_holm"] = np.nan
            for _, idx in ppg.groupby(["protocol", "measure"]).groups.items():
                ppg.loc[idx, f"p_{c_}_holm"] = _holm(ppg.loc[idx, f"p_{c_}"].to_numpy())
        ppg.to_csv(table_dir / f"pupil_position_performance_group{AGG_SUFFIX}.csv", index=False)
        fig, axes = plt.subplots(len(MEAS), 2, figsize=(10.5, 2.9 * len(MEAS)), squeeze=False, sharex=True)
        for r_i, (m_, mlab) in enumerate(MEAS):
            for c_i, (c_, clab) in enumerate([("top_minus_bottom", "highest - lowest pupil bin"),
                                               ("center_minus_edges", "centre - edges (> 0 = inverted U)")]):
                ax = axes[r_i][c_i]
                shade_untrusted(ax)
                for protocol in protocols_present:
                    line_by_position(ax, ppg[(ppg["protocol"] == protocol) & (ppg["measure"] == m_)], f"mean_{c_}",
                                     f"sem_{c_}" if AGG != "pooled" else None, PROTOCOL_COLORS[protocol], protocol,
                                     f"p_{c_}_holm")
                ax.axhline(0, color="grey", lw=0.8, ls="--")
                ax.set_ylabel(f"{mlab}: {clab}", fontsize=7)
                if r_i == 0:
                    ax.set_title(clab, fontsize=9)
        axes[0][0].legend(fontsize=7, frameon=False)
        for ax in axes[-1]:
            ax.set_xlabel("window centre (cm from stimulus onset)", fontsize=8)
        fig.suptitle(f"Performance vs pupil area by position window (all engaged trials; {DPRIME_BINS} pupil bins; unit = {UNIT}; "
                     f"circle = Holm-corrected Wilcoxon p < {ALPHA})", y=1.0, fontsize=9)
        fig.tight_layout()
        save_fig(fig, figdir, "10d_pupil_performance_by_position")
        plt.close(fig)

print(f"\nWrote tables to {table_dir}, figures to {figdir}")
