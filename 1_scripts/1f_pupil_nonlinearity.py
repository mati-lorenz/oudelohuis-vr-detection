# -*- coding: utf-8 -*-
"""
1f_pupil_nonlinearity
======================
Is the relationship between PUPIL AREA and other variables NON-LINEAR -- specifically an INVERTED U
(Yerkes-Dodson: best performance / most "engaged" behavior at intermediate arousal)? Across DP and DN (PROTOCOLS_USED, edit at the top).

PERFORMANCE vs PUPIL (figure 2, separate from everything above): ALL engaged trials. hit rate = P(lick | signal > 0) over every signal
strength, FA rate = P(lick | signal = 0) (the only no-signal trials), balanced accuracy = (hit + 1 - FA) / 2. d' and criterion use
the NOISE trials only (0 < signal < 100 %, no 0 % / 100 % trials): a psychometric (probit) fit per session and pupil bin, with the
floor / lapse fixed from 1b, gives the predicted 0 % / 100 % hit and FA rates, from which the classical d' and criterion follow
(DPRIME_METHOD / CRITERION_METHOD; infotheory/noise_sdt.py explains the variants and what cannot be recovered without the 0 % trials). Figure 2b puts the
variants and the classical 0 % / 100 % version side by side. A bin needs >= MIN_PER_CLASS_BIN trials of each class (hit / FA) and
>= MIN_NOISE_PER_BIN noise trials (d' / criterion), one subplot per window (D_WINDOWS: pre_stim,
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
                   comparable between variables). CROSS-VALIDATED (see below): beta_quadratic_cv < 0 = inverted
                   U, > 0 = U. The in-sample beta_quadratic / p_quadratic (full-sample fit, HC3 SEs) are kept
                   as a reference column but are no longer what is tested or plotted.
  B. LIND-MEHLUM   the proper "is it really a hump" test: the quadratic coefficient alone can be significant
                   for a curve that is merely saturating. Inverted U requires the fitted slope at the LOW end
                   of the pupil range to be significantly > 0 AND at the HIGH end significantly < 0
                   (intersection-union, so no extra correction), with the turning point inside the range.
                   Range = RANGE_PCT percentiles of z (not the extremes, which are leverage points; this
                   endpoint definition is fixed in advance and is not itself cross-validated -- see the CV note
                   below for why).
  C. TWO HALVES    linear slope in the lower and in the upper half of the session's pupil (split at the
                   within-session median -- fixed in advance, so no breakpoint hunting). Inverted U =
                   slope_low > 0 and slope_high < 0. Shown as paired lines per unit (figure 3). Each half is
                   fit AND read off its own, disjoint trials, so there is no fit-on-what-you-score-on bias here
                   and this check is NOT cross-validated (same scoping rule as 2j/2k's fixed-bin statistics).
  D. MODEL-FREE    bias-corrected histogram MI(pupil; y) with the SHARED project bins (RESPONSE_BINS) versus
                   the MI implied by the LINEAR r^2: the excess is information a straight line cannot see.
                   (Same estimator as 1d / 2c / 3a-3c, so the numbers are comparable before bias correction.)
                   Not cross-validated: this is the same per-pair, no-extra-selection Pearson r/MI comparison
                   used throughout the project; there is no multi-parameter in-sample fit to inflate here.

CROSS-VALIDATION (A and B; fit_cv, 2-fold, odd/even trial index parity, mirroring 2j_pupil_activity_shape.py):
the quadratic (and linear) model is fit on one half of the unit's trials and scored on the OTHER half, in both
directions, pooling SSE/SST before computing R2 -- the same pattern as 2j and 2k. beta_linear_cv/beta_quadratic_cv
are the mean of the two half-fits' coefficients (never a coefficient fit on the same trials it is tested against).
dR2_cv = R2_quad_cv - R2_lin_cv can now be NEGATIVE (impossible for an in-sample nested-model comparison, where
adding the z^2 term can only reduce or keep the same in-sample SSE). cv_gain = 1 - SSE_quad_cv / SSE_lin_cv is kept
under its previous name for continuity, but is now computed on the SAME odd/even split as everything else here,
not an independent random 5-fold split. Significance (p_quadratic_cv, p_low_end_positive_cv, p_high_end_negative_cv,
p_cv_gain) comes from a trial-label permutation null that recomputes the WHOLE cross-validated statistic
(fit_cv on the shuffled labels, same odd/even split) at every one of N_SHUFFLES draws -- never from the old HC3
sandwich-SE formula, which assumes a single in-sample fit and would be the wrong reference distribution for an
averaged, cross-validated coefficient. inverted_U_cv / U_shape_cv / turning_point_z_cv are the CV analogues of
inverted_U / U_shape / turning_point_z, using the averaged CV coefficients and the permutation p-values.
SUPPLEMENTARY 5-FOLD (fit_kfold, idea 2/3): the same fit-on-the-other-folds/score-on-the-held-out-fold logic,
generalised from 2 to 5 folds -- beta_linear_5fold/beta_quadratic_5fold (idea 3: averaged over the 5 per-fold
training fits), dR2_5fold/cv_gain_5fold (idea 2: held-out loss averaged/pooled over the 5 folds), p_quadratic_5fold/
p_dR2_5fold. Kept alongside the odd/even fit_cv headline as a cross-check (the two should agree), not a replacement.
Figure 1{a..d}_cv_fit_vs_heldout_points_<x> (idea 1): the quadratic fit on the protocol's ODD trials (pooled across
units) plotted as a line against the EVEN-trial bin means it did not see (points) -- a visual version of what
beta_quadratic_cv/dR2_cv already test numerically: if the line and the points disagree, the curve does not replicate.
Why beta_quadratic itself needed this (unlike C and D): fitting the quadratic on ALL of a unit's trials and then
testing it with a sandwich SE is standard regression inference and is not, by itself, biased -- but the Lind-Mehlum
slopes and the turning point are all read off that SAME single fit, and cv_gain already existed as a reminder that
the project treats "does the curvature generalize" as a question that needs held-out trials, not an in-sample
R2 comparison. Cross-validating the coefficients and the associated tests keeps every A/B number consistent with
cv_gain's held-out logic instead of mixing an in-sample estimate with a held-out one for the same curve.
Group level (needs >= MIN_UNITS units): Wilcoxon signed-rank across units for beta_quadratic_cv != 0, slope_low
> 0, slope_high < 0, cv_gain > 0 and MI excess > 0, Holm-corrected across the y variables within a protocol
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

TUNING CURVES (section 11, figures 11-13, pupil_tuning_curves[__<aggregate>].csv; the MIRROR of 1g, which does the same vs running speed, code in
infotheory/tuning_common.py): every other variable (speed, pupil position, motion energy, lick rate, video PCs, position in the trial,
P(lick) per window) against N_QUANTILE_BINS pupil quantile bins within session. Figure 11 = the whole trial as POOLED FRAMES, figure 12 =
the windows (TUNING_WINDOWS), figure 13 = the whole trial as SECTION means (SECTION_WIDTH cm sections; independent-ish samples, the
version to use for any test, since frames are autocorrelated). Figures 11w-13w = the SAME curves with FIXED, EQUAL-WIDTH pupil bins
(EQUALWIDTH_RANGE, same z range in every session) instead of the within-session quantiles -- equipopulated bins give every bin the
same power but distort the x-axis spacing where the pupil distribution isn't uniform; equal-width bins keep the real spacing but can
leave a sparse bin noisy. Both are shown; 11-13 stay the default (unchanged), 11w-13w are the additional check.

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
                         linear coefficients, p-values, slopes at the range ends, turning point, inverted-U flags
                         (in-sample, reference only) AND their _cv cross-validated counterparts (headline: fit
                         on odd/even trials, scored on the other half, permutation p-values -- see docstring),
                         plus the supplementary _5fold columns (idea 2/3), two-halves slopes (not cross-validated),
                         cv_gain / dR2_cv, MI / MI-from-r2 / excess
            pupil_nonlinearity_group_tests[__<aggregate>].csv  Wilcoxon tests across units, Holm-corrected
            pupil_nonlinearity_curves[__<aggregate>].csv      mean +- SEM of each y per pupil quantile bin
            pupil_performance_by_quantile_windows[__<aggregate>].csv   hit / FA rate, balanced accuracy (all trials), d' / criterion (noise trials: DPRIME_METHOD / CRITERION_METHOD in `dprime`/`criterion`, every variant in dprime_<method>/criterion_<method>; classical 0/100 in *_0_100) per pupil quantile, per window
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
from infotheory.continuous import merge_behavior_video, restrict_to_engaged, restrict_to_trials, remove_pupil_outliers, \
    zscore_per_session, compute_trial_position_window_means
from infotheory.info_theory import mutual_information_shuffle, compare_mi_to_linear, linear_fit
from infotheory.behavior import compute_dprime
from infotheory.noise_sdt import noise_sdt_session, NOISE_COLS, NOISE_METHODS
from infotheory import tuning_common as tc
from infotheory import performance_common as pc
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
TUNING_WINDOWS = D_WINDOWS            # the tuning curves of figures 11-13 use the same windows as figure 2 (and as 1g)
SECTION_RANGE, SECTION_WIDTH = POS_RANGE, POS_WINDOW     # the whole trial as section means (figure 13)
X_VAR = "pupil_area"                  # x of the tuning curves of figures 11-13 (1g: runspeed)
TUNING_Y = [v for v in tc.TUNING_VARS if v != X_VAR]     # the SAME y variables as 1g
LICK_SMOOTH_S = 1.0                   # lick events -> lick rate (Hz) for the Spearman values of figure 11
MIN_PER_BIN_FRAMES = 30               # whole-trial (frame) level: samples needed in a bin of a session
MIN_PER_BIN_TRIALS = 5                # window / section level: trials (samples) needed in a bin of a session
DPRIME_BINS = 6                       # pupil quantile bins (within session, among ALL engaged trials) of the performance figure
# d' and criterion are computed on NOISE trials only (0 < signal < 100 %; see infotheory/noise_sdt.py for the methods and what is lost):
#   "probit_pred"   psychometric fit on the noise trials of each (session, bin), floor / lapse fixed from 1b, 0 % / 100 % rates predicted
#                   from the fit, classical formulas -> same scale as the classical d' / criterion (recommended)
#   "probit_shift"  scale-free, interpolated: criterion = threshold shift of the bin's curve in units of the session's 1b sigma
#                   (> 0 = conservative, 0 = session average), d' = slope relative to the session's 1b slope (1 = average)
#   "split"         weak vs strong noise trials as the two classes (model-free; d' is much smaller)
DPRIME_METHOD = "probit_shift"        # d' drawn in figure 2 and figure 10d (all variants are in figure 2b and the csv)
# "probit_shift" is the inverse-sigma sensitivity: dprime_probit_shift = b / b0 = (1/sigma of this bin's psychometric fit) /
# (1/sigma of the session's own 1b curve), i.e. sensitivity relative to the session's own baseline (1 = average, >1 = more
# sensitive than usual) -- not an absolute 1/sigma, which would not be comparable across sessions of different difficulty.
# The classical criterion from 0 % / 100 % rates is dominated by the floor / lapse balance, so a shift of the psychometric THRESHOLD barely
# moves it -- and so does "probit_pred". "probit_shift" follows threshold shifts and stays inside the tested stimulus range.
CRITERION_METHOD = "probit_shift"    # criterion drawn in figure 2 and figure 10d
MIN_NOISE_PER_BIN = 20                # noise trials needed in one (session, pupil bin) for its d' / criterion
PROBIT_PRIOR_A_SD = 1.0               # sd of the prior on the probit offset a (z units), centred on the session's 1b curve
PROBIT_PRIOR_B_FRAC = 0.75            # sd of the prior on the probit slope b, as a fraction of the session's 1b slope (smaller = more shrinkage)
RATE_CLIP = 0.005                     # predicted hit / FA rates are clipped to [RATE_CLIP, 1 - RATE_CLIP] before the z transform
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
N_QUANTILE_BINS = 8                   # pupil quantile bins for the curves (within session); the same as 1g's speed bins
RANGE_PCT = (5, 95)                   # percentiles of z that define the "low" and "high" end of the range
EQUALWIDTH_RANGE = (-3.0, 3.0)        # pupil area (z): FIXED range for the equal-width tuning-curve bins (figures 11w-13w),
                                       # alongside the default N_QUANTILE_BINS within-session quantile bins (figures 11-13)
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
    merged_full = tc.add_lick_rate(merged_full, LICK_SMOOTH_S)          # smoothed lick rate (Hz), before any trial restriction
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

    # tuning curves vs X_VAR (figures 11-13, shared with 1g via infotheory/tuning_common.py): whole trial (frames), windows, sections
    sel = ses.trialdata[ses.trialdata["engaged"] == 1] if "engaged" in ses.trialdata.columns else ses.trialdata
    if NOISE_TRIALS_ONLY:
        sel = sel[(sel["signal"] > 0) & (sel["signal"] < 100)]
    tun, spm = pd.DataFrame(), pd.DataFrame()
    if len(sel):
        mt = restrict_to_trials(merged_full, ses.trialdata, sel["trialNumber"])
        if len(mt) and X_VAR in mt.columns:
            mt = tc.add_pos_rel(mt, sel)
            ys_t = [v for v in TUNING_Y if v in mt.columns and mt[v].notna().any()]
            tun = tc.session_tuning(mt, sel, x_var=X_VAR, y_vars=ys_t, rate_vars=RATE_VARIABLES, fs=tc.sampling_rate(merged_full),
                                    windows=TUNING_WINDOWS, section_range=SECTION_RANGE, section_width=SECTION_WIDTH,
                                    n_bins=N_QUANTILE_BINS, stop_bin=False, min_frames=MIN_PER_BIN_FRAMES,
                                    min_trials=MIN_PER_BIN_TRIALS, fixed_range=EQUALWIDTH_RANGE)
            spm = tc.spearman_with_x(mt, X_VAR, ys_t)
    for tbl in (tun, spm):
        if len(tbl):
            tbl["session_id"], tbl["animal_id"], tbl["protocol"] = ses.session_id, ses.animal_id, ses.protocol
    return dict(trials=t, tuning=tun, spearman=spm)


session_tables = run_sessions(
    included_ids, process_session, list(PROTOCOLS), paths.store, "pupil_trials",
    key_params=dict(version="1f-v6", tuning=(X_VAR, TUNING_Y, N_QUANTILE_BINS, SECTION_RANGE, SECTION_WIDTH, LICK_SMOOTH_S, MIN_PER_BIN_FRAMES, MIN_PER_BIN_TRIALS, NOISE_TRIALS_ONLY, EQUALWIDTH_RANGE), variables=BEHAVIOR_VARS, rate_variables=RATE_VARIABLES, window=PRESTIM_WINDOW,
                    d_windows=D_WINDOWS, pos_windows=POS_WINDOWS),
    load_kwargs=dict(load_behaviordata=True, load_videodata=True),
    n_jobs=N_JOBS, recompute=args.recompute, verbose=JOBLIB_VERBOSE,
    extra_key_by_session={sid: data_stamp(*triple_of[sid]) for sid in included_ids}, flatten=False)
session_results = [r for r in session_tables if r is not None and len(r["trials"])]
if not session_results:
    raise SystemExit("No selected session had behavior + video data -- nothing to analyze.")
trials = pd.concat([r["trials"] for r in session_results], ignore_index=True)
_cat = lambda k: pd.concat([r[k] for r in session_results if len(r[k])], ignore_index=True) if any(len(r[k]) for r in session_results) else pd.DataFrame()
tuning_all, spearman_all = _cat("tuning"), _cat("spearman")
del session_tables, session_results
trials_all = trials                    # every engaged trial (hit / FA / balanced accuracy and the classical d' need the 0 % / 100 % ones)
FIT_PARAMS = fits_df.set_index("session_id")[["mu", "sigma", "lapse_rate", "guess_rate"]].to_dict("index")
_noise_all = trials_all[(trials_all["signal"] > 0) & (trials_all["signal"] < 100)]
SPLIT_AT = _noise_all.groupby("session_id")["signal"].median().to_dict()      # weak / strong split of the noise trials, per session
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


def _fit_half(z, yy):
    """Quadratic and linear OLS fit on one half of the trials. Returns (beta_quad[3], beta_lin[2])."""
    X = np.column_stack([np.ones_like(z), z, z ** 2])
    bq = np.linalg.lstsq(X, yy, rcond=None)[0]
    bl = np.linalg.lstsq(X[:, :2], yy, rcond=None)[0]
    return bq, bl


def _score_half(bq, bl, z, yy):
    """Held-out SSE (quadratic, linear model) and SST of yy at z, using coefficients fit on the OTHER half."""
    X = np.column_stack([np.ones_like(z), z, z ** 2])
    sse_q = float(np.sum((yy - X @ bq) ** 2))
    sse_l = float(np.sum((yy - X[:, :2] @ bl) ** 2))
    sst = float(np.sum((yy - yy.mean()) ** 2))
    return sse_q, sse_l, sst


def fit_cv(z, yy, odd):
    """2-FOLD (odd/even trials) CROSS-VALIDATED quadratic-vs-linear fit of yy ~ z (see module docstring). Fit on
    one half, score the OTHER half, in both directions, pooling SSE/SST before computing R2 -- the same pattern
    as 2j_pupil_activity_shape.py's fit_cv. Returns the CV-averaged coefficients (mean of the two half-fits, never
    fit on the trials they are tested against), the pooled held-out R2 of each model, their difference dR2 (can
    be negative), and cv_gain (the SSE-ratio version of the same comparison, kept for continuity)."""
    bq_o, bl_o = _fit_half(z[odd], yy[odd])
    bq_e, bl_e = _fit_half(z[~odd], yy[~odd])
    sse_q1, sse_l1, sst1 = _score_half(bq_o, bl_o, z[~odd], yy[~odd])
    sse_q2, sse_l2, sst2 = _score_half(bq_e, bl_e, z[odd], yy[odd])
    sst = sst1 + sst2
    sse_q, sse_l = sse_q1 + sse_q2, sse_l1 + sse_l2
    with np.errstate(invalid="ignore", divide="ignore"):
        r2q = 1.0 - sse_q / sst
        r2l = 1.0 - sse_l / sst
        cvg = 1.0 - sse_q / sse_l if sse_l > 0 else np.nan
    return dict(beta1=(bq_o[1] + bq_e[1]) / 2.0, beta2=(bq_o[2] + bq_e[2]) / 2.0,
                dR2=r2q - r2l, cv_gain=cvg)


def fit_kfold(z, yy, fold, K):
    """K-FOLD generalisation of fit_cv (idea 2/3): each fold's quadratic/linear coefficients are fit on the OTHER
    K-1 folds and scored on the held-out fold, SSE/SST pooled over all K held-out scorings (idea 2: the mean of the
    K held-out losses); beta1/beta2 are the average of the K per-fold TRAINING coefficients (idea 3: the average
    curve across the subset fits). `fold` in 0..K-1. K=2 with fold=odd.astype(int) reproduces fit_cv exactly.
    Supplementary cross-check alongside the odd/even fit_cv headline, not a replacement for it."""
    bqs, bls = [], []
    for f in range(K):
        tr = fold != f
        bq, bl = _fit_half(z[tr], yy[tr])
        bqs.append(bq)
        bls.append(bl)
    sse_q = sse_l = sst = 0.0
    for f in range(K):
        te = fold == f
        sq, sl, ss = _score_half(bqs[f], bls[f], z[te], yy[te])
        sse_q += sq
        sse_l += sl
        sst += ss
    with np.errstate(invalid="ignore", divide="ignore"):
        r2q = 1.0 - sse_q / sst
        r2l = 1.0 - sse_l / sst
        cvg = 1.0 - sse_q / sse_l if sse_l > 0 else np.nan
    return dict(beta1=float(np.mean([b[1] for b in bqs])), beta2=float(np.mean([b[2] for b in bqs])),
                dR2=r2q - r2l, cv_gain=cvg)


def fit_unit(protocol, unit, sub, x, y, seed):
    d = sub[[x, f"{y}__c", y]].dropna()
    if len(d) < MIN_TRIALS_FIT or d[x].std() == 0 or d[f"{y}__c"].std() == 0:
        return None
    z = d[x].to_numpy(float)
    z = (z - z.mean()) / z.std()
    yy = d[f"{y}__c"].to_numpy(float)
    yy = (yy - yy.mean()) / yy.std()
    odd = (np.arange(len(d)) % 2 == 1)        # project convention (2j/2k): trial-index parity, not trial number
    fold5 = (np.arange(len(d)) % 5)           # supplementary 5-fold split (idea 2/3), same trial-index convention
    X = np.column_stack([np.ones(len(z)), z, z ** 2])
    beta, cov = _hc3_ols(X, yy)                # in-sample fit, kept as a REFERENCE column only (see docstring)
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

    # CROSS-VALIDATED quadratic fit (headline): fit_cv on the observed labels, then on N_SHUFFLES permutations of
    # yy (same odd/even split every time) for a null that recomputes the WHOLE cv statistic at each draw.
    obs = fit_cv(z, yy, odd)
    obs5 = fit_kfold(z, yy, fold5, 5)         # supplementary 5-fold cross-check (idea 2/3), see module docstring
    turning_cv = -obs["beta1"] / (2 * obs["beta2"]) if obs["beta2"] != 0 else np.nan
    s_lo_cv = obs["beta1"] + 2 * obs["beta2"] * lo
    s_hi_cv = obs["beta1"] + 2 * obs["beta2"] * hi
    rng = np.random.default_rng(seed)
    null_b2 = np.empty(N_SHUFFLES)
    null_lo = np.empty(N_SHUFFLES)
    null_hi = np.empty(N_SHUFFLES)
    null_dR2 = np.empty(N_SHUFFLES)
    null_b2_5 = np.empty(N_SHUFFLES)
    null_dR2_5 = np.empty(N_SHUFFLES)
    for i in range(N_SHUFFLES):
        yp = rng.permutation(yy)
        f = fit_cv(z, yp, odd)
        null_b2[i], null_dR2[i] = f["beta2"], f["dR2"]
        null_lo[i] = f["beta1"] + 2 * f["beta2"] * lo
        null_hi[i] = f["beta1"] + 2 * f["beta2"] * hi
        f5 = fit_kfold(z, yp, fold5, 5)       # same permutation draw as the 2-fold null, no extra rng calls
        null_b2_5[i], null_dR2_5[i] = f5["beta2"], f5["dR2"]
    p_quad_cv = (1 + (np.abs(null_b2) >= abs(obs["beta2"])).sum()) / (1 + N_SHUFFLES)
    p_quad_5fold = (1 + (np.abs(null_b2_5) >= abs(obs5["beta2"])).sum()) / (1 + N_SHUFFLES)
    p_dR2_5fold = (1 + (null_dR2_5 >= obs5["dR2"]).sum()) / (1 + N_SHUFFLES)
    p_lo_pos_cv = (1 + (null_lo >= s_lo_cv).sum()) / (1 + N_SHUFFLES)     # H1 (inverted U): low end slope > 0
    p_hi_neg_cv = (1 + (null_hi <= s_hi_cv).sum()) / (1 + N_SHUFFLES)     # H1 (inverted U): high end slope < 0
    p_lo_neg_cv = (1 + (null_lo <= s_lo_cv).sum()) / (1 + N_SHUFFLES)     # H1 (U shape): low end slope < 0
    p_hi_pos_cv = (1 + (null_hi >= s_hi_cv).sum()) / (1 + N_SHUFFLES)     # H1 (U shape): high end slope > 0
    p_cv_gain = (1 + (null_dR2 >= obs["dR2"]).sum()) / (1 + N_SHUFFLES)   # H1: curvature generalizes, dR2 > chance
    inv_u_cv = bool(obs["beta2"] < 0 and p_lo_pos_cv < ALPHA and p_hi_neg_cv < ALPHA and lo < turning_cv < hi)
    u_shape_cv = bool(obs["beta2"] > 0 and p_lo_neg_cv < ALPHA and p_hi_pos_cv < ALPHA and lo < turning_cv < hi)

    # two halves, split at the within-unit median of z (fixed, disjoint subsets -- no CV needed, see docstring)
    med = np.median(z)
    lowm, highm = z <= med, z > med
    slope_low = np.polyfit(z[lowm], yy[lowm], 1)[0] if lowm.sum() >= 10 else np.nan
    slope_high = np.polyfit(z[highm], yy[highm], 1)[0] if highm.sum() >= 10 else np.nan
    # MI vs linear (shared bins, bias-corrected); only when enough trials
    mi_bits = mi_lin = excess = np.nan
    r = r2 = np.nan
    if len(d) >= MIN_TRIALS_ANALYSIS:
        rng2 = np.random.default_rng(seed)
        mi = mutual_information_shuffle(d[x], d[y], bins=MI_BINS, n_shuffles=N_SHUFFLES, rng=rng2)
        lin = linear_fit(d[x], d[y])
        cmp = compare_mi_to_linear(mi.bits_corrected, lin.r2)
        mi_bits, mi_lin, r, r2 = mi.bits_corrected, cmp.mi_from_r2_bits, lin.r, lin.r2
        excess = mi_bits - mi_lin
    row = {"protocol": protocol, "x": x, "y": y, "n_trials": len(d), "beta_linear": beta[1],
           "beta_quadratic": beta[2], "p_quadratic": p_quad, "slope_low_end": s_lo, "slope_high_end": s_hi,
           "p_low_end_positive": p_lo, "p_high_end_negative": p_hi, "turning_point_z": turning,
           "range_low_z": lo, "range_high_z": hi, "inverted_U": inv_u, "U_shape": u_shape,
           "beta_linear_cv": obs["beta1"], "beta_quadratic_cv": obs["beta2"], "p_quadratic_cv": p_quad_cv,
           "slope_low_end_cv": s_lo_cv, "slope_high_end_cv": s_hi_cv, "p_low_end_positive_cv": p_lo_pos_cv,
           "p_high_end_negative_cv": p_hi_neg_cv, "turning_point_z_cv": turning_cv,
           "inverted_U_cv": inv_u_cv, "U_shape_cv": u_shape_cv,
           "slope_lower_half": slope_low, "slope_upper_half": slope_high,
           "dR2_cv": obs["dR2"], "cv_gain": obs["cv_gain"], "p_cv_gain": p_cv_gain,
           "beta_linear_5fold": obs5["beta1"], "beta_quadratic_5fold": obs5["beta2"],
           "dR2_5fold": obs5["dR2"], "cv_gain_5fold": obs5["cv_gain"],
           "p_quadratic_5fold": p_quad_5fold, "p_dR2_5fold": p_dR2_5fold,
           "r": r, "r2": r2, "mi_bits_corrected": mi_bits,
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
    tests = [("beta_quadratic_cv", "two-sided"), ("slope_lower_half", "greater"), ("slope_upper_half", "less"),
             ("cv_gain", "greater"), ("mi_excess_bits", "greater")]
    for (protocol, x), g in per_unit.groupby(["protocol", "x"]):
        for stat, alt in tests:
            recs = []
            for y, gy in g.groupby("y"):
                v = gy[stat].dropna()
                recs.append({"protocol": protocol, "x": x, "y": y, "statistic": stat, "alternative": alt,
                             "n_units": len(v), "mean": v.mean() if len(v) else np.nan,
                             "median": v.median() if len(v) else np.nan, "p": _wilcoxon(v, alt),
                             "frac_inverted_U": float(gy["inverted_U_cv"].mean())})
            ps = _holm([r["p"] for r in recs])
            for r, pa in zip(recs, ps):
                r["p_holm"] = pa
            group_rows.extend(recs)
group_tests = pd.DataFrame(group_rows)
group_tests.to_csv(table_dir / f"pupil_nonlinearity_group_tests{AGG_SUFFIX}.csv", index=False)

print(f"\nPer-unit summary (mean over {UNIT}s; beta_quadratic_cv < 0 = inverted U; cross-validated):")
summ = (per_unit.groupby(["protocol", "x", "y"])
        .agg(n_units=("n_trials", "size"), beta_quadratic_cv=("beta_quadratic_cv", "mean"),
             frac_inverted_U=("inverted_U_cv", "mean"), cv_gain=("cv_gain", "mean"),
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
            star = sig_star(sig_lookup(protocol, x, y, "beta_quadratic_cv"))
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

# idea 1: cross-validated check on figure 1 -- NOT tied to the quadratic model (that's what beta_quadratic_cv /
# dR2_cv already test numerically): here the curve itself is just the ODD trials' own bin means (whatever shape
# they take, like figure 1), plotted as a line against the EVEN-trial bin means it did not see (points). If the two
# don't resemble each other, that (protocol, y)'s tuning does not replicate on independent trials -- true regardless
# of whether the disagreement looks quadratic, linear, or anything else.
for xi, x in enumerate(xs):
    ncol = 4
    nrow = int(np.ceil(len(ys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.4 * nrow), squeeze=False, sharex=True)
    for ax, y in zip(axes.ravel(), ys):
        if y == x:
            ax.axis("off")
            continue
        for protocol in protocols_present:
            d = add_bins(trials[trials["protocol"] == protocol], x).dropna(subset=[f"{y}__c", x]).reset_index(drop=True)
            if len(d) < 2 * MIN_TRIALS_FIT:
                continue
            oddp = (np.arange(len(d)) % 2 == 1)           # project convention, pooled across the protocol's trials
            m_odd, _ = unit_curve(d[oddp], "pupil_bin", f"{y}__c", AGG)           # ODD trials' own bin means, trained
            ax.plot(m_odd.index, m_odd.values, color=PROTOCOL_COLORS[protocol], lw=1.3, alpha=0.9, zorder=2)
            m_even, sem_even = unit_curve(d[~oddp], "pupil_bin", f"{y}__c", AGG)  # EVEN-trial points, held out
            ax.errorbar(m_even.index, m_even.values, yerr=sem_even.values, color=PROTOCOL_COLORS[protocol],
                        marker="o", markersize=4, lw=0, capsize=3, alpha=0.9, zorder=3)
        ax.axhline(0, color="grey", ls="--", lw=0.8)
        ax.set_title(Y_LABELS.get(y, y), fontsize=9)
        ax.set_xticks(range(1, N_QUANTILE_BINS + 1))
        ax.set_ylabel("change from session mean", fontsize=8)
    for ax in axes.ravel()[len(ys):]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel(f"{X_LABELS.get(x, x)} quantile (within session; 1 = smallest)", fontsize=8)
    handles = [plt.Line2D([0], [0], color="grey", lw=1.3, label="odd trials' own bin-mean curve"),
               plt.Line2D([0], [0], color="grey", marker="o", lw=0, markersize=4, label="even-trial points (held out)")]
    axes[0][0].legend(handles=handles, fontsize=7, frameon=False, loc="lower left")
    fig.suptitle(f"{X_LABELS.get(x, x)}: cross-validated check on figure 1 -- the ODD trials' own tuning curve (line, "
                 f"no shape assumed) vs EVEN-trial points it did not see (idea 1; unit = {UNIT})", y=1.0)
    fig.tight_layout()
    save_fig(fig, figdir, f"1{'abcd'[xi]}_cv_fit_vs_heldout_points_{x}")
    plt.close(fig)

# d' and criterion per pupil quantile, 0 % and 100 % trials ONLY, one subplot per window (pre_stim / stim / reward); the x of each
# subplot is the pupil of THAT window (z-scored within session), quantile bins within session among all engaged trials
d01 = trials_all[trials_all["protocol"].isin(PROTOCOLS_USED)]          # ALL engaged trials (bins, hit rate, FA rate, balanced accuracy)


def _pupil_bin(v):
    """Pupil quantile bin (1..DPRIME_BINS) of the trials of ONE session (equal occupancy, ties broken in time order)."""
    if len(v) < 4 * DPRIME_BINS:
        return np.full(len(v), np.nan)
    return pd.qcut(pd.Series(v).rank(method="first"), DPRIME_BINS, labels=False).to_numpy() + 1.0


def perf_by_bins(d01, xcol):
    """Performance per (protocol, unit, pupil quantile bin of `xcol`): hit / FA / balanced accuracy on all engaged trials, d' and
    criterion on the NOISE trials (DPRIME_METHOD / CRITERION_METHOD), classical 0/100 d' for reference. Shared with 1g
    (infotheory/performance_common.py)."""
    return pc.perf_by_bins(d01, xcol, _pupil_bin, protocols=PROTOCOLS_USED, unit_keys=UNIT_KEYS, fit_params=FIT_PARAMS, split_at=SPLIT_AT,
                           dprime_method=DPRIME_METHOD, criterion_method=CRITERION_METHOD, min_per_class=MIN_PER_CLASS_BIN,
                           min_noise=MIN_NOISE_PER_BIN, sd_a=PROBIT_PRIOR_A_SD, sd_b_frac=PROBIT_PRIOR_B_FRAC, rate_clip=RATE_CLIP,
                           bin_col="pupil_bin")


dp_rows = []
for w in D_WINDOWS:
    if f"pupil_{w}" in d01.columns:
        dp_rows += [dict(r, window=w) for r in perf_by_bins(d01, f"pupil_{w}")]
dp_df = pd.DataFrame(dp_rows)
if len(dp_df):
    dp_df.to_csv(table_dir / f"pupil_performance_by_quantile_windows{AGG_SUFFIX}.csv", index=False)
    pc.plot_performance_figures(dp_df, figdir=figdir, names=("2_pupil_performance_windows", "2b_pupil_dprime_criterion_methods"),
                                windows=D_WINDOWS, protocols=PROTOCOLS_USED, unit_keys=UNIT_KEYS, unit=UNIT, n_bins=DPRIME_BINS,
                                x_name="pupil", dprime_method=DPRIME_METHOD, criterion_method=CRITERION_METHOD, bin_col="pupil_bin")


# --------------------------------------------------------------------- #
# 6. Quadratic coefficient, inverted-U fraction and MI excess per (y, protocol)
# --------------------------------------------------------------------- #
for xi, x in enumerate(xs):
    px = per_unit[per_unit["x"] == x]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
    width = 0.8 / max(len(protocols_present), 1)
    for ax, col, lab, hl in [(axes[0], "beta_quadratic_cv", "cross-validated quadratic coefficient\n(< 0 = inverted U)", 0),
                              (axes[1], "inverted_U_cv", f"fraction of {UNIT}s with a cross-validated Lind-Mehlum inverted U", None),
                              (axes[2], "mi_excess_bits", "MI - MI implied by linear r$^2$ (bits)", 0)]:
        for pi, protocol in enumerate(protocols_present):
            for yi, y in enumerate(ys):
                v = px[(px["protocol"] == protocol) & (px["y"] == y)][col].astype(float).dropna()
                if v.empty:
                    continue
                xpos = yi + (pi - (len(protocols_present) - 1) / 2) * width
                ax.bar(xpos, v.mean(), width=width * 0.9, color=PROTOCOL_COLORS[protocol], alpha=0.5,
                       edgecolor="black", linewidth=0.4, label=protocol if yi == 0 else None)
                if AGG != "pooled" and col != "inverted_U_cv" and len(v) > 1:
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
        if col == "inverted_U_cv":
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
tp = per_unit[(per_unit["x"] == "cur_pupil_area") & per_unit["inverted_U_cv"]]
width = 0.8 / max(len(protocols_present), 1)
for pi, protocol in enumerate(protocols_present):
    for yi, y in enumerate(ys):
        v = tp[(tp["protocol"] == protocol) & (tp["y"] == y)]["turning_point_z_cv"]
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
ax.set_title(f"Peak of the inverted U, {UNIT}s that pass the cross-validated Lind-Mehlum test (black = median)", fontsize=9)
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
                                 mean_beta_quadratic_cv=gp["beta_quadratic_cv"].mean(),
                                 p_quadratic=_wilcoxon(gp["beta_quadratic_cv"].dropna(), "two-sided")))
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
# (hit / FA / balanced accuracy: all engaged trials; d' / criterion: noise trials only, DPRIME_METHOD, as in figure 2): top-minus-bottom pupil bin and
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
                          mean_beta_quadratic=g["beta_quadratic_cv"].mean(),
                          sem_beta_quadratic=g["beta_quadratic_cv"].sem() if len(g) > 1 else 0.0,
                          mean_r=g["r"].mean(), sem_r=g["r"].sem() if len(g) > 1 else 0.0,
                          frac_inverted_U=float(g["inverted_U_cv"].mean()), mean_cv_gain=g["cv_gain"].mean(),
                          p_beta_quadratic=_wilcoxon(g["beta_quadratic_cv"].dropna(), "two-sided") if AGG != "pooled" else np.nan,
                          p_r=_wilcoxon(g["r"].dropna(), "two-sided") if AGG != "pooled" else np.nan))
    pos_group = pd.DataFrame(grows)
    for col in ("p_beta_quadratic", "p_r"):
        pos_group[col + "_holm"] = np.nan
        for _, idx in pos_group.groupby(["protocol", "y"]).groups.items():
            pos_group.loc[idx, col + "_holm"] = _holm(pos_group.loc[idx, col].to_numpy())
    pos_group.to_csv(table_dir / f"pupil_position_windows_group{AGG_SUFFIX}.csv", index=False)

    def shade_untrusted(ax):
        lo_t, hi_t = TRUSTED_POSITION_RANGE
        ax.axvspan(-30, -10, color="grey", alpha=0.12, lw=0)
        ax.axvspan(0, 20, color="grey", alpha=0.12, lw=0)
        ax.axvspan(25, 45, color="grey", alpha=0.12, lw=0)
        ax.axvline(0, color="black", ls=":", lw=0.8)
        ax.set_xlim(-40, 50)

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
                  "cross-validated quadratic coefficient (< 0 = inverted U)", 0.0),
                 ("10b_pupil_effect_by_position_linear", "mean_r", "sem_r", "p_r_holm", "linear r with pupil area", 0.0),
                 ("10c_pupil_inverted_u_by_position", "frac_inverted_U", None, None,
                  f"fraction of {UNIT}s with a cross-validated Lind-Mehlum inverted U", None)]
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
    MEAS = [("hit_rate", "hit rate"), ("fa_rate", "FA rate"), ("balanced_accuracy", "balanced accuracy"), ("dprime", "d' (noise)"),
            ("criterion", "criterion (noise)")]
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
        fig.suptitle(f"Performance vs pupil area by position window (hit / FA / balanced accuracy: all engaged trials; d' / criterion: noise trials, {DPRIME_METHOD} / {CRITERION_METHOD}; {DPRIME_BINS} pupil bins; unit = {UNIT}; "
                     f"circle = Holm-corrected Wilcoxon p < {ALPHA})", y=1.0, fontsize=9)
        fig.tight_layout()
        save_fig(fig, figdir, "10d_pupil_performance_by_position")
        plt.close(fig)

# --------------------------------------------------------------------- #
# 11. Tuning curves vs pupil area -- the mirror of 1g (everything vs speed): same y variables, levels, bins and figures
# --------------------------------------------------------------------- #
#   figure 11  whole trial, POOLED FRAMES (every time sample of the selected trials)
#   figure 12  per-trial window means (TUNING_WINDOWS) and P(lick)
#   figure 13  whole trial as SECTION means (SECTION_WIDTH cm, SECTION_RANGE): the independent-sample version of figure 11
# Bins = N_QUANTILE_BINS quantiles of the pupil WITHIN session (no stationary bin: pupil is not zero-inflated). Each curve = the mean over
# units of the unit's bin means (+- SEM); pooled = sample-weighted over every session, no error bars.
if len(tuning_all):
    tuning_curves = tc.unit_curves(tuning_all, UNIT_KEYS, AGG, "pupil_area")
    tuning_curves.to_csv(table_dir / f"pupil_tuning_curves{AGG_SUFFIX}.csv", index=False)
    if len(spearman_all):
        tuning_corr = tc.corr_summary(spearman_all, AGG)
        tuning_corr.to_csv(table_dir / f"pupil_tuning_correlations{AGG_SUFFIX}.csv", index=False)
    else:
        tuning_corr = pd.DataFrame()

    def _rho(protocol, y):
        if tuning_corr.empty:
            return np.nan
        m = tuning_corr[(tuning_corr["protocol"] == protocol) & (tuning_corr["var_b"] == y)]
        return float(m["rho"].iloc[0]) if len(m) else np.nan

    tc.plot_tuning_figures(
        tuning_curves, figdir=figdir, names=("11_pupil_tuning_whole_trial", "12_pupil_tuning_by_window", "13_pupil_tuning_sections"),
        x_col="pupil_area", x_label="pupil area (z)", x_name="pupil area",
        protocols=[q for q in PROTOCOLS_USED if (tuning_curves["protocol"] == q).any()], agg=AGG, unit=UNIT,
        ys=[y for y in TUNING_Y if (tuning_curves["y"] == y).any()], windows=TUNING_WINDOWS, rho_lookup=_rho,
        bins_note=f"Pupil bins: {N_QUANTILE_BINS} quantiles within session",
        trial_note="noise trials (0 < signal < 100 %)" if NOISE_TRIALS_ONLY else "all engaged trials",
        section_range=SECTION_RANGE, section_width=SECTION_WIDTH)
    # the same figures with the QUANTILE (bin number) on the x axis instead of the mean pupil of the bin
    tc.plot_tuning_figures(
        tuning_curves, figdir=figdir, names=("11q_pupil_tuning_whole_trial_by_quantile", "12q_pupil_tuning_by_window_by_quantile",
                                             "13q_pupil_tuning_sections_by_quantile"),
        x_col="pupil_area", x_label="pupil area (z)", x_name="pupil",
        protocols=[q for q in PROTOCOLS_USED if (tuning_curves["protocol"] == q).any()], agg=AGG, unit=UNIT,
        ys=[y for y in TUNING_Y if (tuning_curves["y"] == y).any()], windows=TUNING_WINDOWS, rho_lookup=_rho,
        bins_note=f"Pupil bins: {N_QUANTILE_BINS} quantiles within session",
        trial_note="noise trials (0 < signal < 100 %)" if NOISE_TRIALS_ONLY else "all engaged trials",
        section_range=SECTION_RANGE, section_width=SECTION_WIDTH, x_mode="quantile", stop_bin=False)
    # the same figures again with FIXED, EQUAL-WIDTH pupil bins (same z range in every session) instead of the default
    # within-session quantiles -- does the curve's shape depend on which binning scheme is used? (older figures above
    # are unchanged; this is an additional view, not a replacement)
    tc.plot_tuning_figures(
        tuning_curves, figdir=figdir, names=("11w_pupil_tuning_whole_trial_equal_width", "12w_pupil_tuning_by_window_equal_width",
                                             "13w_pupil_tuning_sections_equal_width"),
        x_col="pupil_area", x_label="pupil area (z)", x_name="pupil area",
        protocols=[q for q in PROTOCOLS_USED if (tuning_curves["protocol"] == q).any()], agg=AGG, unit=UNIT,
        ys=[y for y in TUNING_Y if (tuning_curves["y"] == y).any()], windows=TUNING_WINDOWS, rho_lookup=_rho,
        bins_note=f"Pupil bins: {N_QUANTILE_BINS} FIXED equal-width bins over z in [{EQUALWIDTH_RANGE[0]:g}, {EQUALWIDTH_RANGE[1]:g}) "
                  f"(same range in every session, unlike the within-session quantiles of figure 11-13)",
        trial_note="noise trials (0 < signal < 100 %)" if NOISE_TRIALS_ONLY else "all engaged trials",
        section_range=SECTION_RANGE, section_width=SECTION_WIDTH, scheme="equal_width")
else:
    print("\nTuning curves vs pupil: no session had samples in the selected trials.")

print(f"\nWrote tables to {table_dir}, figures to {figdir}")
