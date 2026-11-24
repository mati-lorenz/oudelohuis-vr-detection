# -*- coding: utf-8 -*-
"""
2k_neuron_tuning_curves
=======================
TUNING CURVES of single neurons to behavioral and task variables, per cell, then summarised per group (the six project groups
+ deep PM, always all shown). DN sessions; only NOISE trials (0 < signal < 100 %) of the engaged trials (NOISE_TRIALS_ONLY).

Activity = the cell's deconvolved activity, z-scored per cell over the analysed frames (SD units), as in 2j.

FRAME-LEVEL variables (every imaging frame of the selected trials inside [-S_PRE, +S_POST) cm of the stimulus onset;
behavior is brought onto the imaging clock; the curve = mean activity in each bin of the variable):
    position      position in the corridor relative to the stimulus onset, fixed POSITION_BIN_CM bins over POSITION_RANGE (cm)
    speed         running speed (cm/s): bin 0 = stationary (<= STOP_SPEED), the others = equal-occupancy quantiles of the moving frames
    pupil_area    pupil area (blink outliers removed, z-scored within the session), N_BINS quantile bins
    lick_rate     lick events smoothed with a LICK_SMOOTH_S boxcar (Hz): bin 0 = no licking, the others = quantiles of the licking frames
    motionenergy  video motion energy (z-scored within the session), N_BINS quantile bins
    Quantile bins are defined WITHIN each session, so every session spans the same percentiles of its own range (the x axis is a
    percentile of the range, not an absolute value; tick labels give the typical value of each bin).
TRIAL-LEVEL variables (one value per trial; the response = the cell's mean activity of the trial in a position window, z-scored
across the trials, so these curves are in SD of trial responses and are not comparable in size to the frame-level ones):
    signal        signal strength = the shared stimulus binning (SIMPLE_STRATEGIES: 4 bins of the psychometric curve), STIM window
    p_lick        lick / no lick of the trial (the task's lick response; P(lick) is the height of the 'lick' bin), REWARD window

PER CELL AND VARIABLE (tuning_stats_per_cell.csv)
    eta2          fraction of the cell's activity variance explained by the bins of the variable (biased upwards for few frames)
    eta2_cv       CROSS-VALIDATED effect size, the number used in the figures/tests: the bin means (the tuning curve) are
                  fit on ONE half of the trials (odd / even) and scored (held-out R^2, SSE/SST pooled over both directions)
                  on the OTHER half. Unlike plain in-sample eta2 (fit and scored on the same data, kept alongside for
                  reference -- it is always >= eta2_cv), eta2_cv is not inflated by having many bins or few frames per bin,
                  and comes out <= 0 when the curve does not replicate on independent trials.
    eta2_cv5      supplementary 5-FOLD version of eta2_cv (p_value5 the matching p-value): the curve is fit on 4/5 of
                  the trials and scored on the held-out 1/5, SSE/SST pooled over the 5 held-out scorings -- the same
                  logic as eta2_cv generalised from 2 to 5 folds, kept as a cross-check (the two should agree, not a
                  replacement for the odd/even eta2_cv used in the figures). mean_cv5 (tuning_curves_per_cell.csv) is
                  the average curve across the 5 per-fold fits.
    p_value       null = CIRCULAR SHIFT of the variable against the activity (frame-level: both signals keep their autocorrelation,
                  frames are not independent, a plain test would be far too liberal) / PERMUTATION of the trials (trial-level);
                  the null is recomputed with the SAME fit-on-one-half/score-on-the-other procedure at every draw, so it tests
                  eta2_cv itself. tuned = p_value < ALPHA (about 5 % of cells by chance).
    modulation    max - min of the curve (SD units; biased up: picking the extreme BINS and reading their value off the SAME
                  data inflates it). modulation_cv = the cross-validated counterpart (peak/trough bin picked from one half,
                  value read off the other, both directions averaged -- same logic as figure 5's cross-validated heatmap
                  sorting, as a per-cell scalar); diff_last_first = last bin - first bin (signed, not cross-validated: a FIXED
                  pair of bins, not a data-dependent selection, so it carries no such bias)
    split_half_r  correlation of the curves of the ODD and EVEN trials (same bin edges); NaN with fewer than 3 bins.
                  diff_h1 / diff_h2 = the signed last-first difference in each half (for 2-bin variables: do the halves agree?)
    peak_bin      bin of the maximum of the curve (in-sample; for display / sorting only, see figure 5)
SCOPES (edit SCOPES / PRESTIM_WINDOW at the top): everything is run twice and the figures / tables carry the scope.
    all      = frames within [-S_PRE, +S_POST) cm; signal curve from the stim window, lick curve from the reward window
    prestim  = ONLY the pre-stim window (-30..-10 cm): frame-level curves from those frames (position in 5 cm bins), and the
               pre-stim window is also the response window of the trial-level variables. There the stimulus has not appeared:
               the signal curve is a control (should be flat), the lick curve is anticipatory activity.
    stim / reward = the same restricted to the stim window (0..20 cm) / the reward window (25..45 cm), both trial-level
               variables taking that window as response window (signal in the reward window = does the reward-window
               activity still carry the stimulus; lick in the stim window = does stim-window activity predict the later lick).
    Figures of the restricted scopes have the suffix _prestim / _stim / _reward. Figure 6_by_group_<variable> = one figure per variable, one subplot per group
    (thin = sessions / mice / example cells, thick = group mean), y shared.
Group level: fraction of tuned cells, eta2_cv and split-half reliability per group at the chosen unit; mean tuning curve per
group (mean over units of the unit's mean over its cells, band = SEM over units; pooled: SEM over cells), plotted together
with the ODD-trained fit (line) vs EVEN-trial points (markers) as a visual cross-check (figure 1b); per-cell heatmaps
SORTED ON THE ODD TRIALS AND SHOWN ON THE EVEN TRIALS (cross-validated: a sorted heatmap of the same data would show tuning in pure noise).

Project-wide rules
    * selection : --animals / --sessions restrict the sessions (a restricted run writes to out/selection/<tag>/);
                  --max-cells caps the cells per group and session.
    * aggregate : --aggregate pooled|session|mouse = the unit of the group bar plots and of the mean-curve band.
    * memory    : one session in RAM per worker; per-session results cached in store/ (no ledger).
    * cells     : 1b-included sessions; cells passing 2a (proximity, depth rule) and 2b QC.

Reads: 1b psychometric_fits.csv, 2a celldata_combined.csv, 2b qc_per_cell.csv.  Loads: raw behavior / video / calcium, one session at a time.

Outputs
-------
2_pipeline/2k_neuron_tuning_curves/
    out/    tuning_curves_per_cell.csv     one row per (cell, variable, bin): n, mean, mean of odd / even trials, mean_cv5
            tuning_stats_per_cell.csv      one row per (cell, variable): see above, + group
            tuning_bins.csv                typical value (median) of each bin per session and variable
            tuning_by_group[__<aggregate>].csv        n_cells, mean eta2_cv / eta2_cv5 / modulation_cv / split-half r, fraction tuned per group x variable (x unit)
            tuning_unit_curves[__<aggregate>].csv     mean curve per group x variable x bin (+ SEM, n units)
            figures/1_mean_tuning_curves.png, 1b_cv_fit_vs_heldout_points.png (idea 1), 2_fraction_tuned.png,
                    3_variance_explained.png, 4_split_half_reliability.png, 5_heatmap_<variable>.png (cross-validated)
    out/selection/<tag>/...  the same for a restricted run
    store/  tuning_session_<hash>.pkl      per-session cache (reused unless --recompute)
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.stats import rankdata

from infotheory import project_config as cfg
from infotheory.pipeline import get_pipeline_paths
from infotheory.session import discover_sessions
from infotheory.neural_encoding import load_cell_inclusion
from infotheory.continuous import merge_behavior_video, remove_pupil_outliers, zscore_per_session
from infotheory.celldata_utils import SIX_GROUP_ORDER, SIX_GROUP_LABELS, SIX_GROUP_COLORS
from infotheory.groups import group_key
from infotheory.plotting import set_style, save_fig, clear_figures
from infotheory.scope import add_scope_args, scope_from_args, select_sessions, scoped_dirs, data_stamp, unit_noun
from infotheory.session_jobs import run_sessions_units
from infotheory.unit_plots import group_panels, mouse_palette
from infotheory.stim_binning import SIMPLE_STRATEGIES, psychometric_pct
from infotheory import tensors as tn
from infotheory import cell_tables as ct

PROTOCOL = "DN"
NOISE_TRIALS_ONLY = True               # True: only the noise trials (0 < signal < 100 %) of the engaged trials; False: all engaged trials

# --- what is analysed (edit freely) ------------------------------------------------------------------------
VARIABLES = ["position", "speed", "pupil_area", "lick_rate", "motionenergy", "signal", "p_lick"]
VAR_LABELS = {"position": "position rel. to stimulus (cm)", "speed": "running speed (cm/s)", "pupil_area": "pupil area (z)",
              "lick_rate": "lick rate (Hz, smoothed)", "motionenergy": "motion energy (z)",
              "signal": "signal strength (psychometric bin, %)", "p_lick": "lick response (reward window)"}
FRAME_VARIABLES = ["position", "speed", "pupil_area", "lick_rate", "motionenergy"]
TRIAL_VARIABLES = ["signal", "p_lick"]
S_PRE, S_POST = 40.0, 60.0             # frames inside [-S_PRE, +S_POST) cm of stimStart are used (beyond ~ +-40 cm: neighbouring trials, see 2h)
POSITION_RANGE = (-40.0, 60.0)         # cm; fixed bins of POSITION_BIN_CM
POSITION_BIN_CM = 10.0
PRESTIM_WINDOW = (-30.0, -10.0)        # cm rel. to stimStart: the pre-stim window of 2c; the "prestim" scope uses only these frames
PRESTIM_POSITION_BIN_CM = 5.0
N_BINS = 8                             # quantile bins of speed / pupil / lick rate / motion energy (including the stationary / no-lick bin)
STOP_SPEED = 1.0                       # cm/s: speed <= this = stationary bin
LICK_SMOOTH_S = 1.0                    # lick events -> lick rate (Hz): boxcar of this width
SIGNAL_WINDOW = (0.0, 20.0)            # cm rel. to stimStart: response window of the signal-strength curve
PLICK_WINDOW = (25.0, 45.0)            # cm rel. to stimStart: response window of the lick / no-lick curve (reward window)
STIM_STRATEGY = SIMPLE_STRATEGIES[0]   # shared stimulus binning
MIN_FRAMES = 400                       # analysed frames needed in a session
MIN_FRAMES_PER_BIN = 30                # frames needed per bin of a frame-level variable (else the variable is skipped in that session)
MIN_TRIALS = 30                        # trials needed for a trial-level variable
MIN_TRIALS_PER_BIN = 5
N_SHIFTS = 100                         # circular shifts / trial permutations of the null
MIN_SHIFT_S = 20.0
ALPHA = 0.05
RANDOM_STATE = 0
N_EXAMPLE_CELLS = 40                   # pooled mode, figure 6: thin lines = this many random cells per group
MIN_CELLS_PER_UNIT = 5                 # cells in a group needed for a session / mouse value
MAX_HEATMAP_CELLS = 400                # cells drawn per group in the heatmaps (evenly spaced in the sorted order)
JOBLIB_VERBOSE = 5

# SCOPES: the same analysis is run for each. "all" = every frame within [-S_PRE, +S_POST) cm, signal response in the stim window,
# lick response in the reward window. "prestim" = ONLY frames of the pre-stim window (frame-level curves) and the pre-stim window
# as response window of both trial-level variables (the stimulus has not appeared yet there: the signal curve is a control that
# should be flat, the lick curve is anticipatory activity).
SCOPES = {
    "all": dict(label="whole window", frame_range=(-S_PRE, S_POST), pos_range=POSITION_RANGE, pos_bin=POSITION_BIN_CM,
                sig_win=SIGNAL_WINDOW, lick_win=PLICK_WINDOW),
    "prestim": dict(label="pre-stim window only", frame_range=PRESTIM_WINDOW, pos_range=PRESTIM_WINDOW,
                    pos_bin=PRESTIM_POSITION_BIN_CM, sig_win=PRESTIM_WINDOW, lick_win=PRESTIM_WINDOW),
    "stim": dict(label="stim window only", frame_range=SIGNAL_WINDOW, pos_range=SIGNAL_WINDOW,
                 pos_bin=PRESTIM_POSITION_BIN_CM, sig_win=SIGNAL_WINDOW, lick_win=SIGNAL_WINDOW),
    "reward": dict(label="reward window only", frame_range=PLICK_WINDOW, pos_range=PLICK_WINDOW,
                   pos_bin=PRESTIM_POSITION_BIN_CM, sig_win=PLICK_WINDOW, lick_win=PLICK_WINDOW),
}

parser = argparse.ArgumentParser()
add_scope_args(parser)
parser.add_argument("--n-shifts", type=int, default=N_SHIFTS)
args = parser.parse_args()
scope = scope_from_args(args)
AGG = scope.aggregate
UNIT = unit_noun(AGG)
UNITS = "mice" if UNIT == "mouse" else f"{UNIT}s"
AGG_SUFFIX = "" if AGG == "pooled" else f"__{AGG}"
n_shifts = args.n_shifts
UNIT_KEYS = {"pooled": [], "session": ["animal_id", "session_id"], "mouse": ["animal_id"]}[AGG]
TRIAL_NOTE = "noise trials (0 < signal < 100 %)" if NOISE_TRIALS_ONLY else "all engaged trials"

paths = get_pipeline_paths(__file__)
table_dir, figdir = scoped_dirs(paths, scope)
clear_figures(figdir)
set_style()
print(scope.describe())

# --------------------------------------------------------------------- #
# 1. Included sessions (1b) and cells (2a + 2b)
# --------------------------------------------------------------------- #
fits_path = paths.out_from("1b_psychometric") / "psychometric_fits.csv"
cell_dist_out = paths.out_from("2a_cell_distribution")
activity_out = paths.out_from("2b_activity_statistics")
for f, hint in [(fits_path, "1b_psychometric.py"), (cell_dist_out / "celldata_combined.csv", "2a_cell_distribution.py"),
                (activity_out / "qc_per_cell.csv", "2b_activity_statistics.py")]:
    if not f.exists():
        raise SystemExit(f"Run {hint} first -- expected {f}")
fits_df = pd.read_csv(fits_path)
mu_sigma_lookup = fits_df.set_index("session_id")[["mu", "sigma"]].to_dict("index")
included_all = set(fits_df[fits_df["included"] & (fits_df["protocol"] == PROTOCOL)]["session_id"])
all_triples = discover_sessions([PROTOCOL])
triples = select_sessions(all_triples, scope)
selected = {f"{a}_{d}": (p, a, d) for p, a, d in triples}
session_ids = [s for s in selected if s in included_all]
if not session_ids:
    raise SystemExit("None of the selected sessions passed 1b_psychometric's inclusion criteria.")
mouse_colors = mouse_palette(a for _, a, _ in all_triples)
cell_inclusion = load_cell_inclusion(cell_dist_out, activity_out)
cell_group = dict(zip(cell_inclusion["cell_id"], cell_inclusion["group"]))      # per-group cell caps (--max-cells)
cells_by_session = {sid: sorted(sub.loc[sub["include"], "cell_id"]) for sid, sub in cell_inclusion.groupby("session_id")
                    if sid in session_ids}
session_ids = [s for s in session_ids if cells_by_session.get(s)]
if not session_ids:
    raise SystemExit("No included cells in the selected sessions.")
print(f"{len(session_ids)} sessions, {sum(len(v) for v in cells_by_session.values())} included cells")
stamps = {s: data_stamp(*selected[s]) for s in session_ids}
qc_mtime = int((activity_out / "qc_per_cell.csv").stat().st_mtime)

try:
    STIM_LABELS = [f"{a}%" for a in STIM_STRATEGY.split("_")[1:]]
except Exception:
    STIM_LABELS = []
TRIAL_LABELS = {"signal": STIM_LABELS or ["bin 1", "bin 2", "bin 3", "bin 4"], "p_lick": ["no lick", "lick"]}


# --------------------------------------------------------------------- #
# 2. Per-session computation (one session per worker, cached)
# --------------------------------------------------------------------- #
def fixed_bins(x, ok, lo, hi, step):
    """Fixed-width bins of x over [lo, hi): (bin index or -1, n bins, bin centres)."""
    nb = int(round((hi - lo) / step))
    v = ok & np.isfinite(x) & (x >= lo) & (x < hi)
    b = np.where(v, np.clip(np.floor((np.nan_to_num(x) - lo) / step).astype(int), 0, nb - 1), -1)
    return b, nb, lo + (np.arange(nb) + 0.5) * step


def quantile_bins(x, ok, n_bins, zero_thresh=None):
    """Within-session bins: with `zero_thresh`, bin 0 = x <= zero_thresh (stationary / no licking) and the other n_bins-1 bins are
    equal-occupancy quantiles (rank-based, ties broken in time order) of the remaining values; otherwise n_bins quantiles."""
    x = np.asarray(x, float)
    v = ok & np.isfinite(x)
    b = np.full(len(x), -1, dtype=int)
    if zero_thresh is not None:
        z = v & (x <= zero_thresh)
        b[z] = 0
        mov, first, nq = v & ~z, 1, n_bins - 1
    else:
        mov, first, nq = v, 0, n_bins
    n = int(mov.sum())
    if n >= nq:
        r = rankdata(x[mov], method="ordinal")
        b[mov] = first + np.minimum((r - 1) * nq // n, nq - 1)
    centers = np.array([np.median(x[b == i]) if (b == i).any() else np.nan for i in range(n_bins)])
    return b, n_bins, centers


def tune_frames(Y, Y2, b, nb, odd, fold5, ok, shifts):
    """Bin means (all / odd / even trials), eta2 (in-sample, kept for reference) and eta2_cv + p-value of every column
    of Y (T, N) for the bin index b (T,) (-1 = not used).

    eta2_cv is the CROSS-VALIDATED variance explained: the bin means are fit on ONE half of the trials (odd or even)
    and scored (held-out SSE / SST) on the OTHER half, pooled over both directions -- unlike the plain in-sample eta2
    (`Sb`/`cnt` fit and scored on the SAME frames), this cannot be inflated just by having many bins or few frames per
    bin, and can come out negative (the model generalizes worse than the test half's own mean) when a curve does not
    replicate. The null (p-value) is the circular shift of b against the activity, recomputed with the SAME
    fit-on-one-half/score-on-the-other procedure at each shift, so p tests whether eta2_cv itself -- not the biased
    in-sample eta2 -- is bigger than chance.

    eta2_cv5 is the same idea generalised to 5 folds (`fold5` in 0..4, trial-aligned like `odd`): each fold's curve
    is fit on the OTHER 4 folds and scored on the held-out fold, SSE/SST pooled over all 5 held-out scorings (idea 2:
    mean of the 5 held-out losses); curve_cv5 = the average of the 5 per-fold training curves (idea 3). Kept as a
    supplementary cross-check alongside the odd/even eta2_cv, not a replacement for it."""
    T = len(b)

    def sums(bb):
        idx = np.where(bb >= 0)[0]
        B = sparse.csr_matrix((np.ones(len(idx)), (bb[idx], idx)), shape=(nb, T))
        return B @ Y, np.asarray(B.sum(1)).ravel(), (bb >= 0)

    def kfold_cv(bb, fold, K):
        """K-fold generalisation of halves_cv below: trains each fold's curve on the OTHER K-1 folds, scores on the
        held-out fold, pools SSE/SST over all K foldings. Returns (average curve across the K per-fold fits, eta2_cv)."""
        curves_ = []
        for f in range(K):
            S_, c_, _ = sums(np.where(fold != f, bb, -1))
            with np.errstate(invalid="ignore", divide="ignore"):
                curves_.append(np.where(c_[:, None] > 0, S_ / c_[:, None], np.nan))
        sse, sst = np.zeros(Y.shape[1]), np.zeros(Y.shape[1])
        for f in range(K):
            idx = np.where((fold == f) & (bb >= 0))[0]
            if len(idx) == 0:
                continue
            pred, yt = curves_[f][bb[idx]], Y[idx]
            sse += np.nansum((yt - pred) ** 2, axis=0)
            sst += np.nansum((yt - np.nanmean(yt, axis=0)) ** 2, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            eta2_ = 1 - sse / sst
        return np.nanmean(np.stack(curves_), axis=0), eta2_

    def eta(Sb, cnt, valid):
        vf = valid.astype(float)
        n = vf.sum()
        tot, tot2 = vf @ Y, vf @ Y2
        ssb = (Sb ** 2 / np.maximum(cnt, 1)[:, None]).sum(0) - tot ** 2 / n
        sst = tot2 - tot ** 2 / n
        with np.errstate(invalid="ignore", divide="ignore"):
            return ssb / sst

    def halves_cv(bb):
        """(h1, h2, eta2_cv) for bin assignment bb: h1 = bin means fit on ODD trials, h2 = fit on EVEN trials;
        eta2_cv = pooled held-out R^2 of predicting EVEN frames from h1 and ODD frames from h2."""
        hh = []
        for sel in (odd, ~odd):
            S_, c_, _ = sums(np.where(sel, bb, -1))
            with np.errstate(invalid="ignore", divide="ignore"):
                hh.append(np.where(c_[:, None] > 0, S_ / c_[:, None], np.nan))
        h1_, h2_ = hh
        sse, sst = np.zeros(Y.shape[1]), np.zeros(Y.shape[1])
        for train_curve, test_sel in ((h1_, ~odd), (h2_, odd)):
            idx = np.where(test_sel & (bb >= 0))[0]
            if len(idx) == 0:
                continue
            pred, yt = train_curve[bb[idx]], Y[idx]
            sse += np.nansum((yt - pred) ** 2, axis=0)
            sst += np.nansum((yt - np.nanmean(yt, axis=0)) ** 2, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return h1_, h2_, 1 - sse / sst

    Sb, cnt, valid = sums(b)
    with np.errstate(invalid="ignore", divide="ignore"):
        curve = np.where(cnt[:, None] > 0, Sb / cnt[:, None], np.nan)
    obs_eta2 = eta(Sb, cnt, valid)
    h1, h2, obs_cv = halves_cv(b)
    null_cv = np.vstack([halves_cv(np.where(ok, np.roll(b, s), -1))[2] for s in shifts])
    p = (1 + (null_cv >= obs_cv).sum(0)) / (1 + len(shifts))
    curve_cv5, obs_cv5 = kfold_cv(b, fold5, 5)
    null_cv5 = np.vstack([kfold_cv(np.where(ok, np.roll(b, s), -1), fold5, 5)[1] for s in shifts])
    p5 = (1 + (null_cv5 >= obs_cv5).sum(0)) / (1 + len(shifts))
    return dict(curve=curve, h1=h1, h2=h2, n=cnt, eta2=obs_eta2, eta2_cv=obs_cv, p=p, curve_cv5=curve_cv5,
                eta2_cv5=obs_cv5, p5=p5, n_samples=int(valid.sum()))


def tune_trials(R, xb, nb, odd, fold5, rng, n_perm):
    """Same for trial-level data: R (n_trials, N) trial responses, xb (n_trials,) bin index (-1 = not used).
    eta2_cv / p as in tune_frames, but the null is a permutation of the trial-bin labels (as before), and the h1/h2
    bin means used for prediction are recomputed from the SAME permuted labels at every draw (not just the pooled
    in-sample eta2), so the null matches exactly what the observed eta2_cv does. eta2_cv5 / p5 = the 5-fold
    generalisation (idea 2/3), `fold5` in 0..4 kept fixed while the bin labels are permuted, same logic as tune_frames."""
    valid = (xb >= 0) & np.isfinite(R).all(1)
    Rv, xv, ov, fv = R[valid], xb[valid], odd[valid], fold5[valid]
    mu, sd = Rv.mean(0), Rv.std(0)
    sd = np.where(sd > 0, sd, np.nan)
    Z = (Rv - mu) / sd
    cnt = np.bincount(xv, minlength=nb).astype(float)
    if (cnt < MIN_TRIALS_PER_BIN).any() or valid.sum() < MIN_TRIALS:
        return None

    def means(Zs, xs):
        return np.vstack([Zs[xs == i].mean(0) if (xs == i).any() else np.full(Zs.shape[1], np.nan) for i in range(nb)])

    def eta(xs):
        S = np.vstack([Z[xs == i].sum(0) for i in range(nb)])
        c = np.array([(xs == i).sum() for i in range(nb)], float)
        tot = Z.sum(0)
        n = len(xs)
        ssb = (S ** 2 / np.maximum(c, 1)[:, None]).sum(0) - tot ** 2 / n
        sst = (Z ** 2).sum(0) - tot ** 2 / n
        with np.errstate(invalid="ignore", divide="ignore"):
            return ssb / sst

    def eta_cv(xs):
        """h1 (fit on odd trials), h2 (fit on even trials) of label vector xs, then the pooled held-out R^2."""
        h1_, h2_ = means(Z[ov], xs[ov]), means(Z[~ov], xs[~ov])
        sse, sst = np.zeros(Z.shape[1]), np.zeros(Z.shape[1])
        for train_curve, test_sel in ((h1_, ~ov), (h2_, ov)):
            idx = np.where(test_sel)[0]
            pred, yt = train_curve[xs[idx]], Z[idx]
            sse += np.nansum((yt - pred) ** 2, axis=0)
            sst += np.nansum((yt - np.nanmean(yt, axis=0)) ** 2, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return h1_, h2_, 1 - sse / sst

    def kfold_eta(xs, K):
        """5-fold generalisation of eta_cv: trains each fold's bin means on the OTHER K-1 folds (by `fv`), scores on
        the held-out fold; returns (average curve across the K per-fold fits, eta2_cv)."""
        curves_ = [means(Z[fv != f], xs[fv != f]) for f in range(K)]
        sse, sst = np.zeros(Z.shape[1]), np.zeros(Z.shape[1])
        for f in range(K):
            idx = np.where(fv == f)[0]
            pred, yt = curves_[f][xs[idx]], Z[idx]
            sse += np.nansum((yt - pred) ** 2, axis=0)
            sst += np.nansum((yt - np.nanmean(yt, axis=0)) ** 2, axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nanmean(np.stack(curves_), axis=0), 1 - sse / sst

    obs = eta(xv)
    h1, h2, obs_cv = eta_cv(xv)
    null_cv = np.vstack([eta_cv(rng.permutation(xv))[2] for _ in range(n_perm)])
    p = (1 + (null_cv >= obs_cv).sum(0)) / (1 + n_perm)
    curve_cv5, obs_cv5 = kfold_eta(xv, 5)
    null_cv5 = np.vstack([kfold_eta(rng.permutation(xv), 5)[1] for _ in range(n_perm)])
    p5 = (1 + (null_cv5 >= obs_cv5).sum(0)) / (1 + n_perm)
    return dict(curve=means(Z, xv), h1=h1, h2=h2, n=cnt, eta2=obs, eta2_cv=obs_cv, p=p, curve_cv5=curve_cv5,
                eta2_cv5=obs_cv5, p5=p5, n_samples=int(valid.sum()), good=np.isfinite(sd))


def extreme_cv(train, test):
    """Cross-validated max-min: for each cell, the peak/trough BIN is picked from `train`, and the value read off at
    `test`. Picking (arg)which-bin-is-extreme from the same noisy data you then measure is exactly the kind of
    selection that biases `modulation` upward (it is what figure 5's cross-validated heatmap sorting already avoids
    for display; this is the same idea as a per-cell scalar)."""
    fin_tr, fin_te = np.isfinite(train), np.isfinite(test)
    ok_cell = fin_tr.all(0) & fin_te.all(0)
    hi = np.argmax(np.where(fin_tr, train, -np.inf), axis=0)
    lo = np.argmin(np.where(fin_tr, train, np.inf), axis=0)
    cols = np.arange(train.shape[1])
    return np.where(ok_cell, test[hi, cols] - test[lo, cols], np.nan)


def pack(sid, cell_ids, variable, res, nb):
    """Long curve rows + per-cell stats rows of one variable."""
    n_cells = len(cell_ids)
    cur, h1, h2, cv5 = res["curve"], res["h1"], res["h2"], res["curve_cv5"]
    rows = pd.DataFrame({"session_id": sid, "cell_id": np.tile(cell_ids, nb), "variable": variable,
                         "bin": np.repeat(np.arange(nb), n_cells), "n": np.repeat(res["n"], n_cells),
                         "mean": cur.ravel(), "mean_h1": h1.ravel(), "mean_h2": h2.ravel(), "mean_cv5": cv5.ravel()})
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mod = np.nanmax(cur, 0) - np.nanmin(cur, 0)
        mod_cv = np.nanmean(np.vstack([extreme_cv(h1, h2), extreme_cv(h2, h1)]), axis=0)
        dl = cur[-1] - cur[0]
        d1, d2 = h1[-1] - h1[0], h2[-1] - h2[0]
        pk = np.where(np.isfinite(cur).all(0), np.nanargmax(np.where(np.isfinite(cur), cur, -np.inf), axis=0), -1)
        if nb >= 3:
            a, c = h1 - np.nanmean(h1, 0), h2 - np.nanmean(h2, 0)
            sh = np.nansum(a * c, 0) / np.sqrt(np.nansum(a ** 2, 0) * np.nansum(c ** 2, 0))
        else:
            sh = np.full(n_cells, np.nan)
    st = pd.DataFrame({"session_id": sid, "cell_id": cell_ids, "variable": variable, "n_samples": res["n_samples"],
                       "n_bins": nb, "eta2": res["eta2"], "eta2_cv": res["eta2_cv"], "p_value": res["p"],
                       "eta2_cv5": res["eta2_cv5"], "p_value5": res["p5"],
                       "modulation": mod, "modulation_cv": mod_cv, "diff_last_first": dl, "diff_h1": d1, "diff_h2": d2,
                       "peak_bin": pk, "split_half_r": sh})
    return rows, st


def run_scope(sid, sname, sc, ok, k, rel, A, ids_all, sel, ts_F, dt, xvars, mu_sig, seed):
    """All variables for one scope (frame set + response windows). Returns (curves, stats, bins) lists of DataFrames."""
    if ok.sum() < MIN_FRAMES:
        return [], [], []
    Y = tn.zscore_frames(np.nan_to_num(A), ok)
    good = np.isfinite(Y[ok]).all(0)
    if not good.any():
        return [], [], []
    ids = np.asarray(ids_all)[good]
    Y = np.where(ok[:, None], np.nan_to_num(Y[:, good]), 0.0)
    Y2 = Y ** 2
    odd = (k % 2 == 1)
    fold5 = (k % 5)
    rng = np.random.default_rng((int(seed), list(SCOPES).index(sname)))
    n_ok = int(ok.sum())
    min_shift = min(int(round(MIN_SHIFT_S / dt)), n_ok // 4)
    shifts = rng.integers(min_shift, max(min_shift + 1, n_ok - min_shift), n_shifts)   # shifts inside the concatenated analysed frames
    Yc, Y2c = Y[ok], Y2[ok]

    specs = {"position": lambda: fixed_bins(rel, ok, sc["pos_range"][0], sc["pos_range"][1], sc["pos_bin"])}
    if xvars["speed"] is not None:
        specs["speed"] = lambda: quantile_bins(xvars["speed"], ok, N_BINS, zero_thresh=STOP_SPEED)
    if xvars["pupil_area"] is not None:
        specs["pupil_area"] = lambda: quantile_bins(xvars["pupil_area"], ok, N_BINS)
    if xvars["lick_rate"] is not None:
        specs["lick_rate"] = lambda: quantile_bins(xvars["lick_rate"], ok, N_BINS, zero_thresh=1e-9)
    if xvars["motionenergy"] is not None:
        specs["motionenergy"] = lambda: quantile_bins(xvars["motionenergy"], ok, N_BINS)

    curves, stats, bins = [], [], []

    def add(r_, s_, b_):
        curves.append(r_.assign(scope=sname))
        stats.append(s_.assign(scope=sname))
        bins.append(b_.assign(scope=sname))

    for v in FRAME_VARIABLES:
        if v not in specs:
            continue
        b, nb, centers = specs[v]()
        cnt = np.bincount(b[b >= 0], minlength=nb)
        if (b >= 0).sum() < nb * MIN_FRAMES_PER_BIN or (cnt < MIN_FRAMES_PER_BIN).sum() > nb // 2:
            continue                                     # variable absent / almost constant in this scope and session
        res = tune_frames(Yc, Y2c, b[ok], nb, odd[ok], fold5[ok], np.ones(n_ok, bool), shifts)
        r_, s_ = pack(sid, ids, v, res, nb)
        add(r_, s_, pd.DataFrame({"session_id": sid, "variable": v, "bin": np.arange(nb), "x_center": centers, "n": cnt}))

    # trial-level variables: trial responses in a position window
    R = tn.trial_window_means(Y, k, rel, ok, len(sel), {"stim": sc["sig_win"], "reward": sc["lick_win"]})
    odd_tr = (np.arange(len(sel)) % 2 == 1)
    fold5_tr = (np.arange(len(sel)) % 5)
    tr_x = {}
    if "signal" in TRIAL_VARIABLES and mu_sig is not None:
        tab = pd.DataFrame({"signal": sel["signal"].to_numpy(float),
                            "stim_pct": psychometric_pct(sel["signal"], mu_sig[0], mu_sig[1])})
        sb = ct.add_stimulus_columns(tab, [STIM_STRATEGY])[ct.stim_predictor(STIM_STRATEGY)].to_numpy(float)
        tr_x["signal"] = (np.where(np.isfinite(sb), sb, -1).astype(int), "stim")
    if "p_lick" in TRIAL_VARIABLES and "lickResponse" in sel.columns:
        tr_x["p_lick"] = (sel["lickResponse"].fillna(-1).astype(int).clip(lower=-1, upper=1).to_numpy(), "reward")
    for v, (xb, w) in tr_x.items():
        nb = int(xb.max()) + 1 if (xb >= 0).any() else 0
        if nb < 2:
            continue
        res = tune_trials(R[w], xb, nb, odd_tr, fold5_tr, rng, n_shifts)
        if res is None:
            continue
        g_ = res["good"]
        res = {k_: (val[:, g_] if isinstance(val, np.ndarray) and val.ndim == 2
                    else (val[g_] if isinstance(val, np.ndarray) and val.ndim == 1 and len(val) == len(g_) else val))
               for k_, val in res.items()}
        r_, s_ = pack(sid, ids[g_], v, res, nb)
        add(r_, s_, pd.DataFrame({"session_id": sid, "variable": v, "bin": np.arange(nb),
                                  "x_center": np.arange(nb, dtype=float), "n": res["n"]}))
    return curves, stats, bins


def process_session(ses, seed, new_cells, new_pairs, req_cells):
    sid = ses.session_id
    cell_ids = list(new_cells)          # the requested cells (session_jobs.run_sessions_units)
    td = ses.trialdata
    if (not cell_ids or ses.calciumdata is None or ses.ts_F is None or td is None or ses.videodata is None
            or ses.behaviordata is None or "engaged" not in td.columns or "signal" not in td.columns):
        return None
    sel = td[td["engaged"] == 1]
    if NOISE_TRIALS_ONLY:
        sel = sel[(sel["signal"] > 0) & (sel["signal"] < 100)]
    sel = sel.sort_values("stimStart").reset_index(drop=True)
    if len(sel) < MIN_TRIALS:
        return None
    merged = zscore_per_session(remove_pupil_outliers(merge_behavior_video(ses.behaviordata, ses.videodata)))
    ts_F = np.asarray(ses.ts_F, float)
    dt = float(np.median(np.diff(ts_F)))

    def frames(col):
        return tn.behavior_to_imaging_frames(merged["ts"], merged[col], ts_F, "mean") if col in merged.columns else None

    x_lick = None
    if "lick" in merged.columns:
        ev = np.nan_to_num(tn.behavior_to_imaging_frames(merged["ts"], merged["lick"], ts_F, "rate"))     # events / s
        kb = max(1, int(round(LICK_SMOOTH_S / dt)))
        x_lick = np.convolve(ev, np.ones(kb) / kb, mode="same")
    xvars = {"speed": frames("runspeed"), "pupil_area": frames("pupil_area"), "motionenergy": frames("motionenergy"),
             "lick_rate": x_lick}
    ms = mu_sigma_lookup.get(sid, {})
    mu_sig = (ms["mu"], ms["sigma"]) if ms and np.isfinite(ms.get("mu", np.nan)) and np.isfinite(ms.get("sigma", np.nan)) else None

    A = ses.calciumdata.loc[:, cell_ids].to_numpy(dtype=float)
    zpos_F = tn.interp_to_imaging_frames(ses.behaviordata["ts"], ses.behaviordata["zpos"], ts_F)
    k, rel = tn.position_relative_to_stim(zpos_F, sel["stimStart"].to_numpy(float), S_PRE, S_POST)
    base_ok = tn.trial_frame_mask(ts_F, td, sel["trialNumber"]) & (k >= 0) & np.isfinite(A).all(1)
    curves, stats, bins = [], [], []
    for sname, sc in SCOPES.items():
        lo, hi = sc["frame_range"]
        ok = base_ok & (rel >= lo) & (rel < hi)
        c_, s_, b_ = run_scope(sid, sname, sc, ok, k, rel, A, cell_ids, sel, ts_F, dt, xvars, mu_sig, seed)
        curves += c_
        stats += s_
        bins += b_
    if not stats:
        return None
    return dict(curves=pd.concat(curves, ignore_index=True), stats=pd.concat(stats, ignore_index=True),
                bins=pd.concat(bins, ignore_index=True))



results = run_sessions_units(
    session_ids, process_session, [PROTOCOL], paths.store, "tuning_session",
    key_params=dict(version="2k-v4", scopes=SCOPES, noise=NOISE_TRIALS_ONLY, variables=VARIABLES, s_pre=S_PRE, s_post=S_POST,
                    pos=(POSITION_RANGE, POSITION_BIN_CM), n_bins=N_BINS, stop=STOP_SPEED, lick_smooth=LICK_SMOOTH_S,
                    windows=(SIGNAL_WINDOW, PLICK_WINDOW), strategy=STIM_STRATEGY,
                    mins=(MIN_FRAMES, MIN_FRAMES_PER_BIN, MIN_TRIALS, MIN_TRIALS_PER_BIN), n_shifts=n_shifts,
                    min_shift_s=MIN_SHIFT_S, qc_mtime=qc_mtime, fits_mtime=int(fits_path.stat().st_mtime)),
    load_kwargs=dict(load_behaviordata=True, load_videodata=True, load_celldata=True, load_calciumdata=True),
    cells_by_session=cells_by_session, parts={"curves": "cell", "stats": "cell"}, groups_by_cell=cell_group,
    max_cells=scope.max_cells, base_seed=RANDOM_STATE, n_jobs=scope.n_jobs, recompute=args.recompute,
    verbose=JOBLIB_VERBOSE, extra_key_by_session=stamps)
results = [r for r in results if r]
if not results:
    raise SystemExit("No results (sessions without video, too few frames or too few noise trials).")

# --------------------------------------------------------------------- #
# 3. Tables: cells with their group, summaries per group
# --------------------------------------------------------------------- #
cd = pd.read_csv(cell_dist_out / "celldata_combined.csv", low_memory=False)
stats = ct.attach_cell_meta(pd.concat([r["stats"] for r in results], ignore_index=True), cd)
GROUP_KEYS = [group_key(g) for g in SIX_GROUP_ORDER]
stats = stats[stats["group"].isin(GROUP_KEYS)].reset_index(drop=True)
if stats.empty:
    raise SystemExit("No cells in any group.")
stats["tuned"] = stats["p_value"] < ALPHA
meta = stats[["session_id", "cell_id", "group", "animal_id"]].drop_duplicates(["session_id", "cell_id"])
curves = pd.concat([r["curves"] for r in results], ignore_index=True).merge(meta, on=["session_id", "cell_id"], how="inner")
bins_tbl = pd.concat([r["bins"] for r in results], ignore_index=True)
del results
stats.to_csv(table_dir / "tuning_stats_per_cell.csv", index=False)
curves.to_csv(table_dir / "tuning_curves_per_cell.csv", index=False)
bins_tbl.to_csv(table_dir / "tuning_bins.csv", index=False)
print(f"{stats['cell_id'].nunique()} cells, {stats['session_id'].nunique()} sessions")

tg = ct.unit_group_summary(stats, ["scope", "variable"], AGG, ["eta2_cv", "eta2_cv5", "split_half_r", "modulation_cv"], sig_col="tuned")
tg.to_csv(table_dir / f"tuning_by_group{AGG_SUFFIX}.csv", index=False)
for SC in SCOPES:
    print(f"\nFraction of tuned cells (p<{ALPHA}; chance ~{ALPHA:.0%}), scope '{SC}', mean over {'cells' if AGG == 'pooled' else UNITS}, per group:")
    t_ = tg[tg["scope"] == SC]
    print(t_.groupby(["variable", "group"])["frac_sig"].mean().unstack("group")
          .reindex(index=[v for v in VARIABLES if v in set(t_["variable"])], columns=GROUP_KEYS).round(2).to_string())
stats_all, curves_all, bins_all = stats, curves, bins_tbl

agg_note = {"pooled": "all cells pooled (no error bars)", "session": "bar = mean +- SEM over sessions, dots = sessions",
            "mouse": "bar = mean +- SEM over mice, dots = mice"}[AGG]


# --------------------------------------------------------------------- #
# 4. Figures
# --------------------------------------------------------------------- #
def unit_mean_curves(var, col="mean"):
    """Mean tuning curve per group for column `col` (mean / mean_h1 / mean_h2 / mean_cv5): mean over units of the
    unit's mean over its cells (units with < MIN_CELLS_PER_UNIT cells left out), SEM over units; pooled: mean over
    all cells, SEM over cells."""
    d = curves[curves["variable"] == var]
    if AGG == "pooled":
        g = d.groupby(["group", "bin"])[col].agg(["mean", "sem", "size"]).reset_index()
        return g.rename(columns={"size": "n_units"})
    keys = ["group"] + UNIT_KEYS + ["bin"]
    per = d.groupby(keys)[col].agg(["mean", "size"]).reset_index()
    per = per[per["size"] >= MIN_CELLS_PER_UNIT]
    g = per.groupby(["group", "bin"])["mean"].agg(["mean", "sem", "size"]).reset_index()
    return g.rename(columns={"size": "n_units"})


def bin_ticks(var):
    """x positions and tick labels of the bins of a variable (typical value of each bin: median over sessions)."""
    b = bins_tbl[bins_tbl["variable"] == var].groupby("bin")["x_center"].median()
    if var in TRIAL_LABELS:
        return b.index.to_numpy(), TRIAL_LABELS[var][:len(b)]
    if var == "position":
        return b.to_numpy(), [f"{v:g}" for v in b.to_numpy()]
    return b.index.to_numpy(), [f"{v:.2g}" for v in b.to_numpy()]





unit_curve_rows = []
for SC, scd in SCOPES.items():
    stats, curves, bins_tbl = (t[t["scope"] == SC] for t in (stats_all, curves_all, bins_all))
    variables = [v for v in VARIABLES if (stats["variable"] == v).any()]
    if not variables:
        continue
    SFX = "" if SC == "all" else f"_{SC}"
    SNOTE = scd["label"]
    ncols = 4
    nrows = int(np.ceil(len(variables) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.5 * nrows), squeeze=False, sharey=True)
    for ax, v in zip(axes.ravel(), variables):
        g = unit_mean_curves(v)
        xpos, xlab = bin_ticks(v)
        xmap = dict(zip(range(len(xpos)), xpos if v == "position" else range(len(xpos))))
        for gi, grp in enumerate(SIX_GROUP_ORDER):
            key = group_key(grp)
            s = g[g["group"] == key].sort_values("bin")
            if s.empty:
                continue
            x = np.array([xmap[int(b)] for b in s["bin"]])
            col = SIX_GROUP_COLORS[grp]
            ax.plot(x, s["mean"], color=col, marker="o", markersize=3, lw=1.3, label=SIX_GROUP_LABELS[grp])
            if np.isfinite(s["sem"]).any():
                ax.fill_between(x, s["mean"] - s["sem"].fillna(0), s["mean"] + s["sem"].fillna(0), color=col, alpha=0.15, lw=0)
            unit_curve_rows.append(s.assign(variable=v, scope=SC))
        ax.axhline(0, color="grey", lw=0.6, ls="--")
        ax.set_xticks(list(xmap.values()))
        ax.set_xticklabels(xlab, fontsize=7, rotation=0 if v in TRIAL_LABELS or v == "position" else 30)
        ax.set_title(VAR_LABELS[v], fontsize=9)
        ax.set_xlabel("bin (median value)" if v not in TRIAL_LABELS and v != "position" else "", fontsize=8)
    for ax in axes.ravel()[len(variables):]:
        ax.axis("off")
    for r_ in range(nrows):
        axes[r_][0].set_ylabel("activity (SD)", fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=7, ncol=2, loc="best")
    fig.suptitle(f"Mean tuning curves per group ({PROTOCOL}, {TRIAL_NOTE}, {SNOTE}); "
                 f"{'mean +- SEM over cells' if AGG == 'pooled' else f'mean +- SEM over {UNITS}'}; y shared; frame-level panels in SD of the "
                 f"frames, signal / lick in SD of the trial responses", y=1.0, fontsize=10)
    fig.tight_layout()
    save_fig(fig, figdir, f"1_mean_tuning_curves{SFX}")
    plt.close(fig)

    # idea 1: the curve fit on ODD trials (line) plotted together with the EVEN-trial points (markers), per group --
    # a cross-validated check on figure 1's curve: if the line and the points for a group disagree, that group's
    # figure-1 curve does not replicate on independent trials.
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.5 * nrows), squeeze=False, sharey=True)
    for ax, v in zip(axes.ravel(), variables):
        xpos, xlab = bin_ticks(v)
        xmap = dict(zip(range(len(xpos)), xpos if v == "position" else range(len(xpos))))
        g1, g2 = unit_mean_curves(v, "mean_h1"), unit_mean_curves(v, "mean_h2")
        for grp in SIX_GROUP_ORDER:
            key = group_key(grp)
            col = SIX_GROUP_COLORS[grp]
            s1 = g1[g1["group"] == key].sort_values("bin")
            s2 = g2[g2["group"] == key].sort_values("bin")
            if s1.empty or s2.empty:
                continue
            x1 = np.array([xmap[int(b)] for b in s1["bin"]])
            x2 = np.array([xmap[int(b)] for b in s2["bin"]])
            ax.plot(x1, s1["mean"], color=col, lw=1.1, alpha=0.9, zorder=2)
            ax.plot(x2, s2["mean"], color=col, marker="o", markersize=4, lw=0, alpha=0.9, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.4)
        ax.axhline(0, color="grey", lw=0.6, ls="--")
        ax.set_xticks(list(xmap.values()))
        ax.set_xticklabels(xlab, fontsize=7, rotation=0 if v in TRIAL_LABELS or v == "position" else 30)
        ax.set_title(VAR_LABELS[v], fontsize=9)
        ax.set_xlabel("bin (median value)" if v not in TRIAL_LABELS and v != "position" else "", fontsize=8)
    for ax in axes.ravel()[len(variables):]:
        ax.axis("off")
    for r_ in range(nrows):
        axes[r_][0].set_ylabel("activity (SD)", fontsize=9)
    handles = [plt.Line2D([0], [0], color="grey", lw=1.1, label="curve fit on odd trials"),
               plt.Line2D([0], [0], color="grey", marker="o", lw=0, markersize=4, label="even-trial points (held out)")]
    axes[0][0].legend(handles=handles, frameon=False, fontsize=7, loc="best")
    fig.suptitle(f"Cross-validated check on figure 1 ({PROTOCOL}, {TRIAL_NOTE}, {SNOTE}): line = mean curve fit on "
                 f"ODD trials, markers = EVEN-trial points the fit did not see (idea 1); the same comparison the "
                 f"other way round (fit on even, points from odd) is eta2_cv's other half -- disagreement here is "
                 f"what a low / negative eta2_cv for that group means; y shared", y=1.0, fontsize=10)
    fig.tight_layout()
    save_fig(fig, figdir, f"1b_cv_fit_vs_heldout_points{SFX}")
    plt.close(fig)

    stats_sig = stats.assign(sig=stats["tuned"].astype(float))
    group_panels(stats_sig, "variable", variables, "sig", AGG, figdir, f"2_fraction_tuned{SFX}",
                 f"Fraction of cells with significant tuning (shift / permutation null) -- {PROTOCOL}, {TRIAL_NOTE}, {SNOTE}; dashed = chance ({ALPHA}); {agg_note}",
                 ylabel=f"fraction of cells p<{ALPHA}", hline=ALPHA, mouse_colors=mouse_colors,
                 min_cells_per_unit=MIN_CELLS_PER_UNIT, ncols=4, panel_size=(4.6, 3.2))
    group_panels(stats, "variable", variables, "eta2_cv", AGG, figdir, f"3_variance_explained{SFX}",
                 f"Variance of the activity explained by the variable's bins, CROSS-VALIDATED (fit on odd/even trials, scored "
                 f"on the other half) -- {PROTOCOL}, {TRIAL_NOTE}, {SNOTE}; {agg_note}",
                 ylabel="eta2 (cross-validated)", hline=0, mouse_colors=mouse_colors, min_cells_per_unit=MIN_CELLS_PER_UNIT, ncols=4,
                 panel_size=(4.6, 3.2))
    group_panels(stats[stats["n_bins"] >= 3], "variable", [v for v in variables if v != "p_lick"], "split_half_r", AGG, figdir,
                 f"4_split_half_reliability{SFX}",
                 f"Split-half reliability of the tuning curve (odd vs even trials, correlation over bins) -- {PROTOCOL}, {TRIAL_NOTE}, {SNOTE}; {agg_note}",
                 ylabel="split-half r", hline=0, mouse_colors=mouse_colors, min_cells_per_unit=MIN_CELLS_PER_UNIT, ncols=4,
                 panel_size=(4.6, 3.2))

    # cross-validated heatmaps: cells sorted on the odd-trial curve, shown for the even-trial curve
    for v in variables:
        d = curves[curves["variable"] == v]
        nb = int(d["bin"].max()) + 1
        if nb < 3:
            continue
        xpos, xlab = bin_ticks(v)
        fig, axes = plt.subplots(1, len(SIX_GROUP_ORDER), figsize=(2.3 * len(SIX_GROUP_ORDER), 4.6), squeeze=False)
        im = None
        for ax, grp in zip(axes[0], SIX_GROUP_ORDER):
            key = group_key(grp)
            dd = d[d["group"] == key]
            ax.set_title(f"{SIX_GROUP_LABELS[grp]}", fontsize=9)
            if dd.empty:
                ax.axis("off")
                continue
            H1 = dd.pivot_table(index=["session_id", "cell_id"], columns="bin", values="mean_h1").reindex(columns=range(nb))
            H2 = dd.pivot_table(index=["session_id", "cell_id"], columns="bin", values="mean_h2").reindex(columns=range(nb)).loc[H1.index]
            keep = np.isfinite(H1.to_numpy()).all(1) & np.isfinite(H2.to_numpy()).all(1)
            H1, H2 = H1[keep], H2[keep]
            if len(H1) < 2:
                ax.axis("off")
                continue
            order = np.argsort(H1.to_numpy().argmax(1), kind="stable")
            M = H2.to_numpy()[order]
            M = M - M.mean(1, keepdims=True)
            M = M / np.maximum(np.abs(M).max(1, keepdims=True), 1e-9)
            if len(M) > MAX_HEATMAP_CELLS:
                M = M[np.linspace(0, len(M) - 1, MAX_HEATMAP_CELLS).astype(int)]
            im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
            ax.set_xticks(range(nb))
            ax.set_xticklabels(xlab, fontsize=6, rotation=90)
            ax.set_yticks([])
            ax.set_xlabel(f"n={int(keep.sum())} cells", fontsize=7)
        if im is not None:
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, pad=0.01, label="curve of the even trials,\ncentred, scaled to +-1")
        fig.suptitle(f"{VAR_LABELS[v]}: per-cell tuning curves, cells sorted by the peak bin of the ODD trials and drawn from the EVEN "
                     f"trials (cross-validated) -- {PROTOCOL}, {TRIAL_NOTE}, {SNOTE}", y=1.02, fontsize=9)
        save_fig(fig, figdir, f"5_heatmap_{v}{SFX}")
        plt.close(fig)

    print(f"\nWrote tables to {table_dir}, figures to {figdir}")

    # figure 6: one figure per variable, one subplot per group; thin lines = units (sessions / mice; pooled: example cells), thick = group mean
    rs = np.random.default_rng(0)
    ukeys = UNIT_KEYS if AGG != "pooled" else ["session_id", "cell_id"]
    for v in variables:
        d = curves[curves["variable"] == v]
        xpos, xlab = bin_ticks(v)
        xmap = dict(zip(range(len(xpos)), xpos if v == "position" else range(len(xpos))))
        g = unit_mean_curves(v)
        fig, axes = plt.subplots(2, 4, figsize=(4.2 * 4, 3.3 * 2), squeeze=False, sharex=True, sharey=True)
        allv = []
        for ax, grp in zip(axes.ravel(), SIX_GROUP_ORDER):
            key = group_key(grp)
            col = SIX_GROUP_COLORS[grp]
            dd = d[d["group"] == key]
            ax.set_title(f"{SIX_GROUP_LABELS[grp]}", fontsize=9)
            if dd.empty:
                continue
            if AGG == "pooled":
                keys = dd[ukeys].drop_duplicates()
                keys = keys.iloc[rs.permutation(len(keys))[:N_EXAMPLE_CELLS]]
                thin = dd.merge(keys, on=ukeys).groupby(ukeys + ["bin"])["mean"].mean().reset_index()
                thin["animal_id"] = None
            else:
                thin = dd.groupby(["animal_id"] + [k_ for k_ in ukeys if k_ != "animal_id"] + ["bin"])["mean"].agg(["mean", "size"]).reset_index()
                thin = thin[thin["size"] >= MIN_CELLS_PER_UNIT].rename(columns={"mean": "mean"})
            for _, u in thin.groupby(ukeys):
                u = u.sort_values("bin")
                lc = col if AGG == "pooled" else mouse_colors.get(u["animal_id"].iloc[0], "grey")
                ax.plot([xmap[int(b)] for b in u["bin"]], u["mean"], color=lc, lw=0.7, alpha=0.25 if AGG == "pooled" else 0.6)
            allv += list(thin["mean"].dropna())
            s = g[g["group"] == key].sort_values("bin")
            if not s.empty:
                x = np.array([xmap[int(b)] for b in s["bin"]])
                ax.plot(x, s["mean"], color=col if AGG == "pooled" else "black", marker="o", markersize=3, lw=2)
                if np.isfinite(s["sem"]).any():
                    ax.fill_between(x, s["mean"] - s["sem"].fillna(0), s["mean"] + s["sem"].fillna(0), color="grey", alpha=0.3, lw=0)
                allv += list(s["mean"] + s["sem"].fillna(0)) + list(s["mean"] - s["sem"].fillna(0))
            ax.axhline(0, color="grey", lw=0.6, ls="--")
            ax.set_xticks(list(xmap.values()))
            ax.set_xticklabels(xlab, fontsize=7, rotation=0 if v in TRIAL_LABELS or v == "position" else 30)
        allv = [x_ for x_ in allv if np.isfinite(x_)]
        if allv:
            lo_, hi_ = np.percentile(allv, [1, 99])
            pad = 0.1 * (hi_ - lo_ + 1e-9)
            axes[0][0].set_ylim(lo_ - pad, hi_ + pad)
        for r_ in range(2):
            axes[r_][0].set_ylabel("activity (SD)", fontsize=9)
        what = {"pooled": f"thin = {N_EXAMPLE_CELLS} random cells, thick = mean over all cells",
                "session": "thin = sessions (colour = mouse), black = mean +- SEM over sessions",
                "mouse": "thin = mice, black = mean +- SEM over mice"}[AGG]
        fig.suptitle(f"{VAR_LABELS[v]} -- one panel per group ({PROTOCOL}, {TRIAL_NOTE}, {SNOTE}); {what}; y shared (1-99 percentile)",
                     fontsize=9, y=1.0)
        fig.tight_layout()
        save_fig(fig, figdir, f"6_by_group_{v}{SFX}")
        plt.close(fig)

pd.concat(unit_curve_rows, ignore_index=True).to_csv(table_dir / f"tuning_unit_curves{AGG_SUFFIX}.csv", index=False)
print(f"\nWrote tables to {table_dir}, figures to {figdir}")
