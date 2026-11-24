# -*- coding: utf-8 -*-
"""
2c_information
================
Mutual information between each recorded cell's trial-by-trial activity and (a) ongoing behavioral state
(running speed, pupil area, motion energy, lick rate) and (b) task variables (the stimulus, binned by the
shared strategies of stim_binning.py; choice; correctness) -- DN sessions exclusively (the only protocol with
cell recordings).

Dataset: 1b-included sessions, ENGAGED trials, cells that pass 2a's proximity filter AND 2b's QC
(infotheory.neural_encoding.load_cell_inclusion). Response windows mirror 1c_behavior (pre_stim / stim / reward,
positions in cm relative to stimStart); a cell's activity is its mean in the window per trial, and the
behavioral predictors come from the SAME window (a same-epoch encoding question, not 1d's prediction question).
Face quantities (pupil, motion energy) are z-scored within the session before the window means (running speed and
licks stay in physical units; project_config.NORMALIZE_FACE_PER_SESSION).

TWO SAMPLING UNITS for the behavior-vs-cell MI (the stimulus / choice / correct predictors are trial-level only):
    per TRIAL   one value per trial and window (mean over the window), project_config.RESPONSE_BINS bins -- shared
                with 2d/3x so the measures agree before bias correction;
    pooled FRAMES (predictors "cur_<var>__frames", --no-frames to skip)   every imaging frame inside the window of every
                engaged trial pooled into one distribution, project_config.RESPONSE_BINS_FRAMES bins (thousands of
                samples afford finer bins). Frames are autocorrelated, so the null is a circular shift of the cell
                trace, not a permutation. Table: cell_predictor_mi_frames.csv; shown next to the per-trial ones in the figures.

Project-wide rules (infotheory/project_config.py) -- what applies here
    * selection : --animals / --sessions restrict the sessions; a restricted run writes to out/selection/<tag>/
                  and NEVER overwrites the tables in out/ that 2d (and later steps) read.
    * aggregate : --aggregate pooled|session|mouse = the unit for the group comparisons (bars of MI, fraction of
                  significant cells): pooled = all cells of a group together, no error bars; session / mouse = one
                  value per session / mouse (MIN_CELLS_PER_UNIT cells in the group needed), bar = mean +- SEM over
                  units, dots = units coloured by mouse. Always all the groups of the depth rule (the six + deep PM). No violin plots.
    * cell cap  : --max-cells N limits the cells PER GROUP per session (--max-pairs N the cell pairs per pair of groups, between-area MI);
                  defaults: project_config.MAX_CELLS_PER_SESSION / MAX_PAIRS_PER_SESSION (None = all). Incremental: the
                  cache keeps results per cell / per pair, so raising the cap computes only the missing units, lowering it
                  reads the stored ones. Capped runs write to out/selection/<tag>/ (e.g. maxcells-100), never to out/.
    * memory    : one session in RAM per worker (--n-jobs), reduced to small tables and cached per session in
                  store/. The dataset is never loaded at once.
    * binning   : the information measures here and in 2d use the SAME trials and the SAME bins, so they agree
                  before bias correction: per-pair trials from infotheory.cell_tables.pair_arrays (finite values,
                  stimulus/choice bins with >= MIN_TRIALS_PER_TARGET_BIN trials, >= MIN_TRIALS_ANALYSIS trials),
                  RESPONSE_BINS equal-occupancy bins for the cell response AND for continuous predictors, stimulus
                  predictors from stim_binning (ONE strategy, `stim__pct_0-25_25-50_50-75_75-100`: 4 equal steps of the
                  psychometric curve = lick probability 0-25 / 25-50 / 50-75 / 75-100 %), choice and correct as-is.
                  CHANGE vs. the previous 2c: MI used 12 bins and the quantile `stim_bin`; now RESPONSE_BINS (3)
                  and the strategies. MI values are therefore not comparable with old runs (fewer bins = lower
                  absolute MI, less bias); consumers of `stim_bin` (2e/2f) must switch to `stim__<strategy>`.

Estimator: histogram MI (mutual_information_shuffle), not KSG: hundreds to thousands of cells x ~9 predictors x 3
windows is too many pairs for KSG. Bias correction = subtract the mean MI of N_SHUFFLES shuffles; p-value = fraction of
shuffles >= raw (resolution 1/(N_SHUFFLES+1)). The raw MI is saved next to the corrected one.

MI GIVEN RUNNING SPEED (--no-given-runspeed to skip): I(X;Y | z) with z = the window-mean running speed, same
bins (RESPONSE_BINS quantile bins, z too), same trials as the plain MI for that pair (restricted to finite z), shuffle
null permuting X WITHIN z bins (infotheory.conditional_info). Reported next to the plain MI recomputed on the same trials,
so the two are directly comparable (figures 7-8). A drop means the information about X is (also) carried by what running
speed explains; it is not evidence that the cell is "movement-driven". With only 3 z bins, conditioning removes the
coarse part of the running-speed relation. Engaged trials only (not in the tail test).

CAVEAT (read before trusting choice / correct): position windows give each trial a different number of imaging frames,
and running speed depends on choice (slower on lick trials -> more frames -> more precise window mean). MI reads this
precision difference as information about choice. The script prints cur_runspeed's significant fraction next to choice
/ correct per window: similar magnitudes suggest the confound rather than genuine encoding. (2d's ROC part uses
position-matched frame sampling to remove it.)

INFORMATION BETWEEN AREAS (--no-between to skip; added on top of everything else): the MI between the window-mean activity
of every PAIR of included cells of a session -- cross-area pairs (V1 x PM, V1 x AL, ...) AND within-area pairs for reference --
for the engaged trials. Everything for all pairs at once with one-hot contingency matrices (RESPONSE_BINS equal-occupancy bins per cell,
infotheory.info_theory.quantile_bin_edges, i.e. the binning of mutual_information_hist). Two measures per pair:
    mi            I(X;Y): bias-corrected (minus the mean of N_SHUFFLES shuffles of one cell's trials), p = fraction of shuffles >= raw.
                  It contains what both cells share because of the stimulus, behavior or shared drive ("signal + noise").
    mi_given_stim I(X;Y | S), S = the stimulus bin of BETWEEN_STRATEGY (bins with < MIN_TRIALS_PER_TARGET_BIN trials dropped), the
                  shuffle permutes trials WITHIN S bins: what the cells share beyond the stimulus (the information analogue of the
                  noise correlation; behavior and choice are NOT conditioned out).
Cells with non-finite responses on some trials are handled pair-wise (those trials are left out of that pair). Summaries only
(per session, window, pair of the six groups): n pairs, mean raw / corrected MI, fraction significant. `--check-between N` compares N
random pairs per window with mutual_information_hist (the single-pair code) and prints the largest difference.

EXPLORATORY TEST -- end-of-session DISENGAGED trials (--no-tail to skip)
    Same MI on the disengaged block at the END of each session (not on sporadic lapses earlier on; definition and
    matching in infotheory.engagement: longest suffix with >= --tail-min-frac disengaged trials, analysed = its
    engaged==0 trials) versus an EQUAL number of the last engaged trials before it. Equal n keeps the estimation
    bias equal. This is a look, not a null analysis: no correction across cells, one block per session, few
    trials (windows with < MIN_TRIALS_ANALYSIS valid trials give no value). Disengaged animals hardly lick, so
    choice / correct usually have no valid pairs there (silently missing, see tail_trials.csv for the counts);
    an MI drop can reflect reduced behavioral variance, not only reduced encoding. Sessions without a tail are
    reported in tail_trials.csv and left out of the tail figures.

Outputs
-------
2_pipeline/2c_information/
    out/    cell_predictor_mi_given_runspeed.csv   plain MI and I(X;Y|running speed) on the SAME trials per (session, window,
                                           cell, predictor != runspeed); mi_given_runspeed_by_group[__<aggregate>].csv
            cell_predictor_mi.csv          engaged trials, one row per (session, window, cell, predictor): MI raw / shuffle
                                           mean / corrected, p-value, n_trials, + animal_id, area, label, group (read by 2d)
            cell_predictor_mi_frames.csv   the POOLED-FRAMES MI (predictor "<cur_var>__frames", n_trials = number of frames), same
                                           columns; kept apart so that what 2d reads does not change
            mi_by_group[__<aggregate>].csv per group (and unit) x window x predictor: n_cells, mean MI, fraction significant
            confound_check[__<aggregate>].csv   cur_runspeed vs choice / correct fraction significant
            tail_trials.csv               per session: tail found?, start, n, fraction disengaged, control trials
            tail_cell_predictor_mi.csv    the same MI table for the tail test, column `condition`
                                          (engaged_matched | disengaged)
            tail_mi_by_group[__<aggregate>].csv
            between_area_mi_by_group_pair[__<aggregate>].csv, between_area_mi_by_area_pair[__<aggregate>].csv,
            between_area_mi_per_session.csv   (information between areas, see above)
            figures/*.png                 1a/1b MI and 2a/2b fraction significant (rows = predictor ordered by mean MI, columns = pre_stim / stim / reward; a = per-trial, b = pooled frames), 3 heatmap; 4-6 tail test; 9-11 between areas
    out/selection/<tag>/...               the same for a restricted run
    store/  mi_session_<hash>.pkl         per-session caches
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from infotheory import project_config as cfg
from infotheory.pipeline import get_pipeline_paths
from infotheory.session import discover_sessions
from infotheory.neural_encoding import load_cell_inclusion, build_session_tables, build_window_frames
from infotheory.info_theory import mutual_information_shuffle, mutual_information_hist, quantile_bin_edges
from infotheory.celldata_utils import SIX_GROUP_ORDER, SIX_GROUP_LABELS, SIX_GROUP_COLORS, DEEP_PM_GROUPS
from infotheory.groups import group_key, add_six_group_column
from scipy import stats as sps
from infotheory.plotting import set_style, save_fig, clear_figures
from infotheory.scope import add_scope_args, scope_from_args, select_sessions, scoped_dirs, data_stamp, unit_noun
from infotheory.session_jobs import run_sessions_units
from infotheory.unit_stats import stable_seed
from infotheory.unit_plots import group_panels, group_grid, mouse_palette, unit_bars
from infotheory.stim_binning import STIM_BINNING_STRATEGIES, SIMPLE_STRATEGIES, target_signature, describe_strategies
from infotheory import cell_tables as ct
from infotheory import engagement as eg
from infotheory.conditional_info import mi_and_conditional_mi

PROTOCOL = "DN"

# Response windows: [start_cm, end_cm] relative to stimStart (a position), as in 1c_behavior.MI_WINDOWS
WINDOWS = {"pre_stim": (-30.0, -10.0), "stim": (0.0, 20.0), "reward": (25.0, 45.0)}

BINNING_STRATEGIES = list(SIMPLE_STRATEGIES)    # ONE stimulus binning: 4 bins of the psychometric curve, 0-25 / 25-50 / 50-75 / 75-100 %
#                                                 (stim_binning.SIMPLE_STRATEGIES; override with --strategies)
N_SHUFFLES = 20                                 # shuffles for bias correction + p-value (compute budget)
MIN_CELLS_PER_UNIT = 5                          # session / mouse aggregate: cells needed in a group for a unit's value
ALPHA = 0.05
RANDOM_STATE = 0
RUN_TAIL = True
SORT_PREDICTORS = True                          # figures 1 / 2: rows ordered by mean information (strongest on top); False = the fixed order
JOBLIB_VERBOSE = 5
BETWEEN_STRATEGY = "pct_0-25_25-50_50-75_75-100"       # stimulus bins S of I(X;Y | S) between areas (first strategy if not available)
BETWEEN_MIN_PAIRS = 10                          # session / mouse aggregate: pairs needed in a group pair for a unit's value
BETWEEN_CHUNK = 150                             # cells per block of the pair matrix (bounds memory)
_DEEP_ON = all(g in SIX_GROUP_ORDER for g in DEEP_PM_GROUPS)       # depth rule (project_config): deep PM as extra "area" "PMdeep"
AREAS = ["V1", "PM"] + (["PMdeep"] if _DEEP_ON else []) + ["AL", "RSP"]
AREA_NAME = {"PMdeep": "PM (deep)"}                                  # display names (tokens in the tables stay "PMdeep")
CONDITIONS = ("engaged_matched", "disengaged")  # tail test; "engaged" (all engaged trials) is the main analysis

parser = argparse.ArgumentParser()
add_scope_args(parser)
parser.add_argument("--strategies", nargs="+", default=None, choices=sorted(STIM_BINNING_STRATEGIES),
                    help=f"stimulus binning strategies (default {BINNING_STRATEGIES})")
parser.add_argument("--n-shuffles", type=int, default=N_SHUFFLES)
parser.add_argument("--no-tail", action="store_true", help="skip the disengaged-tail test")
parser.add_argument("--no-between", action="store_true", help="skip the information between areas (pairs of cells)")
parser.add_argument("--between-strategy", default=None, choices=sorted(STIM_BINNING_STRATEGIES),
                    help=f"stimulus bins conditioned on in I(X;Y|S) between areas (default {BETWEEN_STRATEGY})")
parser.add_argument("--check-between", type=int, default=0, help="compare N random pairs per window with the single-pair code")
parser.add_argument("--no-frames", action="store_true", help="skip the pooled-frames MI (every imaging frame of the windows pooled)")
parser.add_argument("--no-given-runspeed", action="store_true", help="skip the MI | running speed part")
parser.add_argument("--tail-min-frac", type=float, default=eg.TAIL_MIN_FRAC,
                    help="a session's tail = longest end-of-session block with at least this fraction of "
                         "disengaged trials")
parser.add_argument("--tail-min-trials", type=int, default=eg.TAIL_MIN_TRIALS,
                    help="minimum analysed disengaged trials (and matched engaged trials) per session")
args = parser.parse_args()
scope = scope_from_args(args)
AGG = scope.aggregate
UNIT = unit_noun(AGG)
UNITS = "mice" if UNIT == "mouse" else f"{UNIT}s"
N_JOBS = scope.n_jobs
AGG_SUFFIX = "" if AGG == "pooled" else f"__{AGG}"
STRATEGIES = args.strategies or BINNING_STRATEGIES
PREDICTORS = ct.predictor_list(STRATEGIES)
N_SHUF = args.n_shuffles
RUN_TAIL = RUN_TAIL and not args.no_tail
RUN_GIVEN = not args.no_given_runspeed
RUN_BETWEEN = not args.no_between
RUN_FRAMES = not args.no_frames
_bs = args.between_strategy or BETWEEN_STRATEGY
BETWEEN_STRAT = _bs if _bs in STRATEGIES else STRATEGIES[0]
if RUN_BETWEEN and BETWEEN_STRAT != _bs:
    print(f"Between areas: strategy {_bs!r} is not among the run's strategies, using {BETWEEN_STRAT!r}")
GIVEN = "cur_runspeed"   # the movement variable that is conditioned out (same-window mean running speed)
BINS = cfg.RESPONSE_BINS
FBINS = cfg.RESPONSE_BINS_FRAMES                 # bins of the pooled-frames MI (many more samples than trials)
FRAME_PREDICTORS = [f"{c}__frames" for c in ct.BEHAVIOR_COLUMNS] if RUN_FRAMES else []
PANEL_PREDICTORS = PREDICTORS + FRAME_PREDICTORS  # everything that gets a panel / row in the figures

paths = get_pipeline_paths(__file__)
table_dir, figdir = scoped_dirs(paths, scope)
clear_figures(figdir)
set_style()
print(scope.describe())
print(f"Stimulus predictors:\n{describe_strategies(STRATEGIES)}")
if RUN_FRAMES:
    print(f"Pooled-frames MI: {FBINS} {cfg.RESPONSE_BIN_METHOD} bins, >= {cfg.MIN_FRAMES_ANALYSIS} frames per pair, null = circular "
          f"shift of the cell trace by >= {cfg.FRAME_SHIFT_MIN_TRIALS} windows' worth of frames")
print(f"Binning (shared with 2d/3x): {BINS} {cfg.RESPONSE_BIN_METHOD} bins, >= {cfg.MIN_TRIALS_PER_TARGET_BIN} trials "
      f"per target bin, >= {cfg.MIN_TRIALS_ANALYSIS} trials per pair")

# --------------------------------------------------------------------- #
# 1. Included sessions (1b), cell inclusion (2a + 2b), selection
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
if not included_all:
    raise SystemExit(f"No {PROTOCOL} session passed 1b_psychometric's inclusion criteria.")

all_triples = discover_sessions([PROTOCOL])
triples = select_sessions(all_triples, scope)
selected = {f"{a}_{d}": (p, a, d) for p, a, d in triples}
session_ids = [s for s in selected if s in included_all]
if not session_ids:
    raise SystemExit("None of the selected sessions passed 1b_psychometric's inclusion criteria.")
mouse_colors = mouse_palette(a for _, a, _ in all_triples)

cell_inclusion = load_cell_inclusion(cell_dist_out, activity_out)
cell_group = dict(zip(cell_inclusion["cell_id"], cell_inclusion["group"]))      # per-group cell / pair caps (--max-cells / --max-pairs)
cells_by_session = {sid: sorted(sub.loc[sub["include"], "cell_id"]) for sid, sub in cell_inclusion.groupby("session_id")
                    if sid in session_ids}
session_ids = [s for s in session_ids if cells_by_session.get(s)]
if not session_ids:
    raise SystemExit("No included cells in the selected sessions.")
print(f"{len(session_ids)} sessions, {sum(len(v) for v in cells_by_session.values())} included cells "
      f"(passing 2a proximity AND 2b QC)")
stamps = {s: data_stamp(*selected[s]) for s in session_ids}


# --------------------------------------------------------------------- #
# 1b. Cell yield by group: where do the cells (and sessions) of each group get lost?  (cell_yield_*.csv, printed)
# --------------------------------------------------------------------- #
# stages: recorded -> 2a proximity filter -> 2a depth rule -> 2b QC (= the cells analysed here); area / label are taken from the raw
# annotation, BEFORE the depth rule moves a cell into 'excluded' / deep PM, so the table shows what each filter removes.
# A session counts for a group in the session / mouse aggregates only with >= MIN_CELLS_PER_UNIT cells in it.
def cell_yield():
    cd = pd.read_csv(cell_dist_out / "celldata_combined.csv", low_memory=False)
    n_not_1b = int((cd["session_id"].isin(selected) & ~cd["session_id"].isin(included_all)).sum())
    cd = cd[cd["session_id"].isin([x for x in selected if x in included_all])]      # only the 1b-included sessions are analysed
    cd = add_six_group_column(cd[["session_id", "cell_id", "roi_name", "labeled", "passes_proximity_filter"]
                                + [c for c in ("depth_class", "depth", "passes_depth_filter") if c in cd.columns]])
    qc = pd.read_csv(activity_out / "qc_per_cell.csv")[["session_id", "cell_id", "qc_pass", "qc_fail_reason"]]
    cd = cd.merge(qc, on=["session_id", "cell_id"], how="left")
    cd["area_label"] = cd["area"].astype(str) + "_" + cd["label"].astype(str)
    prox = cd["passes_proximity_filter"].astype(bool)
    depth = cd["passes_depth_filter"].astype(bool) if "passes_depth_filter" in cd.columns else pd.Series(True, index=cd.index)
    qcp = cd["qc_pass"].fillna(False).astype(bool)
    stages = {"1_recorded": np.ones(len(cd), bool), "2_proximity": prox.to_numpy(), "3_depth_rule": (prox & depth).to_numpy(),
              "4_qc_pass_(analysed)": (prox & depth & qcp).to_numpy()}
    rows = []
    for ag, g in cd.groupby("area_label"):
        for st, m in stages.items():
            gm = g[pd.Series(m, index=cd.index).loc[g.index]]
            per = gm.groupby("session_id").size()
            rows.append(dict(area_label=ag, stage=st, n_cells=len(gm), n_sessions_any=int((per >= 1).sum()),
                             n_sessions_ge_min=int((per >= MIN_CELLS_PER_UNIT).sum()),
                             median_cells_per_session=float(per.median()) if len(per) else 0.0))
    ydf = pd.DataFrame(rows)
    ydf.to_csv(table_dir / "cell_yield_by_group.csv", index=False)
    # per session: cells at each stage for the groups with few sessions
    per_s = []
    for st, m in stages.items():
        c = cd[pd.Series(m, index=cd.index)].groupby(["session_id", "area_label"]).size().rename(st)
        per_s.append(c)
    pd.concat(per_s, axis=1).fillna(0).astype(int).reset_index().to_csv(table_dir / "cell_yield_per_session.csv", index=False)
    # what the depth rule and the QC remove, by reason
    lost_depth = cd[prox & ~depth].groupby(["area_label", "depth_class"]).size().rename("n_cells").reset_index()
    lost_qc = cd[prox & depth & ~qcp].groupby(["area_label", "qc_fail_reason"], dropna=False).size().rename("n_cells").reset_index()
    lost_depth.to_csv(table_dir / "cell_yield_lost_by_depth.csv", index=False)
    lost_qc.to_csv(table_dir / "cell_yield_lost_by_qc.csv", index=False)
    wide = ydf.pivot(index="area_label", columns="stage", values="n_cells")
    wide2 = ydf[ydf["stage"] == "4_qc_pass_(analysed)"].set_index("area_label")[["n_sessions_any", "n_sessions_ge_min"]]
    print(f"\nCell yield by group, 1b-included sessions only ({n_not_1b} cells of sessions excluded by 1b are not counted); "
          f"sessions with >=1 / >={MIN_CELLS_PER_UNIT} analysed cells. Independent of --max-cells.")
    print(wide.join(wide2).to_string())


try:
    cell_yield()
except Exception as e:          # diagnostics only -- never stop the analysis
    print(f"(cell yield table skipped: {type(e).__name__}: {e})")


# --------------------------------------------------------------------- #
# 2. Per session (one in RAM at a time): condition tables -> MI records
# --------------------------------------------------------------------- #
def mi_records(table, condition, window, cell_cols, session_id, seed):
    rng = np.random.default_rng(stable_seed(session_id, condition, window, base=seed))
    recs = []
    for cell in cell_cols:
        for pred in PREDICTORS:
            pair = ct.pair_arrays(table, pred, cell)
            if pair is None:
                continue
            x, y, _ = pair
            mi = mutual_information_shuffle(x, y, bins=BINS, n_shuffles=N_SHUF, rng=rng)
            recs.append(dict(session_id=session_id, condition=condition, window=window, cell_id=cell,
                             predictor=pred, n_trials=mi.n, mi_bits_raw=mi.bits_raw,
                             mi_bits_shuffle_mean=mi.bits_shuffle_mean, mi_bits_corrected=mi.bits_corrected,
                             mi_p_value=mi.p_value))
    return recs


def cmi_records(table, window, cell_cols, session_id, seed):
    """Engaged trials only: for every (cell, predictor != runspeed) the plain MI and I(X;Y|runspeed) on the SAME trials
    (the pair's trials from pair_arrays, further restricted to finite running speed)."""
    if GIVEN not in table.columns:
        return []
    rng = np.random.default_rng(stable_seed(session_id, "given_runspeed", window, base=seed))
    z_by_trial = table.set_index("trialNumber")[GIVEN]
    recs = []
    for cell in cell_cols:
        for pred in PREDICTORS:
            if pred == GIVEN:
                continue
            pair = ct.pair_arrays(table, pred, cell)
            if pair is None:
                continue
            x, y, tnum = pair
            z = z_by_trial.reindex(tnum).to_numpy(float)
            ok = np.isfinite(z)
            if ok.sum() < cfg.MIN_TRIALS_ANALYSIS:
                continue
            r = mi_and_conditional_mi(x[ok], y[ok], z[ok], BINS, N_SHUF, rng)
            if not np.isfinite(r["cmi_corrected"]):
                continue
            recs.append(dict(session_id=session_id, window=window, cell_id=cell, predictor=pred, n_trials=r["n"],
                             mi_bits_raw=r["mi_raw"], mi_bits_corrected=r["mi_corrected"], mi_p_value=r["mi_p"],
                             cmi_bits_raw=r["cmi_raw"], cmi_bits_corrected=r["cmi_corrected"], cmi_p_value=r["cmi_p"]))
    return recs


def _codes(X, bins):
    """trials x cells -> int codes 0..bins-1 (-1 where non-finite); each cell binned on its own finite trials with the
    project's quantile edges (as mutual_information_hist)."""
    out = np.full(X.shape, -1, dtype=np.int16)
    for j in range(X.shape[1]):
        x = X[:, j]
        ok = np.isfinite(x)
        if ok.sum() < 2:
            continue
        edges = quantile_bin_edges(x[ok], bins)
        out[ok, j] = np.clip(np.digitize(x[ok], edges[1:-1]), 0, bins - 1)
    return out


def _onehot(codes, bins):
    T, n = codes.shape
    oh = np.zeros((T, n, bins), dtype=np.float32)
    r, c = np.nonzero(codes >= 0)
    oh[r, c, codes[r, c]] = 1.0
    return oh.reshape(T, n * bins)


def _mi_from_counts(N, bins, min_n):
    """N: (nA * bins, nB * bins) joint counts -> (nA, nB) MI in bits, NaN where fewer than min_n shared trials."""
    nA, nB = N.shape[0] // bins, N.shape[1] // bins
    C = N.reshape(nA, bins, nB, bins).astype(np.float64)
    tot = C.sum((1, 3))
    with np.errstate(divide="ignore", invalid="ignore"):
        p = C / tot[:, None, :, None]
        px = p.sum(3, keepdims=True)
        py = p.sum(1, keepdims=True)
        term = np.where(p > 0, p * np.log2(p / (px * py)), 0.0)
    mi = term.sum((1, 3))
    mi[tot < min_n] = np.nan
    return mi, tot


def _pair_mi(OA, OB, bins, strata, perms, min_n):
    """MI (strata None) or I(.;.|S) of every (A cell, B cell) pair. Returns raw, null mean, p (fraction of nulls >= raw).
    `perms`: row permutations of B (global, or within strata) used for the null."""
    def once(OBx):
        if strata is None:
            mi, _ = _mi_from_counts(OA.T @ OBx, bins, min_n)
            return mi
        num, den = 0.0, 0.0
        for s in np.unique(strata[strata >= 0]):
            idx = strata == s
            mi_s, tot_s = _mi_from_counts(OA[idx].T @ OBx[idx], bins, 1)
            w = np.where(np.isfinite(mi_s), tot_s, 0.0)
            num = num + w * np.nan_to_num(mi_s)
            den = den + w
        with np.errstate(divide="ignore", invalid="ignore"):
            mi = num / den
        mi[den < min_n] = np.nan
        return mi
    raw = once(OB)
    null_sum = np.zeros_like(raw)
    ge = np.zeros_like(raw)
    for perm in perms:
        m = once(OB[perm])
        null_sum += np.nan_to_num(m)
        ge += (np.nan_to_num(m) >= np.nan_to_num(raw))
    k = max(len(perms), 1)
    return raw, null_sum / k, ge / k


def between_pair_records(table, window, groups, new_pairs, session_id, seed):
    """MI (and MI given the stimulus) of the given cell PAIRS, engaged trials of `table`: one row per (pair, measure) with
    raw MI, bias-corrected MI (raw - mean of the permuted MIs), significance and the groups of the two cells. The permutations
    are drawn from a generator seeded by (session, window) BEFORE anything depends on the cells, so a pair gets the same null
    whichever other pairs are computed in the same call. Group-pair summaries (sums, n pairs) are built from these rows
    inside the worker (process_session): the per-pair rows themselves are never stored."""
    rng = np.random.default_rng(stable_seed(session_id, "between", window, base=seed))
    known = set(groups.index)
    pairs = [q for q in new_pairs if q[0] in known and q[1] in known and q[0] in table.columns and q[1] in table.columns]
    if not pairs or len(table) < cfg.MIN_TRIALS_ANALYSIS:
        return [], []
    U = sorted({c for q in pairs for c in q})
    pos = {c: i for i, c in enumerate(U)}
    X = table[U].to_numpy(float)
    grp = groups.loc[U].to_numpy()
    OH = _onehot(_codes(X, BINS), BINS)
    s_col = ct.stim_predictor(BETWEEN_STRAT)
    strata = None
    if s_col in table.columns:
        sv = table[s_col].to_numpy(float)
        strata = np.where(np.isfinite(sv), sv, -1).astype(int)
        vals, cnt = np.unique(strata[strata >= 0], return_counts=True)
        strata[np.isin(strata, vals[cnt < cfg.MIN_TRIALS_PER_TARGET_BIN])] = -1
        if len(np.unique(strata[strata >= 0])) < 2:
            strata = None
    T = len(table)
    glob_perms = [rng.permutation(T) for _ in range(N_SHUF)]
    strat_perms = []
    if strata is not None:
        for _ in range(N_SHUF):
            perm = np.arange(T)
            for s_ in np.unique(strata[strata >= 0]):
                idx = np.flatnonzero(strata == s_)
                perm[idx] = rng.permutation(idx)
            strat_perms.append(perm)
    measures = [("mi", None, glob_perms)] + ([("mi_given_stim", strata, strat_perms)] if strata is not None else [])
    pi = np.array([pos[q[0]] for q in pairs])
    pj = np.array([pos[q[1]] for q in pairs])
    ga, gb = grp[pi], grp[pj]
    lo, hi = np.where(ga <= gb, ga, gb), np.where(ga <= gb, gb, ga)
    ui = np.unique(pi)
    rows, checks = [], []
    for name, st, perms in measures:
        raw = np.full(len(pairs), np.nan)
        nul = np.full(len(pairs), np.nan)
        pv = np.full(len(pairs), np.nan)
        for a0 in range(0, len(ui), BETWEEN_CHUNK):
            rows_idx = ui[a0:a0 + BETWEEN_CHUNK]
            cols_a = (rows_idx[:, None] * BINS + np.arange(BINS)).ravel()
            r_, m_, p_ = _pair_mi(OH[:, cols_a], OH, BINS, st, perms, cfg.MIN_TRIALS_ANALYSIS)
            sel = np.flatnonzero(np.isin(pi, rows_idx))
            ridx = np.searchsorted(rows_idx, pi[sel])
            raw[sel], nul[sel], pv[sel] = r_[ridx, pj[sel]], m_[ridx, pj[sel]], p_[ridx, pj[sel]]
        good = np.isfinite(raw)
        rows.append(pd.DataFrame({
            "session_id": session_id, "window": window, "measure": name, "cell_i": [pairs[k][0] for k in np.flatnonzero(good)],
            "cell_j": [pairs[k][1] for k in np.flatnonzero(good)], "group_a": lo[good], "group_b": hi[good],
            "raw": raw[good], "corr": np.maximum(raw[good] - nul[good], 0.0), "sig": (pv[good] < ALPHA).astype(float)}))
        if args.check_between and name == "mi":                  # parity with the single-pair code (finite pairs only)
            fin = np.isfinite(X).all(0)
            cand = [k for k in np.flatnonzero(good) if fin[pi[k]] and fin[pj[k]]]
            pick = rng.choice(cand, size=min(args.check_between, len(cand)), replace=False) if cand else []
            diffs = [abs(mutual_information_hist(X[:, pi[k]], X[:, pj[k]], bins=BINS) - raw[k]) for k in pick]
            if diffs:
                checks.append(dict(session_id=session_id, window=window, n_checked=len(diffs), max_abs_diff=max(diffs)))
    return rows, checks


def _frame_codes(x, bins):
    """Bin codes of x for the pooled-frames MI; returns (codes, number of bins actually used).
    Continuous x: the project's equal-occupancy edges (info_theory.quantile_bin_edges), identical to mutual_information_hist.
    ZERO-INFLATED x (one value, typically 0, holds more than 1/bins of the frames: lick rate, deconvolved activity): plain
    quantile edges would merge most bins (with > 90% zeros a single bin is left and the MI is exactly 0), so that dominant
    value gets its own bin and the remaining bins are equal-occupancy over the other frames."""
    vals, cnt = np.unique(x, return_counts=True)
    top = vals[cnt.argmax()]
    if len(vals) > bins and cnt.max() / len(x) > 1.0 / bins:
        m = x != top
        e = quantile_bin_edges(x[m], bins - 1)
        codes = np.zeros(len(x), int)
        codes[m] = np.clip(np.searchsorted(e[1:-1], x[m], side="right"), 0, len(e) - 2) + 1
        return codes, len(e)
    e = quantile_bin_edges(x, bins)
    return np.clip(np.searchsorted(e[1:-1], x, side="right"), 0, len(e) - 2), len(e) - 1


def _frame_mi(cx, nx, cy, ny):
    c = np.bincount(cx * ny + cy, minlength=nx * ny).reshape(nx, ny).astype(float)
    p = c / c.sum()
    prod = p.sum(1, keepdims=True) * p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / prod[nz])).sum())


def frame_mi_records(frames, window, cell_cols, session_id, seed):
    """Pooled-frames MI: for every (cell, behavior) the frames of ALL engaged trials' windows are pooled into one distribution
    (FBINS equal-occupancy bins each). Frames are autocorrelated, so the null circularly rolls the cell trace (offset >=
    FRAME_SHIFT_MIN_TRIALS windows' worth of frames) instead of permuting frames: bias correction = mean rolled MI, p = fraction
    of rolled MIs >= raw. `n_trials` holds the number of FRAMES of these predictors (names end in __frames)."""
    rng = np.random.default_rng(stable_seed(session_id, "frames", window, base=seed))
    per_trial = frames.groupby("trialNumber").size()
    min_shift = int(max(1, round(cfg.FRAME_SHIFT_MIN_TRIALS * float(per_trial.median()))))
    recs = []
    for fp in FRAME_PREDICTORS:
        col = fp[:-len("__frames")]
        if col not in frames.columns:
            continue
        xv = frames[col].to_numpy(float)
        for cell in cell_cols:
            yv = frames[cell].to_numpy(float)
            ok = np.isfinite(xv) & np.isfinite(yv)
            n = int(ok.sum())
            if n < cfg.MIN_FRAMES_ANALYSIS or n <= 2 * min_shift + 1:
                continue
            cx, nx = _frame_codes(xv[ok], FBINS)
            cy, ny = _frame_codes(yv[ok], FBINS)
            raw = _frame_mi(cx, nx, cy, ny)
            null = np.array([_frame_mi(cx, nx, np.roll(cy, int(o)), ny)
                             for o in rng.integers(min_shift, n - min_shift, N_SHUF)])
            recs.append(dict(session_id=session_id, condition="engaged", window=window, cell_id=cell, predictor=fp,
                             n_trials=n, mi_bits_raw=raw, mi_bits_shuffle_mean=float(null.mean()),
                             mi_bits_corrected=max(raw - float(null.mean()), 0.0), mi_p_value=float(np.mean(null >= raw))))
    return recs


def process_session(ses, seed, new_cells, new_pairs, req_cells):
    sid = ses.session_id
    cells = set(new_cells)                                     # the requested cells (session_jobs.run_sessions_units)
    need = cells | {c for q in new_pairs for c in q}           # cells whose data the new pairs need
    td = ses.trialdata
    if not need or td is None or "engaged" not in td.columns:
        return None
    engaged_nums = td.loc[td["engaged"] == 1, "trialNumber"].astype(int).tolist()
    tail = eg.tail_condition_trials(td, args.tail_min_frac, args.tail_min_trials) if RUN_TAIL else None
    wanted = set(engaged_nums) | (set(tail["disengaged"]) if tail else set())
    tables = build_session_tables(ses, mu_sigma_lookup, WINDOWS, included_cell_ids=need, trial_numbers=wanted)
    if not tables:
        return None
    sets = {"engaged": set(engaged_nums)}
    if tail:
        sets.update({"engaged_matched": set(tail["engaged_matched"]), "disengaged": set(tail["disengaged"])})
    recs, crecs, brecs, bchecks = [], [], [], []
    if RUN_FRAMES and cells:
        frames = build_window_frames(ses, WINDOWS, included_cell_ids=cells, trial_numbers=engaged_nums)
        for window, fdf in frames.items():
            recs += frame_mi_records(fdf, window, ct.cell_columns(fdf, cells), sid, seed)
        del frames
    groups = None
    if RUN_BETWEEN and new_pairs and ses.celldata is not None:
        cdi = ses.celldata[ses.celldata["cell_id"].isin(need)]
        groups = add_six_group_column(cdi, area_var="roi_name", label_var="labeled").set_index("cell_id")["group"]
    for window, table in tables.items():
        if table.empty:
            continue
        cell_cols = ct.cell_columns(table, cells)
        if groups is not None:
            b_, c_ = between_pair_records(table[table["trialNumber"].isin(sets["engaged"])], window, groups, new_pairs, sid, seed)
            brecs += b_
            bchecks += c_
        if not cells:
            continue
        for cond, nums in sets.items():
            sub = table[table["trialNumber"].isin(nums)]
            if len(sub) < cfg.MIN_TRIALS_ANALYSIS:
                continue
            sub = ct.add_stimulus_columns(sub, STRATEGIES)
            recs += mi_records(sub, cond, window, cell_cols, sid, seed)
            if RUN_GIVEN and cond == "engaged":
                crecs += cmi_records(sub, window, cell_cols, sid, seed)
    between = pd.DataFrame()           # group-pair summary of the between-area MI; the per-pair rows are NOT kept
    if brecs:
        bp = pd.concat(brecs, ignore_index=True)
        between = (bp.groupby(["window", "measure", "group_a", "group_b"])
                     .agg(n_pairs=("raw", "size"), sum_raw=("raw", "sum"), sum_corr=("corr", "sum"), n_sig=("sig", "sum")).reset_index())
        between.insert(0, "animal_id", ses.animal_id)
        between.insert(0, "session_id", sid)
        del bp
    return dict(session_id=sid, animal_id=ses.animal_id, records=pd.DataFrame(recs), cmi=pd.DataFrame(crecs),
                between=between, between_checks=bchecks, tail=tail, trials=td[["trialNumber", "engaged"]].copy())


results = run_sessions_units(
    session_ids, process_session, [PROTOCOL], paths.store, "mi_session",
    key_params=dict(version="2c-v6", frames=[RUN_FRAMES, FBINS, cfg.MIN_FRAMES_ANALYSIS, cfg.FRAME_SHIFT_MIN_TRIALS, cfg.NORMALIZE_FACE_PER_SESSION], given=[RUN_GIVEN, GIVEN], between=[RUN_BETWEEN, BETWEEN_STRAT, args.check_between], windows=WINDOWS, strategies=target_signature(STRATEGIES),
                    bins=BINS, method=cfg.RESPONSE_BIN_METHOD, n_shuffles=N_SHUF,
                    min_trials=cfg.MIN_TRIALS_ANALYSIS, min_per_bin=cfg.MIN_TRIALS_PER_TARGET_BIN,
                    tail=[RUN_TAIL, args.tail_min_frac, args.tail_min_trials],
                    fits_mtime=int(fits_path.stat().st_mtime)),
    load_kwargs=dict(load_behaviordata=True, load_videodata=True, load_celldata=True, load_calciumdata=True),
    cells_by_session=cells_by_session,
    parts={"records": "cell", "cmi": "cell", **({"between": "pair"} if RUN_BETWEEN else {})},
    extras={"between_checks": "extend"}, groups_by_cell=cell_group, max_cells=scope.max_cells, max_pairs=scope.max_pairs,
    base_seed=RANDOM_STATE, n_jobs=N_JOBS, recompute=args.recompute,
    verbose=JOBLIB_VERBOSE, extra_key_by_session=stamps)
results = [r for r in results if r is not None]
if not results:
    raise SystemExit("No session produced tables (no celldata / calciumdata / behaviordata?).")
for r in results:
    if r.get("between") is None:
        r["between"] = pd.DataFrame()
    r.setdefault("between_checks", [])

all_rec = pd.concat([r["records"] for r in results], ignore_index=True)
if all_rec.empty:
    raise SystemExit("No (cell, predictor) pair had enough valid trials.")
cd = pd.read_csv(cell_dist_out / "celldata_combined.csv", low_memory=False)
all_rec = ct.attach_cell_meta(all_rec, cd)
all_rec["significant"] = all_rec["mi_p_value"] < ALPHA
all_rec["sampling"] = np.where(all_rec["predictor"].str.endswith("__frames"), "frame", "trial")

mi_all = all_rec[all_rec["condition"] == "engaged"].drop(columns="condition")
mi_all[mi_all["sampling"] == "trial"].to_csv(table_dir / "cell_predictor_mi.csv", index=False)   # per-trial: what 2d reads
if RUN_FRAMES:
    mi_all[mi_all["sampling"] == "frame"].to_csv(table_dir / "cell_predictor_mi_frames.csv", index=False)
mi_table = mi_all                      # figures / group tables below show both families (predictor "<cur_var>__frames")
print(f"\n{len(mi_table)} (session, window, cell, predictor) MI values (engaged trials)")
if RUN_FRAMES and (mi_table["sampling"] == "frame").any():
    fr_ = mi_table[mi_table["sampling"] == "frame"].groupby("window")["n_trials"].median()
    print(f"Pooled-frames MI: median frames per (cell, behavior) pair by window = {fr_.round(0).astype(int).to_dict()} "
          f"-> ~{(fr_ / FBINS ** 2).round(0).astype(int).to_dict()} frames per joint bin ({FBINS}x{FBINS}) if uniform")

BY = ["window", "predictor"]


def write_group_table(df, by, name, mean_cols=("mi_bits_corrected", "mi_bits_raw")):
    t = ct.unit_group_summary(df, by, AGG, list(mean_cols), sig_col="significant")
    t.to_csv(table_dir / f"{name}{AGG_SUFFIX}.csv", index=False)
    return t


write_group_table(mi_table, BY, "mi_by_group")

# --------------------------------------------------------------------- #
# 3. Reporting: strongest combinations + confound check (unit-level when not pooled)
# --------------------------------------------------------------------- #
UNIT_KEYS = {"pooled": [], "session": ["session_id"], "mouse": ["animal_id"]}[AGG]


def frac_sig_by(df, by):
    """Fraction of significant cells per `by`, at the chosen unit: mean over units of the per-unit fraction."""
    per = df.groupby(by + UNIT_KEYS)["significant"].mean().reset_index()
    return per.groupby(by)["significant"].mean()


frac = frac_sig_by(mi_table, BY)
print(f"\nStrongest (window, predictor) by fraction of significant cells (p<{ALPHA}; mean over {UNITS} of the "
      f"per-{UNIT} fraction):" if AGG != "pooled" else f"\nStrongest (window, predictor) by fraction of significant cells (p<{ALPHA}):")
print(frac.sort_values(ascending=False).head(10).round(3).to_string())
check_cols = [p for p in ["cur_runspeed", "choice", "correct"] if p in set(mi_table["predictor"])]
if check_cols:
    chk = frac.unstack("predictor")[check_cols]
    print("\nConfound check: cur_runspeed's significant fraction vs choice / correct (similar magnitudes suggest the "
          "frame-count/precision confound, see docstring):")
    print(chk.round(3).to_string())
    chk.reset_index().to_csv(table_dir / f"confound_check{AGG_SUFFIX}.csv", index=False)


# --------------------------------------------------------------------- #
# 4. Figures: six groups at the chosen unit, one panel per predictor
# --------------------------------------------------------------------- #
def predictor_panels(df, window, value, name, ylabel, title, hline=None, series_col=None, series_order=None,
                     series_colors=None):
    group_panels(df[df["window"] == window], "predictor", PANEL_PREDICTORS, value, AGG, figdir, name, title,
                 ylabel=ylabel, hline=hline, series_col=series_col, series_order=series_order,
                 series_colors=series_colors, mouse_colors=mouse_colors, min_cells_per_unit=MIN_CELLS_PER_UNIT)


agg_note = {"pooled": "all cells pooled (no error bars)",
            "session": "bar = mean +- SEM over sessions, dots = sessions",
            "mouse": "bar = mean +- SEM over mice, dots = mice"}[AGG]
# Figures 1 / 2: one row per predictor, one column per window (pre_stim / stim / reward), y shared along each row so the windows
# can be compared directly. The bars inside a panel keep the fixed project group order and colours (so a group is always in the
# same place and colour in every panel); what is ordered is the ROWS: strongest predictor (mean MI over groups, windows and
# units) on top. The trial-window predictors and the pooled-frames ones (another estimator, 8 bins) are drawn in separate figures.
mi_win = [w for w in WINDOWS if w in set(mi_table["window"])]
win_titles = {w: f"{w}  [{WINDOWS[w][0]:g}, {WINDOWS[w][1]:g}] cm" for w in mi_win}


def ordered_predictors(preds):
    """`preds` sorted by mean MI (mean over windows, units and the project groups, equal weight per group), strongest first."""
    preds = [p for p in preds if p in set(mi_table["predictor"])]
    if not SORT_PREDICTORS or not preds:
        return preds
    m = (mi_table[mi_table["predictor"].isin(preds)].groupby(["predictor", "group"] + UNIT_KEYS)["mi_bits_corrected"].mean()
         .groupby(["predictor", "group"]).mean().groupby("predictor").mean())
    return list(m.sort_values(ascending=False).index) + [p for p in preds if p not in m.index]


mi_table_sig = mi_table.assign(sig=mi_table["significant"].astype(float))
for part, preds, what in (("a", PREDICTORS, "per-trial window means"), ("b", FRAME_PREDICTORS, "pooled frames")):
    order_ = ordered_predictors(preds)
    if not order_:
        continue
    suffix = "" if part == "a" else "_frames"
    order_note = "rows ordered by mean MI" if SORT_PREDICTORS else "rows in fixed order"
    group_grid(mi_table, "predictor", order_, "window", mi_win, "mi_bits_corrected", AGG, figdir,
               f"1{part}_mi_by_predictor_and_window{suffix}",
               f"Per-cell MI with each predictor ({what}) in the three windows -- {PROTOCOL}; {agg_note}; {order_note}; y shared along each row",
               ylabel="MI (bits, bias-corrected)", mouse_colors=mouse_colors, min_cells_per_unit=MIN_CELLS_PER_UNIT,
               col_titles=win_titles)
    group_grid(mi_table_sig, "predictor", order_, "window", mi_win, "sig", AGG, figdir,
               f"2{part}_frac_significant_by_predictor_and_window{suffix}",
               f"Fraction of cells with significant MI ({what}) -- {PROTOCOL}; dashed = chance ({ALPHA}); {agg_note}; same row order as figure 1{part}",
               ylabel=f"fraction of cells p<{ALPHA}", hline=ALPHA, mouse_colors=mouse_colors,
               min_cells_per_unit=MIN_CELLS_PER_UNIT, col_titles=win_titles)

piv = frac.unstack("window").reindex(index=PANEL_PREDICTORS, columns=[w for w in WINDOWS if w in set(mi_table["window"])])
fig, ax = plt.subplots(figsize=(6, max(3, 0.5 * len(piv))))
im = ax.imshow(piv.to_numpy(float), aspect="auto", cmap="viridis", vmin=0, vmax=1)
ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns)
ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index, fontsize=8)
for i in range(piv.shape[0]):
    for j in range(piv.shape[1]):
        v = piv.to_numpy(float)[i, j]
        if np.isfinite(v):
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", color="white" if v < 0.5 else "black", fontsize=8)
fig.colorbar(im, ax=ax, label="fraction of cells with p<0.05")
ax.set_title(f"Fraction of cells encoding each predictor -- {PROTOCOL}\n"
             f"({'all cells pooled' if AGG == 'pooled' else f'mean over {UNITS} of the per-{UNIT} fraction'})", fontsize=9)
fig.tight_layout()
save_fig(fig, figdir, "3_frac_significant_heatmap")
plt.close(fig)


# --------------------------------------------------------------------- #
# 4b. Information that remains once running speed is accounted for: I(X;Y | runspeed)
#     (after Safaai et al. 2025, Fig. 3b: what is each class enriched for once movement is conditioned out?)
# --------------------------------------------------------------------- #
if RUN_GIVEN:
    cparts = [r["cmi"] for r in results if len(r["cmi"])]
    if cparts:
        cmi_table = ct.attach_cell_meta(pd.concat(cparts, ignore_index=True), cd)
        cmi_table["significant"] = cmi_table["mi_p_value"] < ALPHA
        cmi_table["significant_given"] = cmi_table["cmi_p_value"] < ALPHA
        cmi_table.to_csv(table_dir / "cell_predictor_mi_given_runspeed.csv", index=False)
        print(f"\n{len(cmi_table)} (session, window, cell, predictor) pairs with MI and MI | running speed "
              f"(same trials; running speed itself excluded)")
        t = ct.unit_group_summary(cmi_table, BY, AGG, ["mi_bits_corrected", "cmi_bits_corrected"], sig_col="significant")
        tg = ct.unit_group_summary(cmi_table, BY, AGG, ["significant_given"])
        t["frac_sig_given_runspeed"] = tg["significant_given"].to_numpy()
        t = t.rename(columns={"frac_sig": "frac_sig_plain", "n_sig": "n_sig_plain"})
        t.to_csv(table_dir / f"mi_given_runspeed_by_group{AGG_SUFFIX}.csv", index=False)
        summ_ = (cmi_table.assign(sp=cmi_table["significant"].astype(float), sg=cmi_table["significant_given"].astype(float))
                 .groupby(UNIT_KEYS + ["predictor"])[["sp", "sg"]].mean().groupby("predictor").mean())
        print(f"Fraction of cells significant, plain vs given running speed (mean over {UNITS}):" if UNIT_KEYS
              else "Fraction of cells significant, plain vs given running speed:")
        print(summ_.rename(columns={"sp": "MI", "sg": "MI | runspeed"}).round(3).to_string())
        long_bits = pd.concat([cmi_table.assign(measure="MI", val=cmi_table["mi_bits_corrected"]),
                               cmi_table.assign(measure="MI | runspeed", val=cmi_table["cmi_bits_corrected"])],
                              ignore_index=True)
        long_sig = pd.concat([cmi_table.assign(measure="MI", val=cmi_table["significant"].astype(float)),
                              cmi_table.assign(measure="MI | runspeed", val=cmi_table["significant_given"].astype(float))],
                             ignore_index=True)
        g_preds = [p for p in PREDICTORS if p != GIVEN]
        for w in WINDOWS:
            for df_, val, nm, yl, ttl in ((long_bits, "val", f"7_mi_given_runspeed_{w}", "MI (bits, bias-corrected)",
                                           f"MI with and without conditioning on running speed -- {w}; {agg_note}"),
                                          (long_sig, "val", f"8_frac_sig_given_runspeed_{w}", f"fraction of cells p<{ALPHA}",
                                           f"Fraction of significant cells, plain vs given running speed -- {w}; dashed = chance; {agg_note}")):
                d_ = df_[df_["window"] == w]
                if d_.empty:
                    continue
                group_panels(d_, "predictor", [p for p in g_preds if p in set(d_["predictor"])], val, AGG, figdir, nm, ttl,
                             ylabel=yl, hline=ALPHA if nm.startswith("8") else None, series_col="measure",
                             series_order=["MI", "MI | runspeed"], series_colors={"MI": "#7f7f7f", "MI | runspeed": "#d62728"},
                             mouse_colors=mouse_colors, min_cells_per_unit=MIN_CELLS_PER_UNIT)


# --------------------------------------------------------------------- #
# 4c. Information between areas: MI of every pair of cells (cross-area and within-area), see the docstring
# --------------------------------------------------------------------- #
bsess = None
if RUN_BETWEEN:
    bparts = [r["between"] for r in results if len(r["between"])]
    if bparts:
        bsess = pd.concat(bparts, ignore_index=True)
        bsess["mean_mi_raw"] = bsess["sum_raw"] / bsess["n_pairs"]
        bsess["mean_mi_corrected"] = bsess["sum_corr"] / bsess["n_pairs"]
        bsess["frac_sig"] = bsess["n_sig"] / bsess["n_pairs"]
        bsess.drop(columns=["sum_raw", "sum_corr"]).to_csv(table_dir / "between_area_mi_per_session.csv", index=False)
        if args.check_between:
            chk = pd.DataFrame([c for r in results for c in r["between_checks"]])
            if len(chk):
                chk.to_csv(table_dir / "between_area_mi_check.csv", index=False)
                print(f"\nBetween areas, check vs the single-pair code: {int(chk['n_checked'].sum())} pairs, largest |difference| "
                      f"= {chk['max_abs_diff'].max():.2e} bits")
        area_of = lambda g: g.split("_")[0]
        order_a = {a: i for i, a in enumerate(AREAS)}

        def unit_table(df, keys):
            u = df.groupby(keys + ["window", "measure"] + UNIT_KEYS)[["n_pairs", "sum_corr", "sum_raw", "n_sig"]].sum().reset_index()
            u["mi_corrected"] = u["sum_corr"] / u["n_pairs"]
            u["mi_raw"] = u["sum_raw"] / u["n_pairs"]
            u["frac_sig"] = u["n_sig"] / u["n_pairs"]
            if AGG != "pooled":
                u = u[u["n_pairs"] >= BETWEEN_MIN_PAIRS]
                u["unit"] = u["session_id"] if AGG == "session" else u["animal_id"]
                if AGG == "session":
                    u["animal_id"] = u["session_id"].str.split("_").str[0]
            return u

        bg = unit_table(bsess, ["group_a", "group_b"])
        bg.drop(columns=["sum_corr", "sum_raw", "n_sig"]).to_csv(table_dir / f"between_area_mi_by_group_pair{AGG_SUFFIX}.csv", index=False)
        ba = bsess.assign(area_a=bsess["group_a"].map(area_of), area_b=bsess["group_b"].map(area_of))
        ba = ba[ba["area_a"].isin(AREAS) & ba["area_b"].isin(AREAS)]          # drop cells outside the analysed groups ("excluded")
        flip = ba["area_a"].map(order_a) > ba["area_b"].map(order_a)
        ba.loc[flip, ["area_a", "area_b"]] = ba.loc[flip, ["area_b", "area_a"]].to_numpy()
        ba = unit_table(ba, ["area_a", "area_b"])
        ba["area_pair"] = ba["area_a"] + " x " + ba["area_b"]
        ba.drop(columns=["sum_corr", "sum_raw", "n_sig"]).to_csv(table_dir / f"between_area_mi_by_area_pair{AGG_SUFFIX}.csv", index=False)

        measures = [m for m in ("mi", "mi_given_stim") if m in set(ba["measure"])]
        mlabel = {"mi": "I(X;Y)", "mi_given_stim": "I(X;Y | S)"}
        wins = [w for w in WINDOWS if w in set(ba["window"])]
        pair_order = [f"{a} x {b}" for i, a in enumerate(AREAS) for b in AREAS[i:]]
        pair_order = [p for p in pair_order if p in set(ba["area_pair"])]

        def unit_mean(df, keycols, val):
            """mean over units of the per-unit value (pooled: the single pooled value)."""
            return df.groupby(keycols)[val].mean()

        print(f"\nInformation between areas, mean bias-corrected MI (bits) over {UNITS}; diagonal = within area:" if AGG != "pooled"
              else "\nInformation between areas, mean bias-corrected MI (bits); diagonal = within area:")
        for m in measures:
            t = unit_mean(ba[ba["measure"] == m], ["window", "area_pair"], "mi_corrected").unstack("window")
            print(f"  {mlabel[m]}\n" + t.reindex(pair_order).round(4).to_string())

        # 9. area x area heatmaps (rows: measure, columns: window)
        fig, axes = plt.subplots(len(measures), len(wins), figsize=(4.9 * len(wins), 4.1 * len(measures)), squeeze=False)
        for i, m in enumerate(measures):
            sub_m = ba[ba["measure"] == m]
            mats = {}
            for w in wins:
                v = unit_mean(sub_m[sub_m["window"] == w], ["area_a", "area_b"], "mi_corrected")
                M = np.full((len(AREAS), len(AREAS)), np.nan)
                for (a, b), x in v.items():
                    M[AREAS.index(a), AREAS.index(b)] = M[AREAS.index(b), AREAS.index(a)] = x
                mats[w] = M
            vmax = np.nanmax(np.concatenate([M.ravel() for M in mats.values()])) if mats else 1
            for j, w in enumerate(wins):
                ax = axes[i][j]
                im = ax.imshow(mats[w], cmap="viridis", vmin=0, vmax=vmax)
                ax.set_xticks(range(len(AREAS))); ax.set_xticklabels([AREA_NAME.get(a, a) for a in AREAS], rotation=30, ha="right", fontsize=8)
                ax.set_yticks(range(len(AREAS))); ax.set_yticklabels([AREA_NAME.get(a, a) for a in AREAS], fontsize=8)
                for a in range(len(AREAS)):
                    for b in range(len(AREAS)):
                        if np.isfinite(mats[w][a, b]):
                            ax.text(b, a, f"{mats[w][a, b]:.3f}", ha="center", va="center", fontsize=7,
                                    color="white" if mats[w][a, b] < 0.6 * vmax else "black")
                ax.set_title(f"{mlabel[m]} -- {w}", fontsize=8)
            fig.colorbar(im, ax=axes[i].tolist(), label="bits (bias-corrected)", shrink=0.8)
        fig.suptitle(f"Information between areas: mean MI of cell pairs, diagonal = within area; S = stimulus (4 lick-probability bins); "
                     f"{'all pairs pooled' if AGG == 'pooled' else f'mean over {UNITS}'}", y=1.02, fontsize=9)
        save_fig(fig, figdir, "9_between_area_mi_heatmap")
        plt.close(fig)

        # 10. bars per area pair (one figure per window, one panel per measure, shared y)
        for w in wins:
            fig, axes = plt.subplots(1, len(measures), figsize=(5.2 * len(measures), 3.8), squeeze=False, sharey=True)
            for ax, m in zip(axes[0], measures):
                d = ba[(ba["measure"] == m) & (ba["window"] == w)].copy()
                unit_bars(ax, d, "area_pair", "mi_corrected", pair_order, AGG, mouse_colors=mouse_colors,
                          ylabel="MI (bits, bias-corrected)")
                ax.set_xticks(range(len(pair_order)))
                ax.set_xticklabels([" x ".join(AREA_NAME.get(t, t) for t in p_.split(" x ")) for p_ in pair_order], rotation=45, ha="right", fontsize=7)
                ax.set_title(mlabel[m], fontsize=9)
            fig.suptitle(f"Information between areas, pairs of cells -- {w} window; S = stimulus (4 lick-probability bins); "
                         f"{agg_note.replace('cells', 'pairs')}", y=1.03, fontsize=9)
            fig.tight_layout()
            save_fig(fig, figdir, f"10_between_area_mi_bars_{w}")
            plt.close(fig)

        # 11. group x group heatmap (the six project groups)
        gorder = [group_key(g) for g in SIX_GROUP_ORDER]
        gnames = [SIX_GROUP_LABELS[g] for g in SIX_GROUP_ORDER]
        fig, axes = plt.subplots(len(measures), len(wins), figsize=(3.9 * len(wins), 3.6 * len(measures)), squeeze=False)
        for i, m in enumerate(measures):
            sub_m = bg[bg["measure"] == m]
            mats = {}
            for w in wins:
                v = unit_mean(sub_m[sub_m["window"] == w], ["group_a", "group_b"], "mi_corrected")
                M = np.full((len(gorder), len(gorder)), np.nan)
                for (a, b), x in v.items():
                    if a in gorder and b in gorder:
                        M[gorder.index(a), gorder.index(b)] = M[gorder.index(b), gorder.index(a)] = x
                mats[w] = M
            vmax = np.nanmax(np.concatenate([M.ravel() for M in mats.values()])) if mats else 1
            for j, w in enumerate(wins):
                ax = axes[i][j]
                im = ax.imshow(mats[w], cmap="viridis", vmin=0, vmax=vmax)
                ax.set_xticks(range(len(gnames))); ax.set_xticklabels(gnames, rotation=45, ha="right", fontsize=7)
                ax.set_yticks(range(len(gnames))); ax.set_yticklabels(gnames, fontsize=7)
                ax.set_title(f"{mlabel[m]} -- {w}", fontsize=8)
            fig.colorbar(im, ax=axes[i].tolist(), label="bits (bias-corrected)", shrink=0.8)
        fig.suptitle(f"Information between the six groups (cell pairs; diagonal = within group); S = stimulus (4 lick-probability bins)", y=1.02, fontsize=9)
        save_fig(fig, figdir, "11_between_group_mi_heatmap")
        plt.close(fig)
    else:
        print("\nBetween areas: no session had enough cells / trials")


# --------------------------------------------------------------------- #
# 4d. LABELED vs UNLABELED, PAIRED: the same session (or mouse) gives one value for the labeled and one for the unlabeled cells
#     of an area (V1, PM -- AL and RSP have no labeled split), joined by a line. Test = Wilcoxon signed-rank on the units'
#     differences (paired t-test next to it), Holm-corrected across the comparisons of one figure. A unit needs
#     MIN_CELLS_PER_UNIT cells (BETWEEN_MIN_PAIRS pairs) on BOTH sides. Pooled aggregate: the pairing is still made per SESSION
#     (a paired test needs units). Tables: labeled_vs_unlabeled_paired.csv (one row per comparison, with the statistics) and
#     labeled_vs_unlabeled_units.csv (the paired values).
#       12 / 13  cell MI / fraction significant, per predictor (section 4's data)
#       14       information between areas: MI of labeled-in-A x area P vs unlabeled-in-A x area P cell pairs (P = A: pairs
#                with both cells labeled vs both unlabeled)
# --------------------------------------------------------------------- #
PAIR_AREAS = ["V1", "PM"] + (["PMdeep"] if _DEEP_ON else [])
PAIR_KEY = ["animal_id"] if AGG == "mouse" else ["session_id"]
PAIR_UNIT = "mouse" if AGG == "mouse" else "session"
LAB_COLORS = {a: (SIX_GROUP_COLORS[(a, "lab")], SIX_GROUP_COLORS[(a, "unl")]) for a in PAIR_AREAS
              if (a, "lab") in SIX_GROUP_COLORS and (a, "unl") in SIX_GROUP_COLORS}


def _holm(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok])]
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(order) - rank) * p[i]))
        out[i] = run
    return out


def _paired_stats(wide):
    """wide: one row per unit with columns lab, unl. -> dict of statistics."""
    d = wide.dropna(subset=["lab", "unl"])
    n = len(d)
    out = dict(n_units=n, mean_lab=d["lab"].mean() if n else np.nan, mean_unl=d["unl"].mean() if n else np.nan)
    out["mean_diff_lab_minus_unl"] = out["mean_lab"] - out["mean_unl"] if n else np.nan
    out["n_lab_higher"] = int((d["lab"] > d["unl"]).sum())
    diff = (d["lab"] - d["unl"]).to_numpy(float)
    out["p_wilcoxon"] = out["p_ttest"] = np.nan
    if n >= 5 and np.any(diff != 0):
        try:
            out["p_wilcoxon"] = float(sps.wilcoxon(diff).pvalue)
        except ValueError:
            pass
        out["p_ttest"] = float(sps.ttest_rel(d["lab"], d["unl"]).pvalue)
    return out


def _stars(p):
    return "" if not np.isfinite(p) else "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def paired_figure(panels, comps, name, title, ylabel, panel_ncols):
    """panels: list of (panel title, {area: wide DataFrame(unit, animal_id, lab, unl)}); comps: stats rows are appended
    by the caller beforehand (with `p_holm`); draws one panel per entry, one pair of columns per area, lines per unit."""
    n = len(panels)
    nrows = int(np.ceil(n / panel_ncols))
    fig, axes = plt.subplots(nrows, panel_ncols, figsize=(3.6 * panel_ncols, 3.4 * nrows), squeeze=False, sharey=True)
    for ax, (ptitle, byarea) in zip(axes.flat, panels):
        xt, xl = [], []
        for k, area in enumerate(PAIR_AREAS):
            w = byarea.get(area)
            x0 = 2.4 * k
            xt += [x0, x0 + 1]
            xl += [f"{AREA_NAME.get(area, area)}-lab", f"{AREA_NAME.get(area, area)}-unl"]
            if w is None or w.dropna(subset=["lab", "unl"]).empty:
                continue
            w = w.dropna(subset=["lab", "unl"])
            for _, r in w.iterrows():
                ax.plot([x0, x0 + 1], [r["lab"], r["unl"]], color="grey", alpha=0.45, linewidth=0.8, zorder=1)
            cl, cu = LAB_COLORS.get(area, ("#4c72b0", "#c44e52"))
            for xx, col, key in ((x0, cl, "lab"), (x0 + 1, cu, "unl")):
                dc = [mouse_colors.get(a, "k") for a in w["animal_id"]]
                ax.scatter(np.full(len(w), xx), w[key], s=14, c=dc, edgecolor="black", linewidth=0.3, zorder=3)
                ax.hlines(w[key].mean(), xx - 0.28, xx + 0.28, color=col, linewidth=3, zorder=4)
            st = comps.get((ptitle, area))
            if st is not None:
                top = max(w["lab"].max(), w["unl"].max())
                ax.text(x0 + 0.5, top, f"{_stars(st['p_holm'])}\np={st['p_wilcoxon']:.3f} (n={st['n_units']})" if np.isfinite(st["p_wilcoxon"])
                        else f"n={st['n_units']}", ha="center", va="bottom", fontsize=6)
        ax.set_xticks(xt)
        ax.set_xticklabels(xl, rotation=45, ha="right", fontsize=7)
        ax.set_title(ptitle, fontsize=8)
        ax.set_xlim(-0.6, 2.4 * (len(PAIR_AREAS) - 1) + 1.6)
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel, fontsize=8)
    fig.suptitle(title, y=1.0, fontsize=9)
    fig.tight_layout()
    save_fig(fig, figdir, name)
    plt.close(fig)


def _unit_meta(df):
    df = df.copy()
    if "animal_id" not in df.columns:
        df["animal_id"] = df["session_id"].str.split("_").str[0]
    return df


paired_rows, unit_rows = [], []


def run_family(family, window, items, name, title, ylabel, ncols):
    """items: list of (panel title, area, wide). Computes the statistics, Holm within the figure, draws it."""
    comps, rows = {}, []
    for ptitle, area, wide in items:
        st = _paired_stats(wide)
        st.update(family=family, window=window, comparison=ptitle, area=area)
        rows.append(st)
        for _, r in wide.dropna(subset=["lab", "unl"]).iterrows():
            unit_rows.append(dict(family=family, window=window, comparison=ptitle, area=area, unit=r["unit"],
                                  animal_id=r["animal_id"], lab=r["lab"], unl=r["unl"]))
    for st, ph in zip(rows, _holm([r["p_wilcoxon"] for r in rows])):
        st["p_holm"] = ph
        comps[(st["comparison"], st["area"])] = st
    paired_rows.extend(rows)
    panels = []
    for ptitle in dict.fromkeys(i[0] for i in items):
        panels.append((ptitle, {a: w for t, a, w in items if t == ptitle}))
    paired_figure(panels, comps, name, title, ylabel, ncols)


def _wide(df, key_cols, lab_col):
    """long table with column `label` in {lab, unl} and `val` -> one row per unit with lab / unl."""
    w = df.pivot_table(index=key_cols, columns="label", values="val", aggfunc="mean").reset_index()
    for c in ("lab", "unl"):
        if c not in w.columns:
            w[c] = np.nan
    w["unit"] = w[key_cols[0]]
    return _unit_meta(w.rename(columns={key_cols[0]: key_cols[0]}))


cells_ = mi_table[mi_table["group"].isin([f"{a}_{l}" for a in PAIR_AREAS for l in ("lab", "unl")])].copy()
cells_["area_"] = cells_["group"].str.split("_").str[0]
cells_["label"] = cells_["group"].str.split("_").str[1]
cells_["sig_"] = cells_["significant"].astype(float)
for w_ in WINDOWS:
    cw = cells_[cells_["window"] == w_]
    if cw.empty:
        continue
    for val, fam, nm, yl, ttl in (("mi_bits_corrected", "cell_mi", f"12_labeled_vs_unlabeled_mi_{w_}", "MI (bits, bias-corrected)",
                                   f"Labeled vs unlabeled cells, per-cell MI -- {w_} window"),
                                  ("sig_", "cell_frac_sig", f"13_labeled_vs_unlabeled_frac_sig_{w_}", f"fraction of cells p<{ALPHA}",
                                   f"Labeled vs unlabeled cells, fraction significant -- {w_} window")):
        items = []
        for pred in PANEL_PREDICTORS:
            cp = cw[cw["predictor"] == pred]
            for area in PAIR_AREAS:
                ca = cp[cp["area_"] == area]
                g = ca.groupby(PAIR_KEY + ["label"])[val].agg(["mean", "size"]).reset_index()
                g = g[g["size"] >= MIN_CELLS_PER_UNIT].rename(columns={"mean": "val"})
                if g.empty:
                    continue
                items.append((pred, area, _wide(g, PAIR_KEY, "label")))
        if items:
            run_family(fam, w_, items, nm, f"{ttl}; line = one {PAIR_UNIT} ({'dots coloured by mouse'}), bar = mean, "
                       f"Wilcoxon signed-rank (Holm within the figure)", yl, 4)

if RUN_BETWEEN and bsess is not None:
    roles = []
    for _, r in bsess.iterrows():
        ga, gb = r["group_a"], r["group_b"]
        for G, H in ((ga, gb), (gb, ga)) if ga != gb else ((ga, gb),):
            parts = G.split("_")
            if parts[0] not in PAIR_AREAS or len(parts) < 2:
                continue
            a, lab = parts[0], parts[1]
            pa = H.split("_")[0]
            if pa == a and H != G:
                continue            # labeled x unlabeled pairs of the same area belong to neither side
            roles.append((r["session_id"], r["animal_id"], r["window"], r["measure"], a, lab, pa, r["n_pairs"], r["sum_corr"]))
    rl = pd.DataFrame(roles, columns=["session_id", "animal_id", "window", "measure", "area_", "label", "partner", "n_pairs", "sum_corr"])
    rl = rl.groupby(PAIR_KEY + ["window", "measure", "area_", "label", "partner"])[["n_pairs", "sum_corr"]].sum().reset_index()
    rl = rl[rl["n_pairs"] >= BETWEEN_MIN_PAIRS]
    rl["val"] = rl["sum_corr"] / rl["n_pairs"]
    for w_ in WINDOWS:
        for m in [m for m in ("mi", "mi_given_stim") if m in set(rl["measure"])]:
            items = []
            for part in AREAS:
                for area in PAIR_AREAS:
                    d = rl[(rl["window"] == w_) & (rl["measure"] == m) & (rl["area_"] == area) & (rl["partner"] == part)]
                    if d.empty:
                        continue
                    ttl_ = f"with {part}" + (" (within the area)" if part == area else "")
                    items.append((f"{m}: pairs with {part}" if part != area else f"{m}: within-area pairs", area,
                                  _wide(d, PAIR_KEY, "label")))
            if items:
                run_family(f"between_{m}", w_, items, f"14_labeled_vs_unlabeled_between_area_{m}_{w_}",
                           f"Labeled vs unlabeled: information with each area ({mlabel[m]}{f', S = stimulus [{BETWEEN_STRAT}]' if m == 'mi_given_stim' else ''}) -- {w_} window; "
                           f"line = one {PAIR_UNIT}; Wilcoxon (Holm within the figure)", "MI (bits, bias-corrected)", 4)

if paired_rows:
    pt_ = pd.DataFrame(paired_rows)
    pt_.to_csv(table_dir / "labeled_vs_unlabeled_paired.csv", index=False)
    pd.DataFrame(unit_rows).to_csv(table_dir / "labeled_vs_unlabeled_units.csv", index=False)
    print(f"\nLabeled vs unlabeled (paired per {PAIR_UNIT}): {len(pt_)} comparisons; Holm-significant (p<{ALPHA}):")
    sig_ = pt_[pt_["p_holm"] < ALPHA][["family", "window", "comparison", "area", "n_units", "mean_lab", "mean_unl", "p_wilcoxon", "p_holm"]]
    print(sig_.round(4).to_string(index=False) if len(sig_) else "  none")


# --------------------------------------------------------------------- #
# 5. Disengaged-tail test
# --------------------------------------------------------------------- #
if RUN_TAIL:
    rows = []
    for r in results:
        t = r["tail"]
        rows.append(dict(session_id=r["session_id"], animal_id=r["animal_id"], has_tail=t is not None,
                         **(t["info"] if t else {})))
    tail_trials = pd.DataFrame(rows)
    tail_trials.to_csv(table_dir / "tail_trials.csv", index=False)
    n_tail = int(tail_trials["has_tail"].sum())
    print(f"\nDisengaged-tail test: {n_tail}/{len(tail_trials)} sessions have a tail "
          f"(>= {args.tail_min_frac:.0%} disengaged, >= {args.tail_min_trials} trials, and as many engaged trials "
          f"before it)")
    if n_tail:
        eg.tail_overview_figure({r["session_id"]: r["tail"] for r in results if r["tail"]},
                                {r["session_id"]: r["trials"] for r in results if r["tail"]},
                                figdir, "4_tail_trials")
        tail_mi = all_rec[all_rec["condition"].isin(CONDITIONS)].copy()
        tail_mi.to_csv(table_dir / "tail_cell_predictor_mi.csv", index=False)
        write_group_table(tail_mi, ["condition", "window", "predictor"], "tail_mi_by_group")
        if tail_mi.empty:
            print("  no valid pairs in the tail conditions")
        else:
            tf = frac_sig_by(tail_mi, ["condition", "window", "predictor"]).unstack("condition")
            tm = tail_mi.groupby(UNIT_KEYS + ["condition", "window", "predictor"])["mi_bits_corrected"].mean() \
                .reset_index().groupby(["condition", "window", "predictor"])["mi_bits_corrected"].mean().unstack("condition")
            summary = pd.concat({"mean_mi_bits": tm, "frac_significant": tf}, axis=1)
            print(f"  per-predictor summary, tail sessions only (mean over {UNITS}):")
            print(summary.round(3).to_string())
            cond_colors = {"engaged_matched": "#1f77b4", "disengaged": "#d62728"}
            tail_sig = tail_mi.assign(sig=tail_mi["significant"].astype(float))
            for w in WINDOWS:
                predictor_panels(tail_mi, w, "mi_bits_corrected", f"5_tail_mi_{w}", "MI (bits, bias-corrected)",
                                 f"Tail test, {w}: matched engaged vs. end-of-session disengaged trials ({n_tail} sessions)",
                                 series_col="condition", series_order=list(CONDITIONS), series_colors=cond_colors)
                predictor_panels(tail_sig, w, "sig", f"6_tail_frac_significant_{w}", f"fraction of cells p<{ALPHA}",
                                 f"Tail test, {w}: fraction significant (dashed = chance {ALPHA})", hline=ALPHA,
                                 series_col="condition", series_order=list(CONDITIONS), series_colors=cond_colors)
    else:
        print("  no session has a usable tail -- try a lower --tail-min-frac or --tail-min-trials")

print(f"\nWrote tables to {table_dir}, figures to {figdir}")
