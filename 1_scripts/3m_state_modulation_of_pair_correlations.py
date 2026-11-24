# -*- coding: utf-8 -*-
"""
3m_state_modulation_of_pair_correlations
=========================================
Does the brain / arousal STATE change how strongly two cells co-fluctuate?  The pair-level counterpart of 2m (which asked the same for the
single-cell lick behavior) and of 1f / 1g / 2l (single-cell tuning).  STAGE 1 = NOISE CORRELATIONS; the information-theoretic part
(pair information / PID per state bin) is a separate step (not in this file yet).

STATES (one value per trial, z-scored within the session and then within the pair; mean over STATE_WINDOW, default pre-stim -30..-10 cm)
    PM_lab    mean of the PM-lab cells, LEAVE-PAIR-OUT (the pair's own cells never enter its own state); needs >= MIN_SOURCE_CELLS others
    global    mean of all included cells outside PM-lab, leave-pair-out
    pupil, speed    pupil area and running speed in the window (face normalisation as the rest of the project)

PAIRS  within each group (the 8 project groups: the pairs of 3a's pair_list.csv, pair_rank < --max-pairs = the SAME pairs as 3a / 3b / 3c)
       + CROSS pairs V1-lab x PM-lab and V1-unl x PM-lab (sampled here, at most --max-pairs per session and pair of groups).

CO-FLUCTUATION (per pair, response window = pre_stim / stim / reward of 3a; the SAME trials as 3a: infotheory.pair_tables.pair_trials with the
target = TARGET_STRATEGY of stim_binning, so the numbers equal 3a's noise correlations):
    residuals r_i = response - mean response of that trial's target bin (= 3a's noise correlation), z-scored across the pair's trials;
    mean(r_1 * r_2) = the noise correlation.  Two versions (REPORTED BOTH):
        raw            responses as they are
        resid_global   each cell first regressed on the LEAVE-PAIR-OUT global signal of the same window (mean of all other included cells
                       in that trial), the correlation of what is left: removes the brain-wide shared gain 2l found
MODULATION by the state x (STATE_TRANSFORM = 'rank': normal scores of the state's ranks within the pair, so x and x2 = x^2 - 1 are orthogonal even for the skewed
states; 'z': plain z-score; terciles are the same either way), x2 = x^2 - 1:
    covariance regression   p_t = z_1t * z_2t  ~  1 + x + x2   (b_x, b_x2: change of the co-fluctuation per SD of state; linear and inverted U)
    correlation regression  the same on q_t = p_t / sqrt(v_1(x_t) v_2(x_t)), v_i(x) = the state-dependent variance of cell i (regression of
                            z_i^2 on 1 + x + x2): a state that merely raises BOTH variances raises the covariance but not the correlation.
                            THE CORRELATION REGRESSION (bc_x, bc_x2) IS THE PRIMARY MEASURE; the covariance version is kept for reference.
    state terciles  Pearson correlation of the residuals in the low / mid / high third of the state (within pair); hi_minus_lo = c_high - c_low;
                    mid_minus_ends = c_mid - (c_low + c_high) / 2 (> 0 = inverted U).  With REMOVE_STATE_MEAN (default) each cell's residual is first regressed on
                    the state (1, x, x^2 - 1), so a state-dependent MEAN of the two cells cannot pass for a state-dependent correlation.
TESTS: per (group, window, state, version, measure): the pair mean per SESSION (sessions with >= MIN_PAIRS_PER_UNIT pairs), one-sample t and
Wilcoxon across sessions, the same across MICE, number of mice with a positive effect; BH q over (groups x states) within (version, window, measure).
Everything is EXPLORATORY (no pre-specified primary test).  Pairs of one session share cells and trials: only the session / mouse level tests
count, never the number of pairs.

--noise-only: restrict every window table to the NOISE trials (0 < signal < 100 %) before anything else; outputs in out/noise_only/ (the default uses all engaged
trials, like 3a / 3b / 3c, so that the unsplit correlations agree with 3a).

Project-wide rules
    * selection : --animals / --sessions restrict the sessions (restricted run -> out/selection/<tag>/).  --window LO HI changes the state window.
    * aggregate : --aggregate pooled|session|mouse = the unit of the figures (pooled: pair-weighted mean over all sessions, no error bars;
                  session / mouse: mean over the unit's pairs first, mean +- SEM over units).  Always all groups (8 + 2 cross).
    * memory    : one session per worker; each session is reduced to a SMALL per-(group, window, state, version) summary, cached in store/.
                  (No pair-level table is written: it would be ~10^6 rows.)
    * binning   : same pairs, same trials, same residual definition as 3a (shared pair_tables / build_session_tables).
    * y axes    : panels of a figure row share one y range.

CAVEATS  (1) leave-pair-out removes the pair's own contribution, but a brain-wide gain also contaminates the cells' own fluctuations: the
resid_global version assumes one shared slope per cell.  (2) [fixed in v4, REMOVE_STATE_MEAN] the states shift the MEAN response of cells; before v4 this leaked into the x^2 term (see the setting).  Old note: pupil / speed shift the mean response of cells as well as their correlations;
the target-bin means are removed, state-dependent means are not (a state that shifts the means of both cells creates a co-fluctuation along x
that is regressed away only in the correlation / tercile versions through the within-tercile centring).  (3) 9 PM-lab sessions: the PM_lab
state exists only in sessions with enough PM-lab cells.

Reads: 1b psychometric_fits.csv, 2a celldata_combined.csv, 2b qc_per_cell.csv, 3a out/pair_list.csv.
Outputs  2_pipeline/3m_state_modulation_of_pair_correlations/out/
    pair_state_modulation_by_session.csv   one row per (session, group, window, state, version): n_pairs, mean corr, b_x / b_x2 (cov, corr), terciles
    pair_state_modulation_tests.csv        across-session / across-mouse tests
    figures/ 1_corr_by_state_tercile_<version>, 2_modulation_<version>, 3_raw_vs_resid_global, 4_tstat_<measure>_<version>
"""
from __future__ import annotations

import argparse
import hashlib
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sps

from infotheory import project_config as cfg
from infotheory.pipeline import get_pipeline_paths
from infotheory.session import discover_sessions
from infotheory.neural_encoding import load_cell_inclusion, build_session_tables
from infotheory.celldata_utils import SIX_GROUP_ORDER, SIX_GROUP_LABELS, SIX_GROUP_COLORS
from infotheory.groups import group_key
from infotheory.plotting import set_style, save_fig, clear_figures
from infotheory.scope import add_scope_args, scope_from_args, select_sessions, scoped_dirs, data_stamp, unit_noun
from infotheory.session_jobs import run_sessions
from infotheory.unit_stats import stable_seed
from infotheory.unit_plots import mouse_palette
from infotheory.stim_binning import SIMPLE_STRATEGIES, STIM_BINNING_STRATEGIES, target_signature
from infotheory import pair_tables as pt

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom")
warnings.filterwarnings("ignore", category=RuntimeWarning)

PROTOCOL = "DN"
WINDOWS = {"pre_stim": (-30.0, -10.0), "stim": (0.0, 20.0), "reward": (25.0, 45.0)}   # response windows (as 3a)
STATE_WINDOW = (-30.0, -10.0)          # state window, cm rel. to stimStart (--window LO HI)
TARGET_STRATEGY = SIMPLE_STRATEGIES[0]  # residuals are taken within the bins of this target (3a's noise correlation)
SOURCE_GROUP = "PM_lab"
CROSS_A_GROUPS = ["V1_lab", "V1_unl"]   # cross pairs: each of these x SOURCE_GROUP
MIN_SOURCE_CELLS = 5
MAX_PAIRS_PER_GROUP = 200               # fallback for cfg.MAX_PAIRS_PER_SESSION
STATES = [SOURCE_GROUP, "global", "pupil", "speed"]
VERSIONS = ["raw", "resid_global"]
STATE_TRANSFORM = "rank"                # "rank": the state enters the regressions as its NORMAL SCORES within the pair (rank -> N(0,1) quantile): symmetric, so the
                                        # linear (x) and quadratic (x^2 - 1) terms are orthogonal even for skewed states (global, pupil, speed). "z": plain z-score.
MIN_STATE_TRIALS = max(cfg.MIN_TRIALS_ANALYSIS, 60)
MIN_TERCILE_TRIALS = 15
REMOVE_STATE_MEAN = True                # regress each cell's residual on the state (1, x, x^2 - 1) BEFORE the products / tercile correlations. Without it, two cells
                                        # whose MEAN activity follows the state (b1 * x and b2 * x) give a product b1 * b2 * x^2: a spurious positive x^2 term
                                        # (and a spurious U shape) that is a single-cell effect, not a change of their correlation.  (`corr` stays 3a's value.)
VAR_FLOOR = 0.1                         # floor of the state-dependent variance (fraction of the mean variance)
MIN_PAIRS_PER_UNIT = 3
MIN_SESSIONS_TEST = 4
ALPHA = 0.05
RANDOM_STATE = 0
SHARE_Y = True
JOBLIB_VERBOSE = 5
MEASURES = ["bc_x", "bc_x2", "hi_minus_lo", "mid_minus_ends", "b_x", "b_x2"]
MEASURE_LABEL = {"mid_minus_ends": "correlation in the mid tercile minus the mean of low and high (> 0 = inverted U)", "bc_x": "correlation change per SD of state (linear)", "bc_x2": "correlation change, inverted-U term (x^2 - 1)",
                 "hi_minus_lo": "correlation in high minus low state tercile", "b_x": "co-fluctuation change per SD (linear, covariance)",
                 "b_x2": "co-fluctuation, inverted-U term (covariance)"}
STATE_COLORS = {"PM_lab": "#d62728", "global": "#1f77b4", "pupil": "#2ca02c", "speed": "#9467bd"}

parser = argparse.ArgumentParser()
add_scope_args(parser)
parser.add_argument("--window", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="state window in cm rel. to stimStart (default STATE_WINDOW); outputs go to out/window_<lo>_<hi>/")
parser.add_argument("--noise-only", action="store_true",
                    help="use only the NOISE trials (0 < signal < 100 %), not the 0 % / 100 % catch trials; outputs go to out/noise_only/ (the default run "
                         "uses all engaged trials like 3a / 3b / 3c)")
args = parser.parse_args()
NOISE_ONLY = bool(args.noise_only)
if args.window:
    STATE_WINDOW = (float(args.window[0]), float(args.window[1]))
scope = scope_from_args(args)
AGG = scope.aggregate
UNIT = unit_noun(AGG)
UNITS = "mice" if UNIT == "mouse" else f"{UNIT}s"
MAX_PAIRS = args.max_pairs or cfg.MAX_PAIRS_PER_SESSION or MAX_PAIRS_PER_GROUP
TARGET = TARGET_STRATEGY

paths = get_pipeline_paths(__file__)
table_dir, figdir = scoped_dirs(paths, scope)
if args.window:
    _w = f"window_{STATE_WINDOW[0]:g}_{STATE_WINDOW[1]:g}"
    table_dir, figdir = table_dir / _w, figdir / _w
    table_dir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
if NOISE_ONLY:
    table_dir, figdir = table_dir / "noise_only", figdir / "noise_only"
    table_dir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
clear_figures(figdir)
set_style()
print(scope.describe())
print(f"state window {STATE_WINDOW}; residuals within the bins of target {TARGET!r}")

# --------------------------------------------------------------------- #
# 1. Inputs
# --------------------------------------------------------------------- #
fits_path = paths.out_from("1b_psychometric") / "psychometric_fits.csv"
cell_dist_out = paths.out_from("2a_cell_distribution")
activity_out = paths.out_from("2b_activity_statistics")
pair_list_path = paths.out_from("3a_pair_activity_statistics") / "pair_list.csv"
for f, hint in [(fits_path, "1b_psychometric.py"), (cell_dist_out / "celldata_combined.csv", "2a_cell_distribution.py"),
                (activity_out / "qc_per_cell.csv", "2b_activity_statistics.py"), (pair_list_path, "3a_pair_activity_statistics.py")]:
    if not f.exists():
        raise SystemExit(f"Run {hint} first -- expected {f}")
fits_df = pd.read_csv(fits_path)
mu_sigma_lookup = fits_df.set_index("session_id")[["mu", "sigma"]].to_dict("index")
included_all = set(fits_df[fits_df["included"] & (fits_df["protocol"] == PROTOCOL)]["session_id"])
all_triples = discover_sessions([PROTOCOL])
selected = {f"{a}_{d}": (p, a, d) for p, a, d in select_sessions(all_triples, scope)}
session_ids = [s for s in selected if s in included_all]
if not session_ids:
    raise SystemExit("None of the selected sessions passed 1b_psychometric's inclusion criteria.")
mouse_colors = mouse_palette(a for _, a, _ in all_triples)
cell_inclusion = load_cell_inclusion(cell_dist_out, activity_out)
cell_group = dict(zip(cell_inclusion["cell_id"], cell_inclusion["group"]))
incl_by_session = {sid: sorted(sub.loc[sub["include"], "cell_id"]) for sid, sub in cell_inclusion.groupby("session_id") if sid in session_ids}

# pairs: within groups = 3a's list (finalised -> group key), cross = sampled here
within = pt.load_pair_list(pair_list_path, session_ids, MAX_PAIRS)
if len(within):
    within = pt.finalize_pair_table(within)
    within["kind"] = "within"
cross_rows = []
for sid in session_ids:
    cells = incl_by_session.get(sid, [])
    tgt = [c for c in cells if cell_group.get(c) == SOURCE_GROUP]
    for ga in CROSS_A_GROUPS:
        src = [c for c in cells if cell_group.get(c) == ga]
        if not src or not tgt:
            continue
        ii, jj = np.meshgrid(np.arange(len(src)), np.arange(len(tgt)), indexing="ij")
        ii, jj = ii.ravel(), jj.ravel()
        rng = np.random.default_rng(stable_seed(sid, "cross", ga, SOURCE_GROUP, base=RANDOM_STATE))
        pick = rng.permutation(len(ii))[:MAX_PAIRS]
        for rank, k in enumerate(pick):
            cross_rows.append(dict(session_id=sid, animal_id=sid.split("_")[0], group=f"{ga}~{SOURCE_GROUP}", cell_i=src[ii[k]],
                                   cell_j=tgt[jj[k]], pair_rank=rank, kind="cross"))
cross = pd.DataFrame(cross_rows)
pairs_all = pd.concat([within[["session_id", "animal_id", "group", "cell_i", "cell_j", "pair_rank", "kind"]] if len(within) else None,
                       cross if len(cross) else None], ignore_index=True)
session_ids = [s for s in session_ids if s in set(pairs_all["session_id"])]
if not session_ids:
    raise SystemExit("No pairs (run 3a on the full dataset first).")
pairs_by_session = {sid: sub.reset_index(drop=True) for sid, sub in pairs_all.groupby("session_id")}
cells_by_session = {sid: sorted(incl_by_session[sid]) for sid in session_ids}
print(f"{len(session_ids)} sessions, {len(pairs_all)} pairs ({int((pairs_all['kind'] == 'within').sum())} within-group, "
      f"{int((pairs_all['kind'] == 'cross').sum())} cross; pair_rank < {MAX_PAIRS})")
pair_hash = {s: hashlib.sha1(pairs_by_session[s].to_csv(index=False).encode()).hexdigest()[:10] for s in session_ids}
stamps = {s: f"{data_stamp(*selected[s])}-{pair_hash[s]}" for s in session_ids}
qc_mtime = int((activity_out / "qc_per_cell.csv").stat().st_mtime)


# --------------------------------------------------------------------- #
# 2. Per session
# --------------------------------------------------------------------- #
def zcols(M):
    M = np.asarray(M, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (M - np.nanmean(M, 0)) / np.nanstd(M, 0)


def zvec(v):
    v = np.asarray(v, float)
    sd = np.nanstd(v)
    return (v - np.nanmean(v)) / sd if sd > 0 else np.full_like(v, np.nan)


def nscore(v):
    """Normal scores of the ranks (ties share the mean rank): a symmetric N(0, 1)-like version of the state."""
    v = np.asarray(v, float)
    return sps.norm.ppf((sps.rankdata(v) - 0.5) / len(v))


def basis(x):
    return np.column_stack([np.ones(len(x)), x, x * x - 1.0])


def resid_by_label(a, inv, k):
    cnt = np.bincount(inv, minlength=k)
    return a - (np.bincount(inv, weights=a, minlength=k) / np.maximum(cnt, 1))[inv]


def corr_(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def modulation(z1, z2, x):
    """Covariance and correlation regression + tercile correlations of one pair, one state. z1, z2 = z-scored residuals, x = z-scored state."""
    B = basis(x)
    p = z1 * z2
    cov = np.linalg.lstsq(B, p, rcond=None)[0]
    v1 = B @ np.linalg.lstsq(B, z1 * z1, rcond=None)[0]
    v2 = B @ np.linalg.lstsq(B, z2 * z2, rcond=None)[0]
    v1 = np.maximum(v1, VAR_FLOOR * np.mean(z1 * z1))
    v2 = np.maximum(v2, VAR_FLOOR * np.mean(z2 * z2))
    q = p / np.sqrt(v1 * v2)
    cc = np.linalg.lstsq(B, q, rcond=None)[0]
    e1, e2 = np.quantile(x, [1 / 3, 2 / 3])
    terc = [x <= e1, (x > e1) & (x <= e2), x > e2]
    ct_ = [corr_(z1[m], z2[m]) if m.sum() >= MIN_TERCILE_TRIALS else np.nan for m in terc]
    return [float(np.mean(p)), cov[1], cov[2], cc[1], cc[2]] + ct_ + [ct_[2] - ct_[0], ct_[1] - 0.5 * (ct_[0] + ct_[2])]


COLS = ["corr", "b_x", "b_x2", "bc_x", "bc_x2", "c_t1", "c_t2", "c_t3", "hi_minus_lo", "mid_minus_ends"]


def process_session(ses, seed):
    sid = ses.session_id
    pairs = pairs_by_session.get(sid)
    cells = [c for c in cells_by_session.get(sid, [])]
    if pairs is None or not cells or ses.celldata is None:
        return None
    wins = dict(WINDOWS)
    wins["_state"] = STATE_WINDOW
    tables = build_session_tables(ses, mu_sigma_lookup, wins, included_cell_ids=set(cells))
    if not tables or "_state" not in tables:
        return None
    st = tables.pop("_state")
    cells = [c for c in cells if c in st.columns]
    cidx = {c: i for i, c in enumerate(cells)}
    grp = np.array([cell_group.get(c) for c in cells], dtype=object)
    src = grp == SOURCE_GROUP
    oth = ~src
    st = st.sort_values("trialNumber").reset_index(drop=True)
    Zs = zcols(st[cells].to_numpy(float))
    Zs0 = np.nan_to_num(Zs)
    Cs = np.isfinite(Zs).astype(float)
    state_src = (Zs0[:, src].sum(1), Cs[:, src].sum(1))
    state_oth = (Zs0[:, oth].sum(1), Cs[:, oth].sum(1))
    face = {"pupil": zvec(st["cur_pupil_area"]) if "cur_pupil_area" in st else None,
            "speed": zvec(st["cur_runspeed"]) if "cur_runspeed" in st else None}
    st_tn = st["trialNumber"].to_numpy()
    rows = []
    for window, table in tables.items():
        if table.empty or "signal" not in table.columns:
            continue
        if NOISE_ONLY:                                                  # noise trials only: 0 < signal < 100 %
            table = table[(table["signal"] > 0) & (table["signal"] < 100)]
        table = pt.add_target_columns(table, [TARGET]).reset_index(drop=True)
        tnum = table["trialNumber"].to_numpy()
        srow = pd.Index(st_tn).get_indexer(tnum)                      # state-table row of each trial of this window (-1 = none)
        has = srow >= 0
        Z = zcols(table[cells].to_numpy(float))
        Z0, C = np.nan_to_num(Z), np.isfinite(Z).astype(float)
        Sall, Nall = Z0.sum(1), C.sum(1)

        def at(v):                                                     # state-table vector -> this window's trials
            out = np.full(len(tnum), np.nan)
            out[has] = v[srow[has]]
            return out
        for r in pairs.itertuples(index=False):
            if r.cell_i not in cidx or r.cell_j not in cidx:
                continue
            ci, cj = cidx[r.cell_i], cidx[r.cell_j]
            d = pt.pair_trials(table, TARGET, r.cell_i, r.cell_j)
            if d is None:
                continue
            ix = pd.Index(tnum).get_indexer(d["trial"])
            G = (Sall - Z0[:, ci] - Z0[:, cj]) / np.maximum(Nall - C[:, ci] - C[:, cj], 1)
            G = G[ix]
            lab = d["label"]
            _, inv = np.unique(lab, return_inverse=True)
            k = inv.max() + 1
            # pair-specific states at this window's trials
            sv = {}
            for name, (S_, N_), member in (("PM_lab", state_src, src), ("global", state_oth, oth)):
                own_s = sum(Zs0[:, c] * member[c] for c in (ci, cj))
                own_n = sum(Cs[:, c] * member[c] for c in (ci, cj))
                n_ = N_ - own_n
                v = np.where(n_ >= MIN_SOURCE_CELLS, (S_ - own_s) / np.maximum(n_, 1), np.nan)
                sv[name] = at(zvec(v))[ix]
            for name in ("pupil", "speed"):
                sv[name] = at(face[name])[ix] if face[name] is not None else np.full(len(ix), np.nan)
            for ver in VERSIONS:
                a1, a2 = d["x1"], d["x2"]
                fin = np.isfinite(G)
                if ver == "resid_global":
                    if fin.sum() < MIN_STATE_TRIALS:
                        continue
                    Bg = np.column_stack([np.ones(fin.sum()), G[fin]])
                    a1 = a1.copy()
                    a2 = a2.copy()
                    a1[fin] = a1[fin] - Bg @ np.linalg.lstsq(Bg, a1[fin], rcond=None)[0]
                    a2[fin] = a2[fin] - Bg @ np.linalg.lstsq(Bg, a2[fin], rcond=None)[0]
                r1, r2 = resid_by_label(a1, inv, k), resid_by_label(a2, inv, k)
                for name in STATES:
                    m = fin & np.isfinite(sv[name])
                    if m.sum() < MIN_STATE_TRIALS:
                        continue
                    z1, z2 = zvec(r1[m]), zvec(r2[m])
                    x = nscore(sv[name][m]) if STATE_TRANSFORM == "rank" else zvec(sv[name][m])
                    corr0 = float(np.mean(z1 * z2)) if np.isfinite(z1).all() and np.isfinite(z2).all() else np.nan
                    if REMOVE_STATE_MEAN and np.isfinite(x).all() and np.isfinite(corr0):
                        Bx = basis(x)
                        z1 = zvec(z1 - Bx @ np.linalg.lstsq(Bx, z1, rcond=None)[0])
                        z2 = zvec(z2 - Bx @ np.linalg.lstsq(Bx, z2, rcond=None)[0])
                    if not (np.isfinite(z1).all() and np.isfinite(z2).all() and np.isfinite(x).all()):
                        continue
                    mod = modulation(z1, z2, x)
                    mod[0] = corr0                                     # the noise correlation as in 3a (before removing the state means)
                    rows.append([r.group, window, name, ver] + mod)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["group", "window", "state", "version"] + COLS)
    g = df.groupby(["group", "window", "state", "version"], sort=False)
    out = g[COLS].mean()
    out["n_pairs"] = g.size()
    out = out.reset_index()
    out.insert(0, "animal_id", ses.animal_id)
    out.insert(0, "session_id", sid)
    print(f"  [{sid}] {len(df)} (pair, window, state, version) rows -> {len(out)} summary rows")
    return out


results = run_sessions(
    session_ids, process_session, [PROTOCOL], paths.store, "pair_state",
    key_params=dict(version="3m-v4", remove_state_mean=REMOVE_STATE_MEAN, state_transform=STATE_TRANSFORM, noise_only=NOISE_ONLY, windows=WINDOWS, state_window=STATE_WINDOW, target=target_signature([TARGET]), source=SOURCE_GROUP,
                    cross=CROSS_A_GROUPS, min_source=MIN_SOURCE_CELLS, min_state=MIN_STATE_TRIALS, var_floor=VAR_FLOOR,
                    min_trials=cfg.MIN_TRIALS_ANALYSIS, min_per_bin=cfg.MIN_TRIALS_PER_TARGET_BIN, max_pairs=MAX_PAIRS, qc_mtime=qc_mtime,
                    fits_mtime=int(fits_path.stat().st_mtime)),
    load_kwargs=dict(load_behaviordata=True, load_videodata=True, load_celldata=True, load_calciumdata=True),
    cells_by_session=cells_by_session, base_seed=RANDOM_STATE, n_jobs=scope.n_jobs, recompute=args.recompute,
    verbose=JOBLIB_VERBOSE, extra_key_by_session=stamps, flatten=False)
results = [r for r in results if r is not None and len(r)]
if not results:
    raise SystemExit("No session produced pair modulation results.")
S = pd.concat(results, ignore_index=True)
del results
S.to_csv(table_dir / "pair_state_modulation_by_session.csv", index=False, float_format="%.5g")
print(f"\n{S['session_id'].nunique()} sessions; {len(S)} summary rows")
for st_ in STATES:
    n = S[S["state"] == st_].groupby("group")["session_id"].nunique()
    print(f"  sessions with state {st_}: {int(S[S['state'] == st_]['session_id'].nunique())}  (per group: {n.to_dict()})")

# --------------------------------------------------------------------- #
# 3. Tests across sessions / mice
# --------------------------------------------------------------------- #
GROUPS = [group_key(g) for g in SIX_GROUP_ORDER] + [f"{a}~{SOURCE_GROUP}" for a in CROSS_A_GROUPS]
GLABEL = {group_key(g): SIX_GROUP_LABELS[g] for g in SIX_GROUP_ORDER}
GCOLOR = {group_key(g): SIX_GROUP_COLORS[g] for g in SIX_GROUP_ORDER}
for a, c in zip(CROSS_A_GROUPS, ["#333333", "#8c564b"]):
    GLABEL[f"{a}~{SOURCE_GROUP}"] = f"{a.replace('_', '-')} x {SOURCE_GROUP.replace('_', '-')}"
    GCOLOR[f"{a}~{SOURCE_GROUP}"] = c
WIN = list(WINDOWS)


def bh(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if ok.any():
        pv = p[ok]
        o = np.argsort(pv)
        adj = np.minimum.accumulate((pv[o] * len(pv) / np.arange(1, len(pv) + 1))[::-1])[::-1]
        r = np.empty(len(pv))
        r[o] = np.minimum(adj, 1.0)
        q[ok] = r
    return q


def one_sample(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 3 or np.std(v) == 0:
        return len(v), (float(np.mean(v)) if len(v) else np.nan), np.nan, np.nan, np.nan
    t, p = sps.ttest_1samp(v, 0.0)
    try:
        pw = sps.wilcoxon(v).pvalue
    except ValueError:
        pw = np.nan
    return len(v), float(np.mean(v)), float(t), float(p), float(pw)


trows = []
Sok = S[S["n_pairs"] >= MIN_PAIRS_PER_UNIT]
for (grp, win, state, ver), sub in Sok.groupby(["group", "window", "state", "version"]):
    for meas in MEASURES:
        n, mean, t, p, pw = one_sample(sub[meas])
        if n < MIN_SESSIONS_TEST:
            continue
        mice = sub.groupby("animal_id").apply(lambda d: np.average(d[meas].dropna(), weights=d.loc[d[meas].notna(), "n_pairs"])
                                              if d[meas].notna().any() else np.nan, include_groups=False)
        nm, mmean, mt, mp, _ = one_sample(mice.to_numpy())
        trows.append(dict(group=grp, window=win, state=state, version=ver, measure=meas, n_sessions=n, mean=mean, t=t, p_session=p,
                          p_wilcoxon=pw, n_mice=nm, mean_mice=mmean, t_mice=mt, p_mice=mp,
                          n_mice_positive=int((mice > 0).sum()), n_mice_negative=int((mice < 0).sum())))
tests = pd.DataFrame(trows)
if len(tests):
    tests["q_session"] = np.nan
    for _, idx in tests.groupby(["version", "window", "measure"]).groups.items():
        tests.loc[idx, "q_session"] = bh(tests.loc[idx, "p_session"].to_numpy())
    tests.to_csv(table_dir / "pair_state_modulation_tests.csv", index=False, float_format="%.5g")
    print("\nAcross-session tests of the PRIMARY measures (correlation regression), sorted by p; q = BH within version x window x measure:")
    show = tests[tests["measure"].isin(["bc_x", "bc_x2"])].sort_values("p_session")
    print(show.head(20)[["group", "window", "state", "version", "measure", "n_sessions", "mean", "p_session", "q_session", "n_mice",
                         "p_mice", "n_mice_positive"]].round(4).to_string(index=False))
    print(f"\n{int((tests['q_session'] < ALPHA).sum())} / {len(tests)} tests with q < {ALPHA} (all measures); "
          f"{int(((tests['q_session'] < ALPHA) & tests['measure'].isin(['bc_x', 'bc_x2'])).sum())} among the primary measures")

# --------------------------------------------------------------------- #
# 4. Figures
# --------------------------------------------------------------------- #
agg_note = {"pooled": "pair-weighted mean over all sessions (no error bars)", "session": "mean +- SEM over sessions",
            "mouse": "mean +- SEM over mice"}[AGG]
KEYS = {"pooled": [], "session": ["animal_id", "session_id"], "mouse": ["animal_id"]}[AGG]


def unit_values(df, value, by):
    """Pair-weighted value per (by..., unit): pooled = no unit; session / mouse = mean over the unit's pairs first."""
    d = df[np.isfinite(df[value]) & (df["n_pairs"] >= (1 if AGG == "pooled" else MIN_PAIRS_PER_UNIT))].copy()
    d["_v"] = d[value] * d["n_pairs"]
    g = d.groupby(by + KEYS).agg(_v=("_v", "sum"), _w=("n_pairs", "sum")).reset_index()
    g["value"] = g["_v"] / g["_w"]
    return g


def mean_sem(g, by):
    m = g.groupby(by)["value"].agg(["mean", "sem", "count"]).reset_index()
    if AGG == "pooled":
        m["sem"] = np.nan
    return m


def lim_pad(lo, hi):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    pad = 0.08 * (hi - lo if hi > lo else 1.0)
    return lo - pad, hi + pad


for ver in VERSIONS:
    Sv = S[S["version"] == ver]
    if Sv.empty:
        continue
    # 1: correlation in state terciles, lines per group
    fig, axes = plt.subplots(len(STATES), len(WIN), figsize=(4.3 * len(WIN), 2.9 * len(STATES)), squeeze=False, sharey="row")
    for ri, state in enumerate(STATES):
        for ci_, w in enumerate(WIN):
            ax = axes[ri][ci_]
            sub = Sv[(Sv["state"] == state) & (Sv["window"] == w)]
            for g in GROUPS:
                ms = []
                for t_ in (1, 2, 3):
                    uv = unit_values(sub[sub["group"] == g], f"c_t{t_}", ["group"])
                    ms.append(mean_sem(uv, ["group"]) if len(uv) else None)
                if any(m is None or m.empty for m in ms):
                    continue
                y = [float(m["mean"].iloc[0]) for m in ms]
                e = [0 if not np.isfinite(m["sem"].iloc[0]) else float(m["sem"].iloc[0]) for m in ms]
                ax.errorbar([1, 2, 3], y, yerr=e, color=GCOLOR[g], marker="o", ms=3.5, lw=1.1, capsize=2, label=GLABEL[g])
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(["low", "mid", "high"])
            if ri == 0:
                ax.set_title(f"{w} ({WINDOWS[w][0]:g}..{WINDOWS[w][1]:g} cm)", fontsize=10)
            if ci_ == 0:
                ax.set_ylabel(f"{state}\nnoise correlation")
            if ri == len(STATES) - 1:
                ax.set_xlabel("state tercile (within session / pair)")
    axes[0][0].legend(fontsize=5.5, frameon=False, ncol=2)
    fig.suptitle(f"Noise correlation by state tercile [{ver}]; state window {STATE_WINDOW[0]:g}..{STATE_WINDOW[1]:g} cm; {agg_note}", y=1.0, fontsize=9)
    fig.tight_layout()
    save_fig(fig, figdir, f"1_corr_by_state_tercile_{ver}")
    plt.close(fig)

    # 2: modulation coefficients per group, states as series
    terms = [("bc_x", "linear"), ("bc_x2", "inverted-U (x^2-1)")]
    fig, axes = plt.subplots(len(terms), len(WIN), figsize=(4.8 * len(WIN), 3.4 * len(terms)), squeeze=False, sharey="row")
    xs = np.arange(len(GROUPS))
    for ri, (meas, mlab) in enumerate(terms):
        for ci_, w in enumerate(WIN):
            ax = axes[ri][ci_]
            for si, state in enumerate(STATES):
                sub = Sv[(Sv["state"] == state) & (Sv["window"] == w)]
                uv = unit_values(sub, meas, ["group"])
                ms = mean_sem(uv, ["group"]).set_index("group") if len(uv) else pd.DataFrame()
                for gi, g in enumerate(GROUPS):
                    if g in ms.index:
                        off = (si - (len(STATES) - 1) / 2) * 0.17
                        ax.errorbar(gi + off, ms.loc[g, "mean"], yerr=0 if not np.isfinite(ms.loc[g, "sem"]) else ms.loc[g, "sem"],
                                    color=STATE_COLORS[state], marker="o", ms=4, capsize=2, lw=1,
                                    label=state if gi == 0 or g == GROUPS[-1] else None)
            ax.axhline(0, color="grey", lw=0.7)
            ax.set_xticks(xs)
            ax.set_xticklabels([GLABEL[g] for g in GROUPS], rotation=60, ha="right", fontsize=7)
            if ri == 0:
                ax.set_title(f"{w} ({WINDOWS[w][0]:g}..{WINDOWS[w][1]:g} cm)", fontsize=10)
            if ci_ == 0:
                ax.set_ylabel(f"{mlab}\ncorrelation change per SD")
    h, l_ = axes[0][0].get_legend_handles_labels()
    uniq = dict(zip(l_, h))
    axes[0][0].legend(list(uniq.values()), list(uniq.keys()), fontsize=7, frameon=False)
    fig.suptitle(f"State modulation of pair noise correlations [{ver}] (correlation regression; {agg_note})", y=1.0, fontsize=9)
    fig.tight_layout()
    save_fig(fig, figdir, f"2_modulation_{ver}")
    plt.close(fig)

# 3: raw vs resid_global
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), squeeze=False)
for ax, (meas, mlab) in zip(axes[0], [("bc_x", "linear"), ("bc_x2", "inverted-U")]):
    vals = {}
    for ver in VERSIONS:
        uv = unit_values(S[S["version"] == ver], meas, ["group", "window", "state"])
        vals[ver] = mean_sem(uv, ["group", "window", "state"]).set_index(["group", "window", "state"])["mean"] if len(uv) else pd.Series(dtype=float)
    both = pd.concat(vals, axis=1).dropna()
    for state in STATES:
        b = both[both.index.get_level_values("state") == state]
        ax.scatter(b["raw"], b["resid_global"], s=18, color=STATE_COLORS[state], label=state, alpha=0.8)
    if len(both):
        lo, hi = both.to_numpy().min(), both.to_numpy().max()
        ax.plot([lo, hi], [lo, hi], color="grey", lw=0.8, ls="--")
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel(f"{mlab} modulation, raw")
    ax.set_ylabel(f"{mlab} modulation, global removed")
    ax.set_title(f"{mlab}: each dot = (group, window, state)")
axes[0][0].legend(fontsize=8, frameon=False)
fig.suptitle(f"Does the modulation survive removing the shared (leave-pair-out) global signal? {agg_note}", y=1.02, fontsize=9)
fig.tight_layout()
save_fig(fig, figdir, "3_raw_vs_resid_global")
plt.close(fig)

# 4: t-statistics across sessions
if len(tests):
    for ver in VERSIONS:
        for meas in ("bc_x", "bc_x2"):
            tv = tests[(tests["version"] == ver) & (tests["measure"] == meas)]
            cols = [(w, s_) for w in WIN for s_ in STATES]
            M = np.full((len(GROUPS), len(cols)), np.nan)
            Q = np.full_like(M, np.nan)
            for gi, g in enumerate(GROUPS):
                for ci_, (w, s_) in enumerate(cols):
                    r = tv[(tv["group"] == g) & (tv["window"] == w) & (tv["state"] == s_)]
                    if len(r):
                        M[gi, ci_], Q[gi, ci_] = r["t"].iloc[0], r["q_session"].iloc[0]
            fig, ax = plt.subplots(figsize=(1.0 * len(cols) + 3, 0.5 * len(GROUPS) + 2.5))
            im = ax.imshow(M, cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto")
            for gi in range(len(GROUPS)):
                for ci_ in range(len(cols)):
                    if np.isfinite(M[gi, ci_]):
                        ax.text(ci_, gi, f"{M[gi, ci_]:.1f}" + ("*" if Q[gi, ci_] < ALPHA else ""), ha="center", va="center", fontsize=6.5)
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels([f"{w}\n{s_}" for w, s_ in cols], fontsize=6.5)
            ax.set_yticks(range(len(GROUPS)))
            ax.set_yticklabels([GLABEL[g] for g in GROUPS], fontsize=8)
            fig.colorbar(im, ax=ax, label="t across sessions")
            ax.set_title(f"{MEASURE_LABEL[meas]} [{ver}]: t across sessions (* = BH q < {ALPHA}; blank = < {MIN_SESSIONS_TEST} sessions)", fontsize=8)
            fig.tight_layout()
            save_fig(fig, figdir, f"4_tstat_{meas}_{ver}")
            plt.close(fig)

print(f"\nWrote tables to {table_dir}, figures to {figdir}")
