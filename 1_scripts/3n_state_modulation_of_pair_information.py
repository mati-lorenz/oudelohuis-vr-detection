# -*- coding: utf-8 -*-
"""
3n_state_modulation_of_pair_information
========================================
Does the STATE change what a PAIR of cells tells about the target?  The information counterpart of 3m (which asked it for the noise correlation):
the pair's information breakdown (3b: I_total, I_lin, I_sigsim, I_cor, RSI) and partial information decomposition (3c: redundancy, unique,
synergy) computed separately in the LOW / MID / HIGH tercile of a state, from the SAME pairs, trials and response bins as 3a / 3b / 3c.

STATES, PAIRS, GLOBAL REMOVAL, windows: as 3m (infotheory/pair_state.py).
    states    PM_lab (leave-pair-out), global (leave-pair-out), pupil, speed; z-scored within the session; terciles within the PAIR's trials;
              state window = STATE_WINDOW (default pre-stim -30..-10 cm; --window LO HI)
    versions  raw  |  resid_global (each cell regressed on the leave-pair-out global signal of the response window BEFORE binning)
    pairs     3a's pair_list (pair_rank < INFO_MAX_PAIRS = 5 per group and session, as 3b / 3c; MAX_PAIRS_BY_GROUP raises it to 20 for PM-lab, deep PM-lab and the two
              V1 x PM-lab cross groups; --max-pairs N sets every group) + V1-lab x PM-lab and
              V1-unl x PM-lab cross pairs (same count)
    windows   INFO_WINDOWS (default stim, reward: the windows where the target is encoded; pre_stim can be added at the top)
    targets   TARGET_STRATEGY of stim_binning and "choice"

BINNING (project rule 4): each cell's responses are cut into RESPONSE_BINS quantile bins ONCE, over all the valid trials of the pair (exactly 3b / 3c),
and the SAME bin edges are used in every state tercile -- so the terciles differ only in which trials they hold, never in how the responses were cut.
The "all" value (no state split; `<measure>_all` columns) is computed with the identical code and trials as 3b / 3c; a printed check compares it with
3b's / 3c's own tables when they exist (same pairs, so the means must agree up to the random shuffle bias correction).
BALANCED TERCILES (BALANCE_TERCILES, default on): the three terciles are subsampled to the same number of trials in every target bin (stimulus bins or
choice), so a state that shifts the lick rate or the stimulus mix cannot change I through the target entropy (this was the likely explanation of the
reward-window pupil / speed columns of the first real-data run), and the floor / bias of the shuffle-corrected MI (floored at 0) is the same in the three
terciles.  Terciles are comparable with EACH OTHER, never with the unsplit `_all` value (smaller n -> more flooring).
A pair enters a state's row only if ALL THREE terciles are analysable (>= MIN_TERCILE_TRIALS trials, 2 target labels after dropping target bins with
< MIN_TRIALS_PER_TARGET_BIN trials), so terciles are compared on the same pairs.

MEASURES (bits): I_total, I_lin, I_sigsim, I_cor, RSI (= I_total - I_lin; > 0 net synergy) | redundancy (I_min, shuffle-corrected), unique (U1 + U2),
synergy.  Per pair: value in tercile 1..3 (`_t1.._t3`), `_hml` = high - low, `_inv` = mid - (low + high) / 2 (> 0 = inverted U), `_all` = no split.
All are averaged over the pairs of a (session, group) first.

--noise-only: restrict every window table to the NOISE trials (0 < signal < 100 %) before anything else (pairs, states, bins, residuals); outputs in out/noise_only/.
The unsplit "all" value then no longer matches 3b / 3c (which use all trials), so the agreement check is skipped.
TESTS: as 3m -- the session mean of `_hml` / `_inv` (sessions with >= MIN_PAIRS_PER_UNIT pairs), t and Wilcoxon across sessions, t across mice,
BH per (version, window, target, measure).  Exploratory.  Pairs of a session share cells and trials: only sessions / mice count.

CAVEATS  (1) A tercile has ~1/3 of the trials: shuffle-corrected information is unbiased on average but noisier, and with few pairs per group
(default 5) group-level effects of a few hundredths of a bit are below what this can detect; raise --max-pairs if the budget allows.
(2) The information measures use 3 response bins per cell: a state that only shifts the cells' mean activity changes which trials fall in which
bin (the bins are fixed over the pair's trials), i.e. it can change I through the MEAN, not only through the noise structure.  That is why the
resid_global version matters and why `I_lin` (single-cell information) is reported beside the pair terms: an effect only in I_lin is single-cell.
(3) Cross pairs have no 3a counterpart; their "all" values cannot be checked against 3b / 3c.

Reads: 1b, 2a, 2b, 3a pair_list.csv (and 3b / 3c outputs for the agreement check).
Outputs  2_pipeline/3n_state_modulation_of_pair_information/out/
    pair_info_state_by_session.csv   per (session, group, window, target, state, version): n_pairs, <measure>_t1/_t2/_t3/_hml/_inv/_all
    pair_info_state_tests.csv        across-session / across-mouse tests
    figures/ 1_itotal_by_state_tercile_<target>_<version>, 2_modulation_<target>_<version>, 3_tstat_<target>_<version>
"""
from __future__ import annotations

import argparse
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
from infotheory.unit_plots import mouse_palette
from infotheory.stim_binning import SIMPLE_STRATEGIES, CHOICE_TARGET, target_signature
from infotheory.info_theory import mutual_information_shuffle
from infotheory.pair_information import digitize_quantile, joint_code, plugin_breakdown_terms
from infotheory.pid import build_joint_distribution, partial_information_decomposition, REDUNDANCY_MEASURES
from infotheory import pair_tables as pt
from infotheory import pair_state as ps

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROTOCOL = "DN"
ALL_WINDOWS = {"pre_stim": (-30.0, -10.0), "stim": (0.0, 20.0), "reward": (25.0, 45.0)}
INFO_WINDOWS = ["stim", "reward"]
WINDOWS = {w: ALL_WINDOWS[w] for w in INFO_WINDOWS}
STATE_WINDOW = (-30.0, -10.0)
TARGET_STRATEGY = SIMPLE_STRATEGIES[0]
TARGETS = [TARGET_STRATEGY, CHOICE_TARGET]
SOURCE_GROUP = "PM_lab"
CROSS_A_GROUPS = ["V1_lab", "V1_unl"]
MIN_SOURCE_CELLS = 5
INFO_MAX_PAIRS = 200                      # default pairs per (session, group) -- the cost per pair is ~40x 3a's, as in 3b / 3c
MAX_PAIRS_BY_GROUP = {"PM_lab": 20, "PMdeep_lab": 20, "V1_lab~PM_lab": 20, "V1_unl~PM_lab": 20}   # more pairs where the PM-lab hint was (pair_rank < cap:
                                        # always a subset of 3a's / 3b's / 3c's pairs, so agreement is unaffected); --max-pairs N overrides every group
BALANCE_TERCILES = True                 # equal number of trials PER TARGET BIN in the three terciles (random subsample, seeded): the terciles then have the
                                        # SAME target distribution and the SAME size, so a state that shifts the lick rate cannot change I through the label
                                        # entropy, and the floor / bias of the shuffle-corrected MI is equal across terciles.  False: all trials of each tercile.
STATES = [SOURCE_GROUP, "global", "pupil", "speed"]
VERSIONS = ["raw", "resid_global"]
R_BINS = cfg.RESPONSE_BINS              # shared with 2c / 3b / 3c
MI_SHUFFLES = 4                         # as 3b
PID_SHUFFLES = 30                       # as 3c
REDUNDANCY = "imin"
MIN_STATE_TRIALS = 60
MIN_TERCILE_TRIALS = 30
MIN_PAIRS_PER_UNIT = 3
MIN_SESSIONS_TEST = 4
ALPHA = 0.05
RANDOM_STATE = 0
JOBLIB_VERBOSE = 5
MEAS = ["I_total", "I_lin", "I_sigsim", "I_cor", "RSI", "redundancy", "unique", "synergy"]
FIG_MEAS = ["I_total", "RSI", "redundancy", "synergy"]
TEST_KINDS = ["hml", "inv"]
STATE_COLORS = {"PM_lab": "#d62728", "global": "#1f77b4", "pupil": "#2ca02c", "speed": "#9467bd"}

parser = argparse.ArgumentParser()
add_scope_args(parser)
parser.add_argument("--window", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="state window in cm rel. to stimStart (default STATE_WINDOW); outputs go to out/window_<lo>_<hi>/")
parser.add_argument("--mi-shuffles", type=int, default=MI_SHUFFLES)
parser.add_argument("--pid-shuffles", type=int, default=PID_SHUFFLES)
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
MAX_PAIRS = args.max_pairs or max([INFO_MAX_PAIRS] + list(MAX_PAIRS_BY_GROUP.values()))
CAPS = {} if args.max_pairs else MAX_PAIRS_BY_GROUP          # per-group caps (the default cap INFO_MAX_PAIRS for the other groups)

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
print(f"{'NOISE TRIALS ONLY (0 < signal < 100 %)' if NOISE_ONLY else 'all engaged trials'}; state window {STATE_WINDOW}; response windows {INFO_WINDOWS}; targets {TARGETS}; pairs per group and session: {INFO_MAX_PAIRS} (caps {CAPS})")

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

pairs_all = ps.build_pair_list(session_ids, pair_list_path, MAX_PAIRS, incl_by_session, cell_group, SOURCE_GROUP, CROSS_A_GROUPS, RANDOM_STATE)
if not pairs_all.empty and not args.max_pairs:
    cap = pairs_all["group"].map(lambda g: MAX_PAIRS_BY_GROUP.get(g, INFO_MAX_PAIRS))
    pairs_all = pairs_all[pairs_all["pair_rank"] < cap].reset_index(drop=True)
if pairs_all.empty:
    raise SystemExit("No pairs (run 3a on the full dataset first).")
session_ids = [s for s in session_ids if s in set(pairs_all["session_id"])]
pairs_by_session = {sid: sub.reset_index(drop=True) for sid, sub in pairs_all.groupby("session_id")}
cells_by_session = {sid: sorted(incl_by_session[sid]) for sid in session_ids}
print(f"{len(session_ids)} sessions, {len(pairs_all)} pairs ({int((pairs_all['kind'] == 'within').sum())} within-group, "
      f"{int((pairs_all['kind'] == 'cross').sum())} cross; pair_rank < {MAX_PAIRS})")
stamps = {s: f"{data_stamp(*selected[s])}-{ps.pair_hash(pairs_by_session[s])}" for s in session_ids}
qc_mtime = int((activity_out / "qc_per_cell.csv").stat().st_mtime)


# --------------------------------------------------------------------- #
# 2. Information of one pair on given trials (binned responses -> breakdown + PID)
# --------------------------------------------------------------------- #
def pair_information(b1, b2, lab, rng):
    """3b's breakdown terms and 3c's PID of PRE-BINNED responses b1, b2 (ints in [0, R_BINS)) about the discrete target `lab`; same estimators and
    bias corrections as compute_pair_breakdown / pid_from_binned_responses."""
    mi1 = mutual_information_shuffle(b1, lab, bins=R_BINS, n_shuffles=args.mi_shuffles, rng=rng).bits_corrected
    mi2 = mutual_information_shuffle(b2, lab, bins=R_BINS, n_shuffles=args.mi_shuffles, rng=rng).bits_corrected
    i_tot = mutual_information_shuffle(joint_code(b1, b2, R_BINS), lab, bins=R_BINS * R_BINS, n_shuffles=args.mi_shuffles, rng=rng).bits_corrected
    terms = plugin_breakdown_terms(b1, b2, lab, R_BINS)
    i_lin = mi1 + mi2
    out = dict(I_total=i_tot, I_lin=i_lin, I_sigsim=terms["I_sigsim"], I_cor=i_tot - i_lin - terms["I_sigsim"], RSI=i_tot - i_lin)
    P = build_joint_distribution(b1, b2, lab)
    res = partial_information_decomposition(P, redundancy_measure=REDUNDANCY)
    red = res.redundancy
    if args.pid_shuffles > 0:
        red -= float(np.mean([REDUNDANCY_MEASURES[REDUNDANCY](build_joint_distribution(b1, b2, rng.permutation(lab)))
                              for _ in range(args.pid_shuffles)]))
    u1, u2 = res.I1 - red, res.I2 - red
    out.update(redundancy=red, unique=u1 + u2, synergy=res.I12 - red - u1 - u2)
    return out


def balance_indices(lab, sel, rng):
    """Row indices of the three terciles (boolean masks `sel` over `lab`) after equalising the number of trials PER TARGET BIN: n_k = the smallest
    count of bin k over the terciles; bins with n_k < MIN_TRIALS_PER_TARGET_BIN are dropped from all three. None if too little is left."""
    vals = np.unique(lab[np.any(sel, axis=0)])
    n_k = {v: min(int(((lab == v) & s_).sum()) for s_ in sel) for v in vals}
    vals = [v for v in vals if n_k[v] >= cfg.MIN_TRIALS_PER_TARGET_BIN]
    if len(vals) < 2 or sum(n_k[v] for v in vals) < MIN_TERCILE_TRIALS:
        return None
    return [np.concatenate([rng.choice(np.flatnonzero((lab == v) & s_), n_k[v], replace=False) for v in vals]) for s_ in sel]


def tercile_ok(lab):
    """The pair_trials rule applied to a tercile: bins with < MIN_TRIALS_PER_TARGET_BIN trials dropped; >= MIN_TERCILE_TRIALS and 2 labels left."""
    vals, counts = np.unique(lab, return_counts=True)
    keep = ~np.isin(lab, vals[counts < cfg.MIN_TRIALS_PER_TARGET_BIN])
    return keep if (keep.sum() >= MIN_TERCILE_TRIALS and len(np.unique(lab[keep])) >= 2) else None


# --------------------------------------------------------------------- #
# 3. Per session
# --------------------------------------------------------------------- #
def process_session(ses, seed):
    sid = ses.session_id
    pairs = pairs_by_session.get(sid)
    cells = list(cells_by_session.get(sid, []))
    if pairs is None or not cells or ses.celldata is None:
        return None
    rng = np.random.default_rng(seed)
    wins = dict(WINDOWS)
    wins["_state"] = STATE_WINDOW
    tables = build_session_tables(ses, mu_sigma_lookup, wins, included_cell_ids=set(cells))
    if not tables or "_state" not in tables:
        return None
    st = tables.pop("_state")
    cells = [c for c in cells if c in st.columns]
    cidx = {c: i for i, c in enumerate(cells)}
    ctx = ps.session_state_context(st, cells, cell_group, SOURCE_GROUP)
    rows = []
    for window, table in tables.items():
        if table.empty or "signal" not in table.columns:
            continue
        if NOISE_ONLY:                                                  # noise trials only: 0 < signal < 100 %
            table = table[(table["signal"] > 0) & (table["signal"] < 100)]
        table = pt.add_target_columns(table, [TARGET_STRATEGY]).reset_index(drop=True)
        wc = ps.window_context(ctx, table, cells)
        for r in pairs.itertuples(index=False):
            if r.cell_i not in cidx or r.cell_j not in cidx:
                continue
            ci, cj = cidx[r.cell_i], cidx[r.cell_j]
            for target in TARGETS:
                d = pt.pair_trials(table, target, r.cell_i, r.cell_j)
                if d is None:
                    continue
                ix = wc["tindex"].get_indexer(d["trial"])
                G = ps.pair_global(wc, ci, cj, ix)
                sv = ps.pair_states(ctx, wc, ci, cj, ix, STATES, SOURCE_GROUP, MIN_SOURCE_CELLS)
                lab = d["label"]
                for ver in VERSIONS:
                    fin = np.isfinite(G) if ver == "resid_global" else np.ones(len(lab), bool)
                    if fin.sum() < MIN_STATE_TRIALS:
                        continue
                    a1 = ps.remove_global(d["x1"], G) if ver == "resid_global" else d["x1"]
                    a2 = ps.remove_global(d["x2"], G) if ver == "resid_global" else d["x2"]
                    a1, a2, lf = a1[fin], a2[fin], lab[fin]
                    b1, b2 = digitize_quantile(a1, R_BINS), digitize_quantile(a2, R_BINS)      # bins over ALL the pair's trials (3b / 3c)
                    keep = tercile_ok(lf)
                    if keep is None:
                        continue
                    all_ = pair_information(b1[keep], b2[keep], lf[keep], rng)
                    for name in STATES:
                        x = sv[name][fin]
                        m = np.isfinite(x)
                        if m.sum() < MIN_STATE_TRIALS:
                            continue
                        e1, e2 = np.quantile(x[m], [1 / 3, 2 / 3])
                        sel = [m & (x <= e1), m & (x > e1) & (x <= e2), m & (x > e2)]
                        res3, ok = [], True
                        if BALANCE_TERCILES:
                            groups_idx = balance_indices(lf, sel, rng)
                            if groups_idx is None:
                                continue
                        else:
                            groups_idx = []
                            for s_ in sel:
                                kp = tercile_ok(lf[s_])
                                if kp is None:
                                    ok = False
                                    break
                                groups_idx.append(np.flatnonzero(s_)[kp])
                            if not ok:
                                continue
                        for idx in groups_idx:
                            res3.append(pair_information(b1[idx], b2[idx], lf[idx], rng))
                        row = dict(group=r.group, window=window, target=target, state=name, version=ver)
                        for k in MEAS:
                            t1, t2, t3 = (res3[0][k], res3[1][k], res3[2][k])
                            row.update({f"{k}_t1": t1, f"{k}_t2": t2, f"{k}_t3": t3, f"{k}_hml": t3 - t1,
                                        f"{k}_inv": t2 - 0.5 * (t1 + t3), f"{k}_all": all_[k]})
                        rows.append(row)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    keys = ["group", "window", "target", "state", "version"]
    g = df.groupby(keys, sort=False)
    out = g[[c for c in df.columns if c not in keys]].mean()
    out["n_pairs"] = g.size()
    out = out.reset_index()
    out.insert(0, "animal_id", ses.animal_id)
    out.insert(0, "session_id", sid)
    print(f"  [{sid}] {len(df)} (pair, window, target, state, version) rows -> {len(out)} summary rows")
    return out


results = run_sessions(
    session_ids, process_session, [PROTOCOL], paths.store, "pair_info_state",
    key_params=dict(version="3n-v3", balance=BALANCE_TERCILES, noise_only=NOISE_ONLY, windows=WINDOWS, state_window=STATE_WINDOW, targets=target_signature(TARGETS), source=SOURCE_GROUP,
                    cross=CROSS_A_GROUPS, min_source=MIN_SOURCE_CELLS, min_state=MIN_STATE_TRIALS, min_tercile=MIN_TERCILE_TRIALS,
                    r_bins=R_BINS, mi_shuffles=args.mi_shuffles, pid_shuffles=args.pid_shuffles, redundancy=REDUNDANCY,
                    min_per_bin=cfg.MIN_TRIALS_PER_TARGET_BIN, max_pairs=MAX_PAIRS, qc_mtime=qc_mtime,
                    fits_mtime=int(fits_path.stat().st_mtime)),
    load_kwargs=dict(load_behaviordata=True, load_videodata=True, load_celldata=True, load_calciumdata=True),
    cells_by_session=cells_by_session, base_seed=RANDOM_STATE, n_jobs=scope.n_jobs, recompute=args.recompute,
    verbose=JOBLIB_VERBOSE, extra_key_by_session=stamps, flatten=False)
results = [r for r in results if r is not None and len(r)]
if not results:
    raise SystemExit("No session produced pair information results.")
S = pd.concat(results, ignore_index=True)
del results
S.to_csv(table_dir / "pair_info_state_by_session.csv", index=False, float_format="%.5g")
print(f"\n{S['session_id'].nunique()} sessions; {len(S)} summary rows")

# agreement of the unsplit ("all") values with 3b / 3c (same pairs, same trials, same bins)
for name, fn, col_map in (() if NOISE_ONLY else (("3b", paths.out_from("3b_pair_information_breakdown") / "pair_information_breakdown.csv",
                           {"I_total": "I_total", "I_lin": "I_lin", "RSI": "RSI"}),
                          ("3c", paths.out_from("3c_pair_information_pid") / "pair_pid.csv",
                           {"redundancy": "redundancy", "synergy": "synergy"}))):
    if not fn.exists():
        print(f"(agreement check with {name}: {fn.name} not found)")
        continue
    ref = pd.read_csv(fn)
    ref = ref[ref["window"].isin(INFO_WINDOWS)]
    if "group" not in ref.columns:
        ref = pt.finalize_pair_table(ref)
    ref = ref[ref["target"].isin(TARGETS)]
    cap_ref = ref["group"].map(lambda g: INFO_MAX_PAIRS if args.max_pairs is None else args.max_pairs) if args.max_pairs or not CAPS else \
        ref["group"].map(lambda g: MAX_PAIRS_BY_GROUP.get(g, INFO_MAX_PAIRS))
    rk = ref[ref["pair_rank"] < cap_ref].groupby(["session_id", "group", "window", "target"])[list(col_map)].mean()
    mine = S[(S["state"] == "global") & (S["version"] == "raw")].set_index(["session_id", "group", "window", "target"])
    j = rk.join(mine[[f"{k}_all" for k in col_map]], how="inner")
    for k in col_map:
        if len(j) > 3:
            print(f"  agreement with {name} {k}: n={len(j)} (session, group, window, target) cells, r = {np.corrcoef(j[k], j[f'{k}_all'])[0, 1]:.3f}, "
                  f"mean |diff| = {(j[k] - j[f'{k}_all']).abs().mean():.4f} bits")

# --------------------------------------------------------------------- #
# 4. Tests across sessions / mice
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
for (grp, win, tgt, state, ver), sub in Sok.groupby(["group", "window", "target", "state", "version"]):
    for k in MEAS:
        for kind in TEST_KINDS:
            col = f"{k}_{kind}"
            n, mean, t, p, pw = one_sample(sub[col])
            if n < MIN_SESSIONS_TEST:
                continue
            mice = sub.groupby("animal_id").apply(lambda d: np.average(d[col].dropna(), weights=d.loc[d[col].notna(), "n_pairs"])
                                                  if d[col].notna().any() else np.nan, include_groups=False)
            nm, mmean, mt, mp, _ = one_sample(mice.to_numpy())
            trows.append(dict(group=grp, window=win, target=tgt, state=state, version=ver, measure=k, kind=kind, n_sessions=n, mean=mean,
                              t=t, p_session=p, p_wilcoxon=pw, n_mice=nm, mean_mice=mmean, t_mice=mt, p_mice=mp,
                              n_mice_positive=int((mice > 0).sum()), n_mice_negative=int((mice < 0).sum())))
tests = pd.DataFrame(trows)
if len(tests):
    tests["q_session"] = np.nan
    for _, idx in tests.groupby(["version", "window", "target", "measure", "kind"]).groups.items():
        tests.loc[idx, "q_session"] = bh(tests.loc[idx, "p_session"].to_numpy())
    tests.to_csv(table_dir / "pair_info_state_tests.csv", index=False, float_format="%.5g")
    print("\nAcross-session tests (high - low state tercile), sorted by p; q = BH within version x window x target x measure x kind:")
    show = tests[tests["kind"] == "hml"].sort_values("p_session")
    print(show.head(20)[["group", "window", "target", "state", "version", "measure", "n_sessions", "mean", "p_session", "q_session",
                         "n_mice", "p_mice", "n_mice_positive"]].round(4).to_string(index=False))
    print(f"\n{int((tests['q_session'] < ALPHA).sum())} / {len(tests)} tests with q < {ALPHA}")

# --------------------------------------------------------------------- #
# 5. Figures
# --------------------------------------------------------------------- #
agg_note = {"pooled": "pair-weighted mean over all sessions (no error bars)", "session": "mean +- SEM over sessions",
            "mouse": "mean +- SEM over mice"}[AGG]
KEYS = {"pooled": [], "session": ["animal_id", "session_id"], "mouse": ["animal_id"]}[AGG]


def unit_values(df, value, by):
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


for tgt in TARGETS:
    for ver in VERSIONS:
        Sv = S[(S["version"] == ver) & (S["target"] == tgt)]
        if Sv.empty:
            continue
        # 1: I_total in state terciles, lines per group
        fig, axes = plt.subplots(len(STATES), len(WIN), figsize=(4.3 * len(WIN), 2.9 * len(STATES)), squeeze=False, sharey="row")
        for ri, state in enumerate(STATES):
            for ci_, w in enumerate(WIN):
                ax = axes[ri][ci_]
                sub = Sv[(Sv["state"] == state) & (Sv["window"] == w)]
                for g in GROUPS:
                    ms = [mean_sem(unit_values(sub[sub["group"] == g], f"I_total_t{t_}", ["group"]), ["group"]) for t_ in (1, 2, 3)]
                    if any(m.empty for m in ms):
                        continue
                    y = [float(m["mean"].iloc[0]) for m in ms]
                    e = [0 if not np.isfinite(m["sem"].iloc[0]) else float(m["sem"].iloc[0]) for m in ms]
                    ax.errorbar([1, 2, 3], y, yerr=e, color=GCOLOR[g], marker="o", ms=3.5, lw=1.1, capsize=2, label=GLABEL[g])
                ax.set_xticks([1, 2, 3])
                ax.set_xticklabels(["low", "mid", "high"])
                if ri == 0:
                    ax.set_title(f"{w} ({WINDOWS[w][0]:g}..{WINDOWS[w][1]:g} cm)", fontsize=10)
                if ci_ == 0:
                    ax.set_ylabel(f"{state}\nI_total (bits)")
                if ri == len(STATES) - 1:
                    ax.set_xlabel("state tercile (within pair)")
        axes[0][0].legend(fontsize=5.5, frameon=False, ncol=2)
        fig.suptitle(f"Joint pair information about {tgt} by state tercile [{ver}]; state window {STATE_WINDOW[0]:g}..{STATE_WINDOW[1]:g} cm; {agg_note}",
                     y=1.0, fontsize=9)
        fig.tight_layout()
        save_fig(fig, figdir, f"1_itotal_by_state_tercile_{tgt}_{ver}")
        plt.close(fig)

        # 2: high - low per group, states as series, one row per measure
        fig, axes = plt.subplots(len(FIG_MEAS), len(WIN), figsize=(4.8 * len(WIN), 2.9 * len(FIG_MEAS)), squeeze=False, sharey="row")
        xs = np.arange(len(GROUPS))
        for ri, meas in enumerate(FIG_MEAS):
            for ci_, w in enumerate(WIN):
                ax = axes[ri][ci_]
                for si, state in enumerate(STATES):
                    sub = Sv[(Sv["state"] == state) & (Sv["window"] == w)]
                    uv = unit_values(sub, f"{meas}_hml", ["group"])
                    ms = mean_sem(uv, ["group"]).set_index("group") if len(uv) else pd.DataFrame()
                    for gi, g in enumerate(GROUPS):
                        if g in ms.index:
                            ax.errorbar(gi + (si - (len(STATES) - 1) / 2) * 0.17, ms.loc[g, "mean"],
                                        yerr=0 if not np.isfinite(ms.loc[g, "sem"]) else ms.loc[g, "sem"], color=STATE_COLORS[state],
                                        marker="o", ms=4, capsize=2, lw=1, label=state)
                ax.axhline(0, color="grey", lw=0.7)
                ax.set_xticks(xs)
                ax.set_xticklabels([GLABEL[g] for g in GROUPS] if ri == len(FIG_MEAS) - 1 else [], rotation=60, ha="right", fontsize=7)
                if ri == 0:
                    ax.set_title(f"{w} ({WINDOWS[w][0]:g}..{WINDOWS[w][1]:g} cm)", fontsize=10)
                if ci_ == 0:
                    ax.set_ylabel(f"{meas}\nhigh - low (bits)")
        h, l_ = axes[0][0].get_legend_handles_labels()
        axes[0][0].legend(list(dict(zip(l_, h)).values()), list(dict(zip(l_, h)).keys()), fontsize=7, frameon=False)
        fig.suptitle(f"State modulation of pair information about {tgt} [{ver}]: high minus low state tercile; {agg_note}", y=1.0, fontsize=9)
        fig.tight_layout()
        save_fig(fig, figdir, f"2_modulation_{tgt}_{ver}")
        plt.close(fig)

        # 3: t across sessions (I_total and RSI), group x (window, state)
        if len(tests):
            cols = [(w, s_) for w in WIN for s_ in STATES]
            fig, axes = plt.subplots(1, 2, figsize=(2 * (0.9 * len(cols) + 3), 0.5 * len(GROUPS) + 2.5), squeeze=False)
            for ax, meas in zip(axes[0], ["I_total", "RSI"]):
                tv = tests[(tests["version"] == ver) & (tests["target"] == tgt) & (tests["measure"] == meas) & (tests["kind"] == "hml")]
                M = np.full((len(GROUPS), len(cols)), np.nan)
                Q = np.full_like(M, np.nan)
                for gi, g in enumerate(GROUPS):
                    for ci_, (w, s_) in enumerate(cols):
                        r = tv[(tv["group"] == g) & (tv["window"] == w) & (tv["state"] == s_)]
                        if len(r):
                            M[gi, ci_], Q[gi, ci_] = r["t"].iloc[0], r["q_session"].iloc[0]
                im = ax.imshow(M, cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto")
                for gi in range(len(GROUPS)):
                    for ci_ in range(len(cols)):
                        if np.isfinite(M[gi, ci_]):
                            ax.text(ci_, gi, f"{M[gi, ci_]:.1f}" + ("*" if Q[gi, ci_] < ALPHA else ""), ha="center", va="center", fontsize=6.5)
                ax.set_xticks(range(len(cols)))
                ax.set_xticklabels([f"{w}\n{s_}" for w, s_ in cols], fontsize=6.5)
                ax.set_yticks(range(len(GROUPS)))
                ax.set_yticklabels([GLABEL[g] for g in GROUPS], fontsize=8)
                ax.set_title(f"{meas}: high - low state tercile, t across sessions", fontsize=8)
            fig.colorbar(im, ax=axes[0].tolist(), label="t across sessions")
            fig.suptitle(f"{tgt} [{ver}] (* = BH q < {ALPHA}; blank = < {MIN_SESSIONS_TEST} sessions)", fontsize=9)
            save_fig(fig, figdir, f"3_tstat_{tgt}_{ver}")
            plt.close(fig)

print(f"\nWrote tables to {table_dir}, figures to {figdir}")
