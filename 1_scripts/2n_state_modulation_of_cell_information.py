# -*- coding: utf-8 -*-
"""
2n_state_modulation_of_cell_information
========================================
Does the STATE change how much ONE CELL tells about the choice / the stimulus -- and is that explained by LICKING?

3n found that the state effect on pair information is a single-cell effect: in the reward window, cells carry MORE information about the choice
when the pre-stimulus running speed is high (and less with high global activity / pupil), with no change of the pairwise terms.  The choice IS the
lick, and the reward window is where the mouse licks, so the simplest explanation is motor: at high speed the mouse licks more (or moves more) and the
lick-related activity is larger.  This script (a) repeats the test on single cells (all included cells -> more power than 5-20 pairs per group) and
(b) controls for licking.

PER CELL: the response (window mean; INFO_WINDOWS = stim, reward) is cut into RESPONSE_BINS quantile bins ONCE over all the cell's trials (as 2c / 3b);
the shuffle-corrected mutual information with the target (choice, or the stimulus bins of stim_binning's TARGET_STRATEGY) is computed in the LOW / MID /
HIGH tercile of each state.  `_hml` = high - low, `_inv` = mid - (low + high) / 2, `_all` = no split.
    states    PM_lab (leave-cell-out), global (all included cells outside PM-lab, leave-cell-out), pupil, speed; state window STATE_WINDOW (--window)
    versions  raw | resid_global (the cell regressed on the leave-cell-out global signal of the response window before binning)

MATCHING MODES (how the trials of the three terciles are subsampled before the information is computed; random, seeded):
    balanced            same number of trials per TARGET BIN in the three terciles (as 3n): removes label-entropy / lick-RATE differences
    lick_matched        [choice target] additionally the same number of trials per LICK STRATUM: the lick trials are split into N_LICK_STRATA quantile
                        strata of the lick rate (events / s) in the RESPONSE window, the no-lick trials into "no stray lick" / "stray licks".  After
                        matching, the three terciles have the same distribution of licking VIGOUR.
    lick_speed_matched  [choice target] additionally matched on the running speed in the RESPONSE window (median split within lick / no-lick trials)
HOW TO READ IT: if the high - low difference of the choice information is still there under lick_matched (and lick_speed_matched), the state changes the
gain of the choice / lick-related activity beyond the amount of licking; if it disappears, the effect was licking vigour (or locomotion) itself.
The behaviour table / figure 3 shows directly whether the lick rate and the speed in the response window differ between the state terciles.
Matching leaves fewer trials (the strictest mode the fewest): compare modes by the SIGN and size of the effect, not only by significance.

TESTS: per (group, window, target, state, version, mode): mean over the cells of a session, t / Wilcoxon across sessions, t across mice.  The group
"ALL" = all analysed cells of the session together: ONE confirmatory test per state (the groups share sessions and mice, so the per-group tests are not
independent replications).  BH q across (groups x states) within (version, window, target, mode, kind), "ALL" kept apart (q over the states).

Project-wide rules: --animals / --sessions / --aggregate / --max-cells (cap per group and session; the STATES always use all included cells) /
--noise-only (0 < signal < 100 %) / --window LO HI.  One session per worker; a session is reduced to a small summary (no per-cell table on disk).
Bins: RESPONSE_BINS and the estimator are those of 2c / 3b; the terciles use the same bin edges as the unsplit value.

Reads: 1b psychometric_fits.csv, 2a celldata_combined.csv, 2b qc_per_cell.csv.
Outputs  2_pipeline/2n_state_modulation_of_cell_information/out/
    cell_info_state_by_session.csv   per (session, group, window, target, state, version, mode): n_cells, MI_t1/_t2/_t3/_hml/_inv/_all, n_trials (per tercile)
    cell_info_state_tests.csv        tests (group "ALL" = the confirmatory ones)
    state_tercile_behaviour.csv      lick rate / running speed in the response window per state tercile (lick and no-lick trials)
    figures/ 1_modulation_<target>_<version>, 2_all_cells_modes_<version>, 3_behaviour_by_state_tercile
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
from infotheory.unit_stats import stable_seed
from infotheory.unit_plots import mouse_palette
from infotheory.stim_binning import SIMPLE_STRATEGIES, CHOICE_TARGET, target_signature
from infotheory.info_theory import mutual_information_shuffle
from infotheory.pair_information import digitize_quantile
from infotheory import pair_tables as pt
from infotheory import pair_state as ps

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROTOCOL = "DN"
ALL_WINDOWS = {"pre_stim": (-30.0, -10.0), "stim": (0.0, 20.0), "reward": (25.0, 45.0)}
INFO_WINDOWS = ["stim", "reward"]
WINDOWS = {w: ALL_WINDOWS[w] for w in INFO_WINDOWS}
STATE_WINDOW = (-30.0, -10.0)
TARGET_STRATEGY = SIMPLE_STRATEGIES[0]
TARGETS = [CHOICE_TARGET, TARGET_STRATEGY]
SOURCE_GROUP = "PM_lab"
MIN_SOURCE_CELLS = 5
STATES = [SOURCE_GROUP, "global", "pupil", "speed"]
VERSIONS = ["raw", "resid_global"]
MODES = ["balanced", "lick_matched", "lick_speed_matched"]     # the last two only for the choice target
N_LICK_STRATA = 3                       # quantile strata of the lick rate among the lick trials
R_BINS = cfg.RESPONSE_BINS
MI_SHUFFLES = 4
MIN_STATE_TRIALS = 60
MIN_TERCILE_TRIALS = 30
MIN_CELLS_PER_UNIT = 5
MIN_SESSIONS_TEST = 4
ALPHA = 0.05
RANDOM_STATE = 0
JOBLIB_VERBOSE = 5
STATE_COLORS = {"PM_lab": "#d62728", "global": "#1f77b4", "pupil": "#2ca02c", "speed": "#9467bd"}

parser = argparse.ArgumentParser()
add_scope_args(parser)
parser.add_argument("--window", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="state window in cm rel. to stimStart (default STATE_WINDOW); outputs go to out/window_<lo>_<hi>/")
parser.add_argument("--noise-only", action="store_true", help="only the noise trials (0 < signal < 100 %); outputs go to out/noise_only/")
parser.add_argument("--mi-shuffles", type=int, default=MI_SHUFFLES)
args = parser.parse_args()
NOISE_ONLY = bool(args.noise_only)
if args.window:
    STATE_WINDOW = (float(args.window[0]), float(args.window[1]))
scope = scope_from_args(args)
AGG = scope.aggregate
UNIT = unit_noun(AGG)
MAX_CELLS = scope.max_cells

paths = get_pipeline_paths(__file__)
table_dir, figdir = scoped_dirs(paths, scope)
for sub in ([f"window_{STATE_WINDOW[0]:g}_{STATE_WINDOW[1]:g}"] if args.window else []) + (["noise_only"] if NOISE_ONLY else []):
    table_dir, figdir = table_dir / sub, figdir / sub
    table_dir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
clear_figures(figdir)
set_style()
print(scope.describe())
print(f"{'NOISE TRIALS ONLY' if NOISE_ONLY else 'all engaged trials'}; state window {STATE_WINDOW}; response windows {INFO_WINDOWS}; targets {TARGETS}; "
      f"cells per group and session: {MAX_CELLS or 'all'}")

# --------------------------------------------------------------------- #
# 1. Inputs
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
selected = {f"{a}_{d}": (p, a, d) for p, a, d in select_sessions(all_triples, scope)}
session_ids = [s for s in selected if s in included_all]
if not session_ids:
    raise SystemExit("None of the selected sessions passed 1b_psychometric's inclusion criteria.")
mouse_colors = mouse_palette(a for _, a, _ in all_triples)
cell_inclusion = load_cell_inclusion(cell_dist_out, activity_out)
cell_group = dict(zip(cell_inclusion["cell_id"], cell_inclusion["group"]))
cells_by_session = {sid: sorted(sub.loc[sub["include"], "cell_id"]) for sid, sub in cell_inclusion.groupby("session_id") if sid in session_ids}
session_ids = [s for s in session_ids if cells_by_session.get(s)]
GROUP_KEYS = [group_key(g) for g in SIX_GROUP_ORDER]
stamps = {s: data_stamp(*selected[s]) for s in session_ids}
qc_mtime = int((activity_out / "qc_per_cell.csv").stat().st_mtime)
print(f"{len(session_ids)} sessions, {sum(len(v) for v in cells_by_session.values())} included cells")


# --------------------------------------------------------------------- #
# 2. Helpers
# --------------------------------------------------------------------- #
def balance_indices(strat, sel, rng):
    """Row indices of the three terciles (boolean masks `sel`) after equalising the number of trials per value of `strat` (n_k = the smallest
    count over the terciles; strata with n_k < MIN_TRIALS_PER_TARGET_BIN are dropped from all three)."""
    vals = np.unique(strat[np.any(sel, axis=0)])
    n_k = {v: min(int(((strat == v) & s_).sum()) for s_ in sel) for v in vals}
    vals = [v for v in vals if n_k[v] >= cfg.MIN_TRIALS_PER_TARGET_BIN]
    if not vals or sum(n_k[v] for v in vals) < MIN_TERCILE_TRIALS:
        return None
    return [np.concatenate([rng.choice(np.flatnonzero((strat == v) & s_), n_k[v], replace=False) for v in vals]) for s_ in sel]


def quantile_strata(v, n):
    """0..n-1 quantile strata of v (ties stay together; fewer strata if v has few distinct values)."""
    v = np.asarray(v, float)
    edges = np.unique(np.quantile(v, np.linspace(0, 1, n + 1)[1:-1])) if len(v) else np.array([])
    return np.searchsorted(edges, v, side="left")


def window_strata(table):
    """Per trial of a window table: lick stratum (within choice) and speed stratum (median split within choice), as integers."""
    choice = table["choice"].to_numpy(float)
    lick = table["cur_lick"].to_numpy(float) if "cur_lick" in table else np.zeros(len(table))
    speed = table["cur_runspeed"].to_numpy(float) if "cur_runspeed" in table else np.zeros(len(table))
    lick = np.nan_to_num(lick)
    ls = np.zeros(len(table), int)
    yes = choice == 1
    if yes.any():
        ls[yes] = quantile_strata(lick[yes], N_LICK_STRATA)
    ls[~yes] = (lick[~yes] > 0).astype(int)
    ss = np.zeros(len(table), int)
    for cls in (yes, ~yes):
        if cls.any() and np.isfinite(speed[cls]).any():
            ss[cls] = (np.nan_to_num(speed[cls], nan=np.nanmedian(speed[cls])) > np.nanmedian(speed[cls])).astype(int)
    return ls, ss, lick, speed


def mi(b, lab, rng):
    return mutual_information_shuffle(b, lab, bins=R_BINS, n_shuffles=args.mi_shuffles, rng=rng).bits_corrected


# --------------------------------------------------------------------- #
# 3. Per session
# --------------------------------------------------------------------- #
def process_session(ses, seed):
    sid = ses.session_id
    cells = list(cells_by_session.get(sid, []))
    if not cells or ses.celldata is None:
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
    # analysed cells: all, or the first MAX_CELLS of each group in a fixed pseudo-random order
    analysed = []
    for g in GROUP_KEYS:
        gc = [c for c in cells if cell_group.get(c) == g]
        if MAX_CELLS and len(gc) > MAX_CELLS:
            order = np.random.default_rng(stable_seed(sid, "cells", g, base=RANDOM_STATE)).permutation(len(gc))[:MAX_CELLS]
            gc = [gc[i] for i in sorted(order)]
        analysed += gc
    rows, beh = [], []
    for window, table in tables.items():
        if table.empty or "signal" not in table.columns:
            continue
        if NOISE_ONLY:
            table = table[(table["signal"] > 0) & (table["signal"] < 100)]
        table = pt.add_target_columns(table, [TARGET_STRATEGY]).reset_index(drop=True)
        if len(table) < MIN_STATE_TRIALS:
            continue
        wc = ps.window_context(ctx, table, cells)
        lick_s, speed_s, lick, speed = window_strata(table)
        choice = table["choice"].to_numpy(float)

        def at(v):
            out = np.full(len(wc["tnum"]), np.nan)
            out[wc["has"]] = v[wc["srow"][wc["has"]]]
            return out
        face = {k: (at(ctx["face"][k]) if ctx["face"][k] is not None else np.full(len(table), np.nan)) for k in ("pupil", "speed")}
        S_src, N_src = ctx["sums"]["src"]
        S_oth, N_oth = ctx["sums"]["oth"]
        # behaviour per state tercile (session-level states, no cell left out)
        sess_state = {SOURCE_GROUP: at(ps.zvec(np.where(N_src >= MIN_SOURCE_CELLS, S_src / np.maximum(N_src, 1), np.nan))),
                      "global": at(ps.zvec(np.where(N_oth >= MIN_SOURCE_CELLS, S_oth / np.maximum(N_oth, 1), np.nan))),
                      "pupil": face["pupil"], "speed": face["speed"]}
        for name in STATES:
            x = sess_state[name]
            m = np.isfinite(x)
            if m.sum() < MIN_STATE_TRIALS:
                continue
            e1, e2 = np.quantile(x[m], [1 / 3, 2 / 3])
            for t_, s_ in enumerate([m & (x <= e1), m & (x > e1) & (x <= e2), m & (x > e2)], 1):
                for cname, cls in (("lick", choice == 1), ("no_lick", choice == 0)):
                    k = s_ & cls
                    beh.append(dict(window=window, state=name, tercile=t_, trials=cname, n=int(k.sum()), p_lick=float(np.mean(choice[s_])),
                                    lick_rate=float(np.mean(lick[k])) if k.any() else np.nan,
                                    run_speed=float(np.nanmean(speed[k])) if k.any() else np.nan))
        labels = {t: table[pt.target_column(t)].to_numpy(float) for t in TARGETS if pt.target_column(t) in table.columns}
        for c in analysed:
            ci = cidx[c]
            xr = table[c].to_numpy(float)
            G = (wc["Sall"] - wc["Z0"][:, ci]) / np.maximum(wc["Nall"] - wc["C"][:, ci], 1)
            sv = {}
            for name, (S_, N_), member in ((SOURCE_GROUP, (S_src, N_src), ctx["src"]), ("global", (S_oth, N_oth), ctx["oth"])):
                n_ = N_ - ctx["Cs"][:, ci] * member[ci]
                sv[name] = at(ps.zvec(np.where(n_ >= MIN_SOURCE_CELLS, (S_ - ctx["Zs0"][:, ci] * member[ci]) / np.maximum(n_, 1), np.nan)))
            sv["pupil"], sv["speed"] = face["pupil"], face["speed"]
            for target, lab_all in labels.items():
                ok = np.isfinite(xr) & np.isfinite(lab_all)
                vals, counts = np.unique(lab_all[ok], return_counts=True)
                ok &= ~np.isin(lab_all, vals[counts < cfg.MIN_TRIALS_PER_TARGET_BIN])
                if ok.sum() < MIN_STATE_TRIALS or len(np.unique(lab_all[ok])) < 2:
                    continue
                modes = MODES if target == CHOICE_TARGET else ["balanced"]
                for ver in VERSIONS:
                    use = ok & np.isfinite(G) if ver == "resid_global" else ok
                    if use.sum() < MIN_STATE_TRIALS:
                        continue
                    a = ps.remove_global(xr[use], G[use]) if ver == "resid_global" else xr[use]
                    b = digitize_quantile(a, R_BINS)
                    lab = lab_all[use]
                    all_ = mi(b, lab, rng)
                    strat = {"balanced": lab, "lick_matched": lab * 100 + lick_s[use],
                             "lick_speed_matched": lab * 100 + lick_s[use] * 10 + speed_s[use]}
                    for name in STATES:
                        x = sv[name][use]
                        m = np.isfinite(x)
                        if m.sum() < MIN_STATE_TRIALS:
                            continue
                        e1, e2 = np.quantile(x[m], [1 / 3, 2 / 3])
                        sel = [m & (x <= e1), m & (x > e1) & (x <= e2), m & (x > e2)]
                        for mode in modes:
                            idx3 = balance_indices(strat[mode], sel, rng)
                            if idx3 is None or any(len(np.unique(lab[i])) < 2 for i in idx3):
                                continue
                            t1, t2, t3 = (mi(b[i], lab[i], rng) for i in idx3)
                            rows.append((cell_group.get(c), window, target, name, ver, mode, t1, t2, t3, t3 - t1, t2 - 0.5 * (t1 + t3), all_, len(idx3[0])))
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["group", "window", "target", "state", "version", "mode", "MI_t1", "MI_t2", "MI_t3", "MI_hml", "MI_inv", "MI_all",
                                     "n_trials"])
    keys = ["group", "window", "target", "state", "version", "mode"]
    vals = [c for c in df.columns if c not in keys]

    def summarise(d):
        g = d.groupby(keys, sort=False)
        o = g[vals].mean()
        o["n_cells"] = g.size()
        return o.reset_index()
    out = pd.concat([summarise(df), summarise(df.assign(group="ALL"))], ignore_index=True)
    out.insert(0, "animal_id", ses.animal_id)
    out.insert(0, "session_id", sid)
    bdf = pd.DataFrame(beh)
    bdf.insert(0, "animal_id", ses.animal_id)
    bdf.insert(0, "session_id", sid)
    print(f"  [{sid}] {len(analysed)} cells, {len(df)} rows -> {len(out)} summary rows")
    return dict(summary=out, behaviour=bdf)


results = run_sessions(
    session_ids, process_session, [PROTOCOL], paths.store, "cell_info_state",
    key_params=dict(version="2n-v1", windows=WINDOWS, state_window=STATE_WINDOW, targets=target_signature(TARGETS), source=SOURCE_GROUP,
                    min_source=MIN_SOURCE_CELLS, min_state=MIN_STATE_TRIALS, min_tercile=MIN_TERCILE_TRIALS, r_bins=R_BINS,
                    mi_shuffles=args.mi_shuffles, modes=MODES, lick_strata=N_LICK_STRATA, noise_only=NOISE_ONLY, max_cells=MAX_CELLS,
                    min_per_bin=cfg.MIN_TRIALS_PER_TARGET_BIN, qc_mtime=qc_mtime, fits_mtime=int(fits_path.stat().st_mtime)),
    load_kwargs=dict(load_behaviordata=True, load_videodata=True, load_celldata=True, load_calciumdata=True),
    cells_by_session=cells_by_session, base_seed=RANDOM_STATE, n_jobs=scope.n_jobs, recompute=args.recompute,
    verbose=JOBLIB_VERBOSE, extra_key_by_session=stamps, flatten=False)
results = [r for r in results if r is not None]
if not results:
    raise SystemExit("No session produced results.")
S = pd.concat([r["summary"] for r in results], ignore_index=True)
BEH = pd.concat([r["behaviour"] for r in results], ignore_index=True)
del results
S.to_csv(table_dir / "cell_info_state_by_session.csv", index=False, float_format="%.5g")
BEH.to_csv(table_dir / "state_tercile_behaviour.csv", index=False, float_format="%.5g")
print(f"\n{S['session_id'].nunique()} sessions; {len(S)} summary rows")

# --------------------------------------------------------------------- #
# 4. Tests
# --------------------------------------------------------------------- #
GROUPS = GROUP_KEYS
GLABEL = {group_key(g): SIX_GROUP_LABELS[g] for g in SIX_GROUP_ORDER}
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
Sok = S[S["n_cells"] >= MIN_CELLS_PER_UNIT]
for (grp, win, tgt, state, ver, mode), sub in Sok.groupby(["group", "window", "target", "state", "version", "mode"]):
    for kind in ("hml", "inv"):
        col = f"MI_{kind}"
        n, mean, t, p, pw = one_sample(sub[col])
        if n < MIN_SESSIONS_TEST:
            continue
        mice = sub.groupby("animal_id")[col].mean()
        nm, mmean, mt, mp, _ = one_sample(mice.to_numpy())
        trows.append(dict(group=grp, window=win, target=tgt, state=state, version=ver, mode=mode, kind=kind, n_sessions=n, mean=mean, t=t,
                          p_session=p, p_wilcoxon=pw, n_mice=nm, mean_mice=mmean, t_mice=mt, p_mice=mp,
                          n_mice_positive=int((mice > 0).sum()), n_mice_negative=int((mice < 0).sum()),
                          mean_n_trials=float(sub["n_trials"].mean()), mean_MI_all=float(sub["MI_all"].mean())))
tests = pd.DataFrame(trows)
if len(tests):
    tests["q_session"] = np.nan
    tests["_all"] = tests["group"] == "ALL"
    for _, idx in tests.groupby(["_all", "version", "window", "target", "mode", "kind"]).groups.items():
        tests.loc[idx, "q_session"] = bh(tests.loc[idx, "p_session"].to_numpy())
    tests = tests.drop(columns="_all")
    tests.to_csv(table_dir / "cell_info_state_tests.csv", index=False, float_format="%.5g")
    conf = tests[(tests["group"] == "ALL") & (tests["kind"] == "hml")].sort_values(["target", "window", "version", "state", "mode"])
    print("\nCONFIRMATORY tests (all cells of a session together; high - low state tercile, bits):")
    print(conf[["target", "window", "version", "state", "mode", "n_sessions", "mean", "p_session", "n_mice", "p_mice", "n_mice_positive",
                "mean_n_trials"]].round(4).to_string(index=False))

# --------------------------------------------------------------------- #
# 5. Figures
# --------------------------------------------------------------------- #
agg_note = {"pooled": "cell-weighted mean over all sessions (no error bars)", "session": "mean +- SEM over sessions",
            "mouse": "mean +- SEM over mice"}[AGG]
KEYS = {"pooled": [], "session": ["animal_id", "session_id"], "mouse": ["animal_id"]}[AGG]


def unit_values(df, value, by):
    d = df[np.isfinite(df[value]) & (df["n_cells"] >= (1 if AGG == "pooled" else MIN_CELLS_PER_UNIT))].copy()
    d["_v"] = d[value] * d["n_cells"]
    g = d.groupby(by + KEYS).agg(_v=("_v", "sum"), _w=("n_cells", "sum")).reset_index()
    g["value"] = g["_v"] / g["_w"]
    return g


def mean_sem(g, by):
    m = g.groupby(by)["value"].agg(["mean", "sem", "count"]).reset_index()
    if AGG == "pooled":
        m["sem"] = np.nan
    return m


for tgt in TARGETS:
    modes = MODES if tgt == CHOICE_TARGET else ["balanced"]
    for ver in VERSIONS:
        Sv = S[(S["version"] == ver) & (S["target"] == tgt) & (S["group"] != "ALL")]
        if Sv.empty:
            continue
        fig, axes = plt.subplots(len(modes), len(WIN), figsize=(4.8 * len(WIN), 3.0 * len(modes) + 0.8), squeeze=False, sharey=True)
        for ri, mode in enumerate(modes):
            for ci_, w in enumerate(WIN):
                ax = axes[ri][ci_]
                for si, state in enumerate(STATES):
                    sub = Sv[(Sv["state"] == state) & (Sv["window"] == w) & (Sv["mode"] == mode)]
                    uv = unit_values(sub, "MI_hml", ["group"])
                    ms = mean_sem(uv, ["group"]).set_index("group") if len(uv) else pd.DataFrame()
                    for gi, g in enumerate(GROUPS):
                        if g in ms.index:
                            ax.errorbar(gi + (si - (len(STATES) - 1) / 2) * 0.17, ms.loc[g, "mean"],
                                        yerr=0 if not np.isfinite(ms.loc[g, "sem"]) else ms.loc[g, "sem"], color=STATE_COLORS[state],
                                        marker="o", ms=4, capsize=2, lw=1, label=state)
                ax.axhline(0, color="grey", lw=0.7)
                ax.set_xticks(range(len(GROUPS)))
                ax.set_xticklabels([GLABEL[g] for g in GROUPS] if ri == len(modes) - 1 else [], rotation=60, ha="right", fontsize=7)
                if ri == 0:
                    ax.set_title(f"{w} ({WINDOWS[w][0]:g}..{WINDOWS[w][1]:g} cm)", fontsize=10)
                if ci_ == 0:
                    ax.set_ylabel(f"{mode}\nMI high - low (bits)")
        h, l_ = axes[0][0].get_legend_handles_labels()
        axes[0][0].legend(list(dict(zip(l_, h)).values()), list(dict(zip(l_, h)).keys()), fontsize=7, frameon=False)
        fig.suptitle(f"State modulation of single-cell information about {tgt} [{ver}]: high minus low state tercile; {agg_note}", y=1.0, fontsize=9)
        fig.tight_layout()
        save_fig(fig, figdir, f"1_modulation_{tgt}_{ver}")
        plt.close(fig)

# 2: the confirmatory picture -- all cells, choice target, modes side by side, one dot per unit
for ver in VERSIONS:
    Sv = S[(S["version"] == ver) & (S["target"] == CHOICE_TARGET) & (S["group"] == "ALL")]
    if Sv.empty:
        continue
    fig, axes = plt.subplots(len(WIN), len(STATES), figsize=(3.2 * len(STATES), 3.0 * len(WIN)), squeeze=False, sharey=True)
    for ri, w in enumerate(WIN):
        for ci_, state in enumerate(STATES):
            ax = axes[ri][ci_]
            for mi_, mode in enumerate(MODES):
                uv = unit_values(Sv[(Sv["state"] == state) & (Sv["window"] == w) & (Sv["mode"] == mode)], "MI_hml", ["mode"])
                if uv.empty:
                    continue
                if KEYS:
                    cols = [mouse_colors.get(a, "grey") for a in uv["animal_id"]]
                    ax.scatter(mi_ + np.linspace(-0.15, 0.15, len(uv)), uv["value"], s=14, c=cols, alpha=0.8, zorder=2)
                sem = uv["value"].sem() if (KEYS and len(uv) > 1) else 0
                ax.errorbar(mi_, uv["value"].mean(), yerr=sem, color="black", marker="_", ms=14, capsize=3, lw=1.3, zorder=3)
                if len(tests):
                    r = tests[(tests["group"] == "ALL") & (tests["kind"] == "hml") & (tests["version"] == ver) & (tests["target"] == CHOICE_TARGET)
                              & (tests["window"] == w) & (tests["state"] == state) & (tests["mode"] == mode)]
                    if len(r):
                        ax.annotate(f"p={r['p_session'].iloc[0]:.2g}\nmice {int(r['n_mice_positive'].iloc[0])}/{int(r['n_mice'].iloc[0])}+",
                                    (mi_, 1.0), xycoords=("data", "axes fraction"), ha="center", va="top", fontsize=6)
            ax.axhline(0, color="grey", lw=0.7)
            ax.set_xticks(range(len(MODES)))
            ax.set_xticklabels(["balanced", "lick\nmatched", "lick+speed\nmatched"], fontsize=7)
            ax.set_xlim(-0.6, len(MODES) - 0.4)
            if ri == 0:
                ax.set_title(state, fontsize=10)
            if ci_ == 0:
                ax.set_ylabel(f"{w}\nchoice MI high - low (bits)")
    fig.suptitle(f"All cells [{ver}]: does the state effect on choice information survive matching the licking? (dots = {UNIT}s coloured by mouse; "
                 f"p across sessions)", y=1.0, fontsize=9)
    fig.tight_layout()
    save_fig(fig, figdir, f"2_all_cells_modes_{ver}")
    plt.close(fig)

# 3: behaviour in the response window per state tercile
bvars = [("p_lick", "P(lick)", None), ("lick_rate", "lick rate, lick trials (1/s)", "lick"), ("run_speed", "running speed, lick trials", "lick"),
         ("run_speed", "running speed, no-lick trials", "no_lick")]
fig, axes = plt.subplots(len(bvars), len(STATES), figsize=(3.2 * len(STATES), 2.6 * len(bvars)), squeeze=False, sharey="row")
for ri, (col, lab_, cls) in enumerate(bvars):
    for ci_, state in enumerate(STATES):
        ax = axes[ri][ci_]
        d = BEH[(BEH["state"] == state) & (BEH["trials"] == (cls or "lick"))]
        for w, ls_ in zip(WIN, ["-", "--"]):
            dw = d[d["window"] == w]
            if dw.empty:
                continue
            per = dw.groupby((KEYS or ["session_id"]) + ["tercile"])[col].mean().reset_index()
            m = per.groupby("tercile")[col].agg(["mean", "sem"])
            ax.errorbar(m.index, m["mean"], yerr=m["sem"].fillna(0) if KEYS else None, marker="o", ms=4, capsize=2, ls=ls_, color="black", label=w)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["low", "mid", "high"])
        if ri == 0:
            ax.set_title(state, fontsize=10)
        if ci_ == 0:
            ax.set_ylabel(lab_, fontsize=8)
        if ri == len(bvars) - 1:
            ax.set_xlabel("state tercile")
axes[0][0].legend(fontsize=7, frameon=False, title="response window")
fig.suptitle(f"Behaviour in the response window by tercile of the (pre-stimulus) state; {agg_note if KEYS else 'mean over sessions'}", y=1.0, fontsize=9)
fig.tight_layout()
save_fig(fig, figdir, "3_behaviour_by_state_tercile")
plt.close(fig)

print(f"\nWrote tables to {table_dir}, figures to {figdir}")
