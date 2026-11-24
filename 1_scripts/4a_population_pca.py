# -*- coding: utf-8 -*-
"""
4a_population_pca
===================
STEP 1 of the population-level ("section 4") analyses: how many dimensions does the SHARED (noise) variability of a
group's simultaneously recorded cells actually have, and what is its leading shared mode -- the same leave-pair-out
"global" signal that 2l/3m found dominates pairwise correlations, a state/arousal/locomotion signal, or something
tied to the stimulus/choice? This is purely descriptive/diagnostic and comes BEFORE any coding-axis or inter-area
("communication subspace") analysis (4b/4c), which need to know first whether there is anything low-dimensional to
find, and in what.

PER (session, group, window): the included (and --max-cells-capped, per group) cells' responses are residualised
against their own TARGET_STRATEGY bin mean (same convention/target as 3a's noise correlation -- project rule #4:
shared binning before bias correction) and z-scored per cell. Call this matrix R (trials x cells); only trials with
every one of the group's cells finite are kept.
    DIMENSIONALITY      plain PCA of R: participation_ratio = (sum lambda)^2 / sum(lambda^2) (effective number of
                         dimensions, 1 = one dominates, n_cells = all equal), frac_var_pc1, n_dims_90 (components for
                         90% of the variance) -- fast, always computed.
                         CROSS-VALIDATED factor-analysis dimensionality (the one that actually asks "is this
                         low-dimensional structure real, or does it just look that way on the same trials it was
                         found in"): k-fold CV, FactorAnalysis(n_components=k) for k = 0 (independent cells, the
                         per-cell-variance-only baseline) .. MAX_K, scored by held-out average log-likelihood.
                         cv_k_star = the k with the best held-out likelihood; cv_ll_gain = its advantage over k=0
                         (k_star == 0 or ll_gain <= 0 means: no reliable shared dimension in this group/window/session).
    LEADING MODE         PC1 (sign fixed so it correlates positively with the simple row-mean of R, a stable anchor)
                         correlated with: the LEAVE-GROUP-OUT global signal (mean z-scored residual of every OTHER
                         included cell of the session, same window -- generalises 3m/pair_state's leave-pair-out
                         "global" from a pair to a whole group), the STATE_WINDOW (pre-stim) pupil / running speed,
                         and, in the SAME window, choice. (NOT the stimulus target bin: R is already residualised
                         against it, so corr(any linear combination of R, target bin) = 0 by construction --
                         correlating PC1 with it would be a tautology, not a finding.)

Does NOT yet build coding axes / decoders (4b) or compare across groups (4c, inter-area communication subspace) --
this step only establishes the dimensionality and identity of each group's own shared variability, which those need
as a starting point (per the person's own proposed order: within-area dimensionality first).

Project-wide rules: --animals / --sessions / --aggregate / --max-cells (cap PER GROUP and session; a changed cap
recomputes the session) / --noise-only (0 < signal < 100 %) / --window LO HI (the pre-stim STATE_WINDOW the
pupil/speed/choice/target correlations use the SAME window's own target/choice, see below -- --window only moves
where pupil/speed come from). One session in RAM per worker (--n-jobs). Target strategy/binning = 3a's (TARGET_STRATEGY).

Reads: 1b psychometric_fits.csv, 2a celldata_combined.csv, 2b qc_per_cell.csv.
Outputs  2_pipeline/4a_population_pca/out/
    population_dimensionality_by_session.csv   one row per (session, group, window): n_cells, n_trials,
                                                participation_ratio, frac_var_pc1, n_dims_90, cv_k_star, cv_ll_gain,
                                                corr_pc1_global/_pupil/_speed/_choice
    population_dimensionality_tests.csv        one-sample tests of each corr_pc1_* measure across sessions/mice
                                                (BH q across the 8 groups, within window x measure)
    figures/ 1_dimensionality_by_group, 1b_cv_dimensionality_by_group, 2_mode_identity_correlations,
             3_dimensionality_vs_ncells
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sps
from sklearn.decomposition import FactorAnalysis

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
from infotheory.unit_plots import mouse_palette, group_panels
from infotheory.population import plot_metric_vs_population_size
from infotheory.stim_binning import SIMPLE_STRATEGIES
from infotheory import pair_tables as pt

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROTOCOL = "DN"
WINDOWS = {"pre_stim": (-30.0, -10.0), "stim": (0.0, 20.0), "reward": (25.0, 45.0)}
STATE_WINDOW = (-30.0, -10.0)                 # pre-stim window pupil/speed are read from
TARGET_STRATEGY = SIMPLE_STRATEGIES[0]        # residualisation + target_bin correlation target (matches 3a/3n)
MIN_CELLS_PCA = 5                             # cells needed in a (session, group, window) to run PCA/FA
MIN_OTHER_CELLS = 5                           # cells needed OUTSIDE the group for the leave-group-out global
MIN_TRIALS_PCA = 60                           # trials (after dropping incomplete rows) needed to run PCA/FA
MAX_K = 8                                     # upper cap on the cross-validated FA dimensionality search
CV_FOLDS = 5                                  # k-fold CV for the FA dimensionality search (fewer if few trials)
MIN_TRIALS_PER_FOLD = 20
VAR_FLOOR = 1e-3                              # floor on the k=0 (independent-cell) baseline variance
MIN_SESSIONS_TEST = 4
ALPHA = 0.05
RANDOM_STATE = 0
JOBLIB_VERBOSE = 5
CORR_MEASURES = ["corr_pc1_global", "corr_pc1_pupil", "corr_pc1_speed", "corr_pc1_choice"]
MEASURE_COLORS = {"global": "#1f77b4", "pupil": "#2ca02c", "speed": "#9467bd", "choice": "#d62728"}

parser = argparse.ArgumentParser()
add_scope_args(parser)
parser.add_argument("--window", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="pre-stim state window (pupil/speed) in cm rel. to stimStart (default STATE_WINDOW); outputs go to out/window_<lo>_<hi>/")
parser.add_argument("--noise-only", action="store_true", help="only the noise trials (0 < signal < 100 %); outputs go to out/noise_only/")
parser.add_argument("--max-k", type=int, default=MAX_K, help="upper cap on the cross-validated factor-analysis dimensionality search")
parser.add_argument("--cv-folds", type=int, default=CV_FOLDS)
args = parser.parse_args()
NOISE_ONLY = bool(args.noise_only)
if args.window:
    STATE_WINDOW = (float(args.window[0]), float(args.window[1]))
MAX_K, CV_FOLDS = args.max_k, args.cv_folds
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
print(f"{'NOISE TRIALS ONLY' if NOISE_ONLY else 'all engaged trials'}; state window {STATE_WINDOW}; response windows {list(WINDOWS)}; "
      f"target strategy {TARGET_STRATEGY}; cells per group and session: {MAX_CELLS or 'all'}; max-k {MAX_K}, cv-folds {CV_FOLDS}")

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
GLABEL = {group_key(g): SIX_GROUP_LABELS[g] for g in SIX_GROUP_ORDER}
GCOLOR = {group_key(g): SIX_GROUP_COLORS[g] for g in SIX_GROUP_ORDER}
stamps = {s: data_stamp(*selected[s]) for s in session_ids}
qc_mtime = int((activity_out / "qc_per_cell.csv").stat().st_mtime)
print(f"{len(session_ids)} sessions, {sum(len(v) for v in cells_by_session.values())} included cells")


# --------------------------------------------------------------------- #
# 2. Helpers
# --------------------------------------------------------------------- #
def zvec(v):
    v = np.asarray(v, float)
    sd = np.nanstd(v)
    return (v - np.nanmean(v)) / sd if sd > 0 else np.full_like(v, np.nan)


def zscore_cols(M):
    M = np.asarray(M, float)
    mean = np.nanmean(M, axis=0)
    std = np.nanstd(M, axis=0)
    std = np.where(std > 0, std, np.nan)
    return (M - mean) / std


def residualize(X, labels):
    """Subtract each column's within-label (target-bin) mean; NaN wherever the label is NaN."""
    X = np.asarray(X, float)
    labels = np.asarray(labels)
    out = np.full_like(X, np.nan)
    ok = pd.notna(labels)
    if not ok.any():
        return out
    lab_ok = labels[ok]
    for j in range(X.shape[1]):
        s = pd.Series(X[ok, j])
        out[ok, j] = (s - s.groupby(lab_ok).transform("mean")).to_numpy()
    return out


def participation_ratio(eigvals):
    ev = np.asarray(eigvals, float)
    ev = ev[ev > 0]
    if len(ev) == 0:
        return np.nan
    return float((ev.sum() ** 2) / (ev ** 2).sum())


def fa_cv_loglik(R, k_grid, n_folds, rng):
    """{k: mean held-out per-sample log-likelihood} over `n_folds` CV folds; k=0 = independent-Gaussian baseline
    (no shared structure). NaN for a k with no valid fold."""
    n = R.shape[0]
    idx = rng.permutation(n)
    folds = np.array_split(idx, n_folds)
    out = {k: [] for k in k_grid}
    for i in range(n_folds):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
        if len(train_idx) < 10 or len(test_idx) < 3:
            continue
        train, test = R[train_idx], R[test_idx]
        for k in k_grid:
            if k == 0:
                mu = train.mean(0)
                var = np.maximum(train.var(0, ddof=1), VAR_FLOOR)
                ll = -0.5 * (np.log(2 * np.pi * var) + (test - mu) ** 2 / var).sum(1)
                out[k].append(float(ll.mean()))
            else:
                try:
                    fa = FactorAnalysis(n_components=k, random_state=0).fit(train)
                    out[k].append(float(fa.score(test)))
                except Exception:
                    pass
    return {k: (float(np.mean(v)) if v else np.nan) for k, v in out.items()}


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 10 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return np.nan
    return float(np.corrcoef(a[m], b[m])[0, 1])


# --------------------------------------------------------------------- #
# 3. Per session (one in RAM at a time)
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
    st = tables.pop("_state").sort_values("trialNumber").reset_index(drop=True)
    cells = [c for c in cells if c in st.columns]
    if not cells:
        return None
    # analysed cells: all, or the first MAX_CELLS of each group in a fixed pseudo-random order (never the whole dataset at once)
    analysed = []
    for g in GROUP_KEYS:
        gc = [c for c in cells if cell_group.get(c) == g]
        if MAX_CELLS and len(gc) > MAX_CELLS:
            order = np.random.default_rng(stable_seed(sid, "cells", g, base=RANDOM_STATE)).permutation(len(gc))[:MAX_CELLS]
            gc = [gc[i] for i in sorted(order)]
        analysed += gc
    if not analysed:
        return None
    pupil_state = zvec(st["cur_pupil_area"].to_numpy()) if "cur_pupil_area" in st else None
    speed_state = zvec(st["cur_runspeed"].to_numpy()) if "cur_runspeed" in st else None
    st_tn = st["trialNumber"].to_numpy()

    rows = []
    for window, table in tables.items():
        if table.empty or "signal" not in table.columns:
            continue
        if NOISE_ONLY:
            table = table[(table["signal"] > 0) & (table["signal"] < 100)]
        table = pt.add_target_columns(table, [TARGET_STRATEGY]).reset_index(drop=True)
        cells_here = [c for c in analysed if c in table.columns]
        if len(cells_here) < MIN_CELLS_PCA:
            continue
        labels = table[pt.target_column(TARGET_STRATEGY)].to_numpy()
        X = table[cells_here].to_numpy(float)
        R = residualize(X, labels)
        Z = zscore_cols(R)
        Z0 = np.nan_to_num(Z)
        C = np.isfinite(Z).astype(float)
        Sall, Nall = Z0.sum(1), C.sum(1)
        choice = table["choice"].to_numpy(float)
        tnum = table["trialNumber"].to_numpy()
        srow = pd.Index(st_tn).get_indexer(tnum)
        has_state = srow >= 0

        def state_aligned(v):
            out_ = np.full(len(tnum), np.nan)
            if v is not None:
                out_[has_state] = v[srow[has_state]]
            return out_
        pupil_w, speed_w = state_aligned(pupil_state), state_aligned(speed_state)

        for g in GROUP_KEYS:
            in_g = np.array([cell_group.get(c) == g for c in cells_here])
            n_cells_g = int(in_g.sum())
            if n_cells_g < MIN_CELLS_PCA:
                continue
            row_ok = np.isfinite(Z[:, in_g]).all(axis=1)
            n_trials_g = int(row_ok.sum())
            if n_trials_g < MIN_TRIALS_PCA:
                continue
            Rg = Z[row_ok][:, in_g]
            Sg, Ng = Z0[:, in_g].sum(1), C[:, in_g].sum(1)
            n_other = Nall - Ng
            global_full = np.where(n_other >= MIN_OTHER_CELLS, (Sall - Sg) / np.maximum(n_other, 1), np.nan)
            global_g, choice_g = global_full[row_ok], choice[row_ok]
            pupil_g, speed_g = pupil_w[row_ok], speed_w[row_ok]

            Rc = Rg - Rg.mean(0)
            try:
                U, S_, Vt = np.linalg.svd(Rc, full_matrices=False)
            except np.linalg.LinAlgError:
                continue
            eig = (S_ ** 2) / max(n_trials_g - 1, 1)
            eig_sum = eig.sum()
            if eig_sum <= 0:
                continue
            pr = participation_ratio(eig)
            frac_pc1 = float(eig[0] / eig_sum)
            cumvar = np.cumsum(eig) / eig_sum
            n_dims_90 = int(np.searchsorted(cumvar, 0.9) + 1)
            pc1 = U[:, 0] * S_[0]
            row_mean = Rc.mean(1)
            if np.std(pc1) > 0 and np.std(row_mean) > 0 and np.corrcoef(pc1, row_mean)[0, 1] < 0:
                pc1 = -pc1

            k_grid = list(range(0, min(MAX_K, n_cells_g - 1, max(1, n_trials_g // MIN_TRIALS_PER_FOLD)) + 1))
            n_folds = min(CV_FOLDS, max(2, n_trials_g // MIN_TRIALS_PER_FOLD))
            k_star, ll_gain = np.nan, np.nan
            if len(k_grid) > 1:
                cv_ll = fa_cv_loglik(Rg, k_grid, n_folds, np.random.default_rng(stable_seed(sid, "fa", g, window, base=RANDOM_STATE)))
                finite_ll = {k: v for k, v in cv_ll.items() if np.isfinite(v)}
                if finite_ll:
                    k_star = max(finite_ll, key=finite_ll.get)
                    if np.isfinite(cv_ll.get(0, np.nan)):
                        ll_gain = finite_ll[k_star] - cv_ll[0]

            rows.append(dict(session_id=sid, animal_id=ses.animal_id, window=window, group=g,
                              n_cells=n_cells_g, n_trials=n_trials_g,
                              participation_ratio=pr, frac_var_pc1=frac_pc1, n_dims_90=n_dims_90,
                              cv_k_star=k_star, cv_ll_gain=ll_gain,
                              corr_pc1_global=corr(pc1, global_g), corr_pc1_pupil=corr(pc1, pupil_g),
                              corr_pc1_speed=corr(pc1, speed_g), corr_pc1_choice=corr(pc1, choice_g)))
    if not rows:
        return None
    print(f"  [{sid}] {len(analysed)} cells analysed, {len(rows)} (window, group) rows")
    return dict(summary=pd.DataFrame(rows))


results = run_sessions(
    session_ids, process_session, [PROTOCOL], paths.store, "population_dimensionality",
    key_params=dict(version="4a-v2", windows=WINDOWS, state_window=STATE_WINDOW, target=TARGET_STRATEGY,
                    min_cells_pca=MIN_CELLS_PCA, min_other=MIN_OTHER_CELLS, min_trials_pca=MIN_TRIALS_PCA,
                    max_k=MAX_K, cv_folds=CV_FOLDS, min_trials_per_fold=MIN_TRIALS_PER_FOLD, var_floor=VAR_FLOOR,
                    noise_only=NOISE_ONLY, max_cells=MAX_CELLS, qc_mtime=qc_mtime, fits_mtime=int(fits_path.stat().st_mtime)),
    load_kwargs=dict(load_behaviordata=True, load_videodata=True, load_celldata=True, load_calciumdata=True),
    cells_by_session=cells_by_session, base_seed=RANDOM_STATE, n_jobs=scope.n_jobs, recompute=args.recompute,
    verbose=JOBLIB_VERBOSE, extra_key_by_session=stamps, flatten=False)
results = [r for r in results if r is not None]
if not results:
    raise SystemExit("No session produced a (session, group, window) with enough cells/trials for PCA.")
S = pd.concat([r["summary"] for r in results], ignore_index=True)
S.to_csv(table_dir / "population_dimensionality_by_session.csv", index=False, float_format="%.5g")
print(f"\n{len(S)} (session, window, group) rows from {S['session_id'].nunique()} sessions")
WIN = list(WINDOWS)


# --------------------------------------------------------------------- #
# 4. Tests: is each corr_pc1_* measure different from 0 across sessions/mice?
# --------------------------------------------------------------------- #
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
for (grp, win), sub in S.groupby(["group", "window"]):
    for measure in CORR_MEASURES:
        n, mean, t, p, pw = one_sample(sub[measure])
        if n < MIN_SESSIONS_TEST:
            continue
        mice = sub.groupby("animal_id")[measure].mean()
        nm, mmean, mt, mp, _ = one_sample(mice.to_numpy())
        trows.append(dict(group=grp, window=win, measure=measure, n_sessions=n, mean=mean, t=t, p_session=p,
                           p_wilcoxon=pw, n_mice=nm, mean_mice=mmean, t_mice=mt, p_mice=mp,
                           n_mice_positive=int((mice > 0).sum()), n_mice_negative=int((mice < 0).sum())))
tests = pd.DataFrame(trows)
if len(tests):
    tests["q_session"] = np.nan
    for _, idx in tests.groupby(["window", "measure"]).groups.items():
        tests.loc[idx, "q_session"] = bh(tests.loc[idx, "p_session"].to_numpy())
    tests.to_csv(table_dir / "population_dimensionality_tests.csv", index=False, float_format="%.5g")
    print("\nCorrelation of PC1 with each candidate signal, sorted by p; q = BH within window x measure (across the 8 groups):")
    print(tests.sort_values("p_session").head(20)[["group", "window", "measure", "n_sessions", "mean", "p_session", "q_session",
                                                     "n_mice", "p_mice", "n_mice_positive"]].round(4).to_string(index=False))
    print(f"\n{int((tests['q_session'] < ALPHA).sum())} / {len(tests)} tests with q < {ALPHA}")

print("\nDimensionality summary (mean over sessions) by group x window:")
print(S.groupby(["group", "window"])[["n_cells", "n_trials", "participation_ratio", "frac_var_pc1", "n_dims_90",
                                       "cv_k_star", "cv_ll_gain"]].mean().round(3).to_string())

# --------------------------------------------------------------------- #
# 5. Figures
# --------------------------------------------------------------------- #
agg_note = {"pooled": "mean over sessions (no error bars)", "session": "mean +- SEM over sessions",
            "mouse": "mean +- SEM over mice"}[AGG]


def panels(df, name, value, ylabel, title, **kw):
    kw.setdefault("sharey", True)
    group_panels(df, "window", WIN, value, AGG, figdir, name, title, ylabel=ylabel, mouse_colors=mouse_colors,
                 ncols=len(WIN), panel_size=(4.6, 3.6), **kw)


panels(S, "1_dimensionality_by_group", "participation_ratio", "participation ratio (eff. # dims)",
       f"Dimensionality of shared variability -- {PROTOCOL}; {agg_note}")
panels(S, "1b_cv_dimensionality_by_group", "cv_k_star", "cross-validated FA dimensionality (k*)",
       f"Cross-validated factor-analysis dimensionality (0 = no reliable shared structure) -- {PROTOCOL}; {agg_note}")

long_rows = []
for measure in CORR_MEASURES:
    name = measure.replace("corr_pc1_", "")
    long_rows.append(S[["session_id", "animal_id", "window", "group", measure]].rename(columns={measure: "corr"}).assign(mode=name))
S_long = pd.concat(long_rows, ignore_index=True)
MODE_ORDER = [m.replace("corr_pc1_", "") for m in CORR_MEASURES]
panels(S_long, "2_mode_identity_correlations", "corr", "corr(PC1, signal)",
       f"What is the leading shared mode? -- {PROTOCOL}; {agg_note}", series_col="mode", series_order=MODE_ORDER,
       series_colors=MEASURE_COLORS, hline=0.0)

fig, axes = plt.subplots(1, len(WIN), figsize=(4.8 * len(WIN), 4.2), squeeze=False)
for ax, w in zip(axes[0], WIN):
    sub = S[S["window"] == w]
    plot_metric_vs_population_size(ax, sub, "n_cells", "participation_ratio", "group",
                                    "participation ratio", GCOLOR)
    ax.set_title(w)
    ax.legend(fontsize=6, frameon=False, ncol=2)
fig.suptitle(f"Dimensionality vs. population size, one point per (session, group) -- {PROTOCOL}", y=1.03)
fig.tight_layout()
save_fig(fig, figdir, "3_dimensionality_vs_ncells")
plt.close(fig)

print(f"\nWrote tables to {table_dir}, figures to {figdir}")
