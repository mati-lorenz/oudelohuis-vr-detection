# -*- coding: utf-8 -*-
"""
2m_state_modulation_of_perception
=================================
Does the PRE-STIMULUS state of the brain change how the stimulus is perceived? Two hypotheses
    FEEDBACK   the activity of the feedback-projecting PM-lab cells before the stimulus (SOURCE_GROUP) modulates the behavioral
               response to the stimulus (shifts / steepens the psychometric curve);
    GLOBAL     the global activity (mean of all other cells = the brain-wide gain 2l found) and arousal (pupil, running speed)
               modulate perception following the YERKES-DODSON law: performance is an INVERTED U of arousal (best at intermediate).
DN sessions, engaged trials (all stimulus levels so that the psychometric curve is anchored by 0 % / 100 %; set NOISE_TRIALS_ONLY=True
for the intermediate levels only). Behavior = lick response; stimulus = position on the session's fitted psychometric curve
(1b mu / sigma; stim_pct in [0, 1]; s = 2 * (stim_pct - 0.5) in [-1, 1] below).

STATES (one value per trial, z-scored within the session; mean trial response in the STATE_WINDOW (default pre-stim, -30..-10 cm)):
    PM_lab        mean of the SOURCE_GROUP cells (z-scored across trials first); sessions with >= MIN_SOURCE_CELLS only
    global        mean of all included cells OUTSIDE the source group
    pupil, speed  pupil area and running speed in the window (arousal references; need video)
    + any group in STATE_GROUPS (e.g. another area as a control feedback source)

ANALYSES
    1  psychometric curves by state tercile (low / mid / high, within session): does the curve shift or steepen?
    2  performance vs state (5 equal-occupancy bins within session): bias P(lick), discriminability P(lick | s > 0) - P(lick | s <= 0),
       accuracy P(lick == (s > 0)); an inverted U = Yerkes-Dodson
    3  per-session logistic regression (ridge, lambda = RIDGE):  lick ~ s + x + x2 + s*x + s*x2   (x = state, x2 = x^2 - 1)
           x         state shifts the response bias (more licking when the state is high)
           x2        inverted U of the bias (<0: licks most at intermediate states)
           s*x       state changes the sensitivity / slope (gain)
           s*x2      INVERTED U OF SENSITIVITY = Yerkes-Dodson for perception (<0 expected)
       Two versions: raw, and ADJUSTED (the other states of ADJUST_FOR enter as main effects and as interactions with s, so e.g. the
       PM-lab effect is the part not explained by global activity, pupil or running)
       Session-level tests (one-sample t / Wilcoxon of the session coefficients; mouse-level p; BH q over states x terms in each
       version) -> state_glm_tests.csv
    4  predicted sensitivity and bias as a function of the state, from the mean coefficients (the Yerkes-Dodson curve)
    6  CONFIRMATORY POOLED MODEL: one trial-level logistic regression over all sessions, session-specific intercepts and stimulus slopes,
       SHARED state terms (x, x2, s*x, s*x2; adjusted version adds the other states as covariates). Uncertainty: cluster-robust
       (session) Wald p and a MOUSE-cluster bootstrap (CI and p), plus leave-one-mouse-out estimates. PRIMARY (pre-specified, Holm):
       s*x2 for pupil and for global (Yerkes-Dodson on sensitivity) and s*x for PM_lab (feedback gain). -> state_pooled_model.csv, figure 6
    5  SHAPE-FREE tests on the binned performance of analysis 2 (per session, then across sessions): middle bins minus extremes
       (> 0 = inverted U), lowest bin minus the rest (a deficit at low arousal), highest bin minus the rest, linear trend

Project-wide rules
    * selection : --animals / --sessions restrict the sessions (a restricted run writes to out/selection/<tag>/)
    * aggregate : --aggregate pooled|session|mouse sets the unit of analyses 1 and 2 (pooled: all trials of all sessions,
                  states are z-scored within session so they can be pooled); 3 and 4 and the tests are always per session
    * memory    : one session in RAM per worker; per-session trial tables cached in store/ (no ledger)
    * cells     : 1b-included sessions; cells passing 2a (proximity, depth rule) and 2b QC.

Reads: 1b psychometric_fits.csv, 2a celldata_combined.csv, 2b qc_per_cell.csv.  Loads: raw behavior / video / calcium, one session at a time.

Outputs
-------
2_pipeline/2m_state_modulation_of_perception/
    out/    state_trials.csv                  one row per trial: session, signal, stim_pct, lick, the states
            state_glm_coefficients.csv        per session x state x version x term
            state_glm_tests.csv               across-session tests
            state_binned_performance[__<agg>].csv, state_psychometric[__<agg>].csv
            figures/1_psychometric_by_state, 2_performance_vs_state, 3_glm_coefficients_<version>, 4_sensitivity_bias_vs_state_<version>,
                    5_performance_shape_contrasts;  state_binned_shape_tests.csv (shape-free tests of the binned performance)
    store/  state_session_<hash>.pkl
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sps
from scipy.stats import rankdata

from infotheory.pipeline import get_pipeline_paths
from infotheory.session import discover_sessions
from infotheory.neural_encoding import load_cell_inclusion
from infotheory.continuous import merge_behavior_video, remove_pupil_outliers, zscore_per_session
from infotheory.plotting import set_style, save_fig, clear_figures
from infotheory.scope import add_scope_args, scope_from_args, select_sessions, scoped_dirs, data_stamp, unit_noun
from infotheory.session_jobs import run_sessions_units
from infotheory.unit_plots import mouse_palette
from infotheory.stim_binning import psychometric_pct
from infotheory import tensors as tn

warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom")
PROTOCOL = "DN"
NOISE_TRIALS_ONLY = False              # False: all engaged trials (0 % and 100 % anchor the psychometric curve); True: 0 < signal < 100 only
SOURCE_GROUP = "PM_lab"                # the feedback population (group key)
STATE_GROUPS = []                      # extra groups whose prestim activity is used as a state, e.g. ["PM_unl", "V1_lab"]
STATE_WINDOW = (-30.0, -10.0)          # cm rel. to stimStart (pre-stim window); (25, 45) would be the reward window etc.
S_PRE, S_POST = 40.0, 60.0
MIN_SOURCE_CELLS = 5
MIN_TRIALS = 80                        # trials per session
N_TERCILES = 3
N_PERF_BINS = 5                        # equal-occupancy state bins of analysis 2
PSY_BIN_EDGES = (0.0, 0.1, 0.3, 0.7, 0.9, 1.0001)    # stim_pct bins of analysis 1
RIDGE = 1.0                            # L2 on the non-intercept coefficients of the session logistic regressions
ADJUST_FOR = {"PM_lab": ["global", "pupil", "speed"], "global": ["pupil", "speed"], "pupil": ["speed"], "speed": []}   # default: other states
MIN_SESSIONS_TEST = 4
N_BOOT = 500                           # confirmatory pooled model: mouse-cluster bootstrap draws (resample MICE with replacement)
RIDGE_POOLED = 0.1                     # small L2 on the session slopes and the shared terms of the pooled model (intercepts free)
# PRE-SPECIFIED primary tests of the pooled model (family-wise: Holm over these). Everything else in the table is exploratory.
PRIMARY = [("pupil", "s*x2"), ("global", "s*x2"), (SOURCE_GROUP, "s*x")]
PRED_RANGE = 2.0                       # analysis 4: state from -PRED_RANGE to +PRED_RANGE SD
LICK_SMOOTH_S = 1.0
ALPHA = 0.05
RANDOM_STATE = 0
JOBLIB_VERBOSE = 5

parser = argparse.ArgumentParser()
add_scope_args(parser)
parser.add_argument("--window", type=float, nargs=2, metavar=("LO", "HI"), default=None,
                    help="state window in cm rel. to stimStart, e.g. --window 0 20 (default: STATE_WINDOW); outputs go to out/window_<lo>_<hi>/")
args = parser.parse_args()
if args.window:
    STATE_WINDOW = (float(args.window[0]), float(args.window[1]))
scope = scope_from_args(args)
AGG = scope.aggregate
UNIT = unit_noun(AGG)
UNITS = "mice" if UNIT == "mouse" else f"{UNIT}s"
AGG_SUFFIX = "" if AGG == "pooled" else f"__{AGG}"
TRIAL_NOTE = "noise trials (0 < signal < 100 %)" if NOISE_TRIALS_ONLY else "all engaged trials"
STATES = ["PM_lab" if SOURCE_GROUP == "PM_lab" else SOURCE_GROUP] + ["global", "pupil", "speed"] + list(STATE_GROUPS)
STATES = [SOURCE_GROUP, "global", "pupil", "speed"] + [g for g in STATE_GROUPS if g != SOURCE_GROUP]
ADJ = {s: list(ADJUST_FOR.get(s, [x for x in STATES if x != s and x in ("global", "pupil", "speed")])) for s in STATES}
ADJ[SOURCE_GROUP] = list(ADJUST_FOR.get("PM_lab", ["global", "pupil", "speed"]))

paths = get_pipeline_paths(__file__)
table_dir, figdir = scoped_dirs(paths, scope)
if args.window:
    _w = f"window_{STATE_WINDOW[0]:g}_{STATE_WINDOW[1]:g}"
    table_dir, figdir = table_dir / _w, figdir / _w
    table_dir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
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
animal_of = {sid: t[1] for sid, t in selected.items()}
session_ids = [s for s in selected if s in included_all]
if not session_ids:
    raise SystemExit("None of the selected sessions passed 1b_psychometric's inclusion criteria.")
mouse_colors = mouse_palette(a for _, a, _ in all_triples)
cell_inclusion = load_cell_inclusion(cell_dist_out, activity_out)
cell_group = dict(zip(cell_inclusion["cell_id"], cell_inclusion["group"]))
cells_by_session = {sid: sorted(sub.loc[sub["include"], "cell_id"]) for sid, sub in cell_inclusion.groupby("session_id")
                    if sid in session_ids}
session_ids = [s for s in session_ids if cells_by_session.get(s)]
if not session_ids:
    raise SystemExit("No included cells in the selected sessions.")
print(f"{len(session_ids)} sessions, {sum(len(v) for v in cells_by_session.values())} included cells")
stamps = {s: data_stamp(*selected[s]) for s in session_ids}
qc_mtime = int((activity_out / "qc_per_cell.csv").stat().st_mtime)


# --------------------------------------------------------------------- #
# 2. Per-session trial table (one session per worker, cached)
# --------------------------------------------------------------------- #
def zcols(M):
    M = np.asarray(M, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (M - np.nanmean(M, 0)) / np.nanstd(M, 0)


def process_session(ses, seed, new_cells, new_pairs, req_cells):
    sid = ses.session_id
    cell_ids = list(new_cells)
    td = ses.trialdata
    if (not cell_ids or ses.calciumdata is None or ses.ts_F is None or td is None or ses.behaviordata is None
            or "engaged" not in td.columns or "signal" not in td.columns or "lickResponse" not in td.columns):
        return None
    ms = mu_sigma_lookup.get(sid, {})
    if not ms or not np.isfinite(ms.get("mu", np.nan)) or not np.isfinite(ms.get("sigma", np.nan)):
        return None
    sel = td[td["engaged"] == 1]
    if NOISE_TRIALS_ONLY:
        sel = sel[(sel["signal"] > 0) & (sel["signal"] < 100)]
    sel = sel.sort_values("stimStart").reset_index(drop=True)
    if len(sel) < MIN_TRIALS:
        return None
    ts_F = np.asarray(ses.ts_F, float)
    dt = float(np.median(np.diff(ts_F)))
    A = ses.calciumdata.loc[:, cell_ids].to_numpy(dtype=float)
    zpos_F = tn.interp_to_imaging_frames(ses.behaviordata["ts"], ses.behaviordata["zpos"], ts_F)
    k, rel = tn.position_relative_to_stim(zpos_F, sel["stimStart"].to_numpy(float), S_PRE, S_POST)
    ok = tn.trial_frame_mask(ts_F, td, sel["trialNumber"]) & (k >= 0) & np.isfinite(A).all(1)
    if ok.sum() < 400:
        return None
    Y = tn.zscore_frames(np.nan_to_num(A), ok)
    good = np.isfinite(Y[ok]).all(0)
    if not good.any():
        return None
    ids = np.asarray(cell_ids)[good]
    Y = np.where(ok[:, None], np.nan_to_num(Y[:, good]), 0.0)
    R = zcols(tn.trial_window_means(Y, k, rel, ok, len(sel), {"w": STATE_WINDOW})["w"])
    grp = np.array([cell_group.get(c) for c in ids])
    out = pd.DataFrame({"session_id": sid, "animal_id": animal_of.get(sid), "trial": sel["trialNumber"].to_numpy(),
                        "signal": sel["signal"].to_numpy(float),
                        "stim_pct": psychometric_pct(sel["signal"], ms["mu"], ms["sigma"]),
                        "lick": sel["lickResponse"].to_numpy(float)})
    src = grp == SOURCE_GROUP
    out[SOURCE_GROUP] = np.nanmean(R[:, src], axis=1) if src.sum() >= MIN_SOURCE_CELLS else np.nan
    rest = ~src if src.sum() >= MIN_SOURCE_CELLS else np.ones(len(ids), bool)
    out["global"] = np.nanmean(R[:, rest], axis=1) if rest.sum() >= MIN_SOURCE_CELLS else np.nan
    for g in STATE_GROUPS:
        m = grp == g
        out[g] = np.nanmean(R[:, m], axis=1) if m.sum() >= MIN_SOURCE_CELLS else np.nan
    out["n_source_cells"] = int(src.sum())
    out["pupil"] = np.nan
    out["speed"] = np.nan
    if ses.videodata is not None:
        merged = zscore_per_session(remove_pupil_outliers(merge_behavior_video(ses.behaviordata, ses.videodata)))
        for col, name in (("pupil_area", "pupil"), ("runspeed", "speed")):
            if col in merged.columns:
                x = tn.behavior_to_imaging_frames(merged["ts"], merged[col], ts_F, "mean")
                fin = np.isfinite(x)
                num = tn.trial_window_means(np.where(fin, x, 0.0)[:, None], k, rel, ok, len(sel), {"w": STATE_WINDOW})["w"][:, 0]
                den = tn.trial_window_means(fin.astype(float)[:, None], k, rel, ok, len(sel), {"w": STATE_WINDOW})["w"][:, 0]
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[name] = zcols((num / np.where(den > 0, den, np.nan))[:, None])[:, 0]
    for s_ in [SOURCE_GROUP, "global"] + list(STATE_GROUPS):
        out[s_] = zcols(out[s_].to_numpy()[:, None])[:, 0]
    return dict(trials=out, cells_used=pd.DataFrame({"session_id": sid, "cell_id": ids}))


results = run_sessions_units(
    session_ids, process_session, [PROTOCOL], paths.store, "state_session",
    key_params=dict(version="2m-v1", noise=NOISE_TRIALS_ONLY, source=SOURCE_GROUP, groups=STATE_GROUPS, window=STATE_WINDOW,
                    s=(S_PRE, S_POST), min_source=MIN_SOURCE_CELLS, min_trials=MIN_TRIALS, qc_mtime=qc_mtime,
                    fits_mtime=int(fits_path.stat().st_mtime)),
    load_kwargs=dict(load_behaviordata=True, load_videodata=True, load_celldata=True, load_calciumdata=True),
    cells_by_session=cells_by_session, parts={"cells_used": "cell"}, groups_by_cell=cell_group,
    max_cells=scope.max_cells, base_seed=RANDOM_STATE, n_jobs=scope.n_jobs, recompute=args.recompute,
    verbose=JOBLIB_VERBOSE, extra_key_by_session=stamps)
results = [r for r in results if r and "trials" in r]
if not results:
    raise SystemExit("No results (sessions without video / lickResponse / psychometric fit or with too few trials).")
T = pd.concat([r["trials"] for r in results], ignore_index=True)
del results
T["s"] = 2 * (T["stim_pct"] - 0.5)
T = T[np.isfinite(T["s"]) & T["lick"].isin([0, 1])].reset_index(drop=True)
T.to_csv(table_dir / "state_trials.csv", index=False, float_format="%.4g")
states = [s for s in STATES if s in T.columns and T.groupby("session_id")[s].apply(lambda v: v.notna().sum() >= MIN_TRIALS).any()]
n_ses = {s: int(T.groupby("session_id")[s].apply(lambda v: v.notna().sum() >= MIN_TRIALS).sum()) for s in states}
print(f"{T['session_id'].nunique()} sessions, {len(T)} trials; sessions per state: {n_ses}")


# --------------------------------------------------------------------- #
# 3. Logistic regression per session (ridge IRLS)
# --------------------------------------------------------------------- #
def logit_ridge(X, y, lam, iters=60):
    """Ridge logistic regression (intercept unpenalised) by IRLS. X (n, p) without intercept."""
    Xd = np.column_stack([np.ones(len(X)), X])
    P = np.eye(Xd.shape[1]) * lam
    P[0, 0] = 0.0
    b = np.zeros(Xd.shape[1])
    for _ in range(iters):
        eta = np.clip(Xd @ b, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        w = np.maximum(mu * (1 - mu), 1e-6)
        z = eta + (y - mu) / w
        bn = np.linalg.solve(Xd.T @ (Xd * w[:, None]) + P, Xd.T @ (w * z))
        if np.max(np.abs(bn - b)) < 1e-7:
            b = bn
            break
        b = bn
    return b


TERMS = ["x", "x2", "s*x", "s*x2"]
TERM_LABEL = {"x": "x: bias shift", "x2": "x2: inverted U of bias", "s*x": "s*x: sensitivity shift (gain)",
              "s*x2": "s*x2: inverted U of sensitivity"}


def design(d, state, adjust):
    x = d[state].to_numpy()
    s = d["s"].to_numpy()
    cols = [s, x, x * x - 1, s * x, s * (x * x - 1)]
    names = ["s", "x", "x2", "s*x", "s*x2"]
    for a in (ADJ[state] if adjust else []):
        if a in d.columns and d[a].notna().sum() >= MIN_TRIALS:
            cols += [d[a].to_numpy(), s * d[a].to_numpy()]
            names += [a, "s*" + a]
    return np.column_stack(cols), names


coef_rows = []
for sid, d0 in T.groupby("session_id"):
    for st in states:
        for adjust in (False, True):
            if adjust and not ADJ[st]:
                continue
            need = [st] + [a for a in (ADJ[st] if adjust else []) if a in d0.columns and d0[a].notna().sum() >= MIN_TRIALS]
            d = d0.dropna(subset=need + ["s", "lick"])
            if len(d) < MIN_TRIALS or d["lick"].nunique() < 2:
                continue
            X, names = design(d, st, adjust)
            b = logit_ridge(X, d["lick"].to_numpy(float), RIDGE)
            row = dict(session_id=sid, animal_id=d["animal_id"].iloc[0], state=st, version="adjusted" if adjust else "raw",
                       n_trials=len(d), intercept=b[0])
            row.update({n: v for n, v in zip(names, b[1:])})
            coef_rows.append(row)
coef = pd.DataFrame(coef_rows)
coef.to_csv(table_dir / "state_glm_coefficients.csv", index=False, float_format="%.4g")


def bh(pv):
    pv = np.asarray(pv, float)
    q = np.full(len(pv), np.nan)
    ok = np.isfinite(pv)
    if ok.sum():
        o = np.argsort(pv[ok])
        ranked = pv[ok][o] * ok.sum() / (np.arange(ok.sum()) + 1)
        q[np.where(ok)[0][o]] = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    return q


trows = []
for (st, ver), d in coef.groupby(["state", "version"]):
    for t in ["s"] + TERMS:
        v = d[t].dropna()
        if len(v) < MIN_SESSIONS_TEST or v.std() == 0:
            continue
        mice = d.loc[v.index].groupby("animal_id")[t].mean()
        try:
            pw = sps.wilcoxon(v).pvalue
        except Exception:
            pw = np.nan
        trows.append(dict(state=st, version=ver, term=t, n_sessions=len(v), n_mice=len(mice), mean=v.mean(), sem=v.sem(),
                          frac_negative=float((v < 0).mean()), p_t=sps.ttest_1samp(v, 0).pvalue, p_wilcoxon=pw,
                          p_mice=sps.ttest_1samp(mice, 0).pvalue if len(mice) >= 3 and mice.std() > 0 else np.nan))
tests = pd.DataFrame(trows)
if len(tests):
    tests["q_t"] = np.nan
    for _, ix in tests[tests["term"].isin(TERMS)].groupby("version").groups.items():
        tests.loc[ix, "q_t"] = bh(tests.loc[ix, "p_t"])
    tests.to_csv(table_dir / "state_glm_tests.csv", index=False, float_format="%.4g")
    print("\nSession-level tests of the regression coefficients (mean over sessions; q = BH over states x the 4 state terms per version):")
    print(tests[tests["term"].isin(TERMS)].sort_values("p_t")[["state", "version", "term", "n_sessions", "n_mice", "mean", "p_t", "q_t",
                                                                 "p_mice"]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))

# --------------------------------------------------------------------- #
# 4. Binned descriptive analyses (analyses 1 and 2)
# --------------------------------------------------------------------- #
def qbin(x, nb):
    """Equal-occupancy bin index within one session (-1 where NaN)."""
    x = np.asarray(x, float)
    out = np.full(len(x), -1, dtype=int)
    m = np.isfinite(x)
    n = int(m.sum())
    if n >= nb * 5:
        out[m] = np.minimum((rankdata(x[m], method="ordinal") - 1) * nb // n, nb - 1)
    return out


def unit_col(d):
    return {"pooled": pd.Series("all", index=d.index), "session": d["session_id"], "mouse": d["animal_id"]}[AGG]


T["psy_bin"] = pd.cut(T["stim_pct"], PSY_BIN_EDGES, right=False, labels=False)
T["strong"] = (T["s"] > 0).astype(float)
T["correct"] = (T["lick"] == T["strong"]).astype(float)
T["unit"] = unit_col(T)
psy_rows, perf_rows = [], []
for st in states:
    d_all = T.dropna(subset=[st]).copy()
    d_all["tercile"] = d_all.groupby("session_id")[st].transform(lambda v: qbin(v, N_TERCILES))
    d_all["pbin"] = d_all.groupby("session_id")[st].transform(lambda v: qbin(v, N_PERF_BINS))
    d_all = d_all[(d_all["tercile"] >= 0) & (d_all["pbin"] >= 0)]
    g = d_all.groupby(["unit", "tercile", "psy_bin"]).agg(p_lick=("lick", "mean"), n=("lick", "size"), s=("s", "median")).reset_index()
    psy_rows.append(g.assign(state=st))
    for (u, pb), dd in d_all.groupby(["unit", "pbin"]):
        hi, lo = dd[dd["strong"] == 1]["lick"], dd[dd["strong"] == 0]["lick"]
        perf_rows.append(dict(state=st, unit=u, bin=pb, n=len(dd), bias=dd["lick"].mean(),
                              discriminability=(hi.mean() - lo.mean()) if len(hi) >= 3 and len(lo) >= 3 else np.nan,
                              accuracy=dd["correct"].mean(), x=dd[st].mean()))
psy = pd.concat(psy_rows, ignore_index=True)
perf = pd.DataFrame(perf_rows)
psy.to_csv(table_dir / f"state_psychometric{AGG_SUFFIX}.csv", index=False, float_format="%.4g")
perf.to_csv(table_dir / f"state_binned_performance{AGG_SUFFIX}.csv", index=False, float_format="%.4g")
unit_note = {"pooled": "all trials pooled (states z-scored within session); error = binomial SEM for P(lick), none otherwise",
             "session": "mean +- SEM over sessions", "mouse": "mean +- SEM over mice"}[AGG]


def mean_sem(g, col):
    m = g.groupby("bin" if "bin" in g.columns else "x")[col]
    return m.mean(), (m.sem() if AGG != "pooled" else m.mean() * 0)


# --------------------------------------------------------------------- #
# 5. Figures
# --------------------------------------------------------------------- #
LOHI = ["tab:blue", "grey", "tab:red"]
TER_LAB = ["low", "mid", "high"]
if N_TERCILES != 3:
    TER_LAB = [str(i) for i in range(N_TERCILES)]
ncol = len(states)
# 1: psychometric by tercile
fig, axes = plt.subplots(1, ncol, figsize=(3.6 * ncol, 3.4), squeeze=False, sharey=True)
for ax, st in zip(axes[0], states):
    g = psy[psy["state"] == st]
    for t in range(N_TERCILES):
        gt = g[g["tercile"] == t]
        if gt.empty:
            continue
        m = gt.groupby("psy_bin")["p_lick"].mean()
        e = gt.groupby("psy_bin")["p_lick"].sem() if AGG != "pooled" else np.sqrt(m * (1 - m) / gt.groupby("psy_bin")["n"].sum())
        xm = gt.groupby("psy_bin")["s"].median()
        ax.errorbar(xm, m, yerr=e, color=LOHI[t % 3] if N_TERCILES == 3 else None, marker="o", ms=3, lw=1.2, capsize=2,
                    label=f"{TER_LAB[t]} {st}")
    ax.set_title(f"state = {st}" + (f" (n={n_ses[st]} sessions)" if AGG != "pooled" else ""), fontsize=9)
    ax.set_xlabel("stimulus s (-1 weak ... +1 strong)", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
axes[0][0].set_ylabel("P(lick)", fontsize=9)
fig.suptitle(f"Psychometric curve by pre-stim state tercile (state window {STATE_WINDOW} cm) -- {PROTOCOL}, {TRIAL_NOTE}; {unit_note}", fontsize=9, y=1.0)
fig.tight_layout()
save_fig(fig, figdir, "1_psychometric_by_state")
plt.close(fig)

# 2: performance vs state bins
metrics = [("bias", "P(lick)  (bias)"), ("discriminability", "P(lick|strong) - P(lick|weak)"), ("accuracy", "accuracy")]
fig, axes = plt.subplots(len(metrics), ncol, figsize=(3.6 * ncol, 2.8 * len(metrics)), squeeze=False, sharex=True)
for j, st in enumerate(states):
    g = perf[perf["state"] == st]
    for i, (m_, lab) in enumerate(metrics):
        ax = axes[i][j]
        grp_ = g.groupby("bin")[m_]
        mu_ = grp_.mean()
        er = grp_.sem() if AGG != "pooled" else None
        ax.errorbar(mu_.index, mu_, yerr=er, color="black", marker="o", ms=4, lw=1.3, capsize=2)
        if AGG != "pooled":
            for u, gu in g.groupby("unit"):
                ax.plot(gu["bin"], gu[m_], color="grey", lw=0.5, alpha=0.4)
        ax.set_xticks(range(N_PERF_BINS))
        if i == 0:
            ax.set_title(f"state = {st}", fontsize=9)
        if j == 0:
            ax.set_ylabel(lab, fontsize=8)
        if i == len(metrics) - 1:
            ax.set_xlabel(f"{st} bin (0 = lowest ... {N_PERF_BINS - 1} = highest)", fontsize=8)
fig.suptitle(f"Performance vs pre-stim state ({N_PERF_BINS} equal-occupancy bins within session): an inverted U = Yerkes-Dodson -- "
             f"{PROTOCOL}, {TRIAL_NOTE}; thin = {UNIT}s; {unit_note}", fontsize=9, y=1.0)
fig.tight_layout()
save_fig(fig, figdir, "2_performance_vs_state")
plt.close(fig)

# 5: shape-free tests on the binned performance (session level): is performance highest at intermediate states (Yerkes-Dodson),
#    lower at the lowest / highest bin, or monotonic?  (the quadratic term of the regression cannot see a plateau or a one-sided deficit)
ps_rows = []
for st in states:
    d_all = T.dropna(subset=[st]).copy()
    d_all["pbin"] = d_all.groupby("session_id")[st].transform(lambda v: qbin(v, N_PERF_BINS))
    d_all = d_all[d_all["pbin"] >= 0]
    for sid, d in d_all.groupby("session_id"):
        m = {}
        for b_, dd in d.groupby("pbin"):
            hi, lo = dd[dd["strong"] == 1]["lick"], dd[dd["strong"] == 0]["lick"]
            m[b_] = dict(bias=dd["lick"].mean(), accuracy=dd["correct"].mean(),
                         discriminability=(hi.mean() - lo.mean()) if len(hi) >= 3 and len(lo) >= 3 else np.nan)
        if len(m) < N_PERF_BINS:
            continue
        for met in ("bias", "accuracy", "discriminability"):
            v = np.array([m[b_][met] for b_ in range(N_PERF_BINS)])
            if not np.isfinite(v).all():
                continue
            mid = v[1:-1].mean()
            ps_rows.append(dict(state=st, session_id=sid, animal_id=d["animal_id"].iloc[0], metric=met,
                                mid_minus_extremes=mid - 0.5 * (v[0] + v[-1]), lowest_minus_rest=v[0] - v[1:].mean(),
                                highest_minus_rest=v[-1] - v[:-1].mean(),
                                trend=np.polyfit(np.arange(N_PERF_BINS), v, 1)[0] * (N_PERF_BINS - 1)))
shape = pd.DataFrame(ps_rows)
CONTR = ["mid_minus_extremes", "lowest_minus_rest", "highest_minus_rest", "trend"]
CONTR_LAB = {"mid_minus_extremes": "middle bins - extremes\n(>0: inverted U)", "lowest_minus_rest": "lowest bin - rest\n(<0: deficit at low state)",
             "highest_minus_rest": "highest bin - rest\n(<0: deficit at high state)", "trend": "linear trend (highest - lowest)"}
shape_tests = []
for (st, met), d in shape.groupby(["state", "metric"]):
    for c_ in CONTR:
        v = d[c_].dropna()
        if len(v) < MIN_SESSIONS_TEST or v.std() == 0:
            continue
        mice = d.loc[v.index].groupby("animal_id")[c_].mean()
        try:
            pw = sps.wilcoxon(v).pvalue
        except Exception:
            pw = np.nan
        shape_tests.append(dict(state=st, metric=met, contrast=c_, n_sessions=len(v), n_mice=len(mice), mean=v.mean(), sem=v.sem(),
                                p_t=sps.ttest_1samp(v, 0).pvalue, p_wilcoxon=pw,
                                p_mice=sps.ttest_1samp(mice, 0).pvalue if len(mice) >= 3 and mice.std() > 0 else np.nan))
shape_tests = pd.DataFrame(shape_tests)
if len(shape_tests):
    shape_tests["q_t"] = np.nan
    for _, ix in shape_tests.groupby("metric").groups.items():
        shape_tests.loc[ix, "q_t"] = bh(shape_tests.loc[ix, "p_t"])
    shape_tests.to_csv(table_dir / "state_binned_shape_tests.csv", index=False, float_format="%.4g")
    shape.to_csv(table_dir / "state_binned_shape_per_session.csv", index=False, float_format="%.4g")
    print("\nShape of performance across state bins (session level; q = BH over states x 4 contrasts within metric). Smallest p:")
    print(shape_tests.sort_values("p_t").head(20)[["state", "metric", "contrast", "n_sessions", "n_mice", "mean", "p_t", "q_t", "p_mice"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    mets = ["accuracy", "discriminability", "bias"]
    fig, axes = plt.subplots(len(mets), len(CONTR), figsize=(3.4 * len(CONTR), 2.8 * len(mets)), squeeze=False)
    for i, met in enumerate(mets):
        for j, c_ in enumerate(CONTR):
            ax = axes[i][j]
            for k_, st in enumerate(states):
                v = shape[(shape["state"] == st) & (shape["metric"] == met)]
                ax.scatter(np.full(len(v), k_) + np.random.default_rng(0).uniform(-.18, .18, len(v)), v[c_],
                           c=[mouse_colors.get(a, "grey") for a in v["animal_id"]], s=12, zorder=3, edgecolor="none")
                if len(v) > 1:
                    ax.errorbar(k_, v[c_].mean(), yerr=v[c_].sem(), color="black", capsize=3, marker="_", ms=12, zorder=4)
                tt = shape_tests[(shape_tests["state"] == st) & (shape_tests["metric"] == met) & (shape_tests["contrast"] == c_)]
                if len(tt) and np.isfinite(tt["q_t"].iloc[0]) and tt["q_t"].iloc[0] < 0.1:
                    ax.text(k_, ax.get_ylim()[1], "*", ha="center", va="top", fontsize=12)
            ax.axhline(0, color="grey", lw=0.6, ls="--")
            ax.set_xticks(range(len(states)))
            ax.set_xticklabels(states, fontsize=7, rotation=30)
            if i == 0:
                ax.set_title(CONTR_LAB[c_], fontsize=8)
            if j == 0:
                ax.set_ylabel(met, fontsize=9)
    fig.suptitle(f"Shape of performance across {N_PERF_BINS} state bins, per session (dots = sessions, colour = mouse, bar = mean +- SEM, "
                 f"* = BH q < 0.1) -- {PROTOCOL}, {TRIAL_NOTE}", fontsize=9, y=1.0)
    fig.tight_layout()
    save_fig(fig, figdir, "5_performance_shape_contrasts")
    plt.close(fig)

# 6: confirmatory pooled model
def fe_logit(y, s_, Xs, labels, lam, iters=40):
    """Logistic regression with session-specific intercepts and stimulus slopes (fixed effects) and shared columns Xs.
    Returns the coefficients of the shared columns, the full coefficient vector, the design and mu."""
    u, code = np.unique(labels, return_inverse=True)
    D = np.zeros((len(y), len(u)))
    D[np.arange(len(y)), code] = 1.0
    X = np.column_stack([D, D * s_[:, None], Xs])
    pen = np.r_[np.zeros(len(u)), np.full(X.shape[1] - len(u), lam)]
    P = np.diag(pen)
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = np.clip(X @ b, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        w = np.maximum(mu * (1 - mu), 1e-6)
        z = eta + (y - mu) / w
        bn = np.linalg.solve(X.T @ (X * w[:, None]) + P + 1e-9 * np.eye(X.shape[1]), X.T @ (w * z))
        done = np.max(np.abs(bn - b)) < 1e-7
        b = bn
        if done:
            break
    return b[2 * len(u):], b, X, mu, P, code


pm_rows = []
rng_b = np.random.default_rng(RANDOM_STATE)
for st in states:
    for adjust in (False, True):
        if adjust and not ADJ[st]:
            continue
        need = [st] + [a for a in (ADJ[st] if adjust else []) if a in T.columns and T[a].notna().sum() >= MIN_TRIALS]
        d = T.dropna(subset=need + ["s", "lick"]).reset_index(drop=True)
        d = d[d.groupby("session_id")["lick"].transform("size") >= MIN_TRIALS].reset_index(drop=True)
        if d["session_id"].nunique() < MIN_SESSIONS_TEST:
            continue
        x = d[st].to_numpy()
        sv = d["s"].to_numpy()
        cols = [x, x * x - 1, sv * x, sv * (x * x - 1)]
        names = list(TERMS)
        for a in need[1:]:
            cols += [d[a].to_numpy(), sv * d[a].to_numpy()]
            names += [a, "s*" + a]
        Xs = np.column_stack(cols)
        y = d["lick"].to_numpy(float)
        lab = d["session_id"].to_numpy()
        est, b_all, X, mu, P, code = fe_logit(y, sv, Xs, lab, RIDGE_POOLED)
        # cluster-robust (session) covariance of the shared terms
        H = X.T @ (X * np.maximum(mu * (1 - mu), 1e-6)[:, None]) + P + 1e-9 * np.eye(X.shape[1])
        Hinv = np.linalg.inv(H)
        sc = X * (y - mu)[:, None]
        G = int(code.max()) + 1
        S = np.zeros((G, X.shape[1]))
        np.add.at(S, code, sc)
        meat = S.T @ S * G / max(G - 1, 1)
        V = Hinv @ meat @ Hinv
        k0 = 2 * G
        se = np.sqrt(np.diag(V)[k0:k0 + len(names)])
        # mouse-cluster bootstrap and leave-one-mouse-out
        mice = d["animal_id"].to_numpy()
        um = np.unique(mice)
        idx_by = {m: np.where(mice == m)[0] for m in um}
        boots = []
        for _ in range(N_BOOT):
            draw = rng_b.choice(um, len(um), replace=True)
            ii = np.concatenate([idx_by[m] for m in draw])
            labs = np.concatenate([np.char.add(lab[idx_by[m]].astype(str), f"#{c}") for c, m in enumerate(draw)])
            try:
                boots.append(fe_logit(y[ii], sv[ii], Xs[ii], labs, RIDGE_POOLED)[0][:len(names)])
            except np.linalg.LinAlgError:
                continue
        boots = np.array(boots)
        loo = np.array([fe_logit(y[mice != m], sv[mice != m], Xs[mice != m], lab[mice != m], RIDGE_POOLED)[0][:len(names)] for m in um]) \
            if len(um) > 2 else np.full((1, len(names)), np.nan)
        for j, t in enumerate(names):
            bj = boots[:, j]
            pb = 2 * min((1 + (bj >= 0).sum()) / (1 + len(bj)), (1 + (bj <= 0).sum()) / (1 + len(bj)))
            pm_rows.append(dict(state=st, version="adjusted" if adjust else "raw", term=t, estimate=est[j], se_session_cluster=se[j],
                                p_session_cluster=2 * (1 - sps.norm.cdf(abs(est[j] / se[j]))) if se[j] > 0 else np.nan,
                                ci_low=np.percentile(bj, 2.5), ci_high=np.percentile(bj, 97.5), p_mouse_bootstrap=min(pb, 1.0),
                                loo_min=np.nanmin(loo[:, j]), loo_max=np.nanmax(loo[:, j]), n_trials=len(d),
                                n_sessions=int(d["session_id"].nunique()), n_mice=len(um)))
pooled = pd.DataFrame(pm_rows)
if len(pooled):
    pooled["primary"] = [(r.state, r.term) in PRIMARY and r.version == "adjusted" for r in pooled.itertuples()]
    pooled["holm_p"] = np.nan
    prim = pooled[pooled["primary"]]
    if len(prim):
        o = prim.sort_values("p_mouse_bootstrap")
        m_ = len(o)
        holm = np.maximum.accumulate([min(1.0, (m_ - i) * p_) for i, p_ in enumerate(o["p_mouse_bootstrap"])])
        pooled.loc[o.index, "holm_p"] = holm
    pooled.to_csv(table_dir / "state_pooled_model.csv", index=False, float_format="%.4g")
    print(f"\nCONFIRMATORY pooled model (session-specific intercepts and slopes; shared state terms; {N_BOOT} mouse-cluster bootstrap draws).")
    nm = int(pooled["n_mice"].min())
    print(f"NOTE: the bootstrap resamples mice; with {nm}-{int(pooled['n_mice'].max())} mice there are only a few hundred to a few thousand distinct draws,"
          " so p_mouse_bootstrap is coarse and anti-conservative; read the CI and the leave-one-mouse-out range (loo_min..loo_max: a real "
          "effect keeps its sign when any mouse is left out) first.")
    print("Primary tests, adjusted version (pre-specified, Holm over the primary family):")
    print(pooled[pooled["primary"]][["state", "term", "estimate", "ci_low", "ci_high", "p_mouse_bootstrap", "p_session_cluster", "holm_p",
                                     "loo_min", "loo_max", "n_sessions", "n_mice"]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    print("\nAll state terms (exploratory):")
    print(pooled[pooled["term"].isin(TERMS)][["state", "version", "term", "estimate", "ci_low", "ci_high", "p_mouse_bootstrap",
                                                 "p_session_cluster"]].to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    fig, axes = plt.subplots(1, len(TERMS), figsize=(3.6 * len(TERMS), 3.4), squeeze=False)
    for ax, t in zip(axes[0], TERMS):
        for i, st in enumerate(states):
            for off, ver, col in [(-0.15, "raw", "grey"), (0.15, "adjusted", "tab:red")]:
                r = pooled[(pooled["state"] == st) & (pooled["term"] == t) & (pooled["version"] == ver)]
                if r.empty:
                    continue
                r = r.iloc[0]
                ax.errorbar(i + off, r["estimate"], yerr=[[r["estimate"] - r["ci_low"]], [r["ci_high"] - r["estimate"]]], color=col,
                            marker="o", ms=4, capsize=3, label=ver if i == 0 else None)
                if (st, t) in PRIMARY and ver == "adjusted":
                    ax.text(i + off, ax.get_ylim()[1], "P", ha="center", va="top", fontsize=9, color="tab:red")
        ax.axhline(0, color="grey", lw=0.6, ls="--")
        ax.set_xticks(range(len(states)))
        ax.set_xticklabels(states, fontsize=8, rotation=30)
        ax.set_title(TERM_LABEL[t], fontsize=8)
    axes[0][0].set_ylabel("shared logistic coefficient", fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.suptitle(f"Pooled confirmatory model: shared state coefficients with 95 % mouse-cluster bootstrap CIs (P = pre-specified primary test) -- "
                 f"{PROTOCOL}, {TRIAL_NOTE}", fontsize=9, y=1.02)
    fig.tight_layout()
    save_fig(fig, figdir, "6_pooled_model_coefficients")
    plt.close(fig)

# 3: coefficients; 4: predicted sensitivity / bias
for ver in ["raw", "adjusted"]:
    c = coef[coef["version"] == ver]
    if c.empty:
        continue
    sts = [s for s in states if (c["state"] == s).any()]
    fig, axes = plt.subplots(1, len(TERMS), figsize=(3.4 * len(TERMS), 3.2), squeeze=False, sharey=False)
    for ax, t in zip(axes[0], TERMS):
        for i, st in enumerate(sts):
            v = c[c["state"] == st]
            vals = v[t].dropna()
            ax.scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-.18, .18, len(v)), v[t],
                       c=[mouse_colors.get(a, "grey") for a in v["animal_id"]], s=14, zorder=3, edgecolor="none")
            if len(vals) > 1:
                ax.errorbar(i, vals.mean(), yerr=vals.sem(), color="black", capsize=3, marker="_", ms=14, zorder=4)
            tt = tests[(tests["state"] == st) & (tests["version"] == ver) & (tests["term"] == t)]
            if len(tt) and np.isfinite(tt["q_t"].iloc[0]) and tt["q_t"].iloc[0] < 0.1:
                ax.text(i, ax.get_ylim()[1], "*", ha="center", va="top", fontsize=12)
        ax.axhline(0, color="grey", lw=0.6, ls="--")
        ax.set_xticks(range(len(sts)))
        ax.set_xticklabels(sts, fontsize=8, rotation=30)
        ax.set_title(TERM_LABEL[t], fontsize=8)
    axes[0][0].set_ylabel("logistic coefficient", fontsize=9)
    fig.suptitle(f"Session-level regression coefficients ({ver}); dots = sessions (colour = mouse), bar = mean +- SEM, * = BH q < 0.1 "
                 f"(one-sample t over sessions); {PROTOCOL}, {TRIAL_NOTE}", fontsize=9, y=1.02)
    fig.tight_layout()
    save_fig(fig, figdir, f"3_glm_coefficients_{ver}")
    plt.close(fig)

    xg = np.linspace(-PRED_RANGE, PRED_RANGE, 41)
    fig, axes = plt.subplots(2, len(sts), figsize=(3.6 * len(sts), 5.6), squeeze=False, sharex=True)
    for j, st in enumerate(sts):
        v = c[c["state"] == st]
        sens = np.array([r["s"] + r["s*x"] * xg + r["s*x2"] * (xg ** 2 - 1) for _, r in v.iterrows()])
        bias = np.array([r["intercept"] + r["x"] * xg + r["x2"] * (xg ** 2 - 1) for _, r in v.iterrows()])
        for i, (M, lab) in enumerate([(sens, "sensitivity (slope of the logit on s)"), (bias, "bias (logit P(lick) at s = 0)")]):
            ax = axes[i][j]
            for r_ in M:
                ax.plot(xg, r_, color="grey", lw=0.4, alpha=0.4)
            mm, se = M.mean(0), M.std(0, ddof=1) / np.sqrt(len(M)) if len(M) > 1 else np.zeros(len(xg))
            ax.plot(xg, mm, color="black", lw=2)
            ax.fill_between(xg, mm - se, mm + se, color="black", alpha=0.2, lw=0)
            if i == 0:
                ax.set_title(f"state = {st} (n={len(v)} sessions)", fontsize=9)
            if j == 0:
                ax.set_ylabel(lab, fontsize=8)
            if i == 1:
                ax.set_xlabel(f"{st} (SD, within session)", fontsize=8)
    fig.suptitle(f"Predicted sensitivity and bias vs pre-stim state from the session regressions ({ver}) -- an inverted U of sensitivity = "
                 f"Yerkes-Dodson; thin = sessions, black = mean +- SEM; {PROTOCOL}, {TRIAL_NOTE}", fontsize=9, y=1.0)
    fig.tight_layout()
    save_fig(fig, figdir, f"4_sensitivity_bias_vs_state_{ver}")
    plt.close(fig)

print(f"\nWrote tables to {table_dir}, figures to {figdir}")
