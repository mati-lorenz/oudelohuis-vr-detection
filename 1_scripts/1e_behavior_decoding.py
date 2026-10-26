# -*- coding: utf-8 -*-
"""
1e_behavior_decoding
=====================
Companion of 1d_performance_predictors (1d is untouched): how well does the animal's BEHAVIOR, window by window, tell the
STIMULUS or the CHOICE? One decoder is trained per position bin / time bin (and per protocol DM, DP, DN) and the plot is its
cross-validated accuracy against position / time.

DECODERS (infotheory/decoder.py, numpy / scipy only; --decoders selects, default all three, compared in figure 0):
    lda   linear discriminant analysis (shared covariance, ridge 1e-2, equal priors)
    logr  multinomial logistic regression (L2, C = 1, balanced class weights; matches sklearn's LogisticRegression)
    svm   linear SVM (squared hinge, one-vs-rest, C = 1, balanced class weights; a 2-class problem equals sklearn's LinearSVC)
All score BALANCED accuracy (mean recall over classes), so chance = 1 / n_classes whatever the class imbalance; stratified 5-fold CV
repeated 3 times. The chance level is not assumed: the same CV is run on N_PERM label permutations (grey line + band = mean +- SD
of the null), per decoder. LOGR and SVM are about 5-10x slower than LDA (--n-perm lowers the cost).
BINS (set at the top of this script, POSITION_* / TIME_*; they do NOT come from 3i-3l): fixed = POSITION_BIN cm / TIME_BIN s bins,
sliding = POSITION_WINDOW cm moved by POSITION_STEP (TIME_* likewise).
FEATURES: the trial mean of each behavioral variable in the window (runspeed, pupil_area, motionenergy, lick; 1d's variables, the
engaged trials, each sample inside its own trial), z-scored within session and window. A trial needs finite values of all the
features of the set. Lick: mean of the 0/1 lick channel over the window's samples (proportional to the lick rate).
TARGETS: the stimulus strategies of stim_binning (DEFAULT_STRATEGIES; those defined on the psychometric curve exist only for
sessions with a 1b fit, i.e. mainly DN) and `choice` (lick / no lick).
FEATURE SETS (STAGE 1 -> 2):
    single          each behavioral variable alone
    cumulative      variables added one at a time in ADD_ORDER (editable below; default runspeed, lick, pupil_area, motionenergy)
                    -> "runspeed+lick", "runspeed+lick+pupil_area", all four
    leave_one_out   all four minus one variable: accuracy(all) - accuracy(all but X) = what X adds to the rest
WINDOWS: position (cm rel. to stimulus onset) and time (s rel. to the stimulus onset time); binnings `windows` (3a-3c's pre_stim /
stim / reward, position only), `fixed`, `sliding` (as 3i-3l, infotheory/behavior_pairs.py).
READING IT: before the stimulus a behavior-only decoder of the stimulus should sit at chance (that is a check); after onset the
animal's own running / licking carries the choice, and through it the stimulus. Choice decoded from licking in the response window is
nearly circular (the lick IS the choice): look at the pre-response windows for what behavior predicts before the animal acts.

Project-wide rules (infotheory/project_config.py) -- what applies here
    * selection : --animals / --sessions restrict the sessions; a restricted run writes to out/selection/<tag>/.
    * aggregate : --aggregate pooled|session|mouse = the unit a decoder is trained on. pooled: one decoder per protocol on all its
                  trials (features z-scored per session first; labels permuted within session for the null), no error bars;
                  session / mouse: one decoder per session / mouse, curves = mean +- SEM over units. Always per protocol.
    * memory    : stage 1 (one session per worker) reduces each session to a small trial x window x behavior table, cached in
                  store/; stage 2 decodes from those tables only (a few MB for the whole dataset).
    * binning   : stimulus bins from the shared strategies, trial filters MIN_TRIALS_PER_TARGET_BIN / MIN_TRIALS_ANALYSIS (a window
                  needs >= MIN_TRIALS_ANALYSIS trials and every class >= MIN_TRIALS_PER_TARGET_BIN), as 1d / 2c / 3a-3l.

Outputs
-------
2_pipeline/1e_behavior_decoding/
    out/    behavior_decoding[__<aggregate>].csv       per (protocol, [unit], decoder, axis, binning, window, s_rel, target, feature set):
                                                       accuracy, null mean / SD, n_trials, n_classes, chance
            behavior_decoding_curves[__<aggregate>].csv  mean and SEM over units per (protocol, ..., feature set)
            figures/<axis>/<binning>/<target>/0_decoder_comparison.png  (one row per feature set, lines = decoders)
            figures/<axis>/<binning>/<target>/<decoder>/1_single_variables.png, 2_cumulative_sets.png, 3_leave_one_out.png
            figures/position/windows/<target>_by_window_<decoder>.png, <target>_decoders_by_window.png
    out/selection/<tag>/...                            the same for a restricted run
    store/  behavior_windows_<hash>.pkl                per-session window tables
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from infotheory import project_config as cfg
from infotheory.pipeline import get_pipeline_paths
from infotheory.session import discover_sessions, PROTOCOLS
from infotheory.plotting import set_style, save_fig, clear_figures, PROTOCOL_COLORS
from infotheory.scope import add_scope_args, scope_from_args, select_sessions, scoped_dirs, data_stamp, unit_noun
from infotheory.session_jobs import run_sessions
from infotheory.unit_stats import stable_seed
from infotheory.decoder import decode, DECODERS
from infotheory.stim_binning import (STIM_BINNING_STRATEGIES, DEFAULT_STRATEGIES, CHOICE_TARGET, psychometric_pct, bin_stimulus,
                                     target_signature, target_bin_names)
from infotheory.behavior_pairs import BehaviorWindows, BEHAVIOR_VARS, AXES_DEF, WINDOWS, BINNINGS

ADD_ORDER = ["runspeed", "lick", "pupil_area", "motionenergy"]    # order in which variables are added (edit freely)
STIM_STRATEGIES = list(DEFAULT_STRATEGIES)
N_FOLDS, N_REPEATS, N_PERM = 5, 3, 20
# --- bins of this script (position in cm, time in s); edit here -------------------------------------------------------
POSITION_BIN, POSITION_WINDOW, POSITION_STEP = 10.0, 20.0, 5.0     # fixed bin; sliding window and its step
TIME_BIN, TIME_WINDOW, TIME_STEP = 1.0, 2.0, 0.5
BIN_DEFAULTS = {"position": (POSITION_BIN, POSITION_WINDOW, POSITION_STEP), "time": (TIME_BIN, TIME_WINDOW, TIME_STEP)}
DECODER_COLORS = {"lda": "tab:blue", "logr": "tab:orange", "svm": "tab:green"}
RANDOM_STATE = 0
JOBLIB_VERBOSE = 5
SHARE_Y = True
VAR_COLORS = {"runspeed": "tab:blue", "pupil_area": "tab:orange", "motionenergy": "tab:green", "lick": "tab:red"}

parser = argparse.ArgumentParser()
add_scope_args(parser)
parser.add_argument("--strategies", nargs="+", default=None, choices=sorted(STIM_BINNING_STRATEGIES))
parser.add_argument("--axes", nargs="+", default=list(AXES_DEF), choices=list(AXES_DEF))
parser.add_argument("--binnings", nargs="+", default=BINNINGS, choices=BINNINGS)
parser.add_argument("--n-perm", type=int, default=N_PERM, help="label permutations for the null (0 = none)")
parser.add_argument("--decoders", nargs="+", default=DECODERS, choices=DECODERS)
for ax_ in AXES_DEF:
    parser.add_argument(f"--{ax_}-bin", type=float, default=BIN_DEFAULTS[ax_][0])
    parser.add_argument(f"--{ax_}-window", type=float, default=BIN_DEFAULTS[ax_][1])
    parser.add_argument(f"--{ax_}-step", type=float, default=BIN_DEFAULTS[ax_][2])
args = parser.parse_args()
axes_def = {k: dict(v) for k, v in AXES_DEF.items()}
for ax_ in axes_def:
    axes_def[ax_]["binsize"] = getattr(args, f"{ax_}_bin")
    axes_def[ax_]["window"] = getattr(args, f"{ax_}_window")
    axes_def[ax_]["step"] = getattr(args, f"{ax_}_step")
    r_ = axes_def[ax_]["window"] / axes_def[ax_]["step"]
    if abs(r_ - round(r_)) > 1e-9:
        raise SystemExit(f"{ax_}: the sliding window must be a multiple of the step")
scope = scope_from_args(args)
AGG = scope.aggregate
UNIT = unit_noun(AGG)
AGG_SUFFIX = "" if AGG == "pooled" else f"__{AGG}"
agg_note = {"pooled": "one decoder per protocol (pooled trials)", "session": "mean +- SEM over sessions",
            "mouse": "mean +- SEM over mice"}[AGG]
STRATS = args.strategies or STIM_STRATEGIES
TARGETS = STRATS + [CHOICE_TARGET]
AXES, BINS_RUN, DEC_RUN = list(args.axes), list(args.binnings), list(args.decoders)
assert set(ADD_ORDER) == set(BEHAVIOR_VARS), "ADD_ORDER must list every behavioral variable once"

# feature sets: name -> (kind, variables)
SETS = {}
for v in ADD_ORDER:
    SETS[v] = ("single", [v])
for n in range(2, len(ADD_ORDER) + 1):
    SETS["+".join(ADD_ORDER[:n])] = ("cumulative", ADD_ORDER[:n])
ALL_NAME = "+".join(ADD_ORDER)
for v in ADD_ORDER:
    SETS[f"all_but_{v}"] = ("leave_one_out", [u for u in ADD_ORDER if u != v])
CUM_NAMES = [k for k, (kind, _) in SETS.items() if kind == "cumulative"]

paths = get_pipeline_paths(__file__)
table_dir, figdir = scoped_dirs(paths, scope)
clear_figures(figdir)
for sub in sorted({q.parent for q in figdir.rglob("*.png")}):
    clear_figures(sub)
set_style()
print(scope.describe())
print(f"Decoders {DEC_RUN}: balanced accuracy, {N_FOLDS}-fold CV x {N_REPEATS}, {args.n_perm} permutations; targets {TARGETS}; "
      f"axes {AXES}; binnings {BINS_RUN}")

# --------------------------------------------------------------------- #
# 1. Included sessions (1b) in the selection
# --------------------------------------------------------------------- #
fits_path = paths.out_from("1b_psychometric") / "psychometric_fits.csv"
if not fits_path.exists():
    raise SystemExit(f"Run 1b_psychometric.py first -- expected {fits_path}")
fits_df = pd.read_csv(fits_path)
all_triples = discover_sessions(PROTOCOLS)
selected_ids = {f"{a}_{d}" for _, a, d in select_sessions(all_triples, scope)}
session_ids = [s for s in fits_df[fits_df["included"]]["session_id"].tolist() if s in selected_ids]
if not session_ids:
    raise SystemExit("No selected session passed 1b_psychometric's inclusion criteria.")
triple_of = {f"{a}_{d}": (p, a, d) for p, a, d in all_triples}
mu_sigma = fits_df.set_index("session_id")[["mu", "sigma"]].to_dict("index")
print(f"{len(session_ids)} sessions")


# --------------------------------------------------------------------- #
# 2. Stage 1: per session, trial x window x behavior tables (small), cached
# --------------------------------------------------------------------- #
def process_session(ses, seed):
    bw = BehaviorWindows(ses, [], BEHAVIOR_VARS, axes_def, AXES, BINS_RUN, imaging=False)
    if not bw.ok:
        return None
    tr = bw.trials
    ms = mu_sigma.get(ses.session_id, {})
    pct = psychometric_pct(tr["signal"], ms.get("mu"), ms.get("sigma"))
    labels = {}
    for s in STRATS:
        labels[s] = np.asarray(bin_stimulus(tr["signal"], pct, STIM_BINNING_STRATEGIES[s]), float)
    labels[CHOICE_TARGET] = tr["lickResponse"].to_numpy(float)
    wins = []
    for w in bw.windows():
        tmb = np.full((bw.nt, len(BEHAVIOR_VARS)), np.nan, dtype=np.float32)
        for k, b in enumerate(bw.behaviors):
            tmb[:, BEHAVIOR_VARS.index(b)] = w["TMB"][:, k]
        tmb[w["cnt"] == 0] = np.nan
        wins.append(dict(axis=w["axis"], binning=w["binning"], window=w["window"], s_rel=round(float(w["s_rel"]), 6), X=tmb))
    print(f"  [{ses.session_id}] {bw.nt} engaged trials, {len(wins)} windows, variables {bw.behaviors}")
    return dict(session_id=ses.session_id, animal_id=ses.animal_id, protocol=ses.protocol, labels=labels, windows=wins)


tables = run_sessions(
    session_ids, process_session, list(PROTOCOLS), paths.store, "behavior_windows",
    key_params=dict(axes_def={a: axes_def[a] for a in AXES}, axes=AXES, binnings=BINS_RUN, strategies=target_signature(STRATS),
                    behaviors=BEHAVIOR_VARS, windows=WINDOWS, version="1e-v1"),
    load_kwargs=dict(load_behaviordata=True, load_videodata=True), n_jobs=scope.n_jobs, recompute=args.recompute,
    verbose=JOBLIB_VERBOSE, extra_key_by_session={s: f"{data_stamp(*triple_of[s])}|{mu_sigma.get(s)}" for s in session_ids},
    flatten=False)
tables = [t for t in tables if t is not None]
if not tables:
    raise SystemExit("No session had usable behavior data.")
print(f"{len(tables)} sessions with tables")


# --------------------------------------------------------------------- #
# 3. Stage 2: one decoder per (protocol, unit, window, target, feature set)
# --------------------------------------------------------------------- #
def zscore_cols(X):
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    return (X - mu) / np.where(sd > 1e-12, sd, np.nan)


def decode_unit(protocol, unit, animal, sess_tables, n_perm, seed):
    rng = np.random.default_rng(seed)
    keys = {}
    for t in sess_tables:
        for w in t["windows"]:
            keys.setdefault((w["axis"], w["binning"], w["window"], w["s_rel"]), []).append((t, w))
    rows = []
    for (axis, binning, window, s_rel), items in keys.items():
        X = np.vstack([zscore_cols(w["X"].astype(float)) for _, w in items])
        grp = np.concatenate([np.full(len(w["X"]), i) for i, (_, w) in enumerate(items)])
        for target in TARGETS:
            y = np.concatenate([t["labels"][target] for t, _ in items])
            for name, (kind, vars_) in SETS.items():
                cols = [BEHAVIOR_VARS.index(v) for v in vars_]
                ok = np.isfinite(y) & np.isfinite(X[:, cols]).all(1)
                if ok.sum() < cfg.MIN_TRIALS_ANALYSIS:
                    continue
                vals, cnt = np.unique(y[ok], return_counts=True)
                small = vals[cnt < cfg.MIN_TRIALS_PER_TARGET_BIN]
                if len(small):
                    ok &= ~np.isin(y, small)
                if ok.sum() < cfg.MIN_TRIALS_ANALYSIS or len(np.unique(y[ok])) < 2:
                    continue
                for dec in DEC_RUN:
                    res = decode(X[ok][:, cols], y[ok], groups=grp[ok], n_folds=N_FOLDS, n_repeats=N_REPEATS, n_perm=n_perm,
                                 rng=rng, method=dec)
                    if res is None:
                        continue
                    rows.append(dict(protocol=protocol, unit=unit, animal_id=animal, decoder=dec, axis=axis, binning=binning,
                                     window=window, s_rel=s_rel, target=target, feature_set=name, kind=kind, **res))
    print(f"  [{protocol} {unit}] {len(rows)} decoders")
    return rows


units = []
for p in PROTOCOLS:
    pt_ = [t for t in tables if t["protocol"] == p]
    if not pt_:
        continue
    if AGG == "pooled":
        units.append((p, "all", "", pt_))
    elif AGG == "session":
        units += [(p, t["session_id"], t["animal_id"], [t]) for t in pt_]
    else:
        for a in sorted({t["animal_id"] for t in pt_}):
            units.append((p, a, a, [t for t in pt_ if t["animal_id"] == a]))
print(f"\n{len(units)} decoding units ({UNIT}); every decoder: {N_REPEATS} CV repeats + {args.n_perm} permutations")
nested = Parallel(n_jobs=scope.n_jobs, backend="loky", verbose=JOBLIB_VERBOSE)(
    delayed(decode_unit)(p, u, a, st, args.n_perm, stable_seed("1e", p, u, base=RANDOM_STATE)) for p, u, a, st in units)
res = pd.DataFrame([r for g in nested for r in g])
if res.empty:
    raise SystemExit("No window had enough trials for any decoder.")
res.to_csv(table_dir / f"behavior_decoding{AGG_SUFFIX}.csv", index=False)
KEYS = ["protocol", "decoder", "axis", "binning", "window", "s_rel", "target", "feature_set", "kind"]
g = res.groupby(KEYS)
curves = g[["acc", "null_mean", "null_sd", "chance"]].mean()
curves["acc_sem"] = g["acc"].sem().fillna(0.0) if AGG != "pooled" else 0.0
curves["n_units"] = g.size()
curves["n_trials"] = g["n"].mean()
curves = curves.reset_index()
curves.to_csv(table_dir / f"behavior_decoding_curves{AGG_SUFFIX}.csv", index=False)

# wide table: leave-one-out contribution = acc(all) - acc(all but X), per unit then averaged
wide = res.pivot_table(index=["protocol", "unit", "decoder", "axis", "binning", "window", "s_rel", "target"], columns="feature_set",
                       values="acc").reset_index()
loo_rows = []
for v in ADD_ORDER:
    if ALL_NAME in wide and f"all_but_{v}" in wide:
        d = wide[["protocol", "unit", "decoder", "axis", "binning", "window", "s_rel", "target"]].copy()
        d["variable"] = v
        d["delta"] = wide[ALL_NAME] - wide[f"all_but_{v}"]
        loo_rows.append(d.dropna(subset=["delta"]))
loo = pd.concat(loo_rows) if loo_rows else pd.DataFrame()
if len(loo):
    gl = loo.groupby(["protocol", "decoder", "axis", "binning", "window", "s_rel", "target", "variable"])
    loo_c = gl["delta"].mean().rename("delta").to_frame()
    loo_c["delta_sem"] = gl["delta"].sem().fillna(0.0) if AGG != "pooled" else 0.0
    loo_c = loo_c.reset_index()
    loo_c.to_csv(table_dir / f"behavior_decoding_leave_one_out{AGG_SUFFIX}.csv", index=False)
else:
    loo_c = pd.DataFrame()


# --------------------------------------------------------------------- #
# 4. Figures
# --------------------------------------------------------------------- #
protocols_present = [p for p in PROTOCOLS if (curves["protocol"] == p).any()]


def decorate(ax, axis, ylabel):
    if axis == "position":
        for lo, hi in WINDOWS.values():
            ax.axvspan(lo, hi, color="grey", alpha=0.12, lw=0, zorder=0)
    ax.axvline(0, color="black", ls=":", lw=1)
    ax.set_xlabel(axes_def[axis]["label"])
    ax.set_ylabel(ylabel)


def line(ax, c, y, ysem, color, label, ls="-", lw=1.5):
    c = c.sort_values("s_rel")
    if c.empty:
        return
    ax.plot(c["s_rel"], c[y], color=color, ls=ls, lw=lw, label=label)
    if ysem is not None and (c[ysem] > 0).any():
        ax.fill_between(c["s_rel"], c[y] - c[ysem], c[y] + c[ysem], color=color, alpha=0.16, lw=0)


def chance_band(ax, c):
    c = c.sort_values("s_rel")
    if c.empty:
        return
    ax.plot(c["s_rel"], c["null_mean"], color="grey", lw=1, ls="--")
    ax.fill_between(c["s_rel"], c["null_mean"] - c["null_sd"], c["null_mean"] + c["null_sd"], color="grey", alpha=0.2, lw=0)


def bn_note(axis, binning):
    ad = axes_def[axis]
    return (f"fixed {ad['binsize']:g} {ad['unit']} bins" if binning == "fixed"
            else f"sliding {ad['window']:g} {ad['unit']} window, step {ad['step']:g}")


SINGLE_NAMES = list(ADD_ORDER) + [ALL_NAME]
for axis in AXES:
    for binning in BINS_RUN:
        if binning == "windows":
            continue
        cb = curves[(curves["axis"] == axis) & (curves["binning"] == binning)]
        if cb.empty:
            continue
        note = bn_note(axis, binning)
        for target in TARGETS:
            ct_all = cb[cb["target"] == target]
            if ct_all.empty:
                continue
            tag = f"{target} [{' | '.join(target_bin_names(target))}]"
            prots = [p for p in protocols_present if (ct_all["protocol"] == p).any()]
            tdir = figdir / axis / binning / target
            tdir.mkdir(parents=True, exist_ok=True)

            # 0. decoders compared: one row per feature set, one column per protocol
            fig, axs = plt.subplots(len(SINGLE_NAMES), len(prots), figsize=(5.0 * len(prots), 2.7 * len(SINGLE_NAMES)),
                                    squeeze=False, sharex=True, sharey=SHARE_Y)
            for r, name in enumerate(SINGLE_NAMES):
                for ax, p in zip(axs[r], prots):
                    cp = ct_all[(ct_all["protocol"] == p) & (ct_all["feature_set"] == name)]
                    chance_band(ax, cp[cp["decoder"] == DEC_RUN[0]])
                    for dec in DEC_RUN:
                        line(ax, cp[cp["decoder"] == dec], "acc", "acc_sem", DECODER_COLORS[dec], dec)
                    decorate(ax, axis, "balanced accuracy")
                    ax.set_title(f"{p}: {name if name != ALL_NAME else 'all four'}", fontsize=8,
                                 color=PROTOCOL_COLORS.get(p, "k"))
                    if r == 0 and p == prots[0]:
                        ax.legend(fontsize=7, frameon=False)
            fig.suptitle(f"Decoders compared -- {tag} over {axis} ({note}); grey = permutation null (first decoder); {agg_note}",
                         y=1.0, fontsize=9)
            fig.tight_layout()
            save_fig(fig, tdir, "0_decoder_comparison")
            plt.close(fig)

            for dec in DEC_RUN:
                ct_ = ct_all[ct_all["decoder"] == dec]
                ddir = tdir / dec
                ddir.mkdir(parents=True, exist_ok=True)
                for fname, names, colors, title in [
                        ("1_single_variables", SINGLE_NAMES, {**VAR_COLORS, ALL_NAME: "k"},
                         "single behavioral variables (black dashed = all four)"),
                        ("2_cumulative_sets", list(ADD_ORDER[:1]) + CUM_NAMES,
                         {n: c for n, c in zip(list(ADD_ORDER[:1]) + CUM_NAMES,
                                               plt.cm.viridis(np.linspace(0.05, 0.9, len(CUM_NAMES) + 1)))},
                         "variables added one at a time")]:
                    fig, axs = plt.subplots(1, len(prots), figsize=(5.2 * len(prots), 4.4), squeeze=False, sharey=SHARE_Y)
                    for ax, p in zip(axs[0], prots):
                        cp = ct_[ct_["protocol"] == p]
                        chance_band(ax, cp[cp["feature_set"] == names[0]])
                        for n in names:
                            line(ax, cp[cp["feature_set"] == n], "acc", "acc_sem", colors[n], n,
                                 ls="--" if n == ALL_NAME and fname.startswith("1") else "-", lw=1.8 if n == ALL_NAME else 1.4)
                        decorate(ax, axis, "balanced accuracy")
                        nu = int(cp["n_units"].max()) if len(cp) else 0
                        ax.set_title(f"{p} (n={nu} {UNIT}{'s' if nu != 1 else ''})", color=PROTOCOL_COLORS.get(p, "k"),
                                     fontsize=10)
                        ax.legend(fontsize=6, frameon=False)
                    fig.suptitle(f"[{dec}] Decoding {tag} from {title}, over {axis} ({note}); grey = permutation null; "
                                 f"{agg_note}", y=1.03, fontsize=9)
                    fig.tight_layout()
                    save_fig(fig, ddir, fname)
                    plt.close(fig)
                if len(loo_c):
                    cl = loo_c[(loo_c["decoder"] == dec) & (loo_c["axis"] == axis) & (loo_c["binning"] == binning)
                               & (loo_c["target"] == target)]
                    if not cl.empty:
                        fig, axs = plt.subplots(1, len(prots), figsize=(5.2 * len(prots), 4.4), squeeze=False, sharey=SHARE_Y)
                        for ax, p in zip(axs[0], prots):
                            for v in ADD_ORDER:
                                line(ax, cl[(cl["protocol"] == p) & (cl["variable"] == v)], "delta", "delta_sem",
                                     VAR_COLORS[v], v)
                            ax.axhline(0, color="grey", lw=0.7)
                            decorate(ax, axis, "accuracy(all four) - accuracy(all but X)")
                            ax.set_title(p, color=PROTOCOL_COLORS.get(p, "k"), fontsize=10)
                            ax.legend(fontsize=6, frameon=False)
                        fig.suptitle(f"[{dec}] What each variable adds to the others, {tag}, over {axis} ({note}); {agg_note}",
                                     y=1.03, fontsize=9)
                        fig.tight_layout()
                        save_fig(fig, ddir, "3_leave_one_out")
                        plt.close(fig)

# bars in 3a-3c's three windows
cw_all = curves[(curves["axis"] == "position") & (curves["binning"] == "windows")]
if not cw_all.empty:
    wdir = figdir / "position" / "windows"
    wdir.mkdir(parents=True, exist_ok=True)
    wnames = list(WINDOWS)

    def bars(ax, cp, names, colors, labels):
        width = 0.8 / len(names)
        for i, (n, col, lab) in enumerate(zip(names, colors, labels)):
            c = cp[cp["_series"] == n].set_index("window")
            xs = [wi - 0.4 + width * (i + 0.5) for wi in range(len(wnames))]
            ax.bar(xs, [c["acc"].get(w, np.nan) for w in wnames], width=width * 0.92, color=col, alpha=0.85,
                   yerr=[c["acc_sem"].get(w, 0.0) for w in wnames], error_kw=dict(lw=0.8, capsize=1.5), label=lab)
        chance = cp.groupby("window")["null_mean"].mean()
        for wi, w in enumerate(wnames):
            if w in chance.index:
                ax.hlines(chance[w], wi - 0.4, wi + 0.4, color="grey", ls="--", lw=1)
        ax.set_xticks(range(len(wnames)))
        ax.set_xticklabels([f"{w}\n{WINDOWS[w][0]:g}..{WINDOWS[w][1]:g} cm" for w in wnames], fontsize=7)
        ax.set_ylabel("balanced accuracy")
        ax.legend(fontsize=6, frameon=False)

    for target in TARGETS:
        ct_all = cw_all[cw_all["target"] == target]
        prots = [p for p in protocols_present if (ct_all["protocol"] == p).any()]
        if not prots:
            continue
        for dec in DEC_RUN:                                   # feature sets, one file per decoder
            fig, axs = plt.subplots(1, len(prots), figsize=(5.2 * len(prots), 4.2), squeeze=False, sharey=SHARE_Y)
            for ax, p in zip(axs[0], prots):
                cp = ct_all[(ct_all["protocol"] == p) & (ct_all["decoder"] == dec)].assign(
                    _series=lambda d: d["feature_set"])
                bars(ax, cp, SINGLE_NAMES, [{**VAR_COLORS, ALL_NAME: "k"}[n] for n in SINGLE_NAMES],
                     [n if n != ALL_NAME else "all four" for n in SINGLE_NAMES])
                ax.set_title(p, color=PROTOCOL_COLORS.get(p, "k"), fontsize=10)
            fig.suptitle(f"[{dec}] Decoding {target} in 3a-3c's position windows (dashed = permutation null); {agg_note}",
                         y=1.03, fontsize=9)
            fig.tight_layout()
            save_fig(fig, wdir, f"{target}_by_window_{dec}")
            plt.close(fig)
        fig, axs = plt.subplots(1, len(prots), figsize=(5.2 * len(prots), 4.2), squeeze=False, sharey=SHARE_Y)
        for ax, p in zip(axs[0], prots):                      # decoders compared on all four variables
            cp = ct_all[(ct_all["protocol"] == p) & (ct_all["feature_set"] == ALL_NAME)].assign(_series=lambda d: d["decoder"])
            bars(ax, cp, DEC_RUN, [DECODER_COLORS[d] for d in DEC_RUN], DEC_RUN)
            ax.set_title(f"{p}: all four variables", color=PROTOCOL_COLORS.get(p, "k"), fontsize=10)
        fig.suptitle(f"Decoders compared, {target}, position windows; {agg_note}", y=1.03, fontsize=9)
        fig.tight_layout()
        save_fig(fig, wdir, f"{target}_decoders_by_window")
        plt.close(fig)
print(f"\nWrote tables to {table_dir}, figures to {figdir}")
