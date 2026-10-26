# -*- coding: utf-8 -*-
"""
2d_linear_encod
=================
Linear and ROC descriptions of each recorded cell's trial-by-trial activity -- DN sessions only. Two parts
(PART 2 was the separate 2g_roc_analysis, now retired).

PART 1 -- linear encoding (compared with 2c's mutual information)
    Pairwise linear fit (r, r^2, p) of each cell's window-mean activity on each predictor 2c used (same-window
    behavior, stimulus binned by each strategy `stim__<strategy>`, choice, correct) and the comparison with 2c's MI:
    how much of a cell's information does a LINEAR fit capture (mi_bits_linear_binned / frac_explained_linearly)? Plus a
    MULTIVARIATE fit per cell and window (all behavior predictors + one stimulus binning + choice + correct;
    predictors without variance in the trials used are dropped; sessions without video lack pupil / motion energy and
    are fitted with what they have), "how well can this cell be explained by everything measured together". The MI side
    is read from 2c's out/ (Rule #1); the trials, bins and pair rules come from the shared
    infotheory.cell_tables.pair_arrays, so r^2 and MI are computed on IDENTICAL data (`n_trials_match` checks it).
    THE COMPARISON IS DONE ON THE SAME BINS (infotheory.linear_reference): the 3-bin MI is NOT comparable with
    -0.5*log2(1-r^2) (the MI of a continuous Gaussian): discretising to 3 bins loses information even for a perfectly
    linear cell (rho=0.6: 0.20 vs 0.33 bits), so the old figure 01 had nearly every strong pair "above the diagonal".
    `mi_bits_linear_binned` = the 3-bin MI a linear (Gaussian) relation with this pair's adjusted r^2 would give, with
    the pair's actual bin occupancies; `..._rank` the same from Spearman's rho (the quantile binning only sees ranks, so
    this is the matched measure). `frac_explained_linearly` = linear-binned / max(corrected MI, linear-binned).
    Reading figure 01: ON the diagonal = the cell is linear; BELOW (MI above the line's prediction) = information
    beyond a line (candidates for 2e); ABOVE = the Pearson r comes from a few extreme trials (outliers, zero-inflated
    lick) -- the rank version then drops to the diagonal -- or noise. The legacy `mi_bits_from_r2` is kept in the
    table for reference only.
PART 2 -- ROC analysis of single cells (after Chen et al. 2013): how well can an ideal observer tell two trial classes
    apart from ONE cell? Per cell, window and analysis: auc, si = 2*auc-1, permutation null (null_lo/hi), p_value,
    significant (outside the central 95% of the null). Analyses (edit ANALYSES): choice (lick vs no lick), choice|stim
    (within each stimulus level, pooled), stim (signal>0 vs 0), stim|choice (within lick / no-lick, pooled). Methods
    (METHODS): "mean" = AUC of the window mean (2-sided); "template" = Chen et al. leave-one-out template discriminant
    on the within-window profile (1-sided; named "<analysis> [tmpl]"). Fixed-condition ROC = pooled within-group AUC,
    labels permuted within groups (infotheory.roc). Only cells/analyses with enough trials per class (MIN_PER_CLASS)
    have a value; the others are left out of the summaries.
    DIFFERENT RESPONSE DEFINITION than part 1 (kept on purpose): the ROC uses POSITION-MATCHED imaging-frame sampling
    (EQUALIZE_QUANTILE: every trial averages the same number of frames at the same positions) to remove the frame-count
    / running-speed confound that makes plain window means (parts 1, 2c) look choice-selective (false positives ~5%
    instead of ~10% on null cells). Binning rules (RESPONSE_BINS, strategies) do not apply: the ROC has no discretisation.
    Caveats: ~5% of cells pass by chance and no correction across cells is made (compare with the 5% line); speed,
    pupil and licking co-vary with choice (2g_behavior_correlation shows which cells follow behavior).

Dataset: 1b-included sessions, ENGAGED trials, cells passing 2a proximity AND 2b QC (as 2c).

Project-wide rules (infotheory/project_config.py)
    * selection : --animals / --sessions; a restricted run writes to out/selection/<tag>/ and never overwrites out/.
                  Part 1's MI comparison reads the FULL 2c out/ and merges on (session, cell, window, predictor).
    * aggregate : --aggregate pooled|session|mouse for every group figure and summary table: pooled = all cells of a
                  group together (no error bars); session / mouse = one value per unit (MIN_CELLS_PER_UNIT cells in
                  the group needed), bar = mean +- SEM over units, dots = units coloured by mouse. Always the six groups.
                  Cell-level figures (selectivity-index histograms, stim-vs-choice scatters, MI-vs-linear scatter) show
                  cells and are not aggregated. ROC summaries: roc_summary_pooled / _by_session / _by_mouse.csv.
    * memory    : one session in RAM per worker (--n-jobs); per-session results cached in store/.
    * binning   : see PART 1.

EXPLORATORY TEST -- end-of-session DISENGAGED trials (--no-tail to skip)
    Linear r^2 (pairwise + multivariate) and ROC significance on the disengaged block at the END of each session
    (definition / matching: infotheory.engagement; not the sporadic early lapses) versus an equal number of the last
    engaged trials before it. The ROC of the two conditions is computed together so both share the frame-equalization
    target; permutations are reduced (--n-perm-tail) -- a look, not a rigorous null analysis: one block per session,
    few trials, no correction across cells. Disengaged animals rarely lick, so choice analyses are mostly untestable
    there (left out, not zero), and lower r^2/AUC may reflect reduced behavioral variance rather than reduced encoding.

Outputs
-------
2_pipeline/2d_linear_encod/
    out/    cell_predictor_linear.csv          engaged: per (session, window, cell, predictor): r, r2, p_linear, n_trials
            cell_predictor_mi_vs_linear.csv    ... merged with 2c: MI raw/corrected, mi_bits_from_r2, frac_explained_linearly
            cell_multivariate_linear.csv       per (session, window, cell): r2, adj_r2, p_linear
            linear_by_group / multivariate_by_group / mi_vs_linear_by_group [__<aggregate>].csv
            roc_per_cell.csv, roc_summary_pooled|by_session|by_mouse.csv
            tail_trials.csv, tail_cell_predictor_linear.csv, tail_cell_multivariate_linear.csv, tail_roc_per_cell.csv,
            tail_*_by_group[__<aggregate>].csv
            figures/*.png                      01-03 linear, 04-07 ROC, 08-11 tail test
    out/selection/<tag>/...                    the same for a restricted run
    store/  linear_roc_session_<hash>.pkl      per-session caches
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binomtest, spearmanr

from infotheory import project_config as cfg
from infotheory.pipeline import get_pipeline_paths
from infotheory.session import discover_sessions
from infotheory.neural_encoding import load_cell_inclusion, build_session_tables
from infotheory.info_theory import linear_fit, multivariate_linear_fit, mi_from_r2
from infotheory.linear_reference import bin_props, expected_binned_mi, props_to_str
from scipy.stats import rankdata
from infotheory.celldata_utils import SIX_GROUP_ORDER, SIX_GROUP_LABELS, SIX_GROUP_COLORS
from infotheory.groups import six_panel_figure, group_key, summarize_significance
from infotheory.plotting import set_style, save_fig, clear_figures
from infotheory.scope import add_scope_args, scope_from_args, select_sessions, scoped_dirs, data_stamp, unit_noun
from infotheory.session_jobs import run_sessions
from infotheory.unit_stats import stable_seed
from infotheory.unit_plots import group_panels, mouse_palette
from infotheory.stim_binning import STIM_BINNING_STRATEGIES, DEFAULT_STRATEGIES, target_signature
from infotheory.roc import auc_permutation_test, template_auc_permutation_test
from infotheory import tensors as tn
from infotheory import cell_tables as ct
from infotheory import engagement as eg

PROTOCOL = "DN"
WINDOWS = {"pre_stim": (-30.0, -10.0), "stim": (0.0, 20.0), "reward": (25.0, 45.0)}   # as 2c

# ---- part 1 ----
BINNING_STRATEGIES = list(DEFAULT_STRATEGIES)
MV_STIM_STRATEGY = "pct_0-20_20-80_80-100"   # the one stimulus binning used in the multivariate fit
LINEAR_SUFFICIENT_FRAC = 0.8

# ---- part 2 (ROC) ----
S_PRE, S_POST = 80.0, 60.0       # frames within these cm of a stimulus are assigned to its trial
ANALYSES = {                     # name -> (positive-class column, condition column or None)
    "choice":      ("choice", None),
    "choice|stim": ("choice", "signal"),
    "stim":        ("stim_present", None),
    "stim|choice": ("stim_present", "choice"),
}
SCATTER_PAIRS = [("stim", "choice"), ("stim|choice", "choice|stim"),
                 ("stim [tmpl]", "choice [tmpl]"), ("stim|choice [tmpl]", "choice|stim [tmpl]")]
HEATMAP_WINDOW = "stim"
METHODS = ("mean", "template")
N_TEMPLATE_BINS = 4
N_PERM, N_PERM_TEMPLATE = 1000, 200
N_PERM_TAIL, N_PERM_TEMPLATE_TAIL = 300, 100
ALPHA = 0.05
MIN_PER_CLASS = 5
EQUALIZE_QUANTILE = 0.1          # None = plain window mean
ANALYSIS_NAMES = [a + sfx for a in ANALYSES for m, sfx in (("mean", ""), ("template", " [tmpl]")) if m in METHODS]

MIN_CELLS_PER_UNIT = 5
RANDOM_STATE = 0
RUN_TAIL = True
JOBLIB_VERBOSE = 5
CONDITIONS = ("engaged_matched", "disengaged")

parser = argparse.ArgumentParser()
add_scope_args(parser)
parser.add_argument("--strategies", nargs="+", default=None, choices=sorted(STIM_BINNING_STRATEGIES))
parser.add_argument("--n-perm", type=int, default=N_PERM, help="ROC permutations (smaller = faster look)")
parser.add_argument("--n-perm-template", type=int, default=N_PERM_TEMPLATE)
parser.add_argument("--n-perm-tail", type=int, default=N_PERM_TAIL)
parser.add_argument("--n-perm-template-tail", type=int, default=N_PERM_TEMPLATE_TAIL)
parser.add_argument("--no-roc", action="store_true", help="skip part 2")
parser.add_argument("--no-tail", action="store_true", help="skip the disengaged-tail test")
parser.add_argument("--tail-min-frac", type=float, default=eg.TAIL_MIN_FRAC)
parser.add_argument("--tail-min-trials", type=int, default=eg.TAIL_MIN_TRIALS)
args = parser.parse_args()
scope = scope_from_args(args)
AGG = scope.aggregate
UNIT = unit_noun(AGG)
UNITS = "mice" if UNIT == "mouse" else f"{UNIT}s"
N_JOBS = scope.n_jobs
AGG_SUFFIX = "" if AGG == "pooled" else f"__{AGG}"
UNIT_KEYS = {"pooled": [], "session": ["session_id"], "mouse": ["animal_id"]}[AGG]
STRATEGIES = args.strategies or BINNING_STRATEGIES
PREDICTORS = ct.predictor_list(STRATEGIES)
MV_STRAT = MV_STIM_STRATEGY if MV_STIM_STRATEGY in STRATEGIES else STRATEGIES[0]
MV_PREDICTORS = ct.BEHAVIOR_COLUMNS + [ct.stim_predictor(MV_STRAT)] + ct.OUTCOME_COLUMNS
RUN_TAIL = RUN_TAIL and not args.no_tail
RUN_ROC = not args.no_roc

paths = get_pipeline_paths(__file__)
table_dir, figdir = scoped_dirs(paths, scope)
clear_figures(figdir)
set_style()
print(scope.describe())

# --------------------------------------------------------------------- #
# 1. Inputs
# --------------------------------------------------------------------- #
fits_path = paths.out_from("1b_psychometric") / "psychometric_fits.csv"
cell_dist_out = paths.out_from("2a_cell_distribution")
activity_out = paths.out_from("2b_activity_statistics")
information_out = paths.out_from("2c_information")
for f, hint in [(fits_path, "1b_psychometric.py"), (cell_dist_out / "celldata_combined.csv", "2a_cell_distribution.py"),
                (activity_out / "qc_per_cell.csv", "2b_activity_statistics.py"),
                (information_out / "cell_predictor_mi.csv", "2c_information.py")]:
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
cells_by_session = {sid: sorted(sub.loc[sub["include"], "cell_id"]) for sid, sub in cell_inclusion.groupby("session_id")
                    if sid in session_ids}
session_ids = [s for s in session_ids if cells_by_session.get(s)]
if not session_ids:
    raise SystemExit("No included cells in the selected sessions.")
print(f"{len(session_ids)} sessions, {sum(len(v) for v in cells_by_session.values())} included cells")
stamps = {s: data_stamp(*selected[s]) for s in session_ids}


# --------------------------------------------------------------------- #
# 2. Per session: linear fits (all conditions) + ROC (engaged; tail pair)
# --------------------------------------------------------------------- #
def linear_records(table, cond, window, cell_cols, sid):
    pw, mv = [], []
    mv_avail = [p for p in MV_PREDICTORS if p in table.columns]    # sessions without video lack pupil/motion
    for cell in cell_cols:
        for pred in PREDICTORS:
            pair = ct.pair_arrays(table, pred, cell)
            if pair is None:
                continue
            x, y, _ = pair
            lin = linear_fit(x, y)
            # what a LINEAR relation predicts for the 3-bin MI of 2c (same bins, same trials; infotheory.linear_reference)
            xp, yp = bin_props(x, cfg.RESPONSE_BINS), bin_props(y, cfg.RESPONSE_BINS)
            r2_adj = max(1.0 - (1.0 - lin.r2) * (lin.n - 1) / max(lin.n - 2, 1), 0.0) if np.isfinite(lin.r2) else np.nan
            rho_s = np.corrcoef(rankdata(x), rankdata(y))[0, 1] if len(xp) > 1 and len(yp) > 1 else np.nan
            pw.append(dict(session_id=sid, condition=cond, window=window, cell_id=cell, predictor=pred,
                           n_trials=lin.n, r=lin.r, r2=lin.r2, p_linear=lin.p_value, r2_adj=r2_adj, rho_spearman=rho_s,
                           mi_bits_linear_binned=expected_binned_mi(np.sqrt(r2_adj), xp, yp) if np.isfinite(r2_adj) else np.nan,
                           mi_bits_linear_binned_rank=expected_binned_mi(rho_s, xp, yp),
                           x_props=props_to_str(xp), y_props=props_to_str(yp)))
        d = table[mv_avail + [cell]].replace([np.inf, -np.inf], np.nan).dropna()
        use = [p for p in mv_avail if d[p].std() > 0]
        if len(d) >= max(cfg.MIN_TRIALS_ANALYSIS, len(use) + 10) and use:
            fit = multivariate_linear_fit(d[use].to_numpy(float), d[cell].to_numpy(float), predictor_names=use)
            mv.append(dict(session_id=sid, condition=cond, window=window, cell_id=cell, n_predictors=len(use),
                           n_trials=fit.n, r2=fit.r2, adj_r2=fit.adj_r2, p_linear=fit.p_value))
    return pw, mv


def roc_trial_table(td, nums):
    tr = td[td["trialNumber"].isin(nums)].sort_values("trialNumber").reset_index(drop=True)
    return tr.assign(choice=tr["lickResponse"].astype(int), stim_present=(tr["signal"] > 0).astype(int))


def roc_block(ses, activity, zpos_F, trials, masks, cell_ids, n_perm, n_perm_tmpl, rng):
    """ROC for the cells of `activity` on the trials of `trials`; `masks` = {condition: bool array over trials}
    (conditions share the frame sampling). Returns a list of DataFrames."""
    frame_ok = tn.trial_frame_mask(ses.ts_F, ses.trialdata, trials["trialNumber"])
    k, rel = tn.position_relative_to_stim(zpos_F, trials["stimStart"].to_numpy(), S_PRE, S_POST)
    win_means = (tn.trial_window_means(activity, k, rel, frame_ok, len(trials), WINDOWS,
                                       equalize_quantile=EQUALIZE_QUANTILE) if "mean" in METHODS else {})
    win_prof = ({w: tn.trial_window_profiles(activity, k, rel, frame_ok, len(trials), win, N_TEMPLATE_BINS)
                 for w, win in WINDOWS.items()} if "template" in METHODS else {})
    out = []
    for cond, mask in masks.items():
        for wname in WINDOWS:
            for aname, (lab_col, cond_col) in ANALYSES.items():
                for method, M in (("mean", win_means.get(wname)), ("template", win_prof.get(wname))):
                    if M is None:
                        continue
                    ok = np.isfinite(M).reshape(len(M), -1).all(1) & mask
                    if ok.sum() < 2:
                        continue
                    groups = None if cond_col is None else trials[cond_col].to_numpy()[ok]
                    labels = trials[lab_col].to_numpy()[ok].astype(bool)
                    if method == "mean":
                        res = auc_permutation_test(M[ok], labels, groups=groups, n_perm=n_perm, rng=rng,
                                                   min_per_class=MIN_PER_CLASS, alpha=ALPHA)
                        an = aname
                    else:
                        res = template_auc_permutation_test(M[ok], labels, groups=groups, n_perm=n_perm_tmpl,
                                                            rng=rng, min_per_class=MIN_PER_CLASS, alpha=ALPHA)
                        an = aname + " [tmpl]"
                    out.append(pd.DataFrame({
                        "session_id": ses.session_id, "condition": cond, "cell_id": cell_ids, "window": wname,
                        "analysis": an, "auc": res["auc"], "si": res["si"], "null_lo": res["null_lo"],
                        "null_hi": res["null_hi"], "p_value": res["p_value"], "significant": res["significant"],
                        "n_pos": res["n_pos"], "n_neg": res["n_neg"], "n_groups": res["n_groups"],
                        "n_trials_used": int(ok.sum()), "n_trials_total": len(trials)}))
    return out


def process_session(ses, seed):
    sid = ses.session_id
    cells = set(cells_by_session.get(sid, []))
    td = ses.trialdata
    if not cells or td is None or "engaged" not in td.columns:
        return None
    engaged_nums = td.loc[td["engaged"] == 1, "trialNumber"].astype(int).tolist()
    tail = eg.tail_condition_trials(td, args.tail_min_frac, args.tail_min_trials) if RUN_TAIL else None
    wanted = set(engaged_nums) | (set(tail["disengaged"]) if tail else set())
    sets = {"engaged": set(engaged_nums)}
    if tail:
        sets.update({"engaged_matched": set(tail["engaged_matched"]), "disengaged": set(tail["disengaged"])})

    pw, mv = [], []
    tables = build_session_tables(ses, mu_sigma_lookup, WINDOWS, included_cell_ids=cells, trial_numbers=wanted)
    for window, table in tables.items():
        if table.empty:
            continue
        cell_cols = ct.cell_columns(table, cells)
        for cond, nums in sets.items():
            sub = table[table["trialNumber"].isin(nums)]
            if len(sub) < cfg.MIN_TRIALS_ANALYSIS:
                continue
            a, b = linear_records(ct.add_stimulus_columns(sub, STRATEGIES), cond, window, cell_cols, sid)
            pw += a
            mv += b

    roc = []
    if RUN_ROC and ses.calciumdata is not None and ses.ts_F is not None and ses.behaviordata is not None:
        cell_ids = sorted(cells)
        activity = ses.calciumdata.loc[:, cell_ids].to_numpy(dtype=float)
        zpos_F = tn.interp_to_imaging_frames(ses.behaviordata["ts"], ses.behaviordata["zpos"], ses.ts_F)
        rng = np.random.default_rng(stable_seed(sid, "roc", base=seed))
        tr = roc_trial_table(td, engaged_nums)
        if len(tr) >= cfg.MIN_TRIALS_ANALYSIS:
            roc += roc_block(ses, activity, zpos_F, tr, {"engaged": np.ones(len(tr), bool)}, cell_ids,
                             args.n_perm, args.n_perm_template, rng)
        if tail:
            tr = roc_trial_table(td, set(tail["engaged_matched"]) | set(tail["disengaged"]))
            nums = tr["trialNumber"]
            roc += roc_block(ses, activity, zpos_F, tr,
                             {"engaged_matched": nums.isin(tail["engaged_matched"]).to_numpy(),
                              "disengaged": nums.isin(tail["disengaged"]).to_numpy()}, cell_ids,
                             args.n_perm_tail, args.n_perm_template_tail, rng)
    return dict(session_id=sid, animal_id=ses.animal_id, pairwise=pd.DataFrame(pw), multivariate=pd.DataFrame(mv),
                roc=pd.concat(roc, ignore_index=True) if roc else pd.DataFrame(), tail=tail,
                trials=td[["trialNumber", "engaged"]].copy())


results = run_sessions(
    session_ids, process_session, [PROTOCOL], paths.store, "linear_roc_session",
    key_params=dict(version="2d-v3", bins=cfg.RESPONSE_BINS, windows=WINDOWS, strategies=target_signature(STRATEGIES), mv=MV_PREDICTORS,
                    min_trials=cfg.MIN_TRIALS_ANALYSIS, min_per_bin=cfg.MIN_TRIALS_PER_TARGET_BIN,
                    roc=[RUN_ROC, args.n_perm, args.n_perm_template, args.n_perm_tail, args.n_perm_template_tail,
                         list(ANALYSES.items()), METHODS, N_TEMPLATE_BINS, ALPHA, MIN_PER_CLASS, EQUALIZE_QUANTILE,
                         S_PRE, S_POST],
                    tail=[RUN_TAIL, args.tail_min_frac, args.tail_min_trials], fits_mtime=int(fits_path.stat().st_mtime)),
    load_kwargs=dict(load_behaviordata=True, load_videodata=True, load_celldata=True, load_calciumdata=True),
    cells_by_session=cells_by_session, base_seed=RANDOM_STATE, n_jobs=N_JOBS, recompute=args.recompute,
    verbose=JOBLIB_VERBOSE, extra_key_by_session=stamps, flatten=False)
results = [r for r in results if r is not None]
if not results:
    raise SystemExit("No session produced tables.")

cd = pd.read_csv(cell_dist_out / "celldata_combined.csv", low_memory=False)


def stack(key):
    parts = [r[key] for r in results if len(r[key])]
    return ct.attach_cell_meta(pd.concat(parts, ignore_index=True), cd) if parts else pd.DataFrame()


lin_all, mv_all, roc_all = stack("pairwise"), stack("multivariate"), stack("roc")
if lin_all.empty:
    raise SystemExit("No (cell, predictor) pair had enough valid trials.")
lin_all["significant"] = lin_all["p_linear"] < ALPHA
if len(mv_all):
    mv_all["significant"] = mv_all["p_linear"] < ALPHA
UNIT_SIZE = lambda df: df.groupby(UNIT_KEYS).ngroups if UNIT_KEYS else 1  # noqa: E731


def gtable(df, by, name, mean_cols, sig_col=None):
    t = ct.unit_group_summary(df, by, AGG, mean_cols, sig_col=sig_col)
    t.to_csv(table_dir / f"{name}{AGG_SUFFIX}.csv", index=False)
    return t


# --------------------------------------------------------------------- #
# 3. PART 1: linear vs MI (engaged trials)
# --------------------------------------------------------------------- #
lin = lin_all[lin_all["condition"] == "engaged"].drop(columns="condition")
mv = mv_all[mv_all["condition"] == "engaged"].drop(columns="condition") if len(mv_all) else mv_all
lin.to_csv(table_dir / "cell_predictor_linear.csv", index=False)
mv.to_csv(table_dir / "cell_multivariate_linear.csv", index=False)
print(f"\n{len(lin)} pairwise linear fits, {len(mv)} multivariate fits (engaged trials)")

mi_table = pd.read_csv(information_out / "cell_predictor_mi.csv")
keys = ["session_id", "window", "cell_id", "predictor"]
cmp_df = lin.merge(mi_table[keys + ["n_trials", "mi_bits_raw", "mi_bits_corrected", "mi_p_value"]], on=keys,
                   how="inner", suffixes=("", "_mi"))
if len(cmp_df):
    cmp_df["n_trials_match"] = cmp_df["n_trials"] == cmp_df["n_trials_mi"]
    print(f"Agreement with 2c: {len(cmp_df)}/{len(lin)} pairs found in 2c's table; "
          f"{cmp_df['n_trials_match'].mean():.1%} use exactly the same number of trials")
    comps = [mi_from_r2(max(r2, 0.0)) for r2 in cmp_df["r2"]]
    cmp_df["mi_bits_from_r2"] = comps                      # continuous-Gaussian value, NOT comparable with the 3-bin MI
    cmp_df["mi_reference_bits"] = np.fmax(cmp_df["mi_bits_corrected"], cmp_df["mi_bits_linear_binned"])
    ref = cmp_df["mi_reference_bits"]
    cmp_df["frac_explained_linearly"] = np.where(ref > 1e-9, cmp_df["mi_bits_linear_binned"] / ref, np.nan)
    cmp_df["linear_sufficient"] = cmp_df["frac_explained_linearly"] >= LINEAR_SUFFICIENT_FRAC
    cmp_df.to_csv(table_dir / "cell_predictor_mi_vs_linear.csv", index=False)
    gtable(cmp_df, ["window", "predictor"], "mi_vs_linear_by_group",
           ["r2", "mi_bits_raw", "mi_bits_corrected", "mi_bits_linear_binned", "mi_bits_linear_binned_rank",
            "frac_explained_linearly"])
    fin = cmp_df.dropna(subset=["frac_explained_linearly"])
    insuff = fin.groupby(UNIT_KEYS + ["predictor"])["linear_sufficient"].apply(lambda s: (~s).mean()).reset_index()
    insuff = insuff.groupby("predictor")["linear_sufficient"].mean().sort_values(ascending=False)
    print(f"\nShare of (cell, window) pairs where a linear fit misses >{100 * (1 - LINEAR_SUFFICIENT_FRAC):.0f}% of the "
          f"MI (candidates for 2e), mean over {UNITS}:" if UNIT_KEYS else
          f"\nShare of pairs where a linear fit misses >{100 * (1 - LINEAR_SUFFICIENT_FRAC):.0f}% of the MI (candidates for 2e):")
    print(insuff.round(3).to_string())
gtable(lin, ["window", "predictor"], "linear_by_group", ["r2", "r"], sig_col="significant")
if len(mv):
    gtable(mv, ["window"], "multivariate_by_group", ["r2", "adj_r2"], sig_col="significant")

agg_note = {"pooled": "all cells pooled (no error bars)", "session": "bar = mean +- SEM over sessions, dots = sessions",
            "mouse": "bar = mean +- SEM over mice, dots = mice"}[AGG]


def panels(df, window, panel_col, order, value, name, ylabel, title, **kw):
    group_panels(df[df["window"] == window] if window else df, panel_col, order, value, AGG, figdir, name, title,
                 ylabel=ylabel, mouse_colors=mouse_colors, min_cells_per_unit=MIN_CELLS_PER_UNIT, **kw)


if len(cmp_df):
    for w in WINDOWS:
        sub = cmp_df[cmp_df["window"] == w].replace([np.inf, -np.inf], np.nan).dropna(
            subset=["mi_bits_corrected", "mi_bits_linear_binned"])
        if not sub.empty:
            lim = max(sub["mi_bits_corrected"].max(), sub["mi_bits_linear_binned"].max(),
                      sub["mi_bits_linear_binned_rank"].max()) * 1.05 + 1e-9
            fig, axes = plt.subplots(1, 2, figsize=(10.4, 5), squeeze=False)
            pred_order = [p for p in PREDICTORS if p in set(sub["predictor"])]
            for ax, col, lab in ((axes[0, 0], "mi_bits_linear_binned", "Pearson r (adjusted r²)"),
                                 (axes[0, 1], "mi_bits_linear_binned_rank", "Spearman rho")):
                for p_, c_ in zip(pred_order, plt.cm.tab10(np.arange(len(pred_order)))):
                    s_ = sub[sub["predictor"] == p_]
                    ax.scatter(s_["mi_bits_corrected"], s_[col], s=8, alpha=0.5, color=c_, label=p_)
                ax.plot([0, lim], [0, lim], color="grey", ls="--", lw=1)
                ax.set_xlim(0, lim); ax.set_ylim(0, lim)
                rho = spearmanr(sub["mi_bits_corrected"], sub[col])[0]
                ax.set_xlabel("MI from 2c, 3 bins (bits, bias-corrected)")
                ax.set_ylabel(f"MI a LINEAR relation predicts for the same bins (bits)\nfrom {lab}")
                ax.set_title(f"{w}: Spearman {rho:.2f}, {len(sub)} pairs; above diagonal = MI is lower than a line predicts",
                             fontsize=8)
            axes[0, 0].legend(fontsize=6, markerscale=2, frameon=False)
            fig.tight_layout()
            save_fig(fig, figdir, f"01_mi_vs_linear_{w}")
            plt.close(fig)
        panels(cmp_df.dropna(subset=["frac_explained_linearly"]), w, "predictor", PREDICTORS, "frac_explained_linearly",
               f"02_frac_explained_linearly_{w}", "MI(linear) / MI(full)",
               f"Fraction of MI a linear fit captures -- {w}; dotted = 1; {agg_note}", hline=1.0)
if len(mv):
    fig, axes = plt.subplots(1, len(WINDOWS), figsize=(5.0 * len(WINDOWS), 3.6), squeeze=False)
    from infotheory.unit_plots import group_unit_bars, mouse_legend
    for ax, w in zip(axes[0], WINDOWS):
        s = mv[mv["window"] == w]
        ax.set_title(w, fontsize=9)
        if len(s):
            group_unit_bars(ax, s, "adj_r2", AGG, mouse_colors=mouse_colors, min_cells_per_unit=MIN_CELLS_PER_UNIT,
                            hline=0, ylabel="adjusted R² (all predictors jointly)")
    if AGG != "pooled":
        mouse_legend(axes[0, -1], mouse_colors, animals=mv["animal_id"].unique(), loc="upper right")
    fig.suptitle(f"Multivariate fit of each cell's activity -- {PROTOCOL}; {agg_note}", y=1.02)
    fig.tight_layout()
    save_fig(fig, figdir, "03_multivariate_r2")
    plt.close(fig)

# --------------------------------------------------------------------- #
# 4. PART 2: ROC (engaged trials)
# --------------------------------------------------------------------- #
GROUP_KEYS = [group_key(g) for g in SIX_GROUP_ORDER]


def roc_summaries(df, name_prefix):
    d = df[np.isfinite(df["auc"])].copy()
    d["significant"] = d["significant"].astype(bool)
    pooled = summarize_significance(d, ["group", "window", "analysis"], "si", sign_ref=0.0)
    pooled["p_vs_chance"] = [binomtest(int(s), int(n), ALPHA).pvalue if n > 0 else np.nan
                             for s, n in zip(pooled["n_sig"], pooled["n_cells"])]
    if AGG == "pooled":
        pooled.to_csv(table_dir / f"{name_prefix}_pooled.csv", index=False)
        return d, pooled
    ukeys = ["session_id", "animal_id"] if AGG == "session" else ["animal_id"]
    per = summarize_significance(d, ukeys + ["group", "window", "analysis"], "si", sign_ref=0.0)
    per.to_csv(table_dir / f"{name_prefix}_by_{AGG}.csv", index=False)
    return d, per


if len(roc_all) and (roc_all["condition"] == "engaged").any():
    roc = roc_all[roc_all["condition"] == "engaged"].drop(columns="condition")
    roc.to_csv(table_dir / "roc_per_cell.csv", index=False)
    print(f"\n{len(roc)} (session, cell, window, analysis) ROC rows; {roc['cell_id'].nunique()} cells")
    roc_f, roc_sum = roc_summaries(roc, "roc_summary")
    tab = (roc_f.assign(sig=roc_f["significant"].astype(float))
           .groupby(UNIT_KEYS + ["group", "window", "analysis"])["sig"].mean().reset_index()
           .groupby(["window", "analysis", "group"])["sig"].mean().unstack("group").reindex(columns=GROUP_KEYS))
    print(f"\nFraction of cells significant (chance = {ALPHA:.0%}), "
          f"{'pooled' if AGG == 'pooled' else f'mean over {UNITS} of the per-{UNIT} fraction'}:")
    print(tab.round(2).to_string())

    windows = list(WINDOWS)
    bins_ = np.linspace(-1, 1, 41)
    safe = lambda a: a.replace("|", "_given_").replace(" [tmpl]", "_tmpl")  # noqa: E731
    for aname in ANALYSIS_NAMES:                                    # 04: cell-level SI histograms
        fig, axes = plt.subplots(len(windows), 6, figsize=(16, 2.4 * len(windows)), sharex=True, squeeze=False)
        for i, w in enumerate(windows):
            for j, g in enumerate(SIX_GROUP_ORDER):
                ax = axes[i, j]
                s = roc_f[(roc_f["analysis"] == aname) & (roc_f["window"] == w) & (roc_f["group"] == group_key(g))]
                ax.hist(s.loc[~s["significant"], "si"], bins=bins_, color="#bbbbbb")
                ax.hist(s.loc[s["significant"], "si"], bins=bins_, color=SIX_GROUP_COLORS[g])
                ax.axvline(0, color="k", lw=0.6)
                ax.text(0.98, 0.95, f"{s['significant'].mean():.0%} sig\nn={len(s)}" if len(s) else "no cells",
                        transform=ax.transAxes, ha="right", va="top", fontsize=7)
                if i == 0:
                    ax.set_title(SIX_GROUP_LABELS[g], color=SIX_GROUP_COLORS[g], fontsize=10)
                if j == 0:
                    ax.set_ylabel(f"{w}\ncells")
                if i == len(windows) - 1:
                    ax.set_xlabel("selectivity index")
        fig.suptitle(f"ROC selectivity index -- {aname} (colored = outside 95% of null; all cells, not aggregated)")
        fig.tight_layout()
        save_fig(fig, figdir, f"04_roc_si_hist_{safe(aname)}")
        plt.close(fig)

    rs_ = roc_f.assign(sig=roc_f["significant"].astype(float))     # 05: fraction significant at the unit level
    fig, axes = plt.subplots(1, len(windows), figsize=(6.2 * len(windows), 4.0), squeeze=False, sharey=True)
    from infotheory.unit_plots import grouped_series_bars
    for ax, w in zip(axes[0], windows):
        ax.set_title(w, fontsize=9)
        handles = grouped_series_bars(ax, rs_[rs_["window"] == w], "sig", "analysis", ANALYSIS_NAMES, AGG,
                                      mouse_colors=mouse_colors, min_cells_per_unit=MIN_CELLS_PER_UNIT,
                                      hline=ALPHA, tick_size=6, ylabel="fraction of cells significant")
    axes[0, -1].legend(*handles, fontsize=6, frameon=False, title="analysis", title_fontsize=7)
    fig.suptitle(f"Fraction of ROC-significant cells; dashed = chance ({ALPHA:.0%}); {agg_note}", y=1.02)
    fig.tight_layout()
    save_fig(fig, figdir, "05_roc_frac_significant_by_group")
    plt.close(fig)

    for xa, ya in SCATTER_PAIRS:                                    # 06: cell-level stim vs choice scatter
        if xa not in ANALYSIS_NAMES or ya not in ANALYSIS_NAMES:
            continue
        for w in windows:
            wide = roc_f[roc_f["window"] == w].pivot_table(index=["session_id", "cell_id", "group"],
                                                           columns="analysis", values=["si", "significant"])
            if wide.empty or ("si", xa) not in wide or ("si", ya) not in wide:
                continue
            wide = wide.reset_index()
            fig, axd = six_panel_figure()
            for g, ax in axd.items():
                s = wide[wide[("group", "")] == group_key(g)]
                x, y = s[("si", xa)], s[("si", ya)]
                sx, sy = s[("significant", xa)] > 0.5, s[("significant", ya)] > 0.5
                ax.scatter(x[~(sx | sy)], y[~(sx | sy)], s=5, color="#bbbbbb")
                ax.scatter(x[sx & ~sy], y[sx & ~sy], s=6, color="#1f77b4")
                ax.scatter(x[sy & ~sx], y[sy & ~sx], s=6, color="#d62728")
                ax.scatter(x[sx & sy], y[sx & sy], s=8, color="#6a0dad")
                ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
                ax.text(0.03, 0.95, f"n={len(s)}", transform=ax.transAxes, fontsize=7, va="top")
                ax.set_xlabel(f"SI {xa}", fontsize=8); ax.set_ylabel(f"SI {ya}", fontsize=8)
            fig.suptitle(f"{w}: grey = neither, blue = {xa} only, red = {ya} only, purple = both (cells)")
            fig.tight_layout()
            save_fig(fig, figdir, f"06_roc_scatter_{safe(xa)}_vs_{safe(ya)}_{w}")
            plt.close(fig)

    # 06b / 06c: the same scatter as COUNTS -- what share of each group's cells is selective for x only, y only, both
    # or neither (significance classes of the scatter's colours). 06b: stacked bars, one panel per window; 06c: one
    # panel per class at the chosen unit (dots = units), for the conditional stim|choice vs choice|stim pair.
    CLS = ["neither", "x only", "y only", "both"]
    CLS_COL = {"neither": "#bbbbbb", "x only": "#1f77b4", "y only": "#d62728", "both": "#6a0dad"}
    ukeys_ = ["session_id", "animal_id"] if AGG == "session" else (["animal_id"] if AGG == "mouse" else [])

    def class_table(xa, ya):
        """One row per (window, cell) with the class of the pair (xa, ya), or None."""
        d = roc_f[roc_f["analysis"].isin([xa, ya])]
        wide_ = d.pivot_table(index=["window", "session_id", "animal_id", "cell_id", "group"], columns="analysis",
                              values="significant", aggfunc="first").reset_index()
        if xa not in wide_ or ya not in wide_:
            return None
        wide_ = wide_.dropna(subset=[xa, ya])
        sx_, sy_ = wide_[xa].astype(bool), wide_[ya].astype(bool)
        wide_["cls"] = np.select([sx_ & sy_, sx_, sy_], ["both", "x only", "y only"], "neither")
        return wide_

    for xa, ya in SCATTER_PAIRS:
        if xa not in ANALYSIS_NAMES or ya not in ANALYSIS_NAMES:
            continue
        ct_ = class_table(xa, ya)
        if ct_ is None or ct_.empty:
            continue
        for c in CLS:
            ct_[f"is_{c}"] = (ct_["cls"] == c).astype(float)
        fig, axes = plt.subplots(1, len(windows), figsize=(4.6 * len(windows), 3.8), squeeze=False, sharey=True)
        for ax, w in zip(axes[0], windows):
            dw = ct_[ct_["window"] == w]
            tick_labels = []
            for gi, g in enumerate(SIX_GROUP_ORDER):
                dg = dw[dw["group"] == group_key(g)]
                per = dg.groupby(ukeys_)[[f"is_{c}" for c in CLS]].agg(["mean", "size"]) if ukeys_ else None
                if ukeys_:
                    per = per[per[("is_neither", "size")] >= MIN_CELLS_PER_UNIT]
                    n_u = len(per)
                    mean = np.array([per[(f"is_{c}", "mean")].mean() for c in CLS]) if n_u else np.full(len(CLS), np.nan)
                    sem = (np.array([per[(f"is_{c}", "mean")].std(ddof=1) for c in CLS]) / np.sqrt(n_u)
                           if n_u > 1 else np.zeros(len(CLS)))
                    tick_labels.append(f"{SIX_GROUP_LABELS[g]}\n(n={n_u} {UNITS})")
                else:
                    n_u = len(dg)
                    mean = np.array([dg[f"is_{c}"].mean() for c in CLS]) if n_u else np.full(len(CLS), np.nan)
                    sem = np.zeros(len(CLS))
                    tick_labels.append(f"{SIX_GROUP_LABELS[g]}\n(n={n_u})")
                if not np.isfinite(mean).any():
                    continue
                bottom = 0.0
                for c, m_, e_ in zip(CLS, mean, sem):
                    ax.bar(gi, m_, 0.75, bottom=bottom, color=CLS_COL[c], edgecolor="white", lw=0.5)
                    if m_ >= 0.06:
                        ax.text(gi, bottom + m_ / 2, f"{m_:.0%}", ha="center", va="center", fontsize=6,
                                color="white" if c != "neither" else "black")
                    if e_ > 0:
                        ax.errorbar(gi + 0.42, bottom + m_, e_, color="k", lw=0.7, capsize=1.5)
                    bottom += m_
            ax.set_xticks(range(len(SIX_GROUP_ORDER)))
            ax.set_xticklabels(tick_labels, fontsize=6)
            ax.set_title(w, fontsize=9)
            ax.set_ylim(0, 1)
        axes[0, 0].set_ylabel("share of cells" + ("" if AGG == "pooled" else f" (mean over {UNITS})"))
        handles_ = [plt.Rectangle((0, 0), 1, 1, color=CLS_COL[c]) for c in CLS]
        axes[0, -1].legend(handles_, [f"neither", f"{xa} only", f"{ya} only", "both"], fontsize=7, frameon=False,
                           loc="upper left", bbox_to_anchor=(1.01, 1))
        fig.suptitle(f"Selectivity classes (significant vs permutation null): {xa} / {ya}; "
                     f"{'all cells pooled' if AGG == 'pooled' else f'error bars: SEM over {UNITS}'}", y=1.03, fontsize=9)
        fig.tight_layout()
        save_fig(fig, figdir, f"06b_roc_class_fractions_{safe(xa)}_vs_{safe(ya)}")
        plt.close(fig)

        if (xa, ya) == ("stim|choice", "choice|stim"):
            names = {"neither": "neither", "x only": f"{xa} only", "y only": f"{ya} only", "both": "both"}
            for w in windows:
                dl = pd.concat([ct_[ct_["window"] == w].assign(klass=names[c], frac=ct_.loc[ct_["window"] == w, f"is_{c}"])
                                for c in CLS], ignore_index=True)
                group_panels(dl, "klass", [names[c] for c in CLS], "frac", AGG, figdir,
                             f"06c_roc_class_by_group_{safe(xa)}_vs_{safe(ya)}_{w}",
                             f"{w}: share of cells per selectivity class; {agg_note}", ylabel="share of cells",
                             ncols=4, mouse_colors=mouse_colors, min_cells_per_unit=MIN_CELLS_PER_UNIT, panel_size=(4.2, 3.2))

    if HEATMAP_WINDOW in WINDOWS:                                   # 07: per-unit heatmap
        ukey = {"pooled": None, "session": "session_id", "mouse": "animal_id"}[AGG]
        h = rs_[rs_["window"] == HEATMAP_WINDOW].copy()
        h["_unit"] = h[ukey] if ukey else "all"
        fig, axes = plt.subplots(1, len(ANALYSIS_NAMES), figsize=(3.6 * len(ANALYSIS_NAMES),
                                 0.35 * h["_unit"].nunique() + 2), squeeze=False)
        for ax, an in zip(axes[0], ANALYSIS_NAMES):
            piv = (h[h["analysis"] == an].groupby(["_unit", "group"])["sig"].mean().unstack("group")
                   .reindex(columns=GROUP_KEYS))
            im = ax.imshow(piv.to_numpy(float), aspect="auto", cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks(range(6))
            ax.set_xticklabels([SIX_GROUP_LABELS[g] for g in SIX_GROUP_ORDER], rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(piv)))
            ax.set_yticklabels(piv.index if ax is axes[0, 0] else [], fontsize=6)
            ax.set_title(an, fontsize=9)
        fig.colorbar(im, ax=axes[0].tolist(), label="fraction significant", shrink=0.8)
        fig.suptitle(f"Fraction of significant cells per {UNIT} -- {HEATMAP_WINDOW} window")
        save_fig(fig, figdir, "07_roc_per_unit_heatmap")
        plt.close(fig)

# --------------------------------------------------------------------- #
# 5. Disengaged-tail test
# --------------------------------------------------------------------- #
if RUN_TAIL:
    rows = [dict(session_id=r["session_id"], animal_id=r["animal_id"], has_tail=r["tail"] is not None,
                 **(r["tail"]["info"] if r["tail"] else {})) for r in results]
    tail_trials = pd.DataFrame(rows)
    tail_trials.to_csv(table_dir / "tail_trials.csv", index=False)
    n_tail = int(tail_trials["has_tail"].sum())
    print(f"\nDisengaged-tail test: {n_tail}/{len(tail_trials)} sessions have a tail "
          f"(>= {args.tail_min_frac:.0%} disengaged, >= {args.tail_min_trials} trials)")
    if n_tail:
        eg.tail_overview_figure({r["session_id"]: r["tail"] for r in results if r["tail"]},
                                {r["session_id"]: r["trials"] for r in results if r["tail"]}, figdir, "08_tail_trials")
        colors = {"engaged_matched": "#1f77b4", "disengaged": "#d62728"}
        kw = dict(series_col="condition", series_order=list(CONDITIONS), series_colors=colors)
        tl = lin_all[lin_all["condition"].isin(CONDITIONS)]
        tl.to_csv(table_dir / "tail_cell_predictor_linear.csv", index=False)
        gtable(tl, ["condition", "window", "predictor"], "tail_linear_by_group", ["r2"], sig_col="significant")
        tm = mv_all[mv_all["condition"].isin(CONDITIONS)] if len(mv_all) else mv_all
        if len(tm):
            tm.to_csv(table_dir / "tail_cell_multivariate_linear.csv", index=False)
            gtable(tm, ["condition", "window"], "tail_multivariate_by_group", ["r2", "adj_r2"], sig_col="significant")
        s1 = tl.groupby(UNIT_KEYS + ["condition", "window", "predictor"])["r2"].mean().reset_index() \
            .groupby(["condition", "window", "predictor"])["r2"].mean().unstack("condition")
        print(f"  pairwise r^2, tail sessions only (mean over {UNITS}):")
        print(s1.round(3).to_string())
        for w in WINDOWS:
            panels(tl, w, "predictor", PREDICTORS, "r2", f"09_tail_linear_r2_{w}", "r²",
                   f"Tail test, {w}: linear r², matched engaged vs. disengaged trials ({n_tail} sessions)", **kw)
        if len(tm):
            panels(tm, None, "window", list(WINDOWS), "adj_r2", "10_tail_multivariate_r2", "adjusted R\u00b2",
                   f"Tail test: multivariate adjusted R\u00b2, matched engaged vs. disengaged ({n_tail} sessions)",
                   hline=0, **kw)
        troc = roc_all[roc_all["condition"].isin(CONDITIONS)] if len(roc_all) else roc_all
        if len(troc):
            troc.to_csv(table_dir / "tail_roc_per_cell.csv", index=False)
            tf = troc[np.isfinite(troc["auc"])].copy()
            tf["sig"] = tf["significant"].astype(float)
            ts_ = tf.groupby(UNIT_KEYS + ["condition", "window", "analysis"])["sig"].mean().reset_index() \
                .groupby(["condition", "window", "analysis"])["sig"].mean().unstack("condition")
            print(f"  ROC fraction significant, tail sessions only (mean over {UNITS}; NaN = not testable):")
            print(ts_.round(3).to_string())
            gtable(tf, ["condition", "window", "analysis"], "tail_roc_by_group", ["si"], sig_col="significant")
            for w in WINDOWS:
                panels(tf, w, "analysis", ANALYSIS_NAMES, "sig", f"11_tail_roc_frac_significant_{w}",
                       "fraction significant", f"Tail test, {w}: ROC fraction significant (dashed = {ALPHA:.0%}); "
                       f"fewer permutations", ncols=4, hline=ALPHA, **kw)
    else:
        print("  no session has a usable tail -- try a lower --tail-min-frac or --tail-min-trials")

print(f"\nWrote tables to {table_dir}, figures to {figdir}")
