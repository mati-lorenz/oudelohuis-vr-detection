# -*- coding: utf-8 -*-
"""
2e_nonlinear_encod
=====================
For the cell-predictor relationships where 2d found a big gap between the mutual information (2c) and what a
LINEAR fit captures, refit allowing nonlinearity and ask whether that closes the gap. DN sessions only.

Which pairs are refit (the same significance / threshold logic as 2d, no third criterion): rows of 2d's
cell_predictor_mi_vs_linear.csv (engaged trials) with a significant MI (mi_p_value < 0.05), a predictor that CAN show
nonlinearity (below), and frac_explained_linearly < NONLINEAR_GAP_THR (2d's LINEAR_SUFFICIENT_FRAC).

Predictors and the nonlinear model
    continuous behavior (cur_runspeed, cur_pupil_area, cur_motionenergy, cur_lick):
        polynomial of degree 2 and 3 in the standardized predictor (r2_poly2, r2_poly3).
    stimulus binnings `stim__<strategy>` with >= 3 levels (zero_to80_above80, pct_0-20_20-80_80-100):
        CATEGORICAL fit (one mean per bin = one-way ANOVA, r2_categorical). A polynomial makes no sense on 3 ordinal
        codes (degree 3 is rank-deficient); the categorical fit is the most any function of the bins could explain
        about the mean, so it answers "is the stimulus effect non-monotonic / non-linear in the bin index".
    NOT refit: choice / correct and 2-level stimulus binnings (zero_vs_nonzero): with two values a line already passes
        through both group means, nothing for curvature to find. A linear-vs-MI gap there is not curvature in the mean
        (typically a variance difference, e.g. the frame-count/precision confound described in 2c).
All nonlinear R^2 are ADJUSTED (plain R^2 can only grow with free parameters); the comparison is adjusted against
adjusted (r2_linear_adj). `poly_improvement` = r2_best_nonlinear_adj - r2_linear_adj, where the best nonlinear model
is degree 3 (continuous) or categorical (discrete). `frac_explained_nonlinear` = the 3-bin MI that the fit predicts
(mi_bits_from_nonlinear = infotheory.linear_reference.expected_binned_mi of sqrt(adjusted R^2), with the fit's prediction
cut like the data: equal-occupancy bins for the polynomial, the category occupancies for the categorical fit, and the
response's own bin occupancies) relative to max(2d's MI reference, that value); `gap_closed` if >= NONLINEAR_GAP_THR.
The prediction is on the SAME 3-bin scale as 2c's MI (see 2d: the continuous -0.5*log2(1-r^2) is not comparable).

Data: the same trials and pair rules as 2c/2d (infotheory.cell_tables.pair_arrays) -- the refit uses exactly the (x, y)
the MI and the linear fit used; the number of trials is checked against 2d's (`n_trials_match`).

Project-wide rules (infotheory/project_config.py)
    * selection : --animals / --sessions restrict the sessions; a restricted run writes to out/selection/<tag>/. 2d's
                  FULL table is read and the candidates of the selected sessions are used.
    * aggregate : --aggregate pooled|session|mouse for the group figures/tables (the six groups; pooled = all refit
                  cells together; session / mouse = one value per unit with >= MIN_CELLS_PER_UNIT refit cells, bar =
                  mean +- SEM over units, dots = units coloured by mouse). Only REFIT cells enter these means (a
                  conditional statement: "among pairs with a linear gap"); candidate counts per group are reported too.
                  Cell-level scatter figure is not aggregated. No violin / box plots.
    * memory    : one session in RAM per worker; per-session caches in store/.
    * binning   : see above (shared pair rules; MI itself comes from 2c's out/).

Outputs
-------
2_pipeline/2e_nonlinear_encod/
    out/    nonlinear_refits.csv           one row per refit pair: r2_linear(_adj), r2_poly2/3 or r2_categorical (adjusted),
                                           r2_best_nonlinear_adj, poly_improvement, frac_explained_nonlinear, gap_closed
            nonlinear_by_group[__<aggregate>].csv   per group (and unit) x window x predictor: n_refit, mean improvement,
                                           fraction of gaps closed
            nonlinear_candidates_by_group[__<aggregate>].csv   candidate counts: refit pairs / all significant pairs
            figures/*.png                  1 scatter linear vs nonlinear; 2 improvement; 3 fraction of gaps closed
    out/selection/<tag>/...
    store/  refit_session_<hash>.pkl       per-session caches
"""
from __future__ import annotations

import argparse
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from infotheory import project_config as cfg
from infotheory.pipeline import get_pipeline_paths
from infotheory.session import discover_sessions
from infotheory.neural_encoding import build_session_tables
from infotheory.info_theory import multivariate_linear_fit
from infotheory.linear_reference import expected_binned_mi, str_to_props
from infotheory.celldata_utils import SIX_GROUP_ORDER
from infotheory.groups import group_key
from infotheory.plotting import set_style, save_fig, clear_figures
from infotheory.scope import add_scope_args, scope_from_args, select_sessions, scoped_dirs, data_stamp, unit_noun
from infotheory.session_jobs import run_sessions
from infotheory.unit_plots import group_panels, mouse_palette
from infotheory.stim_binning import STIM_BINNING_STRATEGIES, target_signature
from infotheory import cell_tables as ct

PROTOCOL = "DN"
WINDOWS = {"pre_stim": (-30.0, -10.0), "stim": (0.0, 20.0), "reward": (25.0, 45.0)}   # as 2c/2d
CONTINUOUS_PREDICTORS = ["cur_runspeed", "cur_pupil_area", "cur_motionenergy", "cur_lick"]
NONLINEAR_GAP_THR = 0.8            # = 2d's LINEAR_SUFFICIENT_FRAC
MAX_POLY_DEGREE = 3
MIN_CELLS_PER_UNIT = 5
MIN_P = 0.05
JOBLIB_VERBOSE = 5

parser = argparse.ArgumentParser()
add_scope_args(parser)
args = parser.parse_args()
scope = scope_from_args(args)
AGG = scope.aggregate
UNIT = unit_noun(AGG)
UNITS = "mice" if UNIT == "mouse" else f"{UNIT}s"
N_JOBS = scope.n_jobs
AGG_SUFFIX = "" if AGG == "pooled" else f"__{AGG}"
UNIT_KEYS = {"pooled": [], "session": ["session_id"], "mouse": ["animal_id"]}[AGG]

paths = get_pipeline_paths(__file__)
table_dir, figdir = scoped_dirs(paths, scope)
clear_figures(figdir)
set_style()
print(scope.describe())

# --------------------------------------------------------------------- #
# 1. 2d's comparison table -> refit candidates (selected sessions)
# --------------------------------------------------------------------- #
linear_out = paths.out_from("2d_linear_encod")
comparison_path = linear_out / "cell_predictor_mi_vs_linear.csv"
fits_path = paths.out_from("1b_psychometric") / "psychometric_fits.csv"
for f, hint in [(comparison_path, "2d_linear_encod.py"), (fits_path, "1b_psychometric.py")]:
    if not f.exists():
        raise SystemExit(f"Run {hint} first -- expected {f}")
comparison = pd.read_csv(comparison_path)
all_triples = discover_sessions([PROTOCOL])
triples = select_sessions(all_triples, scope)
selected = {f"{a}_{d}": (p, a, d) for p, a, d in triples}
comparison = comparison[comparison["session_id"].isin(selected)]
mouse_colors = mouse_palette(a for _, a, _ in all_triples)


def refit_family(pred):
    """'poly' (continuous), 'categorical' (>= 3 stimulus bins) or None (cannot show nonlinearity)."""
    if pred in CONTINUOUS_PREDICTORS:
        return "poly"
    if pred.startswith(ct.STIM_PREFIX):
        name = pred[len(ct.STIM_PREFIX):]
        rule = STIM_BINNING_STRATEGIES.get(name)
        if rule is not None and len(__import__("infotheory.stim_binning", fromlist=["x"]).strategy_names(rule)) >= 3:
            return "categorical"
    return None


comparison["family"] = comparison["predictor"].map(refit_family)
sig = comparison[(comparison["mi_p_value"] < MIN_P) & comparison["family"].notna()]
candidates = sig[sig["frac_explained_linearly"] < NONLINEAR_GAP_THR].copy()
print(f"{len(candidates)}/{len(sig)} significant (cell, predictor, window) pairs with a refit-able predictor are candidates "
      f"(linear captures <{NONLINEAR_GAP_THR:.0%} of the MI); {len(comparison)} pairs in 2d's table for the selected sessions")
if candidates.empty:
    print("\nNothing to refit -- every significant relationship is already well captured linearly (or none is "
          "significant). No nonlinear analysis needed for this dataset/threshold.")
    raise SystemExit(0)

strategies = sorted({p[len(ct.STIM_PREFIX):] for p in candidates["predictor"] if p.startswith(ct.STIM_PREFIX)})
cand_by_session = {sid: sub for sid, sub in candidates.groupby("session_id")}
session_ids = sorted(cand_by_session)
cells_by_session = {sid: sorted(sub["cell_id"].unique()) for sid, sub in cand_by_session.items()}
mu_sigma_lookup = pd.read_csv(fits_path).set_index("session_id")[["mu", "sigma"]].to_dict("index")
stamps = {s: data_stamp(*selected[s]) for s in session_ids}
cand_hash = {s: hashlib.sha1("|".join(sorted(f"{c}/{w}/{p}" for c, w, p in
                                                  cand_by_session[s][["cell_id", "window", "predictor"]].itertuples(index=False))
                                      ).encode()).hexdigest()[:10] for s in session_ids}


# --------------------------------------------------------------------- #
# 2. Refit (one session in RAM at a time)
# --------------------------------------------------------------------- #
def adj_r2(design, y):
    fit = multivariate_linear_fit(design, y)
    return fit.adj_r2


def process_session(ses, seed):
    sid = ses.session_id
    cand = cand_by_session[sid]
    tables = build_session_tables(ses, mu_sigma_lookup, WINDOWS, included_cell_ids=set(cand["cell_id"]))
    if not tables:
        return []
    tables = {w: ct.add_stimulus_columns(t, strategies) for w, t in tables.items()}
    rows = []
    for _, r in cand.iterrows():
        table = tables.get(r["window"])
        if table is None:
            continue
        pair = ct.pair_arrays(table, r["predictor"], r["cell_id"])
        if pair is None:
            continue
        x, y, _ = pair
        rec = dict(session_id=sid, window=r["window"], cell_id=r["cell_id"], predictor=r["predictor"],
                   family=r["family"], n_trials=len(x), n_trials_2d=r.get("n_trials", np.nan),
                   r2_linear=r["r2"], mi_reference_bits=r["mi_reference_bits"], mi_p_value=r["mi_p_value"],
                   frac_explained_linearly=r["frac_explained_linearly"], y_props=r["y_props"], x_props=r["x_props"])
        rec["r2_linear_adj"] = adj_r2(x, y)
        if r["family"] == "poly":
            xz = (x - x.mean()) / x.std()
            design = xz[:, None]
            for deg in range(2, MAX_POLY_DEGREE + 1):
                design = np.column_stack([design, xz ** deg])
                rec[f"r2_poly{deg}"] = adj_r2(design, y)
            rec["r2_best_nonlinear_adj"] = rec[f"r2_poly{MAX_POLY_DEGREE}"]
        else:
            levels = np.unique(x)
            design = np.column_stack([(x == lv).astype(float) for lv in levels[1:]])
            rec["r2_categorical"] = adj_r2(design, y)
            rec["r2_best_nonlinear_adj"] = rec["r2_categorical"]
        rows.append(rec)
    return rows


results = run_sessions(
    session_ids, process_session, [PROTOCOL], paths.store, "refit_session",
    key_params=dict(version="2e-v3", bins=cfg.RESPONSE_BINS, windows=WINDOWS, strategies=target_signature(strategies), deg=MAX_POLY_DEGREE,
                    min_trials=cfg.MIN_TRIALS_ANALYSIS, min_per_bin=cfg.MIN_TRIALS_PER_TARGET_BIN,
                    thr=NONLINEAR_GAP_THR),
    load_kwargs=dict(load_behaviordata=True, load_videodata=True, load_celldata=True, load_calciumdata=True),
    cells_by_session=cells_by_session, n_jobs=N_JOBS, recompute=args.recompute, verbose=JOBLIB_VERBOSE,
    extra_key_by_session={s: f"{stamps[s]}-{cand_hash[s]}" for s in session_ids}, flatten=True)
refit = pd.DataFrame(results)
if refit.empty:
    print("\nNo candidate could be refit (too few valid trials) -- nothing further to report.")
    raise SystemExit(0)

def predicted_binned_mi(row):
    """3-bin MI the best nonlinear fit predicts (see docstring)."""
    r2 = max(row["r2_best_nonlinear_adj"], 0.0)
    xp = np.ones(cfg.RESPONSE_BINS) / cfg.RESPONSE_BINS if row["family"] == "poly" else str_to_props(row["x_props"])
    return expected_binned_mi(np.sqrt(r2), xp, str_to_props(row["y_props"]))


refit["mi_bits_from_nonlinear"] = refit.apply(predicted_binned_mi, axis=1)
ref_ = np.fmax(refit["mi_reference_bits"], refit["mi_bits_from_nonlinear"])
refit["frac_explained_nonlinear"] = np.where(ref_ > 1e-9, refit["mi_bits_from_nonlinear"] / ref_, np.nan)
refit["gap_closed"] = refit["frac_explained_nonlinear"] >= NONLINEAR_GAP_THR
refit["poly_improvement"] = refit["r2_best_nonlinear_adj"] - refit["r2_linear_adj"]
refit["n_trials_match"] = refit["n_trials"] == refit["n_trials_2d"]
cd = pd.read_csv(paths.out_from("2a_cell_distribution") / "celldata_combined.csv", low_memory=False)
refit = ct.attach_cell_meta(refit, cd)
refit.to_csv(table_dir / "nonlinear_refits.csv", index=False)

print(f"\n{len(refit)} pairs refit ({refit['n_trials_match'].mean():.1%} with exactly the trial count of 2d). "
      f"Nonlinear (adjusted R²) vs linear (adjusted R²):")
per = refit.groupby(UNIT_KEYS + ["predictor"]).agg(imp=("poly_improvement", "median"), closed=("gap_closed", "mean")).reset_index()
tab = per.groupby("predictor")[["imp", "closed"]].mean().rename(columns={"imp": "median improvement", "closed": "fraction gap closed"})
print(f"(mean over {UNITS} of the per-{UNIT} value)" if UNIT_KEYS else "(pooled)")
print(tab.round(4).to_string())

# group tables
ct.unit_group_summary(refit, ["window", "predictor"], AGG, ["poly_improvement", "r2_linear_adj", "r2_best_nonlinear_adj",
                      "frac_explained_nonlinear"], sig_col="gap_closed").rename(
    columns={"n_sig": "n_gap_closed", "frac_sig": "frac_gap_closed", "n_cells": "n_refit"}).to_csv(
    table_dir / f"nonlinear_by_group{AGG_SUFFIX}.csv", index=False)
cmp_all = ct.attach_cell_meta(comparison[(comparison["mi_p_value"] < MIN_P) & comparison["family"].notna()]
                              .assign(is_candidate=lambda d: d["frac_explained_linearly"] < NONLINEAR_GAP_THR), cd)
ct.unit_group_summary(cmp_all, ["window", "predictor"], AGG, [], sig_col="is_candidate").rename(
    columns={"n_sig": "n_candidates", "frac_sig": "frac_candidates", "n_cells": "n_significant_pairs"}).to_csv(
    table_dir / f"nonlinear_candidates_by_group{AGG_SUFFIX}.csv", index=False)

# --------------------------------------------------------------------- #
# 3. Figures
# --------------------------------------------------------------------- #
agg_note = {"pooled": "all refit cells pooled (no error bars)", "session": "bar = mean +- SEM over sessions, dots = sessions",
            "mouse": "bar = mean +- SEM over mice, dots = mice"}[AGG]
preds = [p for p in ct.predictor_list(strategies) if (refit["predictor"] == p).any()]

fig, ax = plt.subplots(figsize=(5.4, 5.2))
for p_, c_ in zip(preds, plt.cm.tab10(np.arange(len(preds)))):
    s = refit[refit["predictor"] == p_]
    ax.scatter(s["r2_linear_adj"], s["r2_best_nonlinear_adj"], s=10, alpha=0.5, color=c_, label=p_)
lo = min(0, refit["r2_linear_adj"].min(), refit["r2_best_nonlinear_adj"].min())
hi = max(refit["r2_linear_adj"].max(), refit["r2_best_nonlinear_adj"].max()) * 1.1 + 1e-6
ax.plot([lo, hi], [lo, hi], color="grey", ls="--", lw=1)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("linear adjusted r²")
ax.set_ylabel("best nonlinear adjusted r² (poly 3 / categorical)")
ax.set_title(f"Does allowing nonlinearity help? {len(refit)} refit pairs (cells, not aggregated)", fontsize=9)
ax.legend(fontsize=6, frameon=False, markerscale=2)
fig.tight_layout()
save_fig(fig, figdir, "1_linear_vs_nonlinear_r2")
plt.close(fig)

for w in WINDOWS:
    sub = refit[refit["window"] == w]
    if sub.empty:
        continue
    group_panels(sub, "predictor", preds, "poly_improvement", AGG, figdir, f"2_improvement_by_group_{w}",
                 f"Improvement from nonlinearity (adj. r² nonlinear - linear), {w}; refit pairs only; {agg_note}",
                 ylabel="Δ adjusted r²", hline=0, mouse_colors=mouse_colors, min_cells_per_unit=MIN_CELLS_PER_UNIT)
    group_panels(sub.assign(closed=sub["gap_closed"].astype(float)), "predictor", preds, "closed", AGG, figdir,
                 f"3_gap_closed_by_group_{w}",
                 f"Fraction of linear gaps closed by the nonlinear fit (>= {NONLINEAR_GAP_THR:.0%} of MI), {w}; {agg_note}",
                 ylabel="fraction closed", mouse_colors=mouse_colors, min_cells_per_unit=MIN_CELLS_PER_UNIT)

print(f"\nWrote tables to {table_dir}, figures to {figdir}")
