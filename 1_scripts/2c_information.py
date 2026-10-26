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

Project-wide rules (infotheory/project_config.py) -- what applies here
    * selection : --animals / --sessions restrict the sessions; a restricted run writes to out/selection/<tag>/
                  and NEVER overwrites the tables in out/ that 2d (and later steps) read.
    * aggregate : --aggregate pooled|session|mouse = the unit for the group comparisons (bars of MI, fraction of
                  significant cells): pooled = all cells of a group together, no error bars; session / mouse = one
                  value per session / mouse (MIN_CELLS_PER_UNIT cells in the group needed), bar = mean +- SEM over
                  units, dots = units coloured by mouse. Always the six groups. No violin plots.
    * memory    : one session in RAM per worker (--n-jobs), reduced to small tables and cached per session in
                  store/. The dataset is never loaded at once.
    * binning   : the information measures here and in 2d use the SAME trials and the SAME bins, so they agree
                  before bias correction: per-pair trials from infotheory.cell_tables.pair_arrays (finite values,
                  stimulus/choice bins with >= MIN_TRIALS_PER_TARGET_BIN trials, >= MIN_TRIALS_ANALYSIS trials),
                  RESPONSE_BINS equal-occupancy bins for the cell response AND for continuous predictors, stimulus
                  predictors from stim_binning (one per strategy, `stim__<strategy>`), choice and correct as-is.
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
            mi_by_group[__<aggregate>].csv per group (and unit) x window x predictor: n_cells, mean MI, fraction significant
            confound_check[__<aggregate>].csv   cur_runspeed vs choice / correct fraction significant
            tail_trials.csv               per session: tail found?, start, n, fraction disengaged, control trials
            tail_cell_predictor_mi.csv    the same MI table for the tail test, column `condition`
                                          (engaged_matched | disengaged)
            tail_mi_by_group[__<aggregate>].csv
            figures/*.png                 1 MI by predictor, 2 fraction significant, 3 heatmap; 4-6 tail test
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
from infotheory.neural_encoding import load_cell_inclusion, build_session_tables
from infotheory.info_theory import mutual_information_shuffle
from infotheory.celldata_utils import SIX_GROUP_ORDER, SIX_GROUP_LABELS, SIX_GROUP_COLORS
from infotheory.groups import group_key
from infotheory.plotting import set_style, save_fig, clear_figures
from infotheory.scope import add_scope_args, scope_from_args, select_sessions, scoped_dirs, data_stamp, unit_noun
from infotheory.session_jobs import run_sessions
from infotheory.unit_stats import stable_seed
from infotheory.unit_plots import group_panels, mouse_palette
from infotheory.stim_binning import STIM_BINNING_STRATEGIES, DEFAULT_STRATEGIES, target_signature, describe_strategies
from infotheory import cell_tables as ct
from infotheory import engagement as eg
from infotheory.conditional_info import mi_and_conditional_mi

PROTOCOL = "DN"

# Response windows: [start_cm, end_cm] relative to stimStart (a position), as in 1c_behavior.MI_WINDOWS
WINDOWS = {"pre_stim": (-30.0, -10.0), "stim": (0.0, 20.0), "reward": (25.0, 45.0)}

BINNING_STRATEGIES = list(DEFAULT_STRATEGIES)   # stimulus predictors (override with --strategies)
N_SHUFFLES = 20                                 # shuffles for bias correction + p-value (compute budget)
MIN_CELLS_PER_UNIT = 5                          # session / mouse aggregate: cells needed in a group for a unit's value
ALPHA = 0.05
RANDOM_STATE = 0
RUN_TAIL = True
JOBLIB_VERBOSE = 5
CONDITIONS = ("engaged_matched", "disengaged")  # tail test; "engaged" (all engaged trials) is the main analysis

parser = argparse.ArgumentParser()
add_scope_args(parser)
parser.add_argument("--strategies", nargs="+", default=None, choices=sorted(STIM_BINNING_STRATEGIES),
                    help=f"stimulus binning strategies (default {BINNING_STRATEGIES})")
parser.add_argument("--n-shuffles", type=int, default=N_SHUFFLES)
parser.add_argument("--no-tail", action="store_true", help="skip the disengaged-tail test")
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
GIVEN = "cur_runspeed"   # the movement variable that is conditioned out (same-window mean running speed)
BINS = cfg.RESPONSE_BINS

paths = get_pipeline_paths(__file__)
table_dir, figdir = scoped_dirs(paths, scope)
clear_figures(figdir)
set_style()
print(scope.describe())
print(f"Stimulus predictors:\n{describe_strategies(STRATEGIES)}")
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
cells_by_session = {sid: sorted(sub.loc[sub["include"], "cell_id"]) for sid, sub in cell_inclusion.groupby("session_id")
                    if sid in session_ids}
session_ids = [s for s in session_ids if cells_by_session.get(s)]
if not session_ids:
    raise SystemExit("No included cells in the selected sessions.")
print(f"{len(session_ids)} sessions, {sum(len(v) for v in cells_by_session.values())} included cells "
      f"(passing 2a proximity AND 2b QC)")
stamps = {s: data_stamp(*selected[s]) for s in session_ids}


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


def process_session(ses, seed):
    sid = ses.session_id
    cells = set(cells_by_session.get(sid, []))
    td = ses.trialdata
    if not cells or td is None or "engaged" not in td.columns:
        return None
    engaged_nums = td.loc[td["engaged"] == 1, "trialNumber"].astype(int).tolist()
    tail = eg.tail_condition_trials(td, args.tail_min_frac, args.tail_min_trials) if RUN_TAIL else None
    wanted = set(engaged_nums) | (set(tail["disengaged"]) if tail else set())
    tables = build_session_tables(ses, mu_sigma_lookup, WINDOWS, included_cell_ids=cells, trial_numbers=wanted)
    if not tables:
        return None
    sets = {"engaged": set(engaged_nums)}
    if tail:
        sets.update({"engaged_matched": set(tail["engaged_matched"]), "disengaged": set(tail["disengaged"])})
    recs, crecs = [], []
    for window, table in tables.items():
        if table.empty:
            continue
        cell_cols = ct.cell_columns(table, cells)
        for cond, nums in sets.items():
            sub = table[table["trialNumber"].isin(nums)]
            if len(sub) < cfg.MIN_TRIALS_ANALYSIS:
                continue
            sub = ct.add_stimulus_columns(sub, STRATEGIES)
            recs += mi_records(sub, cond, window, cell_cols, sid, seed)
            if RUN_GIVEN and cond == "engaged":
                crecs += cmi_records(sub, window, cell_cols, sid, seed)
    return dict(session_id=sid, animal_id=ses.animal_id, records=pd.DataFrame(recs), cmi=pd.DataFrame(crecs),
                tail=tail, trials=td[["trialNumber", "engaged"]].copy(), n_cells=len(cells))


results = run_sessions(
    session_ids, process_session, [PROTOCOL], paths.store, "mi_session",
    key_params=dict(version="2c-v3", given=[RUN_GIVEN, GIVEN], windows=WINDOWS, strategies=target_signature(STRATEGIES),
                    bins=BINS, method=cfg.RESPONSE_BIN_METHOD, n_shuffles=N_SHUF,
                    min_trials=cfg.MIN_TRIALS_ANALYSIS, min_per_bin=cfg.MIN_TRIALS_PER_TARGET_BIN,
                    tail=[RUN_TAIL, args.tail_min_frac, args.tail_min_trials],
                    fits_mtime=int(fits_path.stat().st_mtime)),
    load_kwargs=dict(load_behaviordata=True, load_videodata=True, load_celldata=True, load_calciumdata=True),
    cells_by_session=cells_by_session, base_seed=RANDOM_STATE, n_jobs=N_JOBS, recompute=args.recompute,
    verbose=JOBLIB_VERBOSE, extra_key_by_session=stamps, flatten=False)
results = [r for r in results if r is not None]
if not results:
    raise SystemExit("No session produced tables (no celldata / calciumdata / behaviordata?).")

all_rec = pd.concat([r["records"] for r in results], ignore_index=True)
if all_rec.empty:
    raise SystemExit("No (cell, predictor) pair had enough valid trials.")
cd = pd.read_csv(cell_dist_out / "celldata_combined.csv", low_memory=False)
all_rec = ct.attach_cell_meta(all_rec, cd)
all_rec["significant"] = all_rec["mi_p_value"] < ALPHA

mi_table = all_rec[all_rec["condition"] == "engaged"].drop(columns="condition")
mi_table.to_csv(table_dir / "cell_predictor_mi.csv", index=False)
print(f"\n{len(mi_table)} (session, window, cell, predictor) MI values (engaged trials)")

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
    group_panels(df[df["window"] == window], "predictor", PREDICTORS, value, AGG, figdir, name, title,
                 ylabel=ylabel, hline=hline, series_col=series_col, series_order=series_order,
                 series_colors=series_colors, mouse_colors=mouse_colors, min_cells_per_unit=MIN_CELLS_PER_UNIT)


agg_note = {"pooled": "all cells pooled (no error bars)",
            "session": "bar = mean +- SEM over sessions, dots = sessions",
            "mouse": "bar = mean +- SEM over mice, dots = mice"}[AGG]
for w in WINDOWS:
    predictor_panels(mi_table, w, "mi_bits_corrected", f"1_mi_by_predictor_{w}", "MI (bits, bias-corrected)",
                     f"Per-cell MI with each predictor -- {w} window ({PROTOCOL}); {agg_note}")
    mi_table_sig = mi_table.assign(sig=mi_table["significant"].astype(float))
    predictor_panels(mi_table_sig, w, "sig", f"2_frac_significant_{w}", f"fraction of cells p<{ALPHA}",
                     f"Fraction of cells with significant MI -- {w} window; dashed = chance ({ALPHA}); {agg_note}",
                     hline=ALPHA)

piv = frac.unstack("window").reindex(index=PREDICTORS, columns=[w for w in WINDOWS if w in set(mi_table["window"])])
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
