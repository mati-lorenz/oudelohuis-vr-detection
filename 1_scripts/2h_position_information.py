# -*- coding: utf-8 -*-
"""
2h_position_information
=======================
Single-cell information over POSITION relative to stimulus onset ("information PSTH"): how much each cell's activity
tells about the stimulus, the choice and the behavioural variables, at every position bin of the corridor, computed
with TWO ways of aggregating the time points of a bin (both shown, side by side):

    trial_mean    one sample per TRIAL: the trial's mean activity (and mean behaviour) over its frames in the bin. Variables
                  = the trial's stimulus bin / choice (stim__<strategy>, choice: the targets of 2c-3x, same trial rule and
                  RESPONSE_BINS) and the trial-bin mean of runspeed, pupil_area, motionenergy, lick.
    frame_pooled  one sample per FRAME: all calcium frames of all engaged trials that fall in the bin. Behaviour = the
                  frame's own value; stimulus / choice = the label of the frame's trial (so a trial's frames share a label;
                  frames of one trial are not independent: the curve is descriptive, the p-values are optimistic).

Both use the same frames, position bins (infotheory.position_info) and the 2c estimator: histogram MI with RESPONSE_BINS (3)
quantile bins, bias-corrected by subtracting the mean of N_SHUFFLES shuffles, p = fraction of shuffles >= raw (with 20
shuffles the null "significant" fraction is about 5%). Behaviour is taken at ZERO lag (2f searches the optimal lag).
Engaged trials only. Always the six groups.

Project-wide rules (infotheory/project_config.py) -- what applies here
    * selection : --animals / --sessions restrict the sessions; a restricted run writes to out/selection/<tag>/.
    * aggregate : --aggregate pooled|session|mouse. Curves = per group, method, variable and position bin, the mean over
                  units of the unit's mean over its cells (units with < MIN_CELLS_PER_UNIT cells left out), band = SEM over
                  units. Pooled: mean over all cells, band = SEM over cells (cells treated as independent; cells of one
                  session share trials and animal, so this band is narrower than the session / mouse one).
    * memory    : one session in RAM per worker (--n-jobs); per-bin means are computed from the frame table without
                  long-format copies; results cached per session in store/.
    * binning   : stimulus strategies / choice as 2c (--strategies), RESPONSE_BINS shared with 2c-3x.

INFORMATIVE CELLS (tag): besides the mean over ALL cells, every cell that codes a variable better than chance is tagged,
separately for each method x variable. A bin is significant when p < ALPHA (shuffle null, as above). A cell is INFORMATIVE
when at least MIN_CONSEC_BINS (default 2, --min-consec-bins) consecutive bins inside TRUSTED_POSITION_RANGE are significant:
a single bin at p<0.05 is a false positive 5% of the time, a run of 2 reduces the expected false-tag rate to about
n_bins x 0.05^2 (about 2% for 9 bins; adjacent bins are correlated, so it is approximate; the script prints the observed
rate of tagged cells next to it). Tagged cells are drawn solid on top of the all-cells mean (dashed), per group; the fraction of
informative cells per group x session is also written. CAVEAT: cells are selected with the same data they are plotted on, so
their curves are biased upward; they show the shape / location of the tuning of the cells that carry the information, not an
unbiased effect size. The tag table (informative_cells.csv) can be read by other steps.

KNOWN CAVEAT (as 2f): positions beyond about +-40 cm from stimulus onset can show an elevated MI signature that is similar
across variables and groups -- probably cross-trial contamination (zpos never resets, a far bin can fall in the previous
trial's reward / ITI). The figures shade the range outside TRUSTED_POSITION_RANGE. Behaviour vs activity at zero lag only.

Outputs
-------
2_pipeline/2h_position_information/
    out/    position_information_per_cell.csv       session, cell, group, method, variable, s_rel, n_samples, n_trials,
                                                    mi_bits, p_value
            position_information_by_group[__<aggregate>].csv   per group x method x variable x s_rel (x unit): n_cells,
                                                    mean MI, fraction p<0.05
            figures/<method>/1_task_variables.png, 2_behavior_variables.png, 3_fraction_significant.png
            figures/4_method_comparison.png         trial_mean vs frame_pooled, all cells
            informative_cells.csv                   tag: session, cell, group, method, variable, n_sig_bins, max_run, informative
            informative_fraction_by_group[__<aggregate>].csv   fraction of informative cells per group (x unit) x method x variable
            position_information_informative_by_group[__<aggregate>].csv   MI curve of the informative cells only
            figures/<method>/5_informative_cells_task_variables.png, 6_informative_cells_behavior_variables.png,
                             7_fraction_informative.png
    out/selection/<tag>/...                         the same for a restricted run
    store/  position_info_<hash>.pkl                per-session caches
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from infotheory import project_config as cfg
from infotheory.pipeline import get_pipeline_paths
from infotheory.session import discover_sessions
from infotheory.neural_encoding import load_cell_inclusion, build_session_tables, BEHAVIOR_PREDICTORS
from infotheory.continuous import (merge_behavior_video, restrict_to_engaged, remove_pupil_outliers,
                                   build_calcium_continuous)
from infotheory.celldata_utils import SIX_GROUP_ORDER, SIX_GROUP_LABELS, SIX_GROUP_COLORS
from infotheory.groups import add_six_group_column, group_key
from infotheory.plotting import set_style, save_fig, clear_figures
from infotheory.scope import add_scope_args, scope_from_args, select_sessions, scoped_dirs, data_stamp, unit_noun
from infotheory.session_jobs import run_sessions
from infotheory.unit_stats import stable_seed
from infotheory.unit_plots import mouse_palette
from infotheory.stim_binning import (STIM_BINNING_STRATEGIES, DEFAULT_STRATEGIES, CHOICE_TARGET, target_signature,
                                     describe_strategies)
from infotheory import pair_tables as pt
from infotheory import position_info as pi

PROTOCOL = "DN"
S_PRE, S_POST, BINSIZE = -60.0, 80.0, 5.0       # cm relative to stimulus onset
TRUSTED_POSITION_RANGE = (-40.0, 40.0)
N_SHUFFLES = 20
MIN_FRAMES_PER_BIN = 100                         # frame_pooled: samples needed in a bin
MIN_CELLS_PER_UNIT = 5
ALPHA = 0.05
MIN_CONSEC_BINS = 2                              # consecutive significant bins to tag a cell informative
RANDOM_STATE = 0
JOBLIB_VERBOSE = 5
BEHAVIOR = list(BEHAVIOR_PREDICTORS)             # runspeed, pupil_area, motionenergy, lick
METHODS = ["trial_mean", "frame_pooled"]
METHOD_STYLE = {"trial_mean": dict(ls="-"), "frame_pooled": dict(ls="--")}

parser = argparse.ArgumentParser()
add_scope_args(parser)
parser.add_argument("--strategies", nargs="+", default=None, choices=sorted(STIM_BINNING_STRATEGIES))
parser.add_argument("--n-shuffles", type=int, default=N_SHUFFLES)
parser.add_argument("--min-consec-bins", type=int, default=MIN_CONSEC_BINS,
                    help="consecutive significant bins (trusted range) needed to tag a cell as informative")
args = parser.parse_args()
scope = scope_from_args(args)
AGG = scope.aggregate
UNIT = unit_noun(AGG)
AGG_SUFFIX = "" if AGG == "pooled" else f"__{AGG}"
MIN_C = 1 if AGG == "pooled" else MIN_CELLS_PER_UNIT
STRATEGIES = args.strategies or list(DEFAULT_STRATEGIES)
TASK_VARS = [pt.target_column(t) for t in STRATEGIES + [CHOICE_TARGET]]      # stim__<strategy>..., choice
GROUP_KEYS = [group_key(g) for g in SIX_GROUP_ORDER]
GROUP_COLOR = {group_key(g): SIX_GROUP_COLORS[g] for g in SIX_GROUP_ORDER}
GROUP_LABEL = {group_key(g): SIX_GROUP_LABELS[g] for g in SIX_GROUP_ORDER}

paths = get_pipeline_paths(__file__)
table_dir, figdir = scoped_dirs(paths, scope)
clear_figures(figdir)
for _sub in figdir.glob("*/"):
    clear_figures(_sub)
set_style()
print(scope.describe())
print("Stimulus binning strategies:\n" + describe_strategies(STRATEGIES))

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
mouse_colors = mouse_palette(a for _, a, _ in all_triples)
cell_inclusion = load_cell_inclusion(cell_dist_out, activity_out)
cells_by_session = {sid: sorted(sub.loc[sub["include"], "cell_id"]) for sid, sub in cell_inclusion.groupby("session_id")
                    if sid in session_ids}
session_ids = [s for s in session_ids if cells_by_session.get(s)]
if not session_ids:
    raise SystemExit("No selected session with included cells.")
cd = add_six_group_column(pd.read_csv(cell_dist_out / "celldata_combined.csv", low_memory=False)[
    ["session_id", "cell_id", "roi_name", "labeled"]])
group_of = dict(zip(zip(cd["session_id"], cd["cell_id"]), cd["group"]))
stamps = {s: data_stamp(*selected[s]) for s in session_ids}
print(f"{len(session_ids)} sessions, {sum(len(v) for v in cells_by_session.values())} included cells")


# --------------------------------------------------------------------- #
# 2. Per session
# --------------------------------------------------------------------- #
def process_session(ses, seed):
    sid = ses.session_id
    cells = cells_by_session[sid]
    if ses.trialdata is None or ses.behaviordata is None or ses.calciumdata is None or ses.ts_F is None:
        return None
    rng = np.random.default_rng(seed)
    merged = remove_pupil_outliers(merge_behavior_video(ses.behaviordata, ses.videodata))
    behav = [b for b in BEHAVIOR if b in merged.columns]
    ft = build_calcium_continuous(ses.calciumdata.loc[:, cells], ses.ts_F, merged, extra_columns=behav)
    ft = restrict_to_engaged(ft, ses.trialdata)
    if ft.empty:
        return None
    td = ses.trialdata
    trials = (td[td["engaged"] == 1] if "engaged" in td.columns else td).reset_index(drop=True)
    tabs = build_session_tables(ses, mu_sigma_lookup, {"stim": (0.0, 20.0)}, included_cell_ids={cells[0]})
    if "stim" not in tabs or tabs["stim"].empty:
        return None
    meta = pt.add_target_columns(tabs["stim"], STRATEGIES).set_index("trialNumber")
    lab = {v: trials["trialNumber"].map(meta[v]).to_numpy(float) for v in TASK_VARS if v in meta.columns}
    pos = ft["zpos"].to_numpy(float)
    fi, tr, bb, centers = pi.frame_bin_index(trials["stimStart"].to_numpy(float), pos, S_PRE, S_POST, BINSIZE)
    if len(fi) == 0:
        return None
    nb, nt = len(centers), len(trials)
    A = ft[cells].to_numpy(float)
    B = {b: ft[b].to_numpy(float) for b in behav}
    # trial means of cells + behaviour: (trials, bins, cells+behaviour)
    TM = pi.trial_bin_means(np.column_stack([A] + [B[b] for b in behav]), fi, tr, bb, nt, nb)
    rows = []
    order = np.argsort(bb, kind="stable")
    bounds = np.searchsorted(bb[order], np.arange(nb + 1))
    variables = [(v, True) for v in lab] + [(b, False) for b in behav]
    trial_ids = trials["trialNumber"].to_numpy()
    for b_i, s_rel in enumerate(centers):
        sel = order[bounds[b_i]:bounds[b_i + 1]]
        f_b, t_b = fi[sel], tr[sel]
        for var, discrete in variables:
            # ---- trial_mean: one sample per trial ----
            if discrete:
                x_t = lab[var]
            else:
                x_t = TM[:, b_i, len(cells) + behav.index(var)]
            # ---- frame_pooled: one sample per frame ----
            x_f = lab[var][t_b] if discrete else B[var][f_b]
            for j, c in enumerate(cells):
                for method, x, y, tid in (("trial_mean", x_t, TM[:, b_i, j], trial_ids),
                                          ("frame_pooled", x_f, A[f_b, j], trial_ids[t_b])):
                    if method == "frame_pooled" and len(f_b) < MIN_FRAMES_PER_BIN:
                        continue
                    res = pi.bin_information(x, y, tid, discrete, args.n_shuffles, rng)
                    if res is None:
                        continue
                    rows.append((sid, c, method, var, s_rel) + res)
    df = pd.DataFrame(rows, columns=["session_id", "cell_id", "method", "variable", "s_rel", "n_samples", "n_trials",
                                     "mi_bits", "p_value"])
    print(f"  [{sid}] {len(cells)} cells, {nb} bins, {len(df)} (cell, bin, variable, method) values")
    return df


results = run_sessions(
    session_ids, process_session, [PROTOCOL], paths.store, "position_info",
    key_params=dict(version="2h-v1", window=(S_PRE, S_POST, BINSIZE), bins=cfg.RESPONSE_BINS, shuffles=args.n_shuffles,
                    targets=target_signature(STRATEGIES + [CHOICE_TARGET]), behavior=BEHAVIOR,
                    min_frames=MIN_FRAMES_PER_BIN, min_trials=cfg.MIN_TRIALS_ANALYSIS,
                    min_per_class=cfg.MIN_TRIALS_PER_TARGET_BIN, fits_mtime=int(fits_path.stat().st_mtime)),
    load_kwargs=dict(load_behaviordata=True, load_videodata=True, load_celldata=True, load_calciumdata=True),
    cells_by_session=cells_by_session, base_seed=RANDOM_STATE, n_jobs=scope.n_jobs, recompute=args.recompute,
    verbose=JOBLIB_VERBOSE, extra_key_by_session=stamps, flatten=False)
results = [r for r in results if r is not None and len(r)]
if not results:
    raise SystemExit("No information values (too few trials or frames per position bin).")
tab = pd.concat(results, ignore_index=True)
del results
tab["animal_id"] = tab["session_id"].str.split("_").str[0]
tab["group"] = [group_of.get((s, c), "") for s, c in zip(tab["session_id"], tab["cell_id"])]
tab = tab[tab["group"].isin(GROUP_KEYS)].reset_index(drop=True)
tab["significant"] = (tab["p_value"] < ALPHA).astype(float)
tab["unit"] = {"pooled": "all", "session": tab["session_id"], "mouse": tab["animal_id"]}[AGG]
tab.drop(columns=["unit"]).to_csv(table_dir / "position_information_per_cell.csv", index=False)
print(f"\n{len(tab)} per-cell values; {tab['cell_id'].nunique()} cells")

VARS_ALL = TASK_VARS + BEHAVIOR


def curves(df, value, by=("method", "variable", "group"), min_c=None):
    """Per `by` x s_rel: mean and SEM over units of the unit means over cells (units with < MIN_C cells dropped).
    Pooled: mean and SEM over cells."""
    by = list(by)
    min_c = MIN_C if min_c is None else min_c
    if AGG == "pooled":
        g = df.groupby(by + ["s_rel"], observed=True)[value]
        out = g.mean().to_frame("mean")
        out["sem"] = g.sem().fillna(0.0)
        out["n_units"] = g.size()
        return out.reset_index()
    g = df.groupby(by + ["s_rel", "unit"], observed=True)[value]
    um = g.mean().to_frame("v")
    um["n"] = g.size()
    um = um[um["n"] >= min_c].reset_index()
    g2 = um.groupby(by + ["s_rel"])["v"]
    out = g2.mean().to_frame("mean")
    out["sem"] = g2.sem().fillna(0.0) if AGG != "pooled" else 0.0
    out["n_units"] = g2.size()
    return out.reset_index()


mi_curve = curves(tab, "mi_bits")
sig_curve = curves(tab, "significant")
out = mi_curve.merge(sig_curve[["method", "variable", "group", "s_rel", "mean"]].rename(columns={"mean": "frac_significant"}),
                     on=["method", "variable", "group", "s_rel"], how="left").rename(columns={"mean": "mi_bits"})
out.to_csv(table_dir / f"position_information_by_group{AGG_SUFFIX}.csv", index=False)

agg_note = {"pooled": "mean +- SEM over all cells", "session": "mean +- SEM over sessions",
            "mouse": "mean +- SEM over mice"}[AGG]
VAR_TITLE = {v: v.replace("stim__", "stimulus: ") for v in VARS_ALL}


def shade(ax):
    ax.axvline(0, color="black", ls=":", lw=1)
    ax.axvspan(0, 20, color="grey", alpha=0.12, zorder=0)
    ax.axvspan(25, 45, color="grey", alpha=0.12, zorder=0)
    ax.set_xlabel("position rel. to stimulus onset (cm)")


def lines(ax, cur, method, var, ylabel):
    for g in GROUP_KEYS:
        c = cur[(cur["method"] == method) & (cur["variable"] == var) & (cur["group"] == g)].sort_values("s_rel")
        if c.empty:
            continue
        ax.plot(c["s_rel"], c["mean"], color=GROUP_COLOR[g], label=f"{GROUP_LABEL[g]} (n={int(c['n_units'].max())})")
        ax.fill_between(c["s_rel"], c["mean"] - c["sem"], c["mean"] + c["sem"], color=GROUP_COLOR[g], alpha=0.18, lw=0)
    shade(ax)
    ax.set_title(VAR_TITLE[var], fontsize=9)
    ax.set_ylabel(ylabel)


def family_figure(cur, method, variables, name, ylabel, title, hline=None):
    variables = [v for v in variables if ((cur["method"] == method) & (cur["variable"] == v)).any()]
    if not variables:
        return
    ncols = min(len(variables), 4)
    nrows = int(np.ceil(len(variables) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.6 * nrows), squeeze=False)
    for ax, v in zip(axes.flat, variables):
        lines(ax, cur, method, v, ylabel)
        if hline is not None:
            ax.axhline(hline, color="grey", ls="--", lw=0.7)
    for ax in list(axes.flat)[len(variables):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle(f"{title} -- {method}; {agg_note}", y=1.0, fontsize=9)
    fig.tight_layout()
    d = figdir / method
    d.mkdir(parents=True, exist_ok=True)
    save_fig(fig, d, name)
    plt.close(fig)


for method in METHODS:
    family_figure(mi_curve, method, TASK_VARS, "1_task_variables", "MI (bits, shuffle-corrected)",
                  "Single-cell information about the stimulus and the choice")
    family_figure(mi_curve, method, BEHAVIOR, "2_behavior_variables", "MI (bits, shuffle-corrected)",
                  "Single-cell information about behaviour (zero lag)")
    family_figure(sig_curve.assign(), method, VARS_ALL, "3_fraction_significant", "fraction of cells p<0.05",
                  f"Cells with significant information (dashed = {ALPHA:.0%}; shuffles {args.n_shuffles})", hline=ALPHA)

# ---------------------------------------------------------------------------------------------------------------------
# INFORMATIVE CELLS: tag cells with >= MIN_CONSEC_BINS consecutive significant bins in the trusted range
# ---------------------------------------------------------------------------------------------------------------------
lo, hi = TRUSTED_POSITION_RANGE
core = tab[(tab["s_rel"] >= lo) & (tab["s_rel"] <= hi)].sort_values("s_rel")
KEYS = ["session_id", "group", "method", "variable"]


def longest_run(x):
    best = cur = 0
    for v in x:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


tag = (core.groupby(KEYS + ["cell_id"], observed=True)["significant"]
       .agg(n_sig_bins="sum", n_bins="size", max_run=lambda x: longest_run(x.to_numpy() > 0)).reset_index())
tag["informative"] = tag["max_run"] >= args.min_consec_bins
tag["animal_id"] = tag["session_id"].str.split("_").str[0]
tag.to_csv(table_dir / "informative_cells.csv", index=False)

inf_tab = tab.merge(tag.loc[tag["informative"], KEYS + ["cell_id"]], on=KEYS + ["cell_id"], how="inner")
inf_curve = curves(inf_tab, "mi_bits", min_c=1)
inf_curve.to_csv(table_dir / f"position_information_informative_by_group{AGG_SUFFIX}.csv", index=False)

tag["unit"] = {"pooled": "all", "session": tag["session_id"], "mouse": tag["animal_id"]}[AGG]
fi = tag.assign(v=tag["informative"].astype(float))
g_ = fi.groupby(["method", "variable", "group", "unit"], observed=True)["v"].agg(["mean", "size"]).reset_index()
g_ = g_[g_["size"] >= MIN_C]
frac_inf = (g_.groupby(["method", "variable", "group"], observed=True)["mean"]
            .agg(frac_informative="mean", sem="sem", n_units="size").reset_index())
frac_inf["sem"] = frac_inf["sem"].fillna(0.0)
frac_inf.to_csv(table_dir / f"informative_fraction_by_group{AGG_SUFFIX}.csv", index=False)

nb = int(core.groupby("cell_id")["s_rel"].nunique().median()) if len(core) else 0
exp_fp = nb * ALPHA ** args.min_consec_bins
print(f"\nInformative cells (>= {args.min_consec_bins} consecutive bins p<{ALPHA} in {lo:g}..{hi:g} cm). "
      f"Expected false-tag rate under the null ~ {exp_fp:.1%} (n_bins={nb}). Observed fraction informative, all cells:")
print(tag.groupby(["method", "variable"])["informative"].mean().unstack("method").reindex(VARS_ALL).round(3).to_string())


def inf_figure(method, variables, name, title):
    variables = [v for v in variables if ((inf_curve["method"] == method) & (inf_curve["variable"] == v)).any()]
    if not variables:
        return
    ncols = min(len(variables), 4)
    nrows = int(np.ceil(len(variables) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.6 * nrows), squeeze=False)
    for ax, v in zip(axes.flat, variables):
        for g in GROUP_KEYS:
            sel = lambda cur: cur[(cur["method"] == method) & (cur["variable"] == v) & (cur["group"] == g)].sort_values("s_rel")
            a, t = sel(mi_curve), sel(inf_curve)
            if not a.empty:
                ax.plot(a["s_rel"], a["mean"], color=GROUP_COLOR[g], ls="--", lw=1.0, alpha=0.8)
            if not t.empty:
                ax.plot(t["s_rel"], t["mean"], color=GROUP_COLOR[g], lw=1.8,
                        label=f"{GROUP_LABEL[g]} (n={int(t['n_units'].max())})")
                ax.fill_between(t["s_rel"], t["mean"] - t["sem"], t["mean"] + t["sem"], color=GROUP_COLOR[g], alpha=0.15, lw=0)
        shade(ax)
        ax.set_title(VAR_TITLE[v], fontsize=9)
        ax.set_ylabel("MI (bits, shuffle-corrected)")
    for ax in list(axes.flat)[len(variables):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=6, frameon=False, title="solid = informative cells\ndashed = all cells", title_fontsize=6)
    fig.suptitle(f"{title} -- {method}; informative = >= {args.min_consec_bins} consecutive bins p<{ALPHA} (selected on the "
                 f"same data: biased upward); {agg_note}", y=1.0, fontsize=8)
    fig.tight_layout()
    d = figdir / method
    d.mkdir(parents=True, exist_ok=True)
    save_fig(fig, d, name)
    plt.close(fig)


def frac_figure(method):
    fr = frac_inf[frac_inf["method"] == method]
    variables = [v for v in VARS_ALL if (fr["variable"] == v).any()]
    if not variables:
        return
    fig, ax = plt.subplots(figsize=(1.6 * len(variables) + 2, 3.6))
    w = 0.8 / len(GROUP_KEYS)
    for i, g in enumerate(GROUP_KEYS):
        for j, v in enumerate(variables):
            r = fr[(fr["variable"] == v) & (fr["group"] == g)]
            if r.empty:
                continue
            ax.bar(j + (i - len(GROUP_KEYS) / 2 + 0.5) * w, r["frac_informative"].iloc[0], w, yerr=r["sem"].iloc[0],
                   color=GROUP_COLOR[g], label=GROUP_LABEL[g] if j == 0 else None, error_kw=dict(lw=0.7))
    ax.axhline(exp_fp, color="grey", ls="--", lw=0.8)
    ax.set_xticks(range(len(variables)))
    ax.set_xticklabels([VAR_TITLE[v] for v in variables], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("fraction of informative cells")
    ax.legend(fontsize=6, frameon=False)
    ax.set_title(f"Fraction of informative cells -- {method}; dashed = expected false-tag rate; {agg_note}", fontsize=8)
    fig.tight_layout()
    d = figdir / method
    d.mkdir(parents=True, exist_ok=True)
    save_fig(fig, d, "7_fraction_informative")
    plt.close(fig)


for method in METHODS:
    inf_figure(method, TASK_VARS, "5_informative_cells_task_variables", "Informative cells: stimulus and choice")
    inf_figure(method, BEHAVIOR, "6_informative_cells_behavior_variables", "Informative cells: behaviour (zero lag)")
    frac_figure(method)


# method comparison: all cells together (group-agnostic), one panel per variable
tab_all = tab.assign(group="all")
cmp_curve = curves(tab_all, "mi_bits", by=("method", "variable", "group"))
variables = [v for v in VARS_ALL if (cmp_curve["variable"] == v).any()]
if variables:
    ncols = min(len(variables), 4)
    nrows = int(np.ceil(len(variables) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.6 * nrows), squeeze=False)
    for ax, v in zip(axes.flat, variables):
        for m in METHODS:
            c = cmp_curve[(cmp_curve["method"] == m) & (cmp_curve["variable"] == v)].sort_values("s_rel")
            if c.empty:
                continue
            ax.plot(c["s_rel"], c["mean"], color="#333333" if m == "trial_mean" else "#c0392b", label=m,
                    **METHOD_STYLE[m])
            ax.fill_between(c["s_rel"], c["mean"] - c["sem"], c["mean"] + c["sem"], alpha=0.15, lw=0,
                            color="#333333" if m == "trial_mean" else "#c0392b")
        shade(ax)
        ax.set_title(VAR_TITLE[v], fontsize=9)
        ax.set_ylabel("MI (bits)")
    for ax in list(axes.flat)[len(variables):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Trial-mean vs frame-pooled aggregation, all cells; {agg_note}", y=1.0, fontsize=9)
    fig.tight_layout()
    save_fig(fig, figdir, "4_method_comparison")
    plt.close(fig)

print("\nPeak mean MI (bits) by method and variable, all cells:")
pk = cmp_curve.loc[cmp_curve.groupby(["method", "variable"])["mean"].idxmax()]
print(pk.pivot(index="variable", columns="method", values="mean").reindex(VARS_ALL).round(4).to_string())
print(f"\nWrote tables to {table_dir}, figures to {figdir}")
