# -*- coding: utf-8 -*-
"""
2f_lag_information
=====================
Does a cell's activity relate to a behavioral variable at some TEMPORAL LAG, and where along the corridor? DN
sessions only; behavioral predictors only (run speed, pupil area, motion energy, lick): stim / choice / correct are
per-trial scalars, a temporal lag is meaningless for them. Every computation pools raw imaging FRAMES (no trial-window
means) -- that is the point of this step, averaging inside windows could hide a lagged relationship (trial and
bin-occupancy rules of 2c/2d therefore do not apply; the shared BIN COUNT does, see below).

Per (predictor, project group) -- lags are chosen per GROUP, not per cell (a single cell is too noisy):
 1. LINEAR LAG SEARCH. Linear r^2 between each cell's activity and the lag-shifted predictor over LAG_RANGE_S, per cell.
    Group curve = mean r^2 over the group's analysis units (see aggregate), OPTIMAL LAG = its argmax.
    Positive lag = behavior BEFORE activity (behavior leads, activity is the delayed response); negative = activity
    leads (e.g. motor preparation / efference copy).
 2. GRANGER CAUSALITY at the group's optimal lag: does the lag-shifted predictor improve the prediction of a cell's
    activity beyond the cell's own AR(AR_ORDER_GRANGER) history (F-test)? Reported as the fraction of cells with
    p<0.05. CAVEAT: the lag was chosen because it maximised r^2, so testing at that lag is post-selection inference and
    the fraction is inflated above 5% even for no real effect: read it as an upper bound.
 3. POSITION-RESOLVED INFORMATION at the optimal lag (x = position relative to stimulus onset, binned by POSITION_WINDOW):
    a) shuffle-corrected MUTUAL INFORMATION(lag-shifted predictor; activity) per bin, every included cell;
    b) shuffle-corrected TRANSFER ENTROPY(lag-shifted predictor -> activity) per bin, on a random subsample of
       MAX_CELLS_FOR_TE cells per (session, group) (3-way histograms are far costlier).
    Bins with fewer than MIN_SAMPLES_PER_BIN frames are left out.

Project-wide rules (infotheory/project_config.py)
    * selection : --animals / --sessions; a restricted run writes to out/selection/<tag>/ and never overwrites out/.
                  The optimal lag is chosen from the SELECTED sessions only, so a restricted run can differ from the
                  full run (per-session lag-search caches are reused; later stages re-run when the lags differ).
    * aggregate : --aggregate pooled|session|mouse = the unit of analysis. Per-cell values are averaged within a unit
                  (>= MIN_CELLS_PER_UNIT cells of the group when not pooled) and then across units: curves = mean +- SEM
                  over units (pooled: mean over all cells, no band), Granger bars = mean +- SEM over units, dots = units
                  coloured by mouse. The optimal lag is the argmax of THAT mean curve, so it depends on the aggregate.
                  Always the six groups (V1-lab, V1-unl, PM-lab, PM-unl, AL, RSP).
    * memory    : each session is loaded once per pass (pass A = lag search; pass B = Granger + position MI + TE, which
                  needs the lags of pass A), reduced to small tables, cached per session in store/ and freed. Peak RAM
                  ~ n_jobs x one session. (The old script held every session at once and reloaded it four times.)
    * binning    : MI and TE use project_config.RESPONSE_BINS (3) quantile bins, like 2c/2d/3x, so the information measures
                  are comparable. CHANGE vs. the previous 2f: MI used 10 bins and TE 6, so absolute values are lower now
                  and not comparable with old runs. The linear lag search (r^2) needs no bins.

KNOWN CAVEAT (unchanged): positions beyond about +-40 cm of stimulus onset show an elevated MI signature that is identical
across predictors and groups -- likely cross-trial contamination (zpos never resets, so a far-from-stimulus bin can fall in the
previous trial's reward / ITI). Not resolved; the figures shade TRUSTED_POSITION_RANGE's complement.

Outputs
-------
2_pipeline/2f_lag_information/
    out/    lag_search_per_cell.csv            session, cell, group, predictor, lag_s, r2   (all lags)
            lag_curve_by_group[__<aggregate>].csv   predictor, group, lag_s: mean r^2 over units, SEM, n_units
            optimal_lags[__<aggregate>].csv    predictor, group: optimal_lag_s, r2_at_optimum, n_units
            granger_causality.csv              per (session, cell, predictor): r2_restricted/unrestricted, p_value, lag
            granger_by_group[__<aggregate>].csv fraction significant per group (and unit) x predictor
            position_information_per_cell.csv, position_te_per_cell.csv       per cell and position bin
            position_information[__<aggregate>].csv, position_transfer_entropy[__<aggregate>].csv  per group (and unit)
            figures/*.png                      1 lag curves; 2 Granger; 3 information PSTH; 4 transfer-entropy PSTH
    out/selection/<tag>/...
    store/  lag_session_<hash>.pkl (pass A), lagB_session_<hash>.pkl (pass B)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from infotheory import project_config as cfg
from infotheory.pipeline import get_pipeline_paths
from infotheory.session import discover_sessions
from infotheory.neural_encoding import load_cell_inclusion, BEHAVIOR_PREDICTORS
from infotheory.continuous import build_calcium_continuous, merge_behavior_video, restrict_to_engaged, remove_pupil_outliers
from infotheory.psth import derive_onset_time, compute_position_bin_indices, mi_from_position_bins, te_from_position_bins
from infotheory.info_theory import granger_causality_test
from infotheory.spike_stats import get_frame_rate
from infotheory.celldata_utils import SIX_GROUP_ORDER, SIX_GROUP_LABELS, SIX_GROUP_COLORS
from infotheory.groups import add_six_group_column, group_key
from infotheory.plotting import set_style, save_fig, clear_figures
from infotheory.scope import add_scope_args, scope_from_args, select_sessions, scoped_dirs, data_stamp, unit_noun
from infotheory.session_jobs import run_sessions
from infotheory.unit_stats import stable_seed
from infotheory.unit_plots import grouped_series_bars, mouse_palette, mouse_legend

PROTOCOL = "DN"
PREDICTORS = list(BEHAVIOR_PREDICTORS)

LAG_RANGE_S = (-2.0, 2.0)
LAG_STEP_S = 0.25
AR_ORDER_GRANGER = 3
MAX_CELLS_FOR_TE = 40                 # per (session, group)
N_SHUFFLES_TE = 8
POSITION_WINDOW = (-80.0, 60.0, 8.0)  # s_pre, s_post, binsize (cm)
TRUSTED_POSITION_RANGE = (-40.0, 40.0)
N_SHUFFLES_PSTH = 10
MIN_SAMPLES_PER_BIN = 400
MIN_FRAMES_LAG = 50
MIN_CELLS_PER_UNIT = 5
BINS = cfg.RESPONSE_BINS              # shared with 2c/2d/3x
ALPHA = 0.05
RANDOM_STATE = 0
JOBLIB_VERBOSE = 5

parser = argparse.ArgumentParser()
add_scope_args(parser)
parser.add_argument("--no-te", action="store_true", help="skip the transfer-entropy PSTH (the slowest part)")
args = parser.parse_args()
scope = scope_from_args(args)
AGG = scope.aggregate
UNIT = unit_noun(AGG)
UNITS = "mice" if UNIT == "mouse" else f"{UNIT}s"
N_JOBS = scope.n_jobs
AGG_SUFFIX = "" if AGG == "pooled" else f"__{AGG}"
RUN_TE = not args.no_te
MIN_C = 1 if AGG == "pooled" else MIN_CELLS_PER_UNIT
GROUP_KEYS = [group_key(g) for g in SIX_GROUP_ORDER]
GROUP_COLOR = {group_key(g): SIX_GROUP_COLORS[g] for g in SIX_GROUP_ORDER}
GROUP_LABEL = {group_key(g): SIX_GROUP_LABELS[g] for g in SIX_GROUP_ORDER}

paths = get_pipeline_paths(__file__)
table_dir, figdir = scoped_dirs(paths, scope)
clear_figures(figdir)
set_style()
print(scope.describe())

# --------------------------------------------------------------------- #
# 1. Inputs: included sessions (1b), cells (2a + 2b), groups
# --------------------------------------------------------------------- #
fits_path = paths.out_from("1b_psychometric") / "psychometric_fits.csv"
cell_dist_out = paths.out_from("2a_cell_distribution")
activity_out = paths.out_from("2b_activity_statistics")
for f, hint in [(fits_path, "1b_psychometric.py"), (cell_dist_out / "celldata_combined.csv", "2a_cell_distribution.py"),
                (activity_out / "qc_per_cell.csv", "2b_activity_statistics.py")]:
    if not f.exists():
        raise SystemExit(f"Run {hint} first -- expected {f}")
fits_df = pd.read_csv(fits_path)
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
cd = add_six_group_column(pd.read_csv(cell_dist_out / "celldata_combined.csv", low_memory=False)[["session_id", "cell_id", "roi_name", "labeled"]])
group_of = dict(zip(zip(cd["session_id"], cd["cell_id"]), cd["group"]))
animal_of = {s: selected[s][1] for s in session_ids}
stamps = {s: data_stamp(*selected[s]) for s in session_ids}
print(f"{len(session_ids)} sessions, {sum(len(v) for v in cells_by_session.values())} included cells")
lags = np.round(np.arange(LAG_RANGE_S[0], LAG_RANGE_S[1] + LAG_STEP_S / 2, LAG_STEP_S), 6)


def shift_with_nan(arr: np.ndarray, lag_frames: int) -> np.ndarray:
    """shift_with_nan(behavior, +lag)[t] == behavior[t - lag]: a POSITIVE lag compares activity now with behavior `lag`
    frames EARLIER (behavior leads); negative = behavior from the FUTURE (activity leads). Padded with NaN, never wraps
    around (np.roll would contaminate the edges). Verified earlier against an injected, known lag."""
    shifted = np.full(len(arr), np.nan, dtype=float)
    if lag_frames > 0:
        shifted[lag_frames:] = arr[:-lag_frames] if lag_frames < len(arr) else np.nan
    elif lag_frames < 0:
        shifted[:lag_frames] = arr[-lag_frames:]
    else:
        shifted[:] = arr
    return shifted


def build_frame_table(ses, cell_ids):
    """Every included cell's activity + every behavioral predictor on the calcium clock, restricted to engaged
    periods. No trial-window averaging."""
    if (ses.trialdata is None or ses.behaviordata is None or ses.calciumdata is None or ses.ts_F is None):
        return None, None
    merged_full = remove_pupil_outliers(merge_behavior_video(ses.behaviordata, ses.videodata))
    preds = [p for p in PREDICTORS if p in merged_full.columns]
    frame_table = build_calcium_continuous(ses.calciumdata.loc[:, cell_ids], ses.ts_F, merged_full, extra_columns=preds)
    frame_table = restrict_to_engaged(frame_table, ses.trialdata)
    if frame_table.empty:
        return None, None
    engaged_trials = (ses.trialdata[ses.trialdata["engaged"] == 1] if "engaged" in ses.trialdata.columns
                      else ses.trialdata.copy())
    return frame_table, derive_onset_time(engaged_trials, merged_full)


def r2_columns(A, ref):
    """Pearson r^2 of every column of A (frames x cells) with `ref`, over frames where both are finite; NaN-safe,
    equals linear_fit(ref, A[:, j]).r2 per cell."""
    out = np.full(A.shape[1], np.nan)
    ok_cols = np.isfinite(A).all(0)
    fin = np.isfinite(ref)
    if fin.sum() >= MIN_FRAMES_LAG and ok_cols.any():
        X, r = A[fin][:, ok_cols], ref[fin]
        Xc, rc = X - X.mean(0), r - r.mean()
        den = np.sqrt((Xc ** 2).sum(0) * (rc ** 2).sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            out[ok_cols] = ((Xc * rc[:, None]).sum(0) / den) ** 2
    for j in np.where(~ok_cols)[0]:                         # cells with NaN frames: own mask
        m = np.isfinite(A[:, j]) & fin
        if m.sum() >= MIN_FRAMES_LAG and A[m, j].std() > 0 and ref[m].std() > 0:
            out[j] = np.corrcoef(A[m, j], ref[m])[0, 1] ** 2
    return out


# --------------------------------------------------------------------- #
# 2. PASS A -- linear lag search (per session, cached)
# --------------------------------------------------------------------- #
def lag_session(ses, seed):
    cells = cells_by_session[ses.session_id]
    frame_table, _ = build_frame_table(ses, cells)
    if frame_table is None:
        return None
    fs = get_frame_rate(ses, fallback=8.0)
    A = frame_table[cells].to_numpy(dtype=float)
    groups = [group_of.get((ses.session_id, c), "") for c in cells]
    rows = []
    for pred in [p for p in PREDICTORS if p in frame_table.columns]:
        ref = frame_table[pred].to_numpy(dtype=float)
        for lag_s in lags:
            r2 = r2_columns(A, shift_with_nan(ref, int(round(lag_s * fs))))
            rows.append(pd.DataFrame({"session_id": ses.session_id, "animal_id": ses.animal_id, "cell_id": cells,
                                      "group": groups, "predictor": pred, "lag_s": lag_s, "r2": r2}))
    print(f"  [{ses.session_id}] lag search done ({len(cells)} cells, {len(lags)} lags)")
    return pd.concat(rows, ignore_index=True) if rows else None


LOAD = dict(load_behaviordata=True, load_videodata=True, load_celldata=True, load_calciumdata=True)
resA = run_sessions(session_ids, lag_session, [PROTOCOL], paths.store, "lag_session",
                    key_params=dict(version="2f-v2", lags=list(lags), min_frames=MIN_FRAMES_LAG),
                    load_kwargs=LOAD, cells_by_session=cells_by_session, n_jobs=N_JOBS, recompute=args.recompute,
                    verbose=JOBLIB_VERBOSE, extra_key_by_session=stamps, flatten=False)
lag_tab = pd.concat([r for r in resA if r is not None], ignore_index=True)
del resA
lag_tab = lag_tab[lag_tab["group"].isin(GROUP_KEYS)]
if lag_tab.empty:
    raise SystemExit("No session produced a lag search.")
lag_tab["unit"] = {"pooled": "all", "session": lag_tab["session_id"], "mouse": lag_tab["animal_id"]}[AGG]
lag_tab.drop(columns="unit").to_csv(table_dir / "lag_search_per_cell.csv", index=False)
print(f"\n{len(lag_tab)} (cell, predictor, lag) linear r² values")


def unit_means(df, value, by, min_c=MIN_C):
    """Mean of `value` over the cells of each analysis unit within `by` (+ group); units with fewer than min_c cells
    are dropped. Columns: by..., group, unit, animal_id, value, n_cells."""
    g = df.groupby(by + ["group", "unit"], observed=True)
    out = g[value].mean().to_frame()
    out["n_cells"] = g.size()
    out["animal_id"] = g["animal_id"].first()
    return out[out["n_cells"] >= min_c].reset_index()


def curve_over_units(um, value, x):
    """(predictor, group, x): mean and SEM of the per-unit values over units (SEM = 0 when pooled)."""
    g = um.groupby(["predictor", "group", x])[value]
    out = g.mean().to_frame("mean")
    out["sem"] = g.sem().fillna(0.0) if AGG != "pooled" else 0.0
    out["n_units"] = g.size()
    return out.reset_index()


lag_um = unit_means(lag_tab, "r2", ["predictor", "lag_s"])
lag_curve = curve_over_units(lag_um, "r2", "lag_s")
lag_curve.to_csv(table_dir / f"lag_curve_by_group{AGG_SUFFIX}.csv", index=False)
best = lag_curve.loc[lag_curve.dropna(subset=["mean"]).groupby(["predictor", "group"])["mean"].idxmax()]
optimal = best.rename(columns={"lag_s": "optimal_lag_s", "mean": "r2_at_optimum"})[
    ["predictor", "group", "optimal_lag_s", "r2_at_optimum", "n_units"]].reset_index(drop=True)
optimal.to_csv(table_dir / f"optimal_lags{AGG_SUFFIX}.csv", index=False)
lag_of = {(r.predictor, r.group): float(r.optimal_lag_s) for r in optimal.itertuples()}
print("\nOptimal lag per (predictor, group); positive = behavior BEFORE activity, negative = activity leads:")
print(optimal.pivot(index="predictor", columns="group", values="optimal_lag_s").reindex(columns=GROUP_KEYS).to_string())

preds_present = [p for p in PREDICTORS if (lag_curve["predictor"] == p).any()]


def group_lines(ax, curve, pred, x, y="mean", ylabel=None, shade=None, lag_marks=False):
    for g in GROUP_KEYS:
        c = curve[(curve["predictor"] == pred) & (curve["group"] == g)].sort_values(x)
        if c.empty:
            continue
        n = int(c["n_units"].max()) if "n_units" in c else 0
        ax.plot(c[x], c[y], color=GROUP_COLOR[g], label=f"{GROUP_LABEL[g]} (n={n})")
        if AGG != "pooled" and "sem" in c:
            ax.fill_between(c[x], c[y] - c["sem"], c[y] + c["sem"], color=GROUP_COLOR[g], alpha=0.2, lw=0)
        if lag_marks and (pred, g) in lag_of:
            ax.axvline(lag_of[(pred, g)], color=GROUP_COLOR[g], ls=":", lw=1, alpha=0.7)
    ax.axvline(0, color="black", lw=0.7)
    ax.set_title(pred)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.legend(fontsize=6, frameon=False)


agg_note = {"pooled": "mean over all cells (no band)", "session": "mean +- SEM over sessions",
            "mouse": "mean +- SEM over mice"}[AGG]
fig, axes = plt.subplots(1, len(preds_present), figsize=(4.6 * len(preds_present), 4.2), squeeze=False)
for ax, pred in zip(axes[0], preds_present):
    group_lines(ax, lag_curve, pred, "lag_s", ylabel="linear r²", lag_marks=True)
    ax.set_xlabel("lag (s) [+ = behavior leads]")
fig.suptitle(f"Linear lag search -- {PROTOCOL}; {agg_note}; dotted = optimal lag", y=1.03)
fig.tight_layout()
save_fig(fig, figdir, "1_lag_search")
plt.close(fig)

# --------------------------------------------------------------------- #
# 3. PASS B -- Granger + position MI + position TE (per session, cached)
# --------------------------------------------------------------------- #
def pass_b(ses, seed):
    sid = ses.session_id
    cells = cells_by_session[sid]
    frame_table, engaged_trials = build_frame_table(ses, cells)
    if frame_table is None or engaged_trials is None or engaged_trials.empty:
        return None
    fs = get_frame_rate(ses, fallback=8.0)
    rng = np.random.default_rng(stable_seed(sid, "lagB", base=seed))
    s_pre, s_post, binsize = POSITION_WINDOW
    bincenters, bin_idx = compute_position_bin_indices(engaged_trials, frame_table, s_pre=s_pre, s_post=s_post,
                                                       binsize=binsize)
    cell_group = {c: group_of.get((sid, c), "") for c in cells}
    by_group = {}
    for c in cells:
        if cell_group[c] in GROUP_KEYS:
            by_group.setdefault(cell_group[c], []).append(c)
    te_cells = {g: (list(rng.choice(cs, MAX_CELLS_FOR_TE, replace=False)) if len(cs) > MAX_CELLS_FOR_TE else cs)
                for g, cs in by_group.items()}
    act = {c: frame_table[c].to_numpy(dtype=float) for c in cells}
    gr, pi, te = [], [], []
    for pred in [p for p in PREDICTORS if p in frame_table.columns]:
        raw = frame_table[pred].to_numpy(dtype=float)
        for g, gcells in by_group.items():
            lag_s = lag_of.get((pred, g))
            if lag_s is None:
                continue
            ref = shift_with_nan(raw, int(round(lag_s * fs)))
            for c in gcells:
                r2r, r2u, p = granger_causality_test(act[c], ref, ar_order=AR_ORDER_GRANGER)
                if not np.isnan(p):
                    gr.append(dict(session_id=sid, animal_id=ses.animal_id, cell_id=c, group=g, predictor=pred,
                                   lag_s=lag_s, r2_restricted=r2r, r2_unrestricted=r2u, p_value=p))
                res = mi_from_position_bins(act[c], ref, bincenters, bin_idx, mi_bins=BINS, n_shuffles=N_SHUFFLES_PSTH,
                                            min_samples_per_bin=MIN_SAMPLES_PER_BIN, rng=rng)
                for _, row in res.iterrows():
                    pi.append(dict(session_id=sid, animal_id=ses.animal_id, cell_id=c, group=g, predictor=pred,
                                   s_rel=row["s_rel"], n_samples=row["n_samples"], mi_bits=row["mi_bits"]))
            if RUN_TE:
                for c in te_cells[g]:
                    res = te_from_position_bins(act[c], ref, bincenters, bin_idx, target_horizon_frames=1, te_bins=BINS,
                                                n_shuffles=N_SHUFFLES_TE, min_samples_per_bin=MIN_SAMPLES_PER_BIN, rng=rng)
                    for _, row in res.iterrows():
                        te.append(dict(session_id=sid, animal_id=ses.animal_id, cell_id=c, group=g, predictor=pred,
                                       s_rel=row["s_rel"], n_samples=row["n_samples"], te_bits=row["te_bits"]))
    print(f"  [{sid}] Granger, position MI{', TE' if RUN_TE else ''} done")
    return dict(granger=pd.DataFrame(gr), mi=pd.DataFrame(pi), te=pd.DataFrame(te))


resB = run_sessions(
    session_ids, pass_b, [PROTOCOL], paths.store, "lagB_session",
    key_params=dict(version="2f-v2", lags=json.dumps({f"{k[0]}|{k[1]}": v for k, v in sorted(lag_of.items())}),
                    ar=AR_ORDER_GRANGER, bins=BINS, shuffles=N_SHUFFLES_PSTH, shuffles_te=N_SHUFFLES_TE, te=RUN_TE,
                    max_te=MAX_CELLS_FOR_TE, window=POSITION_WINDOW, min_bin=MIN_SAMPLES_PER_BIN),
    load_kwargs=LOAD, cells_by_session=cells_by_session, base_seed=RANDOM_STATE, n_jobs=N_JOBS,
    recompute=args.recompute, verbose=JOBLIB_VERBOSE, extra_key_by_session=stamps, flatten=False)
resB = [r for r in resB if r is not None]


def stack(key):
    parts = [r[key] for r in resB if len(r[key])]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["unit"] = {"pooled": "all", "session": df["session_id"], "mouse": df["animal_id"]}[AGG]
    return df


gr_tab, mi_tab, te_tab = stack("granger"), stack("mi"), stack("te")
del resB

# ---- Granger ----
if len(gr_tab):
    gr_tab["r2_improvement"] = gr_tab["r2_unrestricted"] - gr_tab["r2_restricted"]
    gr_tab["significant"] = gr_tab["p_value"] < ALPHA
    gr_tab.drop(columns="unit").to_csv(table_dir / "granger_causality.csv", index=False)
    gu = unit_means(gr_tab.assign(sig=gr_tab["significant"].astype(float)), "sig", ["predictor"])
    gu_out = gu.rename(columns={"sig": "frac_significant"})
    (gu_out if AGG != "pooled" else gu_out.drop(columns=["unit", "animal_id"])).to_csv(
        table_dir / f"granger_by_group{AGG_SUFFIX}.csv", index=False)
    print(f"\n{len(gr_tab)} Granger tests. CAVEAT: the lag tested was chosen by maximising r² (post-selection "
          f"inference), so fractions above {ALPHA:.0%} are inflated even for no real effect -- upper bounds.")
    print(gu.groupby(["predictor", "group"])["sig"].mean().unstack("group").reindex(columns=GROUP_KEYS).round(3).to_string())
    fig, ax = plt.subplots(figsize=(10, 4.2))
    handles = grouped_series_bars(ax, gr_tab.assign(sig=gr_tab["significant"].astype(float)), "sig", "predictor",
                                  preds_present, AGG, mouse_colors=mouse_colors, min_cells_per_unit=MIN_CELLS_PER_UNIT,
                                  hline=ALPHA, ylabel="fraction of cells p<0.05", tick_size=8)
    ax.legend(*handles, fontsize=8, frameon=False, title="predictor")
    ax.set_title(f"Granger causality at each group's optimal lag (upper bounds; dashed = {ALPHA:.0%}); unit = {UNIT}", fontsize=9)
    fig.tight_layout()
    save_fig(fig, figdir, "2_granger_causality")
    plt.close(fig)

# ---- position-resolved MI and TE ----
def position_block(tab, value, name, per_cell_name, fig_name, ylabel, title):
    if tab.empty:
        print(f"\nNo {name} values.")
        return
    tab.drop(columns="unit").to_csv(table_dir / f"{per_cell_name}.csv", index=False)
    um = unit_means(tab, value, ["predictor", "s_rel"])
    out = um if AGG != "pooled" else um.drop(columns=["unit", "animal_id"])
    out.to_csv(table_dir / f"{name}{AGG_SUFFIX}.csv", index=False)
    curve = curve_over_units(um, value, "s_rel")
    preds = [p for p in PREDICTORS if (curve["predictor"] == p).any()]
    fig, axes = plt.subplots(1, len(preds), figsize=(4.6 * len(preds), 4.4), squeeze=False)
    for ax, pred in zip(axes[0], preds):
        group_lines(ax, curve, pred, "s_rel", ylabel=ylabel)
        ax.axvline(0, color="black", ls=":", lw=1)
        ax.axvspan(POSITION_WINDOW[0], TRUSTED_POSITION_RANGE[0], color="grey", alpha=0.15, zorder=0)
        ax.axvspan(TRUSTED_POSITION_RANGE[1], POSITION_WINDOW[1], color="grey", alpha=0.15, zorder=0)
        ax.set_xlabel("position rel. to stimulus onset (cm)")
        lags_txt = ", ".join(f"{GROUP_LABEL[g]} {lag_of[(pred, g)]:+.2f}s" for g in GROUP_KEYS if (pred, g) in lag_of)
        ax.set_title(f"{pred}\nlags: {lags_txt}", fontsize=7)
    fig.suptitle(f"{title}; {agg_note}\nshaded: known artifact range (see docstring)", y=1.06)
    fig.tight_layout()
    save_fig(fig, figdir, fig_name)
    plt.close(fig)
    print(f"\n{len(tab)} per-cell {name} values")


position_block(mi_tab, "mi_bits", "position_information", "position_information_per_cell", "3_lagged_information_psth",
               "MI (bits, shuffle-corrected)", f"Lagged information PSTH -- {PROTOCOL} (behavior shifted by each group's optimal lag)")
if RUN_TE:
    position_block(te_tab, "te_bits", "position_transfer_entropy", "position_te_per_cell", "4_transfer_entropy_psth",
                   "transfer entropy (bits, shuffle-corrected)",
                   f"Transfer entropy PSTH -- {PROTOCOL} (behavior -> activity, <= {MAX_CELLS_FOR_TE} cells/group/session)")

print(f"\nWrote tables to {table_dir}, figures to {figdir}")
