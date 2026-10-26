# -*- coding: utf-8 -*-
"""
2a_cell_distribution
======================
Survey of the SPATIAL organization of the recorded cells -- DN sessions exclusively, since cell recording only
happens for DN (see session.py's module docstring). How many cells, where they sit in the brain and in the FOV,
how big they are -- not their activity (that is 2b and the later steps).

What this does
--------------
1. Loads celldata of every DN session (one session at a time, cached per session), optionally restricted by
   `RECOMBINASE_FILTER` ('cre' / 'flp' / None).
2. Spatial PROXIMITY filter for unlabeled V1/PM cells (an unlabeled cell only counts as "near labeled" if
   within PROXIMITY_RADIUS_UM of a labeled cell of the same area in the SAME session; see
   celldata_utils.filter_nearlabeled). Computed and reported either way; `APPLY_PROXIMITY_FILTER` decides whether
   the rest of this script's plots use the filtered or the full population.
3. Cell counts per session / animal / group, depth and layer distribution (depth is absolute, so it is comparable
   across sessions), FOV x/y positions (NEVER pooled across sessions or areas: local coordinates), cell size/shape.

Project-wide rules (infotheory/project_config.py) -- what applies to 2a
    * selection : --animals / --sessions restrict the sessions; a restricted run writes to out/selection/<tag>/
                  and NEVER overwrites celldata_combined.csv in out/ (read by every later step). This step does
                  not use 1b's session list (1b judges task performance, not neurons): all DN sessions.
    * aggregate : --aggregate pooled|session|mouse = the unit for the group comparisons (counts, depth, layer
                  composition, size/shape): pooled = all cells of a group together, no error bars; session / mouse =
                  one value per session / mouse (MIN_CELLS_PER_UNIT cells needed in the group), bars = mean +- SEM
                  over units, dots = units coloured by mouse. Always the six groups (V1-lab, V1-unl, PM-lab, PM-unl,
                  AL, RSP). Per-session diagnostics (cells per session, depth per session, x/y maps, labeled-cell
                  and proximity-pass diagnostics) are session-level by nature and not affected.
    * memory    : celldata of ONE session at a time (per-session cache in store/); only the small combined cell table
                  is held. No violin plots.
    * binning   : not used here (no discrete information measures).

Reads: 0_data/ directly (every DN session with a celldata.csv).

Outputs
-------
2_pipeline/2a_cell_distribution/
    out/    celldata_combined.csv     every DN session's celldata + session_id / animal_id + passes_proximity_filter
                                      (+ labeled_grouped) -- read this from later steps (Rule #1)
            cell_counts_summary.csv   per session / area counts
            cell_properties_by_unit[__<aggregate>].csv   per group (and unit): n_cells, mean depth / size / shape
            figures/*.png
    out/selection/<tag>/...           the same for a restricted run
    store/  celldata_session_<hash>.pkl   per-session cache
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from infotheory.pipeline import get_pipeline_paths
from infotheory.session import discover_sessions
from infotheory.celldata_utils import (
    get_area_label, filter_nearlabeled, nearest_labeled_distance, LABEL_SPLIT_AREAS,
    AREA_COLORS, LABEL_MARKERS, DEFAULT_AREA_ORDER, DEFAULT_LABEL_ORDER,
    SIX_GROUP_ORDER, SIX_GROUP_LABELS, SIX_GROUP_COLORS,
)
from infotheory.groups import add_six_group_column, group_key
from infotheory.plotting import set_style, save_fig, clear_figures
from infotheory.scope import add_scope_args, scope_from_args, select_sessions, scoped_dirs, data_stamp, unit_noun
from infotheory.session_jobs import run_sessions
from infotheory.unit_stats import group_unit_table
from infotheory.unit_plots import unit_bars, group_unit_bars, mouse_palette, mouse_legend

PROTOCOL = "DN"  # cell recording only happens for DN -- see session.py's module docstring

# Optional filter on the 'recombinase' column ('cre'/'flp'/'non'). None = no filter.
RECOMBINASE_FILTER: str | None = None

PROXIMITY_RADIUS_UM = 50.0       # how close an unlabeled V1/PM cell must be to a labeled same-area cell
PROXIMITY_AREAS = ("V1", "PM")
APPLY_PROXIMITY_FILTER = False   # True: sections 4+ use ONLY cells that pass the proximity filter

MIN_CELLS_PER_UNIT = 5           # session / mouse aggregate: a unit needs this many cells in a group for its
                                 # mean to be shown (ignored when pooled)
MAX_SESSIONS_PER_GRID_PAGE = 6   # x/y figures: one ROW per session, one COLUMN per area; this many rows per page

LAYER_ORDER = ["L1", "L2/3", "L4", "L5", "L6", "L6a", "L6b"]
LAYER_COLORS_BASE = {"L1": "#a8dadc", "L2/3": "#8ecae6", "L4": "#219ebc",
                      "L5": "#023047", "L6": "#03045e", "L6a": "#03045e", "L6b": "#012a4a"}
MIN_CELLS_FOR_KDE = 15
JOBLIB_VERBOSE = 5

parser = argparse.ArgumentParser()
add_scope_args(parser)
args = parser.parse_args()
scope = scope_from_args(args)
AGG = scope.aggregate
UNIT = unit_noun(AGG)
UNITS = "mice" if UNIT == "mouse" else f"{UNIT}s"
AGG_SUFFIX = "" if AGG == "pooled" else f"__{AGG}"
UNIT_KEYS = {"pooled": [], "session": ["animal_id", "session_id"], "mouse": ["animal_id"]}[AGG]

paths = get_pipeline_paths(__file__)
table_dir, figdir = scoped_dirs(paths, scope)
clear_figures(figdir)
set_style()
print(scope.describe())


# --------------------------------------------------------------------- #
# 1. Celldata of every selected DN session, one session at a time (cached per session)
# --------------------------------------------------------------------- #
all_triples = discover_sessions([PROTOCOL])
triples = select_sessions(all_triples, scope)
session_ids = [f"{a}_{d}" for _, a, d in triples]
stamps = {f"{a}_{d}": data_stamp(p, a, d) for p, a, d in triples}
mouse_colors = mouse_palette(a for _, a, _ in all_triples)


def process_session(ses, seed):
    if ses.celldata is None or ses.celldata.empty:
        return None
    cd = ses.celldata.copy()
    cd["session_id"] = ses.session_id
    cd["animal_id"] = ses.animal_id
    # proximity is computed per session by definition, so doing it here equals doing it on the combined table
    cd["passes_proximity_filter"] = filter_nearlabeled(cd, radius_um=PROXIMITY_RADIUS_UM, areas=PROXIMITY_AREAS)
    return cd


tables = run_sessions(
    session_ids, process_session, [PROTOCOL], paths.store, "celldata_session",
    key_params=dict(version="2a-v2", radius=PROXIMITY_RADIUS_UM, areas=PROXIMITY_AREAS),
    load_kwargs=dict(load_behaviordata=False, load_celldata=True),
    n_jobs=scope.n_jobs, recompute=args.recompute, verbose=JOBLIB_VERBOSE,
    extra_key_by_session=stamps, flatten=False)
tables = [t for t in tables if t is not None]
if not tables:
    raise SystemExit(f"No selected {PROTOCOL} session had a celldata.csv -- nothing to analyze.")
celldata = pd.concat(tables, ignore_index=True)
del tables

n_sessions_total = celldata["session_id"].nunique()
n_animals_total = celldata["animal_id"].nunique()
print(f"\n{len(celldata)} cells across {n_sessions_total} {PROTOCOL} sessions, {n_animals_total} animals")


# --------------------------------------------------------------------- #
# 2. Recombinase filter (optional)
# --------------------------------------------------------------------- #
if RECOMBINASE_FILTER is not None:
    n_before = len(celldata)
    celldata = celldata[celldata["recombinase"] == RECOMBINASE_FILTER].copy()
    print(f"Recombinase filter = '{RECOMBINASE_FILTER}': kept {len(celldata)}/{n_before} cells "
          f"({celldata['session_id'].nunique()} sessions, {celldata['animal_id'].nunique()} animals)")
else:
    print(f"Recombinase filter: none (values present: {celldata['recombinase'].value_counts().to_dict()})")


# --------------------------------------------------------------------- #
# 3. Proximity filter -- report before/after, decide which population the rest uses
# --------------------------------------------------------------------- #
for area in PROXIMITY_AREAS:
    unl_mask = (celldata["roi_name"] == area) & (celldata["labeled"] == "unl")
    n_unl = int(unl_mask.sum())
    n_pass = int((unl_mask & celldata["passes_proximity_filter"]).sum())
    print(f"Proximity filter ({area}, unlabeled, within {PROXIMITY_RADIUS_UM:.0f}um of a labeled "
          f"{area} cell, same session): {n_pass}/{n_unl} pass"
          f"{' (0/0 -- no unlabeled ' + area + ' cells)' if n_unl == 0 else ''}")

# 'labeled_grouped': 'labeled' with every cell outside LABEL_SPLIT_AREAS (V1/PM) forced to 'unl' (a stray labeled
# AL/RSP cell must not form its own near-empty group). Kept in celldata_combined.csv as before.
_, celldata["labeled_grouped"], _ = get_area_label(celldata, label_split_areas=LABEL_SPLIT_AREAS)

if APPLY_PROXIMITY_FILTER:
    n_before = len(celldata)
    analysis_celldata = celldata[celldata["passes_proximity_filter"]].copy()
    print(f"APPLY_PROXIMITY_FILTER=True: sections 4+ use {len(analysis_celldata)}/{n_before} cells")
else:
    analysis_celldata = celldata.copy()
    print("APPLY_PROXIMITY_FILTER=False: sections 4+ use the full (unfiltered) population")
celldata.to_csv(table_dir / "celldata_combined.csv", index=False)

# the six project groups (V1-lab, V1-unl, PM-lab, PM-unl, AL, RSP) as column `group`, for all group plots
analysis_celldata = add_six_group_column(analysis_celldata)
GROUP_KEYS = [group_key(g) for g in SIX_GROUP_ORDER]


# --------------------------------------------------------------------- #
# 4. Cell counts
# --------------------------------------------------------------------- #
counts_per_session = (analysis_celldata.groupby(["animal_id", "session_id", "roi_name"])
                      .size().rename("n_cells").reset_index())
counts_per_session.to_csv(table_dir / "cell_counts_summary.csv", index=False)

session_order = (analysis_celldata.groupby("session_id")["animal_id"].first().sort_values().index.tolist())
areas_present = [a for a in DEFAULT_AREA_ORDER if a in analysis_celldata["roi_name"].unique()]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
# 4a. per session, stacked by area (session-level by nature)
pivot = (counts_per_session.pivot_table(index="session_id", columns="roi_name", values="n_cells",
                                        fill_value=0).reindex(session_order))
bottom = np.zeros(len(pivot))
for area in areas_present:
    if area not in pivot.columns:
        continue
    axes[0].bar(range(len(pivot)), pivot[area].to_numpy(), bottom=bottom, color=AREA_COLORS.get(area, "gray"),
                label=area, edgecolor="black", linewidth=0.3)
    bottom += pivot[area].to_numpy()
axes[0].set_xticks(range(len(pivot)))
axes[0].set_xticklabels(pivot.index, rotation=90, fontsize=6)
axes[0].set_ylabel("cells")
axes[0].set_title("Cells per session (stacked by area)")
axes[0].legend(fontsize=8, frameon=False)

# 4b. cells per group at the unit of analysis (a unit with no cell of a group counts 0)
cnt = analysis_celldata.groupby(["group"] + UNIT_KEYS).size().rename("n_cells").reset_index()
if UNIT_KEYS:
    units = analysis_celldata[UNIT_KEYS].drop_duplicates()
    full = units.merge(pd.DataFrame({"group": GROUP_KEYS}), how="cross")
    cnt = full.merge(cnt, on=["group"] + UNIT_KEYS, how="left").fillna({"n_cells": 0})
colors = {k: SIX_GROUP_COLORS[g] for k, g in zip(GROUP_KEYS, SIX_GROUP_ORDER)}
unit_bars(axes[1], cnt, "group", "n_cells", GROUP_KEYS, AGG, colors=colors, mouse_colors=mouse_colors,
          ylabel="cells" if AGG == "pooled" else f"cells per {UNIT}")
axes[1].set_xticklabels([SIX_GROUP_LABELS[g] for g in SIX_GROUP_ORDER], fontsize=8)
axes[1].set_title(f"Cells by group (unit = {UNIT}; {n_sessions_total} sessions, {n_animals_total} animals)")
if AGG != "pooled":
    mouse_legend(axes[1], mouse_colors, animals=analysis_celldata["animal_id"].unique(), loc="upper right")
fig.suptitle(f"Cell counts -- {PROTOCOL}" + (f" (recombinase={RECOMBINASE_FILTER})" if RECOMBINASE_FILTER else ""),
             y=1.03)
fig.tight_layout()
save_fig(fig, figdir, "1_cell_counts")

print("\nCells per animal:")
print(analysis_celldata.groupby("animal_id")["session_id"].apply(lambda s: (s.nunique(), len(s)))
      .rename("n_sessions__n_cells").to_string())


# --------------------------------------------------------------------- #
# 5. Depth / laminar distribution (depth is an absolute measurement)
# --------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
group_unit_bars(axes[0], analysis_celldata, "depth", AGG, mouse_colors=mouse_colors,
                min_cells_per_unit=MIN_CELLS_PER_UNIT)
axes[0].invert_yaxis()  # depth increases downward from the pial surface
axes[0].set_ylabel("depth (μm from pial surface)")
axes[0].set_title(f"Depth by group (unit = {UNIT})")

# layer composition: fraction of each group's cells per layer (per unit first, then mean over units)
lay = analysis_celldata.groupby(["group"] + UNIT_KEYS + ["layer"]).size().rename("n").reset_index()
lay["frac"] = lay["n"] / lay.groupby(["group"] + UNIT_KEYS)["n"].transform("sum")
if UNIT_KEYS:   # a unit with no cell in a layer contributes 0 to that layer's mean
    allk = lay[["group"] + UNIT_KEYS].drop_duplicates().merge(
        pd.DataFrame({"layer": lay["layer"].unique()}), how="cross")
    lay = allk.merge(lay, on=["group"] + UNIT_KEYS + ["layer"], how="left").fillna({"frac": 0.0})
layer_pivot = lay.groupby(["group", "layer"])["frac"].mean().unstack("layer").reindex(GROUP_KEYS).fillna(0.0)
layers_present = list(layer_pivot.columns)
ordered_layers = [l for l in LAYER_ORDER if l in layers_present] + [l for l in layers_present if l not in LAYER_ORDER]
layer_pivot = layer_pivot[ordered_layers]
layer_cmap = plt.colormaps["Blues"]
layer_colors = {l: LAYER_COLORS_BASE.get(l, layer_cmap((i + 1) / (len(ordered_layers) + 1)))
                for i, l in enumerate(ordered_layers)}
layer_pivot.plot(kind="bar", stacked=True, ax=axes[1], color=layer_colors, edgecolor="black", linewidth=0.3)
axes[1].set_xticklabels([SIX_GROUP_LABELS[g] for g in SIX_GROUP_ORDER], rotation=0, fontsize=8)
axes[1].set_xlabel("")
axes[1].set_ylabel("fraction of cells")
axes[1].set_title("Layer composition" + ("" if AGG == "pooled" else f" (mean over {UNITS})"))
axes[1].legend(title="layer", fontsize=8, frameon=False)

# depth per SESSION (consistency check: is one session an outlier -- e.g. a different objective/zoom)
sns.stripplot(data=analysis_celldata, x="session_id", y="depth", hue="roi_name", order=session_order,
              palette=AREA_COLORS, size=2, alpha=0.5, ax=axes[2], legend=False)
axes[2].invert_yaxis()
axes[2].tick_params(axis="x", rotation=90, labelsize=6)
axes[2].set_ylabel("depth (μm)")
axes[2].set_title("Depth per session (consistency check)")
fig.suptitle(f"Depth / laminar distribution -- {PROTOCOL}", y=1.03)
fig.tight_layout()
save_fig(fig, figdir, "2_depth_laminar_distribution")

depth_by_session = analysis_celldata.groupby("session_id")["depth"].median()
if len(depth_by_session) >= 4:
    med, spread = depth_by_session.median(), depth_by_session.std()
    outliers = depth_by_session[(depth_by_session - med).abs() > 2 * spread]
    if len(outliers):
        print(f"\nDepth outlier session(s) (median depth >2 SD from the group median of {med:.0f}μm):")
        print(outliers.round(1).to_string())


# --------------------------------------------------------------------- #
# 6. x/y spatial distribution -- one panel PER (SESSION, AREA), never overlaid or pooled
#    (xloc/yloc are local to the (session, area) imaging plane)
# --------------------------------------------------------------------- #
def plot_xy_pages(kind):
    """x/y maps: ONE ROW PER SESSION, one fixed COLUMN PER AREA (same area in the same column on every row;
    an area not recorded in a session leaves its panel empty). kind = "scatter" | "kde" (smoothed 2D density
    per area, labeled cells still individual on top)."""
    n_cols = len(areas_present)
    n_pages = int(np.ceil(len(session_order) / MAX_SESSIONS_PER_GRID_PAGE))
    for page in range(n_pages):
        page_sessions = session_order[page * MAX_SESSIONS_PER_GRID_PAGE:(page + 1) * MAX_SESSIONS_PER_GRID_PAGE]
        fig, axes = plt.subplots(len(page_sessions), n_cols, figsize=(3.4 * n_cols, 3.2 * len(page_sessions)),
                                 squeeze=False, sharex=True, sharey = True)
        for i, session_id in enumerate(page_sessions):
            for j, area in enumerate(areas_present):
                ax = axes[i, j]
                # ax.set_xticks([])
                # ax.set_yticks([])
                if i == 0:
                    ax.set_title(area, color=AREA_COLORS.get(area, "black"), fontsize=10)
                if j == 0:
                    ax.set_ylabel(session_id, fontsize=7)
                sdf = analysis_celldata[(analysis_celldata["session_id"] == session_id)
                                        & (analysis_celldata["roi_name"] == area)]
                if sdf.empty:
                    ax.text(0.5, 0.5, "not recorded", transform=ax.transAxes, ha="center", va="center",
                            fontsize=7, color="grey")
                    for sp in ax.spines.values():
                        sp.set_visible(False)
                    continue
                color = AREA_COLORS.get(area, "gray")
                if kind == "scatter":
                    for label in DEFAULT_LABEL_ORDER:
                        sub = sdf[sdf["labeled"] == label]
                        if sub.empty:
                            continue
                        ax.scatter(sub["xloc"], sub["yloc"], s=10 if label == "unl" else 22, color=color,
                                   marker=LABEL_MARKERS[label], alpha=0.6 if label == "unl" else 0.9,
                                   edgecolor="black" if label == "lab" else "none", linewidth=0.4)
                else:
                    if len(sdf) < MIN_CELLS_FOR_KDE or sdf["xloc"].std() == 0 or sdf["yloc"].std() == 0:
                        ax.scatter(sdf["xloc"], sdf["yloc"], s=8, color=color, alpha=0.5, linewidth=0)
                    else:
                        try:
                            sns.kdeplot(x=sdf["xloc"], y=sdf["yloc"], ax=ax, fill=True, color=color, alpha=0.45,
                                        levels=5, thresh=0.15, bw_adjust=1.2, cut=0.5)
                        except Exception:
                            ax.scatter(sdf["xloc"], sdf["yloc"], s=8, color=color, alpha=0.5, linewidth=0)
                    lab = sdf[sdf["labeled"] == "lab"]
                    if not lab.empty:
                        ax.scatter(lab["xloc"], lab["yloc"], s=14, color=color, marker="^", edgecolor="black",
                                   linewidth=0.4, alpha=0.9)
                    x_pad = max((sdf["xloc"].max() - sdf["xloc"].min()) * 0.05, 1.0)
                    y_pad = max((sdf["yloc"].max() - sdf["yloc"].min()) * 0.05, 1.0)
                    # ax.set_xlim(sdf["xloc"].min() - x_pad, sdf["xloc"].max() + x_pad)
                    # ax.set_ylim(sdf["yloc"].min() - y_pad, sdf["yloc"].max() + y_pad)
                ax.set_aspect("equal")
        if kind == "scatter":
            handles = [plt.Line2D([0], [0], marker=LABEL_MARKERS[l], linestyle="none", markerfacecolor="gray",
                                  markeredgecolor="black" if l == "lab" else "none", markersize=7, label=l)
                       for l in DEFAULT_LABEL_ORDER]
            title = "Cell positions"
            name = f"3_xy_positions_page{page + 1}"
        else:
            handles = [plt.Line2D([0], [0], marker="^", linestyle="none", markerfacecolor="gray",
                                  markeredgecolor="black", markersize=7, label="labeled cell")]
            title = "Cell density (2D KDE; labeled cells on top)"
            name = f"3b_xy_density_page{page + 1}"
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
        fig.suptitle(f"{title}: one row per session, one column per area (FOV/plane-local coordinates, NOT aligned "
                     f"across sessions OR areas) -- page {page + 1}/{n_pages}", y=1.0, fontsize=9)
        fig.tight_layout()
        save_fig(fig, figdir, name)
        plt.close(fig)


plot_xy_pages("scatter")
plot_xy_pages("kde")


# --------------------------------------------------------------------- #
# 7. Cell size/shape properties by group, at the unit of analysis
# --------------------------------------------------------------------- #
size_vars = [("radius", "radius (μm)"), ("npix", "ROI size (pixels)"), ("npix_soma", "soma size (pixels)"),
             ("skew", "trace skewness"), ("iscell_prob", "iscell probability")]
size_vars = [(v, lbl) for v, lbl in size_vars if v in analysis_celldata.columns]
if size_vars:
    fig, axes = plt.subplots(1, len(size_vars), figsize=(4.2 * len(size_vars), 4), squeeze=False)
    for ax, (var, lbl) in zip(axes[0], size_vars):
        group_unit_bars(ax, analysis_celldata, var, AGG, mouse_colors=mouse_colors,
                        min_cells_per_unit=MIN_CELLS_PER_UNIT, ylabel=lbl, tick_size=6)
    fig.suptitle(f"Cell size/shape properties by group -- {PROTOCOL} (unit = {UNIT})", y=1.02)
    fig.tight_layout()
    save_fig(fig, figdir, "4_cell_size_properties")

# report: everything plotted by group, per unit
report_vars = ["depth"] + [v for v, _ in size_vars]
group_unit_table(analysis_celldata, report_vars, AGG).to_csv(
    table_dir / f"cell_properties_by_unit{AGG_SUFFIX}.csv", index=False)


# --------------------------------------------------------------------- #
# 7b. Labeled reference cells per session/area (explains a 0% proximity pass rate)
# --------------------------------------------------------------------- #
n_labeled_rows = []
for session_id in session_order:
    sdf = celldata[celldata["session_id"] == session_id]
    for area in PROXIMITY_AREAS:
        n_labeled_rows.append({"session_id": session_id, "area": area,
                               "n_labeled": int(((sdf["roi_name"] == area) & (sdf["labeled"] == "lab")).sum())})
n_labeled_df = pd.DataFrame(n_labeled_rows)

fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(session_order)), 4))
width = 0.35
x = np.arange(len(session_order))
for i, area in enumerate(PROXIMITY_AREAS):
    sub = n_labeled_df[n_labeled_df["area"] == area].set_index("session_id").reindex(session_order)
    ax.bar(x + (i - 0.5) * width, sub["n_labeled"].to_numpy(), width=width, color=AREA_COLORS.get(area, "gray"),
           label=area, edgecolor="black", linewidth=0.3)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(session_order, rotation=90, fontsize=6)
ax.set_ylabel("labeled cells")
ax.set_title("Labeled reference cells per session (V1/PM) -- a session with 0 here will show 0% pass rate")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
save_fig(fig, figdir, "5a_labeled_cells_per_session")


# --------------------------------------------------------------------- #
# 7c. Distribution of nearest-labeled-cell distances (is PROXIMITY_RADIUS_UM in a natural gap?)
# --------------------------------------------------------------------- #
dist_to_labeled = nearest_labeled_distance(celldata, areas=PROXIMITY_AREAS)
dist_df = pd.DataFrame({"area": celldata["roi_name"], "distance": dist_to_labeled}).dropna()
if len(dist_df):
    fig, axes = plt.subplots(1, len(PROXIMITY_AREAS), figsize=(5 * len(PROXIMITY_AREAS), 4), squeeze=False)
    for ax, area in zip(axes[0], PROXIMITY_AREAS):
        sub = dist_df[dist_df["area"] == area]["distance"]
        if sub.empty:
            continue
        ax.hist(sub, bins=40, color=AREA_COLORS.get(area, "gray"), edgecolor="black", linewidth=0.3)
        ax.axvline(PROXIMITY_RADIUS_UM, color="firebrick", linestyle="--", linewidth=1.5,
                   label=f"{PROXIMITY_RADIUS_UM:.0f}μm threshold")
        ax.set_xlabel("distance to nearest labeled same-area cell (μm)")
        ax.set_ylabel("unlabeled cells")
        ax.set_title(f"{area} (n={len(sub)}, {float((sub <= PROXIMITY_RADIUS_UM).mean()):.0%} within threshold)")
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("Unlabeled V1/PM cells relative to the labeled population\n(sessions with no labeled cells in an "
                 "area are excluded here, not shown as 0)", y=1.05)
    fig.tight_layout()
    save_fig(fig, figdir, "5b_distance_to_labeled_distribution")


# --------------------------------------------------------------------- #
# 8. Proximity-filter pass rate per session (unfiltered diagnostic)
# --------------------------------------------------------------------- #
prox_rows = []
for session_id in session_order:
    sdf = celldata[celldata["session_id"] == session_id]
    for area in PROXIMITY_AREAS:
        unl = sdf[(sdf["roi_name"] == area) & (sdf["labeled"] == "unl")]
        if unl.empty:
            continue
        prox_rows.append({"session_id": session_id, "area": area,
                          "frac_pass": float(unl["passes_proximity_filter"].mean()), "n": len(unl)})
if prox_rows:
    prox_df = pd.DataFrame(prox_rows)
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(session_order)), 4.5))
    for i, area in enumerate(PROXIMITY_AREAS):
        sub = prox_df[prox_df["area"] == area].set_index("session_id").reindex(session_order)
        ax.bar(x + (i - 0.5) * width, sub["frac_pass"].to_numpy(), width=width, color=AREA_COLORS.get(area, "gray"),
               label=area, edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(session_order, rotation=90, fontsize=6)
    ax.set_ylabel(f"fraction of unlabeled cells within {PROXIMITY_RADIUS_UM:.0f}μm of a labeled cell")
    ax.set_ylim(0, 1.05)
    ax.set_title("Proximity-filter pass rate per session (V1/PM, unlabeled cells)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save_fig(fig, figdir, "5_proximity_filter_pass_rate")

print(f"\nWrote tables to {table_dir}, figures to {figdir}")
