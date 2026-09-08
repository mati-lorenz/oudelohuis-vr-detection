# -*- coding: utf-8 -*-
"""
2a_cell_distribution
======================
Single-cell statistical analysis of the SPATIAL organization of
recorded cells -- DN sessions exclusively, since cell recording only
happens for DN (see session.py's module docstring; DM/DP have no
celldata at all). This is a survey of the recorded population itself
(how many cells, where they sit in the brain and in the FOV, how big
they are) -- not of their activity, which is what the later 2_ steps
(single-cell/pairwise information) will cover.

What this does
--------------
1. Loads celldata for every DN session, optionally restricted by
   `RECOMBINASE_FILTER` ('cre'/'flp'/None for no filter -- see
   celldata's 'recombinase' column, which is 'non' for any cell that
   doesn't express that session's driver-line marker).
2. Computes a spatial PROXIMITY filter for unlabeled V1/PM cells (an
   unlabeled V1 cell only counts as "near labeled" if within
   PROXIMITY_RADIUS_UM of a labeled V1 cell in the SAME session; same
   for PM/PM -- see infotheory.celldata_utils.filter_nearlabeled).
   Computed and reported either way; whether the REST of this script's
   plots use the filtered or full population is controlled by
   `APPLY_PROXIMITY_FILTER` below.
3. Cell counts: per session, per animal, per (area, label) group.
4. Depth/laminar distribution: depth is an absolute physical
   measurement (microns from the pial surface), so -- unlike xloc/yloc
   -- it IS comparable and pooled across sessions/animals here.
5. x/y (FOV) spatial distribution -- explicitly NOT pooled across
   sessions: xloc/yloc are local pixel/micron coordinates within each
   session's own field of view, which is independently positioned
   every recording day, so overlaying different sessions' coordinates
   would produce a meaningless blob (see the per-session grid figure).
6. Cell size/shape properties (radius, npix, npix_soma, skew,
   iscell_prob) by area/label.

Reads: 0_data/ directly (celldata isn't behavior-inclusion-filtered by
1b -- 1b's criteria are about task performance, not neurons -- so
there's no upstream out/ to read the session list from; every DN
session with a celldata.csv is included here).

Outputs
-------
2_pipeline/2a_cell_distribution/
    out/    celldata_combined.csv   every DN session's celldata, tagged
                                     with session_id/animal_id and a
                                     passes_proximity_filter column --
                                     read this (not 0_data/) from later
                                     2_ steps that need celldata (Rule #1)
            cell_counts_summary.csv per-session and per-(area,label) counts
            figures/*.png
    store/  celldata_combined.pkl   cached combined table (reused unless
                                     --recompute is passed)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

from infotheory.pipeline import get_pipeline_paths
from infotheory.session import load_sessions
from infotheory.celldata_utils import (
    get_area_label, ordered_groups, bar_by_group, filter_nearlabeled, nearest_labeled_distance,
    LABEL_SPLIT_AREAS,
    AREA_COLORS, LABEL_LINESTYLES, LABEL_MARKERS, DEFAULT_AREA_ORDER, DEFAULT_LABEL_ORDER,
)
from infotheory.plotting import set_style, save_fig, clear_figures

PROTOCOL = "DN"  # cell recording only happens for DN -- see session.py's module docstring

# Optional filter on the 'recombinase' column ('cre'/'flp'/'non', one
# driver line per animal -- see make_fake_data.py). None = no filter
# (all cells, of every recombinase value, kept). Set to 'cre' or 'flp'
# to restrict to sessions/cells from that driver line; note this also
# naturally drops every 'non' (unlabeled-marker) cell, since 'non' never
# equals 'cre' or 'flp'.
RECOMBINASE_FILTER: str | None = None

PROXIMITY_RADIUS_UM = 50.0       # how close an unlabeled V1/PM cell must be to a labeled
                                  # same-area cell (same session) to pass the proximity filter
PROXIMITY_AREAS = ("V1", "PM")
APPLY_PROXIMITY_FILTER = False   # if True, sections 4 onward use ONLY cells that pass the
                                  # proximity filter (labeled cells and cells outside
                                  # PROXIMITY_AREAS are never excluded by it either way);
                                  # if False (default), the filter is still computed and
                                  # reported (section 3) but doesn't restrict the other plots

MAX_SESSIONS_PER_GRID_PAGE = 12  # x/y grid figure: paginate if more DN sessions than this
GRID_NCOLS = 4

# Superficial-to-deep display order + colors for the layer-composition
# figure. Any layer label actually present in the data but not listed
# here (a naming convention this project hasn't seen yet) is still
# plotted -- appended at the end with a colormap-generated color -- see
# section 5 below.
LAYER_ORDER = ["L1", "L2/3", "L4", "L5", "L6", "L6a", "L6b"]
LAYER_COLORS_BASE = {"L1": "#a8dadc", "L2/3": "#8ecae6", "L4": "#219ebc",
                      "L5": "#023047", "L6": "#03045e", "L6a": "#03045e", "L6b": "#012a4a"}

N_JOBS_SESSIONS = -1
JOBLIB_VERBOSE = 5

parser = argparse.ArgumentParser()
parser.add_argument("--recompute", action="store_true",
                     help="Reload sessions from 0_data/ instead of reusing the cached "
                          "store/celldata_combined.pkl")
args = parser.parse_args()

paths = get_pipeline_paths(__file__)
figdir = paths.out / "figures"
clear_figures(figdir)
set_style()


# --------------------------------------------------------------------- #
# 1. Load celldata for every DN session -- or reuse the cache
# --------------------------------------------------------------------- #
combined_cache = paths.store / "celldata_combined.pkl"
REQUIRED_COLUMNS = ["session_id", "animal_id", "roi_name", "labeled", "xloc", "yloc",
                    "depth", "layer", "recombinase", "passes_proximity_filter"]

celldata = None
if not args.recompute and combined_cache.exists():
    try:
        candidate = pd.read_pickle(combined_cache)
        missing = [c for c in REQUIRED_COLUMNS if c not in candidate.columns]
        if missing:
            print(f"Cached celldata is missing columns {missing} -- looks like it was built by an "
                  f"older version of this script. Recomputing from 0_data/ instead of using "
                  f"{combined_cache} (safe to delete that file to silence this check in the future).")
        else:
            celldata = candidate
            print(f"Reusing cached celldata from {paths.store} (pass --recompute to reload 0_data/)")
    except Exception as e:
        print(f"Could not load cached celldata ({e}); recomputing from 0_data/ ...")

if celldata is None:
    sessions = load_sessions(protocols=[PROTOCOL], load_behaviordata=False, load_celldata=True)

    def process_session(ses):
        if ses.celldata is None or ses.celldata.empty:
            return None
        cd = ses.celldata.copy()
        cd["session_id"] = ses.session_id
        cd["animal_id"] = ses.animal_id
        return cd

    print(f"Processing {len(sessions)} {PROTOCOL} sessions (n_jobs={N_JOBS_SESSIONS}) ...")
    session_results = Parallel(n_jobs=N_JOBS_SESSIONS, backend="loky", verbose=JOBLIB_VERBOSE)(
        delayed(process_session)(ses) for ses in sessions
    )
    session_tables = [r for r in session_results if r is not None]
    if not session_tables:
        raise SystemExit(f"No {PROTOCOL} session had a celldata.csv -- nothing to analyze.")
    celldata = pd.concat(session_tables, ignore_index=True)

    # Proximity filter computed once here (not re-derived every time this
    # script's plots run), so it's available in the cache and in
    # celldata_combined.csv for downstream 2_ steps too.
    celldata["passes_proximity_filter"] = filter_nearlabeled(
        celldata, radius_um=PROXIMITY_RADIUS_UM, areas=PROXIMITY_AREAS)

    celldata.to_pickle(combined_cache)

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
    print(f"Recombinase filter: none (values present: "
          f"{celldata['recombinase'].value_counts().to_dict()})")


# --------------------------------------------------------------------- #
# 3. Proximity filter -- report before/after, decide which population
#    the rest of this script uses
# --------------------------------------------------------------------- #
for area in PROXIMITY_AREAS:
    unl_mask = (celldata["roi_name"] == area) & (celldata["labeled"] == "unl")
    n_unl = int(unl_mask.sum())
    n_pass = int((unl_mask & celldata["passes_proximity_filter"]).sum())
    print(f"Proximity filter ({area}, unlabeled, within {PROXIMITY_RADIUS_UM:.0f}um of a labeled "
          f"{area} cell, same session): {n_pass}/{n_unl} pass"
          f"{' (0/0 -- no unlabeled ' + area + ' cells)' if n_unl == 0 else ''}")

# 'labeled_grouped': like 'labeled', but every cell outside LABEL_SPLIT_AREAS
# (V1/PM) is forced to 'unl' -- used for GROUP-SUMMARY plots (violins,
# bars) below, so a stray labeled AL/RSP cell doesn't produce its own
# near-empty, distorting group. The raw 'labeled' column is kept as-is
# for per-CELL displays (the xy scatter/density figures), where showing
# an individual labeled cell accurately is the point, not a group summary.
_, celldata["labeled_grouped"], _ = get_area_label(celldata, label_split_areas=LABEL_SPLIT_AREAS)

if APPLY_PROXIMITY_FILTER:
    n_before = len(celldata)
    analysis_celldata = celldata[celldata["passes_proximity_filter"]].copy()
    print(f"APPLY_PROXIMITY_FILTER=True: sections 4+ use {len(analysis_celldata)}/{n_before} cells")
else:
    analysis_celldata = celldata
    print("APPLY_PROXIMITY_FILTER=False: sections 4+ use the full (unfiltered) population "
          "-- set APPLY_PROXIMITY_FILTER=True at the top of this script to restrict them")

celldata.to_csv(paths.out / "celldata_combined.csv", index=False)


# --------------------------------------------------------------------- #
# 4. Cell counts: per session, per animal, per (area, label)
# --------------------------------------------------------------------- #
counts_per_session = (analysis_celldata.groupby(["animal_id", "session_id", "roi_name"])
                       .size().rename("n_cells").reset_index())
counts_per_session.to_csv(paths.out / "cell_counts_summary.csv", index=False)

session_order = (analysis_celldata.groupby("session_id")["animal_id"].first()
                  .sort_values().index.tolist())
areas_present = [a for a in DEFAULT_AREA_ORDER if a in analysis_celldata["roi_name"].unique()]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 4a. Per-session, stacked by area
pivot = (counts_per_session.pivot_table(index="session_id", columns="roi_name", values="n_cells",
                                         fill_value=0).reindex(session_order))
bottom = np.zeros(len(pivot))
for area in areas_present:
    if area not in pivot.columns:
        continue
    axes[0].bar(range(len(pivot)), pivot[area].to_numpy(), bottom=bottom,
                color=AREA_COLORS.get(area, "gray"), label=area, edgecolor="black", linewidth=0.3)
    bottom += pivot[area].to_numpy()
axes[0].set_xticks(range(len(pivot)))
axes[0].set_xticklabels(pivot.index, rotation=90, fontsize=6)
axes[0].set_ylabel("cells")
axes[0].set_title("Cells per session (stacked by area)")
axes[0].legend(fontsize=8, frameon=False)

# 4b. Pooled counts per (area, label)
area_, label_, _ = get_area_label(analysis_celldata, label_split_areas=LABEL_SPLIT_AREAS)
counts_by_group = {(a, l): np.ones(np.sum((area_ == a) & (label_ == l)))
                    for a in areas_present for l in DEFAULT_LABEL_ORDER}
bar_by_group(axes[1], counts_by_group, ylabel="cells", statistic="count", zero_line=False)
axes[1].set_title(f"Total cells by area/label ({n_sessions_total} sessions, {n_animals_total} animals)")

fig.suptitle(f"Cell counts -- {PROTOCOL}"
             + (f" (recombinase={RECOMBINASE_FILTER})" if RECOMBINASE_FILTER else ""), y=1.03)
fig.tight_layout()
save_fig(fig, figdir, "1_cell_counts")

print("\nCells per animal:")
print(analysis_celldata.groupby("animal_id")["session_id"].apply(lambda s: (s.nunique(), len(s)))
      .rename("n_sessions__n_cells").to_string())


# --------------------------------------------------------------------- #
# 5. Depth / laminar distribution -- pooled across sessions (depth is an
#    absolute physical measurement, unlike xloc/yloc -- see module docstring)
# --------------------------------------------------------------------- #
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

sns.violinplot(data=analysis_celldata, x="roi_name", y="depth", hue="labeled_grouped",
                order=areas_present, hue_order=DEFAULT_LABEL_ORDER, split=True,
                palette={"unl": "#cccccc", "lab": "#d62728"}, ax=axes[0])
axes[0].invert_yaxis()  # depth increases downward from the pial surface
axes[0].set_ylabel("depth (\u03bcm from pial surface)")
axes[0].set_title("Depth by area/label")

layer_counts = (analysis_celldata.groupby(["roi_name", "layer"]).size()
                .rename("n").reset_index())
layer_pivot = layer_counts.pivot_table(index="roi_name", columns="layer", values="n", fill_value=0)
layer_pivot = layer_pivot.reindex([a for a in areas_present if a in layer_pivot.index])

# Order columns superficial-to-deep where the label is one of the usual
# cortical layers; anything else (a naming convention this project
# hasn't seen yet) is appended at the end and given a color from a
# colormap, rather than the plot crashing on an unrecognized label (as
# it did on real data, which turned out to also have 'L4' -- this
# project's own synthetic data only ever generates L2/3/L5).
layers_present = list(layer_pivot.columns)
ordered_layers = ([l for l in LAYER_ORDER if l in layers_present]
                   + [l for l in layers_present if l not in LAYER_ORDER])
layer_pivot = layer_pivot[ordered_layers]
layer_cmap = plt.colormaps["Blues"]
layer_colors = {l: LAYER_COLORS_BASE.get(l, layer_cmap((i + 1) / (len(ordered_layers) + 1)))
                 for i, l in enumerate(ordered_layers)}

layer_pivot.plot(kind="bar", stacked=True, ax=axes[1], color=layer_colors,
                  edgecolor="black", linewidth=0.3)
axes[1].set_xlabel("area")
axes[1].set_ylabel("cells")
axes[1].set_title("Layer composition by area")
axes[1].tick_params(axis="x", rotation=0)
axes[1].legend(title="layer", fontsize=8, frameon=False)

# Depth per SESSION (small-multiples check: does depth range look
# consistent across mice/sessions, or is one session an outlier -- e.g.
# a different objective/zoom setting)
sns.stripplot(data=analysis_celldata, x="session_id", y="depth", hue="roi_name",
              order=session_order, palette=AREA_COLORS, size=2, alpha=0.5, ax=axes[2], legend=False)
axes[2].invert_yaxis()
axes[2].tick_params(axis="x", rotation=90, labelsize=6)
axes[2].set_ylabel("depth (\u03bcm)")
axes[2].set_title("Depth per session (consistency check)")

fig.suptitle(f"Depth / laminar distribution -- {PROTOCOL}", y=1.03)
fig.tight_layout()
save_fig(fig, figdir, "2_depth_laminar_distribution")

# Flag sessions whose median depth is a clear outlier relative to the
# rest (e.g. a different objective/zoom setting that day) -- easy to
# miss by eye in the strip plot above when there are many sessions.
depth_by_session = analysis_celldata.groupby("session_id")["depth"].median()
if len(depth_by_session) >= 4:
    med, spread = depth_by_session.median(), depth_by_session.std()
    outliers = depth_by_session[(depth_by_session - med).abs() > 2 * spread]
    if len(outliers):
        print(f"\nDepth outlier session(s) (median depth >2 SD from the group median of "
              f"{med:.0f}\u03bcm):")
        print(outliers.round(1).to_string())


# --------------------------------------------------------------------- #
# 6. x/y spatial distribution -- one panel PER SESSION, never pooled
#    (xloc/yloc are local to each session's own field of view, and
#    fields of view are not aligned/registered across different
#    recording days -- overlaying them would be meaningless).
# --------------------------------------------------------------------- #
n_pages = int(np.ceil(len(session_order) / MAX_SESSIONS_PER_GRID_PAGE))
for page in range(n_pages):
    page_sessions = session_order[page * MAX_SESSIONS_PER_GRID_PAGE:(page + 1) * MAX_SESSIONS_PER_GRID_PAGE]
    ncols = min(GRID_NCOLS, len(page_sessions))
    nrows = int(np.ceil(len(page_sessions) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows), squeeze=False)
    flat_axes = axes.flatten()

    for ax, session_id in zip(flat_axes, page_sessions):
        sdf = analysis_celldata[analysis_celldata["session_id"] == session_id]
        for area in areas_present:
            for label in DEFAULT_LABEL_ORDER:
                sub = sdf[(sdf["roi_name"] == area) & (sdf["labeled"] == label)]
                if sub.empty:
                    continue
                ax.scatter(sub["xloc"], sub["yloc"], s=10 if label == "unl" else 22,
                           color=AREA_COLORS.get(area, "gray"), marker=LABEL_MARKERS[label],
                           alpha=0.6 if label == "unl" else 0.9,
                           edgecolor="black" if label == "lab" else "none", linewidth=0.4)
        ax.set_title(session_id, fontsize=8)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in flat_axes[len(page_sessions):]:
        ax.set_visible(False)

    handles = [plt.Line2D([0], [0], marker=LABEL_MARKERS[l], linestyle="none", markerfacecolor="gray",
                           markeredgecolor="black" if l == "lab" else "none", markersize=7, label=l)
               for l in DEFAULT_LABEL_ORDER]
    handles += [plt.Line2D([0], [0], marker="s", linestyle="none", markerfacecolor=AREA_COLORS[a],
                            markersize=8, label=a) for a in areas_present]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
    fig.suptitle(f"Cell positions per session (FOV-local coordinates, NOT aligned across sessions) "
                 f"-- page {page + 1}/{n_pages}", y=1.01)
    fig.tight_layout()
    save_fig(fig, figdir, f"3_xy_positions_page{page + 1}")


# --------------------------------------------------------------------- #
# 6b. SUPPLEMENTARY to the above (not a replacement): the same
#    per-session layout, but synthesized into semi-transparent 2D KDE
#    "blobs" per area instead of every individual cell -- a cleaner
#    read of WHERE each area sits in the FOV at a glance across many
#    sessions. Labeled cells are still plotted individually on top,
#    since those (and specifically their clustering) are what the
#    proximity filter depends on -- worth keeping visible even in this
#    smoothed-out view.
# --------------------------------------------------------------------- #
MIN_CELLS_FOR_KDE = 15  # below this a KDE contour is more noise than signal -- fall back
                         # to a plain scatter for that area/session instead

for page in range(n_pages):
    page_sessions = session_order[page * MAX_SESSIONS_PER_GRID_PAGE:(page + 1) * MAX_SESSIONS_PER_GRID_PAGE]
    ncols = min(GRID_NCOLS, len(page_sessions))
    nrows = int(np.ceil(len(page_sessions) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.6 * nrows), squeeze=False)
    flat_axes = axes.flatten()

    for ax, session_id in zip(flat_axes, page_sessions):
        sdf = analysis_celldata[analysis_celldata["session_id"] == session_id]
        for area in areas_present:
            sub = sdf[sdf["roi_name"] == area]
            if sub.empty:
                continue
            if len(sub) < MIN_CELLS_FOR_KDE or sub["xloc"].std() == 0 or sub["yloc"].std() == 0:
                ax.scatter(sub["xloc"], sub["yloc"], s=8, color=AREA_COLORS.get(area, "gray"),
                           alpha=0.5, linewidth=0)
                continue
            try:
                # cut=0.5 (not seaborn's default 3): a KDE naturally
                # extends several bandwidths past the data it's fit on,
                # and with the default cut that padding was large enough
                # to make every panel look "zoomed out" relative to
                # where the cells actually are -- see the explicit
                # set_xlim/set_ylim below too, which crops each panel
                # tightly to that session's own real cell coordinates
                # regardless of how far the fitted contour extends.
                sns.kdeplot(x=sub["xloc"], y=sub["yloc"], ax=ax, fill=True,
                            color=AREA_COLORS.get(area, "gray"), alpha=0.45, levels=5,
                            thresh=0.15, bw_adjust=1.2, cut=0.5)
            except Exception:
                ax.scatter(sub["xloc"], sub["yloc"], s=8, color=AREA_COLORS.get(area, "gray"),
                           alpha=0.5, linewidth=0)

        lab = sdf[sdf["labeled"] == "lab"]
        for area in areas_present:
            sub = lab[lab["roi_name"] == area]
            if sub.empty:
                continue
            ax.scatter(sub["xloc"], sub["yloc"], s=14, color=AREA_COLORS.get(area, "gray"),
                       marker="^", edgecolor="black", linewidth=0.4, alpha=0.9)

        # Crop to this session's actual cell extent (+5% margin) --
        # without this, matplotlib auto-scales to the KDE contours'
        # extent (which always overshoots the real data due to
        # bandwidth smoothing, even with cut reduced above), making the
        # whole panel look zoomed out relative to where the cells are.
        if not sdf.empty:
            x_span = sdf["xloc"].max() - sdf["xloc"].min()
            y_span = sdf["yloc"].max() - sdf["yloc"].min()
            x_pad = max(x_span * 0.05, 1.0)
            y_pad = max(y_span * 0.05, 1.0)
            ax.set_xlim(sdf["xloc"].min() - x_pad, sdf["xloc"].max() + x_pad)
            ax.set_ylim(sdf["yloc"].min() - y_pad, sdf["yloc"].max() + y_pad)

        ax.set_title(session_id, fontsize=8)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in flat_axes[len(page_sessions):]:
        ax.set_visible(False)

    handles = [plt.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=AREA_COLORS[a],
                           markeredgecolor="none", markersize=10, alpha=0.6, label=a) for a in areas_present]
    handles += [plt.Line2D([0], [0], marker="^", linestyle="none", markerfacecolor="gray",
                            markeredgecolor="black", markersize=7, label="labeled cell")]
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
    fig.suptitle(f"Cell density per session, synthesized (2D KDE per area; labeled cells still shown "
                 f"individually on top) -- page {page + 1}/{n_pages}", y=1.01)
    fig.tight_layout()
    save_fig(fig, figdir, f"3b_xy_density_page{page + 1}")


# --------------------------------------------------------------------- #
# 7. Cell size/shape properties by area/label
# --------------------------------------------------------------------- #
size_vars = [("radius", "radius (\u03bcm)"), ("npix", "ROI size (pixels)"),
             ("npix_soma", "soma size (pixels)"), ("skew", "trace skewness"),
             ("iscell_prob", "iscell probability")]
size_vars = [(v, lbl) for v, lbl in size_vars if v in analysis_celldata.columns]

if size_vars:
    fig, axes = plt.subplots(1, len(size_vars), figsize=(4 * len(size_vars), 4), squeeze=False)
    for ax, (var, lbl) in zip(axes[0], size_vars):
        sns.violinplot(data=analysis_celldata, x="roi_name", y=var, hue="labeled_grouped",
                        order=areas_present, hue_order=DEFAULT_LABEL_ORDER, split=True,
                        palette={"unl": "#cccccc", "lab": "#d62728"}, ax=ax, legend=(var == size_vars[0][0]))
        ax.set_ylabel(lbl)
        ax.set_xlabel("")
    fig.suptitle(f"Cell size/shape properties -- {PROTOCOL}", y=1.02)
    fig.tight_layout()
    save_fig(fig, figdir, "4_cell_size_properties")


# --------------------------------------------------------------------- #
# 7b. How many labeled reference cells actually exist per session/area
#    -- directly explains a 0% proximity-filter pass rate for a given
#    session/area: is it because there were NO labeled cells that day
#    (nothing to be near), or because labeled cells existed but were
#    far from the unlabeled population? Distinguishing these matters
#    for whether a session should just be excluded outright.
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
    ax.bar(x + (i - 0.5) * width, sub["n_labeled"].to_numpy(), width=width,
           color=AREA_COLORS.get(area, "gray"), label=area, edgecolor="black", linewidth=0.3)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels(session_order, rotation=90, fontsize=6)
ax.set_ylabel("labeled cells")
ax.set_title("Labeled reference cells per session (V1/PM) -- context for the pass-rate figure below: "
             "a session with 0 labeled cells here will show 0% pass rate regardless of geometry")
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
save_fig(fig, figdir, "5a_labeled_cells_per_session")


# --------------------------------------------------------------------- #
# 7c. Actual distribution of nearest-labeled-cell distances (not just
#    the binary pass/fail) -- shows whether PROXIMITY_RADIUS_UM sits in
#    a natural gap or cuts through a smooth continuum, i.e. whether 50um
#    is actually a meaningful threshold for this data or an arbitrary
#    one. Pooled across sessions (a distance in microns is comparable
#    across sessions, unlike raw xloc/yloc -- see module docstring).
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
                   label=f"{PROXIMITY_RADIUS_UM:.0f}\u03bcm threshold")
        frac_below = float((sub <= PROXIMITY_RADIUS_UM).mean())
        ax.set_xlabel("distance to nearest labeled same-area cell (\u03bcm)")
        ax.set_ylabel("unlabeled cells")
        ax.set_title(f"{area} (n={len(sub)}, {frac_below:.0%} within threshold)")
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("How unlabeled V1/PM cells are actually distributed relative to the labeled "
                 "population\n(sessions with no labeled cells in an area are excluded here, not shown as 0)",
                 y=1.05)
    fig.tight_layout()
    save_fig(fig, figdir, "5b_distance_to_labeled_distribution")


# --------------------------------------------------------------------- #
# 8. Proximity-filter diagnostic: fraction of unlabeled V1/PM cells
#    passing the filter, per session -- flags sessions where the
#    labeling was sparse/absent for one area (proximity filter fails
#    almost everything there) vs. sessions with good coverage.
# --------------------------------------------------------------------- #
prox_rows = []
for session_id in session_order:
    sdf = celldata[celldata["session_id"] == session_id]  # unfiltered -- want the diagnostic
                                                            # regardless of APPLY_PROXIMITY_FILTER
    for area in PROXIMITY_AREAS:
        unl = sdf[(sdf["roi_name"] == area) & (sdf["labeled"] == "unl")]
        if unl.empty:
            continue
        frac_pass = float(unl["passes_proximity_filter"].mean())
        prox_rows.append({"session_id": session_id, "area": area, "frac_pass": frac_pass, "n": len(unl)})

if prox_rows:
    prox_df = pd.DataFrame(prox_rows)
    fig, ax = plt.subplots(figsize=(max(8, 0.4 * len(session_order)), 4.5))
    width = 0.35
    x = np.arange(len(session_order))
    for i, area in enumerate(PROXIMITY_AREAS):
        sub = prox_df[prox_df["area"] == area].set_index("session_id").reindex(session_order)
        ax.bar(x + (i - 0.5) * width, sub["frac_pass"].to_numpy(), width=width,
               color=AREA_COLORS.get(area, "gray"), label=area, edgecolor="black", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(session_order, rotation=90, fontsize=6)
    ax.set_ylabel(f"fraction of unlabeled cells within {PROXIMITY_RADIUS_UM:.0f}\u03bcm of a labeled cell")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Proximity-filter pass rate per session (V1/PM, unlabeled cells)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save_fig(fig, figdir, "5_proximity_filter_pass_rate")

print(f"\nWrote tables to {paths.out}, figures to {figdir}")
print("Run `python b_progress/make_progress_md.py 2a_cell_distribution` to build a markdown summary.")
