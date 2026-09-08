# -*- coding: utf-8 -*-
"""
celldata_utils.py
===================
Small helpers for working with celldata (one row per recorded cell,
DN sessions only -- see session.py's module docstring): consistent
area/label grouping and colors for plots, and a spatial proximity
filter for unlabeled V1/PM cells.

Mirrors the conventions of the lab's own celldata_utils.py (area order,
colors, `bar_by_group`) so figures here look like the rest of the
project's plots.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

DEFAULT_AREA_ORDER = ["V1", "PM", "AL", "RSP"]
DEFAULT_LABEL_ORDER = ["unl", "lab"]

#: consistent color per area, consistent marker/linestyle per label,
#: reused across this project's cell-distribution/single-cell plots so
#: figures stay visually comparable to each other
AREA_COLORS = {
    "V1": "#1f77b4",
    "PM": "#ff7f0e",
    "AL": "#2ca02c",
    "RSP": "#d62728",
}
LABEL_LINESTYLES = {"unl": "-", "lab": "--"}
LABEL_MARKERS = {"unl": "o", "lab": "^"}

#: areas where deliberate labeling actually happens by this project's
#: design (see make_fake_data.py's recombinase-by-area convention) --
#: pass to get_area_label's label_split_areas to pool AL/RSP's (typically
#: stray) labeled cells into their area's main population instead of
#: showing them as their own near-empty, distorting group
LABEL_SPLIT_AREAS = ("V1", "PM")


def get_area_label(celldata: pd.DataFrame, area_var: str = "roi_name",
                    label_var: str = "labeled", label_split_areas: tuple | None = None) -> tuple:
    """
    Parameters
    ----------
    celldata : pandas.DataFrame (session.celldata)
    area_var : column with the brain area of each cell
    label_var : column already coded as 'unl'/'lab' (this project's
        celldata schema stores it this way directly -- see
        make_fake_data.py -- unlike the lab's own celldata, which
        derives it from a 0/1 `redcell` column)
    label_split_areas : optional iterable of area names to actually
        split by label; every OTHER area's cells are all treated as
        'unl' regardless of their true label, collapsing that area's
        subset (typically tiny/stray -- deliberate labeling only ever
        happens in V1/PM by this project's design, see
        make_fake_data.py's recombinase-by-area convention) into the
        main population, rather than showing it as its own near-empty
        group that distorts variance-based plots (a single stray
        labeled cell produces a bar with a meaningless, enormous error
        bar). None (default) = split every area by label, unchanged
        behavior. Pass e.g. ('V1', 'PM') to only split those.

    Returns
    -------
    area : 1D str array, length N
    label : 1D str array, length N, values in {'unl', 'lab'}
    group : 1D str array, length N, '<area>_<label>' combined group id
    """
    area = celldata[area_var].to_numpy().astype(str)
    label = celldata[label_var].to_numpy().astype(str)
    if label_split_areas is not None:
        pool_mask = ~np.isin(area, list(label_split_areas))
        label = np.where(pool_mask, "unl", label)
    group = np.array([f"{a}_{l}" for a, l in zip(area, label)])
    return area, label, group
    label = celldata[label_var].to_numpy().astype(str)
    group = np.array([f"{a}_{l}" for a, l in zip(area, label)])
    return area, label, group


def ordered_groups(areas_present, labels_present=DEFAULT_LABEL_ORDER,
                    area_order=DEFAULT_AREA_ORDER) -> list:
    """Return (area, label) pairs in a fixed, consistent plotting order,
    restricted to the areas/labels actually present in the data."""
    areas_ordered = [a for a in area_order if a in areas_present]
    areas_ordered += sorted(set(areas_present) - set(areas_ordered))
    return [(a, l) for a in areas_ordered for l in labels_present]


def bar_by_group(ax, values_by_group: dict, ylabel: str, log_y: bool = False,
                  zero_line: bool = True, min_n: int = 1,
                  area_order=DEFAULT_AREA_ORDER, labels_present=DEFAULT_LABEL_ORDER,
                  statistic: str = "mean"):
    """Plot mean +/- SEM (or, with statistic="count", just a bar count --
    no error bars -- for things like cell counts) as a bar per
    (area, label) group, colored by area with the label distinguished by
    the bar's edge linestyle and spelled out in the x-tick text along
    with the group's n.

    Parameters
    ----------
    ax : matplotlib Axes
    values_by_group : dict {(area, label): 1D array-like of values}
    ylabel : str
    log_y, zero_line : see bar_by_group's use elsewhere in this project
    min_n : skip groups with fewer than this many finite values
    statistic : "mean" (default, mean +/- SEM) or "count" (just len(vals),
        no error bar -- for cell-count summaries)
    """
    areas_present = sorted({a for a, l in values_by_group.keys()})
    ordered = [g for g in ordered_groups(areas_present, labels_present=labels_present,
                                          area_order=area_order) if g in values_by_group]

    heights, sems, positions, colors, linestyles, xtick_labels = [], [], [], [], [], []
    for i, (a, l) in enumerate(ordered):
        vals = np.asarray(values_by_group[(a, l)], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < min_n:
            continue
        if statistic == "count":
            heights.append(len(vals))
            sems.append(0.0)
        else:
            heights.append(float(np.mean(vals)))
            sems.append(float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
        positions.append(i)
        colors.append(AREA_COLORS.get(a, "gray"))
        linestyles.append(LABEL_LINESTYLES.get(l, "-"))
        xtick_labels.append(f"{a}\n{l}\n(n={len(vals)})")

    ax.set_ylabel(ylabel)
    if not heights:
        return

    for pos, h, sem, color, ls in zip(positions, heights, sems, colors, linestyles):
        ax.bar(pos, h, yerr=(sem if statistic != "count" else None), width=0.7, color=color,
               edgecolor="black", linestyle=ls, linewidth=1.3, capsize=4, alpha=0.85)

    ax.set_xticks(positions)
    ax.set_xticklabels(xtick_labels, fontsize=7)
    if log_y:
        ax.set_yscale("log")
    elif zero_line:
        ax.axhline(0, color="gray", linewidth=0.7, zorder=0)


def filter_nearlabeled(celldata: pd.DataFrame, radius_um: float = 50.0,
                        areas=("V1", "PM"), area_col: str = "roi_name",
                        label_col: str = "redcell", x_col: str = "xloc",
                        y_col: str = "yloc", session_col: str = "session_id") -> np.ndarray:
    """Boolean mask, same length/index as `celldata`: True for every cell
    EXCEPT unlabeled cells in `areas` that are NOT within `radius_um`
    (Euclidean distance in xloc/yloc, assumed already in microns) of a
    LABELED cell in the SAME area. Labeled cells, and any cell outside
    `areas`, always pass through unaffected.

    `label_col` defaults to 'redcell' (0/1), matching the lab's actual
    ground-truth label column (see cellselection_lib.py -- the string
    'labeled' column this project's own celldata also carries is just a
    convenience derived from the same 0/1 value, and isn't what the real
    codebase's own filters check).

    Distances are computed separately PER SESSION (grouped by
    `session_col`), never pooled across sessions -- xloc/yloc are only
    meaningful within one session's own field of view; a session with no
    labeled reference cells in a given area fails ALL of that area's
    unlabeled cells for that session (nothing to be "near").

    This is PURELY the proximity constraint, deliberately kept
    independent of any layer/depth restriction -- see
    `filter_nearlabeled_layer23` below for the lab's actual combined
    proximity+layer filter (bundling the two together would silently
    throw away the full depth distribution 2a_cell_distribution.py's
    depth/laminar figures want to show).
    """
    is_labeled = celldata[label_col].to_numpy()
    if is_labeled.dtype == object or is_labeled.dtype.kind in "US":
        is_labeled = np.isin(is_labeled, ["lab", "1", "True", "true"])
    else:
        is_labeled = is_labeled.astype(bool)

    keep = pd.Series(True, index=celldata.index)
    for _, sdf in celldata.groupby(session_col):
        sdf_labeled = is_labeled[celldata.index.get_indexer(sdf.index)]
        for area in areas:
            area_mask = (sdf[area_col] == area).to_numpy()
            lab = sdf[area_mask & sdf_labeled]
            unl = sdf[area_mask & ~sdf_labeled]
            if unl.empty:
                continue
            if lab.empty:
                keep.loc[unl.index] = False
                continue
            lab_xy = lab[[x_col, y_col]].to_numpy()
            unl_xy = unl[[x_col, y_col]].to_numpy()
            dists = np.sqrt(((unl_xy[:, None, :] - lab_xy[None, :, :]) ** 2).sum(axis=2))
            min_dist = dists.min(axis=1)
            keep.loc[unl.index[min_dist > radius_um]] = False
    return keep.to_numpy()


def nearest_labeled_distance(celldata: pd.DataFrame, areas=("V1", "PM"),
                              area_col: str = "roi_name", label_col: str = "redcell",
                              x_col: str = "xloc", y_col: str = "yloc",
                              session_col: str = "session_id") -> np.ndarray:
    """Same per-session, same-area logic as `filter_nearlabeled`, but
    returns the actual RAW distance (microns) from each unlabeled cell
    in `areas` to its nearest labeled same-area cell, rather than a
    boolean pass/fail against a fixed radius. NaN for: labeled cells,
    cells outside `areas`, and unlabeled cells in a session/area with no
    labeled reference cells at all (nothing to measure a distance to).

    Exists so the actual DISTRIBUTION of distances can be inspected --
    e.g. plotted as a histogram with the chosen `radius_um` threshold
    drawn on top -- rather than only ever seeing a binary filter
    outcome. Useful both for sanity-checking that a given radius is
    reasonable, and for spotting sessions with poor/absent labeling
    (see 2a_cell_distribution.py's proximity-filter diagnostics)."""
    is_labeled = celldata[label_col].to_numpy()
    if is_labeled.dtype == object or is_labeled.dtype.kind in "US":
        is_labeled = np.isin(is_labeled, ["lab", "1", "True", "true"])
    else:
        is_labeled = is_labeled.astype(bool)

    dist = pd.Series(np.nan, index=celldata.index, dtype=float)
    for _, sdf in celldata.groupby(session_col):
        sdf_labeled = is_labeled[celldata.index.get_indexer(sdf.index)]
        for area in areas:
            area_mask = (sdf[area_col] == area).to_numpy()
            lab = sdf[area_mask & sdf_labeled]
            unl = sdf[area_mask & ~sdf_labeled]
            if unl.empty or lab.empty:
                continue
            lab_xy = lab[[x_col, y_col]].to_numpy()
            unl_xy = unl[[x_col, y_col]].to_numpy()
            dists = np.sqrt(((unl_xy[:, None, :] - lab_xy[None, :, :]) ** 2).sum(axis=2))
            dist.loc[unl.index] = dists.min(axis=1)
    return dist.to_numpy()


def filter_nearlabeled_layer23(celldata: pd.DataFrame, radius: float = 50.0, depth_thr: float = 300.0,
                                lateral_only: bool = True, areas=("V1", "PM"),
                                area_col: str = "roi_name", label_col: str = "redcell",
                                x_col: str = "xloc", y_col: str = "yloc", depth_col: str = "depth",
                                session_col: str = "session_id") -> np.ndarray:
    """Faithful port of the lab's own `filter_nearlabeled_layer23`
    (cellselection_lib.py) -- the combined constraint actually used
    upstream of this project's single-cell/information-theoretic
    scripts (e.g. run_singlecell_infotheory.py, plot_mi_behavior.py):
    unlabeled V1/PM cells must be BOTH (1) within `radius` microns of a
    labeled same-area cell AND (2) superficial (depth < depth_thr).
    Labeled cells and non-V1/PM cells always pass, unaffected.

    Provided alongside (not instead of) `filter_nearlabeled` above so
    you can reproduce the lab's exact established filtering when that's
    the goal, while still having a pure-proximity-only option available
    when you specifically want the full depth range (like
    2a_cell_distribution.py's depth/laminar figures).

    `lateral_only=True` (default, matching the upstream reference):
    proximity uses xloc/yloc only, ignoring depth differences. Set
    False for full 3D (xloc, yloc, depth) distance instead.

    Computed per session (grouped by `session_col`), same reasoning as
    `filter_nearlabeled`."""
    is_labeled = celldata[label_col].to_numpy()
    if is_labeled.dtype == object or is_labeled.dtype.kind in "US":
        is_labeled = np.isin(is_labeled, ["lab", "1", "True", "true"])
    else:
        is_labeled = is_labeled.astype(bool)

    keep = pd.Series(True, index=celldata.index)
    for _, sdf in celldata.groupby(session_col):
        sdf_labeled = is_labeled[celldata.index.get_indexer(sdf.index)]
        for area in areas:
            area_mask = (sdf[area_col] == area).to_numpy()
            constrained_mask = area_mask & ~sdf_labeled
            if not constrained_mask.any():
                continue
            lab = sdf[area_mask & sdf_labeled]
            constrained = sdf[constrained_mask]
            if lab.empty:
                keep.loc[constrained.index] = False
                continue

            xu, yu = constrained[x_col].to_numpy(), constrained[y_col].to_numpy()
            xl, yl = lab[x_col].to_numpy(), lab[y_col].to_numpy()
            if lateral_only:
                d = np.sqrt((xu[:, None] - xl[None, :]) ** 2 + (yu[:, None] - yl[None, :]) ** 2)
            else:
                zu, zl = constrained[depth_col].to_numpy(), lab[depth_col].to_numpy()
                d = np.sqrt((xu[:, None] - xl[None, :]) ** 2 + (yu[:, None] - yl[None, :]) ** 2
                            + (zu[:, None] - zl[None, :]) ** 2)
            near_ok = (d <= radius).any(axis=1)
            keep.loc[constrained.index] = near_ok

    # Layer constraint -- ONLY for the constrained (unlabeled, target-area)
    # population, ANDed with (not overwriting) the proximity result above.
    is_constrained = np.isin(celldata[area_col].to_numpy(), areas) & ~is_labeled
    depth = celldata[depth_col].to_numpy().astype(float)
    keep_arr = keep.to_numpy().copy()  # .copy(): some pandas versions return a read-only
                                        # view from Series.to_numpy(), which the in-place
                                        # &= below would otherwise fail on
    keep_arr[is_constrained] &= depth[is_constrained] < depth_thr
    return keep_arr


def filter_target_groups(area, label, target_areas=None, target_labels=None) -> np.ndarray:
    """Boolean mask selecting only cells belonging to the requested
    (area, label) combinations -- the "which groups do I want" argument
    shared across downstream analysis scripts, so the same selector
    works everywhere. Direct port of the lab's own
    cellselection_lib.filter_target_groups.

    Parameters
    ----------
    area, label : array-like, same length -- typically the output of
        `get_area_label` above.
    target_areas : list of str or None -- e.g. ['V1', 'PM'], or None
        for "all areas, no restriction".
    target_labels : list of str or None -- e.g. ['lab'], or None for
        "all labels, no restriction".

    Returns
    -------
    idx_target : boolean numpy array, len(area) -- True for cells
        matching the requested area(s) AND label(s). Combine with other
        filters (QC, proximity, ...) via logical AND, e.g.:
            idx_valid = idx_qc & filter_nearlabeled(celldata) & \\
                filter_target_groups(area, label, target_areas=['V1', 'PM'])

    Examples
    --------
    filter_target_groups(area, label)                                  # everything
    filter_target_groups(area, label, target_labels=['lab'])           # labeled cells only, any area
    filter_target_groups(area, label, target_areas=['PM'])             # PM only, any label
    filter_target_groups(area, label, target_areas=['PM'], target_labels=['lab'])  # PM labeled only
    """
    area = np.asarray(area)
    label = np.asarray(label)
    idx = np.ones(len(area), dtype=bool)
    if target_areas is not None:
        idx &= np.isin(area, target_areas)
    if target_labels is not None:
        idx &= np.isin(label, target_labels)
    return idx


def sample_pairs_within_groups(area, label, max_pairs_per_group: int = 150, rng=None):
    """Randomly sample neuron-index pairs (i, j), i < j, WITHIN each
    (area, label) group -- see the lab's own celldata_utils.py for the
    full rationale (kept here for parity, not currently used by
    2a_cell_distribution.py, but future single-cell steps will want it)."""
    if rng is None:
        rng = np.random.default_rng()

    pairs, pair_group = [], []
    for a in np.unique(area):
        for l in DEFAULT_LABEL_ORDER:
            idx = np.where((area == a) & (label == l))[0]
            if len(idx) < 2:
                continue
            all_pairs = list(itertools.combinations(idx, 2))
            if len(all_pairs) > max_pairs_per_group:
                chosen = rng.choice(len(all_pairs), size=max_pairs_per_group, replace=False)
                all_pairs = [all_pairs[k] for k in chosen]
            pairs.extend(all_pairs)
            pair_group.extend([(a, l)] * len(all_pairs))
    return pairs, pair_group
