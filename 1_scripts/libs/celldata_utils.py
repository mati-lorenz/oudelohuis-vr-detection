# -*- coding: utf-8 -*-
"""
celldata_utils.py
===================
Small helper to consistently group neurons by brain area (`roi_name` in
celldata, e.g. 'V1', 'PM', 'AL', 'RSP') and labeling status ('redcell' in
celldata: 1 = labeled/recombinase+ cell, 0 = unlabeled), used by both
plotting scripts to keep area/label grouping and colors consistent.
"""

import numpy as np

DEFAULT_AREA_ORDER = ['V1', 'PM', 'AL', 'RSP']
DEFAULT_LABEL_ORDER = ['unl', 'lab']

#: consistent color per area, consistent linestyle per label, reused by
#: both plotting scripts so figures are visually comparable
AREA_COLORS = {
    'V1': '#1f77b4',
    'PM': '#ff7f0e',
    'AL': '#2ca02c',
    'RSP': '#d62728',
}
LABEL_LINESTYLES = {'unl': '-', 'lab': '--'}
LABEL_ALPHAS = {'unl': 0.9, 'lab': 0.9}


def get_area_label(celldata, area_var='roi_name', label_var='redcell'):
    """
    Parameters
    ----------
    celldata : pandas.DataFrame (session.celldata)
    area_var : column with the brain area of each cell
    label_var : boolean/0-1 column, 1 = labeled ('lab'), 0 = unlabeled ('unl')

    Returns
    -------
    area : 1D str array, length N
    label : 1D str array, length N, values in {'unl', 'lab'}
    group : 1D str array, length N, '<area>_<label>' combined group id
    """
    area = celldata[area_var].to_numpy().astype(str)
    is_lab = celldata[label_var].to_numpy().astype(bool)
    label = np.where(is_lab, 'lab', 'unl')
    group = np.array([f'{a}_{l}' for a, l in zip(area, label)])
    return area, label, group


def ordered_groups(areas_present, labels_present=DEFAULT_LABEL_ORDER,
                    area_order=DEFAULT_AREA_ORDER):
    """
    Return (area, label) pairs in a fixed, consistent plotting order,
    restricted to the areas/labels actually present in the data.
    """
    areas_ordered = [a for a in area_order if a in areas_present]
    areas_ordered += sorted(set(areas_present) - set(areas_ordered))
    pairs = [(a, l) for a in areas_ordered for l in labels_present]
    return pairs


def sample_pairs_within_groups(area, label, max_pairs_per_group=150, rng=None):
    """
    Randomly sample neuron-index pairs (i, j), i < j, WITHIN each
    (area, label) group (i.e. both neurons of a pair share the same area
    and label) -- the natural pairing choice for the Pola et al. (2003)
    breakdown when you want results split by area/label, since the
    breakdown itself is defined for a single pair, and pairing across
    heterogeneous groups (e.g. a V1-unl cell with a PM-lab cell) would
    make signal-similarity/correlation terms hard to interpret as
    belonging to one area or label.

    Parameters
    ----------
    area, label : 1D arrays, length N, as returned by `get_area_label`
    max_pairs_per_group : int, cap per (area, label) group (all pairs
        used if the group has fewer neurons than would exceed this cap)
    rng : numpy.random.Generator (created fresh with default seeding if
        None)

    Returns
    -------
    pairs : list of (i, j) tuples
    pair_group : list of (area, label) tuples, same length as `pairs`,
        the group each pair belongs to
    """
    import itertools
    if rng is None:
        rng = np.random.default_rng()

    pairs = []
    pair_group = []
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


def bar_by_group(ax, values_by_group, ylabel, log_y=False, zero_line=True,
                  min_n=1, area_order=DEFAULT_AREA_ORDER, labels_present=DEFAULT_LABEL_ORDER):
    """
    Plot mean +/- SEM as a bar per (area, label) group along the
    x-axis, colored by area (AREA_COLORS) with the label distinguished
    by the bar's edge linestyle (LABEL_LINESTYLES) and spelled out in
    the x-tick text along with the group's n. The simplest, most
    directly comparable way to lay out a summary statistic across many
    groups.

    Parameters
    ----------
    ax : matplotlib Axes
    values_by_group : dict {(area, label): 1D array-like of values}
    ylabel : str
    log_y : bool -- use a log-scaled value axis (e.g. for inter-event
        intervals, which span orders of magnitude). Note bars on a log
        axis don't visually "start from zero" (zero isn't on a log
        scale) -- this is normal/expected for log-scale bar charts.
    zero_line : bool -- draw a horizontal reference line at 0 (skipped
        automatically if log_y=True, since 0 isn't on a log axis)
    min_n : int -- skip groups with fewer than this many finite values
    area_order, labels_present : passed to `ordered_groups` to control
        left-to-right group ordering

    Returns
    -------
    None (draws in place). If no group has >= min_n finite values,
    leaves the axes empty except the ylabel.
    """
    areas_present = sorted({a for a, l in values_by_group.keys()})
    ordered = [g for g in ordered_groups(areas_present, labels_present=labels_present,
                                          area_order=area_order) if g in values_by_group]

    means, sems, positions, colors, linestyles, xtick_labels = [], [], [], [], [], []
    for i, (a, l) in enumerate(ordered):
        vals = np.asarray(values_by_group[(a, l)], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < min_n:
            continue
        means.append(float(np.mean(vals)))
        sems.append(float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
        positions.append(i)
        colors.append(AREA_COLORS.get(a, 'gray'))
        linestyles.append(LABEL_LINESTYLES.get(l, '-'))
        xtick_labels.append(f'{a}\n{l}\n(n={len(vals)})')

    ax.set_ylabel(ylabel)
    if not means:
        return

    for pos, mean, sem, color, ls in zip(positions, means, sems, colors, linestyles):
        if log_y:
            # matplotlib can't draw an error bar below zero on a log axis;
            # clip the lower whisker just short of the bar's own value
            lower = min(sem, mean * 0.999) if mean > 0 else 0.0
            yerr = [[lower], [sem]]
        else:
            yerr = sem
        ax.bar(pos, mean, yerr=yerr, width=0.7, color=color, edgecolor='black',
               linestyle=ls, linewidth=1.3, capsize=4, alpha=0.85)

    ax.set_xticks(positions)
    ax.set_xticklabels(xtick_labels, fontsize=7)
    if log_y:
        ax.set_yscale('log')
    elif zero_line:
        ax.axhline(0, color='gray', linewidth=0.7, zorder=0)
