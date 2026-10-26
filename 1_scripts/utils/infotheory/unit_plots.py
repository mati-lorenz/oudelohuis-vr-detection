# -*- coding: utf-8 -*-
"""
unit_plots.py
==============
Plot helpers that respect the chosen unit of analysis (see project_config.AGGREGATE_BY / unit_stats):
    pooled  : one bar per category (the pooled value), no dots, no error bars
    session : bar = mean over sessions, error = SEM over sessions, dots = single sessions
    mouse   : bar = mean over mice,     error = SEM over mice,     dots = single mice
Dots are coloured by mouse (consistent colours across figures) whenever the unit table has `animal_id`.
No violin plots anywhere.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def mouse_palette(animals) -> dict:
    """Stable colour per mouse (sorted ids -> tab20 cycle)."""
    animals = sorted(set(animals))
    cmap = plt.cm.tab20(np.linspace(0, 1, 20))
    return {a: cmap[i % 20] for i, a in enumerate(animals)}


def unit_bars(ax, units, x, y, order, aggregate, colors=None, mouse_colors=None, jitter=0.18, width=0.7,
              ylabel=None, hline=None):
    """Bars/dots of column `y` for each category in `order` (values of column `x`) from a unit table
    (one row per unit and category)."""
    rng = np.random.default_rng(0)
    for xi, cat in enumerate(order):
        sub = units[units[x] == cat]
        vals = sub[y].dropna()
        if not len(vals):
            continue
        col = colors.get(cat) if isinstance(colors, dict) else colors
        col = "#8da0cb" if col is None else col
        if aggregate == "pooled":
            ax.bar(xi, vals.mean(), width=width, color=col, alpha=0.8, edgecolor="black", linewidth=0.4)
            continue
        sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        ax.bar(xi, vals.mean(), width=width, color=col, alpha=0.45, edgecolor="black", linewidth=0.4,
               yerr=sem, error_kw=dict(lw=0.9, capsize=2))
        dot_c = ([mouse_colors.get(a, "k") for a in sub.loc[vals.index, "animal_id"]]
                 if mouse_colors is not None and "animal_id" in sub else "black")
        ax.scatter(xi + rng.uniform(-jitter, jitter, len(vals)), vals, s=16, c=dot_c, edgecolor="black",
                   linewidth=0.3, zorder=3)
    if hline is not None:
        ax.axhline(hline, color="grey", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([str(o) for o in order])
    if ylabel:
        ax.set_ylabel(ylabel)


def mouse_legend(ax, mouse_colors, animals=None, max_n=14, **kw):
    animals = sorted(animals if animals is not None else mouse_colors)
    if not animals or len(animals) > max_n:
        return
    ax.legend(handles=[Line2D([0], [0], marker="o", ls="", markerfacecolor=mouse_colors[a],
                              markeredgecolor="black", markersize=5, label=a) for a in animals],
              fontsize=6, frameon=False, title="mouse", title_fontsize=7, **kw)


def stacked_outcomes(ax, units, aggregate, protocols, colors, label_size=None):
    """Stacked HIT/MISS/FA/CR fractions (from the unit table's n_<outcome>_all counts), one bar per unit
    (pooled: one per protocol). Returns nothing; sets labels/legend."""
    import pandas as pd
    outs = ("HIT", "MISS", "FA", "CR")
    u = units.copy()
    cols = [f"n_{o}_all" for o in outs]
    tot = u[cols].sum(axis=1).replace(0, np.nan)
    for o, c in zip(outs, cols):
        u[o] = u[c] / tot
    u["protocol"] = pd.Categorical(u["protocol"], protocols, ordered=True)
    u = u.sort_values(["protocol"] + [c for c in ("animal_id", "session_id") if c in u.columns])
    labels = (u["protocol"].astype(str) if aggregate == "pooled" else
              u["protocol"].astype(str) + " " + (u["session_id"] if aggregate == "session" else u["animal_id"]))
    bottom = np.zeros(len(u))
    for o in outs:
        v = u[o].fillna(0).to_numpy()
        ax.bar(range(len(u)), v, bottom=bottom, color=colors[o], label=o, width=0.85)
        bottom += v
    ax.set_xticks(range(len(u)))
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=label_size or (6 if len(u) > 12 else 8))
    ax.set_ylabel("fraction of catch trials")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)


def hbar_units(ax, df, label_col, value_col, aggregate, colors=None, mouse_colors=None, sort=True):
    """Horizontal bars, one per label (e.g. variable pair / target). `df` has one row per (label, unit).
    Bar = mean over units (+- SEM over units); dots = single units (coloured by mouse when `animal_id`
    exists); pooled: one bar per label, no dots. Sorted ascending by the bar value. Returns the label order."""
    stats = df.groupby(label_col)[value_col].agg(["mean", "sem", "count"])
    if sort:
        stats = stats.sort_values("mean")
    labels = list(stats.index)
    rng = np.random.default_rng(0)
    for yi, lab in enumerate(labels):
        col = colors.get(lab) if isinstance(colors, dict) else colors
        col = "#8da0cb" if col is None else col
        m, sem = stats.loc[lab, "mean"], stats.loc[lab, "sem"]
        pooled = aggregate == "pooled"
        ax.barh(yi, m, color=col, alpha=0.8 if pooled else 0.5, edgecolor="black", linewidth=0.4,
                xerr=None if pooled or not np.isfinite(sem) else sem, error_kw=dict(lw=0.8, capsize=1.5))
        if not pooled:
            sub = df[df[label_col] == lab].dropna(subset=[value_col])
            dc = ([mouse_colors.get(a, "k") for a in sub["animal_id"]]
                  if mouse_colors is not None and "animal_id" in sub else "black")
            ax.scatter(sub[value_col], yi + rng.uniform(-0.22, 0.22, len(sub)), s=9, c=dc, edgecolor="black",
                       linewidth=0.25, zorder=3)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    return labels


def group_unit_bars(ax, cells, value, aggregate, mouse_colors=None, min_cells_per_unit=1, log_y=False,
                    hline=None, ylabel=None, tick_size=7):
    """One bar per project group (always the SIX groups of celldata_utils.SIX_GROUP_ORDER, empty ones left
    blank) for a per-CELL variable, at the chosen unit of analysis. `cells` has one row per cell with columns
    `group` ('V1_lab' ...), `value`, and session_id / animal_id.
        pooled : bar = mean over all cells of the group, no error bar
        session: each session's mean over its cells of the group (sessions with fewer than
                 `min_cells_per_unit` cells in the group are left out); bar = mean +- SEM over sessions,
                 dots = sessions coloured by mouse
        mouse  : the same with a mouse's sessions pooled first
    Tick labels carry n (cells when pooled, else units)."""
    from .celldata_utils import SIX_GROUP_ORDER, SIX_GROUP_LABELS, SIX_GROUP_COLORS
    keys = {"pooled": [], "session": ["animal_id", "session_id"], "mouse": ["animal_id"]}[aggregate]
    d = cells.dropna(subset=[value])
    g = d.groupby(["group"] + keys)[value].agg(["mean", "size"]).reset_index().rename(columns={"mean": value})
    g = g[g["size"] >= (1 if aggregate == "pooled" else min_cells_per_unit)]
    order = [f"{a}_{l}" for a, l in SIX_GROUP_ORDER]
    colors = {f"{a}_{l}": SIX_GROUP_COLORS[(a, l)] for a, l in SIX_GROUP_ORDER}
    unit_bars(ax, g, "group", value, order, aggregate, colors=colors, mouse_colors=mouse_colors)
    labels = []
    for (a, l), key in zip(SIX_GROUP_ORDER, order):
        sub = g[g["group"] == key]
        n = int(sub["size"].sum()) if aggregate == "pooled" else len(sub)
        labels.append(f"{SIX_GROUP_LABELS[(a, l)]}\n(n={n})")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, fontsize=tick_size)
    if log_y:
        ax.set_yscale("log")
    if hline is not None:
        ax.axhline(hline, color="grey", linestyle="--", linewidth=1)
    if ylabel:
        ax.set_ylabel(ylabel)
    return g


def _draw_grouped(ax, g, aggregate, series_col, series_order, series_colors, mouse_colors, hline, ylabel,
                  tick_size, jitter=0.06):
    """Draw grouped bars from a table `g` with columns group, `series_col`, mean (value of one unit; pooled: of the
    whole group), size (cells in it) and, for session / mouse, animal_id."""
    from .celldata_utils import SIX_GROUP_ORDER, SIX_GROUP_LABELS
    order = [f"{a}_{l}" for a, l in SIX_GROUP_ORDER]
    n_s = max(len(series_order), 1)
    width = 0.8 / n_s
    rng = np.random.default_rng(0)
    cmap = plt.get_cmap("tab10")
    colors = {s_: (series_colors or {}).get(s_, cmap(i % 10)) for i, s_ in enumerate(series_order)}
    ns = []
    for gi, key in enumerate(order):
        n_group = 0
        for si, s_ in enumerate(series_order):
            sub = g[(g["group"] == key) & (g[series_col] == s_)]
            sub = sub[np.isfinite(sub["mean"])]
            if sub.empty:
                continue
            vals = sub["mean"].to_numpy(float)
            x = gi + (si - (n_s - 1) / 2) * width
            n_group = max(n_group, int(sub["size"].sum()) if aggregate == "pooled" else len(sub))
            if aggregate == "pooled":
                ax.bar(x, vals.mean(), width=width * 0.92, color=colors[s_], alpha=0.85, edgecolor="black",
                       linewidth=0.3)
                continue
            sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            ax.bar(x, vals.mean(), width=width * 0.92, color=colors[s_], alpha=0.5, edgecolor="black",
                   linewidth=0.3, yerr=sem, error_kw=dict(lw=0.8, capsize=1.5))
            dc = ([mouse_colors.get(a, "k") for a in sub["animal_id"]] if mouse_colors is not None else "black")
            ax.scatter(x + rng.uniform(-jitter, jitter, len(vals)), vals, s=10, c=dc, edgecolor="black",
                       linewidth=0.25, zorder=3)
        ns.append(n_group)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{SIX_GROUP_LABELS[gr]}\n(n={n})" for gr, n in zip(SIX_GROUP_ORDER, ns)], fontsize=tick_size)
    if hline is not None:
        ax.axhline(hline, color="grey", linestyle="--", linewidth=1)
    if ylabel:
        ax.set_ylabel(ylabel)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[s_], alpha=0.7) for s_ in series_order]
    return handles, [str(s_) for s_ in series_order]


def grouped_series_bars(ax, cells, value, series_col, series_order, aggregate, series_colors=None,
                        mouse_colors=None, min_cells_per_unit=1, hline=None, ylabel=None, tick_size=7,
                        jitter=0.06):
    """Like group_unit_bars, but with SEVERAL bars per project group, one per entry of `series_col`
    (e.g. engaged vs disengaged trials, or the ROC analyses). `cells` has one row per cell and series
    with columns `group`, `value`, `series_col`, session_id / animal_id. Same unit logic: pooled = mean over
    cells (no error); session / mouse = mean over the unit's cells first, then bar = mean +- SEM over units,
    dots = units coloured by mouse. Tick labels carry n (cells when pooled, else units, max over series)."""
    keys = {"pooled": [], "session": ["animal_id", "session_id"], "mouse": ["animal_id"]}[aggregate]
    d = cells.dropna(subset=[value])
    g = d.groupby(["group", series_col] + keys)[value].agg(["mean", "size"]).reset_index()
    g = g[g["size"] >= (1 if aggregate == "pooled" else min_cells_per_unit)]
    return _draw_grouped(ax, g, aggregate, series_col, series_order, series_colors, mouse_colors, hline, ylabel,
                         tick_size, jitter)


def grouped_unit_value_bars(ax, units, value, series_col, series_order, aggregate, series_colors=None,
                            mouse_colors=None, hline=None, ylabel=None, tick_size=7):
    """grouped_series_bars for values that are ALREADY one number per analysis unit (e.g. a miss fraction computed
    as a ratio of means). `units`: rows (group, series, unit) with `value`, `n_cells` and, for session / mouse,
    animal_id; pooled: one row per (group, series)."""
    g = units.rename(columns={value: "mean", "n_cells": "size"})
    return _draw_grouped(ax, g, aggregate, series_col, series_order, series_colors, mouse_colors, hline, ylabel,
                         tick_size)


def group_panels(df, panel_col, panel_order, value, aggregate, figdir, name, title, ylabel=None, ncols=3,
                 hline=None, series_col=None, series_order=None, series_colors=None, mouse_colors=None,
                 min_cells_per_unit=1, panel_size=(5.0, 3.4), sharey=False):
    """One figure, one panel per entry of `panel_col` (e.g. predictor or analysis), each panel = the six project
    groups at the chosen unit of analysis (group_unit_bars; with `series_col` several bars per group,
    grouped_series_bars). sharey=True gives every panel the same y range. `df`: one row per cell (and series) with group, value, session_id, animal_id."""
    from .plotting import save_fig
    if df.empty:
        return
    nrows = int(np.ceil(len(panel_order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_size[0] * ncols, panel_size[1] * nrows), squeeze=False)
    handles = None
    for ax, p in zip(axes.flat, panel_order):
        s = df[df[panel_col] == p]
        ax.set_title(str(p), fontsize=9)
        if s.dropna(subset=[value]).empty:
            ax.text(0.5, 0.5, "no valid values", transform=ax.transAxes, ha="center", fontsize=8, color="grey")
            ax.set_xticks([])
            continue
        if series_col is None:
            group_unit_bars(ax, s, value, aggregate, mouse_colors=mouse_colors,
                            min_cells_per_unit=min_cells_per_unit, hline=hline, ylabel=ylabel)
        else:
            handles = grouped_series_bars(ax, s, value, series_col, series_order, aggregate,
                                          series_colors=series_colors, mouse_colors=mouse_colors,
                                          min_cells_per_unit=min_cells_per_unit, hline=hline, ylabel=ylabel)
    for ax in list(axes.flat)[len(panel_order):]:
        ax.axis("off")
    if sharey:      # one common y range for all panels that drew something (the union of their autoscaled ranges)
        drawn = [ax for ax in list(axes.flat)[:len(panel_order)] if ax.has_data()]
        if drawn:
            lo = min(ax.get_ylim()[0] for ax in drawn)
            hi = max(ax.get_ylim()[1] for ax in drawn)
            for ax in drawn:
                ax.set_ylim(lo, hi)
    if handles:
        axes[0, -1].legend(*handles, fontsize=7, frameon=False, loc="upper right")
    elif aggregate != "pooled" and mouse_colors is not None and "animal_id" in df:
        mouse_legend(axes[0, -1], mouse_colors, animals=df["animal_id"].unique(), loc="upper right")
    fig.suptitle(title, y=1.0)
    fig.tight_layout()
    save_fig(fig, figdir, name)
    plt.close(fig)
