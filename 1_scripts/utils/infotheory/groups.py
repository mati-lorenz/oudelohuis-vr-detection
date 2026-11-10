# -*- coding: utf-8 -*-
"""
groups.py
==========
The six-group layout (V1-lab, V1-unl, PM-lab, PM-unl, AL, RSP) used by
every population-level figure in this project, as small reusable helpers
so each script does not re-implement the same loop.

Group keys are the tuples in celldata_utils.SIX_GROUP_ORDER, e.g.
("V1", "lab"). AL/RSP labeled cells are pooled into "unl"
(celldata_utils.LABEL_SPLIT_AREAS).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import project_config as cfg
from .celldata_utils import (SIX_GROUP_ORDER, SIX_GROUP_LABELS, SIX_GROUP_COLORS,
                             LABEL_SPLIT_AREAS, get_area_label, classify_depth)


def group_meta_columns(celldata_combined: pd.DataFrame) -> list:
    """Columns of 2a's celldata_combined.csv that add_six_group_column needs besides the ids: roi_name, labeled and the depth
    information (`depth_class` when 2a wrote it, else the raw `depth`)."""
    cols = ["roi_name", "labeled"]
    if "depth_class" in celldata_combined.columns:
        return cols + ["depth_class"]
    if "depth" in celldata_combined.columns:
        return cols + ["depth"]
    return cols


def add_six_group_column(df: pd.DataFrame, area_var: str = "roi_name",
                         label_var: str = "labeled") -> pd.DataFrame:
    """Add `area`, `label`, `depth_class` and `group` (the SIX_GROUP_ORDER key as a string like 'V1_lab'; AL/RSP ->
    'AL_unl'/'RSP_unl'; deep PM -> 'PMdeep_lab'/'PMdeep_unl'; cells under no group -> 'excluded') plus `group_name` (display
    name, e.g. 'V1-lab', 'AL', 'PM-lab (deep)'). `area` stays the real area ('PM' for deep PM cells).

    Depth class: the table's `depth_class` column where it is filled, else computed from `depth`. If the depth rule is active
    (project_config.MAX_DEPTH_UM) and the table has neither, this raises -- silently mixing deep PM cells into PM is worse than
    stopping (subset 2a's table with group_meta_columns())."""
    area, label, group = get_area_label(df, area_var=area_var, label_var=label_var,
                                        label_split_areas=LABEL_SPLIT_AREAS)
    out = df.copy()
    out["area"], out["label"] = area, label
    if cfg.MAX_DEPTH_UM is None:
        out["depth_class"] = "shallow"
    else:
        dc = out["depth_class"].astype(object) if "depth_class" in out.columns else pd.Series(np.nan, index=out.index, dtype=object)
        missing = dc.isna().to_numpy()
        if missing.any():
            if "depth" not in out.columns:
                raise ValueError("add_six_group_column: the depth rule (project_config.MAX_DEPTH_UM) needs a filled 'depth_class' "
                                 "(2a's celldata_combined.csv, rerun 2a) or a 'depth' column in the table; neither is available "
                                 f"for {int(missing.sum())} cells")
            dc = dc.copy()
            dc[missing] = classify_depth(area[missing], out["depth"].to_numpy(float)[missing])
        out["depth_class"] = dc.to_numpy(dtype=object)
        dcv = out["depth_class"].to_numpy()
        group = np.where(dcv == "deep_pm", np.array([f"PMdeep_{l}" for l in label]), group)
        group = np.where(dcv == "excluded", "excluded", group)
    out["group"] = group
    name_by_group = {f"{a}_{l}": SIX_GROUP_LABELS[(a, l)] for a, l in SIX_GROUP_ORDER}
    out["group_name"] = out["group"].map(name_by_group)
    return out


def group_key(group: tuple) -> str:
    return f"{group[0]}_{group[1]}"


def six_panel_figure(figsize_per_panel=(3.2, 3.0), sharex=True, sharey=True, ncols=None):
    """Figure with one panel per SIX_GROUP_ORDER entry (6, or 8 with the deep-PM groups; default 3 resp. 4 columns). Returns
    (fig, {group_tuple: ax}). Always one panel per group, even if a group is empty."""
    if ncols is None:
        ncols = 3 if len(SIX_GROUP_ORDER) <= 6 else 4
    nrows = int(np.ceil(len(SIX_GROUP_ORDER) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_panel[0] * ncols,
                                                    figsize_per_panel[1] * nrows),
                             sharex=sharex, sharey=sharey, squeeze=False)
    ax_by_group = {g: axes.flat[i] for i, g in enumerate(SIX_GROUP_ORDER)}
    for ax in axes.flat[len(SIX_GROUP_ORDER):]:
        ax.axis("off")
    for g, ax in ax_by_group.items():
        ax.set_title(SIX_GROUP_LABELS[g], color=SIX_GROUP_COLORS[g], fontsize=10)
    return fig, ax_by_group


def split_by_group(df: pd.DataFrame):
    """Yield (group_tuple, sub_df) for ALL six groups (empty frame if none)."""
    for g in SIX_GROUP_ORDER:
        yield g, df[df["group"] == group_key(g)]


def summarize_significance(df: pd.DataFrame, keys: list, value_col: str, sig_col: str = "significant",
                           sign_col: str | None = None, sign_ref: float = 0.0) -> pd.DataFrame:
    """Count/fraction of significant cells per `keys` combination.

    value_col : per-cell effect size (AUC-derived SI, correlation r ...)
    sign_col  : column whose sign relative to `sign_ref` splits significant
        cells into positive / negative (default: value_col vs sign_ref)
    Returns n_cells, n_sig, n_sig_pos, n_sig_neg, frac_sig, mean_abs_<value>, median_<value>.
    """
    d = df[df["group"].isin([f"{a}_{l}" for a, l in SIX_GROUP_ORDER])].copy() if "group" in df.columns else df.copy()
    sc = sign_col or value_col
    d["_pos"] = d[sig_col] & (d[sc] > sign_ref)
    d["_neg"] = d[sig_col] & (d[sc] < sign_ref)
    d["_abs"] = d[value_col].abs()
    g = d.groupby(keys, observed=True)
    out = g.agg(n_cells=(sig_col, "size"), n_sig=(sig_col, "sum"), n_sig_pos=("_pos", "sum"),
                n_sig_neg=("_neg", "sum"), **{f"mean_abs_{value_col}": ("_abs", "mean"),
                                              f"median_{value_col}": (value_col, "median")}).reset_index()
    out["frac_sig"] = out["n_sig"] / out["n_cells"]
    return out
