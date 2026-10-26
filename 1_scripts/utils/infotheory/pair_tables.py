# -*- coding: utf-8 -*-
"""
pair_tables.py
===============
Shared by 3a (pair correlations), 3b (information breakdown) and 3c (PID): ONE definition of which cell PAIRS are
analysed and which TRIALS enter each (pair, target, window), so the three scripts agree before bias correction
(project rule 4, project_config.py).

PAIRS      3a samples them ONCE per session (not per window / target), inside each (area, label) group only, and writes
           them to out/pair_list.csv with a random `pair_rank` within the group. 3b and 3c READ that file (Rule #1: other
           steps' out/) and use the pairs with pair_rank < --max-pairs, so a smaller budget is always a subset of the
           same pairs and every script describes the same pairs.
TRIALS     `pair_trials`: trials where both cells and the target label are finite; target bins (stimulus strategies
           `stim__<strategy>`, or `choice`) with fewer than MIN_TRIALS_PER_TARGET_BIN of those trials are dropped; at least
           MIN_TRIALS_ANALYSIS trials and 2 distinct labels must remain -- exactly the rule of cell_tables.pair_arrays
           (2c/2d), applied to two cells.
RESPONSE   window mean of the activity (as 2c/2d); information measures cut it into RESPONSE_BINS quantile bins.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import project_config as cfg
from . import cell_tables as ct
from .celldata_utils import get_area_label, sample_pairs_within_groups, LABEL_SPLIT_AREAS
from .stim_binning import CHOICE_TARGET

PAIR_COLUMNS = ["session_id", "animal_id", "roi_name", "labeled_grouped", "cell_i", "cell_j", "pair_rank"]


def target_column(target: str) -> str:
    """Table column holding a target's labels: `choice`, or `stim__<strategy>`."""
    return CHOICE_TARGET if target == CHOICE_TARGET else ct.stim_predictor(target)


def add_target_columns(table: pd.DataFrame, strategies) -> pd.DataFrame:
    return ct.add_stimulus_columns(table, strategies)


def sample_session_pairs(celldata: pd.DataFrame, session_id: str, rng: np.random.Generator, max_pairs: int) -> pd.DataFrame:
    """Pairs of the (already filtered) cells of one session, within each (area, label) group, `max_pairs` per group,
    with a random `pair_rank` inside the group (0 = first)."""
    celldata = celldata.reset_index(drop=True)
    area, label, _ = get_area_label(celldata, label_split_areas=LABEL_SPLIT_AREAS)
    pairs, groups = sample_pairs_within_groups(area, label, max_pairs_per_group=max_pairs, rng=rng)
    if not pairs:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    df = pd.DataFrame({"session_id": session_id, "animal_id": session_id.split("_")[0],
                       "roi_name": [g[0] for g in groups], "labeled_grouped": [g[1] for g in groups],
                       "cell_i": [celldata["cell_id"].iloc[i] for i, _ in pairs],
                       "cell_j": [celldata["cell_id"].iloc[j] for _, j in pairs]})
    df["pair_rank"] = df.groupby(["roi_name", "labeled_grouped"])["cell_i"].transform(
        lambda s: rng.permutation(len(s)))
    return df[PAIR_COLUMNS]


def pair_trials(table: pd.DataFrame, target: str, cell_i, cell_j, min_trials: int = cfg.MIN_TRIALS_ANALYSIS,
                min_per_bin: int = cfg.MIN_TRIALS_PER_TARGET_BIN):
    """dict(x1, x2, label, choice, trial) for one (pair, target, window), or None -- see the module docstring."""
    col = target_column(target)
    if col not in table.columns or cell_i not in table.columns or cell_j not in table.columns:
        return None
    x1 = table[cell_i].to_numpy(float)
    x2 = table[cell_j].to_numpy(float)
    lab = table[col].to_numpy(float)
    ok = np.isfinite(x1) & np.isfinite(x2) & np.isfinite(lab)
    if ok.any():
        vals, counts = np.unique(lab[ok], return_counts=True)
        small = vals[counts < min_per_bin]
        if len(small):
            ok &= ~np.isin(lab, small)
    if ok.sum() < min_trials or len(np.unique(lab[ok])) < 2:
        return None
    return dict(x1=x1[ok], x2=x2[ok], label=lab[ok],
                choice=table["choice"].to_numpy(float)[ok] if "choice" in table.columns else None,
                trial=table["trialNumber"].to_numpy()[ok] if "trialNumber" in table.columns else np.flatnonzero(ok))


def finalize_pair_table(df: pd.DataFrame) -> pd.DataFrame:
    """Add group / group_name (six project groups) and animal_id."""
    from .groups import add_six_group_column
    out = add_six_group_column(df, area_var="roi_name", label_var="labeled_grouped")
    if "animal_id" not in out.columns:
        out["animal_id"] = out["session_id"].str.split("_").str[0]
    return out


def load_pair_list(pair_list_path, session_ids, max_pairs: int) -> pd.DataFrame:
    """3a's pair list restricted to `session_ids` and pair_rank < max_pairs."""
    pl = pd.read_csv(pair_list_path)
    return pl[pl["session_id"].isin(set(session_ids)) & (pl["pair_rank"] < max_pairs)].reset_index(drop=True)


def stacked_group_bars(ax, df, comps, colors, aggregate, mouse_colors=None, min_pairs_per_unit=1, ylabel=None,
                       total=None):
    """Stacked bars of the mean of each component in `comps` per project group, at the chosen unit: pooled = mean over
    all pairs; session / mouse = mean over the unit's pairs first (units with fewer than min_pairs_per_unit pairs are
    left out), bar = mean over units. Dots (non-pooled) = each unit's value of `total` (default: the sum of the
    components), coloured by mouse. Negative components stack downwards."""
    from .celldata_utils import SIX_GROUP_ORDER, SIX_GROUP_LABELS
    keys = {"pooled": [], "session": ["animal_id", "session_id"], "mouse": ["animal_id"]}[aggregate]
    d = df.copy()
    d["_total"] = d[total] if total else d[comps].sum(axis=1)
    g = d.groupby(["group"] + keys)[comps + ["_total"]].agg("mean")
    g["_n"] = d.groupby(["group"] + keys).size()
    g = g[g["_n"] >= (1 if aggregate == "pooled" else min_pairs_per_unit)].reset_index()
    rng = np.random.default_rng(0)
    labels = []
    for gi, (a, l) in enumerate(SIX_GROUP_ORDER):
        key = f"{a}_{l}"
        sub = g[g["group"] == key]
        n = int(sub["_n"].sum()) if aggregate == "pooled" else len(sub)
        labels.append(f"{SIX_GROUP_LABELS[(a, l)]}\n(n={n})")
        if sub.empty:
            continue
        pos, neg = 0.0, 0.0
        for c in comps:
            m = float(sub[c].mean())
            base = pos if m >= 0 else neg
            ax.bar(gi, m, 0.7, bottom=base, color=colors[c], edgecolor="black", linewidth=0.3,
                   label=c if gi == 0 else None)
            if m >= 0:
                pos += m
            else:
                neg += m
        if aggregate != "pooled":
            dc = [mouse_colors.get(a_, "k") for a_ in sub["animal_id"]] if mouse_colors else "black"
            ax.scatter(gi + rng.uniform(-0.15, 0.15, len(sub)), sub["_total"], s=12, c=dc, edgecolor="black",
                       linewidth=0.3, zorder=3)
    ax.axhline(0, color="grey", linewidth=0.7)
    ax.set_xticks(range(len(SIX_GROUP_ORDER)))
    ax.set_xticklabels(labels, fontsize=7)
    if ylabel:
        ax.set_ylabel(ylabel)
