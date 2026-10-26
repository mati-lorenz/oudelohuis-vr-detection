# -*- coding: utf-8 -*-
"""
unit_stats.py
==============
Aggregate session-level behavior summaries to the chosen UNIT OF ANALYSIS (project_config.AGGREGATE_BY):
    pooled  -> one row per condition (e.g. protocol), counts summed over everything
    session -> one row per session
    mouse   -> one row per mouse (its sessions pooled; separate rows per protocol)
Counts are SUMMED within a unit and rates / d' / criterion are RE-COMPUTED from the summed counts, so a
pooled or per-mouse d' is the d' of the pooled trials, not an average of session d' values.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .scope import unit_columns

OUTCOMES = ("HIT", "MISS", "FA", "CR")


def dprime_from_counts(hit, miss, fa, cr) -> tuple[float, float]:
    """Same d' / criterion and the same 1/(2N) clipping as behavior.compute_dprime, from outcome counts."""
    n_present, n_absent = hit + miss, fa + cr
    if n_present == 0 or n_absent == 0:
        return np.nan, np.nan
    h = np.clip(hit / n_present, 0.5 / n_present, 1 - 0.5 / n_present)
    f = np.clip(fa / n_absent, 0.5 / n_absent, 1 - 0.5 / n_absent)
    zh, zf = stats.norm.ppf(h), stats.norm.ppf(f)
    return float(zh - zf), float(-0.5 * (zh + zf))


def unit_summary(summary: pd.DataFrame, aggregate: str, by=("protocol",)) -> pd.DataFrame:
    """One row per analysis unit (see module docstring).

    Output columns: `by`, [animal_id], [session_id], n_sessions, n_trials, duration_min, n_engaged,
    frac_engaged, n_HIT/MISS/FA/CR_(all|engaged), dprime_(all|engaged), criterion_(all|engaged),
    first_disengaged_frac (mean over the unit's sessions; NaN if never disengaged)."""
    by = list(by)
    keys = by + unit_columns(aggregate)
    if aggregate == "session":
        keys = by + ["animal_id", "session_id"]
    df = summary.copy()
    df["n_engaged_"] = df["eng_n_engaged"] if "eng_n_engaged" in df else np.nan
    df["first_dis_frac"] = df["eng_first_disengaged_trial"] / df["n_trials"] if "eng_first_disengaged_trial" in df else np.nan
    df["duration_min"] = df["duration_s"] / 60 if "duration_s" in df else np.nan
    count_cols = ["n_trials", "duration_min", "n_engaged_"] + [c for c in df.columns if c.startswith("n_") and
                 any(c.startswith(f"n_{o}_") for o in OUTCOMES)]
    count_cols = [c for c in dict.fromkeys(count_cols) if c in df.columns]
    g = df.groupby(keys, dropna=False) if keys else None
    if g is None:
        agg = df[count_cols].sum(min_count=1).to_frame().T
        agg["n_sessions"] = len(df)
        agg["first_dis_frac"] = df["first_dis_frac"].mean()
    else:
        agg = g[count_cols].sum(min_count=1)
        agg["n_sessions"] = g["session_id"].nunique()
        agg["first_dis_frac"] = g["first_dis_frac"].mean()
        agg = agg.reset_index()
    if aggregate == "mouse":
        pass
    agg = agg.rename(columns={"n_engaged_": "n_engaged", "first_dis_frac": "first_disengaged_frac"})
    agg["frac_engaged"] = agg["n_engaged"] / agg["n_trials"]
    for label in ("all", "engaged"):
        cols = [f"n_{o}_{label}" for o in OUTCOMES]
        if all(c in agg for c in cols):
            res = [dprime_from_counts(*r) for r in agg[cols].fillna(0).to_numpy()]
            agg[f"dprime_{label}"] = [r[0] for r in res]
            agg[f"criterion_{label}"] = [r[1] for r in res]
    return agg


def unit_curve(df: pd.DataFrame, x: str, y: str, aggregate: str):
    """Mean and error of `y` at each value of `x`, at the chosen unit of analysis.
        session / mouse : mean within each unit first, then mean and SEM ACROSS units
        pooled          : mean over all rows, no error (pooled = no error bars, see project_config)
    Returns (mean Series indexed by x, sem Series (zeros when pooled))."""
    keys = unit_columns(aggregate)
    if keys:
        per_unit = df.groupby(keys + [x])[y].mean().reset_index()
        g = per_unit.groupby(x)[y]
        return g.mean(), g.sem().fillna(0.0)
    m = df.groupby(x)[y].mean()
    return m, m * 0.0


def stable_seed(*parts, base: int = 0) -> int:
    """Deterministic seed from a descriptive key (independent of list order / selection)."""
    import zlib
    return base + zlib.crc32("|".join(map(str, parts)).encode()) % 1_000_000


def group_unit_table(cells: pd.DataFrame, values: list, aggregate: str) -> pd.DataFrame:
    """Reporting table for per-cell variables: mean of each of `values` and the number of cells, per project
    group (column `group`) and analysis unit. pooled: one row per group (original-style, no unit columns);
    session / mouse: adds `unit` and `animal_id`."""
    keys = {"pooled": [], "session": ["animal_id", "session_id"], "mouse": ["animal_id"]}[aggregate]
    g = cells.groupby(["group"] + keys)
    out = g[list(values)].mean()
    out["n_cells"] = g.size()
    out = out.reset_index()
    if aggregate == "session":
        out = out.rename(columns={"session_id": "unit"})
    elif aggregate == "mouse":
        out["unit"] = out["animal_id"]
    cols = ["group"] + (["unit", "animal_id"] if aggregate != "pooled" else []) + ["n_cells"] + list(values)
    return out[cols]
