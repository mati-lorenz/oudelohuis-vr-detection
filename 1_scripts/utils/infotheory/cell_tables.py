# -*- coding: utf-8 -*-
"""
cell_tables.py
===============
Shared by 2c_information (mutual information) and 2d_linear_encod (linear fits): ONE definition of
"which trials and which values enter the (predictor, cell) comparison", so that MI and r^2 are computed from
IDENTICAL data and agree before any bias correction (project rule 4, see project_config.py).

Per (cell, predictor) pair, `pair_arrays` returns (x, y):
    1. trials where predictor and cell response are both finite
    2. discrete predictors (stimulus bins, choice, correct): trials in a bin holding fewer than
       MIN_TRIALS_PER_TARGET_BIN of the remaining trials are dropped
    3. at least MIN_TRIALS_ANALYSIS trials must remain, and the predictor must take >= 2 distinct values
else None (no value is reported for the pair). The cell response is the window mean of the activity (for
2c/2d); MI discretises BOTH x and y into project_config.RESPONSE_BINS quantile bins, the linear fit uses the
numbers as they are (for a discrete predictor, its bin index 0, 1, 2 ...).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import project_config as cfg
from .stim_binning import STIM_BINNING_STRATEGIES, CHOICE_TARGET, bin_stimulus

STIM_PREFIX = "stim__"
BEHAVIOR_COLUMNS = ["cur_runspeed", "cur_pupil_area", "cur_motionenergy", "cur_lick"]
OUTCOME_COLUMNS = [CHOICE_TARGET, "correct"]


def stim_predictor(strategy: str) -> str:
    return f"{STIM_PREFIX}{strategy}"


def predictor_list(strategies, behavior=BEHAVIOR_COLUMNS) -> list[str]:
    """behavior (same-window) predictors, then one stimulus predictor per binning strategy, then choice, correct."""
    return list(behavior) + [stim_predictor(s) for s in strategies] + list(OUTCOME_COLUMNS)


def is_discrete(predictor: str) -> bool:
    return predictor.startswith(STIM_PREFIX) or predictor in OUTCOME_COLUMNS


def add_stimulus_columns(table: pd.DataFrame, strategies, definitions: dict | None = None) -> pd.DataFrame:
    """Add one column `stim__<strategy>` (integer bin index, NaN where undefined) per strategy. Needs the
    table columns signal, stim_pct (and stim for quantile strategies; build_session_tables provides them)."""
    definitions = definitions or STIM_BINNING_STRATEGIES
    out = table.copy()
    for s in strategies:
        out[stim_predictor(s)] = bin_stimulus(out["signal"].to_numpy(), out["stim_pct"].to_numpy(),
                                              definitions[s], stim_z=out["stim"].to_numpy()
                                              if "stim" in out.columns else None)
    return out


def pair_arrays(table: pd.DataFrame, predictor: str, cell_id, min_trials: int = cfg.MIN_TRIALS_ANALYSIS,
                min_per_bin: int = cfg.MIN_TRIALS_PER_TARGET_BIN):
    """(x, y, trial_numbers) for one (cell, predictor) pair, or None. See module docstring."""
    if predictor not in table.columns or cell_id not in table.columns:
        return None
    x = table[predictor].to_numpy(float)
    y = table[cell_id].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y)
    if is_discrete(predictor) and ok.any():
        vals, counts = np.unique(x[ok], return_counts=True)
        small = vals[counts < min_per_bin]
        if len(small):
            ok &= ~np.isin(x, small)
    if ok.sum() < min_trials or len(np.unique(x[ok])) < 2:
        return None
    tn_ = table["trialNumber"].to_numpy()[ok] if "trialNumber" in table.columns else np.flatnonzero(ok)
    return x[ok], y[ok], tn_


def cell_columns(table: pd.DataFrame, cell_ids) -> list:
    """Columns of `table` that are cells (robust to extra bookkeeping columns)."""
    cells = set(cell_ids)
    return [c for c in table.columns if c in cells]


def attach_cell_meta(df: pd.DataFrame, celldata_combined: pd.DataFrame) -> pd.DataFrame:
    """Add roi_name, labeled, area, label, group, group_name and animal_id (from the session id) to a
    per-cell table that has session_id and cell_id."""
    from .groups import add_six_group_column
    meta = celldata_combined[["session_id", "cell_id", "roi_name", "labeled"]].drop_duplicates(
        ["session_id", "cell_id"])
    stale = [c for c in ("roi_name", "labeled", "area", "label", "group", "group_name") if c in df.columns]
    out = add_six_group_column(df.drop(columns=stale).merge(meta, on=["session_id", "cell_id"], how="left"))
    if "animal_id" not in out.columns:
        out["animal_id"] = out["session_id"].str.split("_").str[0]
    return out


def unit_group_summary(df: pd.DataFrame, by: list, aggregate: str, mean_cols: list, sig_col: str | None = None):
    """Per project group x `by` x analysis unit: n_cells, mean of each of `mean_cols`, and (if `sig_col`, a bool
    column) n_sig and frac_sig. pooled: no unit columns; session / mouse: `unit` and `animal_id` added."""
    keys = {"pooled": [], "session": ["animal_id", "session_id"], "mouse": ["animal_id"]}[aggregate]
    g = df.groupby(["group"] + list(by) + keys, observed=True)
    out = g[list(mean_cols)].mean()
    out["n_cells"] = g.size()
    if sig_col:
        out["n_sig"] = g[sig_col].sum()
        out["frac_sig"] = out["n_sig"] / out["n_cells"]
    out = out.reset_index()
    if aggregate == "session":
        out = out.rename(columns={"session_id": "unit"})
    elif aggregate == "mouse":
        out["unit"] = out["animal_id"]
    lead = ["group"] + (["unit", "animal_id"] if aggregate != "pooled" else []) + list(by)
    rest = ["n_cells"] + list(mean_cols) + (["n_sig", "frac_sig"] if sig_col else [])
    return out[lead + rest]
