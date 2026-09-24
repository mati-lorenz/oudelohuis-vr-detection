# -*- coding: utf-8 -*-
"""
population.py
==============
Shared infrastructure for the section-4 population-level scripts
(4a_population_pca.py, 4b_population_fa.py, 4c_population_svm.py):
building a standardized population response matrix, running an
analysis across the conditioned-shuffle framework (infotheory.shuffling)
in a fully modular way, and the plots common to all three techniques.

MODULARITY (per the person's explicit request): which conditions to
shuffle within is a plain dict of {name: array}, built by
`build_standard_condition_specs` for now (stim, choice, stim+choice)
but extendable to ANY set of named conditions -- e.g. a future
behavioral variable -- by adding one more named entry, with zero
changes needed to `run_across_shuffle_conditions`, the plotting
functions, or any of the three technique scripts. See
infotheory.shuffling.make_shuffle_groups_from_conditions, which this
builds on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .shuffling import make_shuffle_groups_from_conditions, shuffle_population_within_groups


def zscore_population(matrix: np.ndarray) -> np.ndarray:
    """Z-score each column (cell) of a (trials, cells) response
    matrix -- necessary before PCA/FA/SVM, which would otherwise be
    dominated by whichever cells happen to have the largest raw
    activity scale, not the most informative trial-to-trial structure.
    Columns with zero variance are left as all-zero (rather than
    dividing by zero) after centering."""
    matrix = np.asarray(matrix, dtype=float)
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std_safe = np.where(std > 0, std, 1.0)
    z = (matrix - mean) / std_safe
    z[:, (std == 0).ravel()] = 0.0
    return z


def build_standard_condition_specs(stim_bin: np.ndarray, choice: np.ndarray) -> dict:
    """The four condition specs 3a/3b/3c already use (global, stim,
    choice, stim_choice), packaged as {name: conditions_dict} for
    `run_across_shuffle_conditions` -- i.e. one level further wrapped
    than infotheory.shuffling.make_shuffle_groups_from_conditions
    itself expects, so callers can iterate a single dict of named
    scenarios. To test a behavioral variable later (e.g. running
    speed), build an equivalent dict yourself:
        {"real": None, "global": {}, "runspeed": {"runspeed": runspeed_bin}}
    and pass it directly to run_across_shuffle_conditions instead of
    this function's output -- nothing else needs to change.
    """
    return {
        "real": None,       # not shuffled at all -- the actual data
        "global": {},        # fully destroy everything (make_shuffle_groups_from_conditions({}))
        "stim": {"stim": stim_bin},
        "choice": {"choice": choice},
        "stim_choice": {"stim": stim_bin, "choice": choice},
    }


def run_across_shuffle_conditions(matrix: np.ndarray, condition_specs: dict,
                                   analysis_fn, n_shuffles: int = 20,
                                   rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Run `analysis_fn(matrix) -> dict of scalar results` on the REAL
    `matrix`, and on `n_shuffles` independent realizations of each
    OTHER named condition in `condition_specs` (see
    build_standard_condition_specs for the dict shape -- "real" itself
    is never shuffled, just analyzed once).

    Returns a tidy DataFrame: one row per (condition, rep) with
    analysis_fn's result dict as columns, plus 'condition' and 'rep'
    ('rep' is always 0 for the unshuffled "real" condition).
    """
    rng = rng or np.random.default_rng()
    n = matrix.shape[0]
    rows = []
    for name, conditions in condition_specs.items():
        if name == "real":
            result = analysis_fn(matrix)
            rows.append({"condition": name, "rep": 0, **result})
            continue
        groups = make_shuffle_groups_from_conditions(conditions, n=n)
        for rep in range(n_shuffles):
            shuffled = shuffle_population_within_groups(matrix, groups, rng=rng)
            result = analysis_fn(shuffled)
            rows.append({"condition": name, "rep": rep, **result})
    return pd.DataFrame(rows)


def plot_shuffle_condition_comparison(ax, results_df: pd.DataFrame, metric_col: str, ylabel: str,
                                       condition_order=("real", "global", "stim", "choice", "stim_choice"),
                                       colors=None):
    """THE plot common to all of 4a/4b/4c: the technique's headline
    metric, one box per shuffle condition (real data plus each
    conditioned-shuffle null), in the same fixed order/styling every
    time so the three scripts' figures are directly comparable side by
    side. 'real' is drawn as a single reference line, since it isn't a
    distribution (only one realization exists); every shuffle condition
    is drawn as a box over its `n_shuffles` realizations.
    """
    order = [c for c in condition_order if c in results_df["condition"].unique()]
    real_val = results_df.loc[results_df["condition"] == "real", metric_col]
    box_data, box_labels = [], []
    for cond in order:
        if cond == "real":
            continue
        box_data.append(results_df.loc[results_df["condition"] == cond, metric_col].dropna())
        box_labels.append(cond)
    if box_data:
        ax.boxplot(box_data, tick_labels=box_labels, showfliers=False)
    if len(real_val):
        ax.axhline(float(real_val.iloc[0]), color="firebrick", linestyle="--", linewidth=1.5,
                   label="real (unshuffled)")
        ax.legend(fontsize=8, frameon=False)
    ax.set_ylabel(ylabel)
    return ax


def plot_metric_vs_population_size(ax, per_group_df: pd.DataFrame, size_col: str, metric_col: str,
                                    area_col: str, ylabel: str, area_colors: dict):
    """The other plot common to all of 4a/4b/4c: does the technique's
    headline metric (on the REAL, unshuffled data) scale with how many
    cells went into it, and does that relationship differ by area."""
    for area in per_group_df[area_col].unique():
        sub = per_group_df[per_group_df[area_col] == area]
        ax.scatter(sub[size_col], sub[metric_col], s=15, alpha=0.6,
                   color=area_colors.get(area, "gray"), label=area)
    ax.set_xlabel("population size (cells)")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, frameon=False)
    return ax


def build_population_group(ses, mu_sigma_lookup: dict, windows: dict, included_cell_ids: set,
                            area: str, label: str, area_label_lookup: dict):
    """Wraps infotheory.neural_encoding.build_session_tables into the
    exact shape 4a/4b/4c_population_*.py all need: per window, one
    (matrix, stim_bin, choice, cell_ids) tuple for the cells in a
    SINGLE (area, label) group (population techniques are run within
    one area/label group at a time, same as the pairwise 3a/3b/3c
    scripts, never mixing areas together into one population).

    Returns {window_name: (matrix, stim_bin, choice, cell_ids)}, matrix
    shape (n_trials, n_cells) IN RAW UNITS (not yet z-scored -- each
    technique script decides whether/when to call zscore_population).
    """
    from .neural_encoding import build_session_tables
    group_cell_ids = [c for c in included_cell_ids if area_label_lookup.get(c) == (area, label)]
    if len(group_cell_ids) < 2:
        return {}
    tables = build_session_tables(ses, mu_sigma_lookup, windows, included_cell_ids=set(group_cell_ids))
    out = {}
    for window_name, table in tables.items():
        if table.empty or "stim_bin" not in table.columns:
            continue
        cell_cols = [c for c in group_cell_ids if c in table.columns]
        if len(cell_cols) < 2:
            continue
        sub = table[cell_cols + ["stim_bin", "choice"]].dropna()
        if len(sub) < 20:
            continue
        out[window_name] = (sub[cell_cols].to_numpy(), sub["stim_bin"].to_numpy(),
                             sub["choice"].to_numpy(), cell_cols)
    return out


def iterate_population_groups(sessions, mu_sigma_lookup: dict, windows: dict,
                               included_cell_ids_by_session: dict, area_label_by_session: dict,
                               min_cells: int = 10):
    """Shared session/area/label/window loop for 4a/4b/4c_population_*.py:
    yields (session_id, window_name, area, label, matrix, stim_bin,
    choice, cell_ids) for every combination with at least `min_cells`
    cells -- so each script's own body is just "what do I do with one
    population matrix", not session/group bookkeeping."""
    for ses in sessions:
        included_cells = included_cell_ids_by_session.get(ses.session_id)
        if not included_cells:
            continue
        area_label_lookup = area_label_by_session.get(ses.session_id, {})
        areas_labels = sorted({area_label_lookup[c] for c in included_cells if c in area_label_lookup})
        for area, label in areas_labels:
            groups = build_population_group(ses, mu_sigma_lookup, windows, included_cells,
                                             area, label, area_label_lookup)
            for window_name, (matrix, stim_bin, choice, cell_ids) in groups.items():
                if len(cell_ids) < min_cells:
                    continue
                yield ses.session_id, window_name, area, label, matrix, stim_bin, choice, cell_ids
