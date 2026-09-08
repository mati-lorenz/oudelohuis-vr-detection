# -*- coding: utf-8 -*-
"""
neural_encoding.py
=====================
Shared dataset-assembly logic for 2c_information.py / 2d_linear_encod.py
/ 2e_nonlinear_encod.py: applying the session/trial/cell filters those
three scripts all need identically (this lives in the library, not in
any one script, specifically so it's never duplicated three times --
Rule #1 is about not reaching into another SCRIPT's store/, not about
avoiding shared LIBRARY code).

Filters applied (all from EARLIER pipeline steps' own out/ -- nothing
here recomputes an inclusion decision another step already made):
    sessions : 1b_psychometric's included_ids
    trials   : engaged == 1 (same convention as every other step)
    cells    : 2a_cell_distribution's passes_proximity_filter AND
               2b_activity_statistics's qc_pass (logical AND -- a cell
               must pass BOTH to be included)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .continuous import (
    merge_behavior_video, build_calcium_continuous, restrict_to_engaged, remove_pupil_outliers,
    compute_trial_position_window_means,
)
from .psth import derive_onset_time

BEHAVIOR_PREDICTORS = ["runspeed", "pupil_area", "motionenergy", "lick"]
RATE_PREDICTORS = ["lick"]
N_STIM_BINS = 4


def load_cell_inclusion(cell_dist_out, activity_stats_out) -> pd.DataFrame:
    """Combine 2a's proximity filter and 2b's QC pass into one final
    per-cell inclusion table.

    Parameters
    ----------
    cell_dist_out, activity_stats_out : Path -- the out/ directories of
        2a_cell_distribution and 2b_activity_statistics respectively
        (e.g. `paths.out_from("2a_cell_distribution")`)

    Returns
    -------
    DataFrame with session_id, cell_id, include (bool: passes BOTH
    2a's proximity filter and 2b's QC)
    """
    celldata = pd.read_csv(cell_dist_out / "celldata_combined.csv")[
        ["session_id", "cell_id", "passes_proximity_filter"]]
    qc = pd.read_csv(activity_stats_out / "qc_per_cell.csv")[["session_id", "cell_id", "qc_pass"]]
    merged = celldata.merge(qc, on=["session_id", "cell_id"], how="inner")
    merged["include"] = merged["passes_proximity_filter"] & merged["qc_pass"]
    return merged[["session_id", "cell_id", "include"]]


def build_session_tables(ses, mu_sigma_lookup: dict, windows: dict,
                          included_cell_ids: set | None = None,
                          behavior_predictors=BEHAVIOR_PREDICTORS,
                          rate_predictors=RATE_PREDICTORS,
                          n_stim_bins: int = N_STIM_BINS) -> dict:
    """Build one session's per-window (predictor + per-cell neural
    response) table. `ses` must already be loaded with
    load_behaviordata=True, load_videodata=True, load_celldata=True,
    load_calciumdata=True.

    Returns
    -------
    dict: window_name -> DataFrame (one row per engaged trial), columns:
        trialNumber, cur_<behavior_predictor> (one per window, same
        position-window convention as 1c_behavior.py's MI_WINDOWS),
        stim_bin, choice, correct (trial-level, same across windows),
        plus one column per included cell (cell_id as the column name,
        that cell's mean activity in this window/trial)
    Returns {} if this session has no usable celldata/calciumdata/
    behaviordata, or no cells survive `included_cell_ids`.
    """
    if (ses.trialdata is None or ses.behaviordata is None or ses.celldata is None
            or ses.calciumdata is None or ses.ts_F is None):
        return {}

    celldata = ses.celldata.reset_index(drop=True)
    if included_cell_ids is not None:
        keep_mask = celldata["cell_id"].isin(included_cell_ids).to_numpy()
    else:
        keep_mask = np.ones(len(celldata), dtype=bool)
    if not keep_mask.any():
        return {}
    cell_ids = celldata.loc[keep_mask, "cell_id"].to_numpy()
    calciumdata = ses.calciumdata.loc[:, cell_ids]

    merged_full = merge_behavior_video(ses.behaviordata, ses.videodata)
    merged_full = remove_pupil_outliers(merged_full)
    if merged_full.empty:
        return {}
    merged = restrict_to_engaged(merged_full, ses.trialdata)
    if merged.empty:
        return {}
    calcium_continuous = build_calcium_continuous(calciumdata, ses.ts_F, ses.behaviordata)

    available_behavior = [v for v in behavior_predictors if v in merged.columns]
    available_rate = [v for v in rate_predictors if v in available_behavior]

    engaged_trials = (ses.trialdata[ses.trialdata["engaged"] == 1]
                       if "engaged" in ses.trialdata.columns else ses.trialdata.copy())
    engaged_trials = derive_onset_time(engaged_trials, merged_full)

    # Trial-level (window-independent) predictors: stim strength
    # (z-scored via this session's own psychometric fit, matching
    # 1c_behavior.py/1d_performance_predictors.py's convention),
    # choice, and generalized correctness (see 1d_performance_
    # predictors.py's module docstring for the "any signal present ->
    # should lick" rule this mirrors).
    mu_sigma = mu_sigma_lookup.get(ses.session_id, {})
    if mu_sigma and np.isfinite(mu_sigma.get("mu", np.nan)) and np.isfinite(mu_sigma.get("sigma", np.nan)):
        stim = (engaged_trials["signal"] - mu_sigma["mu"]) / mu_sigma["sigma"]
    else:
        stim = engaged_trials["signal"].astype(float)
    engaged_trials = engaged_trials.assign(
        stim=stim, choice=engaged_trials["lickResponse"].astype(int),
        correct=(((engaged_trials["signal"] > 0) & (engaged_trials["lickResponse"] == 1))
                 | ((engaged_trials["signal"] == 0) & (engaged_trials["lickResponse"] == 0))).astype(int))

    from .info_theory import robust_qcut
    try:
        stim_bin = robust_qcut(engaged_trials["stim"], n_stim_bins)
        engaged_trials = engaged_trials.assign(
            stim_bin=stim_bin.apply(lambda iv: round(iv.mid, 6) if pd.notna(iv) else np.nan))
    except (ValueError, TypeError):
        engaged_trials = engaged_trials.assign(stim_bin=np.nan)

    trial_meta = engaged_trials[["trialNumber", "stim_bin", "choice", "correct"]]

    tables = {}
    for window_name, (s_start, s_end) in windows.items():
        behav_win = compute_trial_position_window_means(
            engaged_trials, merged, columns=available_behavior,
            s_start=s_start, s_end=s_end, rate_columns=available_rate)
        behav_win = behav_win.rename(columns={v: f"cur_{v}" for v in available_behavior})

        neural_win = compute_trial_position_window_means(
            engaged_trials, calcium_continuous, columns=list(cell_ids),
            s_start=s_start, s_end=s_end)

        table = behav_win.merge(neural_win, on="trialNumber", how="inner").merge(
            trial_meta, on="trialNumber", how="left")
        tables[window_name] = table

    return tables
