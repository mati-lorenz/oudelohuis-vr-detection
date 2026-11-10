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

from . import project_config as cfg
from .continuous import (
    merge_behavior_video, build_calcium_continuous, restrict_to_engaged, remove_pupil_outliers,
    compute_trial_position_window_means, restrict_to_trials, zscore_per_session,
)
from .psth import derive_onset_time

BEHAVIOR_PREDICTORS = ["runspeed", "pupil_area", "motionenergy", "lick"]
RATE_PREDICTORS = ["lick"]
N_STIM_BINS = 4


def load_cell_inclusion(cell_dist_out, activity_stats_out) -> pd.DataFrame:
    """Combine 2a's proximity filter, 2a's depth rule and 2b's QC pass into one final per-cell inclusion table.

    Parameters
    ----------
    cell_dist_out, activity_stats_out : Path -- the out/ directories of
        2a_cell_distribution and 2b_activity_statistics respectively
        (e.g. `paths.out_from("2a_cell_distribution")`)

    Returns
    -------
    DataFrame with session_id, cell_id, group (the project group key, e.g. 'V1_lab', 'PMdeep_unl'; used by the per-group cell caps
    of session_jobs.run_sessions_units), include (bool: passes 2a's proximity filter AND 2a's depth rule AND 2b's QC).
    `passes_depth_filter` (2a) is True for cells of the analysed depth classes: shallower than project_config.MAX_DEPTH_UM
    (all areas) or deep PM (project_config.DEEP_PM_DEPTH_UM); an older celldata_combined.csv without that column is used
    as it is (rerun 2a)."""
    celldata = pd.read_csv(cell_dist_out / "celldata_combined.csv", low_memory=False)
    dcol = ["passes_depth_filter"] if "passes_depth_filter" in celldata.columns else []
    from .groups import add_six_group_column, group_meta_columns          # local import (groups imports plotting)
    gtab = add_six_group_column(celldata[["session_id", "cell_id"] + group_meta_columns(celldata)])[["session_id", "cell_id", "group"]]
    celldata = celldata[["session_id", "cell_id", "passes_proximity_filter"] + dcol].merge(gtab, on=["session_id", "cell_id"])
    qc = pd.read_csv(activity_stats_out / "qc_per_cell.csv")[["session_id", "cell_id", "qc_pass"]]
    merged = celldata.merge(qc, on=["session_id", "cell_id"], how="inner")
    merged["include"] = merged["passes_proximity_filter"] & merged["qc_pass"]
    if dcol:
        merged["include"] &= merged["passes_depth_filter"].astype(bool)
    return merged[["session_id", "cell_id", "include", "group"]]


def prepare_behavior_trace(ses) -> pd.DataFrame:
    """The session's behavior + video trace as the analyses use it: pupil blinks removed, then (project_config.
    NORMALIZE_FACE_PER_SESSION) the face quantities -- pupil area / x / y, motion energy, video PCs -- z-scored over the
    WHOLE session. Running speed and licks stay in physical units. Used by build_session_tables and build_window_frames so the
    per-trial and the pooled-frame analyses see identical values."""
    merged_full = remove_pupil_outliers(merge_behavior_video(ses.behaviordata, ses.videodata))
    if cfg.NORMALIZE_FACE_PER_SESSION and not merged_full.empty:
        merged_full = zscore_per_session(merged_full)
    return merged_full


def build_session_tables(ses, mu_sigma_lookup: dict, windows: dict,
                          included_cell_ids: set | None = None,
                          behavior_predictors=BEHAVIOR_PREDICTORS,
                          rate_predictors=RATE_PREDICTORS,
                          n_stim_bins: int = N_STIM_BINS, trial_numbers=None) -> dict:
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
    trial_numbers : None (default) -> the ENGAGED trials. Otherwise an iterable of trialNumbers: exactly
        those trials are used whatever their engaged flag (end-of-session disengaged-trial tests, see
        engagement.py); the table then also has an `engaged` column.
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

    merged_full = prepare_behavior_trace(ses)
    if merged_full.empty:
        return {}
    if trial_numbers is None:
        merged = restrict_to_engaged(merged_full, ses.trialdata)
    else:
        merged = restrict_to_trials(merged_full, ses.trialdata, trial_numbers)
    if merged.empty:
        return {}
    calcium_continuous = build_calcium_continuous(calciumdata, ses.ts_F, ses.behaviordata)

    available_behavior = [v for v in behavior_predictors if v in merged.columns]
    available_rate = [v for v in rate_predictors if v in available_behavior]

    if trial_numbers is None:
        engaged_trials = (ses.trialdata[ses.trialdata["engaged"] == 1]
                           if "engaged" in ses.trialdata.columns else ses.trialdata.copy())
    else:
        engaged_trials = ses.trialdata[ses.trialdata["trialNumber"].isin(list(trial_numbers))].copy()
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

    # Raw contrast + position on the fitted psychometric curve (0-1), so callers can apply ANY
    # binning from stim_binning.STIM_BINNING_STRATEGIES; `stim_bin` above is the legacy quantile bin.
    from .stim_binning import psychometric_pct
    engaged_trials = engaged_trials.assign(
        stim_pct=psychometric_pct(engaged_trials["signal"], mu_sigma.get("mu"), mu_sigma.get("sigma")))

    meta_cols = ["trialNumber", "stim_bin", "choice", "correct", "signal", "stim", "stim_pct"]
    if trial_numbers is not None and "engaged" in engaged_trials.columns:
        meta_cols.append("engaged")
    trial_meta = engaged_trials[meta_cols]

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


def build_window_frames(ses, windows: dict, included_cell_ids: set | None = None,
                        behavior_predictors=BEHAVIOR_PREDICTORS, rate_predictors=RATE_PREDICTORS,
                        trial_numbers=None) -> dict:
    """Pooled-frames counterpart of build_session_tables: for every window, one row per IMAGING FRAME that lies inside the
    position window ([stimStart + start, stimStart + end) cm) of one of the trials (engaged trials, or `trial_numbers`).

    Columns: trialNumber, cur_<behavior> (the behavior on the imaging clock: mean of the samples in the frame; lick = events/s;
    same face normalization as the per-trial tables) and one column per included cell (its activity in that frame).
    Returns {} when the session lacks calcium / behavior data or has no frames in any window. Frames of different trials are
    stacked in trial order, so a circular shift of a column moves whole stretches of trials against each other."""
    from . import tensors as tn
    td = ses.trialdata
    if (td is None or ses.behaviordata is None or ses.celldata is None or ses.calciumdata is None or ses.ts_F is None
            or "stimStart" not in td.columns):
        return {}
    celldata = ses.celldata.reset_index(drop=True)
    keep = (celldata["cell_id"].isin(included_cell_ids).to_numpy() if included_cell_ids is not None
            else np.ones(len(celldata), bool))
    if not keep.any():
        return {}
    cell_ids = celldata.loc[keep, "cell_id"].to_numpy()
    merged = prepare_behavior_trace(ses)
    if merged.empty:
        return {}
    tr = (td[td["engaged"] == 1] if (trial_numbers is None and "engaged" in td.columns)
          else td[td["trialNumber"].isin(list(trial_numbers))] if trial_numbers is not None else td)
    tr = tr.sort_values("stimStart")
    if tr.empty:
        return {}
    ts_F = np.asarray(ses.ts_F, float)
    zpos_F = tn.interp_to_imaging_frames(ses.behaviordata["ts"], ses.behaviordata["zpos"], ts_F)
    lo = min(w[0] for w in windows.values())
    hi = max(w[1] for w in windows.values())
    k, rel = tn.position_relative_to_stim(zpos_F, tr["stimStart"].to_numpy(float), max(0.0, -lo), max(0.0, hi))
    available = [v for v in behavior_predictors if v in merged.columns]
    beh = {f"cur_{v}": tn.behavior_to_imaging_frames(merged["ts"], merged[v], ts_F,
                                                     mode="rate" if v in rate_predictors else "mean")
           for v in available}
    A = ses.calciumdata.loc[:, cell_ids].to_numpy(dtype=float)
    tnum = tr["trialNumber"].to_numpy()
    out = {}
    for name, (s0, s1) in windows.items():
        m = (k >= 0) & np.isfinite(rel) & (rel >= s0) & (rel < s1)
        if not m.any():
            continue
        df = pd.DataFrame({"trialNumber": tnum[k[m]]})
        for c, x in beh.items():
            df[c] = x[m]
        cells_df = pd.DataFrame(A[m], columns=list(cell_ids))
        out[name] = pd.concat([df, cells_df], axis=1)
    return out

