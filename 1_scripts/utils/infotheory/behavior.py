# -*- coding: utf-8 -*-
"""
Signal-detection statistics, engagement summaries, and session-level
performance tables. Kept free of matplotlib so these are reusable from
non-plotting contexts (e.g. later steps that regress performance against
neural activity). Plotting helpers live in `infotheory.plotting`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def compute_dprime(signal: pd.Series, response: pd.Series,
                    signal_present_value=100, signal_absent_value=0) -> tuple[float, float]:
    """Signal-detection d' and criterion from hit rate and false-alarm
    rate on the 0%/100% ("catch") trials.

    Hit/FA rates of exactly 0 or 1 are clipped away from the extremes
    (scaled by trial count, following Macmillan & Creelman's 1/(2N)
    correction) so `norm.ppf` doesn't blow up to +/-inf.
    """
    signal = np.asarray(signal)
    response = np.asarray(response)

    n_present = np.sum(signal == signal_present_value)
    n_absent = np.sum(signal == signal_absent_value)
    if n_present == 0 or n_absent == 0:
        return np.nan, np.nan

    hit_rate = np.sum((signal == signal_present_value) & (response == 1)) / n_present
    fa_rate = np.sum((signal == signal_absent_value) & (response == 1)) / n_absent

    hit_rate = np.clip(hit_rate, 0.5 / n_present, 1 - 0.5 / n_present)
    fa_rate = np.clip(fa_rate, 0.5 / n_absent, 1 - 0.5 / n_absent)

    dprime = stats.norm.ppf(hit_rate) - stats.norm.ppf(fa_rate)
    criterion = -0.5 * (stats.norm.ppf(hit_rate) + stats.norm.ppf(fa_rate))
    return float(dprime), float(criterion)


def add_trial_outcome(trialdata: pd.DataFrame, signal_present_value=100,
                       signal_absent_value=0) -> pd.DataFrame:
    """Add a `trialOutcome` column (HIT/MISS/FA/CR) if not already
    present. Trials at intermediate signal strength get no SDT label
    (those categories only apply to 0%/100% catch trials); use
    `signal_psy`/psychometric fits for the intermediate trials instead."""
    trialdata = trialdata.copy()
    if "trialOutcome" in trialdata.columns:
        return trialdata

    outcome = pd.Series(np.nan, index=trialdata.index, dtype=object)
    is_signal = trialdata["signal"] == signal_present_value
    is_nosignal = trialdata["signal"] == signal_absent_value
    responded = trialdata["lickResponse"] == 1

    outcome[is_signal & responded] = "HIT"
    outcome[is_signal & ~responded] = "MISS"
    outcome[is_nosignal & responded] = "FA"
    outcome[is_nosignal & ~responded] = "CR"
    trialdata["trialOutcome"] = outcome
    return trialdata


def engagement_summary(trialdata: pd.DataFrame) -> dict:
    """How much of the session was engaged, and how that disengagement is
    distributed (one big block at the end vs. scattered lapses)."""
    if "engaged" not in trialdata.columns or len(trialdata) == 0:
        return {"frac_engaged": np.nan, "n_engaged": np.nan, "n_disengaged": np.nan,
                "longest_disengaged_run": np.nan, "first_disengaged_trial": np.nan}

    engaged = trialdata["engaged"].astype(bool).to_numpy()
    n = len(engaged)

    longest_run, current_run = 0, 0
    for e in engaged:
        current_run = 0 if e else current_run + 1
        longest_run = max(longest_run, current_run)

    disengaged_idx = np.where(~engaged)[0]
    first_disengaged = int(disengaged_idx[0]) if disengaged_idx.size else np.nan

    return {
        "frac_engaged": float(engaged.mean()),
        "n_engaged": int(engaged.sum()),
        "n_disengaged": int(n - engaged.sum()),
        "longest_disengaged_run": longest_run,
        "first_disengaged_trial": first_disengaged,
    }


def rolling_performance(trialdata: pd.DataFrame, window: int = 25,
                         signal_present_value=100, signal_absent_value=0) -> pd.DataFrame:
    """Trial-by-trial rolling hit rate, false-alarm rate and d'/criterion
    (centered window), for plotting the within-session performance
    trajectory (e.g. to sanity-check the `engaged` labeling). Rows for
    signal strengths other than 0%/100% get NaN hit/FA rate but keep
    their trial number so the x-axis still lines up."""
    trialdata = trialdata.reset_index(drop=True)
    n = len(trialdata)

    is_hit_trial = (trialdata["signal"] == signal_present_value)
    is_ca_trial = (trialdata["signal"] == signal_absent_value)

    hit_series = pd.Series(np.where(is_hit_trial, trialdata["lickResponse"], np.nan))
    fa_series = pd.Series(np.where(is_ca_trial, trialdata["lickResponse"], np.nan))

    roll_hit = hit_series.rolling(window, min_periods=max(3, window // 5), center=True).mean()
    roll_fa = fa_series.rolling(window, min_periods=max(3, window // 5), center=True).mean()

    eps = 1e-3
    roll_hit_c = roll_hit.clip(eps, 1 - eps)
    roll_fa_c = roll_fa.clip(eps, 1 - eps)
    roll_dprime = stats.norm.ppf(roll_hit_c) - stats.norm.ppf(roll_fa_c)
    roll_criterion = -0.5 * (stats.norm.ppf(roll_hit_c) + stats.norm.ppf(roll_fa_c))

    out = pd.DataFrame({
        "trialNumber": trialdata["trialNumber"] if "trialNumber" in trialdata.columns else np.arange(1, n + 1),
        "rolling_hitrate": roll_hit.to_numpy(),
        "rolling_farate": roll_fa.to_numpy(),
        "rolling_dprime": roll_dprime,
        "rolling_criterion": roll_criterion,
    })
    if "engaged" in trialdata.columns:
        out["engaged"] = trialdata["engaged"].to_numpy()
    return out


def _session_duration_seconds(sessiondata: pd.DataFrame, trialdata: pd.DataFrame | None) -> float:
    """Best-effort session duration: explicit session-level start/end if
    present, else span of the trial timestamps."""
    for start_col, end_col in (("t_start", "t_stop"), ("tStart", "tEnd")):
        if start_col in sessiondata.columns and end_col in sessiondata.columns:
            return float(sessiondata[end_col].iloc[0] - sessiondata[start_col].iloc[0])
    if trialdata is not None and {"tStart", "tEnd"}.issubset(trialdata.columns) and len(trialdata):
        return float(trialdata["tEnd"].iloc[-1] - trialdata["tStart"].iloc[0])
    return np.nan


def summarize_session(ses, signal_present_value=100, signal_absent_value=0) -> dict:
    """One row of the session_summary table: everything 1a_performance
    reports per session -- counts, duration, engagement, and SDT metrics
    computed both on all trials and on engaged-only trials."""
    row = {
        "session_id": ses.session_id,
        "protocol": ses.protocol,
        "animal_id": ses.animal_id,
        "sessiondate": ses.sessiondate,
        "n_trials": len(ses.trialdata) if ses.trialdata is not None else 0,
        "has_behaviordata": ses.behaviordata is not None,
        "has_videodata": ses.videodata is not None,
    }

    trialdata = ses.trialdata
    if trialdata is None or len(trialdata) == 0:
        return row

    trialdata = add_trial_outcome(trialdata, signal_present_value, signal_absent_value)
    row["duration_s"] = _session_duration_seconds(ses.sessiondata, trialdata)
    row["trial_rate_per_min"] = (60 * row["n_trials"] / row["duration_s"]
                                  if row["duration_s"] not in (0, np.nan) and not np.isnan(row["duration_s"])
                                  else np.nan)
    row["n_signal_levels"] = int(trialdata["signal"].nunique())

    for key, value in engagement_summary(trialdata).items():
        row[f"eng_{key}"] = value

    if "engaged" in trialdata.columns:
        subsets = {"all": trialdata, "engaged": trialdata[trialdata["engaged"].astype(bool)]}
    else:
        subsets = {"all": trialdata}

    for label, subset in subsets.items():
        d, c = compute_dprime(subset["signal"], subset["lickResponse"],
                               signal_present_value, signal_absent_value)
        row[f"dprime_{label}"] = d
        row[f"criterion_{label}"] = c
        for outcome in ("HIT", "MISS", "FA", "CR"):
            row[f"n_{outcome}_{label}"] = int((subset["trialOutcome"] == outcome).sum())

    if ses.behaviordata is not None and "runspeed" in ses.behaviordata.columns:
        row["mean_runspeed"] = float(ses.behaviordata["runspeed"].mean())

    if ses.videodata is not None and "pupil_area" in ses.videodata.columns:
        row["mean_pupil_area"] = float(ses.videodata["pupil_area"].mean())

    return row


def build_session_summary_table(sessions, signal_present_value=100, signal_absent_value=0) -> pd.DataFrame:
    """Run `summarize_session` over a list of Session objects and stack
    the results into one dataframe -- this is what gets written to
    `2_pipeline/1a_performance/out/session_summary.csv` for later steps
    to read (Rule #1: they load from this `out/`, not from 0_data again)."""
    rows = [summarize_session(ses, signal_present_value, signal_absent_value) for ses in sessions]
    return pd.DataFrame(rows)
