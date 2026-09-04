# -*- coding: utf-8 -*-
"""
Generate a small synthetic dataset under 0_data/ with the right shape
(sessiondata/trialdata/behaviordata/videodata csvs) so scripts in
1_scripts/ can be dry-run end-to-end without real recordings -- useful
when developing a new analysis step, or in CI. NOT part of the pipeline
itself (no numbered prefix, doesn't use get_pipeline_paths): copy this
into a new script only as a starting point for a real one.

Schema convention (confirmed against real data): `stimStart` and
`rewardZoneStart`/`rewardZoneEnd` in trialdata are TIMESTAMPS (seconds),
one per trial -- not positions. `zpos` in behaviordata/videodata is a
monotonically ACCUMULATING position (like an odometer -- total distance
run since session start), not a per-trial-resetting corridor coordinate;
a real session's `zpos` can span many thousands of cm. Position-relative-
to-stimulus-onset plots therefore derive their own reference point per
trial (the position at that trial's stimStart time) rather than reading
it directly off a position column -- see infotheory/psth.py.

If your real data instead has `stimStart` as a position and `zpos` that
resets every trial, the pipeline's PSTH alignment needs the opposite
convention -- check with whoever built the raw-data export before
trusting a mismatch silently (see the zpos QC print in 1c_behavior.py).

Trials with a lick response get a realistic anticipatory running-speed
dip and a burst of lick events in a time window before/around when
reward would be delivered, so event-aligned PSTHs have something real
to recover, whether aligned by time or by trial-relative position.

Usage:  python make_fake_data.py [--out 0_data] [--seed 0]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SIGNAL_LEVELS = {
    "DM": [0, 100],
    "DP": [0, 20, 40, 60, 80, 100],
    "DN": [0, 20, 40, 60, 80, 100],
}
ANIMALS_PER_PROTOCOL = {"DM": 2, "DP": 2, "DN": 3}
SESSIONS_PER_ANIMAL = 2
TRIALS_PER_SESSION = 350
DT = 0.02  # behavior/video sampling interval, seconds

BASELINE_SPEED = 20.0        # cm/s
REWARD_FRAC_CENTER = 0.55    # anticipatory dip centered here (fraction of trial duration)
REWARD_FRAC_WIDTH = 0.12


def make_trial_speed(rng: np.random.Generator, duration: float, anticipating_reward: bool):
    """One trial's (t, runspeed) trace: a roughly flat baseline speed,
    with a Gaussian anticipatory slowdown centered on where the reward
    window falls (as a fraction of the trial's own duration) for trials
    where the animal is about to lick."""
    n = max(int(duration / DT), 10)
    t = np.arange(n) * DT
    f = t / t[-1]

    speed = np.full(n, BASELINE_SPEED)
    if anticipating_reward:
        speed = speed - 0.6 * BASELINE_SPEED * np.exp(-0.5 * ((f - REWARD_FRAC_CENTER) / REWARD_FRAC_WIDTH) ** 2)
    speed = np.clip(speed + rng.normal(0, 1.5, n), 0.5, None)
    return t, speed


def make_session(rng: np.random.Generator, protocol: str, animal_id: str, sessiondate: str,
                  n_trials: int = TRIALS_PER_SESSION):
    signal_levels = SIGNAL_LEVELS[protocol]
    signal = rng.choice(signal_levels, size=n_trials)

    # Psychometric-ish lick probability, plus a session-wide engagement
    # drop-off (a block of disengaged trials starting partway through,
    # like a mouse losing motivation near the end):
    mu, sigma, lapse, guess = 45, 18, 0.08, 0.06
    p_lick = guess + (1 - guess - lapse) * 0.5 * (1 + np.tanh((signal - mu) / sigma))

    disengage_start = rng.integers(int(n_trials * 0.5), int(n_trials * 0.95))
    engaged = np.ones(n_trials, dtype=bool)
    engaged[disengage_start:] = rng.random(n_trials - disengage_start) > 0.7
    for _ in range(rng.integers(1, 4)):
        lapse_start = rng.integers(0, disengage_start - 10) if disengage_start > 10 else 0
        engaged[lapse_start: lapse_start + rng.integers(3, 10)] = False

    p_lick_effective = np.where(engaged, p_lick, guess * 0.5)
    lick_response = (rng.random(n_trials) < p_lick_effective).astype(int)

    trial_dt = rng.normal(6.0, 1.0, n_trials).clip(3, 12)  # seconds/trial
    t_start = np.concatenate([[0.0], np.cumsum(trial_dt)[:-1]])
    t_end = t_start + trial_dt * 0.8
    stim_start = t_start + trial_dt * 0.3                                   # TIME (s)
    reward_zone_start = t_start + trial_dt * (REWARD_FRAC_CENTER - REWARD_FRAC_WIDTH)  # TIME (s)
    reward_zone_end = t_start + trial_dt * (REWARD_FRAC_CENTER + REWARD_FRAC_WIDTH)    # TIME (s)

    session_id = f"{animal_id}_{sessiondate}"
    trialdata = pd.DataFrame({
        "trialNumber": np.arange(1, n_trials + 1),
        "signal": signal,
        "lickResponse": lick_response,
        "engaged": engaged.astype(int),
        "tStart": t_start,
        "tEnd": t_end,
        "stimStart": stim_start,
        "rewardZoneStart": reward_zone_start,
        "rewardZoneEnd": reward_zone_end,
        "session_id": session_id,
    })

    # Build each trial's own (t, speed) trace, then concatenate onto the
    # session's absolute time axis. Position is the CUMULATIVE integral
    # of speed over the whole session (an odometer) -- never resets.
    ts_all, speed_all, trialnum_all = [], [], []
    for k in range(n_trials):
        t_rel, speed = make_trial_speed(rng, trial_dt[k], anticipating_reward=bool(lick_response[k]))
        ts_all.append(t_start[k] + t_rel)
        speed_all.append(speed)
        trialnum_all.append(np.full(len(t_rel), k + 1))

    ts = np.concatenate(ts_all)
    runspeed = np.concatenate(speed_all)
    trial_idx = np.concatenate(trialnum_all)
    dt_actual = np.diff(ts, prepend=ts[0] - DT)
    zpos = np.cumsum(runspeed * dt_actual)  # monotonically increasing, like a real odometer

    sessiondata = pd.DataFrame({
        "animal_id": [animal_id],
        "sessiondate": [sessiondate],
        "session_id": [session_id],
        "protocol": [protocol],
        "fs": [30.0],
    })

    behaviordata = pd.DataFrame({
        "ts": ts,
        "zpos": zpos,
        "runspeed": runspeed,
        "trialNumber": trial_idx,
    })

    # Lick channel: individual 0/1 lick-detector events, not just the
    # per-trial lickResponse summary. Bursts concentrated in the reward
    # TIME window on trials where the animal licked; occasional
    # off-target licks elsewhere as a baseline.
    in_reward_window = np.zeros(len(ts), dtype=bool)
    for k in range(n_trials):
        mask = trial_idx == k + 1
        in_reward_window[mask] = (ts[mask] >= reward_zone_start[k]) & (ts[mask] <= reward_zone_end[k])
    anticipating = np.isin(trial_idx, np.where(lick_response == 1)[0] + 1)
    lick_prob = np.where(in_reward_window & anticipating, 0.35, 0.01)
    lick = (rng.random(len(ts)) < lick_prob).astype(int)
    behaviordata["lick"] = lick

    # Video trace: pupil rises anticipating reward (same trials that
    # slow down), motion energy is deliberately unrelated noise (a
    # negative control for the MI/regression comparison). Also include
    # pupil position (gaze) and the first two video PCs, matching the
    # real videodata schema (pupil_xpos/pupil_ypos, videoPC_0..29) --
    # videoPC_0 loosely tracks motion energy (as it typically does in
    # real face/body video PCA), videoPC_1 and pupil position are
    # deliberately unrelated noise.
    pupil_area = 800 + 60 * (in_reward_window & anticipating) + rng.normal(0, 40, len(ts))
    motionenergy = np.abs(rng.normal(0.5, 0.2, len(ts)))
    pupil_xpos = rng.normal(0, 2, len(ts))
    pupil_ypos = rng.normal(0, 2, len(ts))
    videoPC_0 = 0.7 * (motionenergy - motionenergy.mean()) / motionenergy.std() + rng.normal(0, 0.5, len(ts))
    videoPC_1 = rng.normal(0, 1, len(ts))
    videodata = pd.DataFrame({
        "ts": ts,
        "zpos": zpos,
        "pupil_area": pupil_area.clip(0, None),
        "pupil_xpos": pupil_xpos,
        "pupil_ypos": pupil_ypos,
        "motionenergy": motionenergy,
        "videoPC_0": videoPC_0,
        "videoPC_1": videoPC_1,
    })

    return sessiondata, trialdata, behaviordata, videodata


def main(out: Path, seed: int):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=20).strftime("%Y_%m_%d").tolist()
    di = 0

    for protocol, n_animals in ANIMALS_PER_PROTOCOL.items():
        for ia in range(n_animals):
            animal_id = f"{protocol}L{ia:02d}"
            for _ in range(SESSIONS_PER_ANIMAL):
                sessiondate = dates[di % len(dates)]
                di += 1
                folder = out / protocol / animal_id / sessiondate
                folder.mkdir(parents=True, exist_ok=True)

                sessiondata, trialdata, behaviordata, videodata = make_session(
                    rng, protocol, animal_id, sessiondate)

                sessiondata.to_csv(folder / "sessiondata.csv")
                trialdata.to_csv(folder / "trialdata.csv")
                behaviordata.to_csv(folder / "behaviordata.csv")
                videodata.to_csv(folder / "videodata.csv")

    print(f"Wrote synthetic sessions under {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "0_data")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.out, args.seed)
