# -*- coding: utf-8 -*-
"""
Generate a small synthetic dataset under 0_data/ with the right shape
(sessiondata/trialdata/behaviordata/videodata csvs) so scripts in
1_scripts/ can be dry-run end-to-end without real recordings -- useful
when developing a new analysis step, or in CI. NOT part of the pipeline
itself (no numbered prefix, doesn't use get_pipeline_paths): copy this
into a new script only as a starting point for a real one.

Schema convention (confirmed against the real pipeline's tensor_utils.py
and behavior_signals.py, which always compute `zpos_F - z_T[k]` where
`z_T = trialdata['stimStart']`): `stimStart`/`rewardZoneStart`/
`rewardZoneEnd` in trialdata are POSITIONS, on the SAME scale as `zpos`
in behaviordata/videodata -- not timestamps. `zpos` itself is a
monotonically ACCUMULATING position (like an odometer -- total distance
run since session start), not a per-trial-resetting corridor coordinate;
a real session's `zpos` can span many thousands of cm, and each trial's
own `stimStart` sits at the matching point on that same accumulating
scale (computed here from where the animal's cumulative position is at
~30% of the way through that trial's own duration).

Trials with a lick response get a realistic anticipatory running-speed
dip and a burst of lick events around when reward would be delivered,
so event-aligned PSTHs (whether aligned by time via a derived onset
time, or directly by position via `zpos - stimStart`) have something
real to recover.

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
STIM_FRAC = 0.3               # "stimulus" fires at this fraction of the trial's own duration


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

    # stimStart/rewardZone are POSITIONS (see module docstring), read off
    # the just-built zpos trace at the appropriate fractional-time point
    # within each trial -- i.e. "where the animal's odometer reads" when
    # the stimulus/reward-window events happen, not a time value.
    def zpos_at_time(target_times):
        idx = np.clip(np.searchsorted(ts, target_times), 0, len(zpos) - 1)
        return zpos[idx]

    stim_start = zpos_at_time(t_start + trial_dt * STIM_FRAC)
    reward_zone_start = zpos_at_time(t_start + trial_dt * (REWARD_FRAC_CENTER - REWARD_FRAC_WIDTH))
    reward_zone_end = zpos_at_time(t_start + trial_dt * (REWARD_FRAC_CENTER + REWARD_FRAC_WIDTH))

    session_id = f"{animal_id}_{sessiondate}"
    trialdata = pd.DataFrame({
        "trialNumber": np.arange(1, n_trials + 1),
        "signal": signal,
        "lickResponse": lick_response,
        "engaged": engaged.astype(int),
        "tStart": t_start,
        "tEnd": t_end,
        "stimStart": stim_start,                  # POSITION (cm), same scale as zpos
        "rewardZoneStart": reward_zone_start,      # POSITION (cm)
        "rewardZoneEnd": reward_zone_end,          # POSITION (cm)
        "session_id": session_id,
    })

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
        t_rw_start = t_start[k] + trial_dt[k] * (REWARD_FRAC_CENTER - REWARD_FRAC_WIDTH)
        t_rw_end = t_start[k] + trial_dt[k] * (REWARD_FRAC_CENTER + REWARD_FRAC_WIDTH)
        in_reward_window[mask] = (ts[mask] >= t_rw_start) & (ts[mask] <= t_rw_end)
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


CELL_AREA_ORDER = ["V1", "PM", "AL", "RSP"]     # AL/RSP appear in only some sessions, and never labeled
                                                  # (matches the real project's example celldata.csv)
FOV_SIZE_UM = 600.0                              # per-session local coordinate span (NOT shared/aligned
                                                  # across sessions -- see 2a_cell_distribution.py)
CELL_DEPTH_RANGE = (90.0, 450.0)
LAYER_SPLIT_DEPTHS_UM = (300.0, 380.0)           # depth < first -> "L2/3", < second -> "L4", else "L5"
LABEL_FRAC = 0.12                                # fraction of V1/PM cells labeled (redcell=1)


def make_celldata(rng: np.random.Generator, session_id: str, recombinase_by_area: dict) -> pd.DataFrame:
    """One session's celldata (DN only -- see session.py's module
    docstring: cell recordings only happen for DN). Each area gets its
    own region of a shared per-session coordinate canvas (so different
    areas' cells don't spatially overlap, matching real distinct-ROI
    imaging), and that canvas itself is independently randomized PER
    SESSION -- i.e. NOT aligned across sessions, same as real 2p FOVs
    from different recording days/animals. Labeled cells cluster into a
    few small "hotspots" per area so a proximity-based filter (see
    infotheory.celldata_utils.filter_nearlabeled) has real spatial
    structure to find, rather than labeled/unlabeled cells being
    uniformly interspersed.

    `recombinase_by_area`: {'V1': 'cre'/'flp', 'PM': 'cre'/'flp'} --
    recombinase is a property of the INJECTION, which is per (session,
    area), not per animal -- a session can have V1 injected with one
    driver line and PM with another (confirmed against the lab's own
    session.py: `reset_label_threshold` sets celldata['recombinase']
    from sessiondata['<area>_recombinase']). Unlabeled cells always get
    'non' regardless of area, matching that same convention."""
    areas = ["V1", "PM"]
    if rng.random() < 0.5:
        areas.append("AL")
    if rng.random() < 0.4:
        areas.append("RSP")

    rows = []
    cell_counter = 0
    for area in areas:
        n_cells = int(rng.integers(80, 260))
        center_x = rng.uniform(100, FOV_SIZE_UM - 100)
        center_y = rng.uniform(100, FOV_SIZE_UM - 100)
        spread = 180.0

        can_label = area in ("V1", "PM")
        n_labeled = int(n_cells * LABEL_FRAC) if can_label else 0
        n_hotspots = max(1, n_labeled // 8) if n_labeled else 1
        hotspots = [(center_x + rng.uniform(-spread, spread), center_y + rng.uniform(-spread, spread))
                    for _ in range(n_hotspots)]

        for i in range(n_cells):
            is_labeled = i < n_labeled
            if is_labeled:
                hx, hy = hotspots[i % n_hotspots]
                x = float(np.clip(hx + rng.normal(0, 15), 0, FOV_SIZE_UM))
                y = float(np.clip(hy + rng.normal(0, 15), 0, FOV_SIZE_UM))
            else:
                x = float(np.clip(center_x + rng.normal(0, spread), 0, FOV_SIZE_UM))
                y = float(np.clip(center_y + rng.normal(0, spread), 0, FOV_SIZE_UM))

            depth = float(np.clip(rng.normal(200, 80), *CELL_DEPTH_RANGE))
            layer = ("L2/3" if depth < LAYER_SPLIT_DEPTHS_UM[0]
                      else "L4" if depth < LAYER_SPLIT_DEPTHS_UM[1] else "L5")
            radius = float(np.clip(rng.normal(7.0, 1.5), 3, 14))
            npix = float(np.clip(rng.normal(150, 40), 30, 400))
            npix_soma = float(np.clip(npix * rng.uniform(0.5, 0.75), 15, 300))

            rows.append({
                "iscell": 1.0, "iscell_prob": float(np.clip(rng.normal(0.9, 0.1), 0.3, 1.0)),
                "skew": float(np.clip(rng.normal(2.0, 1.0), -1, 8)),
                "radius": radius, "npix_soma": npix_soma, "npix": npix,
                "xloc": x, "yloc": y,
                "redcell": 1.0 if is_labeled else 0.0,
                "frac_of_ROI_red": rng.uniform(0.6, 1.0) if is_labeled else rng.uniform(0, 0.15),
                "frac_red_in_ROI": rng.uniform(0.6, 1.0) if is_labeled else rng.uniform(0, 0.15),
                "chan2_prob": rng.uniform(0.7, 1.0) if is_labeled else rng.uniform(0, 0.3),
                "nredcells": n_labeled,
                "plane_idx": areas.index(area), "roi_idx": areas.index(area), "plane_in_roi_idx": 0,
                "roi_name": area, "depth": depth, "power_mw": 40.0,
                "labeled": "lab" if is_labeled else "unl",
                "arealabel": f"{area}{'lab' if is_labeled else 'unl'}",
                "meanF": rng.uniform(200, 900),
                "meanF_chan2": rng.uniform(50, 400) if is_labeled else rng.uniform(0, 80),
                "noise_level": float(np.clip(rng.normal(14, 6), 3, 70)),
                "event_rate": float(np.clip(rng.normal(0.08, 0.03), 0.005, 0.25)),
                "cell_id": f"{session_id}_{cell_counter}",
                "layer": layer,
                "recombinase": recombinase_by_area.get(area, "non") if is_labeled else "non",
                "session_id": session_id,
            })
            cell_counter += 1

    return pd.DataFrame(rows)


FS_IMAGING = 8.0  # Hz, synthetic imaging frame rate (stored in sessiondata's 'fs' column too)

# Per-cell "quality class" mix -- mostly normal cells, plus a deliberate
# minority of each QC-relevant failure mode, so 2b_activity_statistics.py
# (and the qc_lib thresholds it exercises) has real outliers to actually
# detect, not just a uniformly clean population.
QUALITY_CLASS_PROBS = {
    "normal": 0.82, "flat": 0.03, "lowrate": 0.03, "noisy": 0.04,
    "highfano": 0.04, "nan": 0.02, "lowskew": 0.02,
}


def make_deconvdata(rng: np.random.Generator, celldata: pd.DataFrame, duration_s: float):
    """Deconvolved-activity traces for one session's cells, ALIGNED 1:1
    with `celldata`'s row order (columns = celldata['cell_id'], same
    order) -- matching the real project's deconvdata.csv convention
    exactly (confirmed against an uploaded real example: same column
    names, same order as celldata's cell_id). Also generates the two
    companion files real sessions carry: real per-frame timestamps
    (Ftsdata.csv) and the session-wide red-channel "Fchan2" signal
    (Fchan2data.csv) -- confirmed against preprocesslib.py's
    proc_imaging: a zscored, session-wide ABSOLUTE red-channel
    fluorescence CHANGE signal (a static structural marker -- tdTomato
    -- so abrupt changes flag likely z-motion/refocusing artifacts),
    NOT a per-cell trace.

    Assigns each cell a "quality class" (QUALITY_CLASS_PROBS) and
    generates a trace matching that failure mode, so downstream QC
    logic (infotheory.qc_lib) has genuine outliers to find:
        normal    : sparse positive events, realistic zero-inflation
        flat      : ~zero variance (dead/silent channel)
        lowrate   : real events, but far too sparse (below qc_lib's
                    default RATE_THR)
        noisy     : real events plus heavy noise SPECIFICALLY DURING
                    the Fchan2 artifact-burst windows (elsewhere just
                    light baseline noise) -- a genuine, detectable
                    shared-motion-artifact signature, not just
                    independent per-cell noise. ALSO pushes this cell's
                    celldata['noise_level'] above qc_lib's default
                    NOISE_THR, so all three data sources agree (returns
                    an updated celldata, not just deconvdata, for this
                    reason).
        highfano  : long silent stretches punctuated by big bursts
                    (unstable/artifact-like activity -- high Fano factor)
        nan       : real events, but a chunk of frames are NaN
        lowskew   : high, roughly-constant baseline with occasional
                    downward dips instead of upward events -- looks
                    like a saturating/clipping artifact (negative skew,
                    the opposite of a normal sparse-positive-event trace)

    Returns
    -------
    deconvdata : DataFrame (T frames, N cells)
    celldata : DataFrame, same as input but with noise_level patched
        for cells assigned the "noisy" class (see above)
    ts_F : 1D array, length T -- real per-frame timestamps (seconds)
    fchan2 : 1D array, length T -- the session-wide Fchan2 signal
    """
    n_cells = len(celldata)
    T = max(int(duration_s * FS_IMAGING), 500)
    ts_F = np.linspace(0, duration_s, T)
    classes = rng.choice(list(QUALITY_CLASS_PROBS.keys()), size=n_cells,
                          p=list(QUALITY_CLASS_PROBS.values()))

    # Session-wide Fchan2 (red-channel motion-artifact) signal: mostly
    # N(0,1) noise, with a handful of short "motion artifact" bursts of
    # elevated |z| -- matching what a real z-drift/refocusing event
    # would look like in a static structural-marker channel.
    fchan2 = rng.normal(0, 1, T)
    artifact_mask = np.zeros(T, dtype=bool)
    for _ in range(int(rng.integers(3, 7))):
        start = int(rng.integers(0, max(T - 30, 1)))
        length = int(rng.integers(10, 30))
        seg = slice(start, min(start + length, T))
        fchan2[seg] += rng.choice([-1.0, 1.0]) * rng.uniform(4.0, 8.0)
        artifact_mask[seg] = True

    traces = np.zeros((T, n_cells))
    noise_level = celldata["noise_level"].to_numpy().copy()

    for i, cls in enumerate(classes):
        if cls == "normal":
            rate = rng.uniform(0.03, 0.15)
            events = rng.random(T) < rate
            traces[:, i] = rng.gamma(shape=2.0, scale=25.0, size=T) * events
        elif cls == "flat":
            traces[:, i] = np.abs(rng.normal(0, 1e-8, T))
        elif cls == "lowrate":
            rate = rng.uniform(0.0005, 0.003)
            events = rng.random(T) < rate
            traces[:, i] = rng.gamma(shape=2.0, scale=20.0, size=T) * events
        elif cls == "noisy":
            rate = rng.uniform(0.05, 0.15)
            events = rng.random(T) < rate
            artifact_noise = np.where(artifact_mask, np.abs(rng.normal(0, 60, T)),
                                       np.abs(rng.normal(0, 5, T)))
            traces[:, i] = rng.gamma(shape=2.0, scale=25.0, size=T) * events + artifact_noise
            noise_level[i] = rng.uniform(120, 250)  # above qc_lib.NOISE_THR (100)
        elif cls == "highfano":
            trace = np.zeros(T)
            for _ in range(int(rng.integers(3, 8))):
                start = int(rng.integers(0, max(T - 50, 1)))
                length = int(rng.integers(10, 40))
                seg = min(length, T - start)
                trace[start:start + seg] += rng.gamma(shape=2.0, scale=80.0, size=seg)
            traces[:, i] = trace
        elif cls == "nan":
            rate = rng.uniform(0.03, 0.15)
            events = rng.random(T) < rate
            trace = rng.gamma(shape=2.0, scale=25.0, size=T) * events
            nan_mask = rng.random(T) < rng.uniform(0.02, 0.1)
            trace[nan_mask] = np.nan
            traces[:, i] = trace
        elif cls == "lowskew":
            base = rng.uniform(50, 100)
            dips = rng.random(T) < 0.05
            dip_amp = rng.gamma(shape=2.0, scale=30.0, size=T) * dips
            traces[:, i] = np.clip(base - dip_amp + rng.normal(0, 3, T), 0, None)

    celldata_out = celldata.copy()
    celldata_out["noise_level"] = noise_level
    deconvdata = pd.DataFrame(traces, columns=celldata["cell_id"].to_numpy())
    return deconvdata, celldata_out, ts_F, fchan2


def main(out: Path, seed: int):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=20).strftime("%Y_%m_%d").tolist()
    di = 0

    for protocol, n_animals in ANIMALS_PER_PROTOCOL.items():
        for ia in range(n_animals):
            animal_id = f"{protocol}L{ia:02d}"
            # Recombinase is a property of the INJECTION (per session,
            # per area -- see make_celldata's docstring), fixed here per
            # ANIMAL (injections aren't redone between repeat sessions of
            # the same animal) but independently for V1 vs PM, since a
            # dual-injection animal can have different driver lines in
            # each area.
            recombinase_by_area = {"V1": rng.choice(["cre", "flp"]), "PM": rng.choice(["cre", "flp"])}
            for _ in range(SESSIONS_PER_ANIMAL):
                sessiondate = dates[di % len(dates)]
                di += 1
                folder = out / protocol / animal_id / sessiondate
                folder.mkdir(parents=True, exist_ok=True)

                sessiondata, trialdata, behaviordata, videodata = make_session(
                    rng, protocol, animal_id, sessiondate)

                if protocol == "DN":
                    # 'fs' here is specifically read by spike_stats.get_frame_rate
                    # as the IMAGING frame rate -- distinct from behaviordata's own
                    # sampling (DT=0.02s => 50Hz), which nothing reads from this column.
                    sessiondata["fs"] = FS_IMAGING

                sessiondata.to_csv(folder / "sessiondata.csv")
                trialdata.to_csv(folder / "trialdata.csv")
                behaviordata.to_csv(folder / "behaviordata.csv")
                videodata.to_csv(folder / "videodata.csv")

                if protocol == "DN":
                    session_id = f"{animal_id}_{sessiondate}"
                    celldata = make_celldata(rng, session_id, recombinase_by_area)
                    session_duration_s = float(trialdata["tEnd"].iloc[-1]) if len(trialdata) else 1800.0
                    deconvdata, celldata, ts_F, fchan2 = make_deconvdata(rng, celldata, session_duration_s)
                    celldata.to_csv(folder / "celldata.csv")
                    deconvdata.to_csv(folder / "deconvdata.csv")
                    pd.DataFrame({"ts": ts_F}).to_csv(folder / "Ftsdata.csv")
                    pd.DataFrame({"Fchan2": fchan2}).to_csv(folder / "Fchan2data.csv")

    print(f"Wrote synthetic sessions under {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "0_data")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.out, args.seed)
