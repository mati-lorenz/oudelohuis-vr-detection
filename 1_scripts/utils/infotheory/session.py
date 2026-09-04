# -*- coding: utf-8 -*-
"""
Generic session loader.

A "session" is one behavioral (and, for DN, imaging) recording,
identified by (protocol, animal_id, sessiondate) and stored under

    0_data/<protocol>/<animal_id>/<sessiondate>/
        sessiondata.csv     one row, session-level metadata
        trialdata.csv       one row per trial
        behaviordata.csv    continuous position/running-speed trace
        videodata.csv       continuous pupil/motion-energy trace
        celldata.csv        one row per cell (DN only)
        <calciumversion>data.csv, Ftsdata.csv, Fchan2data.csv (DN only)

This module only discovers sessions and does the shallow/behavioral load
(sessiondata, trialdata, behaviordata, videodata). Calcium-trace loading
for the single-cell/multi-area steps is intentionally left out of this
version -- add a `load_calciumdata` flag here (mirroring the old
`session.py`'s `load_data`) once step 2 needs it, rather than guessing
its shape now.

Trimmed down and reorganized from Matthijs Oude Lohuis' original loader
(Champalimaud, 2023).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)

PROTOCOLS = ("DM", "DP", "DN")


def get_data_folder() -> Path:
    """Root of `0_data/`, found by walking up from this file's location."""
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        data_dir = candidate / "0_data"
        if data_dir.is_dir():
            return data_dir
    raise FileNotFoundError("Could not locate '0_data/' above %s" % here)


class Session:
    """One session's data. `load()` is opt-in per data stream so a script
    that only needs trial data doesn't pay for reading video traces."""

    def __init__(self, protocol: str, animal_id: str, sessiondate: str):
        self.protocol = protocol
        self.animal_id = animal_id
        self.sessiondate = sessiondate
        self.session_id = f"{animal_id}_{sessiondate}"
        self.data_folder = get_data_folder() / protocol / animal_id / sessiondate

        self.sessiondata: pd.DataFrame | None = None
        self.trialdata: pd.DataFrame | None = None
        self.behaviordata: pd.DataFrame | None = None
        self.videodata: pd.DataFrame | None = None
        self.celldata: pd.DataFrame | None = None

    def __repr__(self):
        return f"Session({self.session_id}, protocol={self.protocol})"

    def _read_csv(self, name: str) -> pd.DataFrame | None:
        path = self.data_folder / name
        if not path.exists():
            return None
        return pd.read_csv(path, sep=",", index_col=0)

    def load(self, load_behaviordata: bool = True, load_videodata: bool = False,
              load_celldata: bool = False) -> "Session":
        self.sessiondata = self._read_csv("sessiondata.csv")
        if self.sessiondata is None:
            raise FileNotFoundError(f"No sessiondata.csv in {self.data_folder}")

        self.trialdata = self._read_csv("trialdata.csv")

        if load_behaviordata:
            self.behaviordata = self._read_csv("behaviordata.csv")
            if self.behaviordata is None:
                logger.warning("No behaviordata.csv for %s", self.session_id)

        if load_videodata:
            self.videodata = self._read_csv("videodata.csv")
            if self.videodata is None:
                logger.warning("No videodata.csv for %s", self.session_id)

        if load_celldata:
            self.celldata = self._read_csv("celldata.csv")

        return self


def discover_sessions(protocols: Sequence[str] = PROTOCOLS) -> list[tuple[str, str, str]]:
    """Scan `0_data/<protocol>/<animal_id>/<sessiondate>/` and return
    (protocol, animal_id, sessiondate) for every folder with a
    sessiondata.csv. Read-only: never touches 0_data/ contents."""
    data_folder = get_data_folder()
    found = []
    for protocol in protocols:
        protocol_dir = data_folder / protocol
        if not protocol_dir.is_dir():
            continue
        for animal_dir in sorted(p for p in protocol_dir.iterdir() if p.is_dir()):
            for session_dir in sorted(p for p in animal_dir.iterdir() if p.is_dir()):
                if (session_dir / "sessiondata.csv").exists():
                    found.append((protocol, animal_dir.name, session_dir.name))
    return found


def load_sessions(protocols: Sequence[str] = PROTOCOLS, min_trials: int = 0,
                   load_behaviordata: bool = True, load_videodata: bool = False,
                   load_celldata: bool = False, require_pupil: bool = False,
                   only_session_ids: Sequence[str] | None = None,
                   verbose: bool = True) -> list[Session]:
    """Discover and load every session for the given protocol(s). No
    behavioral exclusions beyond `min_trials`/`require_pupil` are applied
    here on purpose -- step 1a surveys the raw data as-is; heavier
    performance-based exclusion criteria belong in 1b/1d.

    `only_session_ids`: skip everything not in this set (e.g. the
    sessions that passed 1b's inclusion criteria) -- checked before any
    file is opened, so excluded sessions cost nothing to skip."""
    only_session_ids = set(only_session_ids) if only_session_ids is not None else None
    sessions = []
    n_skipped = 0
    for protocol, animal_id, sessiondate in discover_sessions(protocols):
        session_id = f"{animal_id}_{sessiondate}"
        if only_session_ids is not None and session_id not in only_session_ids:
            continue

        ses = Session(protocol, animal_id, sessiondate)
        try:
            ses.load(load_behaviordata=load_behaviordata, load_videodata=load_videodata,
                     load_celldata=load_celldata)
        except Exception as exc:
            logger.warning("Skipping %s: %s", ses.session_id, exc)
            n_skipped += 1
            continue

        if ses.trialdata is not None and len(ses.trialdata) < min_trials:
            continue
        if require_pupil and (ses.videodata is None or "pupil_area" not in ses.videodata.columns):
            continue

        sessions.append(ses)

    if verbose:
        msg = f"Loaded {len(sessions)} sessions ({', '.join(protocols)})"
        if n_skipped:
            msg += f", skipped {n_skipped} with load errors"
        print(msg)
    return sessions
