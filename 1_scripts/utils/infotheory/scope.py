# -*- coding: utf-8 -*-
"""
scope.py
=========
Run scope for every script: WHICH sessions/mice are analysed and at WHICH LEVEL results are
aggregated for plotting/reporting (see project_config.py for the definitions).

    parser = argparse.ArgumentParser(); add_scope_args(parser); args = parser.parse_args()
    scope  = scope_from_args(args)
    triples = select_sessions(discover_sessions(PROTOCOLS), scope)   # (protocol, animal_id, date)
    table_dir, fig_dir = scoped_dirs(paths, scope)

Safety rule for Rule #1 (later steps read this script's out/): the FULL-dataset results always live
in out/ and out/figures/. A run restricted to some animals/sessions never overwrites them -- it
writes to out/selection/<tag>/ instead, so a quick look at one mouse cannot corrupt downstream inputs.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from . import project_config as cfg


@dataclass(frozen=True)
class Scope:
    animals: tuple | None = None
    sessions: tuple | None = None
    aggregate: str = cfg.AGGREGATE_BY
    n_jobs: int = cfg.N_JOBS_SESSIONS

    def __post_init__(self):
        if self.aggregate not in cfg.AGGREGATES:
            raise ValueError(f"aggregate must be one of {cfg.AGGREGATES}, got {self.aggregate!r}")

    @property
    def is_full(self) -> bool:
        return self.animals is None and self.sessions is None

    @property
    def tag(self) -> str:
        """Folder-safe description of the selection ('full' if none)."""
        if self.is_full:
            return "full"
        parts = []
        if self.animals:
            parts.append("animals-" + "+".join(self.animals))
        if self.sessions:
            s = "+".join(self.sessions)
            parts.append("sessions-" + (s if len(s) < 60 else f"n{len(self.sessions)}-" +
                                        hashlib.sha1(s.encode()).hexdigest()[:8]))
        return "_".join(parts)

    def describe(self) -> str:
        sel = "all sessions" if self.is_full else self.tag
        return f"scope: {sel}; aggregate by {self.aggregate}; n_jobs={self.n_jobs}"


def add_scope_args(parser):
    parser.add_argument("--animals", nargs="+", default=None, metavar="ID",
                        help="analyse only these mice (default: project_config.ANIMALS)")
    parser.add_argument("--sessions", nargs="+", default=None, metavar="SESSION_ID",
                        help="analyse only these sessions, '<animal>_<date>' (default: project_config.SESSIONS)")
    parser.add_argument("--aggregate", choices=cfg.AGGREGATES, default=None,
                        help="unit of analysis for plots/reports (default: project_config.AGGREGATE_BY)")
    parser.add_argument("--n-jobs", type=int, default=None,
                        help="sessions processed in parallel (default: project_config.N_JOBS_SESSIONS)")
    parser.add_argument("--recompute", action="store_true",
                        help="ignore the per-session caches in store/ and recompute")
    return parser


def scope_from_args(args) -> Scope:
    animals = args.animals if args.animals else cfg.ANIMALS
    sessions = args.sessions if args.sessions else cfg.SESSIONS
    return Scope(animals=tuple(animals) if animals else None,
                 sessions=tuple(sessions) if sessions else None,
                 aggregate=args.aggregate or cfg.AGGREGATE_BY,
                 n_jobs=args.n_jobs or cfg.N_JOBS_SESSIONS)


def select_sessions(triples, scope: Scope):
    """Filter discover_sessions() output ((protocol, animal_id, sessiondate) tuples) by the scope.
    Nothing is loaded here. Raises with the available ids if the selection matches nothing, and
    warns about requested names that do not exist."""
    triples = list(triples)
    ids = {f"{a}_{d}": (p, a, d) for p, a, d in triples}
    animals_present = sorted({a for _, a, _ in triples})
    if scope.animals:
        for a in scope.animals:
            if a not in animals_present:
                print(f"WARNING: requested animal {a!r} not found (available: {animals_present})")
    if scope.sessions:
        for s in scope.sessions:
            if s not in ids:
                print(f"WARNING: requested session {s!r} not found")
    out = [(p, a, d) for p, a, d in triples
           if (not scope.animals or a in scope.animals) and (not scope.sessions or f"{a}_{d}" in scope.sessions)]
    if not out:
        raise SystemExit(f"Selection matched no sessions. Animals: {animals_present}; "
                         f"sessions: {sorted(ids)[:20]}{' ...' if len(ids) > 20 else ''}")
    return out


def scoped_dirs(paths, scope: Scope):
    """(table_dir, figure_dir) for this run. Full dataset -> out/, out/figures/;
    a selection -> out/selection/<tag>/ (never touches the full-dataset outputs)."""
    base = paths.out if scope.is_full else paths.out / "selection" / scope.tag
    base = Path(base)
    (base / "figures").mkdir(parents=True, exist_ok=True)
    return base, base / "figures"


def data_stamp(protocol: str, animal_id: str, sessiondate: str, files=("sessiondata.csv", "trialdata.csv",
               "behaviordata.csv", "videodata.csv", "celldata.csv")) -> str:
    """Fingerprint (size + mtime) of a session's raw files, to put in cache keys so that changed raw
    data invalidates that session's cache. Reads file metadata only."""
    from .session import get_data_folder
    folder = get_data_folder() / protocol / animal_id / sessiondate
    parts = []
    for f in files:
        p = folder / f
        parts.append(f"{f}:{p.stat().st_size}:{int(p.stat().st_mtime)}" if p.exists() else f"{f}:-")
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:10]


# --------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------- #
def unit_columns(aggregate: str) -> list[str]:
    """Columns that identify one analysis unit (in addition to any condition columns)."""
    return {"pooled": [], "session": ["session_id"], "mouse": ["animal_id"]}[aggregate]


def unit_noun(aggregate: str) -> str:
    return {"pooled": "dataset", "session": "session", "mouse": "mouse"}[aggregate]
