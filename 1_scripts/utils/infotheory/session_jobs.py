# -*- coding: utf-8 -*-
"""
session_jobs.py
================
Memory-safe, cached, per-session execution.

Old pattern: `sessions = load_sessions(...)` loaded EVERY session into RAM at once and then
shipped the whole session objects to the worker processes (pickled copies) -- RAM ~ (all sessions)
x (1 + n_jobs). New pattern: only session IDs travel to the workers; each worker loads ONE
session, computes its records, saves them to a per-session cache file and frees the session
before taking the next one. Peak RAM ~ n_jobs x (one session). `N_JOBS_SESSIONS` is the lever
(1 = lowest memory; the whole run is then strictly sequential).

Per-session results are cached (parameter-hashed file names, see cache.cache_path), so re-running
only computes sessions/settings that are new, and a crash loses at most the sessions in flight.
"""
from __future__ import annotations

import gc
import hashlib
import itertools
import os
import zlib

import pandas as pd
from joblib import Parallel, delayed

from .cache import cache_path


def run_sessions(session_ids, process_fn, protocol, store_dir, name, key_params, load_kwargs,
                 cells_by_session=None, base_seed=0, n_jobs=2, recompute=False, verbose=5,
                 extra_key_by_session=None, flatten=True):
    """Run `process_fn(ses, seed) -> list[dict]` once per session and return the concatenated
    records. `key_params` must hold EVERY setting that changes the result (they are hashed into
    each per-session cache file name, together with the session id, seed and the session's included
    cell ids).

    protocol             : one protocol name or a list of protocols to search for the session ids
    extra_key_by_session : optional {session_id: str} added to that session's cache key (e.g.
                           scope.data_stamp(...) so changed raw files invalidate the cache)
    flatten              : True  -> process_fn returns a list of records, results are concatenated
                           False -> process_fn may return any picklable object; a list with one
                                    entry per session is returned (in session_ids order)"""
    protocols = [protocol] if isinstance(protocol, str) else list(protocol)
    paths, todo = {}, []
    # seed from the SESSION ID (not its position in the list), so a restricted run reuses the caches
    # and reproduces the exact random draws of a full run
    seeds = {sid: base_seed + zlib.crc32(sid.encode()) % 100000 for sid in session_ids}
    for sid in session_ids:
        cells = sorted(cells_by_session.get(sid, [])) if cells_by_session else []
        cell_hash = hashlib.sha1(",".join(map(str, cells)).encode()).hexdigest()[:10]
        extra = {"extra": extra_key_by_session[sid]} if extra_key_by_session and sid in extra_key_by_session else {}
        paths[sid] = cache_path(store_dir, name, session_id=sid, seed=seeds[sid], cells=cell_hash,
                                **extra, **key_params)
        if recompute or not paths[sid].exists():
            todo.append(sid)
    print(f"{len(session_ids) - len(todo)}/{len(session_ids)} sessions cached; computing {len(todo)} "
          f"(n_jobs={n_jobs}, one session in RAM per worker)")

    def _one(sid):
        from infotheory.session import load_sessions
        loaded = load_sessions(protocols=protocols, only_session_ids=[sid], verbose=False, **load_kwargs)
        recs = process_fn(loaded[0], seeds[sid]) if loaded else ([] if flatten else None)
        del loaded
        gc.collect()
        pd.to_pickle(recs, paths[sid])
        return len(recs) if recs is not None else 0

    if todo:
        Parallel(n_jobs=n_jobs, backend="loky", verbose=verbose)(delayed(_one)(sid) for sid in todo)
    out = []
    for sid in session_ids:
        res = pd.read_pickle(paths[sid])
        if flatten:
            out.extend(res)
        else:
            out.append(res)
    return out


# =========================================================================================================
# CAPS ON CELLS / PAIRS (per group) with a per-session cache of the final tables (no ledger, nothing per unit stored)
# =========================================================================================================
ENGINE_VERSION = "units-v2"


def _rank_key(sid, *unit):
    return zlib.crc32("|".join([str(sid), *map(str, unit)]).encode())


def rank_cells(sid, cells):
    """Deterministic pseudo-random order of a session's cells (hash of session id + cell id, so it does not depend on which
    other cells are present). A cap of N means "the first N of this order": the first 50 are always inside the first 100,
    which are inside the first 300 -- the property that makes caps nest and results reusable."""
    return sorted(cells, key=lambda c: (_rank_key(sid, c), str(c)))


def canonical_pair(a, b):
    return (a, b) if a <= b else (b, a)


def requested_units(sid, cells, max_cells=None, need_pairs=False, max_pairs=None, groups=None):
    """(requested cells in rank order, requested pairs). Pairs are all pairs among the requested cells.

    groups = {cell: group label} (optional but what the group analyses need): the caps then apply PER GROUP -- max_cells = the
    first N cells of each group in this session, max_pairs = the first N pairs of each pair of groups -- so a rare group (e.g.
    labeled cells) is not emptied by a cap that is dominated by the abundant ones. Without groups the cap applies to the session
    as a whole (the first N cells in the hash order, the first max_pairs pairs in the hash order of the pair itself)."""
    ranked = rank_cells(sid, cells)
    if max_cells and groups:
        n_in_group, req_cells = {}, []
        for c in ranked:
            g = groups.get(c, "")
            n_in_group[g] = n_in_group.get(g, 0) + 1
            if n_in_group[g] <= int(max_cells):
                req_cells.append(c)
    else:
        req_cells = ranked[:int(max_cells)] if max_cells else ranked
    req_pairs = []
    if need_pairs:
        req_pairs = [canonical_pair(a, b) for a, b in itertools.combinations(req_cells, 2)]
        if max_pairs:
            order = sorted(req_pairs, key=lambda q: (_rank_key(sid, *q), str(q)))
            if groups:
                n_in_pair, keep = {}, []
                for q in order:
                    k = tuple(sorted((groups.get(q[0], ""), groups.get(q[1], ""))))
                    n_in_pair[k] = n_in_pair.get(k, 0) + 1
                    if n_in_pair[k] <= int(max_pairs):
                        keep.append(q)
                req_pairs = keep
            else:
                req_pairs = order[:int(max_pairs)]
    return req_cells, req_pairs


def run_sessions_units(session_ids, process_fn, protocol, store_dir, name, key_params, load_kwargs, cells_by_session,
                       parts, extras=None, unwrap=None, max_cells=None, max_pairs=None, base_seed=0, n_jobs=2,
                       recompute=False, verbose=5, extra_key_by_session=None, groups_by_cell=None):
    """Like run_sessions(flatten=False) plus the cell / pair CAPS, with NO ledger: the cache of a session holds only the final
    tables `process_fn` returns (one file per session and setting), nothing per unit is kept beyond those tables.

    * Which units: the session's cells (`cells_by_session`, e.g. the included ones) are put in a fixed pseudo-random order
      (rank_cells); `max_cells` = N keeps the first N -- of each GROUP when `groups_by_cell` ({cell_id: group}) is given (always
      pass it for group analyses: a cap over the whole session leaves rare groups, e.g. labeled cells, with a handful of
      cells). Pairs (only if some part is level "pair") are all pairs among those cells, or the first `max_pairs` of them
      (per pair of groups with `groups_by_cell`) in a nested hash order.
    * `process_fn(ses, seed, req_cells, req_pairs, req_cells) -> dict` is called ONCE per session with every requested unit
      (the first two unit arguments keep the names new_cells / new_pairs of the former ledger engine) and returns
      {part name: DataFrame, other small session-level facts}. Pair-level parts should be REDUCED inside the worker (sums per
      group pair, histograms) -- a table with a row per pair is what made the ledger huge. None / empty = nothing for the session.
    * Cache key: session, seed, data stamp, `key_params` AND the requested cells (and pairs when capped). A changed cap or cell
      list therefore recomputes the session (no reuse of earlier cells); an unchanged one reads the file.
    * `parts`: {part name: "cell" | "pair"}; the level only tells whether pairs are needed. `extras` is accepted for
      compatibility. `unwrap`: name of a part to return instead of the whole payload."""
    protocols = [protocol] if isinstance(protocol, str) else list(protocol)
    need_pairs = any(v == "pair" for v in parts.values())
    seeds = {sid: base_seed + zlib.crc32(sid.encode()) % 100000 for sid in session_ids}
    plan, paths = {}, {}
    n_cached = 0
    for sid in session_ids:
        req_cells, req_pairs = requested_units(sid, cells_by_session.get(sid, []), max_cells, need_pairs, max_pairs,
                                               groups=groups_by_cell)
        sel = hashlib.sha1(repr((sorted(map(str, req_cells)), sorted(map(str, req_pairs)) if max_pairs else None)).encode()
                           ).hexdigest()[:12]
        extra = {"extra": extra_key_by_session[sid]} if extra_key_by_session and sid in extra_key_by_session else {}
        paths[sid] = cache_path(store_dir, name, session_id=sid, seed=seeds[sid], **extra, **key_params, sel=sel,
                                engine=ENGINE_VERSION)
        plan[sid] = dict(req_cells=req_cells, req_pairs=req_pairs)
        n_cached += int(paths[sid].exists() and not recompute)
    todo = [sid for sid in session_ids if recompute or not paths[sid].exists()]
    cap = []
    if max_cells:
        cap.append(f"max {max_cells} cells{' per group' if groups_by_cell else ''}")
    if need_pairs and max_pairs:
        cap.append(f"max {max_pairs} pairs{' per pair of groups' if groups_by_cell else ''}")
    print(f"{n_cached}/{len(session_ids)} sessions cached{' (' + ', '.join(cap) + ')' if cap else ''}; computing {len(todo)} "
          f"(n_jobs={n_jobs}, one session in RAM per worker)")

    def _one(sid):
        from infotheory.session import load_sessions
        pl = plan[sid]
        loaded = load_sessions(protocols=protocols, only_session_ids=[sid], verbose=False, **load_kwargs)
        payload = process_fn(loaded[0], seeds[sid], pl["req_cells"], pl["req_pairs"], pl["req_cells"]) if loaded else None
        del loaded
        gc.collect()
        tmp = str(paths[sid]) + f".tmp{os.getpid()}"
        pd.to_pickle(payload or {}, tmp)
        os.replace(tmp, paths[sid])
        return 1

    if todo:
        Parallel(n_jobs=n_jobs, backend="loky", verbose=verbose)(delayed(_one)(sid) for sid in todo)
    out = []
    for sid in session_ids:
        payload = pd.read_pickle(paths[sid])
        empty = not payload or all(v is None or (hasattr(v, "__len__") and len(v) == 0) for v in payload.values())
        if unwrap is not None:
            out.append(None if empty else payload.get(unwrap))
        else:
            out.append(None if empty else payload)
    return out
