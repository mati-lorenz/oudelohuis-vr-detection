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
