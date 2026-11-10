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
# INCREMENTAL, UNIT-LEVEL CACHE ("ledger"): a cap on cells / pairs that can be changed freely between runs
# =========================================================================================================
LEDGER_VERSION = "ledger-v1"
CELL_COL = "cell_id"
PAIR_COLS = ("cell_i", "cell_j")


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


def _ledger_path(store_dir, name, sid, seed, extra, key_params):
    return cache_path(store_dir, name + "__ledger", session_id=sid, seed=seed, **extra, **key_params, engine=LEDGER_VERSION)


def _empty_ledger():
    return dict(done_cells=set(), done_pairs=set(), parts={}, meta={})


def _read_ledger(path, recompute):
    if recompute or not path.exists():
        return _empty_ledger()
    return pd.read_pickle(path)


def _filter_part(df, level, req_cells_set, req_pairs_set):
    if df is None or len(df) == 0:
        return pd.DataFrame() if df is None else df
    if level == "cell":
        return df[df[CELL_COL].isin(req_cells_set)].reset_index(drop=True)
    keys = pd.MultiIndex.from_frame(df[list(PAIR_COLS)])
    return df[keys.isin(req_pairs_set)].reset_index(drop=True)


def run_sessions_units(session_ids, process_fn, protocol, store_dir, name, key_params, load_kwargs, cells_by_session,
                       parts, extras=None, unwrap=None, max_cells=None, max_pairs=None, base_seed=0, n_jobs=2,
                       recompute=False, verbose=5, extra_key_by_session=None, groups_by_cell=None):
    """Like run_sessions(flatten=False), but the cache holds RESULTS PER UNIT (cell, or cell pair) and the set of units that
    is analysed can be changed freely between runs.

    * Which units: the session's cells (`cells_by_session`, e.g. the included ones) are put in a fixed pseudo-random order
      (rank_cells); `max_cells` = N keeps the first N -- of each GROUP when `groups_by_cell` ({cell_id: group}) is given (always
      pass it for group analyses: a cap over the whole session leaves rare groups, e.g. labeled cells, with a handful of
      cells). Pairs (only if some part is level "pair") are all pairs among those cells, or the first `max_pairs` of them
      (per pair of groups with `groups_by_cell`) in a nested hash order.
    * The cache file of a session does NOT depend on the cell list or on the cap -- only on the session, seed, data stamp and
      `key_params`. It stores every unit ever computed. A run computes only the requested units that are missing (N=100, then
      300: the 200 new ones; then 50: nothing, the stored rows of the first 50 are read) and returns the requested units' rows.
    * `process_fn(ses, seed, new_cells, new_pairs, req_cells) -> dict` computes ONLY `new_cells` (cell-level parts) and
      `new_pairs` (pair-level parts; `req_cells` is every requested cell, to pair against) and returns {part name: DataFrame};
      cell-level DataFrames need a `cell_id` column, pair-level ones `cell_i`, `cell_j` (canonical_pair order). Returning None
      is fine (the units are then marked done). Any other key of the dict (small session-level facts: trials, tail, checks)
      goes to the ledger's meta: last value wins, or lists are concatenated when `extras[key] == "extend"`.
    * `parts`: {part name: "cell" | "pair"}. `unwrap`: name of a part to return instead of the whole payload (a DataFrame
      or None), for scripts whose per-session result is one table.
    Results of a unit never change once stored, whatever else is computed later (changing a setting in `key_params` starts a
    new ledger). The random draws of a unit do depend on which units were computed in the same call (they come from one
    session-level generator), which is statistically irrelevant but means a one-shot run of N cells and an incremental run
    reaching N cells are not bit-identical."""
    protocols = [protocol] if isinstance(protocol, str) else list(protocol)
    extras = extras or {}
    need_pairs = any(v == "pair" for v in parts.values())
    seeds = {sid: base_seed + zlib.crc32(sid.encode()) % 100000 for sid in session_ids}
    plan, paths = {}, {}
    n_cached, new_c, new_p = 0, 0, 0
    for sid in session_ids:
        extra = {"extra": extra_key_by_session[sid]} if extra_key_by_session and sid in extra_key_by_session else {}
        paths[sid] = _ledger_path(store_dir, name, sid, seeds[sid], extra, key_params)
        led = _read_ledger(paths[sid], recompute)
        req_cells, req_pairs = requested_units(sid, cells_by_session.get(sid, []), max_cells, need_pairs, max_pairs,
                                               groups=groups_by_cell)
        nc = [c for c in req_cells if c not in led["done_cells"]]
        npairs = [q for q in req_pairs if q not in led["done_pairs"]]
        plan[sid] = dict(req_cells=req_cells, req_pairs=req_pairs, new_cells=nc, new_pairs=npairs)
        n_cached += int(not nc and not npairs)
        new_c += len(nc)
        new_p += len(npairs)
    todo = [sid for sid in session_ids if plan[sid]["new_cells"] or plan[sid]["new_pairs"]]
    cap = []
    if max_cells:
        cap.append(f"max {max_cells} cells{' per group' if groups_by_cell else ''}")
    if need_pairs and max_pairs:
        cap.append(f"max {max_pairs} pairs{' per pair of groups' if groups_by_cell else ''}")
    print(f"{n_cached}/{len(session_ids)} sessions fully cached"
          f"{' (' + ', '.join(cap) + ')' if cap else ''}; computing "
          f"{', '.join(x for x in ([f'{new_c} new cells'] if any(v == 'cell' for v in parts.values()) else []) + ([f'{new_p} new pairs'] if need_pairs else []))}"
          f" in {len(todo)} sessions (n_jobs={n_jobs})")

    def _one(sid):
        from infotheory.session import load_sessions
        pl = plan[sid]
        led = _read_ledger(paths[sid], recompute)
        loaded = load_sessions(protocols=protocols, only_session_ids=[sid], verbose=False, **load_kwargs)
        payload = process_fn(loaded[0], seeds[sid], pl["new_cells"], pl["new_pairs"], pl["req_cells"]) if loaded else None
        del loaded
        gc.collect()
        payload = payload or {}
        for key, level in parts.items():
            df = payload.get(key)
            if df is not None and len(df):
                old = led["parts"].get(key)
                led["parts"][key] = df if old is None or len(old) == 0 else pd.concat([old, df], ignore_index=True)
        for key, val in payload.items():
            if key in parts:
                continue
            if extras.get(key) == "extend":
                led["meta"][key] = list(led["meta"].get(key, [])) + list(val or [])
            else:
                led["meta"][key] = val
        led["done_cells"].update(pl["new_cells"])
        led["done_pairs"].update(pl["new_pairs"])
        tmp = str(paths[sid]) + f".tmp{os.getpid()}"
        pd.to_pickle(led, tmp)
        os.replace(tmp, paths[sid])
        return len(pl["new_cells"]) + len(pl["new_pairs"])

    if todo:
        Parallel(n_jobs=n_jobs, backend="loky", verbose=verbose)(delayed(_one)(sid) for sid in todo)
    out = []
    for sid in session_ids:
        led = _read_ledger(paths[sid], False)
        pl = plan[sid]
        rc, rp = set(pl["req_cells"]), set(pl["req_pairs"])
        payload = {key: _filter_part(led["parts"].get(key), level, rc, rp) for key, level in parts.items()}
        payload.update(led["meta"])
        empty = all(v is None or len(v) == 0 for k, v in payload.items() if k in parts) and not led["meta"]
        if unwrap is not None:
            out.append(None if empty else payload.get(unwrap))
        else:
            out.append(None if empty else payload)
    return out

