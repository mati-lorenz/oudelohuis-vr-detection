# -*- coding: utf-8 -*-
"""
cache.py
=========
Small, dependency-free disk cache used by the more expensive analyses in
this pipeline (currently: the time-resolved pairwise information
breakdown, infotheory/temporal_breakdown.py), so that plotting scripts
can be re-run to tweak aggregation/plotting choices without recomputing
information estimates from scratch.

Design: a cache entry is identified by a "name" (e.g. session_id, or
session_id + analysis name) plus a dict of the parameters that affect the
result (protocol window, n_bins, pairs used, random_state, ...). The
parameter dict is hashed into a short, stable, order-independent key; the
cache file itself is a pickle containing whatever object you want
(typically a dict of {'df': DataFrame, 'axis': ..., 'meta': ...}).

Usage
-----
    key_params = dict(session_id=ses.session_id, n_bins=3, s_pre=-60, ...)
    path = cache_path(cache_dir, 'temporal_breakdown', **key_params)

    cached = load_cache(path)
    if cached is not None:
        result = cached
    else:
        result = expensive_computation(...)
        save_cache(path, result)
"""

import os
import json
import pickle
import hashlib


def _stable_hash(payload_dict):
    """Order-independent, stable short hash of a dict of simple
    (JSON-serializable) values."""
    payload = json.dumps(payload_dict, sort_keys=True, default=str)
    return hashlib.md5(payload.encode('utf-8')).hexdigest()[:12]


def cache_path(cache_dir, name, **key_params):
    """
    Build a cache file path from a human-readable `name` plus a hash of
    `key_params`. Two calls with the same name and an equal (order-
    independent) set of key_params always resolve to the same path; any
    change to key_params (e.g. a different n_bins, a different set of
    pairs, a different response window) produces a different path, so
    stale results are never silently reused.
    """
    os.makedirs(cache_dir, exist_ok=True)
    h = _stable_hash(key_params)
    safe_name = str(name).replace(os.sep, '_')
    return os.path.join(cache_dir, f'{safe_name}_{h}.pkl')


def load_cache(path):
    """Return the cached object, or None if no cache file exists yet (or
    it can't be unpickled, e.g. after a code change to the cached
    object's structure -- in that case treat it as a cache miss)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f'[cache] could not load {path} ({e}); treating as cache miss')
        return None


def save_cache(path, obj):
    tmp_path = path + '.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump(obj, f)
    os.replace(tmp_path, path)   # atomic on POSIX, avoids partial files


def load_or_compute(path, compute_fn, force_recompute=False, verbose=True):
    """
    Convenience wrapper: load `path` if it exists (and not
    force_recompute), otherwise call `compute_fn()`, cache the result,
    and return it.
    """
    if not force_recompute:
        cached = load_cache(path)
        if cached is not None:
            if verbose:
                print(f'[cache] loaded {os.path.basename(path)}')
            return cached
    if verbose:
        print(f'[cache] computing {os.path.basename(path)} ...')
    result = compute_fn()
    save_cache(path, result)
    return result
