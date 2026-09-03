# -*- coding: utf-8 -*-
"""
cache.py
========
Small, dependency-free disk cache for expensive analyses (e.g. time-resolved pairwise breakdown/PID), keyed by a hash of the parameters that affect the result -- so a parameter change automatically busts the cache, without ever deleting files by hand.

TODO(you):
- cache_path(cache_dir, name, **key_params) -> path (hash key_params, not the result)
- load_cache(path) -> object or None (treat unpickle errors as a miss, not a crash)
- save_cache(path, obj) -> write atomically (write to .tmp then os.replace)
- load_or_compute(path, compute_fn, force_recompute=False) -> convenience wrapper
"""

import numpy as np  # TODO: trim/add imports as needed



def _stable_hash(payload_dict):
    """TODO: json.dumps(payload_dict, sort_keys=True, default=str) then hash it
    (order-independent, stable, short)."""
    raise NotImplementedError


def cache_path(cache_dir, name, **key_params):
    """TODO: os.makedirs(cache_dir, exist_ok=True); hash key_params via
    _stable_hash; return os.path.join(cache_dir, f"{name}_{hash}.pkl")."""
    raise NotImplementedError


def load_cache(path):
    """TODO: return the unpickled object, or None if missing/corrupted
    (treat unpickle errors as a cache MISS, not a crash)."""
    raise NotImplementedError


def save_cache(path, obj):
    """TODO: pickle.dump to a .tmp path, then os.replace (atomic on POSIX,
    avoids partial files if interrupted mid-write)."""
    raise NotImplementedError


def load_or_compute(path, compute_fn, force_recompute=False, verbose=True):
    """TODO: load_cache(path) unless force_recompute; else call compute_fn(),
    save_cache(path, result), return result."""
    raise NotImplementedError
