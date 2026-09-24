# -*- coding: utf-8 -*-
"""
cache.py
=========
General-purpose, parameter-keyed disk caching -- one cache file per
(name, parameter combination), so that:
    1. changing a parameter (e.g. fewer shuffles for a quick look)
       creates a NEW cache file rather than silently reusing or
       clobbering the old one -- old cache files are just left on disk,
       unused, until you delete them yourself;
    2. running the SAME script again later (e.g. to add a session, or
       to redo one session at heavier settings) only needs to
       (re)compute whatever's actually missing/changed -- everything
       else is loaded straight from disk.

This is the piece 3d/3e/3f_pair_*_temporal.py were missing: they cached
ONE combined result for ALL sessions together, so there was no way to
add a session or change a setting without recomputing everything.
Caching PER SESSION (see those scripts after this fix) is what makes
"run again later with different settings, results accumulate" actually
possible.

Matches the real project's own cache.py (cache_path/load_or_compute)
referenced in the uploaded plot_temporal_*.py reference scripts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


def cache_path(cache_dir, name: str, **key_params) -> Path:
    """Build a cache file path from `name` plus a dict of parameters
    that affect the result -- the params are hashed into the filename,
    so two calls with different `key_params` never collide, and calling
    again with the SAME params always resolves to the SAME file (so a
    later run can find and reuse it).

    Include EVERY parameter that affects the result: shuffle counts,
    bin sizes, pair caps, random seed, and anything upstream whose
    change should invalidate the cache (e.g. an upstream QC file's
    mtime, so re-running 2b_activity_statistics.py and changing which
    cells pass automatically busts any cache that depended on it --
    see the `qc_mtime` pattern in the reference scripts).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key_str = json.dumps(key_params, sort_keys=True, default=str)
    key_hash = hashlib.sha1(key_str.encode()).hexdigest()[:12]
    return cache_dir / f"{name}_{key_hash}.pkl"


def load_or_compute(path: Path, compute_fn, force_recompute: bool = False):
    """Load `path` if it exists (and `force_recompute` is False),
    otherwise call `compute_fn()`, cache its result to `path`, and
    return it. `compute_fn` takes no arguments -- close over whatever
    it needs (see 3d/3e/3f's `_compute()` closures)."""
    path = Path(path)
    if not force_recompute and path.exists():
        try:
            return pd.read_pickle(path)
        except Exception as e:
            print(f"Could not load cache {path} ({e}); recomputing.")
    result = compute_fn()
    pd.to_pickle(result, path)
    return result


def load_all_matching(cache_dir, name_prefix: str) -> list:
    """Load every cache file in `cache_dir` whose name starts with
    `name_prefix` -- used to gather whatever per-session results
    already exist (at the CURRENT parameter settings, since the hash is
    part of the filename) without needing to know in advance exactly
    which sessions were previously computed. Returns a list of
    whatever objects were cached (e.g. a list of per-session dicts)."""
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return []
    results = []
    for p in sorted(cache_dir.glob(f"{name_prefix}_*.pkl")):
        try:
            results.append(pd.read_pickle(p))
        except Exception as e:
            print(f"Skipping unreadable cache file {p} ({e})")
    return results
