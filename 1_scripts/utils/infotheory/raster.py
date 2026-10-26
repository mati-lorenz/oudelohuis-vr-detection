# -*- coding: utf-8 -*-
"""
raster.py
==========
Trial-averaged, position-aligned activity "raster" (cells x position
bins relative to stimulus onset) with a PLUGGABLE sorting rule.

Pipeline
--------
1. per session: trial x cell x bin tensor (tensors.build_position_tensor)
2. `half_profiles`: trial-average -> (cells x bins) profiles for ALL
   trials and for the odd / even halves (this is what gets cached)
3. `zscore_profiles`: z-score each cell's profile across bins
4. `sort_order`: order cells by a sorter from SORTERS, applied to the
   SORT profiles
5. `build_raster`: display profiles in that order

Cross-validated sorting (default): cells are sorted on one half of the
trials (odd) and DISPLAYED with the other half (even). Sorting and
displaying the same data always yields a bright diagonal, even for pure
noise (selection bias); cross-validation keeps the diagonal only if the
tuning is reproducible across trials. Set cv=False for the classic
(biased) version.

Changing the sort
-----------------
Sorting happens at plotting time on cached profiles, so changing
SORT_BY in 2b_activity_statistics.py needs no recompute. To add a new
sorter, write a function `f(P, centers) -> 1D key array (NaN sorts
last)` and register it in SORTERS.
"""
from __future__ import annotations

import warnings

import numpy as np

# all-NaN bins/cells are expected (poorly sampled positions) and handled explicitly
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom")
warnings.filterwarnings("ignore", message="All-NaN")


# --------------------------------------------------------------------- #
# Profiles
# --------------------------------------------------------------------- #
def trial_average(tensor, trial_idx=None, min_trials: int = 10):
    """NaN-aware trial average of a (K, N, B) tensor over `trial_idx`
    (default all). Bins covered by fewer than `min_trials` trials are NaN.
    Returns (profile (N, B), n_trials_per_bin (B,))."""
    T = tensor if trial_idx is None else tensor[np.asarray(trial_idx)]
    n_per_bin = np.isfinite(T).any(axis=1).sum(0)
    with np.errstate(invalid="ignore"):
        prof = np.nanmean(T, axis=0)
    prof[:, n_per_bin < min_trials] = np.nan
    return prof, n_per_bin


def half_profiles(tensor, min_trials: int = 10) -> dict:
    """Profiles from all / odd / even trials (trial order = row order of
    the tensor, i.e. chronological). Odd/even is robust to slow drifts."""
    K = tensor.shape[0]
    idx = np.arange(K)
    out = {}
    for name, sel in (("all", idx), ("odd", idx[0::2]), ("even", idx[1::2])):
        prof, n = trial_average(tensor, sel, min_trials=max(2, min_trials // (1 if name == "all" else 2)))
        out[name] = prof
        out[f"n_{name}"] = n
    return out


def zscore_profiles(P, mode: str = "profile"):
    """Normalize each cell's profile across bins (NaN-aware).
    mode="profile": z-score (mean 0, SD 1)
    mode="minmax" : rescale to [0, 1] (the "normalized dR/R" of Chen et al.
                    2013, Supp. Fig. 6b)
    mode="none"/"frames": unchanged ("frames" z-scoring is done upstream
                    with tensors.zscore_frames)."""
    P = np.asarray(P, float)
    if mode in ("none", "frames"):
        return P
    if mode == "minmax":
        lo = np.nanmin(P, axis=1, keepdims=True)
        hi = np.nanmax(P, axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(hi - lo > 1e-12, (P - lo) / (hi - lo), np.nan)
    m = np.nanmean(P, axis=1, keepdims=True)
    s = np.nanstd(P, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s > 1e-12, (P - m) / s, np.nan)


# --------------------------------------------------------------------- #
# Sorters: f(P (N,B), centers (B,)) -> key (N,) ; NaN keys go last
# --------------------------------------------------------------------- #
def _peak_position(P, centers):
    allnan = np.isnan(P).all(1)
    idx = np.nanargmax(np.where(np.isnan(P), -np.inf, P), axis=1)
    key = centers[idx].astype(float)
    key[allnan] = np.nan
    return key


def _trough_position(P, centers):
    return _peak_position(-P, centers)


def _center_of_mass(P, centers):
    W = np.clip(P - np.nanmin(P, axis=1, keepdims=True), 0, None)
    W = np.where(np.isnan(P), 0.0, W)
    tot = W.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        com = (W * centers[None, :]).sum(1) / tot
    return np.where(tot > 0, com, np.nan)


def _none(P, centers):
    return np.arange(len(P), dtype=float)


SORTERS = {
    "peak_position": _peak_position,       # position of maximum z-activity (default)
    "trough_position": _trough_position,   # position of minimum
    "center_of_mass": _center_of_mass,     # activity-weighted mean position
    "none": _none,                         # keep input order
}


def sort_order(P_sort, centers, method: str = "peak_position"):
    """Indices that order cells by the chosen sorter (NaN keys last).
    Returns (order, key) with key in ORIGINAL cell order."""
    if method not in SORTERS:
        raise KeyError(f"unknown sorter {method!r}; available: {sorted(SORTERS)}")
    key = SORTERS[method](np.asarray(P_sort, float), np.asarray(centers, float))
    order = np.argsort(np.where(np.isnan(key), np.inf, key), kind="stable")
    return order, key


def reliability(P_a, P_b):
    """Per-cell Pearson correlation of two profile sets over bins
    (odd vs even trials): 1 = tuning fully reproducible, ~0 = noise."""
    a = P_a - np.nanmean(P_a, 1, keepdims=True)
    b = P_b - np.nanmean(P_b, 1, keepdims=True)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = np.where(ok, a, 0.0), np.where(ok, b, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))


def build_raster(profiles_by_half: dict, centers, method: str = "peak_position", cv: bool = True,
                 zscore: str = "profile"):
    """Sorted, z-scored display matrix from the cached half-profiles of
    one set of cells (one session, or several sessions stacked).

    Returns dict: display (N,B) in sorted order, order (N,) indices into
    the input, sort_key (N,) in INPUT order, reliability (N,), valid (N,)
    (cells with a usable profile), n_cells."""
    P_sort = zscore_profiles(profiles_by_half["odd" if cv else "all"], zscore)
    P_disp = zscore_profiles(profiles_by_half["even" if cv else "all"], zscore)
    rel = reliability(profiles_by_half["odd"], profiles_by_half["even"])
    order, key = sort_order(P_sort, centers, method)
    valid = np.isfinite(P_sort).any(1) & np.isfinite(P_disp).any(1)
    order = order[valid[order]]
    return dict(display=P_disp[order], order=order, sort_key=key, reliability=rel,
                valid=valid, n_cells=int(valid.sum()))
