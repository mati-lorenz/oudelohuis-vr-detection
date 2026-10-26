# -*- coding: utf-8 -*-
"""
pair_distance.py
================
Distance between the two cells of a pair, and distance-matched controls. Nearby cells share input and are more
correlated, so any difference in pair correlation between groups of pairs (e.g. labelled-labelled vs unlabelled-
unlabelled pairs of the same area) can be a pure difference in the pairs' distances. The control: keep only pairs whose
distance distributions agree.

DISTANCE   Euclidean distance in microns from `xloc`, `yloc` (position in the field of view) and `depth` (microns below
           the surface; cells of different imaging planes differ in depth). Coordinates are only comparable WITHIN a
           session, so distances are computed, and matching is done, within a session (and area).
MATCHING   `match_distances`: greedy nearest-neighbour matching WITHOUT replacement of the pairs of a target set (e.g.
           labelled pairs) to a pool (e.g. unlabelled pairs), taken in random order, accepted when the distances differ
           by at most `caliper` microns; unmatched target pairs are dropped. The matched pool pairs and the kept
           target pairs then have nearly the same distance distribution; check it with `matching_summary`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

POSITION_COLUMNS = ("xloc", "yloc", "depth")


def pair_distances(celldata: pd.DataFrame, cell_i, cell_j) -> np.ndarray:
    """Distances (microns) between cells `cell_i[k]` and `cell_j[k]` of one session's celldata (index = rows with a
    `cell_id` column). NaN where a coordinate is missing; depth is ignored when the column is absent."""
    cols = [c for c in POSITION_COLUMNS if c in celldata.columns]
    if "xloc" not in cols or "yloc" not in cols:
        return np.full(len(cell_i), np.nan)
    pos = celldata.drop_duplicates("cell_id").set_index("cell_id")[cols].astype(float)
    a = pos.reindex(list(cell_i)).to_numpy()
    b = pos.reindex(list(cell_j)).to_numpy()
    return np.sqrt(((a - b) ** 2).sum(axis=1))


def match_distances(dist_target, dist_pool, caliper: float, rng: np.random.Generator):
    """(kept_target_idx, matched_pool_idx), equal-length index arrays: each kept target pair with the nearest unused pool
    pair within `caliper` microns. Target pairs are visited in random order; pairs with NaN distance never match."""
    dt = np.asarray(dist_target, float)
    dp = np.asarray(dist_pool, float)
    free = np.isfinite(dp)
    kept, matched = [], []
    for i in rng.permutation(len(dt)):
        if not np.isfinite(dt[i]) or not free.any():
            continue
        cand = np.flatnonzero(free)
        j = cand[np.argmin(np.abs(dp[cand] - dt[i]))]
        if abs(dp[j] - dt[i]) <= caliper:
            kept.append(i)
            matched.append(j)
            free[j] = False
    return np.asarray(kept, int), np.asarray(matched, int)


def matching_summary(dist_target, dist_pool, kept, matched) -> dict:
    """Median distance of the target set / pool before and after matching (microns) and the number of matched pairs."""
    dt, dp = np.asarray(dist_target, float), np.asarray(dist_pool, float)
    return dict(n_target=len(dt), n_pool=len(dp), n_matched=len(kept),
                median_target=float(np.nanmedian(dt)) if len(dt) else np.nan,
                median_pool_before=float(np.nanmedian(dp)) if len(dp) else np.nan,
                median_target_kept=float(np.nanmedian(dt[kept])) if len(kept) else np.nan,
                median_pool_after=float(np.nanmedian(dp[matched])) if len(kept) else np.nan)


def flag_distance_matched(df: pd.DataFrame, caliper: float, seed_fn, target_label: str = "lab",
                          pool_label: str = "unl", by=("session_id", "roi_name")) -> pd.Series:
    """Boolean column `dm_matched` aligned with `df` (needs distance_um and labeled_grouped). Within every `by` block the
    `target_label` pairs are matched to `pool_label` pairs; matched pairs of both sets are True. Blocks without both
    labels (e.g. AL, RSP) keep every pair (True) because there is nothing to match. `seed_fn(block_key) -> int`."""
    flag = pd.Series(False, index=df.index)
    for key, blk in df.groupby(list(by), observed=True):
        t = blk[blk["labeled_grouped"] == target_label]
        p = blk[blk["labeled_grouped"] == pool_label]
        if t.empty or p.empty:
            flag.loc[blk.index] = True
            continue
        rng = np.random.default_rng(seed_fn(key))
        k, m = match_distances(t["distance_um"].to_numpy(), p["distance_um"].to_numpy(), caliper, rng)
        flag.loc[t.index[k]] = True
        flag.loc[p.index[m]] = True
    return flag
