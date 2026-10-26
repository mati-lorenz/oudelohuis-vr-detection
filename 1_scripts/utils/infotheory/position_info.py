# -*- coding: utf-8 -*-
"""
position_info.py
================
Position-resolved single-cell information (2h): the frame bookkeeping and the per-bin MI shared by the two aggregation
methods, so both use exactly the same frames, position bins and trial labels.

FRAME INDEX   `frame_bin_index`: for every engaged trial, the calcium frames whose position lies within
              [stimStart + s_pre, stimStart + s_post], each with its trial row and position bin (relative to stimulus onset).
TRIAL MEAN    `trial_bin_means`: per (trial, bin) the mean over the trial's frames in that bin -> one sample per trial.
FRAME POOLED  the frames of ALL trials in a bin are used as samples; a trial-level variable (stimulus bin, choice) is
              attached to each frame from its trial. Frames of one trial are not independent -- the MI is a fair descriptive
              curve, the shuffle p-value is optimistic (shuffling frames breaks the within-trial dependence).
MI            `bin_information`: histogram MI with RESPONSE_BINS quantile bins, shuffle-corrected, shuffle p-value -- the
              estimator of 2c (infotheory.info_theory.mutual_information_shuffle). A discrete variable's classes with fewer
              than `min_per_class` trials are dropped; at least `min_trials` distinct trials and 2 classes must remain.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import project_config as cfg
from .info_theory import mutual_information_shuffle


def position_bins(s_pre: float, s_post: float, binsize: float):
    edges = np.arange(s_pre, s_post + binsize, binsize)
    return edges, (edges[:-1] + edges[1:]) / 2


def frame_bin_index(trial_start_pos, pos, s_pre: float, s_post: float, binsize: float):
    """(frame_idx, trial_row, bin, bincenters): positional frame indices into the frame table, the index of the trial
    (row of `trial_start_pos`) and the position bin of every frame inside any trial's window. `pos` = the frame table's
    position trace (non-decreasing)."""
    edges, centers = position_bins(s_pre, s_post, binsize)
    fi, tr, bb = [], [], []
    for t, z0 in enumerate(np.asarray(trial_start_pos, float)):
        lo, hi = np.searchsorted(pos, [z0 + s_pre, z0 + s_post])
        if hi <= lo:
            continue
        b = np.digitize(pos[lo:hi] - z0, edges) - 1
        ok = (b >= 0) & (b < len(centers))
        if ok.any():
            fi.append(np.arange(lo, hi)[ok])
            tr.append(np.full(int(ok.sum()), t))
            bb.append(b[ok])
    if not fi:
        e = np.array([], int)
        return e, e, e, centers
    return np.concatenate(fi), np.concatenate(tr), np.concatenate(bb), centers


def trial_bin_means(values: np.ndarray, fi, tr, bb, n_trials: int, n_bins: int) -> np.ndarray:
    """(n_trials, n_bins, k) NaN-aware mean of the k columns of `values` (frames x k) over each trial's frames in each
    bin; NaN where the trial has no frame in the bin."""
    out = np.full((n_trials, n_bins, values.shape[1]), np.nan)
    if len(fi) == 0:
        return out
    g = pd.DataFrame(values[fi]).groupby(tr * n_bins + bb).mean()
    idx = g.index.to_numpy()
    out[idx // n_bins, idx % n_bins, :] = g.to_numpy()
    return out


def bin_information(x, y, trials, discrete: bool, n_shuffles: int, rng, bins: int = cfg.RESPONSE_BINS,
                    min_samples: int = cfg.MIN_TRIALS_ANALYSIS, min_trials: int = cfg.MIN_TRIALS_ANALYSIS,
                    min_per_class: int = cfg.MIN_TRIALS_PER_TARGET_BIN):
    """(n_samples, n_trials, mi_bits_corrected, p_value) or None. x = variable, y = cell activity, `trials` = the trial
    of each sample (for trial means: one sample per trial)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if discrete and ok.any():
        for c in np.unique(x[ok]):
            if len(np.unique(trials[ok & (x == c)])) < min_per_class:
                ok &= ~(x == c)
    if ok.sum() < min_samples or len(np.unique(trials[ok])) < min_trials or (discrete and len(np.unique(x[ok])) < 2):
        return None
    res = mutual_information_shuffle(x[ok], y[ok], bins=bins, n_shuffles=n_shuffles, rng=rng)
    return int(ok.sum()), int(len(np.unique(trials[ok]))), float(res.bits_corrected), float(res.p_value)
