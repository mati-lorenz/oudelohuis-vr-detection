# -*- coding: utf-8 -*-
"""
pair_information.py
======================
The Pola et al. (2003) pairwise information-breakdown computation
(I_total, I_lin, I_sigsim, I_cor, RSI, plus the approximate I_corind/
I_cordep split), extracted here from 3b_pair_information_breakdown.py
so 3e_pair_information_breakdown_temporal.py can reuse the EXACT same,
already-validated computation at every position bin, rather than a
second, possibly-drifted copy of the same logic.

See 3b_pair_information_breakdown.py's module docstring for the full
derivation, exactness/approximation caveats, and the validation against
known synthetic ground truth (a fully-redundant "copy" case and a
tuning-plus-aligned-noise-correlation case) -- nothing about the
computation itself changed in this extraction, only its location.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .info_theory import mutual_information_shuffle, quantile_bin_edges
from .shuffling import shuffle_within_groups


def digitize_quantile(x: np.ndarray, bins: int) -> np.ndarray:
    edges = quantile_bin_edges(x, bins)
    return np.clip(np.digitize(x, edges[1:-1]), 0, bins - 1)


def joint_code(a_bin: np.ndarray, b_bin: np.ndarray, n_bins: int) -> np.ndarray:
    return a_bin.astype(np.int64) * n_bins + b_bin.astype(np.int64)


def _mi_avg_over_shuffles(a_bin, b_bin_base, stim_bin, n_bins, n_reps, mi_shuffles, rng):
    """Average bias-corrected MI(joint(a_bin, b_bin_shuffled); stim_bin)
    over `n_reps` independent within-stim shuffles of b_bin -- the
    "I_ind"-style surrogate used for both I_sigsim and the approximate
    I_corind/I_cordep split below."""
    vals = []
    for _ in range(n_reps):
        b_shuf = shuffle_within_groups(b_bin_base, stim_bin, rng=rng)
        jc = joint_code(a_bin, b_shuf, n_bins)
        vals.append(mutual_information_shuffle(jc, stim_bin, bins=n_bins * n_bins,
                                                 n_shuffles=mi_shuffles, rng=rng).bits_corrected)
    return float(np.mean(vals))


def _build_avgcor_surrogate(x1, x2, stim_bin, rng):
    """Reattach the OVERALL (stimulus-pooled) bivariate residual
    distribution to each stimulus's own tuning mean -- a surrogate that
    keeps the average correlation strength/sign but removes any
    stimulus-specific modulation of it. See 3b's module docstring for
    the "approximate" caveat."""
    tuning1 = pd.Series(x1).groupby(stim_bin).transform("mean").to_numpy()
    tuning2 = pd.Series(x2).groupby(stim_bin).transform("mean").to_numpy()
    resid1, resid2 = x1 - tuning1, x2 - tuning2
    n = len(x1)
    draw = rng.permutation(n)
    return tuning1 + resid1[draw], tuning2 + resid2[draw]


def compute_pair_breakdown(x1: np.ndarray, x2: np.ndarray, stim_bin: np.ndarray,
                            r_bins: int = 3, mi_shuffles: int = 15, n_ind_reps: int = 8,
                            rng: np.random.Generator | None = None) -> dict:
    """The full per-pair Pola breakdown computation -- exactly what
    3b_pair_information_breakdown.py runs per (cell pair, window), now
    callable directly so 3e_pair_information_breakdown_temporal.py can
    call it once per (cell pair, position bin) too.

    Returns a dict: I_total, I_1, I_2, I_lin, I_ind, I_sigsim, I_cor,
    RSI, I_corind_approx, I_cordep_approx, n_trials.
    """
    rng = rng or np.random.default_rng()
    r1_bin = digitize_quantile(x1, r_bins)
    r2_bin = digitize_quantile(x2, r_bins)

    mi1 = mutual_information_shuffle(r1_bin, stim_bin, bins=r_bins, n_shuffles=mi_shuffles,
                                      rng=rng).bits_corrected
    mi2 = mutual_information_shuffle(r2_bin, stim_bin, bins=r_bins, n_shuffles=mi_shuffles,
                                      rng=rng).bits_corrected
    i_lin = mi1 + mi2

    jc_total = joint_code(r1_bin, r2_bin, r_bins)
    i_total = mutual_information_shuffle(jc_total, stim_bin, bins=r_bins * r_bins,
                                          n_shuffles=mi_shuffles, rng=rng).bits_corrected

    i_ind = _mi_avg_over_shuffles(r1_bin, r2_bin, stim_bin, r_bins, n_ind_reps, mi_shuffles, rng)
    i_sigsim = i_ind - i_lin
    i_cor = i_total - i_ind
    rsi = i_total - i_lin

    avgcor_mis = []
    for _ in range(n_ind_reps):
        y1, y2 = _build_avgcor_surrogate(x1, x2, stim_bin, rng)
        y1b, y2b = digitize_quantile(y1, r_bins), digitize_quantile(y2, r_bins)
        jc = joint_code(y1b, y2b, r_bins)
        avgcor_mis.append(mutual_information_shuffle(jc, stim_bin, bins=r_bins * r_bins,
                                                       n_shuffles=mi_shuffles, rng=rng).bits_corrected)
    i_avgcor = float(np.mean(avgcor_mis))
    i_corind_approx = i_avgcor - i_ind
    i_cordep_approx = i_total - i_avgcor

    return {"I_total": i_total, "I_1": mi1, "I_2": mi2, "I_lin": i_lin, "I_ind": i_ind,
            "I_sigsim": i_sigsim, "I_cor": i_cor, "RSI": rsi,
            "I_corind_approx": i_corind_approx, "I_cordep_approx": i_cordep_approx,
            "n_trials": len(x1)}
