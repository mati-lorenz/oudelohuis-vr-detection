# -*- coding: utf-8 -*-
"""
Information-theoretic and linear-regression comparison utilities:
histogram-based mutual information (bits), shuffle-corrected for
finite-sample bias, plus a matching linear-regression summary so the two
can be compared directly -- "does the linear fit capture everything the
raw MI does" is the same logic the project's a_docs describes reusing
between the later 2c_information and 2d_linear_encod/2e_nonlinear_encod
steps, just applied here at the behavioral level first.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
from scipy.special import digamma


def mutual_information_hist(x, y, bins: int = 20) -> float:
    """Plug-in histogram estimate of mutual information (bits) between
    two continuous variables, using quantile (equal-occupancy) bin edges
    rather than fixed-width ones -- better resolution where the data is
    actually concentrated, and it naturally collapses to fewer bins for
    a variable with few unique values (e.g. a 6-level stimulus strength)
    instead of forcing 20 mostly-empty bins on it.

    Note: histogram MI is known to systematically UNDER-estimate strong
    (high-MI) relationships regardless of bin count, since discretizing
    a tight, smooth dependency always loses some real structure -- this
    is why `mi_from_r2` is used as a floor in `mutual_information_shuffle`'s
    caller rather than trusted blindly when it exceeds the raw estimate."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 10:
        return np.nan

    x_edges = _quantile_bin_edges(x, bins)
    y_edges = _quantile_bin_edges(y, bins)
    c_xy, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    p_xy = c_xy / c_xy.sum()
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    denom = p_x * p_y
    nz = (p_xy > 0) & (denom > 0)
    return float(np.sum(p_xy[nz] * np.log2(p_xy[nz] / denom[nz])))


def _quantile_bin_edges(x: np.ndarray, bins: int) -> np.ndarray:
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.array([x.min() - 1e-9, x.max() + 1e-9])
    return edges


@dataclass
class MIResult:
    n: int
    bits_raw: float
    bits_shuffle_mean: float
    bits_shuffle_std: float
    bits_corrected: float  # raw - shuffle_mean, floored at 0
    p_value: float          # fraction of shuffles with MI >= the observed (raw) MI


def mutual_information_shuffle(x, y, bins: int = 20, n_shuffles: int = 50,
                                rng: np.random.Generator | None = None) -> MIResult:
    """Bias-corrected MI: subtract the mean MI obtained from shuffling y
    (destroys any real relationship but keeps the same marginals/bin
    occupancy, and thus the same finite-sample bias the raw estimate
    has). Also reports a shuffle-test p-value."""
    rng = rng or np.random.default_rng()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    mi_raw = mutual_information_hist(x, y, bins=bins)
    shuffle_mis = np.array([mutual_information_hist(x, rng.permutation(y), bins=bins)
                             for _ in range(n_shuffles)])

    mean_s, std_s = float(np.mean(shuffle_mis)), float(np.std(shuffle_mis))
    corrected = max(mi_raw - mean_s, 0.0)
    p_value = float(np.mean(shuffle_mis >= mi_raw))
    return MIResult(n=len(x), bits_raw=mi_raw, bits_shuffle_mean=mean_s,
                     bits_shuffle_std=std_s, bits_corrected=corrected, p_value=p_value)


def mutual_information_ksg(x, y, k: int = 5) -> float:
    """Kraskov-Stogbauer-Grassberger (2004) k-nearest-neighbor MI
    estimator, in bits.

    Histogram-based MI (`mutual_information_hist`/`_shuffle` above)
    systematically UNDERESTIMATES MI for smooth relationships, because
    binning averages away structure within each bin -- coarse enough
    binning can make a linear relationship's histogram-MI come out
    *below* the MI implied by its own r^2 (`mi_from_r2`), which isn't
    meaningful (r^2's implied MI should never exceed a good measurement
    of the actual MI for that same relationship). KSG doesn't have this
    bias, so it's what `1c_behavior.py` actually compares against
    `mi_from_r2`.

    A tiny amount of noise is added to break exact ties (KSG assumes
    continuous, tie-free data).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n <= k + 1:
        return np.nan

    # Standardize first: KSG's neighbor search uses distance in the JOINT
    # (x, y) space, so if the two variables have very different native
    # scales (e.g. position in the thousands vs. motion energy near 0.5),
    # the larger-scale one dominates every distance computation and the
    # neighbor structure becomes meaningless. Z-scoring is a linear,
    # invertible transform, so it doesn't change the true MI at all --
    # it just makes the distance metric treat both axes fairly.
    x = (x - np.mean(x)) / (np.std(x) or 1.0)
    y = (y - np.mean(y)) / (np.std(y) or 1.0)

    rng = np.random.default_rng(0)
    x = x + rng.normal(0, 1e-6, n)
    y = y + rng.normal(0, 1e-6, n)

    xy = np.column_stack([x, y])
    tree_xy = cKDTree(xy)
    # Chebyshev (max-norm) distance to the k-th nearest neighbor in the
    # joint space (k+1 because the point itself is its own 0-distance
    # neighbor):
    dist_k, _ = tree_xy.query(xy, k=k + 1, p=np.inf)
    eps = dist_k[:, -1]

    tree_x = cKDTree(x[:, None])
    tree_y = cKDTree(y[:, None])
    # Strictly-less-than count in each marginal (subtract a hair from the
    # radius so points exactly at distance eps aren't double-counted).
    # Floored at 0 rather than left to go negative: when eps is already
    # near machine precision (near-duplicate/collinear data), a naive
    # `eps - 1e-12` can go negative, and query_ball_point with a negative
    # radius finds NOTHING -- not even the point itself -- so the "-1"
    # self-correction below produces -1, and digamma(-1+1)=digamma(0) is
    # a pole that blows MI up to +/-inf. Flooring at 0 keeps the
    # self-match (distance exactly 0), giving a finite digamma(1) instead.
    n_x = np.array([tree_x.query_ball_point([xi], r=max(e - 1e-12, 0.0), p=np.inf).__len__() - 1
                     for xi, e in zip(x, eps)])
    n_y = np.array([tree_y.query_ball_point([yi], r=max(e - 1e-12, 0.0), p=np.inf).__len__() - 1
                     for yi, e in zip(y, eps)])

    mi_nats = digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(n)
    return float(max(mi_nats, 0.0) / np.log(2))


def mutual_information_shuffle_ksg(x, y, k: int = 5, n_shuffles: int = 20,
                                    rng: np.random.Generator | None = None) -> MIResult:
    """KSG MI with a shuffle-test null (KSG has much less finite-sample
    bias than histogram MI, but it's not exactly zero, and the shuffle
    null also gives a significance p-value)."""
    rng = rng or np.random.default_rng()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    mi_raw = mutual_information_ksg(x, y, k=k)
    shuffle_mis = np.array([mutual_information_ksg(x, rng.permutation(y), k=k)
                             for _ in range(n_shuffles)])

    mean_s, std_s = float(np.nanmean(shuffle_mis)), float(np.nanstd(shuffle_mis))
    corrected = max(mi_raw - mean_s, 0.0)
    p_value = float(np.mean(shuffle_mis >= mi_raw))
    return MIResult(n=len(x), bits_raw=mi_raw, bits_shuffle_mean=mean_s,
                     bits_shuffle_std=std_s, bits_corrected=corrected, p_value=p_value)


def mi_from_r2(r2: float) -> float:
    """MI (bits) implied by r^2 under a bivariate-Gaussian/linear
    assumption: MI = -0.5*log2(1-r^2). By the data-processing
    inequality, the TRUE mutual information between x and y can never be
    less than this -- a linear fit is just one particular reduction of
    the full relationship, so whatever information it captures is a
    lower bound on the total. If an empirical (histogram) MI estimate
    comes out below this value, that reflects the estimator's own
    downward bias for strong relationships (see `mutual_information_hist`),
    not a real violation -- use `compare_mi_to_linear` below rather than
    comparing the raw numbers directly."""
    r2 = min(max(r2, 0.0), 1 - 1e-12)
    return float(-0.5 * np.log2(1 - r2))


@dataclass
class MIvsLinear:
    mi_corrected_bits: float   # empirical, shuffle-corrected histogram MI
    mi_from_r2_bits: float     # MI implied by the linear fit's r^2
    mi_reference_bits: float   # max(the two above) -- best available lower bound on the true MI
    frac_explained_linearly: float  # in [0, 1] by construction


def compare_mi_to_linear(mi_corrected_bits: float, r2: float) -> MIvsLinear:
    """The honest version of "does the linear fit explain the MI":
    since mi_from_r2 is a provable lower bound on the true MI, take
    whichever of the two estimates is larger as the best current lower
    bound on the true MI, and express the linear fraction relative to
    THAT -- this keeps the ratio in [0, 1] instead of occasionally
    exceeding 100% purely from histogram-MI's downward bias on strong
    relationships."""
    mi_lin = mi_from_r2(r2)
    reference = max(mi_corrected_bits, mi_lin)
    frac = (mi_lin / reference) if reference > 1e-9 else np.nan
    return MIvsLinear(mi_corrected_bits=mi_corrected_bits, mi_from_r2_bits=mi_lin,
                       mi_reference_bits=reference, frac_explained_linearly=frac)


@dataclass
class LinearFit:
    n: int
    slope: float
    intercept: float
    r: float
    r2: float
    p_value: float


def linear_fit(x, y) -> LinearFit:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        # No variance in one variable (e.g. a rate column that's exactly
        # 0 for every trial in this protocol) -- there's no relationship
        # to fit, not an error; report it as zero rather than crashing.
        return LinearFit(n=len(x), slope=0.0, intercept=float(np.mean(y)) if len(y) else np.nan,
                          r=0.0, r2=0.0, p_value=1.0)
    result = stats.linregress(x, y)
    return LinearFit(n=len(x), slope=result.slope, intercept=result.intercept,
                      r=result.rvalue, r2=result.rvalue ** 2, p_value=result.pvalue)
