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


def quantile_bin_edges(x: np.ndarray, bins: int) -> np.ndarray:
    """Quantile (equal-occupancy) bin edges -- EXCEPT for a discrete or
    low-cardinality variable (few unique values relative to `bins`,
    e.g. a binary 0/1 indicator, OR a heavily zero-inflated rate like a
    lick rate where most values are exactly 0), where naive quantile
    edges collapse disastrously: if enough quantiles land on the same
    repeated value, deduplication leaves far fewer bins than requested
    -- in the worst case a single bin spanning the whole range,
    completely erasing any distinction in the data (silently zeroing
    out MI for a binary predictor, or a plotted "tuning curve" that's
    just one point). When the data has few unique values, give each of
    them its own bin instead (edges at the midpoints between
    consecutive sorted unique values), which is exact for discrete data
    and still correct for continuous data. Used both for MI's histogram
    binning and for any exploratory quantile-binned plot (see
    `robust_qcut` below) -- the same failure mode hits both."""
    unique_vals = np.unique(x)
    if len(unique_vals) <= bins:
        if len(unique_vals) == 1:
            return np.array([unique_vals[0] - 1e-9, unique_vals[0] + 1e-9])
        mids = (unique_vals[:-1] + unique_vals[1:]) / 2
        return np.concatenate([[unique_vals[0] - 1e-9], mids, [unique_vals[-1] + 1e-9]])

    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.array([x.min() - 1e-9, x.max() + 1e-9])
    return edges


# Old private name, kept as an alias in case anything imports it directly.
_quantile_bin_edges = quantile_bin_edges


def robust_qcut(x, bins: int):
    """Drop-in replacement for `pandas.qcut(x, bins, duplicates="drop")`
    that doesn't collapse catastrophically for skewed/zero-inflated/
    low-cardinality data (see `quantile_bin_edges` above for why plain
    qcut does) -- e.g. a lick-rate variable that's ~60%+ exactly 0 can
    collapse `pd.qcut(..., 5, duplicates="drop")` down to just 1-2 bins
    instead of 5, making a "binned tuning curve" plot show only one or
    two points. Returns a pandas Categorical of Intervals, same as
    `pd.cut`/`pd.qcut`, so existing `.groupby(bins)` / `interval.mid`
    code keeps working unchanged -- just call this instead of
    `pd.qcut(...duplicates="drop")` everywhere binning is needed for a
    plot or a grouped summary."""
    import pandas as pd
    x = pd.Series(x)
    finite = x.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return pd.cut(x, bins=[-1e-9, 1e-9])
    edges = quantile_bin_edges(finite, bins)
    return pd.cut(x, bins=edges, include_lowest=True)


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

    x_edges = quantile_bin_edges(x, bins)
    y_edges = quantile_bin_edges(y, bins)
    c_xy, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    p_xy = c_xy / c_xy.sum()
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    denom = p_x * p_y
    nz = (p_xy > 0) & (denom > 0)
    return float(np.sum(p_xy[nz] * np.log2(p_xy[nz] / denom[nz])))


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


def conditional_mutual_information_hist(x, y, z, bins: int = 10) -> float:
    """I(X;Y|Z) via histogram binning and the entropy decomposition
    I(X;Y|Z) = H(X,Z) + H(Y,Z) - H(X,Y,Z) - H(Z) (bits).

    Each variable is independently quantile-binned (`quantile_bin_edges`,
    robust to low-cardinality/discrete inputs, same as the rest of this
    module), then joint/marginal entropies are computed from the
    resulting multi-way histogram. This is the building block
    `transfer_entropy_hist` below uses (transfer entropy IS a
    conditional MI, just with a specific choice of X/Y/Z -- a target's
    own future given its own past, conditioned on a source's past).

    Non-negative by construction; floored at 0 to absorb floating-point
    noise from the entropy subtraction.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    n = len(x)
    if n < 10:
        return np.nan

    def digitize(v):
        edges = quantile_bin_edges(v, bins)
        return np.clip(np.digitize(v, edges[1:-1]), 0, bins - 1)

    xb, yb, zb = digitize(x), digitize(y), digitize(z)

    def joint_entropy(*binned_arrays):
        codes = np.zeros(n, dtype=np.int64)
        mult = 1
        for arr in binned_arrays:
            codes += arr.astype(np.int64) * mult
            mult *= int(arr.max()) + 1
        _, counts = np.unique(codes, return_counts=True)
        p = counts / counts.sum()
        return float(-np.sum(p * np.log2(p)))

    h_xz = joint_entropy(xb, zb)
    h_yz = joint_entropy(yb, zb)
    h_xyz = joint_entropy(xb, yb, zb)
    h_z = joint_entropy(zb)
    return max(h_xz + h_yz - h_xyz - h_z, 0.0)


def transfer_entropy_hist(source, target, bins: int = 10, target_horizon_frames: int = 1) -> float:
    """Transfer entropy TE(source -> target) = I(target_future ;
    source_past | target_past), in bits -- how much knowing `source`'s
    CURRENT value reduces uncertainty about `target`'s value
    `target_horizon_frames` samples later, BEYOND what `target`'s own
    current value already predicts about its own future. This is what
    distinguishes TE from plain lagged mutual information: a lagged MI
    between source[t] and target[t+h] can be large purely because
    target is autocorrelated and source happens to be too (no real
    directed link needed) -- TE conditions that out.

    `source` should already be whatever lag-shifted version of the
    behavioral variable the caller wants tested (same convention as
    2f_lag_information.py's shift_with_nan) -- this function only
    handles the target's own future/past split via `target_horizon_frames`.

    Returns NaN if there's insufficient valid (non-NaN, non-edge) data.
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    T = len(target)
    h = target_horizon_frames
    if h <= 0 or h >= T:
        return np.nan
    target_future = target[h:]
    target_past = target[:-h]
    source_past = source[:-h]
    valid = np.isfinite(target_future) & np.isfinite(target_past) & np.isfinite(source_past)
    if valid.sum() < 30:
        return np.nan
    return conditional_mutual_information_hist(source_past[valid], target_future[valid],
                                                 target_past[valid], bins=bins)


def transfer_entropy_shuffle(source, target, bins: int = 10, target_horizon_frames: int = 1,
                              n_shuffles: int = 20, rng: np.random.Generator | None = None) -> MIResult:
    """Bias-corrected transfer entropy: subtract the mean TE obtained
    from randomly permuting `source` in time (destroys any real
    source->target link while leaving target's own (past, future)
    pairing -- and thus target's own autocorrelation structure --
    completely intact, so the shuffle null correctly isolates the bias
    that comes from finite-sample binning rather than from target's own
    predictability). Same MIResult shape as mutual_information_shuffle,
    for consistency with the rest of this module and with
    2f_lag_information.py's existing MI-result handling."""
    rng = rng or np.random.default_rng()
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)

    te_raw = transfer_entropy_hist(source, target, bins=bins, target_horizon_frames=target_horizon_frames)
    if not np.isfinite(te_raw):
        return MIResult(n=0, bits_raw=np.nan, bits_shuffle_mean=np.nan, bits_shuffle_std=np.nan,
                         bits_corrected=np.nan, p_value=np.nan)

    shuffle_tes = []
    for _ in range(n_shuffles):
        te_shuf = transfer_entropy_hist(rng.permutation(source), target, bins=bins,
                                         target_horizon_frames=target_horizon_frames)
        if np.isfinite(te_shuf):
            shuffle_tes.append(te_shuf)
    shuffle_tes = np.array(shuffle_tes) if shuffle_tes else np.array([np.nan])

    mean_s, std_s = float(np.nanmean(shuffle_tes)), float(np.nanstd(shuffle_tes))
    corrected = max(te_raw - mean_s, 0.0) if np.isfinite(mean_s) else np.nan
    p_value = float(np.mean(shuffle_tes >= te_raw)) if np.isfinite(mean_s) else np.nan
    n_valid = int(np.isfinite(source[target_horizon_frames:]) .sum())
    return MIResult(n=n_valid, bits_raw=te_raw, bits_shuffle_mean=mean_s, bits_shuffle_std=std_s,
                     bits_corrected=corrected, p_value=p_value)


def granger_causality_test(target, source_lagged, ar_order: int = 3):
    """Granger causality: does `source_lagged` (already whatever
    lag-shifted version of a candidate driver the caller wants tested)
    improve prediction of `target[t]` beyond target's OWN autoregressive
    history? F-test comparing a restricted model (target[t] ~ target's
    own last `ar_order` samples) against an unrestricted model (same,
    plus source_lagged[t]).

    This is a STRONGER claim than a plain lagged linear correlation:
    two variables can be correlated at a lag purely because both are
    autocorrelated and happen to share a similar timescale, with no
    real directed link -- Granger causality specifically asks whether
    the source adds anything BEYOND what the target's own past already
    explains about itself.

    Returns
    -------
    r2_restricted, r2_unrestricted, p_value : floats (p_value from an
        F-test on the RSS reduction; NaN if too few valid samples)
    """
    target = np.asarray(target, dtype=float)
    source_lagged = np.asarray(source_lagged, dtype=float)
    T = len(target)
    if T <= ar_order + 20:
        return np.nan, np.nan, np.nan

    y = target[ar_order:]
    own_past = np.column_stack([target[ar_order - k:T - k] for k in range(1, ar_order + 1)])
    src = source_lagged[ar_order:]
    valid = np.isfinite(y) & np.all(np.isfinite(own_past), axis=1) & np.isfinite(src)
    y, own_past, src = y[valid], own_past[valid], src[valid]
    n = len(y)
    if n <= ar_order + 20:
        return np.nan, np.nan, np.nan

    own_names = [f"y_t-{k}" for k in range(1, ar_order + 1)]
    restricted = multivariate_linear_fit(own_past, y, predictor_names=own_names)
    unrestricted = multivariate_linear_fit(np.column_stack([own_past, src]), y,
                                            predictor_names=own_names + ["source"])

    tss = float(np.sum((y - y.mean()) ** 2))
    if tss <= 0:
        return restricted.r2, unrestricted.r2, np.nan
    rss_r = (1 - restricted.r2) * tss
    rss_u = (1 - unrestricted.r2) * tss
    q = 1  # one additional predictor (source_lagged) in the unrestricted model
    k_full = ar_order + 1
    dof2 = n - k_full - 1
    if dof2 <= 0 or rss_u <= 0:
        return restricted.r2, unrestricted.r2, np.nan
    f_stat = ((rss_r - rss_u) / q) / (rss_u / dof2)
    if f_stat <= 0:
        return restricted.r2, unrestricted.r2, 1.0
    from scipy import stats
    p_value = float(1 - stats.f.cdf(f_stat, q, dof2))
    return restricted.r2, unrestricted.r2, p_value


def mutual_information_ksg(x, y, k: int = 5) -> float:
    """Kraskov-Stogbauer-Grassberger (2004) k-nearest-neighbor MI
    estimator, in bits.

    `x` may be 1D (a single variable) or 2D (n_samples, n_features), for
    a MULTIVARIATE MI(X_set; y) estimate -- same algorithm, just with a
    higher-dimensional "X" side of the joint space (e.g. "how much do
    speed AND pupil area TOGETHER tell you about choice", not just each
    one pairwise). `y` stays 1D either way.

    Histogram-based MI (`mutual_information_hist`/`_shuffle` above)
    systematically UNDERESTIMATES MI for smooth relationships, because
    binning averages away structure within each bin -- coarse enough
    binning can make a linear relationship's histogram-MI come out
    *below* the MI implied by its own r^2 (`mi_from_r2`), which isn't
    meaningful (r^2's implied MI should never exceed a good measurement
    of the actual MI for that same relationship). KSG doesn't have this
    bias, so it's what `1c_behavior.py` actually compares against
    `mi_from_r2`. It also doesn't suffer the curse-of-dimensionality
    binning problem histogram MI would hit for a multivariate X, which
    is why it's the estimator used for the multivariate case too.

    A tiny amount of noise is added to break exact ties (KSG assumes
    continuous, tie-free data).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    y = np.asarray(y, dtype=float)
    mask = np.all(np.isfinite(x), axis=1) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(y)
    if n <= k + 1:
        return np.nan

    # Standardize first: KSG's neighbor search uses distance in the JOINT
    # (x, y) space, so if the variables have very different native
    # scales (e.g. position in the thousands vs. motion energy near 0.5),
    # the larger-scale one dominates every distance computation and the
    # neighbor structure becomes meaningless. Z-scoring is a linear,
    # invertible transform, so it doesn't change the true MI at all --
    # it just makes the distance metric treat every axis fairly. Each
    # column of a multivariate x is standardized independently.
    x_std = x.std(axis=0)
    x_std[x_std == 0] = 1.0
    x = (x - x.mean(axis=0)) / x_std
    y = (y - np.mean(y)) / (np.std(y) or 1.0)

    rng = np.random.default_rng(0)
    x = x + rng.normal(0, 1e-6, x.shape)
    y = y + rng.normal(0, 1e-6, n)

    joint = np.column_stack([x, y[:, None]])
    tree_joint = cKDTree(joint)
    # Chebyshev (max-norm) distance to the k-th nearest neighbor in the
    # joint space (k+1 because the point itself is its own 0-distance
    # neighbor):
    dist_k, _ = tree_joint.query(joint, k=k + 1, p=np.inf)
    eps = dist_k[:, -1]

    tree_x = cKDTree(x)
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
    # Vectorized (one batched call with a per-point radius array and
    # return_length=True, rather than a Python loop building index lists
    # one sample at a time) -- this is the dominant cost for a
    # multivariate X with many dimensions/samples, so it matters a lot
    # more here than it did for the original pairwise-only version.
    radii = np.maximum(eps - 1e-12, 0.0)
    n_x = tree_x.query_ball_point(x, r=radii, p=np.inf, return_length=True) - 1
    n_y = tree_y.query_ball_point(y[:, None], r=radii, p=np.inf, return_length=True) - 1

    mi_nats = digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(n)
    return float(max(mi_nats, 0.0) / np.log(2))


def mutual_information_shuffle_ksg(x, y, k: int = 5, n_shuffles: int = 20,
                                    rng: np.random.Generator | None = None) -> MIResult:
    """KSG MI with a shuffle-test null (KSG has much less finite-sample
    bias than histogram MI, but it's not exactly zero, and the shuffle
    null also gives a significance p-value). `x` may be 1D or 2D
    (multivariate) -- see `mutual_information_ksg`."""
    rng = rng or np.random.default_rng()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x2d = x if x.ndim > 1 else x[:, None]
    mask = np.all(np.isfinite(x2d), axis=1) & np.isfinite(y)
    x, y = x[mask], y[mask]

    mi_raw = mutual_information_ksg(x, y, k=k)
    shuffle_mis = np.array([mutual_information_ksg(x, rng.permutation(y), k=k)
                             for _ in range(n_shuffles)])

    mean_s, std_s = float(np.nanmean(shuffle_mis)), float(np.nanstd(shuffle_mis))
    corrected = max(mi_raw - mean_s, 0.0)
    p_value = float(np.mean(shuffle_mis >= mi_raw))
    return MIResult(n=len(y), bits_raw=mi_raw, bits_shuffle_mean=mean_s,
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
    if len(x) < 3 or np.std(x) < 1e-9 * (np.abs(np.mean(x)) + 1) or np.std(y) < 1e-9 * (np.abs(np.mean(y)) + 1):
        # No REAL variance in one variable -- exact equality (`std == 0`)
        # misses near-duplicate floats from upstream floating-point noise
        # (e.g. interval-midpoint arithmetic on a low-cardinality binned
        # variable can produce values like 50.0 vs 50.00000000000001,
        # which scipy's linregress then rejects outright with "all x
        # values are identical" since it uses a stricter internal check).
        # A relative tolerance treats those as the constant they really
        # are; there's no relationship to fit either way, not an error.
        return LinearFit(n=len(x), slope=0.0, intercept=float(np.mean(y)) if len(y) else np.nan,
                          r=0.0, r2=0.0, p_value=1.0)
    result = stats.linregress(x, y)
    return LinearFit(n=len(x), slope=result.slope, intercept=result.intercept,
                      r=result.rvalue, r2=result.rvalue ** 2, p_value=result.pvalue)


@dataclass
class MultivariateLinearFit:
    n: int
    predictors: list
    coefficients: dict
    intercept: float
    r2: float
    adj_r2: float
    p_value: float  # overall F-test: does this model explain more than chance?


def multivariate_linear_fit(X, y, predictor_names: list | None = None) -> MultivariateLinearFit:
    """Ordinary least squares with multiple predictors at once (numpy-
    only, no extra dependency): y ~ intercept + b1*x1 + b2*x2 + ...
    Reports R^2 AND adjusted R^2 -- plain R^2 mechanically increases
    with every extra predictor regardless of whether it actually helps,
    so adjusted R^2 (which penalizes predictor count) is the fairer
    "does this multivariate model beat the pairwise ones" comparison --
    plus an overall F-test p-value for "does this model explain more
    than chance."

    Parameters
    ----------
    X : array (n_samples, n_features) or (n_samples,) for a single
        predictor (falls back to the same fit as `linear_fit`, just
        with the multivariate return shape)
    y : array (n_samples,)
    predictor_names : optional labels for X's columns, used as the
        `coefficients` dict's keys (defaults to x0, x1, ...)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, dtype=float)
    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    n, p = X.shape
    if predictor_names is None:
        predictor_names = [f"x{i}" for i in range(p)]

    if n <= p + 1:
        return MultivariateLinearFit(n=n, predictors=predictor_names,
                                      coefficients={pn: np.nan for pn in predictor_names},
                                      intercept=np.nan, r2=np.nan, adj_r2=np.nan, p_value=np.nan)

    design = np.column_stack([np.ones(n), X])
    coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    y_pred = design @ coeffs
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = (1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    dof_resid = n - p - 1
    adj_r2 = (1 - (1 - r2) * (n - 1) / dof_resid) if dof_resid > 0 else np.nan

    if p > 0 and ss_res > 0 and dof_resid > 0:
        f_stat = ((ss_tot - ss_res) / p) / (ss_res / dof_resid)
        p_value = float(stats.f.sf(f_stat, p, dof_resid))
    elif ss_res == 0:
        p_value = 0.0  # perfect fit
    else:
        p_value = np.nan

    return MultivariateLinearFit(n=n, predictors=predictor_names,
                                  coefficients=dict(zip(predictor_names, coeffs[1:])),
                                  intercept=float(coeffs[0]), r2=float(r2),
                                  adj_r2=float(adj_r2) if np.isfinite(adj_r2) else np.nan,
                                  p_value=p_value)


@dataclass
class StepwiseStep:
    step: int
    predictor_added: str
    cumulative_r2: float
    cumulative_adj_r2: float
    delta_r2: float


def forward_stepwise_selection(X, y, predictor_names: list) -> list:
    """Greedy forward stepwise selection: starting from no predictors,
    repeatedly add whichever remaining predictor gives the largest R^2
    improvement, until all are included. Returns a list of StepwiseStep,
    one per step, in the order predictors were added.

    Why this matters alongside the "leave-one-out" multivariate fit
    (`multivariate_linear_fit` with every other variable at once): a
    high combined R^2 alone can't tell you whether that came from many
    variables genuinely adding independent information, or from ONE
    dominant predictor doing nearly all the work while the rest just
    ride along (e.g. two variables that are themselves strongly
    correlated with each other, like a pupil-tracking x/y pair, can
    make a leave-one-out fit look like broad "multivariate structure"
    when it's really just that one pairwise relationship). Step 1's R^2
    here IS the single best predictor's pairwise R^2; if it's already
    close to the final (all-predictors) R^2, the rest are largely
    redundant with it.

    Cheap: only ordinary least squares (no MI/shuffling) at each
    candidate step, O(p^2) fits total for p predictors -- trivial for
    the small predictor counts (~8-12) this project uses.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    remaining = list(range(len(predictor_names)))
    chosen: list = []
    steps: list = []
    prev_r2 = 0.0

    while remaining:
        best_r2, best_idx, best_fit = -np.inf, None, None
        for idx in remaining:
            trial_idx = chosen + [idx]
            fit = multivariate_linear_fit(X[:, trial_idx], y,
                                           predictor_names=[predictor_names[i] for i in trial_idx])
            if np.isfinite(fit.r2) and fit.r2 > best_r2:
                best_r2, best_idx, best_fit = fit.r2, idx, fit
        if best_idx is None:
            break  # every remaining candidate produced a degenerate (NaN) fit
        chosen.append(best_idx)
        remaining.remove(best_idx)
        steps.append(StepwiseStep(step=len(chosen), predictor_added=predictor_names[best_idx],
                                   cumulative_r2=best_fit.r2, cumulative_adj_r2=best_fit.adj_r2,
                                   delta_r2=best_fit.r2 - prev_r2))
        prev_r2 = best_fit.r2

    return steps
