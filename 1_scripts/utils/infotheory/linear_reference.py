# -*- coding: utf-8 -*-
"""
linear_reference.py
====================
What does a LINEAR relationship predict for the BINNED mutual information?

2c measures MI on RESPONSE_BINS (3) quantile bins; 2d fits a line. The old comparison put the binned MI next to
-0.5*log2(1-r^2), the MI of a CONTINUOUS bivariate Gaussian. Those are not the same quantity: discretising to a few bins
throws information away, so even for a perfectly linear-Gaussian cell the 3-bin MI sits well BELOW the continuous
value (rho=0.6: 0.20 vs 0.33 bits; rho=0.8: 0.42 vs 0.73), and "linear MI < full MI" only holds for the true,
continuous MI. The honest comparison uses the same discretisation on both sides:

    expected_binned_mi(rho, x_props, y_props)
        = MI (bits) of two variables that are bivariate normal with correlation rho, each cut into bins with the
          occupancies x_props / y_props (the pair's ACTUAL bin occupancies, from bin_props). It is the value the
          binned estimator converges to if the relationship were linear (Gaussian) with correlation rho.

If the binned MI is larger than this: information beyond a line (nonlinear/ non-Gaussian). If it is smaller:
the Pearson r is carried by few extreme trials (outliers, zero-inflated predictors such as lick), noise, or the
estimator's bias; the quantile binning only sees RANKS, so rho = Spearman's is the matched measure and Pearson's is the
outlier-sensitive one -- 2d reports both.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.stats import norm

from .info_theory import quantile_bin_edges

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(48)


def bin_props(v, bins: int) -> np.ndarray:
    """Occupancy (fractions) of the bins mutual_information_hist uses for `v`."""
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    counts, _ = np.histogram(v, bins=quantile_bin_edges(v, bins))
    return counts / counts.sum()


def props_to_str(p) -> str:
    return ";".join(f"{x:.4f}" for x in p)


def str_to_props(s: str) -> np.ndarray:
    return np.array([float(x) for x in s.split(";")])


def _thresholds(props):
    return np.concatenate([[-np.inf], norm.ppf(np.clip(np.cumsum(props)[:-1], 1e-9, 1 - 1e-9)), [np.inf]])


@lru_cache(maxsize=200000)
def _expected(rho_r: float, px: tuple, py: tuple) -> float:
    px_, py_ = np.array(px), np.array(py)
    if abs(rho_r) < 1e-6 or len(px_) < 2 or len(py_) < 2:
        return 0.0
    tx, ty = _thresholds(px_), _thresholds(py_)
    s = np.sqrt(1.0 - rho_r ** 2)
    P = np.zeros((len(px_), len(py_)))
    for i in range(len(px_)):
        lo, hi = max(tx[i], -9.0), min(tx[i + 1], 9.0)
        u = 0.5 * (hi - lo) * _NODES + 0.5 * (hi + lo)
        w = 0.5 * (hi - lo) * _WEIGHTS * norm.pdf(u)
        cdf = norm.cdf((ty[None, :] - rho_r * u[:, None]) / s)          # (nodes, edges)
        P[i] = (w[:, None] * (cdf[:, 1:] - cdf[:, :-1])).sum(0)
    P = np.clip(P, 0, None)
    P /= P.sum()
    pxm, pym = P.sum(1, keepdims=True), P.sum(0, keepdims=True)
    nz = P > 0
    return float(np.sum(P[nz] * np.log2(P[nz] / (pxm @ pym)[nz])))


def expected_binned_mi(rho: float, x_props, y_props) -> float:
    """Binned MI (bits) expected for a linear-Gaussian relation of correlation `rho` (sign irrelevant)."""
    if not np.isfinite(rho):
        return np.nan
    rho = float(min(abs(rho), 0.9995))
    return _expected(round(rho, 3), tuple(np.round(x_props, 3)), tuple(np.round(y_props, 3)))
