# -*- coding: utf-8 -*-
"""
conditional_info.py
====================
Information about a variable X in a cell's response Y that REMAINS once a third variable Z (e.g. running speed) is
accounted for: I(X;Y|Z), with the same histogram estimator and bins as the plain MI of 2c
(info_theory.mutual_information_hist / conditional_mutual_information_hist, same quantile_bin_edges).

Bias correction mirrors the unconditioned case, but the shuffle must keep the X-Z relation (otherwise the null
would also remove the part of X that is explained by Z and the "conditional" value would be biased upward): X is
permuted WITHIN each Z bin. The plain MI is recomputed on exactly the same trials with its own (global) shuffle null,
so the two numbers are directly comparable.
"""
from __future__ import annotations

import numpy as np

from .info_theory import quantile_bin_edges, mutual_information_hist, conditional_mutual_information_hist


def _digitize(v, bins):
    return np.clip(np.digitize(v, quantile_bin_edges(v, bins)[1:-1]), 0, bins - 1)


def _permute_within(x, groups, rng):
    out = x.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        out[idx] = x[rng.permutation(idx)]
    return out


def mi_and_conditional_mi(x, y, z, bins, n_shuffles, rng):
    """dict with, on the SAME finite trials: n, mi_raw, mi_corrected, mi_p (plain MI, global shuffle) and
    cmi_raw, cmi_corrected, cmi_p (I(X;Y|Z), X shuffled within Z bins). Corrected = max(raw - mean null, 0);
    p = fraction of shuffles >= raw."""
    x, y, z = (np.asarray(v, dtype=float) for v in (x, y, z))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    res = dict(n=len(x), mi_raw=np.nan, mi_corrected=np.nan, mi_p=np.nan,
               cmi_raw=np.nan, cmi_corrected=np.nan, cmi_p=np.nan)
    if len(x) < 10 or len(np.unique(x)) < 2:
        return res
    zb = _digitize(z, bins)
    mi_raw = mutual_information_hist(x, y, bins=bins)
    cmi_raw = conditional_mutual_information_hist(x, y, z, bins=bins)
    mi_null = np.array([mutual_information_hist(rng.permutation(x), y, bins=bins) for _ in range(n_shuffles)])
    cmi_null = np.array([conditional_mutual_information_hist(_permute_within(x, zb, rng), y, z, bins=bins)
                         for _ in range(n_shuffles)])
    res.update(mi_raw=mi_raw, mi_corrected=max(mi_raw - mi_null.mean(), 0.0), mi_p=float(np.mean(mi_null >= mi_raw)),
               cmi_raw=cmi_raw, cmi_corrected=max(cmi_raw - cmi_null.mean(), 0.0),
               cmi_p=float(np.mean(cmi_null >= cmi_raw)))
    return res
