# -*- coding: utf-8 -*-
"""
trial_tuning.py
===============
Trial-level tuning statistics shared by cross-area tuning scripts (2l): for many target cells at once, the curve of the cell's
trial response over the bins of ONE per-trial variable, split-half curves, eta2, a permutation null and per-cell summary rows.
(The same maths is inlined in 2k_neuron_tuning_curves.py.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def tune_trials(R, xb, nb, odd, rng, n_perm, min_trials=30, min_per_bin=5, strata=None):
    """R (n_trials, N) trial responses, xb (n_trials,) bin index (-1 = trial not used), odd (n_trials,) bool.
    Responses are z-scored across the used trials. Null: the bin labels are permuted across trials (within `strata` if given,
    e.g. within stimulus levels, so that a stimulus-driven relation does not count). Returns None if too few trials; else a dict
    with curve / h1 / h2 (nb, N), n (nb,), eta2, eta2_corr, p (N,), n_samples and `good` (N,) = cells with non-zero variance."""
    valid = (xb >= 0) & np.isfinite(R).all(1)
    Rv, xv, ov = R[valid], xb[valid], odd[valid]
    sv = None if strata is None else np.asarray(strata)[valid]
    cnt = np.bincount(xv, minlength=nb).astype(float) if len(xv) else np.zeros(nb)
    if valid.sum() < min_trials or (cnt < min_per_bin).any():
        return None
    mu, sd = Rv.mean(0), Rv.std(0)
    sd = np.where(sd > 0, sd, np.nan)
    Z = (Rv - mu) / sd

    def means(Zs, xs):
        return np.vstack([Zs[xs == i].mean(0) if (xs == i).any() else np.full(Zs.shape[1], np.nan) for i in range(nb)])

    def eta(xs):
        S = np.vstack([Z[xs == i].sum(0) for i in range(nb)])
        c = np.array([(xs == i).sum() for i in range(nb)], float)
        tot = Z.sum(0)
        n = len(xs)
        ssb = (S ** 2 / np.maximum(c, 1)[:, None]).sum(0) - tot ** 2 / n
        sst = (Z ** 2).sum(0) - tot ** 2 / n
        with np.errstate(invalid="ignore", divide="ignore"):
            return ssb / sst

    def perm():
        if sv is None:
            return rng.permutation(xv)
        out = xv.copy()
        for s in np.unique(sv):
            i = np.where(sv == s)[0]
            out[i] = rng.permutation(xv[i])
        return out

    obs = eta(xv)
    null = np.vstack([eta(perm()) for _ in range(n_perm)])
    p = (1 + (null >= obs).sum(0)) / (1 + n_perm)
    return dict(curve=means(Z, xv), h1=means(Z[ov], xv[ov]), h2=means(Z[~ov], xv[~ov]), n=cnt, eta2=obs,
                eta2_corr=obs - null.mean(0), p=p, n_samples=int(valid.sum()), good=np.isfinite(sd))


def pack_stats(sid, cell_ids, variable, res, nb):
    """Long curve rows + per-cell stats rows from a tune_trials result already restricted to the good cells."""
    n_cells = len(cell_ids)
    cur, h1, h2 = res["curve"], res["h1"], res["h2"]
    rows = pd.DataFrame({"session_id": sid, "cell_id": np.tile(cell_ids, nb), "variable": variable,
                         "bin": np.repeat(np.arange(nb), n_cells), "n": np.repeat(res["n"], n_cells),
                         "mean": cur.ravel(), "mean_h1": h1.ravel(), "mean_h2": h2.ravel()})
    with np.errstate(all="ignore"):
        mod = np.nanmax(cur, 0) - np.nanmin(cur, 0)
        dl = cur[-1] - cur[0]
        d1, d2 = h1[-1] - h1[0], h2[-1] - h2[0]
        if nb >= 3:
            a, c = h1 - np.nanmean(h1, 0), h2 - np.nanmean(h2, 0)
            sh = np.nansum(a * c, 0) / np.sqrt(np.nansum(a ** 2, 0) * np.nansum(c ** 2, 0))
        else:
            sh = np.full(n_cells, np.nan)
    st = pd.DataFrame({"session_id": sid, "cell_id": cell_ids, "variable": variable, "n_samples": res["n_samples"],
                       "n_bins": nb, "eta2": res["eta2"], "eta2_corr": res["eta2_corr"], "p_value": res["p"],
                       "modulation": mod, "diff_last_first": dl, "diff_h1": d1, "diff_h2": d2, "split_half_r": sh})
    return rows, st


def restrict_result(res):
    """Keep only the good cells of a tune_trials result."""
    g = res["good"]
    per_cell = ("eta2", "eta2_corr", "p")
    return {k: (v[:, g] if isinstance(v, np.ndarray) and v.ndim == 2 else (v[g] if k in per_cell else v))
            for k, v in res.items()}
