# -*- coding: utf-8 -*-
"""
roc.py
=======
ROC analysis for many cells at once: AUC (area under the ROC curve),
selectivity index, and a permutation test that says which cells
discriminate two trial classes better than chance.

Conventions
-----------
* `labels` is a boolean/0-1 vector: True/1 = the "positive" class
  (e.g. lick trials, stimulus-present trials).
* AUC is the probability that a random positive-class trial has higher
  activity than a random negative-class trial (ties count 1/2). It is
  computed from rank sums (Mann-Whitney U), so it is invariant to any
  monotone transform of the activity -- z-scoring changes nothing.
* AUC > 0.5: the cell fires MORE on the positive class.
  Selectivity index SI = 2*AUC - 1, in [-1, 1] (0 = no discrimination).
* Permutation test: shuffle the labels `n_perm` times, recompute AUC for
  every cell, and call a cell significant if its observed AUC falls
  OUTSIDE the central 95% of ITS OWN null distribution (2.5th / 97.5th
  percentiles). A two-sided p-value (with the +1 correction) is
  reported too.
* Conditioned ("fixed condition") ROC: pass `groups` (e.g. the stimulus
  level of each trial). AUC is then computed WITHIN every group and
  pooled with weights n_pos * n_neg (= the number of positive/negative
  trial pairs each group contributes -- exactly the AUC you would get
  from comparing only trials that share a condition). The null permutes
  labels WITHIN each group (same logic as shuffling.shuffle_within_groups),
  so the test asks "does this cell tell the classes apart beyond what the
  condition itself explains?".

Multiple comparisons: with a 2.5/97.5 criterion ~5% of cells are
expected to pass by chance alone, and NO correction across cells is
applied. Compare fractions of significant cells against 5%, not 0%.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

MIN_PER_CLASS = 5   # a group needs at least this many trials in BOTH classes


def _prep(X, labels):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    labels = np.asarray(labels).astype(bool)
    if len(labels) != len(X):
        raise ValueError("labels and X must have the same number of trials")
    return X, labels


def roc_auc(X, labels) -> np.ndarray:
    """AUC of every column of X (trials x cells) for `labels` (no NaNs
    allowed in X -- drop trials upstream). Returns shape (cells,); NaN if
    either class is empty."""
    X, labels = _prep(X, labels)
    n1 = int(labels.sum())
    n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return np.full(X.shape[1], np.nan)
    R = rankdata(X, axis=0)
    return (R[labels].sum(0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def selectivity_index(auc):
    """SI = 2*AUC - 1 (in [-1, 1])."""
    return 2.0 * np.asarray(auc) - 1.0


def _valid_groups(labels, groups, min_per_class):
    """List of (index array, n1, n0) for groups with enough trials in both classes."""
    if groups is None:
        groups = np.zeros(len(labels), dtype=int)
    groups = np.asarray(groups)
    out = []
    for g in np.unique(groups[~_isnan_any(groups)]):
        idx = np.where(groups == g)[0]
        n1 = int(labels[idx].sum())
        n0 = len(idx) - n1
        if n1 >= min_per_class and n0 >= min_per_class:
            out.append((idx, n1, n0))
    return out


def _isnan_any(a):
    try:
        return np.isnan(a.astype(float))
    except (TypeError, ValueError):
        return np.zeros(len(a), dtype=bool)


def roc_auc_conditioned(X, labels, groups=None, min_per_class: int = MIN_PER_CLASS):
    """Pooled within-group AUC (see module docstring). Returns
    (auc (cells,), n_pos_total, n_neg_total, n_groups_used)."""
    X, labels = _prep(X, labels)
    vg = _valid_groups(labels, groups, min_per_class)
    if not vg:
        return np.full(X.shape[1], np.nan), 0, 0, 0
    num = np.zeros(X.shape[1])
    den = 0.0
    for idx, n1, n0 in vg:
        num += n1 * n0 * roc_auc(X[idx], labels[idx])
        den += n1 * n0
    return num / den, sum(v[1] for v in vg), sum(v[2] for v in vg), len(vg)


def auc_permutation_test(X, labels, groups=None, n_perm: int = 1000,
                         rng: np.random.Generator | None = None,
                         min_per_class: int = MIN_PER_CLASS, alpha: float = 0.05,
                         chunk: int = 250) -> dict:
    """AUC + selectivity index + label-permutation null for every column.

    Parameters
    ----------
    X : (trials, cells) activity, no NaNs
    labels : (trials,) bool
    groups : optional (trials,) condition id; AUC pooled within groups and
        labels permuted within groups (see module docstring)
    n_perm : number of label permutations
    alpha : significance level; cell is significant if its AUC is outside
        the central (1 - alpha) interval of its null

    Returns dict with arrays of length cells:
        auc, si, null_lo, null_hi, p_value (two-sided, +1 corrected),
        significant (bool), plus scalars n_pos, n_neg, n_groups.
    Everything is NaN/False when no group has enough trials.
    """
    rng = rng or np.random.default_rng()
    X, labels = _prep(X, labels)
    n_cells = X.shape[1]
    nan = np.full(n_cells, np.nan)
    vg = _valid_groups(labels, groups, min_per_class)
    if not vg:
        return dict(auc=nan, si=nan, null_lo=nan, null_hi=nan, p_value=nan,
                    significant=np.zeros(n_cells, bool), n_pos=0, n_neg=0, n_groups=0)

    # ranks within each group, computed once
    ranks = [rankdata(X[idx], axis=0) for idx, _, _ in vg]
    weights = np.array([n1 * n0 for _, n1, n0 in vg], dtype=float)
    wsum = weights.sum()

    obs = np.zeros(n_cells)
    for (idx, n1, n0), R, w in zip(vg, ranks, weights):
        obs += w * (R[labels[idx]].sum(0) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    obs /= wsum

    null = np.empty((n_perm, n_cells))
    for start in range(0, n_perm, chunk):
        m = min(chunk, n_perm - start)
        acc = np.zeros((m, n_cells))
        for (idx, n1, n0), R, w in zip(vg, ranks, weights):
            ng = len(idx)
            pick = rng.random((m, ng)).argsort(axis=1)[:, :n1]
            L = np.zeros((m, ng))
            np.put_along_axis(L, pick, 1.0, axis=1)
            acc += w * (L @ R - n1 * (n1 + 1) / 2.0) / (n1 * n0)
        null[start:start + m] = acc / wsum

    lo = np.percentile(null, 100 * alpha / 2, axis=0)
    hi = np.percentile(null, 100 * (1 - alpha / 2), axis=0)
    dev = np.abs(obs - 0.5)
    p = (1.0 + (np.abs(null - 0.5) >= dev).sum(0)) / (n_perm + 1.0)
    return dict(auc=obs, si=selectivity_index(obs), null_lo=lo, null_hi=hi, p_value=p,
                significant=(obs < lo) | (obs > hi),
                n_pos=int(sum(v[1] for v in vg)), n_neg=int(sum(v[2] for v in vg)),
                n_groups=len(vg))


# --------------------------------------------------------------------- #
# Template-based discriminant ROC (Chen et al. 2013, Methods:
# "Single-neuron discrimination analysis")
# --------------------------------------------------------------------- #
def _template_dv(T, labels):
    """Leave-one-out discriminant scores for every cell.

    T : (trials, cells, bins) response profiles (no NaN); labels bool
    (True = class X, False = class Y). Following Chen et al.:
        DV_X,i = T_i . (mean_X(without i) - mean_Y)     for X trials
        DV_Y,j = T_j . (mean_X - mean_Y(without j))     for Y trials
    i.e. how much a trial's profile resembles the X template rather than
    the Y template, with the trial itself left out of its own template.
    Returns (trials, cells).
    """
    labels = np.asarray(labels).astype(bool)
    nx, ny = int(labels.sum()), int((~labels).sum())
    SX = T[labels].sum(0)                       # (cells, bins)
    SY = T[~labels].sum(0)
    MX, MY = SX / nx, SY / ny
    dv = np.empty(T.shape[:2])
    Tx, Ty = T[labels], T[~labels]
    dv[labels] = np.einsum("knb,knb->kn", Tx, (SX[None] - Tx) / (nx - 1) - MY[None])
    dv[~labels] = np.einsum("knb,knb->kn", Ty, MX[None] - (SY[None] - Ty) / (ny - 1))
    return dv


def template_auc(T, labels, groups=None, min_per_class: int = MIN_PER_CLASS):
    """Pooled (within-group) AUC of the template discriminant. Returns
    (auc (cells,), n_pos, n_neg, n_groups)."""
    T = np.asarray(T, float)
    labels = np.asarray(labels).astype(bool)
    vg = _valid_groups(labels, groups, min_per_class)
    if not vg:
        return np.full(T.shape[1], np.nan), 0, 0, 0
    num, den = np.zeros(T.shape[1]), 0.0
    for idx, n1, n0 in vg:
        dv = _template_dv(T[idx], labels[idx])
        num += n1 * n0 * roc_auc(dv, labels[idx])
        den += n1 * n0
    return num / den, sum(v[1] for v in vg), sum(v[2] for v in vg), len(vg)


def template_auc_permutation_test(T, labels, groups=None, n_perm: int = 200,
                                  rng: np.random.Generator | None = None,
                                  min_per_class: int = MIN_PER_CLASS, alpha: float = 0.05) -> dict:
    """Chen-style discrimination: template-discriminant AUC with a
    ONE-SIDED label-permutation threshold (95th percentile of each cell's
    own null; the discriminant is only informative above 0.5). Labels are
    permuted within groups when `groups` is given. Same return keys as
    auc_permutation_test (`null_lo` is NaN: no lower threshold).

    Cost is O(n_perm * trials * cells * bins): use fewer permutations
    (default 200) than the window-mean test, or subsample cells.
    """
    rng = rng or np.random.default_rng()
    T = np.asarray(T, float)
    labels = np.asarray(labels).astype(bool)
    n_cells = T.shape[1]
    nan = np.full(n_cells, np.nan)
    vg = _valid_groups(labels, groups, min_per_class)
    if not vg:
        return dict(auc=nan, si=nan, null_lo=nan, null_hi=nan, p_value=nan,
                    significant=np.zeros(n_cells, bool), n_pos=0, n_neg=0, n_groups=0)
    obs, n_pos, n_neg, n_groups = template_auc(T, labels, groups, min_per_class)
    weights = np.array([n1 * n0 for _, n1, n0 in vg], float)
    null = np.empty((n_perm, n_cells))
    for p in range(n_perm):
        acc = np.zeros(n_cells)
        for (idx, n1, n0), w in zip(vg, weights):
            lab = np.zeros(len(idx), bool)
            lab[rng.permutation(len(idx))[:n1]] = True
            acc += w * roc_auc(_template_dv(T[idx], lab), lab)
        null[p] = acc / weights.sum()
    hi = np.percentile(null, 100 * (1 - alpha), axis=0)
    p_val = (1.0 + (null >= obs[None, :]).sum(0)) / (n_perm + 1.0)
    return dict(auc=obs, si=selectivity_index(obs), null_lo=nan, null_hi=hi, p_value=p_val,
                significant=obs > hi, n_pos=n_pos, n_neg=n_neg, n_groups=n_groups)
