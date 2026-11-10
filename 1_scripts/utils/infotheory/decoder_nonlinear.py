# -*- coding: utf-8 -*-
"""
decoder_nonlinear
=================
RBF-kernel SVM decoder for 1e_behavior_decoding, scored EXACTLY like infotheory.decoder: balanced accuracy (chance = 1 / n_classes
whatever the class imbalance), stratified K-fold CV repeated R times, chance level from the same CV on label permutations
(permuted WITHIN `groups` = sessions, so a pooled decoder keeps the session structure of the null). It returns the same keys as
decoder.decode (acc, null_mean, null_sd, chance, n), so 1e's tables, caches and figures treat it as one more decoder.

Why a separate file: infotheory.decoder is numpy / scipy only; a kernel SVM needs scikit-learn (SVC, libsvm), imported lazily so
that nothing else in the package depends on it.

Settings (fixed, NOT tuned on the test folds; the linear SVM of decoder.py is also fixed at C = 1):
    kernel rbf, C = RBF_C, gamma = "scale" (1 / (n_features * X.var()); the features are z-scored per session and window by 1e),
    class_weight = "balanced".
Folds are drawn from a seed derived from the DATA (labels + group sizes), not from the decoder's name, so two decoders that
are given the same trials see the same splits (paired comparison of linear vs RBF).
"""
from __future__ import annotations

import zlib

import numpy as np

RBF_C = 1.0
RBF_GAMMA = "scale"
RBF_VERSION = "rbf-v1"
RBF_NAME = "svm_rbf"


def _balanced_acc(y_true, y_pred, classes):
    rec = [np.mean(y_pred[y_true == c] == c) for c in classes if np.any(y_true == c)]
    return float(np.mean(rec))


def _stratified_folds(y, n_folds, seed):
    """Stratified K-fold: each class is shuffled and dealt round-robin over the folds (every fold gets a share of every class)."""
    rng = np.random.default_rng(seed)
    fold = np.empty(len(y), int)
    offset = 0
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        fold[idx] = (np.arange(len(idx)) + offset) % n_folds
        offset += len(idx)
    return fold


def _cv_score(X, y, classes, folds_list, C, gamma):
    from sklearn.svm import SVC
    accs = []
    for fold in folds_list:
        pred = np.empty(len(y), y.dtype)
        ok = True
        for k in range(fold.max() + 1):
            te = fold == k
            tr = ~te
            if len(np.unique(y[tr])) < 2:
                ok = False
                break
            clf = SVC(kernel="rbf", C=C, gamma=gamma, class_weight="balanced")
            clf.fit(X[tr], y[tr])
            pred[te] = clf.predict(X[te])
        if ok:
            accs.append(_balanced_acc(y, pred, classes))
    return float(np.mean(accs)) if accs else np.nan


def decode_rbf(X, y, groups=None, n_folds=5, n_repeats=3, n_perm=20, rng=None, C=RBF_C, gamma=RBF_GAMMA, **_ignored):
    """Cross-validated balanced accuracy of an RBF SVM and its permutation null. Returns dict(acc, null_mean, null_sd, chance, n) or None."""
    try:
        from sklearn.svm import SVC          # noqa: F401
    except ImportError as e:                 # pragma: no cover
        raise SystemExit("The 'svm_rbf' decoder needs scikit-learn (pip install scikit-learn).") from e
    X = np.asarray(X, float)
    y = np.asarray(y)
    classes = np.unique(y)
    if len(classes) < 2 or len(y) < n_folds * 2 or min(np.sum(y == c) for c in classes) < n_folds:
        return None
    rng = rng if rng is not None else np.random.default_rng(0)
    seed0 = zlib.crc32(np.ascontiguousarray(y).tobytes() + str(X.shape).encode()) & 0x7FFFFFFF
    folds_list = [_stratified_folds(y, n_folds, seed0 + r) for r in range(n_repeats)]
    acc = _cv_score(X, y, classes, folds_list, C, gamma)
    null = []
    if n_perm and n_perm > 0:
        g = np.zeros(len(y), int) if groups is None else np.asarray(groups)
        for _ in range(n_perm):
            yp = y.copy()
            for gv in np.unique(g):
                idx = np.flatnonzero(g == gv)
                yp[idx] = y[rng.permutation(idx)]
            null.append(_cv_score(X, yp, classes, folds_list, C, gamma))     # same splits as the real data
    null = np.asarray(null, float)
    return dict(acc=acc, null_mean=float(np.nanmean(null)) if null.size else np.nan,
                null_sd=float(np.nanstd(null)) if null.size else np.nan, chance=1.0 / len(classes), n=int(len(y)))
