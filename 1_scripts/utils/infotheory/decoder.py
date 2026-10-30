# -*- coding: utf-8 -*-
"""
decoder.py
==========
Small cross-validated decoders written with numpy / scipy only (no scikit-learn needed) for 1e_behavior_decoding:
    lda   linear discriminant analysis, shared covariance (ridge 1e-2), EQUAL class priors
    logr  multinomial logistic regression, L2 penalty C = 1, class weights balanced (as sklearn LogisticRegression(C=1,
          class_weight="balanced"); checked against it)
    svm   linear SVM, squared hinge, one-vs-rest, C = 1, class weights balanced (as sklearn LinearSVC(C=1, class_weight=
          "balanced", dual=False); a binary problem is a single classifier)
Features are z-scored before they get here. All are scored with BALANCED accuracy (mean recall over classes), so the
chance level is 1 / n_classes whatever the class imbalance (choice and the stimulus bins are very unbalanced). Cross-validation
is stratified K-fold, repeated; the null is the same cross-validation on label permutations (permuted within `groups`, e.g. within
sessions, so a pooled decoder keeps each session's label structure).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def _lda_predict(Xtr, ytr, Xte, n_cls, ridge):
    d = Xtr.shape[1]
    means = np.stack([Xtr[ytr == k].mean(0) for k in range(n_cls)])
    res = Xtr - means[ytr]
    cov = res.T @ res / max(len(Xtr) - n_cls, 1) + ridge * np.eye(d)
    W = np.linalg.solve(cov, means.T)                                   # d x K
    score = Xte @ W - 0.5 * (means * W.T).sum(1)                        # equal priors
    return score.argmax(1)


DECODERS = ["lda", "logr", "svm"]
C_REG = 1.0
RIDGE_LDA = 1e-2
DECODER_VERSION = "dec-v2"            # bump when a decoder's maths changes (part of 1e's cache key)


def _balanced_weights(ytr, n_cls):
    cnt = np.bincount(ytr, minlength=n_cls).astype(float)
    return (len(ytr) / (n_cls * np.maximum(cnt, 1)))[ytr]


def _newton(loss_fn, grad_hess_fn, x, max_iter=30, tol=1e-9):
    """Damped Newton (backtracking line search) for a smooth convex objective; x is a flat vector."""
    f = loss_fn(x)
    for _ in range(max_iter):
        g, H = grad_hess_fn(x)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = g
        dec = float(g @ step)
        if dec < tol:
            break
        t = 1.0
        while t > 1e-6:
            fn = loss_fn(x - t * step)
            if fn <= f - 1e-4 * t * dec:
                break
            t *= 0.5
        x, f = x - t * step, fn
    return x


def _logr_binary(Xtr, ytr, Xte):
    n, d = Xtr.shape
    Xa = np.hstack([Xtr, np.ones((n, 1))])
    sw = _balanced_weights(ytr, 2)
    pen = np.r_[np.full(d, 1.0 / C_REG), 1e-8]
    y1 = ytr.astype(float)

    def loss(w):
        z = Xa @ w
        return (sw * (np.logaddexp(0.0, z) - y1 * z)).sum() + 0.5 * (pen * w * w).sum()

    def grad_hess(w):
        p = 1.0 / (1.0 + np.exp(-(Xa @ w)))
        g = Xa.T @ (sw * (p - y1)) + pen * w
        H = (Xa.T * (sw * p * (1.0 - p))) @ Xa + np.diag(pen)
        return g, H

    w = _newton(loss, grad_hess, np.zeros(d + 1))
    return ((np.hstack([Xte, np.ones((len(Xte), 1))]) @ w) > 0).astype(int)


def _logr_predict(Xtr, ytr, Xte, n_cls, ridge=None):
    if n_cls == 2:
        return _logr_binary(Xtr, ytr, Xte)
    n, d = Xtr.shape
    Xa = np.hstack([Xtr, np.ones((n, 1))])
    d1 = d + 1
    sw = _balanced_weights(ytr, n_cls)
    pen = np.r_[np.full(d, 1.0 / C_REG), 1e-8]                      # intercept not penalised
    idx = np.arange(n)

    def probs(w):
        Z = Xa @ w.reshape(d1, n_cls)
        Z -= Z.max(1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(1, keepdims=True)

    def loss(w):
        P = probs(w)
        return -(sw * np.log(P[idx, ytr] + 1e-300)).sum() + 0.5 * (pen[:, None] * w.reshape(d1, n_cls) ** 2).sum()

    def grad_hess(w):
        P = probs(w)
        G = P.copy()
        G[idx, ytr] -= 1.0
        G *= sw[:, None]
        g = (Xa.T @ G + pen[:, None] * w.reshape(d1, n_cls)).ravel()
        H = np.empty((d1, n_cls, d1, n_cls))
        for k in range(n_cls):
            for l in range(k, n_cls):
                a = sw * ((P[:, k] if k != l else P[:, k] * (1.0 - P[:, k])) if k == l else -P[:, k] * P[:, l])
                blk = (Xa.T * a) @ Xa
                H[:, k, :, l] = blk
                H[:, l, :, k] = blk
        H = H.reshape(d1 * n_cls, d1 * n_cls)
        H[np.diag_indices_from(H)] += np.repeat(pen, n_cls)
        return g, H

    w = _newton(loss, grad_hess, np.zeros(d1 * n_cls)).reshape(d1, n_cls)
    return (np.hstack([Xte, np.ones((len(Xte), 1))]) @ w).argmax(1)


def _svm_fit_binary(Xa, t, sw):
    d1 = Xa.shape[1]
    pen = np.r_[np.ones(d1 - 1), 1e-8]

    def loss(w):
        a = np.maximum(1.0 - t * (Xa @ w), 0.0)
        return C_REG * (sw * a * a).sum() + 0.5 * (pen * w * w).sum()

    def grad_hess(w):
        m = 1.0 - t * (Xa @ w)
        act = m > 0
        g = -2.0 * C_REG * (Xa[act].T @ (sw[act] * m[act] * t[act])) + pen * w
        H = 2.0 * C_REG * (Xa[act].T * sw[act]) @ Xa[act] + np.diag(pen)
        return g, H

    return _newton(loss, grad_hess, np.zeros(d1))


def _svm_predict(Xtr, ytr, Xte, n_cls, ridge=None):
    Xa = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xe = np.hstack([Xte, np.ones((len(Xte), 1))])
    sw = _balanced_weights(ytr, n_cls)
    if n_cls == 2:
        return ((Xe @ _svm_fit_binary(Xa, np.where(ytr == 1, 1.0, -1.0), sw)) > 0).astype(int)
    return np.column_stack([Xe @ _svm_fit_binary(Xa, np.where(ytr == k, 1.0, -1.0), sw) for k in range(n_cls)]).argmax(1)


_PREDICT = {"lda": _lda_predict, "logr": _logr_predict, "svm": _svm_predict}


def _folds(y, k, rng):
    """Stratified fold index per sample."""
    f = np.empty(len(y), int)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        f[rng.permutation(idx)] = np.arange(len(idx)) % k
    return f


def balanced_accuracy_cv(X, y, n_cls, n_folds, rng, ridge=1e-2, method="lda"):
    f = _folds(y, n_folds, rng)
    pred = np.empty(len(y), int)
    for k in range(n_folds):
        te = f == k
        pred[te] = _PREDICT[method](X[~te], y[~te], X[te], n_cls, ridge)
    return float(np.mean([(pred[y == c] == c).mean() for c in range(n_cls)]))


def decode(X, y, groups=None, n_folds=5, n_repeats=3, n_perm=20, rng=None, ridge=1e-2, method="lda"):
    """X: samples x features (finite), y: labels (any values). Returns dict(acc, null_mean, null_sd, n, n_cls, chance) or None
    when a class has too few samples for 2 folds. acc = mean over `n_repeats` CV repeats; the null = `n_perm` single-repeat CVs
    with labels permuted within `groups`."""
    rng = rng or np.random.default_rng(0)
    classes, yi = np.unique(y, return_inverse=True)
    n_cls = len(classes)
    counts = np.bincount(yi)
    k = int(min(n_folds, counts.min()))
    if n_cls < 2 or k < 2:
        return None
    acc = np.mean([balanced_accuracy_cv(X, yi, n_cls, k, rng, ridge, method) for _ in range(n_repeats)])
    null = []
    for _ in range(n_perm):
        yp = yi.copy()
        if groups is None:
            yp = rng.permutation(yp)
        else:
            for g in np.unique(groups):
                m = np.flatnonzero(groups == g)
                yp[m] = yi[rng.permutation(m)]
        null.append(balanced_accuracy_cv(X, yp, n_cls, k, rng, ridge, method))
    return dict(acc=float(acc), null_mean=float(np.mean(null)) if null else np.nan,
                null_sd=float(np.std(null)) if null else np.nan, n=len(yi), n_cls=n_cls, chance=1.0 / n_cls)
