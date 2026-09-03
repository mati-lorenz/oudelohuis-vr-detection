# -*- coding: utf-8 -*-
"""
core.py
========
Core information-theoretic estimators shared by all three analysis stages.

Implements:
    - naive ("plug-in") discrete entropy and mutual information
    - the analytic Panzeri-Treves / Treves-Panzeri limited-sampling bias
      correction (Treves & Panzeri, 1995, Neural Comp.; Panzeri & Treves,
      1996, Network; see also the review in Panzeri, Senatore, Montemurro
      & Petersen, 2007, J. Neurophysiol., and the Information Breakdown
      ToolBox, Magri et al. 2009, BMC Neurosci.)
    - a permutation/shuffle-based null distribution for significance
      testing, which also gives an independent, assumption-free estimate
      of the residual bias (useful as a sanity check on the analytic
      correction, especially for small trial counts)

Notation follows the standard direct-method formulation: a discrete
response variable R (already binned, see discretize.py) and a discrete
"stimulus" variable S (e.g. stimulus category or choice, i.e. any
categorical trial label).

All estimators operate on integer label arrays (0..R-1 for the response,
0..S-1 for the stimulus) so that this module has no dependency on how the
labels were produced upstream (single neuron activity, PCA scores,
population-averaged activity, ...) -- this is what keeps stage 1
(single cell), stage 2 (dimensionality-reduced components) and stage 3
(population code) all built on the same numerical core.
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional


LOG2 = np.log(2)


# --------------------------------------------------------------------------
# Basic counting / probability utilities
# --------------------------------------------------------------------------

def _as_labels(x):
    x = np.asarray(x)
    if x.dtype.kind not in 'iu':
        # map arbitrary hashable labels (e.g. strings) to 0..K-1 integers
        _, x = np.unique(x, return_inverse=True)
    return x.astype(int)


def joint_counts(r, s, n_r=None, n_s=None):
    """Joint occurrence counts n(r, s) as a 2D integer array (n_r, n_s)."""
    r = _as_labels(r)
    s = _as_labels(s)
    if n_r is None:
        n_r = r.max() + 1
    if n_s is None:
        n_s = s.max() + 1
    counts = np.zeros((n_r, n_s), dtype=int)
    np.add.at(counts, (r, s), 1)
    return counts


def _entropy_from_probs(p):
    """Shannon entropy in bits of a probability vector/array (any shape),
    summing over all elements. Zero-probability bins contribute 0."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p) / LOG2))


# --------------------------------------------------------------------------
# Naive (plug-in) entropy & mutual information
# --------------------------------------------------------------------------

def naive_entropy(r):
    """Naive plug-in entropy H(R) in bits."""
    r = _as_labels(r)
    counts = np.bincount(r)
    p = counts / counts.sum()
    return _entropy_from_probs(p)


def naive_mutual_information(r, s):
    """
    Naive plug-in mutual information I(R;S) in bits, computed directly
    from the joint histogram (the "direct method", e.g. Panzeri et al.
    2007). This estimator is biased upward for finite trial counts; use
    `mutual_information` below to obtain the Panzeri-Treves corrected
    estimate.
    """
    r = _as_labels(r)
    s = _as_labels(s)
    counts = joint_counts(r, s)
    n = counts.sum()
    p_rs = counts / n
    p_r = p_rs.sum(axis=1, keepdims=True)
    p_s = p_rs.sum(axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(p_rs > 0, p_rs / (p_r * p_s), 1.0)
        terms = np.where(p_rs > 0, p_rs * np.log(ratio) / LOG2, 0.0)
    return float(terms.sum())


# --------------------------------------------------------------------------
# Panzeri-Treves analytic bias correction
# --------------------------------------------------------------------------

def _relevant_bins(counts_1d):
    """Naive count of response bins with non-zero occupancy (R_naive in
    the Treves-Panzeri formalism)."""
    return int(np.sum(counts_1d > 0))


def panzeri_treves_correction(counts):
    """
    Analytic first-order limited-sampling bias correction for I(R;S),
    following Treves & Panzeri (1995) / Panzeri & Treves (1996): the
    naive (plug-in) entropy of a discrete variable estimated from N
    samples with R relevant (non-empty) bins is biased by
    approximately -(R-1)/(2 N ln 2) bits; propagating this through
    I = H(R) - H(R|S) gives:

        bias(I_naive) ~= (1 / (2 N ln 2)) * [ sum_s (R_s - 1) - (R_tot - 1) ]

    where R_tot is the number of non-empty response bins pooling all
    trials, R_s is the number of non-empty response bins within trials of
    stimulus s, and N is the total number of trials.

    This is the practical, widely used form of the "PT correction" (see
    e.g. the implementation in the Information Breakdown ToolBox, Magri
    et al. 2009). Note it uses the *naive* relevant bin counts; the fully
    Bayesian refinement of R_s described in Panzeri & Treves (1996) can
    give a slightly better estimate in very undersampled regimes and can
    be substituted here if a specific published method needs to be
    matched exactly (e.g. to reproduce Lorenz et al. 2025 figure-by-
    figure) -- swap out `_relevant_bins` for a Bayesian estimator if so.

    Parameters
    ----------
    counts : 2D int array (n_r, n_s), joint response x stimulus counts

    Returns
    -------
    bias : float, in bits (subtract from the naive MI to correct it)
    """
    n_r, n_s = counts.shape
    n_tot = counts.sum()
    r_tot = _relevant_bins(counts.sum(axis=1))

    bias_terms = 0.0
    for si in range(n_s):
        n_s_i = counts[:, si].sum()
        if n_s_i == 0:
            continue
        r_s_i = _relevant_bins(counts[:, si])
        bias_terms += (r_s_i - 1)

    bias = (bias_terms - (r_tot - 1)) / (2.0 * n_tot * LOG2)
    return float(bias)


def mutual_information(r, s, bias_correction=True):
    """
    Mutual information I(R;S) in bits, optionally Panzeri-Treves bias
    corrected.

    Returns
    -------
    dict with keys:
        'I_naive'   : plug-in estimate
        'I_pt'      : PT-corrected estimate (== I_naive if
                       bias_correction=False)
        'bias'      : the subtracted analytic bias term
        'n_trials'  : total trial count used
    """
    r = _as_labels(r)
    s = _as_labels(s)
    counts = joint_counts(r, s)
    n = counts.sum()
    p_rs = counts / n
    p_r = p_rs.sum(axis=1, keepdims=True)
    p_s = p_rs.sum(axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(p_rs > 0, p_rs / (p_r * p_s), 1.0)
        terms = np.where(p_rs > 0, p_rs * np.log(ratio) / LOG2, 0.0)
    i_naive = float(terms.sum())

    bias = panzeri_treves_correction(counts) if bias_correction else 0.0
    i_pt = i_naive - bias

    return {'I_naive': i_naive, 'I_pt': i_pt, 'bias': bias, 'n_trials': int(n)}


# --------------------------------------------------------------------------
# Permutation / shuffle null distribution
# --------------------------------------------------------------------------

def get_worker_rng(random_state, unit_index=0):
    """Deterministic-but-independent RNG stream per parallel worker/unit,
    so that results are reproducible regardless of the number of workers
    or the order jobs happen to run in."""
    seed = None if random_state is None else int(random_state) * 100003 + int(unit_index)
    return np.random.default_rng(seed)


def shuffle_null_distribution(r, s, n_shuffles=500, bias_correction=True,
                               random_state=None, unit_index=0):
    """
    Build a null distribution of MI values by repeatedly shuffling the
    trial-to-stimulus assignment (destroying any true R-S relationship
    while preserving each variable's own marginal statistics and the
    trial count).

    Used both for:
      (a) significance testing: p = fraction of null draws >= observed
      (b) an independent, estimator-free bias estimate: mean(null) should
          be ~0 for a well-corrected (e.g. PT-corrected) estimator, and
          equal to the naive bias if bias_correction=False.

    Parameters
    ----------
    r, s : 1D arrays, single-trial response and stimulus/choice labels
    n_shuffles : int
    bias_correction : bool
        whether the null MI values are themselves PT-corrected (recommended:
        keep this the same as for the observed estimate so that
        `null_mean` is directly interpretable as "residual bias after
        correction", which should be close to zero)
    random_state, unit_index : seeding, see `get_worker_rng`

    Returns
    -------
    null_values : 1D array, length n_shuffles (I_pt or I_naive per shuffle)
    """
    rng = get_worker_rng(random_state, unit_index)
    r = _as_labels(r)
    s = np.asarray(s)
    n = len(s)
    null_values = np.empty(n_shuffles, dtype=float)
    key = 'I_pt' if bias_correction else 'I_naive'
    for i in range(n_shuffles):
        s_shuf = s[rng.permutation(n)]
        res = mutual_information(r, s_shuf, bias_correction=bias_correction)
        null_values[i] = res[key]
    return null_values


def significance_test(observed, null_values, alternative='greater'):
    """
    p-value of an observed statistic against a null distribution.

    alternative='greater': p = (1 + #{null >= observed}) / (1 + n_null)
        (appropriate for MI, which is non-negative and expected to be
        pushed up, not down, by a true R-S relationship). The +1/+1 is
        the standard finite-sample correction (Davison & Hinkley, 1997;
        Phipson & Smyth, 2010) that avoids reporting p = 0.
    alternative='two-sided': symmetric version around the null median.
    """
    null_values = np.asarray(null_values)
    n_null = len(null_values)
    if alternative == 'greater':
        p = (1 + np.sum(null_values >= observed)) / (1 + n_null)
    elif alternative == 'two-sided':
        dev = np.abs(null_values - np.median(null_values))
        obs_dev = np.abs(observed - np.median(null_values))
        p = (1 + np.sum(dev >= obs_dev)) / (1 + n_null)
    else:
        raise ValueError(f'Unknown alternative: {alternative}')
    return float(p)


# --------------------------------------------------------------------------
# One-stop single-unit MI estimate with correction + significance
# --------------------------------------------------------------------------

def mi_with_stats(r, s, n_shuffles=500, bias_correction=True,
                   shuffle_bias_estimate=True, alternative='greater',
                   random_state=None, unit_index=0):
    """
    Convenience wrapper combining `mutual_information`,
    `shuffle_null_distribution` and `significance_test` into a single
    result dictionary -- this is the function called (in parallel, once
    per neuron) by single_cell.py.

    Returns
    -------
    dict with keys:
        I_naive, I_pt, bias, n_trials  (see `mutual_information`)
        I_shuffle_corrected : I_pt (or I_naive) minus the shuffle-based
            bias estimate (mean of the null distribution); reported only
            if shuffle_bias_estimate=True
        null_mean, null_std : summary of the null distribution
        p_value
    """
    res = mutual_information(r, s, bias_correction=bias_correction)
    null_values = shuffle_null_distribution(
        r, s, n_shuffles=n_shuffles, bias_correction=bias_correction,
        random_state=random_state, unit_index=unit_index)

    key = 'I_pt' if bias_correction else 'I_naive'
    res['p_value'] = significance_test(res[key], null_values, alternative=alternative)
    res['null_mean'] = float(np.mean(null_values))
    res['null_std'] = float(np.std(null_values))

    if shuffle_bias_estimate:
        res['I_shuffle_corrected'] = res[key] - res['null_mean']

    return res
