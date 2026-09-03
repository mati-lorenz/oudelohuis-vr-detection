# -*- coding: utf-8 -*-
"""
discretize.py
==============
Binning of continuous single-trial neural responses (e.g. mean deconvolved
activity in a response window, one value per trial per neuron) into a small
number of discrete symbols, as required by the "direct method" of mutual
information estimation used throughout this package (Panzeri lab style,
e.g. Panzeri et al. 2007, Ince et al. 2009).

Equipopulated ("equal-occupancy") binning is the default: bin edges are
placed at quantiles of the pooled response distribution so that every bin
contains (approximately) the same number of trials. This keeps the naive
sampling bias of the MI estimator as small and as uniform as possible
across neurons with different firing statistics, which is important both
for the analytic bias correction (core.py) and for comparing information
values across cells.
"""

import numpy as np


def choose_n_bins(n_trials_per_class, trials_per_bin_target=8, min_bins=2, max_bins=8):
    """
    Heuristic automatic choice of the number of discretization bins, based
    on the least-sampled stimulus/choice class, so that the average number
    of trials per (bin x class) cell stays around `trials_per_bin_target`.

    Parameters
    ----------
    n_trials_per_class : array-like
        number of trials available for each class of the variable(s) you
        will condition on (e.g. per stimulus, per choice, or per
        stimulus-choice combination if you plan on conditioning on both)
    trials_per_bin_target : int
        desired average trial count per bin per class
    min_bins, max_bins : int
        hard bounds on the returned number of bins

    Returns
    -------
    n_bins : int
    """
    n_min = np.min(n_trials_per_class)
    n_bins = int(np.floor(n_min / trials_per_bin_target))
    return int(np.clip(n_bins, min_bins, max_bins))


def equipopulated_edges(x, n_bins):
    """
    Compute bin edges such that each bin contains (as close as possible to)
    an equal number of samples of x.

    Ties (e.g. many identical zero values, common for deconvolved calcium
    "spike" estimates) are handled by using `np.unique` on the quantile
    edges; if this collapses the number of distinct edges below
    n_bins + 1 (i.e. some bins would be empty), n_bins is reduced
    accordingly and a warning value is returned via the second output.

    Parameters
    ----------
    x : 1D array
    n_bins : int

    Returns
    -------
    edges : 1D array, len = n_actual_bins + 1
    n_actual_bins : int
        may be smaller than the requested n_bins if the data has heavy
        ties (e.g. many zeros)
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(x, quantiles)
    edges = np.unique(edges)
    n_actual_bins = len(edges) - 1
    if n_actual_bins < 1:
        # degenerate: constant response, everything falls in one bin
        lo, hi = np.min(x), np.max(x)
        edges = np.array([lo - 1e-9, hi + 1e-9])
        n_actual_bins = 1
    return edges, n_actual_bins


def digitize(x, edges):
    """
    Assign each sample of x to a bin defined by `edges` (as returned by
    `equipopulated_edges` or `np.histogram_bin_edges`).

    Returns 0-indexed integer bin labels in [0, n_bins - 1]. The rightmost
    edge is inclusive (matches np.digitize with right=False semantics on
    an edge set that already covers the data range).
    """
    x = np.asarray(x, dtype=float)
    n_bins = len(edges) - 1
    labels = np.digitize(x, edges[1:-1], right=False)
    return np.clip(labels, 0, n_bins - 1)


def discretize_response(x, n_bins=3, method='equipopulated', edges=None):
    """
    Convenience wrapper: discretize a 1D array of single-trial responses.

    Parameters
    ----------
    x : 1D array, single-trial responses for one neuron (or one component,
        e.g. a PC score, in later pipeline stages)
    n_bins : int
        requested number of bins (may be reduced in case of ties, see
        `equipopulated_edges`)
    method : {'equipopulated', 'equal_width'}
    edges : precomputed bin edges (optional). If given, `n_bins`/`method`
        are ignored and these edges are reused -- this is how you should
        discretize held-out/surrogate data with the SAME bins as the
        original data (important for the breakdown/PID surrogates in
        breakdown.py, so that all information quantities being compared
        use an identical response alphabet).

    Returns
    -------
    labels : 1D int array, same length as x
    edges : the edges actually used
    """
    x = np.asarray(x, dtype=float)
    if edges is None:
        if method == 'equipopulated':
            edges, _ = equipopulated_edges(x, n_bins)
        elif method == 'equal_width':
            # equipopulated_edges (above) already drops NaNs before computing
            # edges from quantiles; np.histogram_bin_edges does NOT do this on
            # its own -- a single NaN makes its internal min/max computation
            # return NaN, which raises "autodetected range ... is not finite"
            # (seen in practice on pupil_area/video_pc1 traces, which have NaN
            # frames from blinks/tracking dropouts). Mirror equipopulated's
            # NaN-dropping here so both methods behave the same way on data
            # with missing samples.
            finite_x = x[~np.isnan(x)]
            if finite_x.size == 0:
                raise ValueError(
                    'discretize_response(method="equal_width"): input is all-NaN, '
                    'cannot compute bin edges. Check the upstream trace (e.g. a '
                    'pupil_area/video_pc1 session with no valid tracking).')
            edges = np.histogram_bin_edges(finite_x, bins=n_bins)
        else:
            raise ValueError(f'Unknown binning method: {method}')
    labels = digitize(x, edges)
    return labels, edges


def discretize_dataframe(respmat, n_bins=3, method='equipopulated'):
    """
    Discretize a full (K trials x N neurons) response matrix, one neuron
    (column) at a time, each with its own bin edges.

    Parameters
    ----------
    respmat : array (K, N) or DataFrame

    Returns
    -------
    labels : int array (K, N)
    edges_list : list of length N, bin edges used for each neuron
    """
    X = np.asarray(respmat, dtype=float)
    K, N = X.shape
    labels = np.zeros((K, N), dtype=int)
    edges_list = []
    for n in range(N):
        lab, edges = discretize_response(X[:, n], n_bins=n_bins, method=method)
        labels[:, n] = lab
        edges_list.append(edges)
    return labels, edges_list
