# -*- coding: utf-8 -*-
"""
temporal_breakdown.py
=======================
Time- (or position-) resolved version of the Pola et al. (2003) pairwise
information breakdown (see breakdown.py for the method itself): for a
given set of simultaneously recorded neuron pairs, compute
I_lin/I_sig_sim/I_cor_indep/I_cor_dep/I_full at every bin of a
(K trials, N neurons, T bins) tensor (see tensor_utils.py).

As in temporal.py, each neuron's bin edges are computed ONCE by pooling
across all time/space bins (not per-bin), so the response alphabet is
fixed over time and apparent changes in the breakdown terms cannot be an
artifact of shifting bin edges (see breakdown.pairwise_information_
breakdown's `edges1`/`edges2` parameters).

This is the most expensive analysis in the pipeline (per pair: T bins x
4 exact mutual-information-derived terms, further multiplied by
`params.bias.breakdown_n_shuffles` + 1 if shuffle-subtraction bias
correction is enabled -- see breakdown.py's module docstring for why
that correction matters even so), so it is meant to be used together
with cache.py: see plot_temporal_breakdown.py for the cached,
per-session driver.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import breakdown as breakdown_mod
from .discretize import equipopulated_edges
from .params import InfoTheoryParams

BREAKDOWN_TERMS = ['I_full', 'I_R1', 'I_R2', 'I_lin', 'I_sig_sim', 'I_cor_indep', 'I_cor_dep']
BREAKDOWN_TERMS_SHUFFCORR = [f'{k}_shuffcorr' for k in BREAKDOWN_TERMS]


def _pair_timecourse_worker(x1_nt, x2_nt, stim, params: InfoTheoryParams, pair_index):
    """
    x1_nt, x2_nt : arrays (K trials, T bins), single-trial responses of
        the two cells across all trials and time/space bins.
    """
    edges1, _ = equipopulated_edges(x1_nt.ravel(), params.binning.n_bins)
    edges2, _ = equipopulated_edges(x2_nt.ravel(), params.binning.n_bins)
    T = x1_nt.shape[1]

    out_keys = list(BREAKDOWN_TERMS)
    if params.bias.breakdown_shuffle_correction:
        out_keys += BREAKDOWN_TERMS_SHUFFCORR
    terms = {k: np.full(T, np.nan) for k in out_keys}

    for t in range(T):
        res = breakdown_mod.pairwise_information_breakdown(
            x1_nt[:, t], x2_nt[:, t], stim,
            n_bins=params.binning.n_bins, binning_method=params.binning.method,
            panzeri_treves=params.bias.panzeri_treves,
            shuffle_correction=params.bias.breakdown_shuffle_correction,
            n_shuffles=params.bias.breakdown_n_shuffles,
            random_state=params.bias.random_state,
            unit_pair_index=pair_index * 1_000_000 + t,
            edges1=edges1, edges2=edges2,
        )
        for k in out_keys:
            terms[k][t] = res[k]

    return pair_index, terms


def compute_time_resolved_breakdown(tensor, stim, pairs, params: InfoTheoryParams = None):
    """
    Parameters
    ----------
    tensor : array (K trials, N neurons, T bins)
    stim : 1D array, length K
    pairs : list of (i, j) neuron-index tuples
    params : InfoTheoryParams. `params.bias.breakdown_shuffle_correction`
        (default True) controls whether the `*_shuffcorr` columns
        (recommended, see breakdown.py) are also computed and returned;
        `params.bias.breakdown_n_shuffles` controls their cost.

    Returns
    -------
    pandas.DataFrame, long format, one row per (pair, bin):
        columns: pair_index, neuron_i, neuron_j, bin_index, the
        BREAKDOWN_TERMS columns, and (if enabled) the
        BREAKDOWN_TERMS_SHUFFCORR columns.
    """
    if params is None:
        params = InfoTheoryParams()

    K, N, T = tensor.shape
    stim = np.asarray(stim)
    assert len(stim) == K

    out_keys = list(BREAKDOWN_TERMS)
    if params.bias.breakdown_shuffle_correction:
        out_keys += BREAKDOWN_TERMS_SHUFFCORR

    results = Parallel(n_jobs=params.parallel.n_jobs,
                        backend=params.parallel.backend,
                        verbose=params.parallel.verbose)(
        delayed(_pair_timecourse_worker)(tensor[:, i, :], tensor[:, j, :], stim, params, pidx)
        for pidx, (i, j) in enumerate(pairs)
    )

    rows = []
    for pair_index, terms in results:
        i, j = pairs[pair_index]
        for t in range(T):
            row = {'pair_index': pair_index, 'neuron_i': i, 'neuron_j': j, 'bin_index': t}
            row.update({k: terms[k][t] for k in out_keys})
            rows.append(row)

    return pd.DataFrame(rows)
