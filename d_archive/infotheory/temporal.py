# -*- coding: utf-8 -*-
"""
temporal.py
============
Time- (or position-) resolved single-cell information, i.e. MI(stim) and
MI(choice) computed independently at every time/space bin of a
(K trials, N neurons, T bins) tensor (see tensor_utils.py).

This intentionally skips the permutation null test used in
single_cell.py's per-session summary (that would cost
N neurons x T bins x n_shuffles MI evaluations, which gets expensive fast
for populations of thousands of neurons) -- it's meant for exploratory
time-course plots of PT-corrected information, grouped/averaged by area
and label, where uncertainty is instead visualized via the neuron-to-
neuron SEM within each group (see plot_temporal_information.py). If you
need a rigorous significance test at a specific time bin, pull that time
slice out and run it through `core.mi_with_stats` directly.

Response discretization uses ONE set of bin edges per neuron, computed by
pooling values across ALL time/space bins and trials together, so that
the response "alphabet" (and thus the number of possible bins) stays
fixed over time -- this is important, otherwise apparent changes in
information over time could simply reflect the bin edges shifting with
e.g. baseline drift.
"""

import numpy as np
from joblib import Parallel, delayed

from . import core
from .discretize import equipopulated_edges, digitize
from .params import InfoTheoryParams


def _neuron_timecourse_worker(x_nt, stim, choice, params: InfoTheoryParams, neuron_index):
    """
    x_nt : array (K trials, T bins), single neuron's responses across all
        trials and time/space bins.
    """
    edges, _ = equipopulated_edges(x_nt.ravel(), params.binning.n_bins)
    T = x_nt.shape[1]

    stim_mi = np.full(T, np.nan)
    choice_mi = np.full(T, np.nan) if choice is not None else None

    for t in range(T):
        r = digitize(x_nt[:, t], edges)
        res = core.mutual_information(r, stim, bias_correction=params.bias.panzeri_treves)
        stim_mi[t] = res['I_pt']
        if choice is not None:
            resc = core.mutual_information(r, choice, bias_correction=params.bias.panzeri_treves)
            choice_mi[t] = resc['I_pt']

    return neuron_index, stim_mi, choice_mi


def compute_time_resolved_information(tensor, stim, choice=None,
                                       params: InfoTheoryParams = None):
    """
    Parameters
    ----------
    tensor : array (K trials, N neurons, T bins) -- see
        tensor_utils.compute_tensor_for_session
    stim : 1D array, length K
    choice : 1D array, length K, optional
    params : InfoTheoryParams (uses defaults if None; only
        params.binning.n_bins/method and params.bias.panzeri_treves and
        params.parallel are used here -- the shuffle/null settings are
        not used, see module docstring)

    Returns
    -------
    stim_mi : array (N, T), PT-corrected MI(stim) per neuron per bin
    choice_mi : array (N, T) or None
    """
    if params is None:
        params = InfoTheoryParams()

    K, N, T = tensor.shape
    stim = np.asarray(stim)
    assert len(stim) == K
    if choice is not None:
        choice = np.asarray(choice)
        assert len(choice) == K

    results = Parallel(n_jobs=params.parallel.n_jobs,
                        backend=params.parallel.backend,
                        verbose=params.parallel.verbose)(
        delayed(_neuron_timecourse_worker)(tensor[:, n, :], stim, choice, params, n)
        for n in range(N)
    )

    stim_mi = np.full((N, T), np.nan)
    choice_mi = np.full((N, T), np.nan) if choice is not None else None
    for n, s_mi, c_mi in results:
        stim_mi[n, :] = s_mi
        if choice is not None:
            choice_mi[n, :] = c_mi

    return stim_mi, choice_mi
