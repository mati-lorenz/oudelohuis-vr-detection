# -*- coding: utf-8 -*-
"""
temporal_pairwise_pid.py
===========================
Time- (or position-) resolved version of the pairwise-neuron PID (see
single_cell.compute_pairwise_pid for the static, full-window version):
for a given set of simultaneously recorded neuron pairs, at every bin of
a (K trials, N neurons, T bins) tensor (see tensor_utils.py), decompose
the information about STIMULUS -- and, separately, about CHOICE --
carried jointly by the pair into redundant / unique-to-neuron-1 /
unique-to-neuron-2 / synergistic components. Here the two neurons are
the PID SOURCES and stim/choice is the TARGET, exactly as in
single_cell.compute_pairwise_pid; this module only adds the time/space
axis.

As in temporal_pid.py and temporal_breakdown.py, each neuron's bin
edges are computed ONCE by pooling across all time/space bins (not
per-bin), so the response alphabet is fixed over time.

This is the single most expensive analysis in the pipeline: per pair,
per bin, it runs the PID machinery TWICE (once for stim, once for
choice), each optionally with `params.bias.pid_n_shuffles` shuffle-
correction resamples. Cost scales as
    n_pairs * T * 2 * (1 + pid_n_shuffles)
so reduce `max_pairs_per_group`, the number of time/space bins, or
`pid_n_shuffles` for a first exploratory pass -- and use the cache
in plot_temporal_pairwise_pid.py.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import pid as pid_mod
from .discretize import equipopulated_edges, digitize
from .params import InfoTheoryParams

PID_TERMS = ['I_total', 'I_n1', 'I_n2', 'redundancy', 'unique_n1', 'unique_n2', 'synergy']


def _pair_pid_timecourse_worker(x1_nt, x2_nt, stim, choice, params: InfoTheoryParams, pair_index):
    """
    x1_nt, x2_nt : arrays (K trials, T bins), single-trial responses of
        the two cells across all trials and time/space bins.
    """
    edges1, _ = equipopulated_edges(x1_nt.ravel(), params.binning.n_bins)
    edges2, _ = equipopulated_edges(x2_nt.ravel(), params.binning.n_bins)
    T = x1_nt.shape[1]

    out_keys = list(PID_TERMS)
    if params.bias.pid_shuffle_correction:
        out_keys += [f'{k}_shuffcorr' for k in PID_TERMS]

    stim_terms = {k: np.full(T, np.nan) for k in out_keys}
    choice_terms = {k: np.full(T, np.nan) for k in out_keys} if choice is not None else None

    for t in range(T):
        r1 = digitize(x1_nt[:, t], edges1)
        r2 = digitize(x2_nt[:, t], edges2)

        res_stim = pid_mod.pid_decomposition(
            stim, r1, r2, s1_name='n1', s2_name='n2',
            shuffle_correction=params.bias.pid_shuffle_correction,
            n_shuffles=params.bias.pid_n_shuffles,
            random_state=params.bias.random_state,
            unit_index=pair_index * 1_000_000 + t,
        )
        for k in out_keys:
            stim_terms[k][t] = res_stim[k]

        if choice is not None:
            res_choice = pid_mod.pid_decomposition(
                choice, r1, r2, s1_name='n1', s2_name='n2',
                shuffle_correction=params.bias.pid_shuffle_correction,
                n_shuffles=params.bias.pid_n_shuffles,
                random_state=params.bias.random_state,
                unit_index=pair_index * 1_000_000 + t + 500_000,
            )
            for k in out_keys:
                choice_terms[k][t] = res_choice[k]

    return pair_index, stim_terms, choice_terms


def compute_time_resolved_pairwise_pid(tensor, stim, choice, pairs, params: InfoTheoryParams = None):
    """
    Parameters
    ----------
    tensor : array (K trials, N neurons, T bins)
    stim : 1D array, length K
    choice : 1D array, length K, or None (if None, only the stim-target
        PID is computed)
    pairs : list of (i, j) neuron-index tuples
    params : InfoTheoryParams

    Returns
    -------
    pandas.DataFrame, long format, one row per (pair, bin):
        columns: pair_index, neuron_i, neuron_j, bin_index, and the PID
        term columns prefixed `stim_pid_` and (if choice is given)
        `choice_pid_` -- matching single_cell.compute_pairwise_pid's
        column naming, with the time/space axis added.
    """
    if params is None:
        params = InfoTheoryParams()

    K, N, T = tensor.shape
    stim = np.asarray(stim)
    assert len(stim) == K
    if choice is not None:
        choice = np.asarray(choice)
        assert len(choice) == K

    out_keys = list(PID_TERMS)
    if params.bias.pid_shuffle_correction:
        out_keys += [f'{k}_shuffcorr' for k in PID_TERMS]

    results = Parallel(n_jobs=params.parallel.n_jobs,
                        backend=params.parallel.backend,
                        verbose=params.parallel.verbose)(
        delayed(_pair_pid_timecourse_worker)(
            tensor[:, i, :], tensor[:, j, :], stim, choice, params, pidx)
        for pidx, (i, j) in enumerate(pairs)
    )

    rows = []
    for pair_index, stim_terms, choice_terms in results:
        i, j = pairs[pair_index]
        for t in range(T):
            row = {'pair_index': pair_index, 'neuron_i': i, 'neuron_j': j, 'bin_index': t}
            row.update({f'stim_pid_{k}': stim_terms[k][t] for k in out_keys})
            if choice_terms is not None:
                row.update({f'choice_pid_{k}': choice_terms[k][t] for k in out_keys})
            rows.append(row)

    return pd.DataFrame(rows)
