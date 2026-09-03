# -*- coding: utf-8 -*-
"""
temporal_pid.py
=================
Time- (or position-) resolved partial information decomposition (PID):
for every neuron, decompose the information carried about stimulus and
choice jointly into redundant / unique-to-stim / unique-to-choice /
synergistic components, at every bin of a (K trials, N neurons, T bins)
tensor (see tensor_utils.py).

As in temporal.py, each neuron's bin edges are computed ONCE by pooling
across all time/space bins (not per-bin), so the response alphabet is
fixed over time and apparent changes in the PID terms over time cannot
be an artifact of shifting bin edges.

Shuffle-subtraction bias correction (see pid.py's module docstring and
tests/test_pid_canonical.py) is applied by default
(`params.bias.pid_shuffle_correction`), since naive PID terms carry the
same kind of limited-sampling bias as the pairwise information
breakdown -- important here because per-bin trial counts are the same
as for the full-window analysis, but responses are noisier (a single
time/space bin instead of an averaged response window), so the naive
bias problem is, if anything, more acute.

This is a more expensive analysis than temporal.py's single-variable
MI time-course (it computes 3 mutual informations plus a redundancy
optimization per neuron per bin, times `params.bias.pid_n_shuffles`+1
if shuffle correction is on), so consider a coarser time/space bin size
or fewer shuffles for a first exploratory pass -- see
plot_temporal_pid.py for a cached, per-session driver.
"""

import numpy as np
from joblib import Parallel, delayed

from . import pid as pid_mod
from .discretize import equipopulated_edges, digitize
from .params import InfoTheoryParams

#: canonical term names returned per neuron/bin (see pid.pid_decomposition,
#: called here with s1_name='stim', s2_name='choice')
PID_TERMS = ['I_total', 'I_stim', 'I_choice', 'redundancy',
             'unique_stim', 'unique_choice', 'synergy']


def _neuron_pid_timecourse_worker(x_nt, stim, choice, params: InfoTheoryParams, neuron_index):
    """
    x_nt : array (K trials, T bins), single neuron's responses across all
        trials and time/space bins.

    Returns
    -------
    neuron_index : int
    terms : dict {term_name: array of length T}, term_name in PID_TERMS
        (plus '<term>_shuffcorr' for each, if
        params.bias.pid_shuffle_correction is True)
    """
    edges, _ = equipopulated_edges(x_nt.ravel(), params.binning.n_bins)
    T = x_nt.shape[1]

    out_keys = list(PID_TERMS)
    if params.bias.pid_shuffle_correction:
        out_keys += [f'{k}_shuffcorr' for k in PID_TERMS]
    terms = {k: np.full(T, np.nan) for k in out_keys}

    for t in range(T):
        r = digitize(x_nt[:, t], edges)
        res = pid_mod.pid_decomposition(
            r, stim, choice, s1_name='stim', s2_name='choice',
            shuffle_correction=params.bias.pid_shuffle_correction,
            n_shuffles=params.bias.pid_n_shuffles,
            random_state=params.bias.random_state,
            unit_index=neuron_index * 1_000_000 + t,
        )
        for k in out_keys:
            terms[k][t] = res[k]

    return neuron_index, terms


def compute_time_resolved_pid(tensor, stim, choice, params: InfoTheoryParams = None):
    """
    Parameters
    ----------
    tensor : array (K trials, N neurons, T bins) -- see
        tensor_utils.compute_tensor_for_session
    stim, choice : 1D arrays, length K (both required -- PID needs two
        source variables; for single-variable time courses use
        temporal.compute_time_resolved_information instead)
    params : InfoTheoryParams. `params.bias.pid_shuffle_correction`
        (default True) controls whether the `*_shuffcorr` terms
        (recommended, see pid.py) are also computed; `params.bias.
        pid_n_shuffles` controls their cost.

    Returns
    -------
    dict {term_name: array (N, T)}, one entry per PID term (plus
    shuffle-corrected versions if enabled) -- e.g. result['redundancy']
    is an (N, T) array, result['synergy_shuffcorr'] likewise.
    """
    if params is None:
        params = InfoTheoryParams()

    K, N, T = tensor.shape
    stim = np.asarray(stim)
    choice = np.asarray(choice)
    assert len(stim) == K and len(choice) == K, (
        'PID requires both stim and choice, each of length K (n_trials)')

    results = Parallel(n_jobs=params.parallel.n_jobs,
                        backend=params.parallel.backend,
                        verbose=params.parallel.verbose)(
        delayed(_neuron_pid_timecourse_worker)(tensor[:, n, :], stim, choice, params, n)
        for n in range(N)
    )

    out_keys = list(PID_TERMS)
    if params.bias.pid_shuffle_correction:
        out_keys += [f'{k}_shuffcorr' for k in PID_TERMS]

    out = {k: np.full((N, T), np.nan) for k in out_keys}
    for neuron_index, terms in results:
        for k in out_keys:
            out[k][neuron_index, :] = terms[k]

    return out
