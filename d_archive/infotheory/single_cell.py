# -*- coding: utf-8 -*-
"""
single_cell.py
================
Stage 1 of the information-theoretic pipeline: single-cell (and cell-pair)
information about stimulus and/or choice.

This is the only stage meant to be run standalone for now; stages 2
(dimensionality reduction, signal/noise correlations) and 3 (population
information in reduced dimensions) will import `params.py` and `core.py`
directly and reuse the same estimators, so that e.g. "the PT correction"
or "the null test" only ever needs to be implemented once.

Public entry points
--------------------
- compute_single_cell_information(respmat, stim, choice, params)
    -> per-neuron DataFrame with MI(stim), MI(choice), PT-corrected
       versions, significance, and the stim/choice PID decomposition.
- compute_pairwise_breakdown(respmat, stim, params, pairs=None)
    -> per-pair DataFrame with the Pola et al. (2003) 4-term breakdown.
- compute_pairwise_pid(respmat, stim, choice, params, pairs=None)
    -> per-pair DataFrame with the PID of STIM (and, separately, CHOICE)
       carried by the pair of neurons -- i.e. here the two neurons are
       the PID SOURCES and stim/choice is the TARGET, the mirror image
       of compute_single_cell_information's PID (where the neuron is
       the target and stim/choice are the sources). This is the
       "N=2 PID" analysis in Lorenz et al. (2025)'s Fig 2 (Syn/Red bars
       per neuron pair about a task variable).
- run_session_single_cell_analysis(session, params)
    -> convenience wrapper that pulls stim/choice out of a loaddata
       Session object's trialdata and runs both of the above.

Everything embarrassingly parallel (independent across neurons / pairs)
is dispatched via joblib.Parallel.
"""

import itertools
import numpy as np
import pandas as pd

from joblib import Parallel, delayed

from . import core
from . import pid as pid_mod
from . import breakdown as breakdown_mod
from .discretize import discretize_response
from .params import InfoTheoryParams


# --------------------------------------------------------------------------
# per-neuron worker
# --------------------------------------------------------------------------

def _single_neuron_worker(x, stim, choice, neuron_index, params: InfoTheoryParams):
    """
    Compute stimulus/choice information (with PT correction + shuffle
    significance) and the stimulus-choice PID for a single neuron's
    single-trial response vector x.
    """
    out = {'neuron_index': neuron_index}

    r_binned, edges = discretize_response(
        x, n_bins=params.binning.n_bins, method=params.binning.method)
    out['n_bins_used'] = len(edges) - 1

    # --- stimulus information ---
    res_stim = core.mi_with_stats(
        r_binned, stim,
        n_shuffles=params.bias.n_shuffles,
        bias_correction=params.bias.panzeri_treves,
        shuffle_bias_estimate=params.bias.shuffle_bias_estimate,
        alternative=params.bias.alternative,
        random_state=params.bias.random_state,
        unit_index=neuron_index,
    )
    for k, v in res_stim.items():
        out[f'stim_{k}'] = v

    # --- choice information (optional) ---
    if choice is not None:
        res_choice = core.mi_with_stats(
            r_binned, choice,
            n_shuffles=params.bias.n_shuffles,
            bias_correction=params.bias.panzeri_treves,
            shuffle_bias_estimate=params.bias.shuffle_bias_estimate,
            alternative=params.bias.alternative,
            random_state=params.bias.random_state,
            # offset the seed stream so choice shuffles != stim shuffles
            unit_index=neuron_index + 1_000_000,
        )
        for k, v in res_choice.items():
            out[f'choice_{k}'] = v

        # --- partial information decomposition of stim & choice ---
        pid_res = pid_mod.pid_decomposition(
            r_binned, stim, choice, s1_name='stim', s2_name='choice',
            shuffle_correction=params.bias.pid_shuffle_correction,
            n_shuffles=params.bias.pid_n_shuffles,
            random_state=params.bias.random_state,
            unit_index=neuron_index + 2_000_000,
        )
        for k, v in pid_res.items():
            out[f'pid_{k}'] = v

    return out


def compute_single_cell_information(respmat, stim, choice=None,
                                     params: InfoTheoryParams = None,
                                     neuron_labels=None):
    """
    Compute single-cell information about `stim` (and, if given, `choice`)
    for every neuron in `respmat`, in parallel.

    Parameters
    ----------
    respmat : array (N neurons, K trials) or (K trials, N neurons)
        single-trial response matrix. Orientation is auto-detected by
        comparing to len(stim); pass explicitly-oriented data to avoid
        ambiguity when N == K.
    stim : 1D array, length K, discrete stimulus/task-variable label
    choice : 1D array, length K, discrete choice/outcome label (optional)
    params : InfoTheoryParams (uses defaults if None)
    neuron_labels : optional list of length N to use as the 'neuron_id'
        column instead of a plain 0..N-1 index (e.g. celldata index or
        cell_ID)

    Returns
    -------
    pandas.DataFrame, one row per neuron
    """
    if params is None:
        params = InfoTheoryParams()

    X = np.asarray(respmat, dtype=float)
    stim = np.asarray(stim)
    K = len(stim)

    if X.shape[0] == K and X.shape[1] != K:
        pass  # already (K, N)
    elif X.shape[1] == K and X.shape[0] != K:
        X = X.T  # was (N, K)
    elif X.shape[0] == K and X.shape[1] == K:
        raise ValueError(
            'respmat is square (N==K trials): cannot auto-detect '
            'orientation, please pass an explicitly oriented (K, N) array.')
    else:
        raise ValueError(
            f'respmat shape {np.asarray(respmat).shape} is not compatible '
            f'with {K} trials along either axis.')

    if choice is not None:
        choice = np.asarray(choice)
        assert len(choice) == K, 'choice must have the same length as stim'

    N = X.shape[1]

    results = Parallel(n_jobs=params.parallel.n_jobs,
                        backend=params.parallel.backend,
                        verbose=params.parallel.verbose)(
        delayed(_single_neuron_worker)(X[:, n], stim, choice, n, params)
        for n in range(N)
    )

    df = pd.DataFrame(results)
    if neuron_labels is not None:
        assert len(neuron_labels) == N
        df.insert(0, 'neuron_id', np.asarray(neuron_labels))
    return df


# --------------------------------------------------------------------------
# pairwise breakdown driver
# --------------------------------------------------------------------------

def _pair_worker(x1, x2, s, i, j, pair_index, params: InfoTheoryParams):
    res = breakdown_mod.pairwise_information_breakdown(
        x1, x2, s,
        n_bins=params.binning.n_bins,
        binning_method=params.binning.method,
        panzeri_treves=params.bias.panzeri_treves,
        shuffle_correction=params.bias.breakdown_shuffle_correction,
        n_shuffles=params.bias.breakdown_n_shuffles,
        random_state=params.bias.random_state,
        unit_pair_index=pair_index,
    )
    res['neuron_i'] = i
    res['neuron_j'] = j
    return res


def compute_pairwise_breakdown(respmat, stim, params: InfoTheoryParams = None,
                                pairs=None, max_pairs=None, neuron_labels=None,
                                random_state=None):
    """
    Compute the Pola et al. (2003) pairwise information breakdown for a
    set of simultaneously recorded neuron pairs.

    Parameters
    ----------
    respmat : array, (N, K) or (K, N) -- see compute_single_cell_information
    stim : 1D array, length K
    params : InfoTheoryParams
    pairs : optional list of (i, j) index tuples. If None, ALL pairs are
        used (N*(N-1)/2 -- can be large; consider `max_pairs` for a
        random subsample when N is big, e.g. > ~60 neurons).
    max_pairs : optional int, randomly subsample this many pairs
    neuron_labels : optional, see compute_single_cell_information

    Returns
    -------
    pandas.DataFrame, one row per pair
    """
    if params is None:
        params = InfoTheoryParams()

    X = np.asarray(respmat, dtype=float)
    stim = np.asarray(stim)
    K = len(stim)
    if X.shape[0] == K and X.shape[1] != K:
        pass
    elif X.shape[1] == K and X.shape[0] != K:
        X = X.T
    else:
        raise ValueError('Could not unambiguously orient respmat; pass (K, N).')

    N = X.shape[1]

    if pairs is None:
        pairs = list(itertools.combinations(range(N), 2))

    if max_pairs is not None and len(pairs) > max_pairs:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[k] for k in idx]

    results = Parallel(n_jobs=params.parallel.n_jobs,
                        backend=params.parallel.backend,
                        verbose=params.parallel.verbose)(
        delayed(_pair_worker)(X[:, i], X[:, j], stim, i, j, pidx, params)
        for pidx, (i, j) in enumerate(pairs)
    )

    df = pd.DataFrame(results)
    if neuron_labels is not None:
        neuron_labels = np.asarray(neuron_labels)
        df.insert(0, 'neuron_j_id', neuron_labels[df['neuron_j'].to_numpy()])
        df.insert(0, 'neuron_i_id', neuron_labels[df['neuron_i'].to_numpy()])
    return df


# --------------------------------------------------------------------------
# pairwise PID driver: two neurons as PID SOURCES, stim/choice as TARGET
# --------------------------------------------------------------------------

def _pair_pid_worker(x1, x2, stim, choice, i, j, pair_index, params: InfoTheoryParams):
    """
    PID of stim (and, if given, choice) carried jointly by two neurons'
    responses. Note the role reversal relative to
    `_single_neuron_worker`'s PID call: there, the neuron is the target
    and stim/choice are the two sources; here, stim (or choice) is the
    target and the two NEURONS are the two sources.
    """
    r1_binned, _ = discretize_response(x1, n_bins=params.binning.n_bins, method=params.binning.method)
    r2_binned, _ = discretize_response(x2, n_bins=params.binning.n_bins, method=params.binning.method)

    out = {'neuron_i': i, 'neuron_j': j}

    pid_stim = pid_mod.pid_decomposition(
        stim, r1_binned, r2_binned, s1_name='n1', s2_name='n2',
        shuffle_correction=params.bias.pid_shuffle_correction,
        n_shuffles=params.bias.pid_n_shuffles,
        random_state=params.bias.random_state,
        unit_index=pair_index,
    )
    for k, v in pid_stim.items():
        out[f'stim_pid_{k}'] = v

    if choice is not None:
        pid_choice = pid_mod.pid_decomposition(
            choice, r1_binned, r2_binned, s1_name='n1', s2_name='n2',
            shuffle_correction=params.bias.pid_shuffle_correction,
            n_shuffles=params.bias.pid_n_shuffles,
            random_state=params.bias.random_state,
            unit_index=pair_index + 1_000_000,
        )
        for k, v in pid_choice.items():
            out[f'choice_pid_{k}'] = v

    return out


def compute_pairwise_pid(respmat, stim, choice=None, params: InfoTheoryParams = None,
                          pairs=None, max_pairs=None, neuron_labels=None,
                          random_state=None):
    """
    Compute the PID of stim (and, separately, choice) carried jointly by
    each pair of neurons in `respmat` -- the two neurons are the PID
    SOURCES, stim/choice is the TARGET (mirror image of the PID computed
    in `compute_single_cell_information`). Redundancy/unique/synergy are
    reported for each target separately (columns prefixed `stim_pid_`
    and, if `choice` is given, `choice_pid_`), from the SAME pairs and
    SAME response binning, so the two decompositions are directly
    comparable pair-by-pair.

    Parameters
    ----------
    respmat : array, (N, K) or (K, N) -- see compute_single_cell_information
    stim : 1D array, length K
    choice : 1D array, length K, optional (if None, only the stim-target
        PID is computed)
    params : InfoTheoryParams
    pairs : optional list of (i, j) index tuples. If None, ALL pairs are
        used (see `compute_pairwise_breakdown` for the same caveat about
        N*(N-1)/2 growing large)
    max_pairs : optional int, randomly subsample this many pairs
    neuron_labels : optional, see compute_single_cell_information

    Returns
    -------
    pandas.DataFrame, one row per pair
    """
    if params is None:
        params = InfoTheoryParams()

    X = np.asarray(respmat, dtype=float)
    stim = np.asarray(stim)
    K = len(stim)
    if X.shape[0] == K and X.shape[1] != K:
        pass
    elif X.shape[1] == K and X.shape[0] != K:
        X = X.T
    else:
        raise ValueError('Could not unambiguously orient respmat; pass (K, N).')

    if choice is not None:
        choice = np.asarray(choice)
        assert len(choice) == K, 'choice must have the same length as stim'

    N = X.shape[1]

    if pairs is None:
        pairs = list(itertools.combinations(range(N), 2))

    if max_pairs is not None and len(pairs) > max_pairs:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[k] for k in idx]

    results = Parallel(n_jobs=params.parallel.n_jobs,
                        backend=params.parallel.backend,
                        verbose=params.parallel.verbose)(
        delayed(_pair_pid_worker)(X[:, i], X[:, j], stim, choice, i, j, pidx, params)
        for pidx, (i, j) in enumerate(pairs)
    )

    df = pd.DataFrame(results)
    if neuron_labels is not None:
        neuron_labels = np.asarray(neuron_labels)
        df.insert(0, 'neuron_j_id', neuron_labels[df['neuron_j'].to_numpy()])
        df.insert(0, 'neuron_i_id', neuron_labels[df['neuron_i'].to_numpy()])
    return df


# --------------------------------------------------------------------------
# convenience wrapper around a loaddata Session
# --------------------------------------------------------------------------

def get_trial_labels(session, params: InfoTheoryParams):
    """
    Pull stimulus and choice trial labels (and, optionally, a boolean
    trial mask) out of a loaddata Session's `trialdata`, according to
    `params.stim_var` / `params.choice_var` / `params.trial_mask_var`.

    If `params.stim_value_map` or `params.stim_binarize_threshold` is
    set, the raw `stim_var` values are collapsed/binarized accordingly
    before being returned -- see the docstrings in params.py. This is
    how you get, e.g., a binary "signal present vs not" stimulus
    variable out of a 3-category 'stimcat' column ('C'/'N'/'M') or out
    of a continuous 'signal' strength column, without needing a
    separate trialdata column.

    Returns
    -------
    stim : 1D array
    choice : 1D array or None
    mask : 1D boolean array, length == n_trials (all True if no mask
        variable is configured)
    """
    trialdata = session.trialdata
    stim_raw = trialdata[params.stim_var].to_numpy()

    if params.stim_value_map is not None:
        stim = np.array([params.stim_value_map.get(v, v) for v in stim_raw])
    elif params.stim_binarize_threshold is not None:
        stim = (stim_raw.astype(float) > params.stim_binarize_threshold).astype(int)
    else:
        stim = stim_raw

    choice = trialdata[params.choice_var].to_numpy() if params.choice_var else None

    if params.trial_mask_var is not None:
        mask = trialdata[params.trial_mask_var].to_numpy().astype(bool)
    else:
        mask = np.ones(len(trialdata), dtype=bool)

    return stim, choice, mask


def run_session_single_cell_analysis(session, params: InfoTheoryParams = None,
                                      do_pairwise=False, max_pairs=2000):
    """
    Run the full stage-1 analysis (single-cell info + optional pairwise
    breakdown) for one already-loaded Session object (see
    loaddata/session.py; the session must have `respmat` computed, e.g.
    via `session.load_respmat(...)`).

    Returns
    -------
    dict with keys 'single_cell' (DataFrame) and, if do_pairwise=True,
    'pairwise' (DataFrame). Both carry a 'session_id' column for easy
    concatenation across sessions.
    """
    if params is None:
        params = InfoTheoryParams()

    assert hasattr(session, 'respmat'), (
        'session.respmat not found -- call session.load_respmat(...) first')

    stim, choice, mask = get_trial_labels(session, params)

    respmat = np.asarray(session.respmat)  # (N neurons, K trials)
    respmat = respmat[:, mask]
    stim = stim[mask]
    choice = choice[mask] if choice is not None else None

    neuron_labels = (session.celldata.index.to_numpy()
                      if hasattr(session, 'celldata') else None)

    df_sc = compute_single_cell_information(
        respmat, stim, choice, params=params, neuron_labels=neuron_labels)
    df_sc.insert(0, 'session_id', session.session_id)

    out = {'single_cell': df_sc}

    if do_pairwise:
        df_pw = compute_pairwise_breakdown(
            respmat, stim, params=params, max_pairs=max_pairs,
            neuron_labels=neuron_labels, random_state=params.bias.random_state)
        df_pw.insert(0, 'session_id', session.session_id)
        out['pairwise'] = df_pw

    return out
