# -*- coding: utf-8 -*-
"""
behavior_mi.py
================
Mutual information between each neuron's continuous, session-long
deconvolved trace and a continuous behavioral/state variable (position,
running speed, pupil area, face-video PC1 -- see behavior_signals.py),
computed over the WHOLE recording (not trial-windowed) since these are
continuously-varying state variables rather than trial-locked events.

This is the information-theoretic upgrade of the simple Pearson
correlation used for the same purpose in spike_stats.runspeed_
correlation: mutual information additionally captures nonlinear and
non-monotonic relationships (e.g. a neuron that fires for BOTH very
slow and very fast running, or a U-shaped pupil-size tuning curve,
would show near-zero Pearson correlation but nonzero MI).

Both the neural trace and the behavioral trace are discretized with
the SAME equipopulated-binning machinery used throughout this package
(discretize.py) -- but not necessarily the same NUMBER of bins: the
behavioral target can be binned more finely than the per-neuron
responses (see compute_behavior_mi's target_n_bins/target_method), since
it's a single, densely-sampled whole-session trace rather than a
per-trial, per-neuron response with a much smaller effective sample
size. MI itself is computed with the Panzeri-Treves bias correction
(core.py) -- consistent with every other MI estimate in the pipeline.

Always parallelized across neurons via joblib (per the project's
convention now).
"""

import numpy as np
from joblib import Parallel, delayed

from . import core
from .discretize import discretize_response
from .params import InfoTheoryParams


def _neuron_worker(x, target_binned, params: InfoTheoryParams, neuron_index, compute_significance):
    r_binned, edges = discretize_response(x, n_bins=params.binning.n_bins, method=params.binning.method)
    if compute_significance:
        res = core.mi_with_stats(
            r_binned, target_binned,
            n_shuffles=params.bias.n_shuffles,
            bias_correction=params.bias.panzeri_treves,
            shuffle_bias_estimate=params.bias.shuffle_bias_estimate,
            alternative=params.bias.alternative,
            random_state=params.bias.random_state,
            unit_index=neuron_index,
        )
    else:
        res = core.mutual_information(r_binned, target_binned, bias_correction=params.bias.panzeri_treves)
    res['neuron_index'] = neuron_index
    res['n_bins_used'] = len(edges) - 1
    return res


def compute_behavior_mi(calciumdata, behavior_trace, params: InfoTheoryParams = None,
                         compute_significance=False, target_n_bins=None, target_method=None):
    """
    MI between every neuron's continuous trace and one continuous
    behavioral trace, parallelized across neurons.

    Parameters
    ----------
    calciumdata : array or DataFrame (T samples, N neurons)
    behavior_trace : 1D array, length T -- same timestamps as
        calciumdata (see behavior_signals.get_behavior_trace)
    params : InfoTheoryParams (uses defaults if None). Note:
        params.parallel controls the joblib settings used here. Each
        NEURON's response is always binned with params.binning (n_bins,
        method) -- see target_n_bins/target_method below to bin the
        behavioral TARGET differently.
    compute_significance : bool
        if True, also run the shuffle-based null test (core.mi_with_
        stats) for a p-value -- more expensive (params.bias.n_shuffles
        extra evaluations per neuron); off by default since this
        analysis is typically run for MANY neurons across the whole
        population as an exploratory pass.
    target_n_bins, target_method : optional int / str
        override params.binning.n_bins / params.binning.method for the
        BEHAVIORAL TARGET only -- the per-neuron response binning is
        unaffected. Use this for variables like position or running
        speed that are smoothly-varying and densely sampled (one value
        per imaging frame over a whole session, so there's no small-
        trial-count penalty to a finer binning the way there is for
        per-trial neural responses elsewhere in this pipeline) and
        benefit from more bins than the coarser binning appropriate for
        single-neuron responses. Defaults to params.binning.n_bins /
        params.binning.method (i.e. no change) if left as None -- see
        plot_mi_behavior.py's `target_n_bins` config dict for a
        per-behavioral-variable example.

    Returns
    -------
    pandas.DataFrame, one row per neuron, columns include at least
    'neuron_index', 'I_naive', 'I_pt', 'bias', 'n_trials' (here:
    n_samples), 'n_bins_used' (neuron response bins),
    'n_target_bins_used' (behavioral target bins, constant across rows),
    and (if compute_significance) 'p_value', 'null_mean', 'null_std',
    'I_shuffle_corrected'.
    """
    import pandas as pd

    if params is None:
        params = InfoTheoryParams()

    X = np.asarray(calciumdata, dtype=float)
    behavior_trace = np.asarray(behavior_trace, dtype=float)
    T, N = X.shape
    assert len(behavior_trace) == T, 'behavior_trace must have the same length as calciumdata'

    target_binned, target_edges = discretize_response(
        behavior_trace,
        n_bins=target_n_bins if target_n_bins is not None else params.binning.n_bins,
        method=target_method if target_method is not None else params.binning.method)

    results = Parallel(n_jobs=params.parallel.n_jobs,
                        backend=params.parallel.backend,
                        verbose=params.parallel.verbose)(
        delayed(_neuron_worker)(X[:, n], target_binned, params, n, compute_significance)
        for n in range(N)
    )

    df = pd.DataFrame(results)
    df['n_target_bins_used'] = len(target_edges) - 1  # constant across neurons; reported per-row
                                                        # for transparency alongside n_bins_used
                                                        # (the per-neuron response bin count)
    return df
