# -*- coding: utf-8 -*-
"""
tensor_utils.py
=================
Protocol-aware helper for getting a (K trials, N neurons, T time/space
bins) tensor out of a loaddata Session object, for the time/position
resolved information analysis (temporal.py).

Mirrors session_utils.py's dispatch logic for the respmat case:
    - time-locked protocols ('IM', 'GR', 'GN') -> Session.load_tensor
      (utils.psth.compute_tensor), response axis = time relative to
      stimulus/trial onset.
    - spatial/VR protocols ('VR', 'DM', 'DN', 'DP') -> compute_tensor_space
      (as used in loaddata.session_info.load_neural_performing_sessions),
      response axis = position (cm) relative to the stimulus zone.

The two underlying functions return their axes in different orders
(see the docstrings in loaddata/session.py and loaddata/session_info.py:
compute_tensor -> "N neurons by K trials by T time bins", while
compute_tensor_space -> "K trials by N neurons by S spatial bins"), so
this module auto-detects the actual axis order by matching array
dimensions against the known number of trials (K, from trialdata) and
neurons (N, from celldata) and returns a CANONICAL (K, N, T) array
regardless of protocol, so temporal.py never has to think about it.
"""

import numpy as np
from utils.psth import compute_tensor_space

TIME_LOCKED_PROTOCOLS = ('IM', 'GR', 'GN')
SPATIAL_PROTOCOLS = ('VR', 'DM', 'DN', 'DP')


def _orient_to_KNT(tensor, n_trials, n_neurons):
    """
    Transpose a 3D tensor of unknown axis order to canonical
    (K trials, N neurons, T bins), by matching axis lengths against the
    known K and N. Raises a clear error if this is ambiguous (e.g.
    K == N) or inconsistent with the tensor's shape.
    """
    tensor = np.asarray(tensor)
    if tensor.ndim != 3:
        raise ValueError(f'Expected a 3D tensor, got shape {tensor.shape}')
    shape = tensor.shape

    k_axis_candidates = [ax for ax, n in enumerate(shape) if n == n_trials]
    n_axis_candidates = [ax for ax, n in enumerate(shape) if n == n_neurons]

    if not k_axis_candidates or not n_axis_candidates:
        raise ValueError(
            f'Could not match tensor shape {shape} to n_trials={n_trials}, '
            f'n_neurons={n_neurons}. Check that the tensor was computed '
            f'for this exact session/trial selection.')

    k_axis = k_axis_candidates[0]
    n_axis = next((ax for ax in n_axis_candidates if ax != k_axis), None)
    if n_axis is None:
        raise ValueError(
            f'Ambiguous tensor axes for shape {shape} with n_trials='
            f'{n_trials} == n_neurons={n_neurons}; cannot auto-orient. '
            f'Pass the tensor axis order explicitly.')

    t_axis = [ax for ax in range(3) if ax not in (k_axis, n_axis)][0]
    return np.transpose(tensor, (k_axis, n_axis, t_axis))


def compute_tensor_for_session(session, calciumversion='deconv',
                                load_videodata=False, filter_hp=None,
                                keepraw=False,
                                t_pre=-1, t_post=2,
                                s_pre=-60, s_post=80, binsize=10):
    """
    Ensure a canonical (K, N, T) response tensor is available for
    `session`, dispatching on `session.protocol`.

    Parameters
    ----------
    t_pre, t_post : float, seconds
        time window for TIME_LOCKED_PROTOCOLS (forwarded to
        Session.load_tensor / utils.psth.compute_tensor)
    s_pre, s_post, binsize : float, cm
        spatial window & bin size for SPATIAL_PROTOCOLS (forwarded to
        utils.psth.compute_tensor_space, matching
        load_neural_performing_sessions's defaults)

    Returns
    -------
    tensor : array (K, N, T)
    axis : 1D array, length T -- time (s) or position (cm) bin centers
    axis_label : str, for plot x-axis labels
    """
    protocol = session.protocol
    n_trials = len(session.trialdata)
    n_neurons = len(session.celldata)

    if protocol in TIME_LOCKED_PROTOCOLS:
        session.load_tensor(
            load_behaviordata=True, load_calciumdata=True,
            load_videodata=load_videodata, calciumversion=calciumversion,
            keepraw=keepraw, filter_hp=filter_hp)
        tensor = _orient_to_KNT(session.tensor, n_trials, n_neurons)
        axis = np.asarray(session.t_axis)
        axis_label = 'Time from stimulus onset (s)'

    elif protocol in SPATIAL_PROTOCOLS:
        session.load_data(
            load_behaviordata=True, load_calciumdata=True,
            load_videodata=load_videodata, calciumversion=calciumversion,
            filter_hp=filter_hp)
        assert hasattr(session, 'zpos_F') and hasattr(session, 'trialnum_F')

        raw_tensor, sbins = compute_tensor_space(
            session.calciumdata, session.ts_F, session.trialdata['stimStart'],
            session.zpo0000000000s_F, session.trialnum_F,
            s_pre=s_pre, s_post=s_post, binsize=binsize, method='binmean')
        tensor = _orient_to_KNT(raw_tensor, n_trials, n_neurons)
        axis = np.asarray(sbins)
        axis_label = 'Position relative to stimulus zone (cm)'

        if not keepraw:
            delattr(session, 'calciumdata')
            if hasattr(session, 'videodata') and session.videodata is not None:
                delattr(session, 'videodata')
            delattr(session, 'behaviordata')

    else:
        raise ValueError(
            f"Don't know how to compute a tensor for protocol '{protocol}'. "
            f"Add it to TIME_LOCKED_PROTOCOLS or SPATIAL_PROTOCOLS.")

    return tensor, axis, axis_label
