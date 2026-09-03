# -*- coding: utf-8 -*-
"""
session_utils.py
==================
Protocol-aware helper for getting a (N neurons, K trials) response matrix
out of a loaddata Session object.

`Session.load_respmat` (loaddata/session.py) only knows how to build a
TIME-locked response window, and only for the static-stimulus protocols
('IM', 'GR', 'GN') -- for the VR corridor detection protocols
('VR', 'DM', 'DN', 'DP') it silently skips and leaves `session.respmat`
unset, because for those protocols the meaningful response window is
defined in SPACE (position in the corridor relative to the stimulus
zone), not in time. The correct spatial computation is
`utils.psth.compute_respmat_space`, exactly as used in
`loaddata.session_info.load_neural_performing_sessions`.

This module wraps both cases behind one call so the information-theory
pipeline (single_cell.py) does not need to know which protocol it is
dealing with.
"""

from utils.psth import compute_respmat_space


#: protocols whose response window is defined in time relative to
#: stimulus/trial onset, handled by Session.load_respmat
TIME_LOCKED_PROTOCOLS = ('IM', 'GR', 'GN')

#: protocols whose response window is defined in space (cm in the VR
#: corridor) relative to the stimulus zone, handled here directly via
#: compute_respmat_space
SPATIAL_PROTOCOLS = ('VR', 'DM', 'DN', 'DP')


def compute_respmat_for_session(session, calciumversion='deconv',
                                 load_videodata=False, filter_hp=None,
                                 keepraw=False,
                                 s_resp_start=0, s_resp_stop=20):
    """
    Ensure `session.respmat` (N neurons x K trials) is computed,
    dispatching to the correct method based on `session.protocol`.

    Parameters
    ----------
    session : loaddata.session.Session
    calciumversion : 'dF' or 'deconv'
    load_videodata : bool, forwarded to the loader (not needed for the
        info-theory pipeline itself, but harmless/useful if you also want
        session.respmat_pupilarea etc. for later stages)
    filter_hp : optional high-pass filter cutoff, forwarded to the loader
    keepraw : if False (default), the raw calciumdata/videodata/
        behaviordata traces are deleted after respmat is computed, to
        save memory -- matches the behavior of Session.load_respmat.
    s_resp_start, s_resp_stop : float, cm
        spatial response window relative to the stimulus zone, used ONLY
        for the spatial protocols (VR/DM/DN/DP). Defaults match
        `load_neural_performing_sessions` (0 to 20 cm into the stimulus
        zone).

    Returns
    -------
    session.respmat : array (N neurons, K trials)
    """
    protocol = session.protocol

    if protocol in TIME_LOCKED_PROTOCOLS:
        session.load_respmat(
            load_behaviordata=True, load_calciumdata=True,
            load_videodata=load_videodata, calciumversion=calciumversion,
            keepraw=keepraw, filter_hp=filter_hp)

    elif protocol in SPATIAL_PROTOCOLS:
        session.load_data(
            load_behaviordata=True, load_calciumdata=True,
            load_videodata=load_videodata, calciumversion=calciumversion,
            filter_hp=filter_hp)

        assert hasattr(session, 'zpos_F') and hasattr(session, 'trialnum_F'), (
            'zpos_F / trialnum_F not found -- these are only created when '
            'both load_behaviordata and load_calciumdata are True in '
            'session.load_data (see loaddata/session.py)')

        session.respmat = compute_respmat_space(
            session.calciumdata, session.ts_F, session.trialdata['stimStart'],
            session.zpos_F, session.trialnum_F,
            s_resp_start=s_resp_start, s_resp_stop=s_resp_stop,
            method='mean', subtr_baseline=False)

        if not keepraw:
            delattr(session, 'calciumdata')
            if hasattr(session, 'videodata') and session.videodata is not None:
                delattr(session, 'videodata')
            delattr(session, 'behaviordata')

    else:
        raise ValueError(
            f"Don't know how to compute a response matrix for protocol "
            f"'{protocol}'. Add it to TIME_LOCKED_PROTOCOLS or "
            f"SPATIAL_PROTOCOLS in session_utils.py, with the right "
            f"response-window convention.")

    return session.respmat
