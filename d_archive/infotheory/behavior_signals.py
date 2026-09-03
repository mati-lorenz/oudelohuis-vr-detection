# -*- coding: utf-8 -*-
"""
behavior_signals.py
======================
Extracts continuous behavioral/state traces aligned to the imaging
frame timestamps (`session.ts_F`), so they can be correlated or
information-theoretically compared against each neuron's continuous
deconvolved activity trace on a frame-by-frame basis.

Supported variables:
    'position'   -- session.zpos_F (position along the VR corridor,
                     already interpolated to imaging frames by
                     loaddata, same convention used for
                     compute_respmat_space/compute_tensor_space)
    'runspeed'   -- session.runspeed_F (already at imaging-frame
                     resolution, see loaddata/just_load_a_session.py)
    'pupil_area' -- session.videodata['pupil_area'], interpolated from
                     video-frame timestamps onto ts_F
    'video_pc1'  -- first principal component of face-video motion,
                     interpolated onto ts_F. Looks for a precomputed
                     column first (a few common naming conventions);
                     if none is found, falls back to computing PC1 on
                     the fly from whatever numeric columns are present
                     in videodata (excluding pupil_area and
                     timestamps) -- this fallback is a coarse
                     approximation (PCA over whatever summary features
                     happen to be in videodata, not a proper
                     motion-energy SVD over raw video frames), so
                     prefer passing `video_pc_column=` explicitly once
                     you know the right column name in your dataset.

Every function raises a clear, actionable ValueError (listing what WAS
found) rather than failing silently or guessing, since attribute/column
names for behavioral data vary across loaddata versions and datasets.
"""

import numpy as np
import pandas as pd


def _interp_to_reference(values, src_times, ref_times):
    values = np.asarray(values, dtype=float)
    src_times = np.asarray(src_times, dtype=float)
    order = np.argsort(src_times)
    return np.interp(np.asarray(ref_times, dtype=float), src_times[order], values[order])


def _find_video_timestamps(session):
    for attr in ('ts_video', 'timestamps_video', 'tsvideo'):
        if hasattr(session, attr):
            return np.asarray(getattr(session, attr))
    if hasattr(session, 'videodata') and session.videodata is not None:
        for col in ('ts', 'timestamps', 'time'):
            if col in session.videodata.columns:
                return session.videodata[col].to_numpy()
    return None


def get_behavior_trace(session, var_name, video_pc_column=None):
    """
    Return a 1D array of the requested behavioral variable, resampled
    onto `session.ts_F` (imaging-frame timestamps).

    Parameters
    ----------
    session : loaddata.session.Session, already loaded with
        load_behaviordata=True (for 'position'/'runspeed') and/or
        load_videodata=True (for 'pupil_area'/'video_pc1')
    var_name : {'position', 'runspeed', 'pupil_area', 'video_pc1'}
    video_pc_column : optional str, explicit column name in
        session.videodata for the face-motion PC (skips the
        auto-detect/fallback logic below)

    Returns
    -------
    1D array, length == len(session.ts_F)
    """
    if not hasattr(session, 'ts_F'):
        raise ValueError('session.ts_F not found -- load calcium data first '
                          '(session.load_data(load_calciumdata=True, ...)).')
    ts_F = np.asarray(session.ts_F)

    if var_name == 'position':
        if not hasattr(session, 'zpos_F') or session.zpos_F is None:
            raise ValueError(
                "session.zpos_F not found -- this requires BOTH "
                "load_behaviordata=True and load_calciumdata=True when "
                "calling session.load_data(...) (zpos_F is only created "
                "when both are loaded together, see loaddata/session.py).")
        trace = np.asarray(session.zpos_F)

    elif var_name == 'runspeed':
        if not hasattr(session, 'runspeed_F') or session.runspeed_F is None:
            raise ValueError(
                "session.runspeed_F not found -- requires "
                "load_behaviordata=True and load_calciumdata=True together.")
        trace = np.asarray(session.runspeed_F)

    elif var_name in ('pupil_area', 'video_pc1'):
        if not hasattr(session, 'videodata') or session.videodata is None:
            raise ValueError(
                f"session.videodata not found -- needed for '{var_name}'. "
                f"Load with load_videodata=True.")
        videodata = session.videodata
        video_ts = _find_video_timestamps(session)
        if video_ts is None:
            raise ValueError(
                "Could not find video timestamps (tried session.ts_video, "
                "session.timestamps_video, and a 'ts'/'timestamps'/'time' "
                f"column in videodata; available videodata columns: "
                f"{list(videodata.columns)}). Adjust "
                f"behavior_signals._find_video_timestamps for your dataset.")

        if var_name == 'pupil_area':
            if 'pupil_area' not in videodata.columns:
                raise ValueError(
                    f"'pupil_area' not in videodata columns: {list(videodata.columns)}")
            raw = videodata['pupil_area'].to_numpy()
        else:  # video_pc1
            if video_pc_column is not None:
                if video_pc_column not in videodata.columns:
                    raise ValueError(
                        f"'{video_pc_column}' not in videodata columns: "
                        f"{list(videodata.columns)}")
                raw = videodata[video_pc_column].to_numpy()
            else:
                raw = None
                for cand in ('motSVD_0', 'motionPC1', 'video_PC1', 'PC1', 'pc1',
                             'motionenergy_PC1'):
                    if cand in videodata.columns:
                        raw = videodata[cand].to_numpy()
                        break
                if raw is None:
                    import pandas as pd
                    exclude = {'pupil_area', 'ts', 'timestamps', 'time'}
                    # use pandas' own dtype check, NOT np.issubdtype: pandas
                    # nullable/extension dtypes (e.g. StringDtype, Int64) are
                    # not interpretable by np.issubdtype and raise a TypeError
                    # instead of just returning False, which crashed this
                    # fallback whenever videodata had a non-numeric string
                    # column (e.g. a filename or condition-label column)
                    numeric_cols = [c for c in videodata.columns if c not in exclude
                                     and pd.api.types.is_numeric_dtype(videodata[c])]
                    if not numeric_cols:
                        raise ValueError(
                            "No precomputed face-motion PC column found in videodata "
                            f"(available: {list(videodata.columns)}), and no other "
                            "numeric columns to fall back on. Pass video_pc_column= "
                            "explicitly once you know the right column name.")
                    from sklearn.decomposition import PCA
                    Xv = videodata[numeric_cols].to_numpy(dtype=float)
                    Xv = Xv - np.nanmean(Xv, axis=0)
                    Xv = np.nan_to_num(Xv)
                    raw = PCA(n_components=1).fit_transform(Xv)[:, 0]
                    print(f"[behavior_signals] WARNING: no precomputed video PC1 "
                          f"column found; computed an on-the-fly PC1 from columns "
                          f"{numeric_cols} as a coarse approximation. Pass "
                          f"video_pc_column= explicitly to use a specific column "
                          f"instead.")
        trace = _interp_to_reference(raw, video_ts, ts_F)

    else:
        raise ValueError(f"Unknown var_name '{var_name}'; expected one of "
                          f"'position', 'runspeed', 'pupil_area', 'video_pc1'.")

    if len(trace) != len(ts_F):
        raise ValueError(f"'{var_name}' trace length ({len(trace)}) does not match "
                          f"ts_F length ({len(ts_F)}) after alignment -- unexpected.")
    return trace


def compute_behavior_tensor_for_session(session, var_name, t_pre=-1, t_post=2,
                                         s_pre=-60, s_post=80, binsize=10,
                                         video_pc_column=None):
    """
    Build a (K trials, T bins) tensor for a continuous behavioral
    variable, using the EXACT SAME time/space binning convention as
    tensor_utils.compute_tensor_for_session (same t_pre/t_post or
    s_pre/s_post/binsize parameters, same axis/axis_label return
    values) -- so a neural tensor and a behavior tensor for the SAME
    session line up bin-for-bin, and any window-collapsing function
    written for one (e.g. run_dimensionality_estimate.py's
    `collapse_window`) works identically on the other.

    Achieved by packaging the behavior trace as a single-column
    "neuron" DataFrame and running it through the identical underlying
    windowing call used for real neural data (compute_tensor_space for
    spatial/VR protocols). Time-locked protocols (IM/GR/GN) are not yet
    supported here (compute_tensor_space's time-locked counterpart has
    an API this module hasn't needed to pin down elsewhere in the
    pipeline) -- raises a clear NotImplementedError rather than
    guessing if you hit this on a time-locked protocol.

    Parameters
    ----------
    session : loaddata.session.Session, already loaded (see
        get_behavior_trace for the load_behaviordata/load_videodata
        requirements of each var_name)
    var_name : {'position', 'runspeed', 'pupil_area', 'video_pc1'}
    t_pre, t_post, s_pre, s_post, binsize, video_pc_column : see
        tensor_utils.compute_tensor_for_session / get_behavior_trace

    Returns
    -------
    tensor : array (K, T)
    axis : 1D array, length T
    axis_label : str
    """
    from .session_utils import TIME_LOCKED_PROTOCOLS, SPATIAL_PROTOCOLS
    from .tensor_utils import _orient_to_KNT

    protocol = session.protocol
    n_trials = len(session.trialdata)

    trace = get_behavior_trace(session, var_name, video_pc_column=video_pc_column)

    if protocol in SPATIAL_PROTOCOLS:
        from utils.psth import compute_tensor_space
        trace_df = pd.DataFrame({'beh': trace})
        raw_tensor, sbins = compute_tensor_space(
            trace_df, session.ts_F, session.trialdata['stimStart'],
            session.zpos_F, session.trialnum_F,
            s_pre=s_pre, s_post=s_post, binsize=binsize, method='binmean')
        tensor = _orient_to_KNT(raw_tensor, n_trials, 1)[:, 0, :]
        axis = np.asarray(sbins)
        axis_label = 'Position relative to stimulus zone (cm)'

    elif protocol in TIME_LOCKED_PROTOCOLS:
        raise NotImplementedError(
            f"compute_behavior_tensor_for_session doesn't yet support "
            f"time-locked protocol '{protocol}' (compute_tensor's exact "
            f"signature hasn't been confirmed elsewhere in this pipeline "
            f"-- see tensor_utils.compute_tensor_for_session's docstring "
            f"for the same limitation on the neural-data side). For a "
            f"per-trial SCALAR (not per-bin tensor) behavior summary on "
            f"time-locked protocols, see gain_model_analysis.py's "
            f"compute_behavior_trial_values instead.")
    else:
        raise ValueError(f"Unknown protocol '{protocol}'.")

    return tensor, axis, axis_label
