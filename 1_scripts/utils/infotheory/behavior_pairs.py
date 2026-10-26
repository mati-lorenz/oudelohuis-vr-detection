# -*- coding: utf-8 -*-
"""
behavior_pairs.py
=================
Data side of the pair analyses whose TARGET is a behavioral variable (3i correlation, 3j information, 3k breakdown, 3l PID).
The pair engine is the one of 3d-3f / 3h (pairs sampled per session, cell activity averaged per trial in a window, project
trial rules); the only difference is the target: instead of a stimulus binning or the choice, the target of a window is the
TRIAL MEAN OF A BEHAVIORAL VARIABLE IN THAT SAME WINDOW (runspeed, pupil_area, motionenergy, lick; BEHAVIOR_PREDICTORS as 2c / 2h),
cut into RESPONSE_BINS equal-occupancy bins (infotheory.pair_information.digitize_quantile, the same binning the cells get;
a heavily zero-inflated variable such as lick gets one bin per distinct value, see info_theory.quantile_bin_edges).

WINDOWS (`BehaviorWindows.windows()`), per axis
    position   cm relative to stimulus onset (frames assigned by the zpos trace)
    time       s relative to the stimulus onset time (derive_onset_time)
and per binning
    windows    position only: 3a-3c's windows pre_stim (-30..-10), stim (0..20), reward (25..45) cm
    fixed      non-overlapping bins of `binsize`
    sliding    windows of `window` moved by `step` (neighbouring windows overlap: smoother, not independent)
A frame is used only if it lies inside ITS OWN engaged trial's [tStart, tEnd]. Cell and behavior values of a trial = the mean of
its frames in the window (behavior: nan-aware, pupil blinks are removed first, as 2h).

NOTE ON WHAT THE TARGET MEANS: the label is defined WITHIN each window, so "high running speed" is relative to the other
trials in the same window. Within a window, trial-to-trial behavior differences are what the analyses read; the
position/time dependence of the behavior itself (the mean running profile) is NOT part of the target, because every window is
labelled separately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

from . import project_config as cfg
from . import position_info as pi
from .continuous import build_calcium_continuous, merge_behavior_video, remove_pupil_outliers
from .neural_encoding import BEHAVIOR_PREDICTORS
from .pair_information import digitize_quantile
from .psth import derive_onset_time

BEHAVIOR_VARS = list(BEHAVIOR_PREDICTORS)
AXES_DEF = {"position": dict(pre=-60.0, post=80.0, binsize=15.0, window=20.0, step=5.0,
                             label="position rel. to stimulus onset (cm)", unit="cm"),
            "time": dict(pre=-4.0, post=6.0, binsize=1.0, window=2.0, step=0.5,
                         label="time rel. to stimulus onset (s)", unit="s")}
WINDOWS = {"pre_stim": (-30.0, -10.0), "stim": (0.0, 20.0), "reward": (25.0, 45.0)}   # 3a-3c's windows (cm)
BINNINGS = ["windows", "fixed", "sliding"]


def behavior_labels(x, has, r_bins: int = cfg.RESPONSE_BINS):
    """Per-trial integer-coded (as float) quantile bins of the trial-mean behavior `x` among the trials with frames (`has`)
    and a finite value; NaN for the other trials."""
    lab = np.full(len(x), np.nan)
    fin = has & np.isfinite(x)
    if fin.sum() >= 2:
        lab[fin] = digitize_quantile(x[fin], r_bins)
    return lab


class BehaviorWindows:
    """One session: trial x cell mean activity and trial mean behavior in every window of every requested axis / binning.
    `windows()` yields dicts (axis, binning, window, s_rel, TM, TMB, cnt): TM = trials x cells, TMB = trials x behaviors,
    cnt = frames per trial. `col_of` maps cell -> TM column, `bad` flags cells with non-finite activity, `behaviors` are the
    variables present in this session."""

    def __init__(self, ses, cells, behaviors, axes_def, axes, binnings, imaging: bool = True):
        """imaging=False: behavior only (no calcium data needed): the frames are the behavior samples, TM has no columns."""
        self.cells = list(cells)
        self.col_of = {c: k for k, c in enumerate(self.cells)}
        self.axes_def, self.axes, self.binnings = axes_def, list(axes), list(binnings)
        self.ok = False
        if ses.trialdata is None or ses.behaviordata is None:
            return
        if imaging and (ses.calciumdata is None or ses.ts_F is None):
            return
        merged = remove_pupil_outliers(merge_behavior_video(ses.behaviordata, ses.videodata))
        self.behaviors = [b for b in behaviors if b in merged.columns]
        if not self.behaviors:
            return
        td = ses.trialdata
        trials = (td[td["engaged"] == 1] if "engaged" in td.columns else td).reset_index(drop=True)
        if trials.empty:
            return
        if imaging:
            cont = build_calcium_continuous(ses.calciumdata[self.cells], ses.ts_F, merged, extra_columns=self.behaviors)
        else:
            cont = merged[["ts", "zpos"] + self.behaviors].reset_index(drop=True)
        self.trials = trials
        trials = derive_onset_time(trials, cont)
        self.A = cont[self.cells].to_numpy(float) if self.cells else np.zeros((len(cont), 0))
        self.bad = ~np.isfinite(self.A).all(0)
        self.A = np.nan_to_num(self.A)
        B = cont[self.behaviors].to_numpy(float)
        self.Bfin = np.isfinite(B).astype(float)
        self.B0 = np.nan_to_num(B)
        self.ts = cont["ts"].to_numpy(float)
        self.zpos = cont["zpos"].to_numpy(float)
        self.t0, self.t1 = trials["tStart"].to_numpy(float), trials["tEnd"].to_numpy(float)
        self.anchor = {"position": trials["stimStart"].to_numpy(float), "time": trials["_onset_time"].to_numpy(float)}
        self.coord = {"position": self.zpos, "time": self.ts}
        self.nt = len(trials)
        self.ok = True

    def _grid(self, axis, binning):
        """(base bin size, [(window name, first base bin, number of base bins)]) or None when not defined."""
        ad = self.axes_def[axis]
        if binning == "fixed":
            return ad["binsize"], None, 1
        m = int(round(ad["window"] / ad["step"]))
        if binning == "sliding":
            return ad["step"], None, m
        if binning == "windows" and axis == "position":
            return ad["step"], WINDOWS, None
        return None

    def windows(self):
        for axis in self.axes:
            ad = self.axes_def[axis]
            for binning in self.binnings:
                grid = self._grid(axis, binning)
                if grid is None:
                    continue
                base, named, m = grid
                fi, tr, bb, centers = pi.frame_bin_index(self.anchor[axis], self.coord[axis], ad["pre"], ad["post"], base)
                if len(fi):
                    own = (self.ts[fi] >= self.t0[tr]) & (self.ts[fi] <= self.t1[tr])
                    fi, tr, bb = fi[own], tr[own], bb[own]
                if len(fi) == 0:
                    continue
                if named is not None:                                   # coarse named windows on the base grid
                    spans = []
                    for name, (lo, hi) in named.items():
                        k0, k1 = (lo - ad["pre"]) / base, (hi - ad["pre"]) / base
                        if abs(k0 - round(k0)) > 1e-9 or abs(k1 - round(k1)) > 1e-9:
                            raise SystemExit(f"window {name} ({lo}..{hi}) is not aligned with the {base:g} cm base grid")
                        spans.append((name, int(round(k0)), int(round(k1 - k0)), (lo + hi) / 2.0))
                else:
                    spans = [("", k, m, (centers[k] + centers[k + m - 1]) / 2.0) for k in range(len(centers) - m + 1)]
                for name, k0, mm, s_rel in spans:
                    sel = (bb >= k0) & (bb < k0 + mm)
                    f_w, t_w = fi[sel], tr[sel]
                    if len(f_w) == 0:
                        continue
                    cnt = np.bincount(t_w, minlength=self.nt).astype(float)
                    S = sparse.csr_matrix((np.ones(len(t_w)), (t_w, np.arange(len(t_w)))), shape=(self.nt, len(t_w)))
                    TM = (S @ self.A[f_w]) / np.maximum(cnt, 1)[:, None]
                    den = S @ self.Bfin[f_w]
                    TMB = np.where(den > 0, (S @ self.B0[f_w]) / np.maximum(den, 1e-12), np.nan)
                    yield dict(axis=axis, binning=binning, window=name, s_rel=s_rel, TM=TM, TMB=TMB, cnt=cnt)
