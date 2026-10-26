# -*- coding: utf-8 -*-
"""
engagement.py
==============
The END-OF-SESSION DISENGAGED BLOCK ("tail") and its matched engaged control -- used by the exploratory
disengaged-trial tests in 2c_information and 2d_linear_encod.

Why not "all engaged==0 trials"? Disengaged trials also occur sporadically in the middle of a session
(a lapse, a pause). The tail is the sustained loss of engagement at the END, where the animal has
stopped working for good. Because the engaged flag is noisy there (the odd engaged==1 trial inside an
otherwise disengaged stretch), the tail is not required to be strictly engaged==0 throughout:

    tail = the LONGEST suffix of the session whose fraction of disengaged trials is >= min_frac
           (and which has >= min_trials trials), trimmed so that it starts on a disengaged trial.
    analysed disengaged trials = the engaged==0 trials inside the tail (the odd engaged==1 trial
           inside the tail is left out of BOTH conditions).
    matched engaged control    = the LAST n engaged trials BEFORE the tail, with n = number of analysed
           disengaged trials (n is reduced to what is available for both). Both samples then have the same
           number of trials (equal estimation bias for MI/r^2) and are adjacent in time to the transition,
           which limits drifts (bleaching, depth) as an explanation for any difference.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TAIL_MIN_FRAC = 0.9       # a suffix counts as "disengaged tail" if >= this fraction of its trials are engaged==0
TAIL_MIN_TRIALS = 20      # ... and it has at least this many analysed disengaged trials (project MIN_TRIALS_ANALYSIS)


def disengaged_tail_trials(trialdata: pd.DataFrame, min_frac: float = TAIL_MIN_FRAC,
                           min_trials: int = TAIL_MIN_TRIALS) -> dict | None:
    """Locate the end-of-session disengaged block. Returns None if there is none, else a dict:
        start_trial     trialNumber of the first tail trial
        n_tail          trials in the tail (engaged and not)
        frac_disengaged fraction of engaged==0 inside the tail
        trial_numbers   trialNumbers of the engaged==0 trials in the tail (the analysed disengaged trials)
        n_engaged_in_tail  engaged==1 trials inside the tail (excluded from the analysis)"""
    if "engaged" not in trialdata.columns or "trialNumber" not in trialdata.columns:
        return None
    td = trialdata.sort_values("trialNumber")
    dis = (td["engaged"].to_numpy() == 0)
    n = len(dis)
    if n == 0 or not dis.any():
        return None
    tail_dis = np.cumsum(dis[::-1])[::-1]                 # disengaged trials in suffix starting at i
    suffix_len = n - np.arange(n)
    frac = tail_dis / suffix_len
    ok = (frac >= min_frac) & (tail_dis >= min_trials)
    if not ok.any():
        return None
    i = int(np.argmax(ok))                                # smallest start index = longest suffix
    while i < n and not dis[i]:                           # start on a disengaged trial
        i += 1
    if i >= n:
        return None
    sub = td.iloc[i:]
    sub_dis = sub[sub["engaged"] == 0]
    if len(sub_dis) < min_trials:
        return None
    return dict(start_trial=int(sub["trialNumber"].iloc[0]), n_tail=len(sub),
                frac_disengaged=float(len(sub_dis) / len(sub)),
                trial_numbers=sub_dis["trialNumber"].astype(int).tolist(),
                n_engaged_in_tail=int(len(sub) - len(sub_dis)))


def matched_engaged_trials(trialdata: pd.DataFrame, tail: dict, n: int | None = None) -> list[int]:
    """The last `n` engaged trials before the tail (default: as many as the tail has analysed trials)."""
    n = len(tail["trial_numbers"]) if n is None else n
    before = trialdata[(trialdata["trialNumber"] < tail["start_trial"]) & (trialdata["engaged"] == 1)]
    return before.sort_values("trialNumber")["trialNumber"].astype(int).tolist()[-n:] if n > 0 else []


def tail_condition_trials(trialdata: pd.DataFrame, min_frac: float = TAIL_MIN_FRAC,
                          min_trials: int = TAIL_MIN_TRIALS) -> dict | None:
    """Tail + matched control with EQUAL n. Returns None if the session has no usable tail, else
    dict(info=..., disengaged=[trialNumbers], engaged_matched=[trialNumbers]). Both lists are cut to the
    same n (the n closest to the transition: the FIRST n tail trials, the LAST n control trials);
    None if fewer than `min_trials` remain in either."""
    tail = disengaged_tail_trials(trialdata, min_frac, min_trials)
    if tail is None:
        return None
    ctrl = matched_engaged_trials(trialdata, tail, len(tail["trial_numbers"]))
    n = min(len(ctrl), len(tail["trial_numbers"]))
    if n < min_trials:
        return None
    dis = tail["trial_numbers"][:n]
    ctrl = ctrl[-n:]
    return dict(info=dict(start_trial=tail["start_trial"], n_tail=tail["n_tail"],
                          frac_disengaged=tail["frac_disengaged"],
                          n_engaged_in_tail=tail["n_engaged_in_tail"], n_used=n,
                          control_first_trial=int(min(ctrl)), control_last_trial=int(max(ctrl))),
                disengaged=dis, engaged_matched=ctrl)


def tail_overview_figure(per_session: dict, trialdata_by_session: dict, figdir, name: str, mouse_colors=None):
    """One row per session with a tail: the engaged flag of every trial (grey = engaged, white = not),
    the matched engaged control (blue) and the analysed disengaged trials (red). A visual check that the
    right trials are used."""
    import matplotlib.pyplot as plt
    from .plotting import save_fig
    sids = [s for s in per_session if per_session[s] is not None]
    if not sids:
        return
    fig, ax = plt.subplots(figsize=(10, max(2.0, 0.42 * len(sids) + 1.2)))
    for yi, sid in enumerate(sids):
        td = trialdata_by_session[sid].sort_values("trialNumber")
        tn_, eng = td["trialNumber"].to_numpy(), td["engaged"].to_numpy()
        ax.scatter(tn_[eng == 1], np.full((eng == 1).sum(), yi), s=5, color="#bbbbbb", marker="|")
        ax.scatter(tn_[eng == 0], np.full((eng == 0).sum(), yi), s=5, color="#eeeeee", marker="|")
        c = per_session[sid]
        ax.scatter(c["engaged_matched"], np.full(len(c["engaged_matched"]), yi), s=14, color="#1f77b4", marker="|")
        ax.scatter(c["disengaged"], np.full(len(c["disengaged"]), yi), s=14, color="#d62728", marker="|")
    ax.set_yticks(range(len(sids)))
    ax.set_yticklabels(sids, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("trial number")
    ax.set_title("Disengaged-tail test: matched engaged control (blue), analysed disengaged tail trials (red)\n"
                 "grey = other engaged trials, pale = other disengaged trials", fontsize=9)
    fig.tight_layout()
    save_fig(fig, figdir, name)
    plt.close(fig)
