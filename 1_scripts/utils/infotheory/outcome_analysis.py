# -*- coding: utf-8 -*-
"""
outcome_analysis.py
====================
Per-session computations for 2b_activity_statistics PART 3, after van Vugt et al. 2018 (Science 360:537): how does
cell activity differ between HITS, MISSES, FALSE ALARMS (FA) and CORRECT REJECTIONS (CR), and how much of the
stimulus-driven activity survives on misses?

Outcomes (engaged trials only): HIT = signal>0 & lick; MISS = signal>0 & no lick; FA = signal==0 & lick;
CR = signal==0 & no lick. Stimulus trials are split into CONTRAST CATEGORIES by the psychometric position of the
contrast (stim_binning rule `pct_0-20_20-80_80-100`): difficult (<=20% of the curve; contrast 0 excluded),
intermediate, easy -- the analogue of van Vugt's qLow / qHigh split.

Activity = deconvolved activity, z-scored per cell over the session's engaged frames (so cells and sessions are
comparable and prestimulus differences are visible). Position axis as in the raster (cm relative to stimStart).

CONTRAST MATCHING (van Vugt's "contrast-matched hits and misses"): within a category, hits and misses are compared on
the contrast levels both have; each hit trial is weighted so that the hit contrast distribution equals the miss
distribution (weight = n_miss(level)/n_hit(level); misses weigh 1; levels without both are dropped). Used for traces
and miss fractions. Choice probability / marker AUROCs are instead conditioned on contrast level (pooled within-level
AUC, infotheory.roc), which asks the same question.

Scalars use position-matched window means (EQUALIZE_QUANTILE, as 2d's ROC): pre_stim (-30,-10), stim (0,20) cm.
Response = stim - pre_stim (baseline-subtracted). MISS FRACTION = 100 * mean response on misses / mean response on
hits (van Vugt: ActivityMiss/ActivityHit x 100%, 0-300 ms); it is computed at the level of the analysis unit in the
script (ratio of means, not mean of ratios: single-cell ratios explode when a cell's hit response is near 0).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from . import tensors as tn
from .roc import auc_permutation_test, roc_auc_conditioned
from .stim_binning import bin_stimulus, psychometric_pct
from .continuous import merge_behavior_video, remove_pupil_outliers, restrict_to_engaged, \
    compute_trial_position_window_means

CATEGORY_NAMES = ("difficult", "intermediate", "easy")
BEHAVIOR_MARKERS = ["runspeed", "pupil_area", "motionenergy", "lick"]


def label_trials(trials: pd.DataFrame, mu, sigma, rule: dict) -> pd.DataFrame:
    """Add stim_pct, outcome (HIT/MISS/FA/CR) and category (0,1,2 for stimulus trials, -1 otherwise)."""
    sig = trials["signal"].to_numpy(float)
    lick = trials["lickResponse"].to_numpy().astype(bool)
    pct = psychometric_pct(sig, mu, sigma)
    cat = bin_stimulus(sig, pct, rule)
    present = sig > 0
    outcome = np.where(present, np.where(lick, "HIT", "MISS"), np.where(lick, "FA", "CR"))
    category = np.where(present & np.isfinite(cat), cat, -1).astype(int)
    return trials.assign(stim_pct=pct, outcome=outcome, category=category)


def matched_weights(signal, outcome, category):
    """Per-trial weights (w_hit, w_miss) that match the hit contrast distribution to the miss one (see docstring)."""
    signal = np.asarray(signal, float)
    w_hit = np.zeros(len(signal))
    w_miss = np.zeros(len(signal))
    for level in np.unique(signal[np.asarray(category) >= 0]):
        mh = (signal == level) & (outcome == "HIT")
        mm = (signal == level) & (outcome == "MISS")
        if mh.sum() and mm.sum():
            w_hit[mh] = mm.sum() / mh.sum()
            w_miss[mm] = 1.0
    return w_hit, w_miss


def wmean(T, w, min_trials: int):
    """Weighted nan-mean over axis 0 of T (trials, ...) with trial weights w; NaN where fewer than `min_trials`
    trials with weight > 0 and finite value contribute."""
    T = np.asarray(T, float)
    w = np.asarray(w, float).reshape((-1,) + (1,) * (T.ndim - 1))
    valid = np.isfinite(T) & (w > 0)
    num = np.where(valid, np.nan_to_num(T) * w, 0.0).sum(0)
    den = np.where(valid, w, 0.0).sum(0)
    cnt = valid.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out = np.where((cnt >= min_trials) & (den > 0), out, np.nan)
    return out


def _auc_rows(session_id, cell_ids, analysis, window, res):
    return pd.DataFrame({"session_id": session_id, "cell_id": cell_ids, "analysis": analysis, "window": window,
                         "auc": res["auc"], "si": res["si"], "p_value": res["p_value"],
                         "significant": res["significant"], "n_pos": res["n_pos"], "n_neg": res["n_neg"]})


def analyze_session(ses, cell_ids, cell_group: dict, mu_sigma: dict, rule: dict, windows: dict, s_pre: float,
                    s_post: float, binsize: float, equalize_quantile, min_trials_cond: int, min_per_class: int,
                    n_perm: int, alpha: float, min_pop_cells: int, rng):
    """Everything 2b part 3 needs from ONE loaded session (calcium, behavior, video, celldata). Returns a dict of
    small arrays / tables, or None if the session cannot be analysed."""
    if (ses.calciumdata is None or ses.ts_F is None or ses.trialdata is None or ses.behaviordata is None
            or "engaged" not in ses.trialdata.columns):
        return None
    mu, sigma = mu_sigma.get("mu", np.nan), mu_sigma.get("sigma", np.nan)
    if not (np.isfinite(mu) and np.isfinite(sigma)):
        return None
    sid = ses.session_id
    trials = ses.trialdata.sort_values("trialNumber")
    trials = trials[trials["engaged"] == 1].reset_index(drop=True)
    if len(trials) < 20:
        return None
    trials = label_trials(trials, mu, sigma, rule)
    if (trials["category"] >= 0).sum() == 0:
        return None
    n = len(trials)
    signal = trials["signal"].to_numpy(float)
    outcome = trials["outcome"].to_numpy()
    category = trials["category"].to_numpy()

    A = ses.calciumdata.loc[:, cell_ids].to_numpy(dtype=float)
    frame_ok = tn.engaged_frame_mask(ses.ts_F, ses.trialdata)
    A = tn.zscore_frames(A, frame_ok)
    zpos_F = tn.interp_to_imaging_frames(ses.behaviordata["ts"], ses.behaviordata["zpos"], ses.ts_F)
    k, rel = tn.position_relative_to_stim(zpos_F, trials["stimStart"].to_numpy(), s_pre, s_post)
    tensor, _, centers = tn.build_position_tensor(A, k, rel, frame_ok, n, s_pre, s_post, binsize)
    wm = tn.trial_window_means(A, k, rel, frame_ok, n, windows, equalize_quantile=equalize_quantile)
    PRE, STIM = wm["pre_stim"], wm["stim"]
    RESP = STIM - PRE

    # ---- conditions and weights ------------------------------------------------------------------------
    weights = {}
    scal_rows = []
    for ci, cname in enumerate(CATEGORY_NAMES):
        in_cat = category == ci
        wh, wm_ = matched_weights(np.where(in_cat, signal, 0.0), outcome, np.where(in_cat, category, -1))
        weights[f"HIT|{cname}"], weights[f"MISS|{cname}"] = wh, wm_
        n_h, n_m = int((wh > 0).sum()), int((wm_ > 0).sum())
        if n_h >= min_trials_cond and n_m >= min_trials_cond:
            scal_rows.append(pd.DataFrame({
                "session_id": sid, "cell_id": cell_ids, "category": cname, "n_hit": n_h, "n_miss": n_m,
                "resp_hit": wmean(RESP, wh, min_trials_cond), "resp_miss": wmean(RESP, wm_, min_trials_cond),
                "pre_hit": wmean(PRE, wh, min_trials_cond), "pre_miss": wmean(PRE, wm_, min_trials_cond)}))
    weights["CR"] = (outcome == "CR").astype(float)
    weights["FA"] = (outcome == "FA").astype(float)
    traces, n_cond = {}, {}
    for cond, w in weights.items():
        n_cond[cond] = int((w > 0).sum())
        traces[cond] = wmean(tensor, w, min_trials_cond) if n_cond[cond] >= min_trials_cond else None
    scalars = pd.concat(scal_rows, ignore_index=True) if scal_rows else pd.DataFrame()
    fa_cr = pd.DataFrame()
    if n_cond["CR"] >= min_trials_cond and n_cond["FA"] >= min_trials_cond:
        fa_cr = pd.DataFrame({
            "session_id": sid, "cell_id": cell_ids, "n_CR": n_cond["CR"], "n_FA": n_cond["FA"],
            "pre_CR": wmean(PRE, weights["CR"], min_trials_cond), "pre_FA": wmean(PRE, weights["FA"], min_trials_cond),
            "resp_CR": wmean(RESP, weights["CR"], min_trials_cond), "resp_FA": wmean(RESP, weights["FA"], min_trials_cond)})

    # ---- hit vs miss / FA vs CR: choice probability (windows and over position) ----------------------
    cp_rows, cp_trace = [], {}
    is_hm = np.isin(outcome, ["HIT", "MISS"]) & (category >= 0)
    sets = {"hit_vs_miss_intermediate": is_hm & (category == 1), "hit_vs_miss_all": is_hm,
            "FA_vs_CR": (signal == 0) & np.isin(outcome, ["FA", "CR"])}
    for name, sel in sets.items():
        positive = "FA" if name == "FA_vs_CR" else "HIT"
        lab = (outcome == positive)[sel]
        grp = None if name == "FA_vs_CR" else signal[sel]
        for wname in ("pre_stim", "stim"):
            M = wm[wname][sel]
            ok = np.isfinite(M).all(1)
            if ok.sum() < 2 * min_per_class:
                continue
            res = auc_permutation_test(M[ok], lab[ok], groups=None if grp is None else grp[ok], n_perm=n_perm,
                                       rng=rng, min_per_class=min_per_class, alpha=alpha)
            cp_rows.append(_auc_rows(sid, cell_ids, name, wname, res))
        if name == "FA_vs_CR":
            continue
        T = tensor[sel]
        tr = np.full((len(cell_ids), T.shape[2]), np.nan)
        for b in range(T.shape[2]):
            X = T[:, :, b]
            ok = np.isfinite(X).all(1)
            if ok.sum() >= 2 * min_per_class:
                tr[:, b] = roc_auc_conditioned(X[ok], lab[ok], grp[ok], min_per_class)[0]
        cp_trace[name] = tr
    cp = pd.concat(cp_rows, ignore_index=True) if cp_rows else pd.DataFrame()

    # ---- prestimulus markers (population + behavior + trial timing) -> AUROC, and per-trial table -----
    mt = pd.DataFrame({"trialNumber": trials["trialNumber"].to_numpy(), "signal": signal, "outcome": outcome,
                       "category": category})
    cell_groups = np.array([cell_group.get(c, "") for c in cell_ids])
    for g in sorted(set(cell_groups) - {""}):
        idx = np.where(cell_groups == g)[0]
        if len(idx) >= min_pop_cells:
            P = PRE[:, idx]
            cnt = np.isfinite(P).sum(1)
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mt[f"popmean__{g}"] = np.where(cnt >= min_pop_cells, np.nanmean(np.where(np.isfinite(P), P, np.nan), 1), np.nan)
                mt[f"popvar__{g}"] = np.where(cnt >= min_pop_cells, np.nanvar(np.where(np.isfinite(P), P, np.nan), 1, ddof=1), np.nan)
    try:
        merged = remove_pupil_outliers(merge_behavior_video(ses.behaviordata, ses.videodata))
        merged = restrict_to_engaged(merged, ses.trialdata)
        cols = [c for c in BEHAVIOR_MARKERS if c in merged.columns]
        if cols and len(merged):
            lo, hi = windows["pre_stim"]
            bw = compute_trial_position_window_means(trials, merged, columns=cols, s_start=lo, s_end=hi,
                                                     rate_columns=["lick"] if "lick" in cols else None)
            for c in cols:
                mt[c] = bw[c].to_numpy()
    except Exception as e:                                   # video missing etc.: markers from behavior only
        print(f"  [{sid}] behavior markers skipped: {e}")
    full = ses.trialdata.sort_values("trialNumber")
    iti = (full["tStart"].to_numpy() - np.r_[np.nan, full["tEnd"].to_numpy()[:-1]])
    mt["iti"] = pd.Series(iti, index=full["trialNumber"].to_numpy()).reindex(mt["trialNumber"]).to_numpy()
    marker_cols = [c for c in mt.columns if c.startswith(("popmean__", "popvar__"))] + \
                  [c for c in BEHAVIOR_MARKERS + ["iti"] if c in mt.columns]
    mk_rows = []
    for cmp_name in ("hit_vs_miss", "FA_vs_CR"):
        sel = is_hm if cmp_name == "hit_vs_miss" else sets["FA_vs_CR"]
        positive = "HIT" if cmp_name == "hit_vs_miss" else "FA"
        lab = (outcome == positive)
        for col in marker_cols:
            x = mt[col].to_numpy(float)
            ok = sel & np.isfinite(x)
            if ok.sum() < 2 * min_per_class:
                continue
            grp = signal[ok] if cmp_name == "hit_vs_miss" else None
            auc, n_pos, n_neg, ng = roc_auc_conditioned(x[ok][:, None], lab[ok], grp, min_per_class)
            if not np.isfinite(auc[0]):
                continue
            kind, _, grp_name = col.partition("__")
            mk_rows.append(dict(session_id=sid, comparison=cmp_name, marker=kind if grp_name else col,
                                group=grp_name, auc=float(auc[0]), n_pos=n_pos, n_neg=n_neg, n_groups=ng))
    markers = pd.DataFrame(mk_rows)
    counts = pd.DataFrame([dict(session_id=sid, category=CATEGORY_NAMES[c] if c >= 0 else "no stimulus",
                                outcome=o, n=int(((category == c) & (outcome == o)).sum()))
                           for c in (-1, 0, 1, 2) for o in ("HIT", "MISS", "FA", "CR")
                           if ((c == -1) == (o in ("FA", "CR")))])
    return dict(session_id=sid, animal_id=ses.animal_id, cell_ids=list(cell_ids), centers=centers, traces=traces,
                n_cond=n_cond, scalars=scalars, fa_cr=fa_cr, cp=cp, cp_trace=cp_trace, markers=markers,
                marker_trials=mt[["trialNumber", "signal", "outcome", "category"] + marker_cols], counts=counts)
