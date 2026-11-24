# -*- coding: utf-8 -*-
"""
performance_common.py
=====================
Behavioral PERFORMANCE per quantile bin of a state variable (pupil in 1f, running speed in 1g) -- the shared code, so the two scripts
mirror each other.

Per (protocol, unit, bin of x) and window:
    hit_rate           P(lick | signal > 0), every signal strength          (ALL engaged trials)
    fa_rate            P(lick | signal = 0)                                 (ALL engaged trials)
    balanced_accuracy  (hit + 1 - FA) / 2                                   (ALL engaged trials)
    dprime, criterion  d' / criterion on the NOISE trials only (0 < signal < 100 %), method = DPRIME_METHOD / CRITERION_METHOD
                       of infotheory/noise_sdt.py; every variant is kept as dprime_<method> / criterion_<method>
    dprime_0_100, criterion_0_100   the classical 0 % / 100 % version (the unit's pooled trials), for reference
Bins: `bin_fn(values of ONE session) -> bin number per trial (1..n, NaN = invalid)`, so each script decides how x is binned
(pupil: quantiles among all engaged trials; speed: the same, quantiles). The x of each window is the mean of x inside that window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .behavior import compute_dprime
from .noise_sdt import noise_sdt_session, NOISE_COLS
from .plotting import PROTOCOL_COLORS, save_fig


def noise_measures(g, fit_params, split_at, *, min_noise, sd_a, sd_b_frac, rate_clip, min_per_class):
    """d' / criterion variants on the NOISE trials of one unit's bin: computed per session (each session has its own 1b curve, floor
    and lapse), then averaged over the sessions of the unit. Returns a dict of NOISE_COLS plus the number of noise trials."""
    gn = g[(g["signal"] > 0) & (g["signal"] < 100)]
    recs = []
    for sid, gs in gn.groupby("session_id"):
        fp = fit_params.get(sid, {})
        recs.append(noise_sdt_session(gs["signal"].to_numpy(float), gs["lickResponse"].to_numpy(float), fp.get("mu"), fp.get("sigma"),
                                      fp.get("guess_rate"), fp.get("lapse_rate"), split_at.get(sid), min_n=min_noise,
                                      sd_a=sd_a, sd_b_frac=sd_b_frac, rate_clip=rate_clip, min_per_class=min_per_class))
    out = {}
    for c in NOISE_COLS:
        v = np.array([r[c] for r in recs], float)
        out[c] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
    out["n_trials_noise"] = int(len(gn))
    return out


def perf_by_bins(d_all, xcol, bin_fn, *, protocols, unit_keys, fit_params, split_at, dprime_method, criterion_method, min_per_class,
                 min_noise, sd_a, sd_b_frac, rate_clip, bin_col="bin"):
    """Performance measures per (protocol, unit, bin of `xcol`). d_all = ALL engaged trials (session_id, protocol, signal,
    lickResponse, xcol, + unit_keys columns). Returns a list of dict records."""
    out = []
    dw = d_all.dropna(subset=[xcol]).copy()
    if dw.empty:
        return out
    dw[bin_col] = dw.groupby("session_id")[xcol].transform(lambda v: pd.Series(bin_fn(v.to_numpy(float)), index=v.index))
    dw = dw.dropna(subset=[bin_col])
    for protocol in [q for q in protocols if (dw["protocol"] == q).any()]:
        d = dw[dw["protocol"] == protocol]
        for keys, g in d.groupby(unit_keys + [bin_col] if unit_keys else [bin_col]):
            keys = keys if isinstance(keys, tuple) else (keys,)
            n0, n100, npos = int((g["signal"] == 0).sum()), int((g["signal"] == 100).sum()), int((g["signal"] > 0).sum())
            ok_hf = min(npos, n0) >= min_per_class
            hit = float(g.loc[g["signal"] > 0, "lickResponse"].mean()) if ok_hf else np.nan        # any signal > 0
            fa = float(g.loc[g["signal"] == 0, "lickResponse"].mean()) if ok_hf else np.nan
            dp0, crit0 = compute_dprime(g["signal"], g["lickResponse"]) if min(n100, n0) >= min_per_class else (np.nan, np.nan)
            nz = noise_measures(g, fit_params, split_at, min_noise=min_noise, sd_a=sd_a, sd_b_frac=sd_b_frac,
                                rate_clip=rate_clip, min_per_class=min_per_class)
            rec = {"protocol": protocol, bin_col: int(keys[-1]),
                   "dprime": nz[f"dprime_{dprime_method}"], "criterion": nz[f"criterion_{criterion_method}"],
                   "dprime_0_100": dp0, "criterion_0_100": crit0,
                   "hit_rate": hit, "fa_rate": fa, "balanced_accuracy": (hit + 1.0 - fa) / 2.0,
                   "n_trials_0": n0, "n_trials_100": n100, "n_trials_signal_pos": npos}
            rec.update(nz)
            rec.update(dict(zip(unit_keys, keys[:-1])))
            out.append(rec)
    return out


def _curve(dd, col, unit_keys, bin_col):
    if unit_keys:
        per = dd.groupby(unit_keys + [bin_col])[col].mean().reset_index()
        g = per.groupby(bin_col)[col]
        return g.mean(), g.sem().fillna(0)
    return dd.groupby(bin_col)[col].mean(), None


def plot_performance_figures(dp_df, *, figdir, names, windows, protocols, unit_keys, unit, n_bins, x_name, dprime_method,
                             criterion_method, bin_col="bin", bin_labels=None):
    """names = (main figure stem, methods-comparison figure stem). x axis = the quantile (bin number), not the mean value of x."""
    if dp_df is None or not len(dp_df):
        return
    wins = [w for w in windows if (dp_df["window"] == w).any()]
    ticks = list(range(1, n_bins + 1))
    ticklabels = bin_labels or [str(t) for t in ticks]
    err = "one value per bin, no error bars" if not unit_keys else "mean +- SEM over units"

    measures = [("hit_rate", "hit rate  P(lick | signal>0)\n(all signal strengths)"), ("fa_rate", "FA rate  P(lick | 0 %)"),
                ("balanced_accuracy", "balanced accuracy\n(hit[signal>0] + CR) / 2"),
                ("dprime", f"d' (noise trials,\n{dprime_method})"), ("criterion", f"criterion (noise trials,\n{criterion_method})")]
    fig, axes = plt.subplots(len(measures), len(wins), figsize=(4.4 * len(wins), 2.9 * len(measures)), squeeze=False, sharey="row")
    for r, (col, lab) in enumerate(measures):
        for ax, w in zip(axes[r], wins):
            for protocol in [q for q in protocols if ((dp_df["window"] == w) & (dp_df["protocol"] == q)).any()]:
                dd = dp_df[(dp_df["window"] == w) & (dp_df["protocol"] == protocol)].dropna(subset=[col])
                if dd.empty:
                    continue
                m, sem = _curve(dd, col, unit_keys, bin_col)
                ax.errorbar(m.index, m.values, yerr=None if sem is None else sem.values, color=PROTOCOL_COLORS[protocol],
                            marker="o", markersize=4, capsize=3, label=protocol)
            ax.set_xticks(ticks)
            ax.set_xticklabels(ticklabels)
            ax.set_ylabel(lab)
            if r == 0:
                lo, hi = windows[w]
                ax.set_title(f"{w} ({lo:g}..{hi:g} cm)", fontsize=10)
            if col == "balanced_accuracy":
                ax.axhline(0.5, color="grey", lw=0.8, ls="--")
            if r == len(measures) - 1:
                ax.set_xlabel(f"{x_name} quantile in the {w} window\n(within session)")
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle(f"Performance by {x_name} quantile; hit / FA / balanced accuracy on ALL engaged trials, d' ({dprime_method}) and criterion "
                 f"({criterion_method}) on the NOISE trials only (unit = {unit}; {err})", y=1.0, fontsize=9)
    fig.tight_layout()
    save_fig(fig, figdir, names[0])
    plt.close(fig)

    # the d' / criterion variants side by side (do the methods agree? project rule 4). Noise-trial methods are averaged over the sessions
    # of a unit; the classical 0 % / 100 % reference is computed on the unit's pooled trials (few catch trials per bin and session).
    variants = [("dprime_probit_pred", "criterion_probit_pred", "noise: probit, predicted 0/100 % rates", "tab:blue"),
                ("dprime_probit_shift", "criterion_probit_shift", "noise: probit shift (criterion = threshold shift / sigma; d' = slope ratio)", "tab:green"),
                ("dprime_split", "criterion_split", "noise: weak vs strong halves", "tab:orange"),
                ("dprime_0_100", "criterion_0_100", "classical: 0 % vs 100 % trials", "black")]
    prots2 = [q for q in protocols if (dp_df["protocol"] == q).any()]
    fig, axes = plt.subplots(2 * len(prots2), len(wins), figsize=(4.4 * len(wins), 2.9 * 2 * len(prots2)), squeeze=False, sharex=True)
    for pi, protocol in enumerate(prots2):
        for mi, (mname, which) in enumerate([("d'", 0), ("criterion", 1)]):
            for ax, w in zip(axes[2 * pi + mi], wins):
                for v in variants:
                    col = v[which]
                    dd = dp_df[(dp_df["window"] == w) & (dp_df["protocol"] == protocol)].dropna(subset=[col])
                    if dd.empty:
                        continue
                    m, sem = _curve(dd, col, unit_keys, bin_col)
                    ax.errorbar(m.index, m.values, yerr=None if sem is None else sem.values, color=v[3], marker="o", markersize=3.5,
                                capsize=2, lw=1.2, label=v[2])
                ax.set_xticks(ticks)
                ax.set_xticklabels(ticklabels)
                ax.set_ylabel(f"{protocol}: {mname}")
                if pi == 0 and mi == 0:
                    lo, hi = windows[w]
                    ax.set_title(f"{w} ({lo:g}..{hi:g} cm)", fontsize=10)
                if 2 * pi + mi == axes.shape[0] - 1:
                    ax.set_xlabel(f"{x_name} quantile in the {w} window\n(within session)")
    handles = {}
    for ax in axes.ravel():                          # one legend with every variant (a panel may lack some)
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(ll, hh)
    axes[0][0].legend(list(handles.values()), list(handles.keys()), fontsize=6, frameon=False)
    fig.suptitle(f"d' and criterion by {x_name} quantile: noise-trial methods vs the classical 0 % / 100 % reference (unit = {unit}; "
                 f"{'one value per bin' if not unit_keys else 'mean +- SEM over units'}). The methods differ in scale (see noise_sdt.py); compare the SHAPE.",
                 y=1.0, fontsize=8)
    fig.tight_layout()
    save_fig(fig, figdir, names[1])
    plt.close(fig)
