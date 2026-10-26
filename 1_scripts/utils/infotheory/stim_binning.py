# -*- coding: utf-8 -*-
"""
stim_binning.py
================
EXPLICIT, EDITABLE stimulus binning for the information / correlation
analyses (3a-3c; to be adopted by the other scripts).

What "position on the psychometric curve" means
-----------------------------------------------
Every session's 1b fit is a cumulative Gaussian
    p(x) = guess + (1 - guess - lapse) * Phi((x - mu) / sigma)
Phi((x - mu) / sigma) runs from 0 to 1 and is "how far up the curve" a
contrast sits, independent of the lapse/guess floors:
    0.0 = bottom of the curve, 0.5 = threshold (x = mu), 1.0 = saturated.
This is what `stim_pct` (0-1) stores per trial; the percent cut points
below refer to it, so "80% of the psychometric curve" = Phi = 0.8 =
x = mu + 0.84 * sigma in that session. Only mu and sigma are needed
(lapse/guess only rescale the curve and cancel in the normalization).
Sessions therefore get session-specific contrast cut points.

Defining / editing strategies
-----------------------------
A strategy is a dict:
    kind        "curve_pct" : cut `stim_pct` (0-1)
                "contrast"  : cut the raw `signal` (same units as the task)
                "quantile"  : n equal-occupancy bins of the z-scored stim
                              (the OLD default, n_bins=4)
    edges       sorted cut points. Bin i holds edges[i-1] < v <= edges[i]
                (right-inclusive), i.e. "up to 80%" includes 80%.
    zero_separate  True: trials with signal == 0 get their own first bin
                (no-stimulus catch trials), the remaining trials are cut by `edges`.
    names       optional labels, one per resulting bin (used on plots)
Add a new strategy by adding an entry to STIM_BINNING_STRATEGIES; select
which ones a script runs with its `BINNING_STRATEGIES` list.

Choice as the target
--------------------
CHOICE_TARGET ("choice") is accepted anywhere a strategy name is: the
trial label is then the animal's lick/no-lick response (0/1), so the
same analyses measure information ABOUT CHOICE.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import special

CHOICE_TARGET = "choice"

STIM_BINNING_STRATEGIES = {
    # 1: contrast 0 | up to 80% of the curve | above 80%
    "zero_to80_above80": dict(
        kind="curve_pct", zero_separate=True, edges=[0.8],
        names=["0", "<=80%", ">80%"]),
    # 2: 0-20% | 20-80% | 80-100% of the curve (contrast 0 falls in the first bin)
    "pct_0-20_20-80_80-100": dict(
        kind="curve_pct", zero_separate=False, edges=[0.2, 0.8],
        names=["0-20%", "20-80%", "80-100%"]),
    # 3: no stimulus vs any stimulus
    "zero_vs_nonzero": dict(
        kind="contrast", zero_separate=False, edges=[0.0],
        names=["0", ">0"]),
    # every distinct stimulus value (contrast level) is its own bin: 0, 20, ..., 100 in the DN task (no psychometric grouping)
    "all_levels": dict(
        kind="levels", zero_separate=False, edges=[],
        names=["0", "20", "40", "60", "80", "100"]),
    # the previous hard-coded behaviour, kept for reference/comparison
    "quantile4": dict(kind="quantile", n_bins=4, zero_separate=False, edges=[]),
}

DEFAULT_STRATEGIES = ["zero_to80_above80", "pct_0-20_20-80_80-100", "zero_vs_nonzero"]


def psychometric_pct(signal, mu, sigma):
    """Position on the fitted curve in [0, 1]: Phi((signal - mu) / sigma).
    NaN if mu/sigma are missing."""
    signal = np.asarray(signal, float)
    if mu is None or sigma is None or not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
        return np.full(signal.shape, np.nan)
    return 0.5 * (1.0 + special.erf((signal - mu) / (np.sqrt(2.0) * sigma)))


def strategy_names(rule: dict) -> list[str]:
    if rule["kind"] == "levels":
        return list(rule.get("names") or [])
    n = len(rule["edges"]) + 1 + (1 if rule.get("zero_separate") else 0)
    if rule["kind"] == "quantile":
        n = rule["n_bins"]
    names = rule.get("names")
    return list(names) if names and len(names) == n else [str(i) for i in range(n)]


def bin_stimulus(signal, stim_pct, rule: dict, stim_z=None) -> np.ndarray:
    """Integer bin index per trial (float array, NaN where undefined).

    signal  : raw contrast per trial
    stim_pct: psychometric_pct(signal, mu, sigma) per trial
    rule    : one entry of STIM_BINNING_STRATEGIES
    stim_z  : z-scored contrast, only used by kind="quantile"
    """
    signal = np.asarray(signal, float)
    kind = rule["kind"]
    if kind == "levels":        # one bin per distinct stimulus value of the session, in increasing order
        out = np.full(signal.shape, np.nan)
        ok = np.isfinite(signal)
        if ok.any():
            v = np.round(signal[ok], 6)
            out[ok] = np.searchsorted(np.unique(v), v).astype(float)
        return out
    if kind == "quantile":
        from .info_theory import robust_qcut
        try:
            iv = robust_qcut(pd.Series(stim_z if stim_z is not None else signal), rule["n_bins"])
            codes = pd.Series(iv).astype("category").cat.codes.to_numpy().astype(float)
            codes[codes < 0] = np.nan
            return codes
        except (ValueError, TypeError):
            return np.full(signal.shape, np.nan)
    if kind == "curve_pct":
        v = np.asarray(stim_pct, float)
    elif kind == "contrast":
        v = signal
    else:
        raise KeyError(f"unknown binning kind {kind!r}")
    edges = np.asarray(rule["edges"], float)
    out = np.full(signal.shape, np.nan)
    ok = np.isfinite(v) & np.isfinite(signal)
    if rule.get("zero_separate"):
        shift = 1
        out[ok & (signal == 0)] = 0
        nz = ok & (signal != 0)
    else:
        shift, nz = 0, ok
    # right-inclusive: edges[i-1] < v <= edges[i]
    out[nz] = np.searchsorted(edges, v[nz], side="left") + shift
    return out


def trial_labels(table: pd.DataFrame, target: str, strategies: dict | None = None) -> np.ndarray:
    """Per-trial label array for a TARGET name: a strategy from
    STIM_BINNING_STRATEGIES (needs table columns `signal`, `stim_pct`,
    optionally `stim`) or CHOICE_TARGET (table column `choice`)."""
    strategies = strategies or STIM_BINNING_STRATEGIES
    if target == CHOICE_TARGET:
        return table["choice"].to_numpy().astype(float)
    return bin_stimulus(table["signal"].to_numpy(), table["stim_pct"].to_numpy(), strategies[target],
                        stim_z=table["stim"].to_numpy() if "stim" in table.columns else None)


def target_signature(targets, strategies: dict | None = None) -> str:
    """Text fingerprint of the definitions of `targets` -- stored with
    cached results so editing a strategy invalidates the cache."""
    strategies = strategies or STIM_BINNING_STRATEGIES
    return json.dumps({t: (strategies[t] if t != CHOICE_TARGET else "choice") for t in targets},
                      sort_keys=True, default=str)


def target_bin_names(target: str, strategies: dict | None = None) -> list[str]:
    strategies = strategies or STIM_BINNING_STRATEGIES
    return ["no lick", "lick"] if target == CHOICE_TARGET else strategy_names(strategies[target])


def describe_strategies(targets, strategies: dict | None = None) -> str:
    strategies = strategies or STIM_BINNING_STRATEGIES
    lines = []
    for t in targets:
        if t == CHOICE_TARGET:
            lines.append(f"  {t}: lick vs no-lick")
        else:
            r = strategies[t]
            lines.append(f"  {t}: {r['kind']}, zero_separate={r.get('zero_separate', False)}, "
                         f"edges={r.get('edges')} -> bins {strategy_names(r)}")
    return "\n".join(lines)


def _session_means(sub, value, session_col):
    """Mean of `value` within each session (pairs inside a session are not independent, so the
    SESSION is the unit of replication): returns an array of per-session means."""
    return sub.groupby(session_col)[value].mean().dropna().to_numpy()


def _draw_bar(ax, x, vals, width, color, rng, label=None):
    """Bar = mean over sessions, error bar = SEM over sessions, dots = individual session means."""
    vals = np.asarray(vals, float)
    if len(vals) == 0:
        return
    m = vals.mean()
    sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    ax.bar(x, m, width=width * 0.95, color=color, alpha=0.55, edgecolor="black", linewidth=0.3,
           yerr=sem, error_kw=dict(lw=0.9), label=label)
    ax.scatter(x + rng.uniform(-width * 0.25, width * 0.25, len(vals)), vals, s=10, color=color,
               edgecolor="black", linewidth=0.3, zorder=3)


def plot_target_comparison(df, value, windows, group_order, group_labels, targets, figdir, name,
                           ylabel=None, title=None, group_col="group", session_col="session_id"):
    """x = the six (area, label) groups, one bar colour per TARGET (binning strategy / choice), one
    panel per window. Bar = mean over SESSIONS of the per-session mean of `value`, error = SEM over
    sessions, dots = single sessions. `df` needs columns window, target, `group_col`, session_col, value."""
    import matplotlib.pyplot as plt
    from .groups import group_key
    from .plotting import save_fig
    rng = np.random.default_rng(0)
    colors = plt.cm.tab10(np.arange(len(targets)))
    fig, axes = plt.subplots(1, len(windows), figsize=(5.2 * len(windows), 4.5), squeeze=False)
    width = 0.8 / max(len(targets), 1)
    for ax, w in zip(axes[0], windows):
        sub = df[df["window"] == w]
        for ti, t in enumerate(targets):
            for gi, g in enumerate(group_order):
                s_ = sub[(sub["target"] == t) & (sub[group_col] == group_key(g))]
                x = gi + (ti - (len(targets) - 1) / 2) * width
                _draw_bar(ax, x, _session_means(s_, value, session_col), width, colors[ti], rng,
                          label=t if gi == 0 else None)
        ax.axhline(0, color="grey", linewidth=0.7)
        ax.set_xticks(np.arange(len(group_order)))
        ax.set_xticklabels([group_labels[g] for g in group_order], rotation=30)
        ax.set_title(w)
        ax.set_ylabel(ylabel or value)
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle(title or f"{value} by group and target", y=1.03)
    fig.tight_layout()
    save_fig(fig, figdir, name)
    plt.close(fig)


def plot_stim_vs_choice_corr(df, strategy, windows, group_order, group_labels, figdir, name, title=None,
                             group_col="group", session_col="session_id"):
    """Signal and noise correlation with the stimulus (binning `strategy`) and with CHOICE as the
    'signal', side by side. Rows: signal correlation, noise correlation. Columns: windows. x: the
    six groups; two bars per group (stimulus strategy vs. choice). Bar = mean over sessions, error =
    SEM over sessions, dots = sessions."""
    import matplotlib.pyplot as plt
    from .groups import group_key
    from .plotting import save_fig
    rng = np.random.default_rng(0)
    targets = [strategy, CHOICE_TARGET]
    colors = {strategy: "#4c72b0", CHOICE_TARGET: "#c44e52"}
    rows = [("signal_corr", "signal correlation"), ("noise_corr", "noise correlation")]
    fig, axes = plt.subplots(2, len(windows), figsize=(5.2 * len(windows), 8), squeeze=False)
    width = 0.38
    for ri, (val, nm) in enumerate(rows):
        for ax, w in zip(axes[ri], windows):
            sub = df[df["window"] == w]
            for ti, t in enumerate(targets):
                for gi, g in enumerate(group_order):
                    s_ = sub[(sub["target"] == t) & (sub[group_col] == group_key(g))]
                    x = gi + (ti - 0.5) * width
                    _draw_bar(ax, x, _session_means(s_, val, session_col), width, colors[t], rng,
                              label=("stimulus: " + t if t != CHOICE_TARGET else "choice") if gi == 0 else None)
            ax.axhline(0, color="grey", linewidth=0.7)
            ax.set_xticks(np.arange(len(group_order)))
            ax.set_xticklabels([group_labels[g] for g in group_order], rotation=30)
            ax.set_ylabel(nm)
            ax.set_title(f"{w} -- {nm}")
    axes[0][0].legend(fontsize=7, frameon=False)
    fig.suptitle(title or f"Signal / noise correlation: stimulus ({strategy}) vs. choice", y=1.0)
    fig.tight_layout()
    save_fig(fig, figdir, name)
    plt.close(fig)


def plot_group_dots(ax, df, value, group_order, group_labels, group_col="group", session_col="session_id",
                    colors=None, ylabel=None):
    """Replacement for violin plots: one bar per group = mean over sessions of the per-session
    mean of `value` (+- SEM over sessions), dots = single sessions, tick labels carry n sessions /
    n pairs. Interpretable at a glance, and honest about the unit of replication."""
    from .groups import group_key
    rng = np.random.default_rng(0)
    for gi, g in enumerate(group_order):
        s_ = df[df[group_col] == group_key(g)]
        col = (colors or {}).get(g, "#4c72b0")
        _draw_bar(ax, gi, _session_means(s_, value, session_col), 0.7, col, rng)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(group_order)))
    ax.set_xticklabels([f"{group_labels[g]}\n(n={df[df[group_col] == group_key(g)][session_col].nunique()} ses, "
                        f"{int((df[group_col] == group_key(g)).sum())} pairs)" for g in group_order],
                       rotation=30, fontsize=7)
    ax.set_ylabel(ylabel or value)
