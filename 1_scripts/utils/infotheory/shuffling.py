# -*- coding: utf-8 -*-
"""
shuffling.py
=============
Condition-conditioned trial-shuffling controls, generalizing this
project's existing "shuffle to destroy ALL structure" null tests
(info_theory.mutual_information_shuffle etc., which permute completely
at random) to also support shuffling WITHIN a condition -- same
stimulus, same choice, or the same (stimulus, choice) pair -- so that
whichever structure a shuffle preserves (each neuron's own tuning to
that condition) can be told apart from whichever structure it destroys
(trial-to-trial correlated variability beyond that condition, i.e.
"noise" correlations/information).

This is the general-purpose version of the shuffling described in Pola
et al. 2003 / Panzeri et al. 1999's Information Breakdown and reused in
the Panzeri lab's more recent MINT toolbox (Lorenz, Engel et al. 2025,
PLOS Comp Biol, https://doi.org/10.1371/journal.pcbi.1012934 --
hShuffle.m): "a shuffled response obtained by collecting randomly
permuted trials to the same stimulus" removes noise correlations while
preserving signal (tuning) correlations (see that paper's Fig 2 legend).

Built for 3a_pair_activity_statistics.py (pairwise noise/signal
correlations) and the 3b/3c_pair_information_*.py Information
Breakdown/PID scripts, which need the same conditioned-shuffle controls
to isolate the contribution of correlated variability -- and kept fully
general (works on any two-array statistic function, not hardcoded to
correlation or MI) specifically so the later population/dimensionality-
reduction ("section 4") scripts can reuse it too.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

SHUFFLE_MODES = ("global", "stim", "choice", "stim_choice")


def make_shuffle_groups(mode: str, stim=None, choice=None, n: int | None = None) -> np.ndarray:
    """Build the group-label array `shuffle_within_groups` shuffles
    within, for one of the four standard conditioned-shuffle controls:
        "global"      : one group for everyone -- unconstrained
                        shuffle, destroys ALL structure (the null this
                        project's other MI-shuffle tests already use)
        "stim"        : group = stimulus condition -- destroys noise
                        correlations while preserving each neuron's
                        own stimulus tuning (signal correlations)
        "choice"      : group = choice (0/1) -- destroys correlations
                        beyond what's explained by shared choice
        "stim_choice" : group = (stimulus, choice) jointly -- destroys
                        correlations beyond what's explained by BOTH

    `stim`/`choice` should already be discretized/binned (e.g. via
    robust_qcut) if continuous -- grouping needs a finite number of
    distinct condition values, not raw continuous ones, or every trial
    would end up alone in its own group and the shuffle would become a
    no-op (nothing left to permute within a group of size 1).
    """
    if mode not in SHUFFLE_MODES:
        raise ValueError(f"Unknown shuffle mode {mode!r}; must be one of {SHUFFLE_MODES}")
    if mode == "global":
        if n is None:
            n = len(stim) if stim is not None else len(choice)
        return np.zeros(n, dtype=int)
    if mode == "stim":
        if stim is None:
            raise ValueError("mode='stim' requires `stim`")
        return np.asarray(stim)
    if mode == "choice":
        if choice is None:
            raise ValueError("mode='choice' requires `choice`")
        return np.asarray(choice)
    if mode == "stim_choice":
        if stim is None or choice is None:
            raise ValueError("mode='stim_choice' requires both `stim` and `choice`")
        return np.array([f"{s}_{c}" for s, c in zip(stim, choice)])


def shuffle_within_groups(y: np.ndarray, groups: np.ndarray,
                           rng: np.random.Generator | None = None) -> np.ndarray:
    """Permute `y` WITHIN each distinct value of `groups` -- leaves
    cross-group structure (e.g. each condition's own mean/tuning)
    intact while destroying within-group structure (e.g. trial-to-
    trial correlated variability at fixed condition -- "noise"
    correlations). Groups with only 1 member are left as-is (nothing to
    permute)."""
    rng = rng or np.random.default_rng()
    y = np.asarray(y)
    groups = np.asarray(groups)
    y_shuffled = y.copy()
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        if len(idx) > 1:
            y_shuffled[idx] = y[rng.permutation(idx)]
    return y_shuffled


def make_shuffle_groups_from_conditions(conditions: dict | None, n: int | None = None) -> np.ndarray:
    """General group-label builder for shuffle_within_groups /
    shuffle_population_within_groups -- the modular counterpart to
    make_shuffle_groups above (which hardcodes the stim/choice/
    stim_choice combination specifically). `conditions` is a dict
    mapping a condition NAME (used only for bookkeeping/labeling
    elsewhere) to an already-discretized 1D array -- e.g.
    {"stim": stim_bin}, {"stim": stim_bin, "choice": choice}, or, to
    add a behavioral variable later with zero changes to any shuffle
    code, {"cur_runspeed": runspeed_bin}. Every condition given is
    combined jointly (AND semantics: two trials land in the same group
    only if EVERY listed condition matches) -- pass as many or as few
    as needed.

    `conditions=None` or `{}` -> one big group (global/unconstrained
    shuffle, destroys everything, matching mode="global" above).

    Built for 4a/4b/4c_population_*.py's population-level shuffle
    tests, which need an easily-extended set of named conditions rather
    than the three fixed modes make_shuffle_groups supports.
    """
    if not conditions:
        if n is None:
            raise ValueError("n is required when conditions is empty/None")
        return np.zeros(n, dtype=int)
    arrays = [np.asarray(v) for v in conditions.values()]
    n = len(arrays[0])
    # String-joined keys, not raw tuples: comparing an object array of
    # tuples against a single tuple (as shuffle_within_groups does via
    # `groups == g`) hits a numpy broadcasting ambiguity -- numpy tries
    # to broadcast a 2-element tuple against the array elementwise
    # instead of treating it as one scalar object, corrupting the
    # group masks. Strings compare as plain scalars with no such issue.
    combined = np.array(["_".join(str(a[idx]) for a in arrays) for idx in range(n)])
    return combined


def shuffle_population_within_groups(response_matrix: np.ndarray, groups: np.ndarray,
                                      rng: np.random.Generator | None = None) -> np.ndarray:
    """Population-scale generalization of shuffle_within_groups: applies
    an INDEPENDENT random permutation to each COLUMN (cell) of
    `response_matrix`, each restricted within `groups` -- i.e. calls
    shuffle_within_groups separately per cell. Destroys cross-cell
    (pairwise AND higher-order) correlations within each group while
    preserving each cell's own group-conditioned distribution exactly
    (same values, just independently reassigned to different trials
    within the group). Used by 4a/4b/4c_population_*.py to test how
    much of a population-level result (dimensionality, decoding
    accuracy, ...) depends on real cross-cell correlation structure
    versus what independent cells with the same individual tuning would
    give."""
    rng = rng or np.random.default_rng()
    response_matrix = np.asarray(response_matrix, dtype=float)
    shuffled = np.empty_like(response_matrix)
    for c in range(response_matrix.shape[1]):
        shuffled[:, c] = shuffle_within_groups(response_matrix[:, c], groups, rng=rng)
    return shuffled


@dataclass
class ShuffleTestResult:
    n: int
    stat_raw: float
    stat_shuffle_mean: float
    stat_shuffle_std: float
    stat_corrected: float   # raw - shuffle_mean (NOT floored at 0, unlike this project's MI
                             # results -- correlation-like statistics can meaningfully come out
                             # below their null mean, e.g. a negative noise correlation)
    p_value: float          # two-tailed: fraction of |shuffle - shuffle_mean| >= |raw - shuffle_mean|


def conditioned_shuffle_test(statistic_fn: Callable[[np.ndarray, np.ndarray], float],
                              x, y, mode: str = "global", stim=None, choice=None,
                              n_shuffles: int = 200,
                              rng: np.random.Generator | None = None) -> ShuffleTestResult:
    """General conditioned-shuffle significance test: compute
    `statistic_fn(x, y)` on the real data, then on `n_shuffles`
    surrogates where y is permuted WITHIN groups defined by `mode` (see
    make_shuffle_groups) -- e.g. mode="stim" tests whether x and y are
    related BEYOND what their shared stimulus tuning predicts (the
    surrogate preserves each neuron's own stimulus-conditioned mean,
    only scrambling trial identity within each stimulus value).

    Works with ANY two-array statistic (Pearson correlation, MI,
    covariance, ...) -- pass e.g. `lambda a, b: np.corrcoef(a, b)[0, 1]`,
    or one of this project's mutual_information_hist-based functions.

    Complements (does not replace) directly computing a "noise
    correlation" as the correlation of stimulus-conditioned RESIDUALS
    (see 3a_pair_activity_statistics.py): that gives you the value,
    this tells you whether it's bigger than chance while explicitly
    controlling for exactly which structure ("stim", "choice", or both)
    should NOT count as part of the effect.
    """
    rng = rng or np.random.default_rng()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if stim is not None:
        mask &= np.isfinite(np.asarray(stim, dtype=float)) if np.asarray(stim).dtype.kind == "f" else True
    x, y = x[mask], y[mask]
    stim_m = np.asarray(stim)[mask] if stim is not None else None
    choice_m = np.asarray(choice)[mask] if choice is not None else None
    groups = make_shuffle_groups(mode, stim=stim_m, choice=choice_m, n=len(x))

    stat_raw = float(statistic_fn(x, y))
    shuffle_stats = np.array([float(statistic_fn(x, shuffle_within_groups(y, groups, rng=rng)))
                               for _ in range(n_shuffles)])
    mean_s, std_s = float(np.mean(shuffle_stats)), float(np.std(shuffle_stats))
    corrected = stat_raw - mean_s
    p_value = float(np.mean(np.abs(shuffle_stats - mean_s) >= np.abs(stat_raw - mean_s)))
    return ShuffleTestResult(n=len(x), stat_raw=stat_raw, stat_shuffle_mean=mean_s,
                              stat_shuffle_std=std_s, stat_corrected=corrected, p_value=p_value)
