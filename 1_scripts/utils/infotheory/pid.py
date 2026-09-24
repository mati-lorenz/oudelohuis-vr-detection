# -*- coding: utf-8 -*-
"""
pid.py
=======
Partial Information Decomposition (PID, Williams & Beer 2010) for two
"source" variables (e.g. two cells' responses) and one "target" (e.g.
stimulus or choice): splits the total information the pair carries
about the target into REDUNDANT (both sources carry it), UNIQUE-to-
each-source, and SYNERGISTIC (only available from the pair jointly, not
either source alone) components.

    I(S1;T)      = R + U1
    I(S2;T)      = R + U2
    I(S1,S2;T)   = R + U1 + U2 + Syn

Only ONE additional quantity beyond the three ordinary mutual
informations above is needed to solve this system: the REDUNDANCY R.
Different PID measures in the literature are, in this sense, just
different choices of how to define/compute R from the joint
distribution -- which is exactly why `partial_information_decomposition`
below takes a `redundancy_fn` argument rather than hardcoding one.

Default redundancy measure: I_min (Williams & Beer 2010) -- the
original, simplest PID redundancy measure. Well known to sometimes
OVERSTATE redundancy in some distributions (see e.g. Bertschinger et
al. 2014's "two-bit-copy" example); this is exactly why the more recent
BROJA-2PID measure (Bertschinger, Rauh, Olbrich, Jost & Ay 2014;
Makkeh, Theis & Vicente 2018) exists, and why this module is structured
to make swapping it in later a one-function change (see
`redundancy_broja_STUB` below and REDUNDANCY_MEASURES) rather than a
rewrite of this whole file.

Validated against the two textbook PID cases before being used
anywhere in this project's pipeline (see this module's own tests, run
once during development): a fully-redundant "copy" distribution
(S1=S2=T) gives R=I(S1;T), U1=U2=Syn=0; an XOR gate (T = S1 XOR S2, S1/
S2 independent fair coins) gives R=U1=U2=0, Syn=1 bit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .info_theory import quantile_bin_edges


def digitize_quantile(x: np.ndarray, bins: int) -> np.ndarray:
    """Quantile-bin `x` into `bins` levels, returning integer bin
    indices 0..bins-1 (reuses this project's own robust quantile edges,
    so low-cardinality/discrete inputs don't collapse to 1 bin)."""
    edges = quantile_bin_edges(x, bins)
    return np.clip(np.digitize(x, edges[1:-1]), 0, bins - 1)


def build_joint_distribution(s1_bin: np.ndarray, s2_bin: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Empirical joint P(s1, s2, t) as a normalized 3D array, shape
    (n_bins_s1, n_bins_s2, n_t_levels). `t` can be any discrete/binned
    array (arbitrary values, not necessarily 0..n-1) -- its distinct
    values are used as-is, in sorted order, for the last axis."""
    t_levels, t_idx = np.unique(t, return_inverse=True)
    n1, n2 = int(s1_bin.max()) + 1, int(s2_bin.max()) + 1
    counts = np.zeros((n1, n2, len(t_levels)))
    for a, b, c in zip(s1_bin, s2_bin, t_idx):
        counts[a, b, c] += 1
    total = counts.sum()
    if total == 0:
        raise ValueError("Empty joint distribution -- no valid samples.")
    return counts / total


def _mi_from_joint_2d(P_x_t: np.ndarray) -> float:
    """MI (bits) between a single discrete variable and target, given
    their joint probability table P(x, t)."""
    P_x = P_x_t.sum(axis=1, keepdims=True)
    P_t = P_x_t.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = P_x_t / (P_x * P_t)
        terms = np.where(P_x_t > 0, P_x_t * np.log2(ratio), 0.0)
    return float(np.nansum(terms))


def _specific_info_per_target(P_x_t: np.ndarray) -> np.ndarray:
    """I_spec(T=t; X) for every t -- the POINTWISE (per target-outcome)
    specific information a single source carries, as opposed to
    I(X;T) which is the average of this over t weighted by P(t).
    I_spec(t) = sum_x P(x|t) log2[P(t|x)/P(t)]."""
    P_x = P_x_t.sum(axis=1, keepdims=True)
    P_t = P_x_t.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        P_t_given_x = P_x_t / P_x
        log_ratio = np.log2(P_t_given_x / P_t)
        P_x_given_t = P_x_t / P_t
        terms = np.where(P_x_t > 0, P_x_given_t * log_ratio, 0.0)
    return np.nansum(terms, axis=0)  # sum over x -> shape (n_t,)


def redundancy_imin(P_joint: np.ndarray) -> float:
    """Williams & Beer (2010) I_min redundancy: for each target
    outcome, take whichever source gives the smaller specific
    information, then average over target outcomes weighted by P(t)."""
    P_s1_t = P_joint.sum(axis=1)  # (n_s1, n_t)
    P_s2_t = P_joint.sum(axis=0)  # (n_s2, n_t)
    ispec1 = _specific_info_per_target(P_s1_t)
    ispec2 = _specific_info_per_target(P_s2_t)
    P_t = P_joint.sum(axis=(0, 1))
    return float(np.sum(P_t * np.minimum(ispec1, ispec2)))


def redundancy_broja_STUB(P_joint: np.ndarray) -> float:
    """NOT IMPLEMENTED -- placeholder showing where to plug in
    BROJA-2PID (Bertschinger et al. 2014's redundancy measure, robustly
    estimated via convex/cone-programming optimization by Makkeh,
    Theis & Vicente 2018: https://github.com/Abzinger/BROJA_2PID, module
    `broja2pid`). Needs the ECOS cone-programming solver installed
    (`pip install ecos`) plus the BROJA_2PID package itself (not on
    PyPI as of this writing -- clone the GitHub repo).

    Sketch of the integration (confirm the exact dict key names against
    your installed version before trusting this -- written from the
    package's documented usage pattern, not by importing and running it):

        from broja2pid import pid as _broja_pid
        def redundancy_broja(P_joint):
            n1, n2, nt = P_joint.shape
            pdf = {(i, j, k): P_joint[i, j, k]
                   for i in range(n1) for j in range(n2) for k in range(nt)
                   if P_joint[i, j, k] > 0}
            result = _broja_pid(pdf, cone_solver="ECOS", output=0)
            return result["SI"]  # TODO: confirm this is the shared/redundant-information key

    Then register it: REDUNDANCY_MEASURES["broja"] = redundancy_broja,
    and everything downstream (partial_information_decomposition,
    3c_pair_information_pid.py's REDUNDANCY_MEASURE constant) works
    unchanged -- the whole point of routing every measure through
    REDUNDANCY_MEASURES / the redundancy_fn argument.
    """
    raise NotImplementedError(
        "BROJA-2PID is not wired in yet -- see this function's docstring for the integration "
        "sketch. Use redundancy_imin (REDUNDANCY_MEASURE='imin') for now.")


REDUNDANCY_MEASURES: dict[str, Callable[[np.ndarray], float]] = {
    "imin": redundancy_imin,
    "broja": redundancy_broja_STUB,
}


@dataclass
class PIDResult:
    n: int
    I1: float
    I2: float
    I12: float
    redundancy: float
    unique1: float
    unique2: float
    synergy: float
    redundancy_shuffle_mean: float | None = None
    redundancy_p_value: float | None = None


def partial_information_decomposition(P_joint: np.ndarray, redundancy_measure: str = "imin") -> PIDResult:
    """Solve R/U1/U2/Syn from a joint P(s1, s2, t) table, using the
    named entry in REDUNDANCY_MEASURES for R. See module docstring for
    the linear system being solved."""
    if redundancy_measure not in REDUNDANCY_MEASURES:
        raise ValueError(f"Unknown redundancy_measure {redundancy_measure!r}; "
                          f"must be one of {list(REDUNDANCY_MEASURES)}")
    redundancy_fn = REDUNDANCY_MEASURES[redundancy_measure]

    P_s1_t = P_joint.sum(axis=1)
    P_s2_t = P_joint.sum(axis=0)
    n1, n2, nt = P_joint.shape
    P_s12_t = P_joint.reshape(n1 * n2, nt)

    I1 = _mi_from_joint_2d(P_s1_t)
    I2 = _mi_from_joint_2d(P_s2_t)
    I12 = _mi_from_joint_2d(P_s12_t)
    R = redundancy_fn(P_joint)
    U1 = I1 - R
    U2 = I2 - R
    Syn = I12 - R - U1 - U2

    return PIDResult(n=0, I1=I1, I2=I2, I12=I12, redundancy=R, unique1=U1, unique2=U2, synergy=Syn)


def pid_from_binned_responses(x1: np.ndarray, x2: np.ndarray, t: np.ndarray, bins: int = 3,
                               redundancy_measure: str = "imin", n_shuffles: int = 0,
                               rng: np.random.Generator | None = None) -> PIDResult:
    """Convenience wrapper: quantile-bin two continuous responses,
    build their joint distribution with `t`, and run
    `partial_information_decomposition`. If `n_shuffles` > 0, also
    bias-corrects the redundancy value the same way every other MI
    estimate in this project is corrected: subtract the mean redundancy
    obtained from `n_shuffles` realizations where `t` is fully shuffled
    (global shuffle -- see infotheory.shuffling -- destroys everything,
    giving the finite-sample-bias null for a distribution with no real
    structure at all)."""
    x1, x2, t = np.asarray(x1, dtype=float), np.asarray(x2, dtype=float), np.asarray(t)
    valid = np.isfinite(x1) & np.isfinite(x2)
    x1, x2, t = x1[valid], x2[valid], t[valid]
    s1_bin, s2_bin = digitize_quantile(x1, bins), digitize_quantile(x2, bins)

    P_joint = build_joint_distribution(s1_bin, s2_bin, t)
    result = partial_information_decomposition(P_joint, redundancy_measure=redundancy_measure)
    result.n = len(x1)

    if n_shuffles > 0:
        rng = rng or np.random.default_rng()
        shuffle_r = []
        for _ in range(n_shuffles):
            t_shuf = rng.permutation(t)
            P_shuf = build_joint_distribution(s1_bin, s2_bin, t_shuf)
            shuffle_r.append(REDUNDANCY_MEASURES[redundancy_measure](P_shuf))
        mean_r = float(np.mean(shuffle_r))
        result.redundancy_shuffle_mean = mean_r
        result.redundancy_p_value = float(np.mean(np.array(shuffle_r) >= result.redundancy))
        # re-derive U1/U2/Syn from the bias-corrected redundancy, keeping I1/I2/I12 as-is
        result.redundancy = result.redundancy - mean_r
        result.unique1 = result.I1 - result.redundancy
        result.unique2 = result.I2 - result.redundancy
        result.synergy = result.I12 - result.redundancy - result.unique1 - result.unique2

    return result
