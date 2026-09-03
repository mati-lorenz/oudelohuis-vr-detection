# -*- coding: utf-8 -*-
"""
pid.py
=======
Partial Information Decomposition (PID) of the information that a single
response variable R (e.g. one neuron's discretized activity) carries about
two source/predictor variables jointly, e.g. stimulus S and choice C:

    I(R; S, C) = Red(R; S, C) + Unq(R; S\\C) + Unq(R; C\\S) + Syn(R; S, C)

    Red  : redundant information about R carried by both S and C
    Unq_S: information about R carried uniquely by S
    Unq_C: information about R carried uniquely by C
    Syn  : synergistic information about R only available when S and C are
           known jointly

This module implements the classic Williams & Beer (2010) redundancy
measure I_min, which is the simplest, most widely used and most
tractable (non-negative, exact for any discrete distribution, no
optimization required) two-source PID measure and is a standard choice
in the systems-neuroscience literature for stimulus/choice/outcome
decompositions (see also Timme & Lapish, 2018, eNeuro, for review, and
Lorenz et al. 2025's MINT toolbox, which offers I_min alongside I_MMI
and BROJA as interchangeable redundancy measures for exactly this kind
of PID).

Validated against the three canonical PID "logic gate" test cases
(see tests/test_pid_canonical.py): XOR (pure synergy, 1 bit), COPY
(pure unique information), and RDN (S1==S2, pure redundancy) all
reproduce their textbook values exactly.

Caveat: I_min is known to overestimate redundancy in some special cases
(it does not fully satisfy the "identity" axiom for continuous-analog
constructions). If exact reproduction of a specific published PID
decomposition is required (e.g. the BROJA-2PID / I_ccs measure), swap out
`redundancy_imin` for another redundancy function -- every other quantity
in this module (unique information, synergy) is defined generically in
terms of whatever redundancy function is supplied, so this is a one-line
change (see `pid_decomposition(..., redundancy_fn=...)`).

Bias correction: as demonstrated in tests/test_pid_canonical.py, the
naive plug-in PID terms carry the same kind of limited-sampling bias as
any other direct-method information estimate (Panzeri & Treves, 1996)
-- at realistic trial counts, even a genuinely INDEPENDENT (r unrelated
to s1, s2) case shows spurious nonzero synergy/unique/redundancy. Pass
`shuffle_correction=True` to `pid_decomposition` to subtract a
shuffle-based bias estimate from every term (shuffling r's trial order,
which destroys the R-{S1,S2} relationship while preserving the S1-S2
joint distribution), matching the "shuffle-subtraction" method MINT
uses throughout (Lorenz et al. 2025).
"""

import numpy as np
from .core import _as_labels, LOG2, get_worker_rng

TERM_KEYS_GENERIC = ['I_total', 'I_1', 'I_2', 'redundancy', 'unique_1', 'unique_2', 'synergy']


def _joint_counts_3d(r, s1, s2):
    """Joint counts n(r, s1, s2) as a 3D integer array."""
    r = _as_labels(r)
    s1 = _as_labels(s1)
    s2 = _as_labels(s2)
    n_r, n_s1, n_s2 = r.max() + 1, s1.max() + 1, s2.max() + 1
    counts = np.zeros((n_r, n_s1, n_s2), dtype=int)
    np.add.at(counts, (r, s1, s2), 1)
    return counts


def redundancy_imin(r, s1, s2):
    """
    Williams & Beer (2010) I_min redundancy between two source variables
    s1, s2 about a target r:

        I_min(R; {S1, S2}) = sum_r p(r) * min( I_spec(r; S1), I_spec(r; S2) )

    where I_spec(r; Si) = sum_{si} p(si|r) log2( p(r|si) / p(r) ) is the
    "specific information" that observing R=r provides about Si.

    Returns
    -------
    red : float, redundancy in bits
    """
    r = _as_labels(r)
    s1 = _as_labels(s1)
    s2 = _as_labels(s2)
    n_r = r.max() + 1
    n_s1 = s1.max() + 1
    n_s2 = s2.max() + 1
    n = len(r)

    counts_r_s1 = np.zeros((n_r, n_s1), dtype=int)
    counts_r_s2 = np.zeros((n_r, n_s2), dtype=int)
    np.add.at(counts_r_s1, (r, s1), 1)
    np.add.at(counts_r_s2, (r, s2), 1)

    p_r = np.bincount(r, minlength=n_r) / n
    p_r_s1 = counts_r_s1 / n            # joint p(r, s1)
    p_r_s2 = counts_r_s2 / n            # joint p(r, s2)

    red = 0.0
    for ri in range(n_r):
        if p_r[ri] == 0:
            continue
        spec1 = 0.0
        p_s1_given_r = p_r_s1[ri, :] / p_r[ri]
        for si in range(n_s1):
            if p_s1_given_r[si] > 0:
                p_s1_marg = p_r_s1[:, si].sum()
                p_r_given_s1 = p_r_s1[ri, si] / p_s1_marg
                spec1 += p_s1_given_r[si] * np.log(p_r_given_s1 / p_r[ri]) / LOG2

        spec2 = 0.0
        p_s2_given_r = p_r_s2[ri, :] / p_r[ri]
        for si in range(n_s2):
            if p_s2_given_r[si] > 0:
                p_s2_marg = p_r_s2[:, si].sum()
                p_r_given_s2 = p_r_s2[ri, si] / p_s2_marg
                spec2 += p_s2_given_r[si] * np.log(p_r_given_s2 / p_r[ri]) / LOG2

        red += p_r[ri] * min(spec1, spec2)

    return float(red)


def _pid_terms(r, s1, s2, redundancy_fn, s1_name, s2_name):
    """
    Core computation: PID of I(R; S1, S2) given already-discretized
    r, s1, s2. Pure function -- reused for both the real data and, with
    r permuted, for each shuffle-correction draw (see
    `pid_decomposition(..., shuffle_correction=True)`).
    """
    from .core import naive_mutual_information

    r = _as_labels(r)
    s1 = _as_labels(s1)
    s2 = _as_labels(s2)

    joint = s1 * (s2.max() + 1) + s2

    i_total = naive_mutual_information(r, joint)
    i_s1 = naive_mutual_information(r, s1)
    i_s2 = naive_mutual_information(r, s2)

    red = redundancy_fn(r, s1, s2)
    unq1 = max(i_s1 - red, 0.0)
    unq2 = max(i_s2 - red, 0.0)
    syn = i_total - red - unq1 - unq2

    return {
        'I_total': float(i_total),
        f'I_{s1_name}': float(i_s1),
        f'I_{s2_name}': float(i_s2),
        'redundancy': float(red),
        f'unique_{s1_name}': float(unq1),
        f'unique_{s2_name}': float(unq2),
        'synergy': float(syn),
    }


def pid_decomposition(r, s1, s2, redundancy_fn=redundancy_imin,
                       s1_name='S1', s2_name='S2',
                       shuffle_correction=False, n_shuffles=30,
                       random_state=None, unit_index=0):
    """
    Full two-source PID of I(R; S1, S2).

    Parameters
    ----------
    r  : 1D array, discretized response (single neuron, or later a
         discretized population/PC score)
    s1, s2 : 1D arrays, the two discrete source variables (e.g. stimulus
         and choice)
    redundancy_fn : callable(r, s1, s2) -> float
        the redundancy measure to use; defaults to Williams-Beer I_min.
    s1_name, s2_name : used only to label the returned dict's keys.
    shuffle_correction : bool
        if True, ALSO report shuffle-subtraction bias-corrected versions
        of all seven terms (`*_shuffcorr` keys). Recommended at
        realistic trial counts -- see module docstring and
        tests/test_pid_canonical.py for why. Shuffles r's trial order
        (destroys the R-{S1,S2} relationship while preserving the S1-S2
        joint distribution, so a real behavioral correlation between
        e.g. stimulus and choice is not itself treated as bias).
    n_shuffles : int
    random_state, unit_index : seeding (see core.get_worker_rng)

    Returns
    -------
    dict with keys:
        'I_total', 'I_<s1_name>', 'I_<s2_name>', 'redundancy',
        'unique_<s1_name>', 'unique_<s2_name>', 'synergy'   (naive)
        same keys with '_shuffcorr' suffix (only if shuffle_correction)
    """
    out = _pid_terms(r, s1, s2, redundancy_fn, s1_name, s2_name)

    if shuffle_correction:
        r = _as_labels(r)
        s1 = _as_labels(s1)
        s2 = _as_labels(s2)
        rng = get_worker_rng(random_state, unit_index)
        n = len(r)
        term_keys = list(out.keys())
        shuf_sums = {k: 0.0 for k in term_keys}
        for _ in range(n_shuffles):
            r_shuf = r[rng.permutation(n)]
            terms_shuf = _pid_terms(r_shuf, s1, s2, redundancy_fn, s1_name, s2_name)
            for k in term_keys:
                shuf_sums[k] += terms_shuf[k]
        for k in term_keys:
            out[f'{k}_shuffcorr'] = out[k] - shuf_sums[k] / n_shuffles

    return out
