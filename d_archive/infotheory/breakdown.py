# -*- coding: utf-8 -*-
"""
breakdown.py
=============
Information breakdown for pairs (and, in later pipeline stages,
populations) of simultaneously recorded neurons, following the exact
decomposition introduced by Pola, Thiele, Hoffmann & Panzeri (2003,
Network: Comput. Neural Syst. 14:35-60), Eqs. 11-17 (as reproduced e.g.
in Ince, Schultz & Panzeri, "Estimating Information-Theoretic
Quantities", 2015). Terminology and validated qualitative behavior
below follow Lorenz, Engel, Celotto et al. (2025) "MINT: a toolbox for
the analysis of multivariate neural information coding and
transmission", PLoS Comput Biol 21(4):e1012934, which implements the
same decomposition (their I_ss / I_ci / I_cd) and documents two
canonical validation scenarios reproduced in
`tests/test_breakdown_canonical.py` alongside this module.

The total information that a pair of neurons (R1, R2) carries jointly
about a stimulus/task variable S is decomposed into four exact,
additive terms:

    I(R1,R2; S) = I_lin + I_sig-sim + I_cor-indep + I_cor-dep

    I_lin       : "linear" term = I(R1;S) + I(R2;S), the information
                  available if the two cells were both independent AND
                  had independent tuning.
    I_sig-sim   : correction for signal similarity -- how much the
                  cells' average tuning curves overlap/oppose each
                  other, evaluated with noise correlations removed.
    I_cor-indep : contribution of the AVERAGE (stimulus-independent)
                  level of noise correlation.
    I_cor-dep   : contribution of genuine STIMULUS-DEPENDENT modulation
                  of the noise correlation.

REVISION HISTORY
-----------------
v1 (surrogate resampling): estimated I_sig-sim/I_cor-indep/I_cor-dep by
building surrogate datasets via random resampling and differencing
naive MI computed on them. Abandoned: this is not how Pola et al. (2003)
define these quantities, and differencing naive MI estimates with
different, resampling-induced biases made I_cor-indep/I_cor-dep saturate
to opposite extremes at realistic trial counts.

v2 (exact closed-form, naive plug-in): replaced the surrogates with the
exact closed-form Pola et al. (2003) expressions (Eqs. 14-17 below),
built deterministically from the estimated probability tables -- no
resampling. This guarantees I_lin + I_sig-sim + I_cor-indep + I_cor-dep
== I_full exactly for the NAIVE (plug-in) probability estimates. This
is correct as an algebraic identity, but plain naive plug-in
probabilities are themselves biased (Panzeri & Treves 1996), and this
bias does not cancel between I_full and I_lin: reproducing the MINT
paper's own canonical simulated example (Fig 2B / SM6.1) at their exact
trial count (200 trials/stimulus) showed the sign of the (small)
redundancy-synergy index flipping from the true negative value to a
spurious positive one purely from this naive bias, converging back to
the correct sign only when trial counts were pushed far beyond
realistic experimental sizes.

v3 (this version): adds SHUFFLE-SUBTRACTION bias correction across the
full breakdown, exactly as MINT does for these examples (SM6.1: "All MI
and PID values were corrected for the limited-sampling bias by using
the shuffle-subtraction procedure ... averaged over 30 shuffles").
Because shuffling only the trial-to-stimulus LABEL assignment (not the
responses, and not the binning) leaves the response alphabet and joint
occupancy pattern statistically exchangeable with the real data, the
naive bias of each of the four terms (plus I_full, I_R1, I_R2) computed
on shuffled data is, to leading order, the SAME as its bias on the real
data -- so subtracting the shuffle-averaged term from the real term
cancels the leading-order bias for EACH term individually, while the
exact-sum identity is preserved automatically (it is linear, so it
holds for the real terms, the shuffle-averaged terms, and therefore
their difference). See `pairwise_information_breakdown(...,
shuffle_correction=True)`.

Definitions
------------
For a joint pair-response symbol r = (r1, r2):

    P_ind(r|s) = P(r1|s) * P(r2|s)                         (product of
        each cell's OWN stimulus-conditional response distribution)
    P_ind(r)   = <P_ind(r|s)>_s = sum_s P(s) P_ind(r|s)     (stimulus-
        averaged independent-model response distribution)
    nu(r)      = P_ind(r) / (P(r1) P(r2)) - 1               (signal
        correlation density)
    gamma(r|s) = P(r|s) / P_ind(r|s) - 1  (0 if P_ind(r|s) = 0)
                                                              (noise
        correlation density at fixed stimulus s)

the four terms are (Pola et al. 2003, Eqs. 14-17):

    I_lin       = I(R1;S) + I(R2;S)
    I_sig-sim   = (1/ln2) * sum_r [P(r1)P(r2)] *
                            [nu(r) + (1+nu(r)) * ln(1/(1+nu(r)))]
    I_cor-indep = sum_r [P(r) - P_ind(r)] * log2(1/(1+nu(r)))
    I_cor-dep   = sum_s P(s) sum_r P(r|s) *
                            log2[ P_ind(r) P(r|s) / (P_ind(r|s) P(r)) ]
"""

import numpy as np
from . import core
from .core import mutual_information, naive_mutual_information, _as_labels, LOG2
from .discretize import discretize_response

TERM_KEYS = ['I_full', 'I_R1', 'I_R2', 'I_lin', 'I_sig_sim', 'I_cor_indep', 'I_cor_dep']


def _joint_response_labels(r1_binned, r2_binned, n_bins_r2):
    """Combine two already-binned response vectors into one joint symbol
    (Cartesian product of bin labels), used to treat the pair as a single
    'population response' for the direct-method MI estimate."""
    return r1_binned * n_bins_r2 + r2_binned


def _breakdown_terms_from_binned(r1_binned, r2_binned, s, n1, n2):
    """
    Core computation: the exact Pola et al. (2003) four-term breakdown
    (Eqs. 14-17) given ALREADY-BINNED responses and integer stimulus
    labels. Pure function of (r1_binned, r2_binned, s) -- used both for
    the real data and, with `s` permuted, for each shuffle-correction
    draw, so that only the stimulus-to-trial association changes and
    the response alphabet/binning is identical in both cases.

    Returns
    -------
    dict with I_full, I_R1, I_R2, I_lin, I_sig_sim, I_cor_indep, I_cor_dep
    """
    n_s = int(s.max()) + 1
    n = len(s)

    counts_r1_s = np.zeros((n1, n_s), dtype=int)
    counts_r2_s = np.zeros((n2, n_s), dtype=int)
    counts_joint_s = np.zeros((n1, n2, n_s), dtype=int)
    np.add.at(counts_r1_s, (r1_binned, s), 1)
    np.add.at(counts_r2_s, (r2_binned, s), 1)
    np.add.at(counts_joint_s, (r1_binned, r2_binned, s), 1)

    n_s_trials = counts_joint_s.sum(axis=(0, 1))            # (n_s,)
    p_s = n_s_trials / n
    safe_n_s_trials = np.where(n_s_trials > 0, n_s_trials, 1)

    p_r1_given_s = counts_r1_s / safe_n_s_trials[None, :]     # (n1, n_s)
    p_r2_given_s = counts_r2_s / safe_n_s_trials[None, :]     # (n2, n_s)
    p_joint_given_s = counts_joint_s / safe_n_s_trials[None, None, :]  # (n1,n2,n_s)

    p_r1 = counts_r1_s.sum(axis=1) / n                        # (n1,)
    p_r2 = counts_r2_s.sum(axis=1) / n                        # (n2,)
    p_joint = counts_joint_s.sum(axis=2) / n                  # (n1, n2)

    # P_ind(r|s) = P(r1|s) * P(r2|s)  (exact independent-model response)
    p_ind_given_s = p_r1_given_s[:, None, :] * p_r2_given_s[None, :, :]   # (n1,n2,n_s)
    p_ind = (p_ind_given_s * p_s[None, None, :]).sum(axis=2)              # (n1,n2)

    # nu(r): signal correlation density
    denom_signal = p_r1[:, None] * p_r2[None, :]                          # (n1,n2)
    valid_signal = denom_signal > 0
    nu = np.zeros((n1, n2))
    nu[valid_signal] = p_ind[valid_signal] / denom_signal[valid_signal] - 1.0
    one_plus_nu = 1.0 + nu
    pos = one_plus_nu > 0

    # I_full, I_R1, I_R2, I_lin (naive)
    joint_full = _joint_response_labels(r1_binned, r2_binned, n2)
    i_full = naive_mutual_information(joint_full, s)
    i_r1 = naive_mutual_information(r1_binned, s)
    i_r2 = naive_mutual_information(r2_binned, s)
    i_lin = i_r1 + i_r2

    # I_sig-sim: Eq. 15
    log_term = np.zeros((n1, n2))
    log_term[pos] = np.log(1.0 / one_plus_nu[pos])
    i_sig_sim = float(np.sum(denom_signal * (nu + one_plus_nu * log_term)) / LOG2)

    # I_cor-indep: Eq. 16, simplified to sum_r [P(r)-P_ind(r)] log2(1/(1+nu))
    log2_inv_one_plus_nu = np.zeros((n1, n2))
    log2_inv_one_plus_nu[pos] = -np.log2(one_plus_nu[pos])
    i_cor_indep = float(np.sum((p_joint - p_ind) * log2_inv_one_plus_nu))

    # I_cor-dep: Eq. 17
    i_cor_dep = 0.0
    for si in range(n_s):
        if p_s[si] == 0:
            continue
        pr_s = p_joint_given_s[:, :, si]
        pind_s = p_ind_given_s[:, :, si]
        valid = (pr_s > 0) & (pind_s > 0) & (p_joint > 0)
        if not np.any(valid):
            continue
        ratio = (p_ind[valid] * pr_s[valid]) / (pind_s[valid] * p_joint[valid])
        i_cor_dep += p_s[si] * float(np.sum(pr_s[valid] * np.log2(ratio)))

    return {
        'I_full': float(i_full),
        'I_R1': float(i_r1),
        'I_R2': float(i_r2),
        'I_lin': float(i_lin),
        'I_sig_sim': i_sig_sim,
        'I_cor_indep': i_cor_indep,
        'I_cor_dep': i_cor_dep,
    }


def pairwise_information_breakdown(x1, x2, s, n_bins=3, binning_method='equipopulated',
                                    edges1=None, edges2=None, panzeri_treves=True,
                                    shuffle_correction=False, n_shuffles=30,
                                    random_state=None, unit_pair_index=0):
    """
    Compute the Pola et al. (2003) four-term EXACT information breakdown
    for one pair of (continuous, single-trial) responses x1, x2 about a
    discrete stimulus/choice variable s.

    Parameters
    ----------
    x1, x2 : 1D arrays, single-trial continuous responses (e.g. mean
        deconvolved activity in the response window) for the two cells
    s : 1D array, discrete stimulus or choice label per trial
    n_bins : int, bins per neuron (see discretize.py); ignored for a
        given neuron if `edges1`/`edges2` is supplied
    edges1, edges2 : optional precomputed bin edges for x1/x2. Pass these
        (e.g. computed once by pooling across all time/space bins) when
        calling this function repeatedly across bins of a temporal/
        spatial tensor for the SAME pair, so that the response alphabet
        stays fixed over time/space -- see temporal_breakdown.py.
    panzeri_treves : bool
        if True (default), also report Panzeri-Treves bias-corrected
        versions of I_full, I_R1, I_R2 and I_lin (`*_pt` keys).
    shuffle_correction : bool
        if True, ALSO report shuffle-subtraction bias-corrected versions
        of all seven terms (`*_shuffcorr` keys), matching the method
        MINT itself uses for this exact decomposition (Lorenz et al.
        2025, SM3.1.1/SM6.1). This is the recommended correction to
        trust the SIGN and magnitude of the small correlational terms
        (I_sig_sim, I_cor_indep, I_cor_dep) and of derived quantities
        like the redundancy-synergy index (I_full - I_lin) at realistic
        trial counts -- see the module docstring and
        `tests/test_breakdown_canonical.py` for a worked demonstration
        of why this matters. Costs `n_shuffles` extra (fast, since
        binning is not repeated) evaluations of the breakdown.
    n_shuffles : int
        number of stimulus-label permutations used for
        `shuffle_correction`.
    random_state, unit_pair_index : seeding for the shuffle permutations
        (see core.get_worker_rng; pass a distinct unit_pair_index per
        pair when calling this in a loop/parallel map so different pairs
        don't share identical shuffle draws).

    Returns
    -------
    dict with:
        I_full, I_R1, I_R2, I_lin, I_sig_sim, I_cor_indep, I_cor_dep
            (bits; naive plug-in, and exactly additive:
            I_lin + I_sig_sim + I_cor_indep + I_cor_dep == I_full)
        I_full_pt, I_R1_pt, I_R2_pt, I_lin_pt
            (bits; Panzeri-Treves bias-corrected, only if
            panzeri_treves=True)
        <term>_shuffcorr for each of the 7 terms above (only if
            shuffle_correction=True); these seven ALSO sum exactly:
            I_lin_shuffcorr + I_sig_sim_shuffcorr + I_cor_indep_shuffcorr
            + I_cor_dep_shuffcorr == I_full_shuffcorr
        n_bins_r1, n_bins_r2
    """
    s = _as_labels(s)

    r1_binned, edges1 = discretize_response(x1, n_bins=n_bins, method=binning_method, edges=edges1)
    r2_binned, edges2 = discretize_response(x2, n_bins=n_bins, method=binning_method, edges=edges2)
    n1 = len(edges1) - 1
    n2 = len(edges2) - 1

    out = _breakdown_terms_from_binned(r1_binned, r2_binned, s, n1, n2)
    out['n_bins_r1'] = n1
    out['n_bins_r2'] = n2

    if panzeri_treves:
        joint_full = _joint_response_labels(r1_binned, r2_binned, n2)
        out['I_full_pt'] = mutual_information(joint_full, s, bias_correction=True)['I_pt']
        out['I_R1_pt'] = mutual_information(r1_binned, s, bias_correction=True)['I_pt']
        out['I_R2_pt'] = mutual_information(r2_binned, s, bias_correction=True)['I_pt']
        out['I_lin_pt'] = out['I_R1_pt'] + out['I_R2_pt']

    if shuffle_correction:
        rng = core.get_worker_rng(random_state, unit_pair_index)
        n = len(s)
        shuf_sums = {k: 0.0 for k in TERM_KEYS}
        for _ in range(n_shuffles):
            s_shuf = s[rng.permutation(n)]
            terms_shuf = _breakdown_terms_from_binned(r1_binned, r2_binned, s_shuf, n1, n2)
            for k in TERM_KEYS:
                shuf_sums[k] += terms_shuf[k]
        for k in TERM_KEYS:
            shuf_mean = shuf_sums[k] / n_shuffles
            out[f'{k}_shuffcorr'] = out[k] - shuf_mean

    return out
