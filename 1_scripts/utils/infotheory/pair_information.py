# -*- coding: utf-8 -*-
"""
pair_information.py
===================
The Pola, Thiele, Hoffmann & Panzeri (2003, Network 14:35) exact information breakdown of a pair of binned responses
(r1, r2) about a discrete target s, shared by 3b (windows) and 3e (position bins).

    I(R;S) = I_lin + I_sigsim + I_corind + I_cordep, with  P_ind(r|s) = P(r1|s) P(r2|s),  P_ind(r) = sum_s P(s) P_ind(r|s)
    gamma(r|s) = P(r|s)/P_ind(r|s) - 1 (eq. 4),  nu(r) = P_ind(r) / (P(r1) P(r2)) - 1 (eq. 6)

    I_lin    = I(R1;S) + I(R2;S)                                                  (eq. 9)
    I_sigsim = sum_r P(r1)P(r2) [nu - (1+nu) ln(1+nu)] / ln 2  =  - KL(P_ind(r) || P(r1)P(r2))  <= 0   (eq. 10)
               ALWAYS NEGATIVE (or zero when the cells' tuning is unrelated): the information lost because the two cells
               respond similarly to the target. Pola et al. prove it; it is computed here from that closed form on the
               plug-in probabilities, so it can never come out positive. Pola et al. also note that I_sigsim is hardly
               biased, so it is NOT bias-corrected (a shuffle-corrected difference of two estimates, as an earlier version
               of this module used, can be positive by estimation noise).
    I_corind = sum_s P(s) sum_r [P(r|s) - P_ind(r|s)] log2( 1/(1+nu(r)) )          (eq. 11), sign not constrained
    I_cordep = sum_s P(s) KL(P(.|s) || P_ind(.|s)) - KL(P(.) || P_ind(.))  >= 0     (eq. 12 written as a difference of
               KLs; non-negative before bias correction, 0 iff gamma(r|s) does not depend on s)
The four terms add up to I(R;S) exactly (checked in the tests of this module's author). I_cor = I_corind + I_cordep and
RSI = I_total - I_lin = I_sigsim + I_cor.

Equivalent entropy forms (Pola appendix A, eqs 27-30; Magri et al. 2009, BMC Neurosci 10:81, the ibTB toolbox):
    I_sigsim = H_ind(R) - sum_c H(R_c),   I_corind = F(R) - H_ind(R),   I_cordep = I - F(R) + sum_c H(R_c|S),
    F(R) = -sum_r P(r) log2 P_ind(r)  -- identical to the formulas above (checked numerically to 1e-15).

BIAS    Each term gets the treatment that works best for it (tested by simulation, see below):
        I_1, I_2, I_total, I_lin  shuffle-corrected as everywhere in this project (info_theory.mutual_information_shuffle,
                                  the same as 2c);
        I_sigsim, I_corind        plug-in, as in Pola et al. (hardly biased: in simulations with 60-400 trials and 3 response
                                  bins the plug-in I_corind was within 0.01 bit of its large-sample value);
        I_cordep                  the remainder I_total - I_lin - I_sigsim - I_corind (inherits the corrected total),
                                  NOT floored: the true I_cordep is >= 0 but after bias correction the estimate may come out
                                  slightly negative, which is kept (I_cordep_raw is the same number, stored for compatibility).
                                  I_cor = I_corind + I_cordep exactly; this is the most biased term of the breakdown.
        Pola's / ibTB's own correction of I_cordep (replace H_ind(R|S) by the within-stimulus shuffled noise entropy
        H_sh(R|S)) was also simulated here and was biased low with only 3 response bins and < 400 trials, so it is not used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .info_theory import mutual_information_shuffle, quantile_bin_edges
from .shuffling import shuffle_within_groups


def digitize_quantile(x: np.ndarray, bins: int) -> np.ndarray:
    edges = quantile_bin_edges(x, bins)
    return np.clip(np.digitize(x, edges[1:-1]), 0, bins - 1)


def joint_code(a_bin: np.ndarray, b_bin: np.ndarray, n_bins: int) -> np.ndarray:
    return a_bin.astype(np.int64) * n_bins + b_bin.astype(np.int64)


def plugin_breakdown_terms(r1_bin, r2_bin, stim, n_bins: int) -> dict:
    """Plug-in (uncorrected) Pola terms in bits for binned responses `r1_bin`, `r2_bin` (ints in [0, n_bins)) and any
    discrete target `stim`: I_total, I_lin, I_sigsim (<= 0), I_corind, I_cordep (>= 0). Exact identities, see the module
    docstring; I_corind is evaluated from its own formula (eq. 11)."""
    s_codes = pd.factorize(np.asarray(stim))[0]
    ns = s_codes.max() + 1
    cnt = np.zeros((ns, n_bins, n_bins))
    np.add.at(cnt, (s_codes, np.asarray(r1_bin, int), np.asarray(r2_bin, int)), 1.0)
    p_s = cnt.sum(axis=(1, 2)) / cnt.sum()
    p_joint = cnt / cnt.sum(axis=(1, 2), keepdims=True)                      # P(r1, r2 | s)
    p1_s, p2_s = p_joint.sum(axis=2), p_joint.sum(axis=1)                    # P(r1|s), P(r2|s)
    p_ind_s = p1_s[:, :, None] * p2_s[:, None, :]                            # P_ind(r|s)
    w = p_s[:, None, None]
    p_r = (w * p_joint).sum(axis=0)                                          # P(r)
    p_ind = (w * p_ind_s).sum(axis=0)                                        # P_ind(r)
    m1, m2 = (p_s[:, None] * p1_s).sum(axis=0), (p_s[:, None] * p2_s).sum(axis=0)
    prod_marg = m1[:, None] * m2[None, :]

    def kl(p, q):                                                            # sum p log2(p/q), 0 log 0 = 0
        ok = p > 0
        return float((p[ok] * np.log2(p[ok] / q[ok])).sum())

    def mi(p_xy_s, p_x):                                                     # I(R;S) of a conditional table
        return sum(p_s[k] * kl(p_xy_s[k], p_x) for k in range(ns))

    i_total = mi(p_joint, p_r)
    i_lin = mi(p1_s, m1) + mi(p2_s, m2)
    i_sigsim = -max(kl(p_ind, prod_marg), 0.0)          # KL >= 0; guards against rounding
    i_cordep = sum(p_s[k] * kl(p_joint[k], p_ind_s[k]) for k in range(ns)) - kl(p_r, p_ind)
    nu = np.where(prod_marg > 0, p_ind / np.where(prod_marg > 0, prod_marg, 1.0) - 1.0, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.where(nu > -1, -np.log2(1.0 + nu), 0.0)
    i_corind = float((w * (p_joint - p_ind_s) * log_term).sum())
    return dict(I_total=i_total, I_lin=i_lin, I_sigsim=i_sigsim, I_corind=i_corind, I_cordep=i_cordep)


def compute_pair_breakdown(x1: np.ndarray, x2: np.ndarray, stim_bin: np.ndarray, r_bins: int = 3, mi_shuffles: int = 15,
                           rng: np.random.Generator | None = None) -> dict:
    """The Pola breakdown of one pair (see the module docstring for the definitions and the bias treatment).

    Returns a dict: I_total, I_1, I_2, I_lin, I_sigsim (<= 0), I_corind, I_cordep, I_cor (= I_corind + I_cordep),
    RSI (= I_total - I_lin), I_ind (= I_lin + I_sigsim), n_trials."""
    rng = rng or np.random.default_rng()
    r1_bin = digitize_quantile(x1, r_bins)
    r2_bin = digitize_quantile(x2, r_bins)
    mi1 = mutual_information_shuffle(r1_bin, stim_bin, bins=r_bins, n_shuffles=mi_shuffles, rng=rng).bits_corrected
    mi2 = mutual_information_shuffle(r2_bin, stim_bin, bins=r_bins, n_shuffles=mi_shuffles, rng=rng).bits_corrected
    i_lin = mi1 + mi2
    jc_total = joint_code(r1_bin, r2_bin, r_bins)
    i_total = mutual_information_shuffle(jc_total, stim_bin, bins=r_bins * r_bins, n_shuffles=mi_shuffles,
                                         rng=rng).bits_corrected
    terms = plugin_breakdown_terms(r1_bin, r2_bin, stim_bin, r_bins)
    i_sigsim, i_corind = terms["I_sigsim"], terms["I_corind"]
    i_cor = i_total - i_lin - i_sigsim                    # exact remainder: the four terms still add up to I_total
    i_cordep_raw = i_cor - i_corind                       # bias-corrected remainder; the true (plug-in) I_cordep is >= 0, this one can be < 0
    i_cordep = i_cordep_raw                               # NOT floored: may be slightly negative after bias correction
    return {"I_total": i_total, "I_1": mi1, "I_2": mi2, "I_lin": i_lin, "I_ind": i_lin + i_sigsim, "I_sigsim": i_sigsim,
            "I_corind": i_corind, "I_cordep": i_cordep, "I_cordep_raw": i_cordep_raw, "I_cor": i_cor, "RSI": i_total - i_lin,
            "n_trials": len(x1)}
