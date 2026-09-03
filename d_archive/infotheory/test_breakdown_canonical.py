# -*- coding: utf-8 -*-
"""
test_breakdown_canonical.py
=============================
Validates `breakdown.pairwise_information_breakdown` against two
canonical, LITERATURE-DOCUMENTED simulated scenarios with known
qualitative results, reproduced from Lorenz, Engel, Celotto et al.
(2025) "MINT: a toolbox for the analysis of multivariate neural
information coding and transmission", PLoS Comput Biol 21(4):e1012934
-- specifically Fig 2A/2B and their exact simulation parameters in
Supplementary Section SM6.1.

Both scenarios simulate a PAIR of Poisson neurons responding to a
binary stimulus s in {0, 1} as the sum of an independent process and a
process SHARED between the two neurons (drawn once per trial, added
identically to both):

    r_i(trial) = Poisson(lambda_indiv(s)) + Poisson(lambda_shared(s))

with lambda_shared(s) controlling the trial-by-trial noise correlation
strength (shared variance / total variance = lambda_shared / (lambda_
indiv + lambda_shared) for same-stimulus trials), and lambda_indiv(s)
controlling single-cell stimulus tuning.

Scenario A -- "pure stimulus-dependent correlation" (paper Fig 2A):
    lambda_indiv(s=0) = 1, lambda_shared(s=0) = 1   (total rate 2)
    lambda_indiv(s=1) = 2, lambda_shared(s=1) = 0   (total rate 2)
    Total firing rate is IDENTICAL across stimuli by construction, so
    single-cell information is ~zero; only the correlation strength
    (1.0 vs 0.0 shared spikes/s) carries stimulus information. Paper's
    described result: single-cell info ~0, joint (pairwise) info > 0,
    strongly positive RSI (synergy), and "all the synergistic
    information is due to stimulus-dependent correlations" -- i.e.
    I_sig_sim ~ 0, I_cor_indep ~ 0, I_cor_dep accounts for essentially
    all of I_full.

Scenario B -- "information-limiting correlation" (paper Fig 2B):
    lambda_indiv(s=0) = 0.8, lambda_shared(s=0) = 0.2   (total 1.0)
    lambda_indiv(s=1) = 1.9, lambda_shared(s=1) = 0.1   (total 2.0)
    Both neurons share identical tuning (redundant signal correlation)
    and positive, only weakly stimulus-modulated noise correlation.
    Paper's described result: negative RSI (redundancy dominates
    synergy) because signal similarity (I_sig_sim) and the average
    noise correlation (I_cor_indep) are both negative and larger in
    magnitude than the small, positive stimulus-dependent term
    (I_cor_dep).

IMPORTANT: as demonstrated interactively while building this test
(and left as a comment here for anyone re-deriving it), the RAW naive
plug-in breakdown gets scenario B's sign WRONG at the paper's own
trial count (200 trials/stimulus) -- the small redundancy-synergy index
sits right at the level of the naive estimator's limited-sampling bias,
and only recovers the correct (negative) sign once trial counts are
pushed far beyond anything experimentally realistic, OR once
shuffle-subtraction bias correction is applied (which is what the paper
itself does for this exact figure, see SM6.1: "shuffle-subtraction
procedure ... averaged over 30 shuffles"). This test therefore validates
the `shuffle_correction=True` output, matching the paper's own method.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from infotheory.breakdown import pairwise_information_breakdown  # noqa: E402


# a joint response symbol above 4 spikes is capped, matching the 5-bin
# scheme MINT itself uses for spike counts in this exact simulation
# (SM6.1: "R=5 bins (0,1,2,3,>3 spike counts)")
SPIKE_COUNT_EDGES = np.array([-0.5, 0.5, 1.5, 2.5, 3.5, np.inf])


def simulate_correlated_poisson_pair(lambda_indiv_s0, lambda_shared_s0,
                                      lambda_indiv_s1, lambda_shared_s1,
                                      n_trials_per_stim=200, rng=None):
    """Simulate one pair of neurons per the SM6.1 recipe above."""
    if rng is None:
        rng = np.random.default_rng()
    n = n_trials_per_stim
    stim = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])

    shared_s0 = rng.poisson(lambda_shared_s0, size=n)
    r1_s0 = rng.poisson(lambda_indiv_s0, size=n) + shared_s0
    r2_s0 = rng.poisson(lambda_indiv_s0, size=n) + shared_s0

    shared_s1 = rng.poisson(lambda_shared_s1, size=n)
    r1_s1 = rng.poisson(lambda_indiv_s1, size=n) + shared_s1
    r2_s1 = rng.poisson(lambda_indiv_s1, size=n) + shared_s1

    r1 = np.concatenate([r1_s0, r1_s1]).astype(float)
    r2 = np.concatenate([r2_s0, r2_s1]).astype(float)
    return r1, r2, stim


TERM_KEYS = ['I_full', 'I_R1', 'I_R2', 'I_lin', 'I_sig_sim', 'I_cor_indep', 'I_cor_dep']


def run_scenario(lambda_indiv_s0, lambda_shared_s0, lambda_indiv_s1, lambda_shared_s1,
                  n_repeats=10, n_trials_per_stim=200, n_shuffles=30):
    """
    Run the given scenario `n_repeats` times (independent simulated
    datasets) and return the mean +/- SEM of each shuffle-subtraction
    bias-corrected breakdown term across repeats.
    """
    shuffcorr_keys = [f'{k}_shuffcorr' for k in TERM_KEYS]
    all_results = []
    for rep in range(n_repeats):
        rng = np.random.default_rng(rep)
        r1, r2, stim = simulate_correlated_poisson_pair(
            lambda_indiv_s0, lambda_shared_s0, lambda_indiv_s1, lambda_shared_s1,
            n_trials_per_stim=n_trials_per_stim, rng=rng)
        res = pairwise_information_breakdown(
            r1, r2, stim, edges1=SPIKE_COUNT_EDGES, edges2=SPIKE_COUNT_EDGES,
            panzeri_treves=False, shuffle_correction=True,
            n_shuffles=n_shuffles, random_state=rep)
        all_results.append(res)

    agg = {k: float(np.mean([r[k] for r in all_results])) for k in shuffcorr_keys}
    sem = {k: float(np.std([r[k] for r in all_results]) / np.sqrt(n_repeats)) for k in shuffcorr_keys}
    return agg, sem


def _print_scenario(name, agg, sem):
    print(f'--- {name} ---')
    for k in TERM_KEYS:
        kk = f'{k}_shuffcorr'
        print(f'  {kk:22s}: {agg[kk]:+.4f} +/- {sem[kk]:.4f}')
    total = sum(agg[f'{k}_shuffcorr'] for k in ['I_lin', 'I_sig_sim', 'I_cor_indep', 'I_cor_dep'])
    print(f'  sum check            : {total:+.4f} (vs I_full_shuffcorr {agg["I_full_shuffcorr"]:+.4f})')


def test_scenario_a_pure_stimulus_dependent_correlation():
    """
    All information should be synergistic and attributed to I_cor_dep;
    I_lin, I_sig_sim, I_cor_indep should all be small/near-zero.
    """
    agg, sem = run_scenario(lambda_indiv_s0=1.0, lambda_shared_s0=1.0,
                             lambda_indiv_s1=2.0, lambda_shared_s1=0.0)
    _print_scenario('Scenario A: pure stimulus-dependent correlation', agg, sem)

    i_full = agg['I_full_shuffcorr']
    i_lin = agg['I_lin_shuffcorr']
    i_cor_dep = agg['I_cor_dep_shuffcorr']

    checks = [
        ('I_full clearly positive (correlations do carry information)',
         i_full > 3 * sem['I_full_shuffcorr']),
        ('I_lin small relative to I_full (single cells ~uninformative)',
         abs(i_lin) < 0.3 * i_full),
        ('I_cor_dep accounts for most of I_full (>70%)',
         i_cor_dep > 0.7 * i_full),
        ('I_sig_sim ~ 0',
         abs(agg['I_sig_sim_shuffcorr']) < 0.15 * i_full),
        ('I_cor_indep ~ 0',
         abs(agg['I_cor_indep_shuffcorr']) < 0.15 * i_full),
        ('exact-sum identity holds',
         np.isclose(i_lin + agg['I_sig_sim_shuffcorr'] + agg['I_cor_indep_shuffcorr'] + i_cor_dep,
                     i_full, atol=1e-9)),
    ]
    return _report(checks)


def test_scenario_b_information_limiting_correlation():
    """
    Redundancy should dominate: RSI = I_full - I_lin < 0, driven mainly
    by negative I_sig_sim and I_cor_indep outweighing a small positive
    I_cor_dep.
    """
    agg, sem = run_scenario(lambda_indiv_s0=0.8, lambda_shared_s0=0.2,
                             lambda_indiv_s1=1.9, lambda_shared_s1=0.1)
    _print_scenario('Scenario B: information-limiting correlation', agg, sem)

    i_full = agg['I_full_shuffcorr']
    i_lin = agg['I_lin_shuffcorr']
    rsi = i_full - i_lin
    sig_sim = agg['I_sig_sim_shuffcorr']
    cor_indep = agg['I_cor_indep_shuffcorr']
    cor_dep = agg['I_cor_dep_shuffcorr']

    checks = [
        ('I_lin clearly positive (single cells are tuned)', i_lin > 0.1),
        ('RSI = I_full - I_lin is negative (net redundancy)', rsi < 0),
        ('I_sig_sim is negative (redundant tuning)', sig_sim < 0),
        ('I_cor_dep is non-negative (small stim-dependent boost)', cor_dep > -0.01),
        ('|I_sig_sim| + |I_cor_indep| > I_cor_dep (redundancy wins)',
         abs(sig_sim) + abs(cor_indep) > cor_dep),
        ('exact-sum identity holds',
         np.isclose(i_lin + sig_sim + cor_indep + cor_dep, i_full, atol=1e-9)),
    ]
    return _report(checks)


def _report(checks):
    n_pass = 0
    for desc, ok in checks:
        print(f'  [{"PASS" if ok else "FAIL"}] {desc}')
        n_pass += int(ok)
    print(f'  {n_pass}/{len(checks)} checks passed.\n')
    return n_pass == len(checks)


if __name__ == '__main__':
    ok_a = test_scenario_a_pure_stimulus_dependent_correlation()
    ok_b = test_scenario_b_information_limiting_correlation()
    if ok_a and ok_b:
        print('ALL CANONICAL VALIDATION SCENARIOS PASSED.')
        sys.exit(0)
    else:
        print('SOME CANONICAL VALIDATION SCENARIOS FAILED -- see above.')
        sys.exit(1)
