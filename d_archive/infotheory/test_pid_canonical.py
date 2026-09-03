# -*- coding: utf-8 -*-
"""
test_pid_canonical.py
========================
Validates `pid.pid_decomposition` against the three canonical two-source
PID "logic gate" test cases used throughout the PID literature (Williams
& Beer, 2010; Timme & Lapish, 2018 eNeuro tutorial; also used as
sanity checks in Lorenz et al. 2025's MINT toolbox):

    XOR  (R = S1 XOR S2, S1/S2 independent fair bits):
        pure SYNERGY -- redundancy = unique1 = unique2 = 0, synergy = 1 bit
    COPY (R = S1, S2 independent of S1 and irrelevant):
        pure UNIQUE information from S1 -- unique1 = 1 bit, everything
        else = 0
    RDN  (S1 == S2 identically, R = S1):
        pure REDUNDANCY -- redundancy = 1 bit, unique1 = unique2 =
        synergy = 0

It also demonstrates why `shuffle_correction=True` matters at realistic
trial counts: a genuinely independent (R unrelated to S1, S2) case shows
substantial spurious information with the naive estimator at n=200
trials, dropping close to zero once shuffle-subtraction is applied.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from infotheory.pid import pid_decomposition  # noqa: E402


def _report(checks):
    n_pass = 0
    for desc, ok in checks:
        print(f'  [{"PASS" if ok else "FAIL"}] {desc}')
        n_pass += int(ok)
    print(f'  {n_pass}/{len(checks)} checks passed.\n')
    return n_pass == len(checks)


def test_xor_gate_pure_synergy(n=20000, tol=0.02):
    rng = np.random.default_rng(0)
    s1 = rng.integers(0, 2, n)
    s2 = rng.integers(0, 2, n)
    r = s1 ^ s2
    res = pid_decomposition(r, s1, s2)
    print('--- XOR gate (expect Red=0, Unq1=0, Unq2=0, Syn=1 bit) ---')
    for k, v in res.items():
        print(f'  {k}: {v:.4f}')
    checks = [
        ('redundancy ~ 0', abs(res['redundancy']) < tol),
        ('unique_S1 ~ 0', abs(res['unique_S1']) < tol),
        ('unique_S2 ~ 0', abs(res['unique_S2']) < tol),
        ('synergy ~ 1 bit', abs(res['synergy'] - 1.0) < tol),
        ('I_total ~ 1 bit', abs(res['I_total'] - 1.0) < tol),
    ]
    return _report(checks)


def test_copy_gate_pure_unique(n=20000, tol=0.02):
    rng = np.random.default_rng(1)
    s1 = rng.integers(0, 2, n)
    s2 = rng.integers(0, 2, n)
    r = s1.copy()
    res = pid_decomposition(r, s1, s2)
    print('--- COPY gate R=S1 (expect Unq1=1 bit, Unq2=0, Red=0, Syn=0) ---')
    for k, v in res.items():
        print(f'  {k}: {v:.4f}')
    checks = [
        ('redundancy ~ 0', abs(res['redundancy']) < tol),
        ('unique_S1 ~ 1 bit', abs(res['unique_S1'] - 1.0) < tol),
        ('unique_S2 ~ 0', abs(res['unique_S2']) < tol),
        ('synergy ~ 0', abs(res['synergy']) < tol),
        ('I_total ~ 1 bit', abs(res['I_total'] - 1.0) < tol),
    ]
    return _report(checks)


def test_rdn_gate_pure_redundancy(n=20000, tol=0.02):
    rng = np.random.default_rng(2)
    s1 = rng.integers(0, 2, n)
    s2 = s1.copy()
    r = s1.copy()
    res = pid_decomposition(r, s1, s2)
    print('--- RDN gate S1=S2, R=S1 (expect Red=1 bit, Unq1=0, Unq2=0, Syn=0) ---')
    for k, v in res.items():
        print(f'  {k}: {v:.4f}')
    checks = [
        ('redundancy ~ 1 bit', abs(res['redundancy'] - 1.0) < tol),
        ('unique_S1 ~ 0', abs(res['unique_S1']) < tol),
        ('unique_S2 ~ 0', abs(res['unique_S2']) < tol),
        ('synergy ~ 0', abs(res['synergy']) < tol),
        ('I_total ~ 1 bit', abs(res['I_total'] - 1.0) < tol),
    ]
    return _report(checks)


def test_shuffle_correction_removes_null_bias(n_trials=200, n_repeats=30, n_shuffles=50):
    """
    For genuinely independent r, s1, s2, the mean naive I_total across
    many independent datasets should be well above zero (bias); the
    mean shuffle-corrected I_total should be much closer to zero.
    """
    naive_vals, corrected_vals = [], []
    for rep in range(n_repeats):
        rng = np.random.default_rng(rep)
        s1 = rng.integers(0, 2, n_trials)
        s2 = rng.integers(0, 2, n_trials)
        r = rng.integers(0, 3, n_trials)
        res = pid_decomposition(r, s1, s2, shuffle_correction=True,
                                 n_shuffles=n_shuffles, random_state=rep)
        naive_vals.append(res['I_total'])
        corrected_vals.append(res['I_total_shuffcorr'])

    naive_mean = float(np.mean(naive_vals))
    corrected_mean = float(np.mean(corrected_vals))
    corrected_sem = float(np.std(corrected_vals) / np.sqrt(n_repeats))

    print(f'--- Null-case bias check (n_trials={n_trials}, n_repeats={n_repeats}) ---')
    print(f'  naive I_total (mean over repeats)      : {naive_mean:+.4f}  (should be clearly > 0, this IS the bias)')
    print(f'  shuffcorr I_total (mean +/- SEM)        : {corrected_mean:+.4f} +/- {corrected_sem:.4f}  (should be ~0)')

    checks = [
        ('naive estimator shows clear positive bias for a null case', naive_mean > 0.01),
        ('shuffle-corrected mean is within ~2 SEM of zero',
         abs(corrected_mean) < 3 * corrected_sem + 1e-3),
        ('shuffle correction reduces |bias| substantially', abs(corrected_mean) < 0.5 * naive_mean),
    ]
    return _report(checks)


if __name__ == '__main__':
    results = [
        test_xor_gate_pure_synergy(),
        test_copy_gate_pure_unique(),
        test_rdn_gate_pure_redundancy(),
        test_shuffle_correction_removes_null_bias(),
    ]
    if all(results):
        print('ALL PID CANONICAL VALIDATION TESTS PASSED.')
        sys.exit(0)
    else:
        print('SOME PID CANONICAL VALIDATION TESTS FAILED -- see above.')
        sys.exit(1)
