# -*- coding: utf-8 -*-
"""
test_gain_model_canonical.py
===============================
Validates every piece of gain_model.py against synthetic data with KNOWN
ground truth, simulated directly from the model being fit
(r_i(t) | G(t) ~ Poisson(mu_i(t) G(t))):

  1. estimate_gain_unconstrained recovers a known G(t) trace (given the
     TRUE mu) with high correlation, and its bias-corrected sigma_G^2
     is accurate -- critically, on a pure-Poisson null (sigma_G^2=0
     truly), the CORRECTED estimate must land at exactly 0, while the
     naive/uncorrected variance would falsely suggest excess gain
     variance (this is the whole point of the bias correction).
  2. estimate_tuning_curves_cv recovers the true per-condition mu(t) to
     high accuracy, and feeding its output into
     estimate_gain_unconstrained still recovers G(t) well (end-to-end
     check, not just each piece in isolation with oracle mu).
  3. fit_poisson_gain_glm recovers known (beta0, beta1) when the gain
     truly depends on a behavioral variable, and correctly finds ~0
     effect (beta1, pseudo_r2) when it's a null relationship.
  4. predicted_noise_correlation matches the empirical pairwise
     correlation structure of simulated shared-gain data.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from infotheory.gain_model import (  # noqa: E402
    estimate_tuning_curves_cv, estimate_gain_unconstrained,
    fit_poisson_gain_glm, predicted_noise_correlation,
    estimate_neuron_coupling_strength, predicted_noise_correlation_coupling)
from infotheory import signal_noise_correlations  # noqa: E402


def _report(checks):
    n_pass = 0
    for desc, ok in checks:
        print(f'  [{"PASS" if ok else "FAIL"}] {desc}')
        n_pass += int(ok)
    print(f'  {n_pass}/{len(checks)} checks passed.\n')
    return n_pass == len(checks)


def test_gain_recovery_with_oracle_mu():
    rng = np.random.default_rng(0)
    N, K = 100, 2000
    stim = rng.integers(0, 2, K)
    tuning = rng.uniform(1, 10, N)
    true_mu = tuning[None, :] * (1 + 0.5 * stim[:, None])

    true_sigma_G_sq = 0.25
    G_true = np.exp(rng.normal(-true_sigma_G_sq / 2, np.sqrt(true_sigma_G_sq), K))
    G_true = G_true / G_true.mean()
    r = rng.poisson(true_mu * G_true[:, None])

    res = estimate_gain_unconstrained(r, true_mu)
    corr = np.corrcoef(res['G_hat'], G_true)[0, 1]
    print(f'--- Gain recovery (oracle mu, large N): corr={corr:.4f}, '
          f'sigma_G_sq={res["sigma_G_sq"]:.4f} (true={true_sigma_G_sq}) ---')

    # Null-case bias-correction check: use a SMALL population, where the
    # precision-weighted estimator's own sampling noise is non-negligible
    # relative to a null (zero) shared-gain variance -- with N=100 above,
    # sampling noise is already so small that both raw and corrected
    # estimates round to ~0, which doesn't actually test the correction.
    N_small, K_small = 5, 3000
    tuning_small = rng.uniform(0.5, 3, N_small)
    stim_small = rng.integers(0, 2, K_small)
    true_mu_small = tuning_small[None, :] * (1 + 0.3 * stim_small[:, None])
    r_null = rng.poisson(true_mu_small)   # NO shared gain at all
    res_null = estimate_gain_unconstrained(r_null, true_mu_small)
    print(f'--- Null case (small N={N_small}): sigma_G_sq corrected={res_null["sigma_G_sq"]:.4f} '
          f'(raw={res_null["sigma_G_sq_raw"]:.4f}, true=0) ---')

    checks = [
        ('corr(G_hat, G_true) > 0.98 with oracle mu', corr > 0.98),
        ('sigma_G_sq within 20% of truth', abs(res['sigma_G_sq'] - true_sigma_G_sq) / true_sigma_G_sq < 0.2),
        ('null case: corrected sigma_G_sq is small (<0.02, near the true 0)', res_null['sigma_G_sq'] < 0.02),
        ('null case: correction removes most of the false-positive raw variance',
         res_null['sigma_G_sq'] < 0.3 * res_null['sigma_G_sq_raw']),
        ('null case: raw (uncorrected) is falsely positive (small N)', res_null['sigma_G_sq_raw'] > 0.03),
    ]
    return _report(checks)


def test_cv_tuning_curves_end_to_end():
    rng = np.random.default_rng(2)
    N, K = 80, 2500
    stim = rng.integers(0, 3, K)
    tuning = rng.uniform(1, 8, (3, N))
    true_mu = tuning[stim, :]

    true_sigma_G_sq = 0.2
    G_true = np.exp(rng.normal(-true_sigma_G_sq / 2, np.sqrt(true_sigma_G_sq), K))
    G_true = G_true / G_true.mean()
    r = rng.poisson(true_mu * G_true[:, None])

    mu_hat_cv = estimate_tuning_curves_cv(r, stim, n_folds=5, random_state=0)
    mu_corr = np.corrcoef(mu_hat_cv.ravel(), true_mu.ravel())[0, 1]

    res = estimate_gain_unconstrained(r, mu_hat_cv)
    corr_G = np.corrcoef(res['G_hat'], G_true)[0, 1]
    print(f'--- CV tuning curves: corr(mu_hat_cv, true_mu)={mu_corr:.4f}, '
          f'corr(G_hat, G_true)={corr_G:.4f}, sigma_G_sq={res["sigma_G_sq"]:.4f} '
          f'(true={true_sigma_G_sq}) ---')

    checks = [
        ('CV mu_hat correlates > 0.99 with true mu', mu_corr > 0.99),
        ('end-to-end gain recovery corr > 0.98', corr_G > 0.98),
        ('sigma_G_sq within 25% of truth', abs(res['sigma_G_sq'] - true_sigma_G_sq) / true_sigma_G_sq < 0.25),
    ]
    return _report(checks)


def test_glm_behavior_driven_gain():
    rng = np.random.default_rng(3)
    N, K = 100, 3000
    stim = rng.integers(0, 2, K)
    tuning = rng.uniform(1, 6, N)
    true_mu = tuning[None, :] * (1 + 0.4 * stim[:, None])

    behavior = rng.normal(0, 1, K)
    true_beta0, true_beta1 = -0.05, 0.35
    G_true = np.exp(true_beta0 + true_beta1 * behavior)
    r = rng.poisson(true_mu * G_true[:, None])

    mu_hat_cv = estimate_tuning_curves_cv(r, stim, n_folds=5, random_state=0)
    res = fit_poisson_gain_glm(r, mu_hat_cv, behavior)
    corr_GB = np.corrcoef(res['G_hat_B'], G_true)[0, 1]
    print(f'--- GLM (behavior-driven): beta0={res["beta0"]:.4f} (true={true_beta0}), '
          f'beta1={res["beta1"]:.4f} (true={true_beta1}), pseudo_r2={res["pseudo_r2"]:.4f}, '
          f'corr(G_hat_B, G_true)={corr_GB:.4f} ---')

    # null: behavior unrelated to gain
    G_flat = np.ones(K)
    r_flat = rng.poisson(true_mu * G_flat[:, None])
    mu_hat_cv2 = estimate_tuning_curves_cv(r_flat, stim, n_folds=5, random_state=1)
    behavior_null = rng.normal(0, 1, K)
    res_null = fit_poisson_gain_glm(r_flat, mu_hat_cv2, behavior_null)
    print(f'--- GLM (null): beta1={res_null["beta1"]:.4f} (true=0), '
          f'pseudo_r2={res_null["pseudo_r2"]:.4f} (true~0) ---')

    checks = [
        ('beta0 within 0.05 of truth', abs(res['beta0'] - true_beta0) < 0.05),
        ('beta1 within 0.05 of truth', abs(res['beta1'] - true_beta1) < 0.05),
        ('G_hat_B correlates ~1.0 with true gain', corr_GB > 0.999),
        ('null case: |beta1| < 0.02', abs(res_null['beta1']) < 0.02),
        ('null case: pseudo_r2 < 0.01', res_null['pseudo_r2'] < 0.01),
    ]
    return _report(checks)


def test_predicted_noise_correlation():
    rng = np.random.default_rng(4)
    N, K = 6, 20000
    mu = rng.uniform(2, 15, N)
    true_sigma_G_sq = 0.3
    G = np.exp(rng.normal(-true_sigma_G_sq / 2, np.sqrt(true_sigma_G_sq), K))
    G = G / G.mean()
    r = rng.poisson(mu[None, :] * G[:, None])

    emp_corr = np.corrcoef(r.T)
    iu = np.triu_indices(N, k=1)
    pred_corr = np.array([predicted_noise_correlation(mu[i], mu[j], true_sigma_G_sq)
                           for i, j in zip(*iu)])
    emp_vals = emp_corr[iu]

    corr_of_corrs = np.corrcoef(emp_vals, pred_corr)[0, 1]
    mean_abs_err = np.mean(np.abs(emp_vals - pred_corr))
    print(f'--- Noise-correlation prediction (uniform model): corr(empirical, predicted)={corr_of_corrs:.4f}, '
          f'mean|error|={mean_abs_err:.4f} ---')

    checks = [
        ('predicted vs empirical pairwise correlations agree (r>0.99)', corr_of_corrs > 0.99),
        ('mean absolute error < 0.05', mean_abs_err < 0.05),
    ]
    return _report(checks)


def test_neuron_coupling_strength_heterogeneous():
    """
    The critical test: with HETEROGENEOUS, mixed-sign per-neuron
    coupling to the shared factor, and at mean-response magnitudes
    matching this pipeline's actual (magnitude-inflated) deconvolved
    data, the uniform-sensitivity model (`predicted_noise_correlation`)
    should be BADLY miscalibrated (saturating near 1 regardless of the
    true coupling), while `estimate_neuron_coupling_strength` +
    `predicted_noise_correlation_coupling` should recover the per-pair
    structure accurately.
    """
    rng = np.random.default_rng(0)
    N, K = 60, 4000
    stim = rng.integers(0, 2, K)
    tuning = rng.uniform(20, 80, N)   # magnitude regime matching real deconv data
    mu = tuning[None, :] * (1 + 0.3 * stim[:, None])

    true_sigma_G_sq = 0.05
    F = np.exp(rng.normal(-true_sigma_G_sq / 2, np.sqrt(true_sigma_G_sq), K))
    F = F / F.mean()
    true_a = np.concatenate([rng.uniform(0.8, 1.5, N // 3), rng.uniform(-0.3, 0.3, N // 3),
                              rng.uniform(-1.0, -0.3, N // 3)])
    rng.shuffle(true_a)
    r = rng.poisson(mu * (F[:, None] ** true_a[None, :]))
    true_c = true_a * np.sqrt(true_sigma_G_sq)

    mu_hat = estimate_tuning_curves_cv(r, stim, n_folds=5, random_state=0)
    c_est, _ = estimate_neuron_coupling_strength(r, mu_hat, stim)
    corr_c = np.corrcoef(true_c, c_est)[0, 1]
    frac_sign_correct = np.mean(np.sign(true_c) == np.sign(c_est))
    print(f'--- Coupling strength recovery: corr(true_c, c_est)={corr_c:.4f}, '
          f'sign accuracy={frac_sign_correct:.3f} ---')

    mu_bar = mu_hat.mean(axis=0)
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rng2 = np.random.default_rng(1)
    pairs = [pairs[k] for k in rng2.choice(len(pairs), size=300, replace=False)]
    signal_corr, noise_corr = signal_noise_correlations(r.T, stim, pairs)

    pred_uniform = np.array([predicted_noise_correlation(mu_bar[i], mu_bar[j], true_sigma_G_sq)
                              for i, j in pairs])
    pred_coupling = np.array([predicted_noise_correlation_coupling(mu_bar[i], mu_bar[j], c_est[i], c_est[j])
                               for i, j in pairs])
    valid = np.isfinite(noise_corr) & np.isfinite(pred_uniform) & np.isfinite(pred_coupling)

    corr_uniform = np.corrcoef(pred_uniform[valid], noise_corr[valid])[0, 1]
    corr_coupling = np.corrcoef(pred_coupling[valid], noise_corr[valid])[0, 1]
    mean_pred_uniform = pred_uniform[valid].mean()
    mean_pred_coupling = pred_coupling[valid].mean()
    mean_observed = noise_corr[valid].mean()
    print(f'--- Uniform model:   corr(pred,obs)={corr_uniform:.4f}, mean_pred={mean_pred_uniform:.4f} '
          f'(mean_obs={mean_observed:.4f}) -- expect BAD (near-zero corr, saturated mean) ---')
    print(f'--- Coupling model:  corr(pred,obs)={corr_coupling:.4f}, mean_pred={mean_pred_coupling:.4f} '
          f'-- expect GOOD ---')

    checks = [
        ('coupling strength recovers true values well (corr>0.99)', corr_c > 0.99),
        ('coupling strength sign mostly correct (>90%)', frac_sign_correct > 0.9),
        ('uniform model is indeed badly miscalibrated here (corr<0.3)', corr_uniform < 0.3),
        ('uniform model saturates high (mean_pred > 3x observed)', mean_pred_uniform > 3 * abs(mean_observed)),
        ('coupling model matches observed correlations well (corr>0.95)', corr_coupling > 0.95),
        ('coupling model mean is close to observed mean (abs diff<0.02)',
         abs(mean_pred_coupling - mean_observed) < 0.02),
    ]
    return _report(checks)


if __name__ == '__main__':
    results = [
        test_gain_recovery_with_oracle_mu(),
        test_cv_tuning_curves_end_to_end(),
        test_glm_behavior_driven_gain(),
        test_predicted_noise_correlation(),
        test_neuron_coupling_strength_heterogeneous(),
    ]
    if all(results):
        print('ALL GAIN-MODEL CANONICAL VALIDATION TESTS PASSED.')
        sys.exit(0)
    else:
        print('SOME GAIN-MODEL CANONICAL VALIDATION TESTS FAILED -- see above.')
        sys.exit(1)
