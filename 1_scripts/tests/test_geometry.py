import numpy as np
from utils.geometry import (
    angle_between, exact_angle_pvalue, exact_angle_pvalue_subspace, participation_ratio,
    gain_ratio, pca_spectrum, cv_pca_reconstruction_r2, effective_dimensionality_from_cv,
    alignment_correlation, cv_pca_output_r2, noise_suppression_ratio,
    relative_susceptibility, effective_connectivity_modes, multi_regressor_selectivity,
    sin_alignment,
)


def test_angle_between_orthogonal():
    assert abs(angle_between([1, 0], [0, 1]) - 90) < 1e-9


def test_angle_between_parallel():
    assert abs(angle_between([1, 2, 3], [2, 4, 6])) < 1e-9


def test_exact_angle_pvalue_matches_monte_carlo():
    """Regression test for the analytic formula against a large Monte
    Carlo null of random directions in R^N -- must match to 2 decimals."""
    rng = np.random.default_rng(0)
    N = 32
    w = np.zeros(N)
    w[0] = 1.0
    draws = rng.standard_normal((100_000, N))
    draws /= np.linalg.norm(draws, axis=1, keepdims=True)
    angles = np.degrees(np.arccos(np.clip(draws @ w, -1, 1)))
    for a in (95, 100, 115):
        mc_p = np.mean(np.abs(angles - 90) >= np.abs(a - 90))
        exact_p = exact_angle_pvalue(a, N)
        assert abs(mc_p - exact_p) < 0.01, (a, mc_p, exact_p)


def test_exact_angle_pvalue_at_90_is_one():
    assert exact_angle_pvalue(90, 50) > 0.99


def test_sin_alignment_orthogonal_is_one():
    assert abs(sin_alignment(90) - 1.0) < 1e-9


def test_sin_alignment_parallel_is_zero():
    assert abs(sin_alignment(0)) < 1e-9
    assert abs(sin_alignment(180)) < 1e-9


def test_sin_alignment_matches_sqrt_one_minus_cos2():
    """sin(angle) should equal sqrt(1 - cos^2(angle)) -- the same
    quantity exact_angle_pvalue's Beta-distribution derivation is built
    on, confirming the two are directly connected."""
    for angle in (10, 37, 63, 89, 91, 145):
        cos2 = np.cos(np.deg2rad(angle)) ** 2
        assert abs(sin_alignment(angle) - np.sqrt(1 - cos2)) < 1e-9


def test_sin_alignment_array_input():
    angles = np.array([0, 45, 90, 135, 180])
    result = sin_alignment(angles)
    expected = np.array([0, np.sqrt(2) / 2, 1, np.sqrt(2) / 2, 0])
    assert np.allclose(result, expected, atol=1e-9)


def test_exact_angle_pvalue_subspace_matches_k1_case():
    """k=1 subspace null must reduce EXACTLY to exact_angle_pvalue."""
    for angle in (75, 90, 105, 120):
        p_line = exact_angle_pvalue(angle, 40)
        p_subspace = exact_angle_pvalue_subspace(angle, 40, k=1)
        assert abs(p_line - p_subspace) < 1e-10


def test_exact_angle_pvalue_subspace_matches_monte_carlo(rng):
    """Regression test: the k>1 subspace null (fixed line vs. random
    k-dim subspace) against a Monte Carlo null built the same way
    04/06's actual analysis draws random subspaces."""
    N, k = 32, 4
    v = np.zeros(N)
    v[0] = 1.0
    cos2_draws = []
    for _ in range(20000):
        A = rng.standard_normal((N, k))
        Q, _ = np.linalg.qr(A)
        proj = Q @ (Q.T @ v)
        cos2_draws.append(np.dot(proj, proj))
    cos2_draws = np.array(cos2_draws)
    for angle in (60, 75, 89):
        c2 = np.cos(np.deg2rad(angle)) ** 2
        mc_p = np.mean(cos2_draws >= c2)
        exact_p = exact_angle_pvalue_subspace(angle, N, k)
        assert abs(mc_p - exact_p) < 0.02, (angle, mc_p, exact_p)


def test_exact_angle_pvalue_subspace_null_shifts_with_k():
    """A random subspace of higher dimension k is more likely to pass
    close to any fixed line just by chance, so for the SAME observed
    angle, the k-corrected p-value should be LARGER (less surprising /
    more consistent with pure chance) for larger k -- i.e. what looks
    like a highly significant non-orthogonality against a k=1 line null
    can look completely unremarkable against a large-k subspace null."""
    angle = 60  # a fairly non-orthogonal angle
    N = 64
    p_k1 = exact_angle_pvalue_subspace(angle, N, k=1)
    p_k4 = exact_angle_pvalue_subspace(angle, N, k=4)
    p_k8 = exact_angle_pvalue_subspace(angle, N, k=8)
    assert p_k1 < p_k4 < p_k8


def test_participation_ratio_bounds():
    rng = np.random.default_rng(0)
    # isotropic noise -> PR should be close to n_features
    X = rng.standard_normal((5000, 10))
    pr = participation_ratio(X)
    assert 8 < pr <= 10

    # rank-1 signal (all variance along one direction) -> PR close to 1
    v = rng.standard_normal(10)
    X_rank1 = np.outer(rng.standard_normal(5000), v)
    pr1 = participation_ratio(X_rank1)
    assert pr1 < 1.5


def test_gain_ratio_positive(rng):
    W = rng.standard_normal((20, 20))
    direction = rng.standard_normal(20)
    ratio, random_gains = gain_ratio(W, direction, n_random=50, rng=rng)
    assert ratio > 0
    assert len(random_gains) == 50


def test_cv_pca_recovers_low_rank_structure(rng):
    """A truly rank-3 signal (plus small noise) should show CV R^2
    plateauing near k=3, not needing all features."""
    N, k_true = 40, 3
    n_trials = 400
    basis = rng.standard_normal((k_true, N))
    latents = rng.standard_normal((n_trials, k_true))
    X = latents @ basis + rng.standard_normal((n_trials, N)) * 0.05
    r2_curve = cv_pca_reconstruction_r2(X, max_components=10, n_folds=4, rng=rng)
    eff_dim = effective_dimensionality_from_cv(r2_curve, threshold=0.9)
    assert eff_dim <= 5
    assert r2_curve[-1] > 0.9


def test_alignment_correlation_aligned_case(rng):
    """Activity confined to the w_out direction itself -> rho ~ 1."""
    N = 20
    w_out = rng.standard_normal(N)
    w_hat = w_out / np.linalg.norm(w_out)
    coeffs = rng.standard_normal((1, 200))
    X = w_hat[:, None] @ coeffs  # (N, 200), rank-1 along w_hat
    rho = alignment_correlation(w_out, X)
    assert rho > 0.99


def test_alignment_correlation_oblique_case(rng):
    """Activity confined to a subspace ORTHOGONAL to w_out -> rho ~ 0."""
    N = 20
    w_out = np.zeros(N)
    w_out[0] = 1.0
    # activity lives entirely in dimensions 1..N-1, none along dim 0
    coeffs = rng.standard_normal((N - 1, 200))
    X = np.zeros((N, 200))
    X[1:, :] = coeffs
    rho = alignment_correlation(w_out, X)
    assert rho < 0.05


def test_cv_pca_output_r2_low_dim_target(rng):
    """z depends only on the TOP PC of X -> near-perfect R^2 with k=1."""
    N, n_trials = 30, 300
    top_direction = rng.standard_normal(N)
    top_direction /= np.linalg.norm(top_direction)
    latent = rng.standard_normal(n_trials) * 3.0  # large-variance top PC
    noise_dirs = rng.standard_normal((n_trials, N)) * 0.05
    X = latent[:, None] * top_direction[None, :] + noise_dirs
    z = latent * 2.0 + rng.standard_normal(n_trials) * 0.05
    r2_curve = cv_pca_output_r2(X, z, max_components=5, n_folds=4, rng=rng)
    assert r2_curve[0] > 0.8  # k=1 already explains most of it


def test_noise_suppression_ratio_suppressed_case(rng):
    """Residuals with near-zero variance along w_out, large variance
    elsewhere -> ratio << 1."""
    N, n_trials = 20, 500
    w_out = np.zeros(N)
    w_out[0] = 1.0
    residuals = rng.standard_normal((n_trials, N))
    residuals[:, 0] *= 0.01  # suppress variance along w_out's direction
    ratio = noise_suppression_ratio(residuals, w_out, n_random=200, rng=rng)
    assert ratio < 0.5


def test_noise_suppression_ratio_amplified_case(rng):
    """Residuals with LARGE variance along w_out -> ratio >> 1."""
    N, n_trials = 20, 500
    w_out = np.zeros(N)
    w_out[0] = 1.0
    residuals = rng.standard_normal((n_trials, N))
    residuals[:, 0] *= 10.0
    ratio = noise_suppression_ratio(residuals, w_out, n_random=200, rng=rng)
    assert ratio > 5


def test_relative_susceptibility():
    strengths = np.linspace(0, 5, 6)
    # perturbing along w_out barely hurts accuracy; along PC hurts a lot
    acc_wout = np.full(6, 0.95)
    acc_pc = np.linspace(0.95, 0.5, 6)
    r = relative_susceptibility(acc_wout, acc_pc, strengths)
    assert r < 0.5


def test_effective_connectivity_modes_recovers_exact_low_rank(rng):
    """A genuinely rank-2 matrix built from known m/n vectors must be
    reconstructed exactly (up to float precision) by SVD extraction."""
    N = 20
    m1, n1 = rng.standard_normal(N), rng.standard_normal(N)
    m2, n2 = rng.standard_normal(N), rng.standard_normal(N)
    W = (np.outer(m1, n1) + np.outer(m2, n2)) / N
    M, Nmat = effective_connectivity_modes(W, rank=2)
    W_recon = (M @ Nmat.T) / N
    assert np.abs(W - W_recon).max() < 1e-10


def test_multi_regressor_selectivity_recovers_known_betas(rng):
    design = rng.standard_normal((300, 2))
    true_beta = rng.standard_normal((2, 15))
    activity = design @ true_beta + rng.standard_normal((300, 15)) * 0.01
    betas = multi_regressor_selectivity(activity, design, ["a", "b"])
    assert np.allclose(betas["a"], true_beta[0], atol=0.15)
    assert np.allclose(betas["b"], true_beta[1], atol=0.15)
    assert set(betas.keys()) == {"a", "b"}
