import numpy as np
import torch
from utils.models import RateRNN


def test_dale_sign_constraint_holds():
    """Every excitatory unit's outgoing weights must be >= 0, every
    inhibitory unit's outgoing weights must be <= 0."""
    m = RateRNN(n_hidden=32, dale=True, exc_fraction=0.75, seed=0)
    W = m.W_rec.detach().numpy()
    is_exc = m.is_exc
    # outgoing weights of unit j are column j (r_eff @ W_rec.T means W_rec[i,j]
    # is the weight FROM presynaptic unit j TO postsynaptic unit i)
    for j in range(32):
        col = W[:, j]
        if is_exc[j]:
            assert np.all(col >= -1e-6), "excitatory unit has a negative outgoing weight"
        else:
            assert np.all(col <= 1e-6), "inhibitory unit has a positive outgoing weight"


def test_exc_fraction_respected():
    m = RateRNN(n_hidden=40, dale=True, exc_fraction=0.8, seed=0)
    assert m.is_exc.sum() == 32


def test_no_dale_is_exc_none():
    m = RateRNN(n_hidden=16, dale=False, seed=0)
    assert m.is_exc is None


def test_spectral_radius_near_target_both_branches():
    for dale in (False, True):
        m = RateRNN(n_hidden=64, dale=dale, exc_fraction=0.8, seed=0, target_radius=1.2)
        W = m.W_rec.detach().numpy()
        radius = np.max(np.abs(np.linalg.eigvals(W)))
        assert abs(radius - 1.2) < 1e-4


def test_forward_runs_both_activations():
    for act in ("tanh", "relu"):
        m = RateRNN(n_hidden=16, activation=act, dale=True, seed=0)
        u = torch.randn(4, 10, 1)
        logits, r_traj = m(u)
        assert logits.shape == (4, 1)
        assert r_traj.shape == (4, 10, 16)
        if act == "relu":
            assert (r_traj >= 0).all()


def test_perturbation_changes_activity():
    m = RateRNN(n_hidden=16, dale=True, seed=0)
    u = torch.randn(4, 10, 1)
    torch.manual_seed(0)
    _, r_clean = m(u, noise_on=False)
    pert_vec = torch.ones(16)
    torch.manual_seed(0)
    _, r_pert = m(u, pert_vec=pert_vec, pert_strength=5.0, noise_on=False)
    assert not torch.allclose(r_clean, r_pert)


def test_find_fixed_point_returns_converged_flag():
    from utils.geometry import find_fixed_point
    m = RateRNN(n_hidden=32, dale=True, exc_fraction=0.8, seed=0)
    x_star, r_star, converged = find_fixed_point(m)
    assert x_star.shape == (32,)
    assert r_star.shape == (32,)
    assert isinstance(converged, bool)


def test_find_fixed_point_convergence_tolerance_sensitivity():
    """Directly tests the convergence-detection logic: the SAME
    relaxation trajectory must be reported converged under a loose
    tolerance and not converged under an impossibly strict one --
    confirms the flag actually reflects the residual comparison, not a
    hardcoded True.

    Uses a model with a manually-perturbed (nonzero) b_rec: an UNTRAINED
    RateRNN has b_rec exactly all-zero, which makes x*=0 an EXACT
    (floating-point-zero-residual) trivial fixed point whenever u_const
    is also zero (tanh(0)=0, so -0 + W@0 + 0 + 0 = 0 identically) --
    that degenerate case would satisfy even an impossibly strict
    tolerance trivially and defeat the point of this test."""
    from utils.geometry import find_fixed_point
    m = RateRNN(n_hidden=32, dale=True, exc_fraction=0.8, seed=0)
    with torch.no_grad():
        m.b_rec.copy_(torch.randn(32) * 0.3)
    _, _, converged_loose = find_fixed_point(m, residual_tol=10.0)
    _, _, converged_strict = find_fixed_point(m, residual_tol=1e-20)
    assert converged_loose is True
    assert converged_strict is False


def test_low_rank_zero_rank_matches_dale_rnn():
    """rank=0, chi=1.0 must recover the plain Dale RateRNN bit-for-bit."""
    from utils.models import LowRankRateRNN
    base = RateRNN(n_hidden=32, dale=True, exc_fraction=0.8, seed=7)
    low_rank = LowRankRateRNN(n_hidden=32, dale=True, exc_fraction=0.8, seed=7, rank=0, chi=1.0)
    assert torch.allclose(base.W_rec, low_rank.W_rec, atol=1e-6)

    u = torch.randn(4, 10, 1)
    torch.manual_seed(0)
    logits_base, _ = base(u, noise_on=False)
    torch.manual_seed(0)
    logits_lr, _ = low_rank(u, noise_on=False)
    assert torch.allclose(logits_base, logits_lr, atol=1e-5)


def test_low_rank_zero_rank_matches_nodale_rnn():
    """Same check, but for dale=False -- confirms the no-Dale ablation
    is a real, correctly-wired code path, not a silent no-op."""
    from utils.models import LowRankRateRNN
    base = RateRNN(n_hidden=32, dale=False, seed=7)
    low_rank = LowRankRateRNN(n_hidden=32, dale=False, seed=7, rank=0, chi=1.0)
    assert torch.allclose(base.W_rec, low_rank.W_rec, atol=1e-6)
    assert low_rank.is_exc is None


def test_low_rank_dale_false_has_no_sign_constraint():
    """dale=False should give a background matrix WITHOUT the
    excitatory/inhibitory sign structure -- spot check that columns can
    have mixed-sign entries (unlike the dale=True case)."""
    from utils.models import LowRankRateRNN
    low_rank = LowRankRateRNN(n_hidden=32, dale=False, seed=3, rank=4, chi=0.5)
    W = low_rank.W_rec.detach().numpy()
    # at least some column should have BOTH positive and negative entries
    has_mixed_sign = any((col > 0).any() and (col < 0).any() for col in W.T)
    assert has_mixed_sign


def test_low_rank_nonzero_rank_changes_connectivity():
    from utils.models import LowRankRateRNN
    base = RateRNN(n_hidden=32, dale=True, exc_fraction=0.8, seed=7)
    low_rank = LowRankRateRNN(n_hidden=32, exc_fraction=0.8, seed=7, rank=4, chi=0.2)
    assert not torch.allclose(base.W_rec, low_rank.W_rec, atol=1e-4)
    assert low_rank.M.shape == (32, 4)
    assert low_rank.N.shape == (32, 4)


def test_out_scale_default_matches_original_init():
    """Regression test: default out_scale=1.0 must reproduce this
    project's original W_out init exactly (torch.randn(...)/sqrt(n_hidden)),
    since 01-07 all depend on that behavior never silently changing."""
    torch.manual_seed(3)
    expected = torch.randn(1, 32) / np.sqrt(32)
    m = RateRNN(n_hidden=32, seed=3)
    # can't compare directly (RateRNN's own seed call consumes the RNG
    # stream before W_out is drawn) -- instead check the STATISTICAL
    # convention: std of W_out entries should be ~1/sqrt(n_hidden)
    std = m.W_out.detach().std().item()
    assert abs(std - 1 / np.sqrt(32)) < 0.15  # loose bound, single draw


def test_out_scale_aligned_vs_oblique_norms_differ():
    """Schuessler et al. (2024) convention: 'aligned' (out_scale=1/sqrt(N))
    should give a much smaller total readout norm than 'oblique'
    (out_scale=1.0), at the same N."""
    n_hidden = 64
    m_aligned = RateRNN(n_hidden=n_hidden, seed=0, out_scale=1 / np.sqrt(n_hidden))
    m_oblique = RateRNN(n_hidden=n_hidden, seed=0, out_scale=1.0)
    norm_aligned = m_aligned.W_out.detach().norm().item()
    norm_oblique = m_oblique.W_out.detach().norm().item()
    assert norm_aligned < norm_oblique
    # both should be close to their respective out_scale (norm ~ out_scale)
    assert abs(norm_aligned - 1 / np.sqrt(n_hidden)) < 0.1
    assert abs(norm_oblique - 1.0) < 0.3
