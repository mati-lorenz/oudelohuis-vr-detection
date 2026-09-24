import numpy as np
from utils.perturbation import make_perturbation_direction, strength_grid


def test_not_normalized_is_unit_norm(rng):
    v = make_perturbation_direction(64, rng, normalize_by_size=False)
    assert abs(float(v.norm()) - 1.0) < 1e-5


def test_normalized_by_size_scales_with_sqrt_n(rng):
    v64 = make_perturbation_direction(64, rng, normalize_by_size=True)
    v256 = make_perturbation_direction(256, rng, normalize_by_size=True)
    assert abs(float(v64.norm()) - np.sqrt(64)) < 1e-4
    assert abs(float(v256.norm()) - np.sqrt(256)) < 1e-4


def test_strength_grid_multiplicative_wider_than_additive():
    add = strength_grid("additive", normalize_by_size=False)
    mul = strength_grid("multiplicative", normalize_by_size=False)
    assert mul.max() > add.max()


def test_strength_grid_normalized_smaller_max():
    unnorm = strength_grid("additive", normalize_by_size=False)
    norm = strength_grid("additive", normalize_by_size=True)
    assert norm.max() < unnorm.max()
