# -*- coding: utf-8 -*-
"""
population/gpfa.py
==================
Single-trial trajectories via GPFA (or a lighter proxy -- see the 'lighter-weight diagnostic' note below) with a null for whether trial-to-trial covariation is genuinely shared, not independent noise around a shared condition mean.

TODO(you):
- fit_gpfa_trajectories(tensor, condition, n_factors, ...) -> per-trial latent trajectories
- report_shared_variance(tensor, condition, rng, n_null_repeats) -> leading eigenvalue fraction of the trial-to-trial covariance, real vs. nulls.shuffle_residuals_within_condition repeats -- a MUCH cheaper diagnostic than refitting full GPFA per shuffle; use this for the null check even if you fit true GPFA once on the real data.
"""

import numpy as np  # TODO: trim/add imports as needed

