# -*- coding: utf-8 -*-
"""
population/jpca.py
==================
Rotational dynamics via jPCA, checked against TWO nulls (see docs/null_hypotheses_framework.md): time-permutation (general overfitting) and phase-randomization (targets the 'oscillation looks like rotation regardless of coordination' confound specifically). Require a result to clear BOTH before calling it real rotational structure.

TODO(you):
- fit_jpca(tensor, condition, n_dims, ...) -> rotation matrices / R^2 of the skew-symmetric fit (trim near-degenerate dimensions before fitting -- more free parameters than samples causes spurious 'rotation')
- jpca_null_time_permutation(tensor, condition, rng) -> null R^2 dist
- jpca_null_phase_randomization(tensor, rng) -> null R^2 dist (uses nulls.phase_randomization_null)
"""

import numpy as np  # TODO: trim/add imports as needed

