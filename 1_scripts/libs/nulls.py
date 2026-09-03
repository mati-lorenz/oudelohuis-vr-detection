# -*- coding: utf-8 -*-
"""
nulls.py
========
Shared null-hypothesis constructors used across stage 2/3 (see docs/null_hypotheses_framework.md). Keep each null as a pure function of (data, rng) -> surrogate data, so the SAME null can be reused by both the significance test and any diagnostic plot.

TODO(you):
- shuffle_neurons_independently(X, rng) -> shuffle each neuron's trial order independently (dimensionality null)
- shuffle_residuals_within_condition(X, condition, rng) -> subtract condition-mean, shuffle residuals per neuron within condition (GPFA null)
- time_permutation_null(X, condition, rng) -> shuffle time bins within condition (jPCA null #1, general overfitting check)
- phase_randomization_null(X, rng) -> preserve each dimension's own power spectrum, randomize cross-dimension phase (jPCA null #2, targets the oscillation/ellipse confound specifically)
- label_permutation_null(labels, rng) -> shuffle trial labels (TDR, breakdown, PID)
"""

import numpy as np  # TODO: trim/add imports as needed

