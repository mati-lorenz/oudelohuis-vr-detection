# -*- coding: utf-8 -*-
"""
population/tdr.py
=================
Targeted Dimensionality Reduction: regression-based encoding of stim/choice in PCA space, with a label-permutation null (small-sample regression with many PCA dims and modest trial counts is exactly the regime where overfitting alone inflates R^2).

TODO(you):
- fit_tdr_encoding(tensor, stim, choice, n_pca_dims, ...) -> R^2(t) per variable
- tdr_null(tensor, stim, choice, rng, n_shuffles) -> null R^2(t) distribution via nulls.label_permutation_null
"""

import numpy as np  # TODO: trim/add imports as needed

