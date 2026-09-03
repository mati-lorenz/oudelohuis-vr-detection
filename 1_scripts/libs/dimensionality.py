# -*- coding: utf-8 -*-
"""
dimensionality.py
=================
Stage 2: cross-validated dimensionality (FA vs PCA, raw and z-scored variants) with a shuffle-null significance check -- see docs/null_hypotheses_framework.md's dimensionality-estimate section. Raw PCA (isotropic noise) needs more components than FA (per-neuron noise) purely from heterogeneous private variance, NOT genuine extra shared structure; z-scoring narrows but may not close that gap.

TODO(you):
- compute_explained_variance(X, n_components, ...) -> dict per method {'pca','pca_zscored','fa','fa_zscored'} -> explained-variance-ratio array
- cv_dimensionality(X, n_components_list, cv_folds, ...) -> DataFrame(method, n_components, cv_score_mean, cv_score_sem); z-scored variants must fit the StandardScaler on the TRAIN FOLD ONLY (no leakage into the held-out score)
- dimensionality_significance_test(X, ..., n_null_repeats) -> compares real cv_dimensionality peak against nulls.shuffle_neurons_independently repeats, per method
- Track 'at_grid_boundary' (did the CV curve peak AT the largest n_components tried, meaning the true optimum might be higher) as its own diagnostic column -- don't silently report a boundary hit as a real optimum.
"""

import numpy as np  # TODO: trim/add imports as needed

