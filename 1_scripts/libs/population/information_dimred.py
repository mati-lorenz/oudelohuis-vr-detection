# -*- coding: utf-8 -*-
"""
population/information_dimred.py
================================
Information carried by a population-level latent (e.g. the top PCA/FA component), shuffle-corrected for 'is there information at all', PLUS the comparison against the best single neuron's own MI -- the remaining open question flagged in docs/null_hypotheses_framework.md: does the population framing add anything beyond one dominant neuron's tuning?

TODO(you):
- latent_mi_with_stats(latent, target, params) -> core.mi_with_stats on the reduced latent
- compare_latent_vs_best_single_neuron(latent_mi, per_neuron_mi_df) -> how much (if any) the population latent exceeds the single best neuron -- this comparison was still TODO as of the last design pass
"""

import numpy as np  # TODO: trim/add imports as needed

