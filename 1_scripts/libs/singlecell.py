# -*- coding: utf-8 -*-
"""
singlecell.py
=============
Stage 1: single-cell and cell-pair information about stimulus and choice, for one session.

TODO(you):
- get_trial_labels(session, params) -> stim, choice, mask (apply stim_value_map / stim_binarize_threshold / trial_mask_var from InfoTheoryParams here, in ONE place, so every stage applies trial selection identically)
- run_session_single_cell_analysis(session, params, do_pairwise, max_pairs) -> dict(single_cell=DataFrame, pairwise=DataFrame) -- single_cell via core.mi_with_stats per neuron per variable; pairwise via breakdown.pairwise_information_breakdown on a random subsample of pairs (see celldata_utils.sample_pairs_within_groups if you want pairs restricted within area/label groups instead of fully random)
"""

import numpy as np  # TODO: trim/add imports as needed

