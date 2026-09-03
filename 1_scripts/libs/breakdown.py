# -*- coding: utf-8 -*-
"""
breakdown.py
============
Pola, Thiele, Hoffmann & Panzeri (2003) exact four-term information breakdown for a pair of neurons about a stimulus/task variable: I_full = I_lin + I_sig_sim + I_cor_indep + I_cor_dep. Naive plug-in probabilities are biased, and that bias does NOT cancel between I_full and I_lin -- use shuffle-subtraction bias correction (matches MINT, Lorenz et al. 2025) rather than trusting the naive breakdown at realistic trial counts.

TODO(you):
- _breakdown_terms_from_binned(r1_binned, r2_binned, s, n1, n2) -> dict of the 7 terms, pure function of already-binned data + labels
- pairwise_information_breakdown(x1, x2, s, n_bins, ..., shuffle_correction=False, n_shuffles=30) -> dict; if shuffle_correction, ALSO report *_shuffcorr terms that sum exactly (the correction is linear, so exact-sum identity survives it)
- Write a canonical-example regression test FIRST (see tests/test_breakdown_canonical.py) before trusting any of this.
"""

import numpy as np  # TODO: trim/add imports as needed

