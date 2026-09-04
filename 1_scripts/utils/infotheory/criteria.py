# -*- coding: utf-8 -*-
"""
Shared QC thresholds. Centralized so 1a_performance's "worth a second
look" flags and 1b_psychometric's hard exclusion criteria can't silently
drift apart from each other.
"""

MIN_TRIALS = 50          # sessions with fewer trials than this: flagged in 1a
MIN_FRAC_ENGAGED = 0.3   # engaged-trial fraction below this: flagged in 1a, excluded in 1b
MIN_DPRIME = 1.0         # d' (engaged trials) below this: excluded in 1b
MAX_FA_RATE = 0.5        # false-alarm rate (engaged trials) above this: excluded in 1b
