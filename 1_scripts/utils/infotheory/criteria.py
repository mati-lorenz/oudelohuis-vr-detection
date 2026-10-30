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
MAX_GUESS_RATE = 0.4     # DN only: fitted guess rate (floor of the psychometric curve) above this: excluded in 1b
                         # (same limit as the other lab pipeline's get_idx_performing_sessions, guess_thr=0.4)

# BORDERLINE zone (DN fit criteria only). A session whose value is within this distance of the cutoff, on either
# side, is KEPT (if it passes or is within the zone) but FLAGGED `borderline` in 1b's psychometric_fits.csv, because
# a different fit (bootstrap seed, single fit vs median) can flip it. Outside the zone the criterion excludes as usual.
BORDERLINE_Z = 0.15      # threshold z-range: distance (in sigma) of the tested range edge from the fitted threshold
BORDERLINE_GUESS = 0.02  # guess rate: distance from MAX_GUESS_RATE
