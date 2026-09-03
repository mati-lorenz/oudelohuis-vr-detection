# -*- coding: utf-8 -*-
"""
pid.py
======
Partial information decomposition (redundancy/unique/synergy) for two sources about one target -- used both for single-cell PID (neuron is target, stim/choice are sources) and pairwise PID (two neurons are sources, stim/choice is target).

TODO(you):
- Pick and document ONE PID definition (e.g. Williams & Beer redundancy lattice, or MMI) -- different definitions give different numbers, don't mix them silently.
- pid_decomposition(sources, target, n_bins, ...) -> dict(redundancy, unique_1, unique_2, synergy, I_total), naive plug-in
- Add shuffle-subtraction bias correction (same rationale as breakdown.py -- naive PID terms are biased at realistic trial counts; see tests/test_pid_canonical.py for a zero-information sanity check that should show near-zero spurious synergy once corrected).
"""

import numpy as np  # TODO: trim/add imports as needed

