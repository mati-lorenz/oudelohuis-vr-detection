# -*- coding: utf-8 -*-
"""
data/tensor_utils.py
====================
Protocol-aware (K, N, T) tensor construction, mirroring session_utils.py's dispatch. If your two underlying windowing functions return axes in different orders, auto-detect and return a CANONICAL (K, N, T) shape so downstream code never has to think about axis order.

TODO(you):
- compute_tensor_for_session(session, ..., t_pre, t_post, s_pre, s_post, binsize) -> tensor, axis, axis_label
- _orient_to_KNT(tensor, n_trials, n_neurons) -> canonical axis order (match axis lengths against known K/N; raise clearly if ambiguous, e.g. K == N)
"""

import numpy as np  # TODO: trim/add imports as needed

