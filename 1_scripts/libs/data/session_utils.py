# -*- coding: utf-8 -*-
"""
data/session_utils.py
=====================
Protocol-aware (N neurons, K trials) response-matrix construction: time-locked window vs. spatial window, dispatched on session.protocol.

TODO(you):
- TIME_LOCKED_PROTOCOLS / SPATIAL_PROTOCOLS -- name your own protocols here
- compute_respmat_for_session(session, ..., s_resp_start, s_resp_stop) -> fills session.respmat, dispatching on protocol
"""

import numpy as np  # TODO: trim/add imports as needed

