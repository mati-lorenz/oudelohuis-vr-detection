# -*- coding: utf-8 -*-
"""
data/behavior_signals.py
========================
Extract continuous behavioral/state traces aligned to imaging-frame timestamps, so they can be compared frame-by-frame against neural activity.

TODO(you):
- get_behavior_trace(session, var_name, ...) -> 1D array, length == len(session.ts_F). Raise a clear, actionable error listing what WAS found if a variable/column is missing -- don't fail silently or guess.
- _interp_to_reference(values, src_times, ref_times) -> np.interp onto ts_F for variables sampled at a different rate (e.g. video)
"""

import numpy as np  # TODO: trim/add imports as needed

