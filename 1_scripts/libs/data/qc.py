# -*- coding: utf-8 -*-
"""
data/qc.py
==========
Per-cell QC pass/fail loading, produced by a separate QC-summary script and consumed everywhere neurons get filtered before analysis.

TODO(you):
- load_qc_pass(session, qc_csv_path) -> boolean array, length == n_neurons, aligned to session.celldata's row order. Decide what happens if qc_csv_path is None or missing -- likely: pass everything, with a printed warning, not a silent all-True.
"""

import numpy as np  # TODO: trim/add imports as needed

