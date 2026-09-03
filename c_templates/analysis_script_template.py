# -*- coding: utf-8 -*-
"""
<one-line description of what this script computes>

Copy this into <step>/scripts/, rename it, and fill in each section. This
skeleton mirrors the shape every script in this project has converged on:
config -> load sessions -> per-session parallel compute (with anatomical +
QC filtering FIRST, before anything is computed) -> pool across sessions
-> null-hypothesis check if applicable -> save tables -> save figures ->
update the step's README with what you found.
"""

#%%
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from infotheory import InfoTheoryParams  # TODO: + whatever else this script needs
from infotheory.celldata_utils import get_area_label, ordered_groups
from infotheory.plotting import apply_style, save_figure, save_table_markdown
from infotheory.data.loaders import YourLabSessionLoader  # TODO: your actual loader

apply_style()

#%% ------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
STEP_DIR = "."          # TODO: path to this step's folder (behavior/single/pairs/area/multi_area)
protocol = ["DN"]
calciumversion = "deconv"
# TODO: analysis-specific params (InfoTheoryParams(...), n_bins, windows, ...)

#%% ------------------------------------------------------------------
# Load sessions
# ----------------------------------------------------------------------
loader = YourLabSessionLoader()  # TODO
session_ids = loader.list_sessions(protocol=protocol)

#%% ------------------------------------------------------------------
# Per-session computation (parallelized across sessions)
# ----------------------------------------------------------------------
def process_session(session_id):
    session = loader.load(session_id)
    # TODO: anatomical filter (projection identity / area) FIRST
    # TODO: QC filter FIRST -- see infotheory.data.qc.load_qc_pass
    # TODO: the actual computation for this script
    raise NotImplementedError


results = Parallel(n_jobs=-1, backend="loky", verbose=10)(
    delayed(process_session)(sid) for sid in session_ids
)

#%% ------------------------------------------------------------------
# Pool across sessions
# ----------------------------------------------------------------------
# TODO: pd.concat(results, ignore_index=True)

#%% ------------------------------------------------------------------
# Null-hypothesis check (if this analysis makes a "is this real" claim --
# see docs/null_hypotheses_framework.md for which null actually applies)
# ----------------------------------------------------------------------
# TODO

#%% ------------------------------------------------------------------
# Save tables + figures
# ----------------------------------------------------------------------
# TODO: save_table_markdown(df, STEP_DIR, "result_name")
# TODO: fig, ax = plt.subplots(...); ...; save_figure(fig, STEP_DIR, "figure_name")

#%% ------------------------------------------------------------------
# Don't forget: update <step>/README.md's Findings section with what
# this script showed, and Open questions with what it raised.
# ----------------------------------------------------------------------
