# -*- coding: utf-8 -*-
"""
Generates figures to characterize the mice behavior across the sessions

If a session is missing video tracking (no videodata) or behavioral
tracking (no zpos_F/runspeed_F), the corresponding variable(s) are
skipped for that session with a warning, rather than failing the whole
run -- results are still pooled across whichever sessions DID have each
variable.

Produces one figure (2x2 grid, one panel per behavioral variable) plus
a per-neuron summary CSV per variable.
"""

#%%
import os, sys, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']  # avoid 'findfont: Arial not found' warning
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from utils.session_info import filter_sessions, report_sessions
# from utils import InfoTheoryParams, BinningParams, BiasCorrectionParams, ParallelParams
from infotheory.behavior_signals import get_behavior_trace
from infotheory.behavior_mi import compute_behavior_mi
from infotheory.celldata_utils import (
    get_area_label, ordered_groups, bar_by_group, AREA_COLORS, LABEL_LINESTYLES, DEFAULT_LABEL_ORDER)
from utils.cellselection_lib import filter_nearlabeled_layer23
from infotheory.qc_lib import load_qc_pass

#%% ------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
NAME = '1_behavior' ## Name of the notebook goes here (without the file extension!)
PROJECT = 'multi_area_detection_task_ff_fb'
PYTHON_VERSION = '3.13'

## Set working directory  
"""
The code below will traverse the path upwards until it finds the root folder of the project.
"""

workdir = re.sub("(?<={})[\w\W]*".format(PROJECT), "", os.getcwd())
os.chdir(workdir)


## Set  up pipeline folder if missing  
"""
The code below will automatically create a pipeline folder for this code file if it does not exist.
"""

if os.path.exists(os.path.join('empirical', '2_pipeline')):
    pipeline = os.path.join('empirical', '2_pipeline', NAME)
else:
    pipeline = os.path.join('2_pipeline', NAME)
    
if not os.path.exists(pipeline):
    os.makedirs(pipeline)
    for folder in ['out', 'store', 'tmp']:
        os.makedirs(os.path.join(pipeline, folder))

protocol = ['DM', 'DP', 'DN']
calciumversion = 'deconv'

behavior_vars = ['position', 'runspeed', 'pupil_area', 'video_pc1']
video_pc_column = None   # set explicitly once you know your dataset's face-motion PC
                          # column name, e.g. 'motSVD_0' -- see behavior_signals.py



# params = InfoTheoryParams(
#     binning=BinningParams(method='equipopulated', n_bins=4),   # neuron response binning
#     bias=BiasCorrectionParams(panzeri_treves=True, random_state=0),
#     parallel=ParallelParams(n_jobs=1, backend='loky', verbose=0),  # neuron-level n_jobs set below
# )
compute_significance = False   # shuffle-test p-values per neuron; expensive at population scale

# n_jobs at the SESSION level (primary parallelism, one worker per session).
# Keep n_jobs_neurons small when running many sessions in parallel to avoid
# oversubscribing CPUs with nested parallel pools; bump n_jobs_neurons if
# you only have 1-2 sessions.
n_jobs_sessions = -1
n_jobs_neurons = 1

#%% ------------------------------------------------------------------
# Load sessions (shallow)
# ----------------------------------------------------------------------
sessions, nSessions = filter_sessions(
    protocols=protocol, load_behaviordata=True, load_calciumdata=False,
    load_videodata=False, min_trials=100)
report_sessions(sessions)
