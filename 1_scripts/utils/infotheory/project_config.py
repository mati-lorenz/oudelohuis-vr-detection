# -*- coding: utf-8 -*-
"""
project_config.py
==================
THE one place for settings that must be identical across scripts.
Scripts import from here (and expose command-line overrides), they do not
re-define these values locally.

1. WHICH DATA (selection)           ANIMALS / SESSIONS
2. HOW TO AGGREGATE (unit)          AGGREGATE_BY  = "pooled" | "session" | "mouse"
3. MEMORY                           N_JOBS_SESSIONS (sessions processed in parallel, one in RAM each)
4. SHARED BINNING                   so correlations, MI, information breakdown and PID are computed on
                                    the SAME trials and the SAME bins and are comparable BEFORE any bias
                                    correction (see the block below)

Command-line overrides (every script that calls scope.add_scope_args):
    python 1a_performance.py --animals DNL00 DNL01
    python 1a_performance.py --sessions DNL00_2024_01_09
    python 1a_performance.py --aggregate mouse
"""
from __future__ import annotations

# --------------------------------------------------------------------- #
# 1. Selection -- None = everything. Animal ids as in 0_data/<protocol>/<animal_id>/.
#    Session ids are "<animal_id>_<sessiondate>". If both are given, a session must match BOTH.
# --------------------------------------------------------------------- #
ANIMALS = None          # e.g. ["DNL00"]
SESSIONS = None         # e.g. ["DNL00_2024_01_09", "DNL00_2024_01_10"]

# --------------------------------------------------------------------- #
# 2. Aggregation = the UNIT OF ANALYSIS used when plotting and reporting
#    "pooled" : everything pooled into one unit per condition (counts are summed over all
#               sessions/mice, rates are recomputed from the pooled counts). No error bars.
#    "session": one unit per session. Bars = mean over sessions, error = SEM over sessions,
#               dots = single sessions (coloured by mouse).
#    "mouse"  : one unit per mouse (a mouse's sessions are pooled together first). Bars = mean over
#               mice, error = SEM over mice, dots = single mice.
# --------------------------------------------------------------------- #
AGGREGATE_BY = "session"
AGGREGATES = ("pooled", "session", "mouse")

# --------------------------------------------------------------------- #
# 3. Memory: sessions are NEVER all loaded at once. Each worker loads ONE session, computes and
#    caches its result, and frees it. Peak RAM ~ N_JOBS_SESSIONS x one session (1 = lowest).
# --------------------------------------------------------------------- #
N_JOBS_SESSIONS = 14

# --------------------------------------------------------------------- #
# 4. Shared binning (consumed by 2c, 3a-3f, 4a-4c once migrated; at the moment only 1a follows the
#    selection/aggregation rules). The rule: any two quantities that are compared with each other (correlation, MI,
#    information breakdown, PID) must be built from the SAME
#       * trials          -> engaged trials, labels finite, >= MIN_TRIALS_PER_TARGET_BIN per target bin,
#                            >= MIN_TRIALS_ANALYSIS in total
#       * target labels   -> stimulus bins from stim_binning.STIM_BINNING_STRATEGIES (+ "choice")
#       * response bins   -> RESPONSE_BINS equal-occupancy (quantile) bins per cell
#    Only the estimator may differ between scripts, and bias correction is applied AFTER.
#    Edit stimulus strategies in stim_binning.py; everything else below.
# --------------------------------------------------------------------- #
RESPONSE_BINS = 3                  # per-cell response bins for discrete information measures
RESPONSE_BIN_METHOD = "quantile"   # equal-occupancy
MIN_TRIALS_PER_TARGET_BIN = 1      # trials needed in a stimulus/choice bin for it to be used
MIN_TRIALS_ANALYSIS = 20           # trials needed (after the filters above) to analyse a cell/pair
