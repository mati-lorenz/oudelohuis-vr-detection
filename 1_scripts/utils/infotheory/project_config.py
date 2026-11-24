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
4. DEPTH RULE                       which cells enter the area comparison (max depth) and the extra deep-PM groups
5. SHARED BINNING                   so correlations, MI, information breakdown and PID are computed on
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
N_JOBS_SESSIONS = 5

#    Caps on the number of units analysed per session (None = all). --max-cells / --max-pairs override them per run.
#    The cap picks a fixed pseudo-random subset (first N of a hash order of the session's cells). The per-session caches hold only
#    the final tables of one selection (no per-cell / per-pair ledger, to save disk): changing the cap recomputes the session.
#    A run with a cap writes to out/selection/<tag>/.
# The caps apply PER GROUP (V1-lab, V1-unl, PM-lab, ... incl. deep PM) within a session: MAX_CELLS_PER_SESSION = 10 keeps up to 10 cells of each
# group, so rare groups (labeled cells) are not emptied; MAX_PAIRS_PER_SESSION counts pairs per pair of groups.
MAX_CELLS_PER_SESSION = None
MAX_PAIRS_PER_SESSION = 400

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
MIN_TRIALS_PER_TARGET_BIN = 3      # trials needed in a stimulus/choice bin for it to be used
MIN_TRIALS_ANALYSIS = 20           # trials needed (after the filters above) to analyse a cell/pair

# --------------------------------------------------------------------- #
# 5. Depth rule. Defined ONCE in 2a_cell_distribution (column `depth_class` of celldata_combined.csv) and inherited by every
#    later step through that file (neural_encoding.load_cell_inclusion, infotheory.groups.add_six_group_column).
#    To COMPARE AREAS only cells shallower than MAX_DEPTH_UM are used (all areas, labeled and unlabeled), so all areas are
#    compared at the same depth. Because deep PM cells matter for the feedback loop, PM cells between DEEP_PM_DEPTH_UM[0] and
#    [1] form two EXTRA groups (PM-lab (deep), PM-unl (deep)) that are shown next to the six shallow groups. Everything else
#    deeper than MAX_DEPTH_UM (deep V1 / AL / RSP, PM deeper than DEEP_PM_DEPTH_UM[1], cells without depth) is excluded.
#      depth_class:  "shallow"  depth < MAX_DEPTH_UM                  -> the six groups
#                    "deep_pm"  PM, DEEP_PM_DEPTH_UM[0] <= depth < [1]  -> the two deep-PM groups
#                    "excluded" everything else                          -> no group, not analysed
#    MAX_DEPTH_UM = None switches the whole rule off (old behaviour: only unlabeled V1/PM cells <300 um were kept in 2a).
#    DEEP_PM_DEPTH_UM = None keeps the max-depth rule but drops the deep-PM groups.
#    Sessions whose depth looks badly measured (2a `depth_suspect`) cannot be classified reliably:
#    DEPTH_SUSPECT_POLICY "exclude" = their cells are `excluded`; "keep" = classified by their (doubtful) depth anyway.
# --------------------------------------------------------------------- #
MAX_DEPTH_UM = 300.0
DEEP_PM_DEPTH_UM = (300.0, 500.0)
DEPTH_SUSPECT_POLICY = "exclude"

# --------------------------------------------------------------------- #
# 6. POOLED-FRAMES information (2c). Besides the per-TRIAL values (one window mean per trial, RESPONSE_BINS bins) 2c also
#    computes MI between a behavior (runspeed, pupil, motion energy, lick) and a cell's activity with EVERY imaging frame
#    inside the window of every engaged trial pooled into one distribution (predictors named "<cur_var>__frames").
#    * more samples allow finer bins: with a few hundred trials 3 bins (9 joint cells) is all the data support, with thousands
#      of frames RESPONSE_BINS_FRAMES bins (64 joint cells, ~100 frames per cell) are affordable. The two families therefore
#      use different bin numbers on purpose; the per-trial one stays at RESPONSE_BINS because it is shared with the
#      correlations / breakdown / PID (rule: shared binning before bias correction).
#    * zero-inflated variables (lick rate, deconvolved activity: one value holds > 1/bins of the frames) get that value as its
#      own bin and equal-occupancy bins over the rest (otherwise the bins collapse and the MI is 0); continuous ones are binned
#      exactly like mutual_information_hist.
#    * frames of one trial are autocorrelated, so the null is NOT a frame permutation: the cell trace is circularly rolled
#      by >= FRAME_SHIFT_MIN_TRIALS windows' worth of frames (keeps the autocorrelation of both signals) and bias
#      correction = mean of the rolled MIs, p = fraction of rolled MIs >= raw. The p-value is still optimistic because
#      frames from one trial are not independent samples of the trial-to-trial variability, so compare the pooled-frame and
#      per-trial results rather than trusting the frame p-values alone.
# --------------------------------------------------------------------- #
RESPONSE_BINS_FRAMES = 8
MIN_FRAMES_ANALYSIS = 300           # pooled frames needed to analyse a (cell, behavior) pair in a window
FRAME_SHIFT_MIN_TRIALS = 10         # smallest circular shift of the null, in units of the median frames per trial window

# --------------------------------------------------------------------- #
# 7. Face-quantity normalization. pupil area / x / y, motion energy and the video PCs are z-scored WITHIN each session
#    (continuous.zscore_per_session, after blink rejection) wherever the behavior tables are built (neural_encoding:
#    build_session_tables / build_window_frames), as in 1c / 1d / 1f. Running speed and licks are NOT normalized (physical
#    units). Per-session quantile-binned measures do not depend on it; it matters for anything that pools raw values
#    across sessions.
# --------------------------------------------------------------------- #
NORMALIZE_FACE_PER_SESSION = True

# --------------------------------------------------------------------- #
# 8. DEPTH CORRECTIONS for sessions whose recorded depth carries a known constant offset (um). Convention:
#       corrected depth = recorded depth - offset      (the recorded values are too DEEP by `offset`;
#                                                       use a negative offset when they are too shallow)
#    Applied in 2a (celldata.depth, before the depth_suspect check and the depth classes, so celldata_combined.csv holds the
#    corrected value + `depth_recorded` + `depth_offset_um`) and by anything that reads a session's raw celldata through
#    infotheory.celldata_utils.apply_depth_correction (2b does). Key = session_id.
# --------------------------------------------------------------------- #
DEPTH_OFFSET_CORRECTIONS_UM = {
    "LPE12013_2024_04_30": 325.0,
}

