# -*- coding: utf-8 -*-
"""
infotheory
==========
Shared analysis library for the multi_area_detection_task_ff_fb project.
See the project README (repo root) for the folder-structure conventions
this package plugs into.

Sub-modules (import what you need, e.g. `from infotheory.behavior import
compute_dprime`):

    pipeline    -- get_pipeline_paths(__file__): resolves each script's own
                   2_pipeline/<name>/{out,store,tmp} folders
    session     -- Session class + discover_sessions/load_sessions:
                   generic loader for 0_data/<protocol>/<animal>/<date>/
    behavior    -- signal-detection stats (d', criterion), engagement,
                   session-level summary tables
    plotting    -- small shared plotting helpers (colors, style, save_fig)

Binning, mutual information, dimensionality reduction (GPFA/jPCA/TDR)
live in their own sub-modules and are added as those pipeline steps are
built out (see a_docs/ progress notes).
"""

from .pipeline import get_pipeline_paths, find_project_root
from .session import Session, discover_sessions, load_sessions
from .behavior import (
    compute_dprime,
    add_trial_outcome,
    engagement_summary,
    rolling_performance,
    summarize_session,
    build_session_summary_table,
)
from .reporting import build_progress_markdown
from .psychometric import psychometric_function, fit_psychometric, PsychometricFit
from . import criteria
from .info_theory import (
    mutual_information_hist,
    mutual_information_shuffle,
    mutual_information_ksg,
    mutual_information_shuffle_ksg,
    mi_from_r2,
    compare_mi_to_linear,
    MIvsLinear,
    MIResult,
    linear_fit,
    LinearFit,
)
from .continuous import (
    merge_behavior_video,
    check_zpos_consistency,
    restrict_to_engaged,
    remove_pupil_outliers,
    compute_trial_window_means,
    bin_by_position,
)
from .psth import align_trials_time, align_trials_position, derive_onset_time

__all__ = [
    "get_pipeline_paths",
    "find_project_root",
    "Session",
    "discover_sessions",
    "load_sessions",
    "compute_dprime",
    "add_trial_outcome",
    "engagement_summary",
    "rolling_performance",
    "summarize_session",
    "build_session_summary_table",
    "build_progress_markdown",
    "psychometric_function",
    "fit_psychometric",
    "PsychometricFit",
    "criteria",
    "mutual_information_hist",
    "mutual_information_shuffle",
    "mutual_information_ksg",
    "mutual_information_shuffle_ksg",
    "mi_from_r2",
    "compare_mi_to_linear",
    "MIvsLinear",
    "MIResult",
    "linear_fit",
    "LinearFit",
    "merge_behavior_video",
    "check_zpos_consistency",
    "restrict_to_engaged",
    "remove_pupil_outliers",
    "compute_trial_window_means",
    "bin_by_position",
    "align_trials_time",
    "align_trials_position",
    "derive_onset_time",
]
