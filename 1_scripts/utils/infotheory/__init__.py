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
from . import spike_stats
from . import qc_lib
from . import neural_encoding
from . import shuffling
from . import cache
from .cache import cache_path, load_or_compute, load_all_matching
from . import pid
from . import pair_information
from .pair_information import compute_pair_breakdown, digitize_quantile, joint_code
from . import population
from .population import (
    zscore_population, build_standard_condition_specs, run_across_shuffle_conditions,
    plot_shuffle_condition_comparison, plot_metric_vs_population_size, build_population_group,
    iterate_population_groups,
)
from .shuffling import (
    make_shuffle_groups, shuffle_within_groups, conditioned_shuffle_test,
    ShuffleTestResult, SHUFFLE_MODES,
    make_shuffle_groups_from_conditions, shuffle_population_within_groups,
)
from .info_theory import (
    mutual_information_hist,
    quantile_bin_edges,
    robust_qcut,
    mutual_information_shuffle,
    conditional_mutual_information_hist,
    transfer_entropy_hist,
    transfer_entropy_shuffle,
    granger_causality_test,
    mutual_information_ksg,
    mutual_information_shuffle_ksg,
    mi_from_r2,
    compare_mi_to_linear,
    MIvsLinear,
    MIResult,
    linear_fit,
    LinearFit,
    multivariate_linear_fit,
    MultivariateLinearFit,
    forward_stepwise_selection,
    StepwiseStep,
)
from .continuous import (
    merge_behavior_video,
    build_calcium_continuous,
    check_zpos_consistency,
    restrict_to_engaged,
    remove_pupil_outliers,
    compute_trial_window_means,
    compute_trial_position_window_means,
    bin_by_position,
)
from .psth import (
    align_trials_time, align_trials_position, derive_onset_time,
    compute_position_binned_information, compute_position_bin_indices, mi_from_position_bins,
    te_from_position_bins, build_position_binned_trial_tensor,
)
from .celldata_utils import (
    get_area_label as get_cell_area_label,
    ordered_groups as cell_ordered_groups,
    bar_by_group as cell_bar_by_group,
    filter_nearlabeled,
    nearest_labeled_distance,
    filter_nearlabeled_layer23,
    filter_target_groups,
    sample_pairs_within_groups,
    AREA_COLORS as CELL_AREA_COLORS,
    LABEL_LINESTYLES as CELL_LABEL_LINESTYLES,
    LABEL_MARKERS as CELL_LABEL_MARKERS,
    DEFAULT_AREA_ORDER as CELL_DEFAULT_AREA_ORDER,
    DEFAULT_LABEL_ORDER as CELL_DEFAULT_LABEL_ORDER,
    LABEL_SPLIT_AREAS,
    SIX_GROUP_ORDER,
    SIX_GROUP_LABELS,
    SIX_GROUP_COLORS,
)

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
    "quantile_bin_edges",
    "robust_qcut",
    "mutual_information_shuffle",
    "conditional_mutual_information_hist",
    "transfer_entropy_hist",
    "transfer_entropy_shuffle",
    "granger_causality_test",
    "make_shuffle_groups",
    "shuffle_within_groups",
    "conditioned_shuffle_test",
    "ShuffleTestResult",
    "SHUFFLE_MODES",
    "make_shuffle_groups_from_conditions",
    "shuffle_population_within_groups",
    "cache_path",
    "load_or_compute",
    "load_all_matching",
    "zscore_population",
    "build_standard_condition_specs",
    "run_across_shuffle_conditions",
    "plot_shuffle_condition_comparison",
    "plot_metric_vs_population_size",
    "build_population_group",
    "compute_pair_breakdown",
    "digitize_quantile",
    "joint_code",
    "iterate_population_groups",
    "mutual_information_ksg",
    "mutual_information_shuffle_ksg",
    "mi_from_r2",
    "compare_mi_to_linear",
    "MIvsLinear",
    "MIResult",
    "linear_fit",
    "LinearFit",
    "multivariate_linear_fit",
    "MultivariateLinearFit",
    "forward_stepwise_selection",
    "StepwiseStep",
    "merge_behavior_video",
    "build_calcium_continuous",
    "check_zpos_consistency",
    "restrict_to_engaged",
    "remove_pupil_outliers",
    "compute_trial_window_means",
    "compute_trial_position_window_means",
    "bin_by_position",
    "align_trials_time",
    "align_trials_position",
    "derive_onset_time",
    "compute_position_binned_information",
    "compute_position_bin_indices",
    "mi_from_position_bins",
    "te_from_position_bins",
    "build_position_binned_trial_tensor",
    "get_cell_area_label",
    "cell_ordered_groups",
    "cell_bar_by_group",
    "filter_nearlabeled",
    "nearest_labeled_distance",
    "filter_nearlabeled_layer23",
    "filter_target_groups",
    "sample_pairs_within_groups",
    "CELL_AREA_COLORS",
    "CELL_LABEL_LINESTYLES",
    "CELL_LABEL_MARKERS",
    "CELL_DEFAULT_AREA_ORDER",
    "CELL_DEFAULT_LABEL_ORDER",
    "LABEL_SPLIT_AREAS",
    "SIX_GROUP_ORDER",
    "SIX_GROUP_LABELS",
    "SIX_GROUP_COLORS",
]
