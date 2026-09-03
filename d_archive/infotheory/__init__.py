# -*- coding: utf-8 -*-
"""
infotheory
===========
Information-theoretic analysis pipeline for calcium imaging data recorded
during the virtual-maze stimulus/choice detection task (see loaddata/).

Stage 1 (this package, ready to use):
    single-cell and cell-pair information about stimulus and choice
    - core.py        : MI estimation, Panzeri-Treves bias correction,
                        permutation null testing
    - discretize.py   : response binning
    - breakdown.py     : Pola et al. (2003) pairwise information breakdown
    - pid.py           : partial information decomposition (stim/choice)
    - single_cell.py    : parallelized driver tying the above together
    - params.py         : shared configuration dataclasses

Stage 2 (planned, not yet implemented):
    dimensionality reduction (PCA, dPCA, LDA, ...) on full & trial-
    averaged data; signal & noise correlation structure.

Stage 3 (planned, not yet implemented):
    information-theoretic analysis (reusing core.py/breakdown.py/pid.py)
    of the reduced-dimensionality population code.
"""

from .params import InfoTheoryParams, BinningParams, BiasCorrectionParams, ParallelParams
from .core import (
    mutual_information,
    mi_with_stats,
    shuffle_null_distribution,
    significance_test,
    naive_mutual_information,
    naive_entropy,
)
from .discretize import discretize_response, discretize_dataframe
from .breakdown import pairwise_information_breakdown
from .pid import pid_decomposition, redundancy_imin
from .single_cell import (
    compute_single_cell_information,
    compute_pairwise_breakdown,
    compute_pairwise_pid,
    run_session_single_cell_analysis,
    get_trial_labels,
)
from .gain_model import (
    estimate_tuning_curves_cv, estimate_gain_unconstrained,
    fit_poisson_gain_glm, predicted_noise_correlation,
    estimate_neuron_coupling_strength, predicted_noise_correlation_coupling)
from .session_utils import compute_respmat_for_session, TIME_LOCKED_PROTOCOLS, SPATIAL_PROTOCOLS
from .tensor_utils import compute_tensor_for_session
from .temporal import compute_time_resolved_information
from .temporal_breakdown import compute_time_resolved_breakdown, BREAKDOWN_TERMS
from .temporal_pid import compute_time_resolved_pid, PID_TERMS
from .temporal_pairwise_pid import compute_time_resolved_pairwise_pid
from .celldata_utils import (
    get_area_label, ordered_groups, sample_pairs_within_groups,
    AREA_COLORS, LABEL_LINESTYLES)
from .cache import cache_path, load_cache, save_cache, load_or_compute
from .spike_stats import (
    get_frame_rate, event_rate, active_frame_rate, sparsity, iei_stats, fano_factor, autocorrelogram,
    signal_noise_correlations, population_coupling, split_half_reliability,
    runspeed_correlation)
from .behavior_signals import get_behavior_trace
from .behavior_mi import compute_behavior_mi

__all__ = [
    'InfoTheoryParams', 'BinningParams', 'BiasCorrectionParams', 'ParallelParams',
    'mutual_information', 'mi_with_stats', 'shuffle_null_distribution',
    'significance_test', 'naive_mutual_information', 'naive_entropy',
    'discretize_response', 'discretize_dataframe',
    'pairwise_information_breakdown',
    'pid_decomposition', 'redundancy_imin',
    'compute_single_cell_information', 'compute_pairwise_breakdown',
    'compute_pairwise_pid',
    'run_session_single_cell_analysis', 'get_trial_labels',
    'compute_respmat_for_session', 'TIME_LOCKED_PROTOCOLS', 'SPATIAL_PROTOCOLS',
    'compute_tensor_for_session', 'compute_time_resolved_information',
    'compute_time_resolved_breakdown', 'BREAKDOWN_TERMS',
    'compute_time_resolved_pid', 'PID_TERMS',
    'compute_time_resolved_pairwise_pid',
    'get_area_label', 'ordered_groups', 'sample_pairs_within_groups',
    'AREA_COLORS', 'LABEL_LINESTYLES',
    'cache_path', 'load_cache', 'save_cache', 'load_or_compute',
    'get_frame_rate', 'event_rate', 'active_frame_rate', 'sparsity', 'iei_stats', 'fano_factor',
    'autocorrelogram', 'signal_noise_correlations', 'population_coupling',
    'split_half_reliability', 'runspeed_correlation',
    'get_behavior_trace', 'compute_behavior_mi',
    'estimate_tuning_curves_cv', 'estimate_gain_unconstrained',
    'fit_poisson_gain_glm', 'predicted_noise_correlation',
    'estimate_neuron_coupling_strength', 'predicted_noise_correlation_coupling',
]
