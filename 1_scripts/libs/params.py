# -*- coding: utf-8 -*-
"""
params.py
=========
Shared configuration objects for the whole pipeline. Plain dataclasses --
no external config framework. This module is mostly declarative, so it's
sketched more completely than the others; adjust fields/defaults freely as
your actual needs surface.

TODO(you):
- Confirm these fields cover what your estimators actually need.
- Add protocol-specific stim_var/choice_var mappings as you add protocols.
- Consider whether you want a `target_binning` override here (per-variable
  binning for continuous behavioral targets) vs. passing overrides as
  explicit function arguments (see design_notes.md).
"""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class BinningParams:
    """How continuous neural responses are discretized before any
    information-theoretic quantity is computed."""
    method: Literal["equipopulated", "equal_width"] = "equipopulated"
    n_bins: int = 3
    auto_nbins: bool = False
    trials_per_bin_target: int = 8


@dataclass
class BiasCorrectionParams:
    """Panzeri-Treves analytic correction + shuffle-based null settings."""
    panzeri_treves: bool = True
    shuffle_bias_estimate: bool = True
    n_shuffles: int = 500
    alternative: Literal["greater", "two-sided"] = "greater"
    random_state: Optional[int] = 0

    # separate, cheaper shuffle-subtraction settings for the pairwise
    # breakdown / PID (see docs/null_hypotheses_framework.md)
    breakdown_shuffle_correction: bool = True
    breakdown_n_shuffles: int = 30
    pid_shuffle_correction: bool = True
    pid_n_shuffles: int = 30


@dataclass
class ParallelParams:
    n_jobs: int = -1
    backend: Literal["loky", "threading", "multiprocessing"] = "loky"
    verbose: int = 5


@dataclass
class InfoTheoryParams:
    """Top-level parameter bundle passed around the pipeline."""
    binning: BinningParams = field(default_factory=BinningParams)
    bias: BiasCorrectionParams = field(default_factory=BiasCorrectionParams)
    parallel: ParallelParams = field(default_factory=ParallelParams)

    stim_var: str = "stimcat"
    choice_var: Optional[str] = "lickResponse"
    trial_mask_var: Optional[str] = None
    stim_value_map: Optional[dict] = None
    stim_binarize_threshold: Optional[float] = None
    min_trials_per_class: int = 5
