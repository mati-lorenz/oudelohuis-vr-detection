# -*- coding: utf-8 -*-
"""
params.py
=========
Shared configuration objects for the information-theoretic analysis pipeline.

The pipeline is organized in three stages, of which this module ships the
common backbone so that stage 2 (dimensionality reduction / signal & noise
correlations) and stage 3 (population-level information in reduced spaces)
can reuse the exact same estimators, binning rules and parallelization
settings as stage 1 (single-cell / cell-pair information).

Matthijs' lab style is followed (see loaddata/*.py): plain dataclasses,
no external config framework required.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Literal


@dataclass
class BinningParams:
    """How continuous neural responses are discretized before any
    information-theoretic quantity is computed."""

    method: Literal['equipopulated', 'equal_width'] = 'equipopulated'
    n_bins: int = 3
    #: If set, n_bins is chosen automatically per neuron as
    #: max(2, floor(min_trials_per_stim / trials_per_bin_target))
    auto_nbins: bool = False
    trials_per_bin_target: int = 8
    #: bins are (re)computed pooling across all trials ('pooled') or
    #: separately per stimulus/choice class then merged ('conditional').
    #: 'pooled' is the standard choice and avoids information leakage.
    bin_reference: Literal['pooled'] = 'pooled'


@dataclass
class BiasCorrectionParams:
    """Settings for the Panzeri-Treves style analytic bias correction and
    the permutation-based (shuffle) null hypothesis test."""

    #: apply the analytic Treves-Panzeri / Miller-Madow style 1/N correction
    panzeri_treves: bool = True
    #: also compute a bootstrap/shuffle-based bias estimate (mean of the
    #: null distribution) and report it alongside the analytic correction
    shuffle_bias_estimate: bool = True
    #: number of shuffles used both for the null-distribution significance
    #: test and (if requested) for the shuffle-based bias estimate
    n_shuffles: int = 500
    #: two-sided or one-sided (information is bounded >=0, so 'greater' is
    #: the natural choice) permutation test
    alternative: Literal['greater', 'two-sided'] = 'greater'
    #: seed for reproducibility (each parallel worker derives its own
    #: sub-seed from this + the unit index, see core.get_worker_rng)
    random_state: Optional[int] = 0

    #: Shuffle-subtraction bias correction for the Pola et al. (2003)
    #: pairwise information breakdown (breakdown.py). This is a SEPARATE,
    #: much cheaper setting than n_shuffles above: it is essential for
    #: trusting the sign/magnitude of the small correlational terms
    #: (I_sig_sim, I_cor_indep, I_cor_dep) and of the redundancy-synergy
    #: index at realistic trial counts -- naive plug-in bias alone can
    #: flip their sign (see breakdown.py's module docstring and
    #: tests/test_breakdown_canonical.py for a worked demonstration
    #: reproducing Lorenz et al. 2025's own validation figures). Matches
    #: the "shuffle-subtraction" method MINT itself uses for exactly
    #: this decomposition (their SM6.1 uses 30 shuffles).
    breakdown_shuffle_correction: bool = True
    breakdown_n_shuffles: int = 30

    #: Same idea, for the partial information decomposition (pid.py):
    #: naive PID terms (redundancy/unique/synergy) carry the same kind
    #: of limited-sampling bias -- see tests/test_pid_canonical.py for a
    #: demonstration that a genuinely independent (zero-information)
    #: case shows substantial spurious synergy/unique information at
    #: realistic trial counts without this correction.
    pid_shuffle_correction: bool = True
    pid_n_shuffles: int = 30


@dataclass
class ParallelParams:
    """Parallelization settings, shared by all stages."""

    n_jobs: int = -1            # joblib convention: -1 = all cores
    backend: Literal['loky', 'threading', 'multiprocessing'] = 'loky'
    verbose: int = 5


@dataclass
class InfoTheoryParams:
    """Top-level parameter bundle passed around the pipeline."""

    binning: BinningParams = field(default_factory=BinningParams)
    bias: BiasCorrectionParams = field(default_factory=BiasCorrectionParams)
    parallel: ParallelParams = field(default_factory=ParallelParams)

    #: trial-selection: which trialdata column(s) define "stimulus" and
    #: "choice" for this protocol. Adapt per protocol, e.g.:
    #:   DN/DM/DP (detection tasks): stim_var='stimcat' (S vs N), choice_var='lickResponse'
    #:   GR/GN: stim_var='Orientation', choice_var=None
    stim_var: str = 'stimcat'
    choice_var: Optional[str] = 'lickResponse'
    #: optional column used to condition/split trials, e.g. only engaged
    #: trials, or only correct trials, before computing information
    trial_mask_var: Optional[str] = None

    #: Collapse/binarize the raw values of `stim_var` before anything
    #: else uses them. At most one of the two should be set.
    #:
    #: stim_value_map: exact value -> new value mapping, for CATEGORICAL
    #:   stim_var. E.g. for the detection task's 'stimcat' column, which
    #:   has three categories ('C' = catch/no-signal trial, 'N' and 'M'
    #:   = two non-zero signal strengths), use
    #:   {'C': 0, 'N': 1, 'M': 1} to get a binary "signal present or
    #:   not" stimulus variable pooling N and M together. Any stim value
    #:   not present as a key is left unchanged, so partial maps are
    #:   fine.
    #: stim_binarize_threshold: for a CONTINUOUS stim_var (e.g. the raw
    #:   'signal' strength column instead of the categorical 'stimcat'),
    #:   threshold it into a binary variable as
    #:   (stim_var > stim_binarize_threshold). E.g.
    #:   stim_var='signal', stim_binarize_threshold=0 reproduces the
    #:   same "signal present (any strength) vs not" split as the
    #:   stim_value_map example above, but starting from the continuous
    #:   variable -- convenient once you're ready to move beyond a
    #:   binary split (e.g. to more than 2 signal-strength bins) since
    #:   you only need to change the binning, not the trialdata column.
    stim_value_map: Optional[dict] = None
    stim_binarize_threshold: Optional[float] = None

    #: minimum number of trials per stimulus/choice class required to
    #: attempt an estimate for a given cell (protects against very biased
    #: MI estimates in poorly sampled conditions)
    min_trials_per_class: int = 5
