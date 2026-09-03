# infotheory — Stage 1: single-cell & cell-pair information

Information-theoretic analysis of calcium imaging data recorded during the
virtual-maze stimulus/choice detection task. Built to be the numerical
backbone for all three planned analysis stages:

1. **Stage 1 (this code)** — single-cell information about stimulus and
   choice (mutual information, Panzeri-Treves bias correction, shuffle
   significance testing), the Pola et al. (2003) pairwise information
   breakdown, and partial information decomposition (PID) of
   stimulus/choice for single cells.
2. **Stage 2 (planned)** — dimensionality reduction (PCA, dPCA, LDA, ...)
   on full and trial-averaged data; signal & noise correlations.
3. **Stage 3 (planned)** — information-theoretic analysis of the
   population code in reduced dimensions, reusing `core.py`,
   `breakdown.py` and `pid.py` unchanged.

## Layout

```
analysis/infotheory/
    params.py        Config dataclasses (BinningParams, BiasCorrectionParams,
                      ParallelParams, InfoTheoryParams) shared by all stages
    discretize.py     Equipopulated / equal-width binning of continuous
                      single-trial responses
    core.py           Naive MI, Panzeri-Treves analytic bias correction,
                      shuffle null distribution + significance test
    breakdown.py       Pola et al. (2003) 4-term pairwise information
                      breakdown (I_lin, I_sig-sim, I_cor-indep, I_cor-dep)
    pid.py             Partial information decomposition (Williams-Beer
                      I_min): redundancy / unique / synergy of stimulus
                      and choice about a single cell's response
    single_cell.py      Parallelized (joblib) driver: per-neuron and
                      per-pair analysis, plus a Session-object wrapper
analysis/run_singlecell_infotheory.py
                      Example/CLI script: loads sessions via loaddata,
                      runs stage 1, saves per-session and combined CSVs
```

## Quick start

```python
from loaddata.session_info import filter_sessions
from analysis.infotheory import InfoTheoryParams, run_session_single_cell_analysis

sessions, n = filter_sessions(protocols=['DN'], min_trials=100)
ses = sessions[0]
ses.load_respmat(load_behaviordata=True, load_calciumdata=True, calciumversion='deconv')

params = InfoTheoryParams(stim_var='stimcat', choice_var='lickResponse')
results = run_session_single_cell_analysis(ses, params=params, do_pairwise=True)

results['single_cell']   # one row per neuron: MI(stim), MI(choice), PID, p-values
results['pairwise']      # one row per cell pair: 4-term Pola breakdown
```

Or just run `analysis/run_singlecell_infotheory.py` directly (edit the
`protocol` / `params` block at the top first).

## Design choices worth knowing about

- **Binning**: equipopulated (quantile) binning per neuron, `n_bins=3` by
  default. Ties (e.g. many exact zeros in deconvolved activity) reduce the
  effective bin count automatically and safely (see `discretize.py`).
- **Bias correction**: the analytic correction implemented in `core.py`
  follows Treves & Panzeri (1995) / Panzeri & Treves (1996) using the
  *naive* count of non-empty response bins per stimulus class. This is the
  standard practical form of "the PT correction" used across the
  literature. If you need to reproduce a specific published number
  exactly (e.g. from Lorenz et al. 2025), check whether that paper's
  Methods use the fully Bayesian refinement of the relevant-bin count —
  swapping it in only requires editing `_relevant_bins` /
  `panzeri_treves_correction`.
- **Significance**: a permutation/shuffle null distribution
  (`n_shuffles`, default 500) is always computed alongside the analytic
  correction, both for a p-value and as an independent bias sanity check
  (`I_shuffle_corrected`, `null_mean` should be ~0 after PT correction).
- **PID**: uses the Williams & Beer (2010) `I_min` redundancy measure —
  simple, exact, non-negative, no optimization required. `pid_decomposition`
  accepts any `redundancy_fn(r, s1, s2) -> float`, so a different
  redundancy measure (e.g. I_ccs / BROJA) can be swapped in later without
  touching the rest of the pipeline.
- **Pairwise breakdown**: the four Pola et al. (2003) terms are computed
  as telescoping differences of directly-estimated MI on the real data
  and two surrogate datasets (trial-shuffled and "stimulus-independent
  correlation" surrogates — see the module docstring in `breakdown.py`
  for exact construction). This guarantees `I_lin + I_sig_sim +
  I_cor_indep + I_cor_dep == I_full` exactly for any binning, which is a
  good sanity check to run on your own data.
- **Parallelization**: every neuron (stage 1) and every pair (breakdown)
  is independent, so both are dispatched via `joblib.Parallel`
  (`params.parallel.n_jobs`, default: all cores). Random seeding is
  per-unit/per-pair deterministic (`core.get_worker_rng`) so results do
  not depend on the number of workers.

## Dependencies

`numpy`, `pandas`, `joblib` (in addition to what `loaddata` already needs).
