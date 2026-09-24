# Global_modes

Do network-wide perturbations (arousal/neuromodulatory-like signals
applied uniformly to every neuron) corrupt stimulus decoding in trained
recurrent networks, and under what conditions does the network avoid
that -- for free, from geometry alone, or only by actively learning to
route around the perturbation? A companion project to
`Perturbation-vs-Stimulus-Coding`, rebuilt with **Dale's law as a
first-class, mandatory ingredient** (not an optional flag) and organized
with the de Kok folder convention (Ties de Kok, "How to keep your
research projects organized, part 1: folder structure",
https://medium.com/data-science/how-to-keep-your-research-projects-organized-part-1-folder-structure-10bd56034d3a).

## Environment setup

```bash
python3 -m venv venv
source venv/bin/activate        # or venv\Scripts\activate on Windows
pip install --upgrade pip
pip install -r requirements.txt
```

Verify:
```bash
python -c "import numpy, scipy, matplotlib, pandas, torch; print('OK')"
```

## Folder structure (de Kok layout)

```
0_data/       empty -- everything in this project is simulated; kept for
              convention / in case real data ever gets added
1_scripts/    ALL main scripts live flat here, numbered in execution
              order (see "Scripts" below) -- NOT one subfolder per script.
              Subfolders are reserved for shared code and tests:
                utils/   shared library (models, tasks, perturbation,
                         training, caching, geometry/dimensionality)
                tests/   pytest suite, pure/fast, no network training
2_pipeline/   one sub-folder per script in 1_scripts/, name-matched:
                <script_name>/out/    end-products a LATER script may load
                <script_name>/store/  this SAME script's own ResultCache
                <script_name>/tmp/    throwaway inspection files
3_output/     hand-picked, presentation-ready figures/tables only --
              not everything a script produces. Flat, prefixed by step,
              e.g. `02_accuracy_curves_dale.png`.
a_docs/       project proposal + one progress file per script
              (Goals / Findings / Open questions)
b_progress/   markdown -> slide-deck pipeline for sharing with coworkers
c_templates/  copy-paste starting points for new scripts
d_archive/    old code
```

**Folder rules** (from the source article, enforced throughout this
project):
1. A script only loads data from `0_data/` or from an `out/` folder
   belonging to a script that runs **before** it. It never reads a later
   script's output, and never reaches into another script's own
   `store/`/`tmp/` -- those are that script's private scratch space. (In
   practice this means e.g. script 03 keeps its own cache rather than
   reusing script 02's, even for network configurations the two happen
   to share -- a small amount of duplicated compute in exchange for the
   rule never having an exception to remember.)
2. Never write into `0_data/`.
3. `utils/pipeline.py`'s `get_pipeline_paths(__file__)` resolves a
   script's own `2_pipeline/<name>/{out,store,tmp}` automatically --
   call it as the first thing in `main()`, rather than hand-maintaining
   paths.

## Scripts (`1_scripts/`, run in this order)

| # | Script | What it does |
|---|--------|---------------|
| 01 | `01_train_coherence_rnn_nodale.py` | **Control.** No Dale's law. Trains on single-stream coherence detection, global uniform perturbation, 2x2 grid (tanh/relu x perturbation-normalized-by-size or not) x clean/perturbed training x seeds. |
| 02 | `02_train_coherence_rnn.py` | **Core.** Same task/grid, Dale's law ON, 80% excitatory / 20% inhibitory (Song, Yang & Wang 2016-style sign-constrained recurrent weights). |
| 03 | `03_ei_ratio_sweep.py` | Does clean-training orthogonalization depend on the exc/inh ratio, and does that depend on network size? Heatmap over `N x exc_fraction`, plus a focused line plot at the project's default N. |
| 04 | `04_structured_perturbation_directions.py` | Low-rank-mixture and power-law-spectrum perturbation *directions* (motivated by Rabinowitz et al. 2015, Okun et al. 2015, Stringer et al. 2019 -- see `a_docs/4_structured_perturbation_directions.md`), applied to Dale's-law networks. |
| 05 | `05_train_low_rank_rnn.py` | Mastrogiuseppe & Ostojic (2018)-style low-rank connectivity, hybrid `chi*W_dale_bg + (1/N)M N^T`, rank R swept at fixed chi. |
| 06 | `06_orthogonalization_analysis.py` | Full geometry/dynamics toolkit (angle + exact p-value, gain ratio, participation ratio, fixed point/Jacobian eigenspectrum, pulse-decay) across curated conditions from 01-05, plus a low-rank-connectivity-level orthogonalization check (05's m^(r)/n^(r) vectors vs. the perturbation direction). |
| 07 | `07_dimensionality_reduction.py` | PCA-based activity dimensionality (Mastrogiuseppe & Ostojic 2018-style participation ratio; Cunningham & Yu 2014-style cross-validated PCA): no-Dale vs. Dale, dimensionality vs. rank R (05), and a direct test of whether higher relative dimensionality predicts easier orthogonalization (03's EI x size sweep). |
| 08 | `08_aligned_oblique_dynamics.py` | Schuessler, Mastrogiuseppe, Ostojic & Barak (2024, eLife)-style aligned/oblique dynamics: does readout-weight-magnitude-controlled alignment (their axis) interact with this project's own perturbation-orthogonalization axis? Supports `--task coherence\|context` (see 09-11) to test whether their dimensionality-based dissociation needs a genuinely multi-variable task to show up. |
| 09 | `09_train_context_rnn_nodale.py` | No-Dale control, Mante et al. (2013)-style context-dependent task (motion/color/context/nuisance inputs, context-GATED choice) -- motivated directly by script 07's finding that the coherence task is solved via near-1D dynamics, leaving dimensionality-based analyses with little room to show anything. |
| 10 | `10_train_context_rnn.py` | Dale core, same context task. Feeds script 11. |
| 11 | `11_context_connectivity_selectivity_analysis.py` | Dubreuil, Valente, Beiran, Mastrogiuseppe & Ostojic (2022, Nat. Neurosci.)-style connectivity-space/selectivity-space analysis: SVD-based effective low-rank extraction from trained full-rank W_rec, per-neuron Mante-style multi-variable regression (motion/color/context selectivity), and whether the two spaces align -- plus where the project's own global perturbation direction sits in both. |
| 12+ | *(planned)* | EI-ratio sweep and structured-perturbation-direction analyses mirrored onto the context task (03/04's counterparts). |

Every training script's CLI:
```bash
cd 1_scripts && python 01_train_coherence_rnn_nodale.py
cd 1_scripts && python 01_train_coherence_rnn_nodale.py --n_seeds 3 --n_iters 500   # faster, coarser
N_WORKERS=4 python 02_train_coherence_rnn.py                                        # limit parallel workers
python 03_ei_ratio_sweep.py --force_recompute                                       # bypass the cache
```

**Tests** (fast, no network training):
```bash
cd 1_scripts && pytest tests/
```

## Result caching (`utils/caching.py`)

Every trained network is a `ResultCache.get_or_compute` call keyed by an
exact config dict (seed, N, activation, dale on/off, exc_fraction,
perturbation-normalization convention, clean/perturbed training,
n_iters, ...). Config -> SHA-256 hash (first 16 hex chars) -> a folder
`<script>/store/<project_name>/<hash>/` containing `config.json`,
`summary.json`, `result.pkl` (trained weights + accuracy curve +
perturbation direction used), and `provenance.json`. Rerunning a script
with the same parameters loads cached results instead of retraining --
tweaking a plot and rerunning completes in seconds. Floats are rounded
to 9 decimals before hashing so floating-point sweep noise (e.g.
`np.linspace` accumulation) can't cause spurious cache misses.

## Design notes worth knowing before extending this project

- **Perturbation convention**: `normalize_by_size=False` draws a
  *unit-norm* random direction in R^N regardless of N (per-neuron
  amplitude shrinks as N grows); `normalize_by_size=True` scales the
  direction to norm sqrt(N) (per-neuron amplitude stays ~constant as N
  grows). Both are run side by side since it's genuinely unclear which
  is the more literal model of a network-wide neuromodulatory signal.
  See `utils/perturbation.py`.
- **Dale's law**: mandatory throughout the core scripts (02+), 80/20
  exc/inh split by default (`exc_fraction=0.8`), implemented via
  `softplus(W_raw) * sign_vec` (Song, Yang & Wang 2016), rescaled to a
  target spectral radius of 1.2 at init. Script 01's no-Dale networks are
  rescaled to the *same* target spectral radius so the two conditions
  start from comparable dynamical regimes.
- **Activation**: `tanh` or `relu`, a single flag on one `RateRNN` class
  (`utils/models.py`) -- never two model classes -- so activation
  comparisons share every other line of code.
- **Orthogonalization angle**: `utils/geometry.py`'s
  `analyze_orthogonalization` regresses time-averaged hidden activity
  against `[stimulus coherence, perturbation strength, intercept]` in a
  single batched forward pass (each trial gets its own randomly sampled
  perturbation strength) to get Mante-style per-neuron loading vectors
  `beta_stim`/`beta_perturb`, then reports the angle between them and an
  *exact* analytic p-value against the random-direction null (Beta
  distribution of a random unit vector's angle to a fixed axis, verified
  against a 100k-draw Monte Carlo in `tests/test_geometry.py`).
- **Orthogonality visualization convention**: every figure that plots an
  orthogonalization angle uses `sin_alignment(angle_deg)` -- bounded
  [0, 1], with **1.0 = perfectly orthogonal, 0.0 = fully aligned** --
  not the raw angle or an `|angle - 90|` degrees deviation. Ties
  directly to the same geometry the exact p-value formulas use
  (`sin^2(angle) = 1 - cos^2(angle)`, and `cos^2(angle)` is exactly the
  Beta-distributed quantity those formulas compute a p-value from).
  Applies to scripts 03, 04, 06, 07, and 11 (the last also sin-transforms
  its principal-angle bars, where 0 means "same subspace" rather than
  "aligned to a fixed line" -- read the axis label on each figure, since
  the 0/1 endpoints mean slightly different things there).
- **Parallelism**: every training script parallelizes independent seeds
  across CPU cores via `concurrent.futures.ProcessPoolExecutor`
  (`utils/experiment_runner.py`), defaulting to `os.cpu_count()` workers,
  overridable with `N_WORKERS=<n>`.

## Reproducibility

All scripts use fixed seeds by default. Trained weights are cached in
full (`model_state.pt`-equivalent, embedded in `result.pkl`), so any
downstream analysis script can reconstruct the exact trained model
without retraining (`utils/experiment_runner.rebuild_model`).
