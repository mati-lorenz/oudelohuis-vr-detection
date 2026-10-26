# Data-cleaning filters used in the pipeline

This document lists every filter that decides which sessions, trials, samples, and cells enter an analysis: what it tests, why, its cutoff, and where the value lives (so you can change it). Values are the current defaults in the code.

**Order in which the filters act**

```
raw session ──► [A] session level (1a flags → 1b exclusion)  ──► included sessions
                [B] trial level   (engaged trials; ≥ trials per analysis)
                [C] sample level  (pupil blinks; position window; frame coverage)
cells ──► [D] cell level (2b QC  AND  2a proximity)  ──► included cells
                [E] statistic-level minimums (trials per bin/class, units per group)
```

2b statistics, 2a surveys, and the 1a/1b overview deliberately use *unfiltered* populations (see "What is NOT filtered"). The filters below are the ones applied from 1b onward.

---

## A. Session level

### A1. 1a flags (informational) and 1b exclusion (hard)
Shared in `utils/infotheory/criteria.py`, so 1a's flags and 1b's exclusions cannot drift apart. 1a only flags sessions "worth a second look"; 1b is where a session is actually excluded. Every later step that reads `1b_psychometric/out/psychometric_fits.csv` uses only sessions with `included == True`. 1a/1b computed per session, so selecting sessions never changes another session's verdict.

| Test                                        | Definition                                                                                                                                                                                                                              | Cutoff                                            | Why                                                                                                                                                         | Where                                |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------ |
| Minimum trials                              | number of trials in the session                                                                                                                                                                                                         | **< 50 → flagged in 1a** (not excluded)           | Too few trials give unstable d′, psychometric fits and information estimates.                                                                               | `MIN_TRIALS`                         |
| Engaged fraction                            | fraction of trials labelled `engaged`                                                                                                                                                                                                   | **< 0.30 → flagged in 1a, excluded in 1b**        | A session where the animal mostly isn't performing the task has too little usable data and is dominated by disengaged behavior.                             | `MIN_FRAC_ENGAGED`                   |
| Sensitivity (d′)                            | d′ = z(hit rate) − z(false-alarm rate), computed on **engaged trials**, with rates clipped to [1/(2N), 1 − 1/(2N)] so d′ stays finite                                                                                                   | **d′ < 1.0 → excluded**                           | If the animal cannot discriminate signal from no-signal trials, "detection" analyses are meaningless.                                                       | `MIN_DPRIME`                         |
| False-alarm rate                            | FA / (FA + CR) on engaged trials                                                                                                                                                                                                        | **> 0.50 → excluded**                             | A high FA rate means the animal licks regardless of the stimulus (a bias/impulsive strategy), not detection.                                                | `MAX_FA_RATE`                        |
| Threshold inside tested range (**DN only**) | The 4-parameter psychometric fit (see below) gives a threshold μ and slope σ. The intermediate contrast levels actually presented are z-scored as (c − μ)/σ; the test requires z_min ≤ 0 ≤ z_max, i.e. the tested contrasts *bracket* μ | **otherwise excluded** ("threshold out of range") | If μ lies outside the range of contrasts tested, it was extrapolated rather than measured, and "% of psychometric curve" bins (below) would be meaningless. | `PsychometricFit.threshold_in_range` |
| Fit converged (**DN only**)                 | The fit needs ≥ 3 distinct signal levels and the optimizer must converge                                                                                                                                                                | **failure → excluded**                            | No curve = no μ/σ for stimulus binning.                                                                                                                     | `fit_psychometric`                   |

**Psychometric fit.** Probability of licking vs. contrast, fitted on trial-by-trial responses (engaged trials only) with 4 parameters: threshold μ ∈ [0, 100], slope σ ∈ [2, 40], lapse rate, guess rate (the last two bounded to ±20 % of the observed rates at the highest/lowest level). Only DN has intermediate contrast levels, so only DN is fitted. DM and DP go through the d′ / FA / engagement criteria only.

**Missing-session rule:** a session missing from 1a's `session_summary.csv` is excluded.

### A2. What "engaged" is
`engaged` is a per-trial 0/1 column that comes with `trialdata.csv`; this pipeline does **not** compute it. Everything below that says "engaged trials" simply keeps `engaged == 1`. (I could not check from the code how the column is derived — it is whatever the acquisition/preprocessing step wrote. If you want its definition documented here, tell me the rule.)

---

## B. Trial level

| Filter                                       | Definition                                                                                     | Cutoff                                                                                                                        | Why                                                                                                                                                       | Where                                                               |
| -------------------------------------------- | ---------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| Engaged trials only                          | keep trials with `engaged == 1`                                                                | —                                                                                                                             | Behavior and neural responses in disengaged stretches reflect a different state (no task strategy). Applied in 1b fits, 1c, 1d, 2b-raster, 2g, 2h, 2c–4c. | `trialdata["engaged"]`                                              |
| Continuous data restricted to engaged trials | keep behavior/video samples (and imaging frames) with tStart ≤ t ≤ tEnd of an engaged trial    | —                                                                                                                             | Same reason, for continuous traces.                                                                                                                       | `restrict_to_engaged`, `tensors.engaged_frame_mask`                 |
| Correctness (generalized)                    | correct = (signal > 0 AND lick) OR (signal = 0 AND no lick)                                    | —                                                                                                                             | "Any stimulus → lick, none → withhold" is the task rule at every contrast, not only at the 0 %/100 % catch levels. Not a filter but defines 1d's target.  | 1d                                                                  |
| "Previous trial"                             | previous *engaged* trial that is analysed (a gap from a disengaged trial is not special-cased) | —                                                                                                                             | Keeps history features defined within analysed data.                                                                                                      | 1d                                                                  |
| Minimum trials per analysis                  | trials left after all filters                                                                  | **< 20 → cell/pair/predictor skipped** (shared `MIN_TRIALS_ANALYSIS = 20`; 2g: < 20 trials → session skipped)                 | Information/correlation estimates from fewer trials are dominated by estimator bias.                                                                      | `project_config.py` (used by 1d; migration of 2c/3a–3c pending), 2g |
| Minimum trials per target bin/class          | each stimulus bin, choice class (lick/no lick) or correct/error class needs this many trials   | **< 3 → that bin is dropped** (shared `MIN_TRIALS_PER_TARGET_BIN = 3`); 2g: **< 5 per class per condition** (`MIN_PER_CLASS`) | A bin with 1–2 trials cannot yield a meaningful conditional distribution (breakdown / PID) or ROC.                                                        | `project_config.py`; `2g MIN_PER_CLASS`                             |
| Minimum trials per position bin (raster)     | trial coverage of a position bin                                                               | **< 10 → bin left blank**                                                                                                     | The corridor before trial start is poorly sampled; averages from a few trials are unreliable.                                                             | 2b `MIN_TRIALS_PER_BIN`                                             |
| Minimum engaged trials for raster            | engaged trials in the session                                                                  | **< 2 × 10 = 20 → session skipped**                                                                                           | Odd/even split needs ≥ 10 trials per half.                                                                                                                | 2b                                                                  |
| History kernel                               | trials in each group (prev correct / prev error) at a given lag                                | **< 5 each → point omitted**                                                                                                  | Difference of two means needs a minimal n in both groups.                                                                                                 | 1d `KERNEL_MIN_TRIALS`                                              |
| Regression/MI minimum                        | pairwise: < 10 complete cases skipped; multivariate: needs ≥ (#predictors + 10) complete cases | as stated                                                                                                                     | A regression with as many parameters as data is meaningless.                                                                                              | 1c, 1d, 2c                                                          |

---

## C. Sample (continuous-data) level

| Filter | Definition | Cutoff | Why | Where |
|---|---|---|---|---|
| Pupil outliers (blinks) | pupil-area samples with \|z\| > 5 relative to the session mean are set to NaN (the other channels at that timestamp are kept) | **\|z\| > 5** | Blinks/tracking failures produce non-physiological pupil values that would dominate means and correlations. | `continuous.remove_pupil_outliers` (`z_thresh=5`) |
| Pre-stimulus window | behavior predictors for "current trial" are averaged only over **[−30, −10] cm before stimStart** | window | Using any window after stimulus onset would be circular for predicting the response. | 1d `PRESTIM_WINDOW` |
| Lick as a rate | lick is an event channel; it is binned as events per window width, never averaged as 0/1 | — | Averaging a 0/1 channel gives a duty cycle that depends on sample rate. | `RATE_VARIABLES` |
| Frame-count equalization | trial window averages use the same number of imaging frames per trial (EQUALIZE_QUANTILE = 0.1) | 0.1 | Lick trials run slower (more frames per cm); for skewed, spike-like activity that biases rank statistics toward false "selectivity" (false positives ~10 % → ~5 % on null cells). | 2g |
| Fchan2 artifact frames (diagnostic) | \|Fchan2 z\| > 3 flags a possible motion / z-drift frame | **\|z\| > 3** | Red-channel (structural marker) changes indicate motion, not activity. Reported per session; not used to drop frames. | 2b `FCHAN2_ARTIFACT_Z` |

---

## D. Cell level (imaging)

A cell enters 2c–4c, 2b-raster, 2g and 2h only if it passes **both** D1 (proximity, from 2a) **and** D2 (QC, from 2b): `include = passes_proximity_filter AND qc_pass` (`neural_encoding.load_cell_inclusion`; the 2b raster uses the same rule in memory).

### D1. Proximity filter (2a)
| | |
|---|---|
| Test | An *unlabeled* V1 or PM cell passes only if it lies within **50 µm** (Euclidean, xloc/yloc) of a *labeled* cell of the **same area in the same session**. |
| Always pass | Labeled cells; every cell in AL and RSP. |
| Fail | Unlabeled V1/PM cells in a session/area with no labeled cell at all (nothing to be "near"). |
| Why | Labeled cells mark where the labelled population sits; unlabeled cells near them are a better-matched comparison population than unlabeled cells far away in the FOV. Distances are per session (xloc/yloc are local to each imaging plane). |
| Where | `PROXIMITY_RADIUS_UM = 50`, `PROXIMITY_AREAS = ("V1","PM")` in 2a |
| Note | `APPLY_PROXIMITY_FILTER = False` in 2a only controls whether 2a's *own plots* use the filtered population. The filter result is always saved (`passes_proximity_filter`) and **always applied downstream**. The distance histogram (fig 5b) shows whether 50 µm sits in a natural gap. |

### D2. Activity QC (2b, `qc_per_cell.csv`)
A cell fails if **any** of six tests fails (all reasons are stored separately, `qc_flag_<reason>`, plus `qc_fail_reason`). The thresholds are the lab's starting points; 2b's QC figure (fig 3) draws them on the metric distributions so they can be re-tuned for this dataset. 2b itself never removes cells from its own statistics.

| Reason | Metric (from the deconvolved trace) | Fails when | Why | Constant |
|---|---|---|---|---|
| `flat` | std of the raw trace | **< 1e-6** | Dead / constant channel. | `FLAT_STD_THR` |
| `lowrate` | `qc_rate` = fraction of frames above (20th percentile + 2 × std) | **< 0.01** (1 % of frames) | Silent neuron: nothing to analyse; information estimates would be degenerate. | `RATE_THR` |
| `nan` | fraction of NaN samples | **> 0.01** | Corrupt/incomplete trace. | `NAN_FRAC_THR` |
| `lowskew` | skewness of the trace | **< −1.0** | Spike-like activity is right-skewed; strong left skew suggests saturation/clipping or artifact. | `SKEW_MIN` |
| `highfano` | count-based Fano factor of active frames (> 0) in 1 s windows | **> 3.0** | Activity unstable/bursty across the session (movement or imaging artifact, drift). Recalibrated from the lab's original 1000, which belonged to a magnitude-weighted, per-trial Fano factor that can legitimately reach the hundreds. | `FANO_THR` |
| `noisy` | `noise_level` from `celldata.csv` | **> 100** | Extraction (suite2p-style) noise estimate too high. | `NOISE_THR` |

QC is computed per cell, so it is independent of which sessions are selected.

### D3. Not used as filters
`iscell_prob`, `depth`/layer and `recombinase` are plotted (2a) but not used to exclude cells. `RECOMBINASE_FILTER = None` in 2a (set to `'cre'`/`'flp'` to restrict the surveys, which also drops all `'non'` cells).

---

## E. Statistic-level minimums and group definitions

| Rule | Definition | Value | Why | Where |
|---|---|---|---|---|
| Six groups | V1-lab, V1-unl, PM-lab, PM-unl, AL, RSP; labeled cells outside V1/PM are pooled into "unl" | — | Deliberate labeling only happens in V1/PM; a stray labeled AL/RSP cell would form a near-empty, variance-distorting group. | `LABEL_SPLIT_AREAS` |
| Cells per unit | In session/mouse aggregate mode, a unit (session or mouse) needs this many cells in a group for its mean to be used | **5** | A mean of 1–4 cells is too noisy to count as one unit. | `MIN_CELLS_PER_UNIT` (2a, 2b) |
| Units per SEM | error bar needs ≥ 2 units | 2 | SEM undefined for 1 unit. | `unit_plots` |
| Pooled mode | no error bars | — | Pooling cells/sessions removes the independent units SEM needs. | `project_config.AGGREGATE_BY` |
| Significance (2g ROC, 2h correlations) | observed value outside the central 95 % of the cell's own null | **α = 0.05** (2g: 1000 permutations "mean", 200 "template"; 2h: 500 circular shifts, shift ≥ 20 s) | Each cell is tested against its own null (2h uses circular shifts, which preserve autocorrelation — a trial shuffle would make nearly everything look significant). No correction across cells/variables: ~5 % pass by chance, so compare against the 5 % line. | 2g, 2h |
| Behavior classification (2h) | cell classed by the behavior with the largest significant r (on position-residual correlations); "mixed" if the runner-up significant behavior is within 0.05; "unclassified" if none significant | **margin 0.05** | Analogue of touch/whisking/unclassified cells in Chen et al. 2013. | `CLASS_MARGIN` |
| Reliability (raster) | odd- vs even-trial profile correlation | **≥ 0.3 = reliable** (display threshold only; no cell is removed) | Separates reproducible position tuning from noise. | `RELIABILITY_MIN` |
| Shared binning for information measures | trials: engaged, finite labels; target labels: shared stimulus strategies (+ choice); response bins: **3 quantile bins** per cell | `RESPONSE_BINS = 3` | Correlation, MI, breakdown and PID must be built from the same trials and the same bins to be comparable *before* bias correction. | `project_config.py`; migration of 2c/3a–3c still pending |
| Stimulus strategies | `zero_to80_above80`, `pct_0-20_20-80_80-100`, `zero_vs_nonzero` (edges right-inclusive; psychometric % = Φ((c − μ)/σ)) | — | See A1: needs a valid DN fit. | `stim_binning.py` |
