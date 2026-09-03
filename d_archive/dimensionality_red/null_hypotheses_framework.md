# Null Hypotheses for the Dimensionality-Reduction Pipeline

Framed around Elsayed & Cunningham (2017, *Nat. Neurosci.*), "Structure in
neural population recordings: an expected byproduct of simpler phenomena?"

## The core principle

Population-level structure — low dimensionality, rotational dynamics,
specific encoding geometry — can be a trivial consequence of simple
single-neuron properties (tuning shape, timing, variance) rather than
evidence of coordinated computation. Their fix: build a null model that
preserves *exactly* the simple statistic you want to rule out as the
explanation, and randomizes everything else. Two failure modes to avoid:

- **Too weak a null** (e.g. full randomization of everything) destroys so
  much that almost anything looks "significant" against it — it doesn't
  test the specific alternative explanation that actually threatens your
  claim.
- **A mismatched null** tests the wrong confound entirely, missing the
  simple phenomenon that could actually explain the result.

This was tested directly during development (see `run_jpca_computation.py`'s
history): two same-frequency oscillatory signals trace an ellipse — i.e.
look exactly like rotation to a linear-dynamics fit — for ANY relative
phase, with zero true coordination required. A null that destroys
timing/frequency content entirely (full time permutation) can under-detect
this specific confound; a null that preserves each dimension's own
frequency content but randomizes cross-dimension phase (phase
randomization) targets it directly. Validated: on synthetic phase-shifted-
copy data (R=0.84, looks dramatic), phase randomization gives a null mean
of 0.85 — correctly flagging the result as unremarkable, which the coarser
permutation null also got right here but with a visibly weaker null
(mean 0.77), i.e. a less specific test that would be more likely to miss
this exact confound in a harder case.

## Stage by stage

### Step 1 — Dimensionality estimate (FA/PCA cross-validated log-likelihood)

**Simple alternative explanation**: neurons are independent; any apparent
multi-dimensional structure is an artifact of overfitting to a small
number of trials, not genuine shared population covariance.

**Null**: shuffle each neuron's trial order independently (breaks
cross-neuron trial-to-trial covariation while preserving each neuron's own
marginal firing-rate distribution across trials exactly). Refit the same
CV FA/PCA procedure on the shuffled data. If the real data's optimal
dimensionality / peak CV log-likelihood doesn't exceed this null, there's
no evidence of genuine shared structure beyond independent per-neuron
variability — the whole premise of a population-level dimensionality
estimate would be unsupported.

**Status**: previously absent (the script only ever fit the real data).
Added `cv_dimensionality_null` — see `run_dimensionality_estimate.py`.

### Step 2 — GPFA single-trial trajectories

**Simple alternative explanation**: apparent single-trial "trajectory"
structure is just each neuron's own condition-mean PSTH plus independent
per-neuron noise, with no genuine shared (population-level) single-trial
fluctuation.

**Null**: subtract each neuron's condition-mean PSTH, then shuffle the
residuals independently per neuron across trials within condition
(preserves each neuron's own residual variance/autocorrelation, destroys
cross-neuron trial-to-trial covariation). Compare the leading eigenvalue
fraction of the trial-to-trial covariance matrix between real and null —
if real data's shared variance isn't bigger, single-trial GPFA
trajectories don't reflect genuine coordinated single-trial dynamics
beyond independent noise around a shared mean.

**Status**: added as a lighter-weight diagnostic (`report_shared_variance`)
alongside the existing GPFA fit, rather than a full second GPFA refit per
shuffle (too expensive to repeat many times) — see
`run_gpfa_trajectories.py`.

### Step 3 — TDR (encoding R²)

**Simple alternative explanation**: the observed R² peak (Steps 3
discussion: 0.14-0.24 right after the stimulus zone) is what you'd expect
by chance given the number of PCA dimensions and trials, not genuine
stim/choice encoding.

**Null**: shuffle stim (and separately choice) labels across trials,
refit the same regression, get a null R²(t) distribution. This was a real
gap — Steps 4 and 5 both got proper significance tests during development,
but Step 3 never did, despite being exactly the kind of small-sample
regression (many PCA dims, modest trial counts) where the jPCA
investigation showed overfitting alone can produce misleadingly large
numbers. **Added** — see `run_tdr_encoding.py`.

### Step 4 — jPCA (rotational dynamics)

**Simple alternative explanations**: (a) overfitting given more free
parameters than samples (already caught and fixed earlier — see the
degenerate-dimension trim and the original time-permutation null); (b)
band-limited/oscillatory signals of a common frequency trace an ellipse
regardless of true coordination (the confound validated above).

**Null**: TWO nulls now reported side by side —
1. Time-permutation within condition (existing): destroys the local
   state→derivative relationship entirely. Good general-purpose null,
   catches gross overfitting.
2. Phase randomization (new): preserves each dimension's own power
   spectrum/autocorrelation, randomizes cross-dimension phase. Targets
   the ellipse confound specifically, per Elsayed & Cunningham's
   philosophy of preserving exactly the simple statistic being ruled out.

A result should be treated as real rotational structure only if it clears
BOTH nulls, not just the more permissive one.

### Step 5 — Information content of the population latent (MI)

**Simple alternative explanation**: the population-latent MI is no
different from what any single, arbitrary linear combination of neurons
would show — i.e. the "population" framing doesn't add anything beyond
what a single well-tuned neuron already provides.

**Status**: the existing shuffle-corrected MI test already targets the
right question for THAT specific claim (is there information at all,
beyond chance) and was validated during development. The remaining
Elsayed & Cunningham-flavored question — does the population-latent
number exceed what the best single neuron alone would show, or is it just
inheriting one dominant neuron's tuning — is the natural next comparison
once single-cell MI values are joined in (flagged previously as the next
step; not yet implemented).
