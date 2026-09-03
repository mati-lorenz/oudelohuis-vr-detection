# Step 4: Neural characterization -- single-area populations

## Goals
1. Compare dimensionality-reduction techniques (PCA, FA, etc.) on each
   area's population activity.
2. Apply different levels of permutation/shuffling to isolate the role of
   neuron-neuron correlations in encoding (see
   `docs/null_hypotheses_framework.md`).
3. Study how the reduced dimensions' geometry/invariance evolves over
   time/distance within a trial.

## Findings so far
(port over the dimensionality-estimate results once re-run under this
repo's structure -- prior runs found raw PCA needs far more components
than FA due to heterogeneous per-neuron variance, not genuine extra shared
structure; see `docs/null_hypotheses_framework.md`.)

## Open questions
- Does the threshold-tuned vs. max-tuned split (see `single/README.md`)
  correspond to a specific direction/component in the reduced space?
- Does projection identity predict position in the reduced space?
