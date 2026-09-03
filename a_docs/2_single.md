# Step 2: Neural characterization -- first order (single cell)

## Goals
Describe single-cell encoding and dynamics: temporal/spatial response
profile, which time window to isolate, task-variable selectivity, and how
projection-identified populations differ from nearby (all-else-equal)
unlabeled neurons.

## Findings so far
- **Analyze in space, not time** for single-area questions: activity is
  reliably spatially locked to the stimulus window in the VR corridor
  (validated space-vs-time: similar PSTH/decoding, but spatial binning
  gives higher/more stable correlation structure). For MULTI-area
  questions with temporal delays between areas, analyze in time instead.
- Neurons show reproducible, cross-validated **background coding** tied to
  the repeating noisy texture (independent of the target stimulus), but it
  explains only a small fraction of response variance.
- Mean population activity scales monotonically with stimulus strength,
  but individual neurons often show **threshold sensitivity**: two
  dominant tuning patterns -- neurons tuned to threshold-level contrast vs.
  neurons tuned to maximum contrast (catch/threshold/max show a graded
  population increase, but some neurons peak at threshold and DROP at
  max). Shows up as a curved 'geometry' in population PCA.
- The fraction of threshold-tuned neurons may differ by projection
  identity: some indication V1->PM and PM->V1 labeled neurons are BOTH
  enriched for threshold tuning vs. unlabeled neighbors. Needs confirming.

## Open questions
- Activity statistics / linear encoding / information theory sub-analyses:
  fill in as they're run (see `scripts/`).
- Confirm the threshold-tuning-by-projection-identity finding is robust
  across sessions/animals, not driven by one or two sessions.
