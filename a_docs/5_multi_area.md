# Step 5: Neural characterization -- inter-area populations

## Goals
1. Relate each area's reduced signals (from `area/`) to each other, both
   jointly and time-lagged (FF should lead, FB should lag, if the
   anatomical labels track function).
2. Jointly reduce the activity of area PAIRS (RRR, CCA).
3. Correlate population-level metrics across areas across trials (e.g. do
   areas' LDA decoding projections coherently track task variables,
   a la Chen et al. 2016?).

## Findings so far
(none logged yet -- depends on `area/` being far enough along)

## Open questions
- What's the right time-lag range to test, given the imaging frame rate
  (~5.35 Hz) and known feedback latencies in the literature (Ciceri et al.
  2024's modeled feedback dynamics is a good starting reference)?
- Does the V1<->PM projection-identified signal behave differently from
  the bulk (unlabeled) V1<->PM relationship, or from V1<->AL (the control
  pair with dissimilar tuning)?
