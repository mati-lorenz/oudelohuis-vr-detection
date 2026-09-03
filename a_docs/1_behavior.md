# Step 1: Behavioral characterization

**Status: mostly done** (per project doc) -- treat this folder as
maintenance + the open questions below, not a from-scratch build.

## Goals
Understand how the mice solve the task: performance, variability across
mice, and which trials/sessions are usable for which downstream neural
questions.

## Findings so far
- Mice perform the VR detection task well overall but with variable
  behavior across sessions/animals: d-prime 2.26 +/- 0.88 (N=26 sessions).
- Hits are reflected in spatially selective licking and running-speed
  slowdown starting in the second half of the stimulus window; lick rate
  and slowdown scale similarly with stimulus intensity.
- Choice is only weakly predictable from ITI behavior in some sessions; in
  most sessions choice-related signal only emerges in the last ~5cm of the
  stimulus window.
- **Session inclusion criteria for hit/miss questions**: exclude if
  d-prime < 1, false-alarm rate > 0.5, or no overlap between tested
  stimulus range and fitted threshold. TODO: how many sessions does this
  actually filter, and can the excluded ones still be used for other
  (non hit/miss) questions?
- **Subjective saliency**: convert %signal to a z-scored position on each
  session's fitted psychometric curve, so trials are comparable across
  mice/sessions with different subjective saliency for the same stimulus.
- **Use only engaged trials** for most questions (255.5+/-101.5 engaged vs.
  137.0+/-103.9 disengaged per session) -- but note disengaged trials are
  always at the END of the session (no interleaving), so
  engaged-vs-disengaged comparisons are confounded with time/bleaching.
- **Visual flow confound**: running speed changes the effective temporal
  frequency of the visual scene: control for running speed in any analysis
  of the sensory stimulus window.

## Open questions
- What actually predicts performance -- arousal, running speed, trial
  number, or a nonlinear combination (optimal intermediate arousal,
  McGinley et al. ~2016)?
