## [[@vanvugt2018threshold]]
- **Outcome predicted from single prestimulus markers, as AUROC per marker** (Fig. 3A)
- **Hit, miss, correct-rejection and false-alarm traces, per area and contrast bin** (Fig. 2A)
- **Miss fraction per area and projection class** (Fig. 2B)
- **Pre-trial activity before false alarms vs. correct rejections** (Fig. 2A)
- **Choice probability before and after stimulus onset** (Fig. 3A)
- **Trial-by-trial coupling between V1→PM activity and the PM response, hits vs. misses** (Fig. 4D)

## [[@chen2013Behaviourdependent]]
- **Population discrimination with and without the individually discriminating neurons** (Supp. Fig. 8)
- **Functional classification of neurons by cross-correlation with behavioural vectors** (Chen, Fig. 1f–h)
- **Single-neuron ROC, ranked, with shuffle threshold** (Chen, Fig. 2c): which neurons separate hits from misses in the intermediate bin?
- **Engaged vs. disengaged trials as the analogue of trained vs. naive** (Fig. 4c–e)
## [[@kwon2016Sensory]]
- **Analysis window set from the reaction-time distribution** (Supp. Fig. 1; cyan band in Fig. 2a): they use the first 0.25 s after onset, which precedes 96.5% of first licks. A direct precedent for your behaviour-free window.
- **Movement-controlled subset of trials** (Supp. Fig. 8): does the hit/miss difference survive in trials with no early whisker or tongue movement on video? Your equivalent is a speed-matched, no-early-lick subset.
- **Fraction of task-responsive neurons per area** (Fig. 1h): 37% in S1, 24% in S2. A simple first figure for V1, PM, AL, RSP and by projection class.
- **Spatial maps and clustering of SP and DP neurons** (Fig. 2e–j): are choice-related neurons clustered, and does SP or DP fall off with distance from the stimulus representation? Your equivalent is distance from the stimulus's retinotopic location, which you can estimate only where RF mapping exists.

## [[@rowland2023Propagation]]
- **Prestimulus population mean vs. population variance, hits vs. misses** (Fig. 4a–b): the mean did not predict outcome in S1; the variance across neurons did, being higher before misses. Computed over the 500 ms before the stimulus.
- **Hit probability against binned prestimulus variance, per session** (Fig. 4c): a negative relationship in 10 of 11 sessions.


### Oude Lohuis et al. 2022

**Step 1: Behaviour**

- **Reaction time by subjective saliency level (sub, thr, sup, max)** (Fig. 1h): does report timing shift with stimulus strength? This uses the same psychometric-position binning you already plan.

**Step 2: Single-cell**

- **Z-scored traces for hit, miss, false alarm and correct rejection** (Fig. 2e): is there an early stimulus wave and a late hit-only wave?
- **Kernel GLM with variance explained over time per predictor group** (Fig. 3c): how much of the late activity is vision, licking, hit or pupil? This is the main tool for separating report from movement.
- **Time-resolved ROC, as the fraction of significantly coding neurons** (Fig. 4a–b): when do stimulus-presence and hit/miss coding appear, by area and projection class?
- **Onset of hit/miss coding against reaction time, per session** (Fig. 4d–f): onset was 162 ms in unisensory vs. 288 ms in multisensory mice, and about 280 ms before the first lick.
- **Lick-aligned activity** (Supp. Fig. 3): is the “late” signal locked to the stimulus or to the response?

**Step 3: Pairwise**

- **Change in noise correlations from baseline, hits vs. misses** (Fig. 7c): they dropped only on hits.
- **Timing of that drop by reaction-time tertile** (Fig. 7d–f): it was locked to the lick, not the stimulus.
- **Noise correlations around stimulus onset, hits vs. misses** (Fig. 7i): higher on hits, the opposite of Rowland’s variance result.

**Step 4: Populations**

- **Stimulus decoding over time, compared across conditions** (Fig. 7b): is sensory information equally present on hits and misses?

**Limits for reuse:** the coding-onset and decorrelation latencies rest on 10–25 ms bins. At 5.35 Hz they are out of reach in time, so use corridor position instead. The optogenetic and laminar results have no equivalent in your data.

### Safaai et al. 2025

**Step 2: Single-cell**

- **Activity time course and peak-time distribution per projection class** (Fig. 1d–f): do V1→PM and PM→V1 neurons peak at different corridor positions from their neighbours?
- **Information per task variable, per projection class, conditioned on movement** (Fig. 3b): what is each channel enriched for once running is accounted for?

**Step 3: Pairwise**

- **Noise correlations for same-target pairs vs. unlabelled pairs, correct vs. error, with and without conditioning on movement** (Fig. 4b). Hit vs. miss is your analogue.
- **Interaction information per pair** (Fig. 4c–e): do a pair’s correlations add or remove information about choice?
- **Distance-matched control** (Ext. Data Fig. 5c): needed, since labelled neurons sit slightly closer together.

**Step 4: Populations**

- **Triplet motif probabilities and clustering coefficient against a shuffled network** (Fig. 5c–d): are information-enhancing pairs clustered?
- **Within-pool vs. across-pool interactions, with pools defined by choice preference** (Fig. 5j–k).
- **Population information split into independent, unstructured and structured parts, against population size** (Fig. 6e).

**Limits for reuse:**

- The network structure did not appear when a GLM replaced their copula model (Ext. Data Fig. 9d), so the main result is method-dependent.
- The effect is small: 0.015 bits at 75 cells.
- It rests on 1,355 same-target pairs against 145,439 unlabelled.
- Correct trials outnumber errors about 4 to 1. I did not check how they equalised trial counts, so match hits and misses yourself.

### Mastrogiuseppe et al. 2026

This is mostly theory, with one data figure, and all of it belongs to Step 5.

- **Cross-covariance asymmetry score between areas** (Fig. 1C): which area leads?
- **Alignment of the dominant shared direction (PLS or CCA) with the population-mean axis vs. a feature-specific axis** (Fig. 7C): is V1–PM shared variability mostly a global fluctuation?
- **Cross-covariance amplitude and asymmetry for unspecific vs. specific latents, stimulus vs. spontaneous** (Fig. 7E–K): your analogue is the stimulus zone vs. the inter-trial corridor.
- **Model prediction to test** (Fig. 4B, 6H): the area receiving stronger input to its excitatory cells leads, so V1 should lead more for stronger stimuli.

**Limits for reuse:**

- The data analysis uses 2 ms bins and lags of ±8 ms. Your frame interval is about 187 ms, so the asymmetry analysis does not transfer.
- At lags of whole frames, different planes are acquired at different times, and indicator kinetics may differ between areas. Both can produce spurious lead or lag.
- The zero-lag analyses (alignment and amplitude) do transfer.
- The paper shows asymmetry can arise with no connection between areas (Fig. 4H), so it is not evidence of transmission by itself.

Francesca and Christian are on your project, so they can tell you which parts they consider applicable.

### Glickfeld et al. 2013

This paper used passive gratings and axon imaging, so it gives few analyses but one important control.

**Step 2: Single-cell**

- **Tuning of projection neurons against the preferences of the target area** (Fig. 2c–d, 5d): are V1→PM neurons matched to PM?
- **Share of matched inputs vs. their response amplitude** (Fig. 5b, 6c–e): is a class difference due to more tuned cells or to stronger responses? This is the same split as Chen’s fraction vs. accuracy.
- **No spatial clustering of tuning within V1** (Fig. 3c–e): supports comparing labelled cells with nearby unlabelled ones.

**The control it implies:** V1 input to PM prefers slow speeds (median 20 deg/s, against 73 for AL), with about threefold larger responses to slow stimuli. Your mice slow down on hits. Slower visual flow should therefore drive V1→PM neurons more than their neighbours, whatever the animal perceived. A larger hit/miss difference in V1→PM cells is exactly what this confound predicts, so speed matching is essential for that comparison. AL should show the opposite sign, which makes it a useful check.