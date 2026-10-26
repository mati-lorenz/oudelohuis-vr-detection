# Project overview: feedforward and feedback signals in a VR detection task

Members: Matías Lorenz, Francesca Mastrogiuseppe, Matthijs Oude Lohuis, Leopoldo Petreanu, Christian Machens
Started: 2026-10-07

This is a working summary of the project: questions, data, decisions, status and key literature. The full background is in [project_proposal.md](project_proposal.md), and the folder layout and pipeline rules are in the [README](../README.md).

## Questions

How do visual cortico-cortical feedforward and feedback signals contribute to sensory and choice dynamics?

1. Do V1 and PM neurons that project to each other differ in encoding or dynamics from neighbouring neurons projecting elsewhere?
2. What mechanisms produce these differences, and do they serve specific roles?
3. Is trial-to-trial variability in the strength or structure of feedforward or feedback signals associated with perception and choice?

## Current state

- Where it stands: behavioural characterization mostly done; single-cell, pairwise and single-area scripts (steps 2 to 4) under way.
- Main obstacle:
- Next milestone:

## Data

Active part of VISTA (VISion Two Areas), Matthijs Oude Lohuis's 2022–2024 dataset (Machens and Petreanu labs). Full description in [Dataset Organization Matthijs.pdf](../0_data/Dataset%20Organization%20Matthijs.pdf).

| Protocol | Trials | Neural recording |
|---|---|---|
| DM (detection maximum) | catch and 100% | No |
| DP (detection psychometric) | catch, 5, 12, 25 and 100% | No |
| DN (detection noise) | catch, near-threshold noise trials (about 8 to 23%) and 100% | Yes |

- **Task:** head-fixed mice run through a VR corridor and lick in a reward zone to report a visual target embedded in a repeating noisy background (period 200 cm). DN uses one stimulus per session and continues after the mouse is sated, giving disengaged trials.
- **Mice and labelling:** Thy1-GCaMP6s (GP4.3); retrograde Cre and Flp viruses with tdTomato label V1→PM (feedforward) and PM→V1 (feedback) neurons. Pairing counterbalanced but unbalanced (5 mice with Flp-tdTomato in V1, 1 with Cre-tdTomato in V1).
- **Imaging:** 2p-RAM mesoscope, 8 planes (3 V1, 3 PM, 1 AL, 1 RSP), 600 × 600 µm at 512 × 512 px, about 5.35 Hz; Suite2p and Cellpose; FaceMap for pupil and face.
- **Counts:** 6 mice, 32 sessions; 52,752 neurons, 2,405 labelled (about 5%); per session 1,648 ± 601 neurons, 392 ± 182 trials (256 ± 102 engaged), 140 ± 51 engaged noise trials.
- **Processed data:** `CorticalCircuitsShare_2/Users/Matthijs/Procdata`, one folder per `Protocol / Animal / Date`, with `sessiondata`, `trialdata`, `behaviordata` (100 Hz), `videodata`, `celldata` (includes projection label), `dFdata` and `deconvdata` as .csv. Use deconvolved activity for stimulus-locked analyses, since dF/F bleeds into the next stimulus.
- **Session overviews:** [DE_Sessions_Overview.xlsx](DE_Sessions_Overview.xlsx) and [MOL_Recordings_Overview.xlsx](MOL_Recordings_Overview.xlsx).

### Known problems

- Neurons are not tracked across days; the densely labelled zone is likely resampled.
- Receptive-field mapping was run after only one session.
- Disengaged trials always come last in a session, so they are confounded with time (bleaching, lower signal-to-noise).
- Stimulus input varies across trials: running speed, eye position, and the mix of target and background.
- Some sessions have no usable hit/miss contrast.
- The documents describe the Cre/Flp injections in opposite ways; check each mouse's labels in `celldata` before pooling.

## Decisions

Choices that would be hard to reconstruct later. One line each: date, what, why.

- From proposal: for hit/miss questions, exclude sessions with d' < 1, false-alarm rate > 0.5, or no overlap between tested stimulus range and fitted threshold.
- From proposal: compare across mice with subjective saliency (z-scored position on each session's psychometric curve).
- From proposal: analyse engaged trials only.
- From proposal: analyse at session level, checking that results hold across animals.
- From proposal: effects in the stimulus window need running-speed controls, because visual flow depends on speed.
- From proposal: spatial binning within areas (more stable correlation structure); time binning for multi-area analyses with delays.
- 2026-10-07: de Kok folder layout; one numbered script per step, with progress notes in `a_docs/` and `b_progress/`.

## Analysis plan and status

| Step | Question | Method | Control | Status |
|---|---|---|---|---|
| 1 | How do mice solve the task, and which sessions and trials suit each question? | Performance, psychometric fits, behavioural predictors | Exclusion criteria; engaged trials only | mostly done |
| 2 | What do single cells encode, and do projection-identified cells differ from neighbours? | Activity statistics, mutual information, linear and nonlinear encoding, ROC | Nearby unlabelled cells; running speed | in progress |
| 3 | Are pairwise correlations explained by tuning or by projection identity? | Pair statistics, information breakdown, PID | Shuffles; pairs matched for distance and tuning | in progress |
| 4 | What is the population structure within each area, and does it depend on correlations? | PCA, FA, SVM | Trial permutations removing correlations | in progress |
| 5 | Do V1 and PM share latents, and do FF and FB neurons carry them? | Lagged latents, RRR, CCA, LDA projections | Lag and trial shuffles; AL as nonspecific control | not started |

### Checklist

**1 Behaviour**
- [ ] 1a performance: raw data, no exclusions, engagement (DM, DP)
- [ ] 1b psychometric fit and subjective saliency (DP, DN)
- [ ] 1c behaviour: speed, position, pupil, video (DM, DP, DN)
- [ ] 1d performance predictors, current and past trial (DM, DP, DN)

**2 Single cells (DN)**
- [ ] 2a cell distribution · 2b activity statistics · 2c mutual information
- [ ] 2d linear encoding vs 2c · 2e nonlinear encoding
- [ ] 2f lagged information · 2g ROC · 2h correlation with behaviour · 2i rasters

**3 Pairs (DN)**
- [ ] 3a pair statistics · 3b information breakdown · 3c PID
- [ ] 3d to 3f temporal versions

**4 Single-area populations (DN)**
- [ ] 4a PCA (choose one of three variants) · 4b FA · 4c SVM
- [ ] Permutations to test the role of correlations
- [ ] Stability of reduced dimensions within a trial

**5 Across areas (DN)**
- [ ] Relate step-4 latents across areas, jointly and time-lagged
- [ ] Joint reduction of two areas (RRR, CCA)
- [ ] Correlate population metrics across trials (as in Chen et al. 2016)

**Housekeeping**
- [ ] Remove or archive `2_pipeline/1_behav`, `1_behavior` and `2_single`, which match no current script

## Results so far

- Behaviour: d' 2.26 ± 0.88 (26 sessions), variable across sessions and animals; hits show slowing and licking in the second half of the stimulus window.
- Behaviour: choice predictable from ITI behaviour only slightly and in some sessions; in most, the choice signal appears in the last ~5 cm of the stimulus window.
- Neural (Matthijs, earlier work): activity is spatially locked to the stimulus window; space and time binning give similar PSTHs and decoding.
- Neural: reproducible coding of the repeating background texture, small share of variance.
- Neural: mean population activity scales with stimulus strength, but many neurons are threshold-tuned, giving a curved population geometry. Threshold tuning appears enriched in both V1→PM and PM→V1 neurons.
- Figures in [3_output](../3_output/); per-step notes in [a_docs](.) and [b_progress](../b_progress/).

## Open questions

- How many sessions remain after the exclusion criteria, and can the full dataset be used for other questions? (to ask Matthijs)
- What predicts performance from prestimulus behaviour: arousal, running speed or trial number? Is the relation nonlinear?
- How does prestimulus activity in V1 and PM, labelled and unlabelled, influence detection?
- Is any part of the engaged vs disengaged comparison usable, given the order confound?
- Are pairwise correlations explained by shared tuning or by shared projection identity?
- Idea: include neurons without significant discrimination in population analyses, to catch synergy (unlike Chen et al. 2013).

## Key literature

**Feedforward vs feedback, threshold perception**
- Lamme & Roelfsema 2000, [doi:10.1016/S0166-2236(00)01657-X](https://doi.org/10.1016/S0166-2236(00)01657-X): feedforward sweep vs recurrent processing.
- van Vugt et al. 2018, [doi:10.1126/science.aar7186](https://doi.org/10.1126/science.aar7186): weak stimuli are lost along the hierarchy; prestimulus state shifts bias and sensitivity.
- Manita et al. 2015, [doi:10.1016/j.neuron.2015.05.006](https://doi.org/10.1016/j.neuron.2015.05.006): a top-down loop needed for accurate perception.
- Makino & Komiyama 2015, [doi:10.1038/nn.4061](https://doi.org/10.1038/nn.4061): learning increases top-down influence in V1.
- Ishizawa et al. 2016, [doi:10.1523/JNEUROSCI.4577-15.2016](https://doi.org/10.1523/JNEUROSCI.4577-15.2016): area-specific dynamics during loss of consciousness.

**Projection-defined recruitment**
- Chen et al. 2013, [doi:10.1038/nature12236](https://doi.org/10.1038/nature12236): S1 neurons projecting to S2 and M1 carry task-dependent information.
- Chen et al. 2015, [doi:10.1038/nn.4046](https://doi.org/10.1038/nn.4046): the two pathways change differently with learning.
- Chen et al. 2016, [doi:10.7554/eLife.14679](https://doi.org/10.7554/eLife.14679): S1-S2 coordination; projection neurons contribute only with touch.
- Glickfeld et al. 2013, [doi:10.1038/nn.3300](https://doi.org/10.1038/nn.3300): V1 sends target-matched tuning to AL and PM.
- Ravishankar et al. 2025, [doi:10.64898/2025.12.22.696069](https://doi.org/10.64898/2025.12.22.696069): PPC→M1 neurons encode like their neighbours.
- Oude Lohuis, Mastrogiuseppe, Petreanu & Machens, preprint: projection neurons amplify shared across-area latents.

**Loops during detection**
- Kwon et al. 2016, [doi:10.1038/nn.4356](https://doi.org/10.1038/nn.4356): stimulus and choice propagate both ways in an S1-S2 loop.
- Condylis et al. 2020, [doi:10.1016/j.neuron.2020.02.004](https://doi.org/10.1016/j.neuron.2020.02.004): context-dependent S1-S2 processing in projection neurons.
- Han & Helmchen 2024, [doi:10.1038/s41593-023-01534-x](https://doi.org/10.1038/s41593-023-01534-x): top-down predictions between PPC and S1.
- Oude Lohuis et al. 2022, [doi:10.1038/s41467-022-30600-4](https://doi.org/10.1038/s41467-022-30600-4): V1 is needed until the late, report-related wave.
- Ciceri et al. 2024, [doi:10.1523/ENEURO.0191-24.2024](https://doi.org/10.1523/ENEURO.0191-24.2024): model in which PPC gates feedback to V1.
- Rowland et al. 2023, [doi:10.1038/s41593-023-01413-5](https://doi.org/10.1038/s41593-023-01413-5): S1-to-S2 propagation depends on prestimulus variability.

**Prestimulus state**
- Vinck et al. 2015, [doi:10.1016/j.neuron.2015.03.028](https://doi.org/10.1016/j.neuron.2015.03.028): arousal and locomotion affect cortical activity differently.
- Also cited in the proposal: Speed et al. 2018; Montijn et al. 2016; McGinley et al. 2015.

**Information and correlations**
- Safaai et al. 2025, [doi:10.1038/s41593-025-02095-x](https://doi.org/10.1038/s41593-025-02095-x): neurons sharing a target have a specific, choice-enhancing correlation structure.
- Cohen & Kohn 2011, [doi:10.1038/nn.2842](https://doi.org/10.1038/nn.2842): measuring and interpreting correlations.

**Population and across-area methods**
- Cunningham & Yu 2014, [doi:10.1038/nn.3776](https://doi.org/10.1038/nn.3776): dimensionality reduction review.
- Semedo et al. 2019, [doi:10.1016/j.neuron.2019.01.026](https://doi.org/10.1016/j.neuron.2019.01.026): communication subspace between V1 and V2.
- Kaufman et al. 2014, [doi:10.1038/nn.3643](https://doi.org/10.1038/nn.3643): output-null and output-potent dimensions.
- Stringer et al. 2019, [doi:10.1038/s41586-019-1346-5](https://doi.org/10.1038/s41586-019-1346-5): high-dimensional V1 responses with a power-law spectrum.
- Mastrogiuseppe et al. 2026, [doi:10.64898/2026.05.18.725829](https://doi.org/10.64898/2026.05.18.725829): lead-lag between areas can follow inputs rather than transmission.
