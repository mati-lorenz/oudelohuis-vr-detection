# Project proposal (source doc, condensed)

**Title.** How do visual cortico-cortical feedforward and feedback signals
contribute to sensory and choice dynamics?

**Members.** Lorenz, Machens, Petreanu, Oude Lohuis, Mastrogiuseppe.

## Background

Perception arises through coordinated interaction of cortical areas
connected by feedforward and feedback projections (Lamme & Roelfsema 2000).
Threshold perception is highly nonlinear and variable; subthreshold stimuli
can evoke measurable V1 activity without a conscious percept (van Vugt et
al. 2019). Trial-to-trial brain state (arousal/pupil, prestimulus gamma,
prestimulus desynchronization) shapes perceptual outcome. Feedback circuits
are causally involved in perception and dynamically recruited with learning
(Manita 2015; Makino & Komiyama 2015); open question whether this shows up
as encoding differences at the projection-neuron level. Precedent for
projection-specific behavioral recruitment exists in S1 (Chen et al. 2013);
worth checking for V1<->PM.

## Research questions
1. Do V1/PM cells projecting between each other differ (encoding/dynamics)
   from cells projecting elsewhere?
2. What mechanisms produce these differences -- do they fulfill specific
   roles?
3. Is trial-by-trial variability in FF/FB signal strength/structure
   associated with perception/choice?

## Task
Head-fixed mice run a VR corridor, lick in a reward zone to report
detection of a visual target embedded in a repeating noisy background
(200 cm repeat period) -- decouples stimulus timing from motor report.

**Protocols**: DM (catch + 100% only), DP (0/5/12/25/100%), DN (catch +
noise + 100%, jittered near-threshold noise trials give the signal-strength
regression dimension; ONLY protocol with simultaneous neural recording;
one stimulus per session; recording continued past satiety for a
disengaged-but-running control epoch).

## Multi-area, projection-identified imaging
Simultaneous mesoscale 2p (2p-RAM), full 4-mm window. Dual retrograde
labeling: PM injection -> PM->V1 (feedback) neurons; V1 injection ->
V1->PM (feedforward) neurons. Areas: V1, PM (primary reciprocal pair),
AL (control, dissimilar tuning), RSP (above PM, navigational
decision-making). 8 planes / 4 areas (3 V1, 3 PM, 1 AL, 1 RSP), 600x600um
@ 512x512px, ~5.35 Hz. Suite2p + Cellpose. RF-mapping control session
confirms spatial corridor position correlates with azimuth (not elevation)
RF. Neurons NOT tracked across days -- analyze at session level, confirm
results hold across animals.

**Dataset**: 6 mice, 32 sessions (3-7/subject), 52,752 neurons (2,405
anatomically labeled). ~1648.5+/-601.0 neurons/session, ~392.5+/-181.5
trials/session (255.5+/-101.5 engaged, 137.0+/-103.9 disengaged;
139.7+/-51.1 engaged noise-type trials/session -- the key trial type for
perception questions).

## Five-step data analysis approach
1. **Behavioral characterization** -- see `behavior/README.md`
2. **Single-cell characterization (first order)** -- see `single/README.md`
3. **Pairwise characterization (second order)** -- see `pairs/README.md`
4. **Single-area population characterization** -- see `area/README.md`
5. **Inter-area population characterization** -- see `multi_area/README.md`

## Key literature to fold into motivation
Kwon 2016, Chen 2016, Condylis 2020, Han & Helmchen 2024, Ishizawa 2016,
Ciceri 2024, Oude Lohuis 2022, Chen 2013, Glickfeld 2013.
