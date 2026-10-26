## multi_area_detection_task_ff_fb

How do visual cortico-cortical feedforward and feedback signals contribute to sensory and choice dynamics? Multi-area (V1/PM/AL/RSP), projection-identified 2p imaging during a VR detection task (protocols DM/DP/DN). See `a_docs/project_proposal.md` for full background.

### Protocols

|Protocol|Signal strengths|Cell recording|
|---|---|---|
|DM|0% and 100% only|No|
|DP|0%, 100%, and intermediate strengths|No|
|DN|0%, 100%, and intermediates derived from DP|Yes|

### Folder structure

This is the de Kok layout (Ties de Kok, "How to keep your research projects organized, part 1: folder structure"), plus lettered folders for infrastructure that isn't itself a pipeline stage:

- `0_data/`: raw, read-only input (one folder per protocol/mouse/date).
- `1_scripts/`: numbered scripts, one per analysis step, in execution order, plus `utils/`, the shared code library.
- `2_pipeline/`: one sub-folder per script, name-matched:
    - `<script_name>/out/`: end-products for a later script.
    - `<script_name>/store/`: this same script's own cache.
    - `<script_name>/tmp/`: throwaway inspection files.
- `3_output/`: hand-picked, presentation-ready figures and tables. Name files with the step prefix, since the folder is flat.
- `a_docs/`: project proposal, null-hypotheses philosophy, per-step progress notes (Goals / Findings / Open questions).
- `b_progress/`: markdown to slide-deck pipeline for sharing with coworkers.
- `c_templates/`: copy-paste starting points for new scripts.
- `d_archive/`: old code.

**Rules**

1. A script only loads data from `0_data/` or from the `out/` folder of a script that runs before it. Never load from a later script, and never reach into another script's `store/` or `tmp/`.
2. Never write into `0_data/`.
3. `1_scripts/utils/pipeline.py` (Python) and `pipeline.R` (R) resolve each script's own `2_pipeline/<name>/` folder from the script's file location. Call `get_pipeline_paths(__file__)` first, rather than hand-maintaining an output path.

### Analysis pipeline

Each step is one numbered script in `1_scripts/` and one progress file in `a_docs/`.

**1 – Behavioral characterization**

- `1a_performance`: raw-data analyses with no exclusions, plus engagement level (DM, DP).
- `1b_psychometric`: psychometric fit, used to establish standards (DP, DN).
- `1c_behavior`: speed, position, pupil size and video analyzed against each other (DM, DP, DN).
- `1d_performance_predictors`: relationships between performance and behavioral variables, current or past trial (DM, DP, DN).

**2 – Single-cell characterization** (DN)

- `2a_cell_distribution`: spatial organization of recorded cells across sessions.
- `2b_activity_statistics`: statistics of single-cell spiking activity.
- `2c_information`: mutual information between cell activity and behavior / performance (stimulus, choice).
- `2d_linear_encod`: linear regressions, compared against `2c_information` to check whether the relationships are already captured linearly.
- `2e_nonlinear_encod` (maybe): nonlinear fits if there is a large gap between total and linearly explained information.

**3 – Pairwise-correlations characterization** (DN)

**4 – Single-area analysis** (DN)

**5 – Multi-area analysis** (DN)

### Shared code: `1_scripts/utils/`

`1_scripts/utils/infotheory/` is the shared analysis library (binning, MI, dimensionality, session loader, plotting and table helpers, the pipeline-path helper). Install once with `pip install -e .`, then any script can `from infotheory import ...`.

### Docs, progress, templates, archive

`a_docs/` holds `project_proposal.md` and one progress file per step. `b_progress/`, `c_templates/` and `d_archive/` each have their own contents.