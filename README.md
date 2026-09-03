# multi_area_detection_task_ff_fb

How do visual cortico-cortical feedforward and feedback signals contribute
to sensory and choice dynamics? Multi-area (V1/PM/AL/RSP), projection-
identified 2p imaging during a VR detection task (protocols DM/DP/DN).
See `a_docs/project_proposal.md` for full background.

## Protocols

| Protocol | Signal strengths                          | Cell recording |
|----------|--------------------------------------------|-----------------|
| DM       | 0% and 100% only                            | No              |
| DP       | 0%, 100%, and intermediate strengths        | No              |
| DN       | 0%, 100%, and intermediates derived from DP | Yes             |

## Folder structure

The original de Kok layout (Ties de Kok, "How to keep your research
projects organized, part 1: folder structure",
https://medium.com/data-science/how-to-keep-your-research-projects-organized-part-1-folder-structure-10bd56034d3a),
plus lettered folders for infrastructure that isn't itself a pipeline
stage:

    0_data/       raw, READ-ONLY input (one folder per protocol/mouse/date)
    1_scripts/    numbered scripts, one per analysis step, in execution order
                  (see "Analysis pipeline" below); plus utils/ -- the shared
                  code library (see below)
    2_pipeline/   one sub-folder per script in 1_scripts/, name-matched:
                      <script_name>/out/    end-products for a LATER script
                      <script_name>/store/  this SAME script's own cache
                      <script_name>/tmp/    throwaway inspection files
    3_output/     hand-picked, presentation-ready figures/tables (NOT
                  everything a script produces -- only what's worth
                  sharing/publishing; name files with the step prefix,
                  e.g. `1_behavior_psychometric_curve.png`, since this
                  folder is flat)
    a_docs/       project proposal, per-step progress notes 
                  (Goals / Findings / Open questions)
    b_progress/   markdown -> slide-deck pipeline for sharing with coworkers
    c_templates/  copy-paste starting points for new scripts
    d_archive/    old code

**Rules** (from the source article):
1. A script only loads data from `0_data/` or from an `out/` folder
   belonging to a script that runs **before** it. Never load from a later
   script, and never reach into another script's `store/`/`tmp/` (those
   are that script's own scratch space).
2. Never write into `0_data/`.
3. `1_scripts/utils/pipeline.py` (Python) and `1_scripts/utils/pipeline.R`
   (R) resolve each script's own `2_pipeline/<name>/` folder automatically
   from the script's file location -- call `get_pipeline_paths(__file__)`
   (Python) or source `pipeline.R` (see any script in `1_scripts/` for the
   exact one-line bootstrap) as the first thing a script does, rather than
   hand-maintaining an output path.

## Analysis pipeline

Each top-level step below corresponds to one numbered folder in
`1_scripts/` and one progress file in `a_docs/`; lettered sub-steps share
that folder. The protocols each sub-step applies to are noted in
parentheses.

**1 - Behavioral characterization**
- `1a_performance` -- raw-data analyses with no exclusions, plus
  engagement level (DM, DP)
- `1b_psychometric` -- psychometric fit, used to establish standards
  (DP, DN)
- `1c_behavior` -- behavioral variables (speed, position, pupil size,
  video) analyzed against each other (DM, DP, DN)
- `1d_performance_predictors` -- relationships between animal performance
  and any combination of behavioral variables, current or past trial
  (DM, DP, DN)

**2 - Single-cell characterization** (DN)
- `2a_cell_distribution` -- spatial organization of recorded cells across
  sessions
- `2b_activity_statistics` -- statistics of single-cell spiking activity
- `2c_information` -- mutual information between cell activity and
  behavioral variables / performance (stim/choice)
- `2d_linear_encod` -- linear regressions relating single-cell activity to
  behavioral variables / performance; compared against `2c_information` to
  check whether the relationships are already captured linearly
- `2e_nonlinear_encod` (maybe) -- if there's a large gap between total and
  linearly-explained mutual information for a given relationship,
  introduce nonlinear fits to describe it better

**3 - Pairwise-correlations characterization** (DN)

**4 - Single-area analysis** (DN)

**5 - Multi-area analysis** (DN)

## Shared code: `1_scripts/libs/`

`1_scripts/libs/` is the shared analysis library

## Docs: `a_docs/`

`project_proposal.md`, and one progress file per analysis step
(`1_behavior.md` .. `5_multi_area.md` -- Goals / Findings / Open
questions, matching the `1_scripts/` numbering above).

## Sharing progress: `b_progress/`   /   Templates: `c_templates/`  /  Old code: `d_archive/`

See each folder's own README/contents.
