# Sharing progress with coworkers

Source of truth stays markdown: `a_docs/<step>.md` (Goals / Findings /
Open questions) per step. This folder turns that + `3_output/` figures
into something presentable.

## Render: pandoc -> reveal.js (live) or .pptx (email)

`3_output/` is flat and prefixed by step (e.g. `1_behavior_psychometric_
by_engagement.png`), so reference figures directly:

    # Weekly progress -- 2026-09-08

    ## Behavior
    ![](../3_output/1_behavior_psychometric_by_engagement.png)
    - Engaged trials reach P(lick)=1.0 by 100% signal; disengaged trials
      stay near 0.18 even at 100% -- confirms a_docs/1_behavior.md

Render:

    pandoc c_progress/archive/2026-09-08.md -o c_progress/archive/2026-09-08.html \
        -t revealjs -s -V theme=white --slide-level=2
    pandoc c_progress/archive/2026-09-08.md -o c_progress/archive/2026-09-08.pptx \
        --slide-level=2

`make_deck.py` drafts this automatically from `3_output/`'s prefixed files.
