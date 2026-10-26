# -*- coding: utf-8 -*-
"""
plot_labeling_strategy.py
=========================
Schematic of the dual retrograde viral labelling strategy, for the lab-meeting
slides: which virus went where, and which population each pair makes red.

Left panel  : dorsal atlas view with V1 and PM, injection sites and the two
              projection directions.
Right panel : the recombinase logic, step by step, for a non-specialist.

Needs plot_visual_areas_dorsal.py in the same folder (it reuses the atlas
helpers), so it also needs no OpenGL.

    python plot_labeling_strategy.py

Output:
    3_output/figures_lab_meeting/labeling_strategy.png / .svg

>>> CHECK THE CONFIG BELOW BEFORE USING THIS FIGURE. <<<
The retrograde virus goes in the TARGET area; the Cre/Flp-dependent reporter
goes in the SOURCE area, where the labelled cell bodies (the ones you image)
sit. Your description names each pair by one area only, so I set the direction
from the population each pair is said to label. Confirm against your surgery
notes and flip RETRO_IN / REPORTER_IN if needed.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from brainglobe_atlasapi import BrainGlobeAtlas

from plot_visual_areas_dorsal import (
    ATLAS, OUT_DIR, surface_labels, draw_dorsal, region_centroid_mm,
)

# ----------------------------------------------------------------------------
# Config: the two labelled populations
# ----------------------------------------------------------------------------
POPULATIONS = [
    dict(
        name="V1\u2192PM",
        kind="feedforward",
        colour="#0072B2",
        source="VISp",          # cell bodies live here; imaged here
        target="VISpm",         # axons end here
        retro_in="VISpm",       # retrograde recombinase injected here
        reporter_in="VISp",     # conditional reporter injected here
        retro="retroAAV-EF1a-Flpo",
        reporter="AAV1-EF1a-fDIO-tdTomato",
        n_mice=5,
    ),
    dict(
        name="PM\u2192V1",
        kind="feedback",
        colour="#E69F00",
        source="VISpm",
        target="VISp",
        retro_in="VISp",
        reporter_in="VISpm",
        retro="retroAAV-pgk-Cre",
        reporter="AAV5-CAG-Flex-tdTomato",
        n_mice=1,
    ),
]

AREA_COLOURS = {
    "VISp":  ("V1", "#0072B2"),
    "VISpm": ("PM", "#E69F00"),
}

CAPTION = (
    "GCaMP reports activity in all neurons; tdTomato marks projection identity.\n"
    "Virus-to-area pairing was counterbalanced across mice, but the groups are\n"
    "unbalanced: n=5 with Flp-tdTomato in V1, n=1 with Cre-tdTomato in V1."
)


# ----------------------------------------------------------------------------
# Left panel: anatomy
# ----------------------------------------------------------------------------
def draw_anatomy(ax, atlas, ids):
    draw_dorsal(ax, atlas, ids, areas=AREA_COLOURS, alpha=0.75)

    centres = {acr: region_centroid_mm(ids, atlas, acr) for acr in AREA_COLOURS}

    for pop in POPULATIONS:
        x0, y0 = centres[pop["source"]]
        x1, y1 = centres[pop["target"]]
        # bow the two arrows apart so they don't overlap
        rad = 0.35 if pop["kind"] == "feedforward" else -0.35
        ax.add_patch(FancyArrowPatch(
            (x0, y0), (x1, y1),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>", mutation_scale=16,
            lw=2, color=pop["colour"], zorder=5,
        ))

    # injection sites
    for pop in POPULATIONS:
        xr, yr = centres[pop["retro_in"]]
        ax.plot(xr, yr, marker="v", ms=11, color=pop["colour"],
                mec="white", mew=1.2, zorder=6)
        xp, yp = centres[pop["reporter_in"]]
        ax.plot(xp, yp, marker="o", ms=10, color=pop["colour"],
                mec="white", mew=1.2, zorder=6)

    legend = [
        plt.Line2D([], [], marker="v", ls="", color="0.3", ms=10,
                   label="retrograde recombinase (target area)"),
        plt.Line2D([], [], marker="o", ls="", color="0.3", ms=9,
                   label="conditional tdTomato (source area)"),
        plt.Line2D([], [], color=POPULATIONS[0]["colour"], lw=2,
                   label=f"{POPULATIONS[0]['name']} (feedforward)"),
        plt.Line2D([], [], color=POPULATIONS[1]["colour"], lw=2,
                   label=f"{POPULATIONS[1]['name']} (feedback)"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.02),
              frameon=False, fontsize=8.5)
    ax.set_title("Where the viruses go", fontsize=12, fontweight="bold", pad=10)


# ----------------------------------------------------------------------------
# Right panel: the logic, as boxes
# ----------------------------------------------------------------------------
def box(ax, x, y, w, h, text, colour, fontsize=8.5, text_colour="black"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.2, edgecolor=colour, facecolor=colour + "22",
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=text_colour, linespacing=1.4)


def arrow(ax, x0, y0, x1, y1, colour, text=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=13, lw=1.4, color=colour))
    if text:
        ax.text((x0 + x1) / 2 + 0.015, (y0 + y1) / 2, text, fontsize=7.5,
                color="0.3", ha="left", va="center")


def draw_logic(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_title("How one population gets labelled", fontsize=12,
                 fontweight="bold", pad=10)

    top = 0.94
    for pop in POPULATIONS:
        c = pop["colour"]
        y = top
        ax.text(0.0, y, f"{pop['name']}  ({pop['kind']}, n={pop['n_mice']})",
                fontsize=10, fontweight="bold", color=c, va="top")
        y -= 0.075

        box(ax, 0.0, y - 0.085, 1.0, 0.085,
            f"1. {pop['retro']} into "
            f"{AREA_COLOURS[pop['retro_in']][0]}", c)
        y -= 0.085
        arrow(ax, 0.5, y, 0.5, y - 0.045, "0.45",
              "travels backwards along axons")
        y -= 0.045

        box(ax, 0.0, y - 0.085, 1.0, 0.085,
            f"2. Recombinase now sits in "
            f"{AREA_COLOURS[pop['source']][0]} cells\n"
            f"that send an axon to "
            f"{AREA_COLOURS[pop['target']][0]}", c)
        y -= 0.085
        arrow(ax, 0.5, y, 0.5, y - 0.045, "0.45", "unlocks the reporter")
        y -= 0.045

        box(ax, 0.0, y - 0.085, 1.0, 0.085,
            f"3. {pop['reporter']} in "
            f"{AREA_COLOURS[pop['reporter_in']][0]}\n"
            f"\u2192 only those cells turn red", c)
        top = y - 0.085 - 0.075

    ax.text(0.0, 0.02, CAPTION, fontsize=7.5, color="0.35",
            va="bottom", linespacing=1.5)


# ----------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    atlas = BrainGlobeAtlas(ATLAS)
    ids = surface_labels(atlas)

    fig, axes = plt.subplots(1, 2, figsize=(11, 6.5), dpi=200,
                             gridspec_kw=dict(width_ratios=[1, 1.15]))
    draw_anatomy(axes[0], atlas, ids)
    draw_logic(axes[1])
    fig.subplots_adjust(wspace=0.05)

    for ext in ("png", "svg"):
        out = OUT_DIR / f"labeling_strategy.{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
