# -*- coding: utf-8 -*-
"""
plot_visual_areas_heatmap.py
============================
Horizontal (top-down) slice of the left hemisphere with V1, PM, AL and RSP
coloured, drawn with brainglobe-heatmap.

    pip install brainglobe-heatmap

Usage
-----
    # sweep depths first to find a slice that shows all four areas
    python plot_visual_areas_heatmap.py --sweep

    # then render the one you picked
    python plot_visual_areas_heatmap.py --depth 1200

Output:
    3_output/figures_lab_meeting/visual_areas_heatmap*.png / .svg

Note: a horizontal slice is a FLAT plane through a CURVED cortex. Shallow
depths catch the midline areas (RSP, PM); lateral areas (AL) sit lower and
need a deeper slice. --sweep renders a grid of depths so you can pick.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import brainglobe_heatmap as bgh

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
ATLAS = "allen_mouse_25um"
OUT_DIR = Path(__file__).resolve().parent.parent / "3_output" / "figures_lab_meeting"

# acronym -> (slide label, colour). Okabe-Ito, colour-blind safe.
AREAS = {
    "VISp":  ("V1",  "#0072B2"),
    "VISpm": ("PM",  "#E69F00"),
    "VISal": ("AL",  "#009E73"),
    "RSP":   ("RSP", "#CC79A7"),
}

HEMISPHERE = "left"
THICKNESS = 100          # slab thickness in um
DEFAULT_DEPTH = 1200     # um below the top of the volume
SWEEP_DEPTHS = [150, 350, 600, 900, 1200, 1500]

# one integer per area, so the colormap is discrete
VALUES = {acr: i + 1 for i, acr in enumerate(AREAS)}
CMAP = ListedColormap([colour for _, colour in AREAS.values()])


def make_heatmap(depth, annotate=True):
    """Build the Heatmap object for one depth. The axis goes to .plot()."""
    return bgh.Heatmap(
        VALUES,
        position=depth,
        orientation="horizontal",
        thickness=THICKNESS,
        atlas_name=ATLAS,
        format="2D",
        cmap=CMAP,
        vmin=0.5,
        vmax=len(AREAS) + 0.5,
        hemisphere=HEMISPHERE,
        annotate_regions=annotate,
        title=f"{depth} \u00b5m",
    )


def legend_handles():
    import matplotlib.patches as mpatches
    return [mpatches.Patch(color=c, label=lbl) for lbl, c in AREAS.values()]


def sweep():
    """Render several depths side by side to choose from."""
    n = len(SWEEP_DEPTHS)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(4 * ((n + 1) // 2), 8),
                             dpi=150)
    for depth, ax in zip(SWEEP_DEPTHS, axes.ravel()):
        make_heatmap(depth, annotate=False).plot_subplot(
            fig=fig, ax=ax, show_legend=False)
        ax.set_title(f"{depth} \u00b5m", fontsize=11)
    for ax in axes.ravel()[n:]:
        ax.set_axis_off()
    fig.legend(handles=legend_handles(), loc="lower center", ncol=len(AREAS),
               frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = OUT_DIR / "visual_areas_heatmap_sweep.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}\nPick a depth, then rerun with --depth <value>.")


def single(depth):
    fig, ax = plt.subplots(figsize=(6.5, 6.5), dpi=200)
    make_heatmap(depth).plot_subplot(fig=fig, ax=ax, show_legend=False)
    ax.set_title(f"Left hemisphere, horizontal slice ({depth} \u00b5m)",
                 fontsize=12, fontweight="bold")
    ax.legend(handles=legend_handles(), loc="lower left",
              bbox_to_anchor=(1.0, 0.0), frameon=False, fontsize=11)
    for ext in ("png", "svg"):
        out = OUT_DIR / f"visual_areas_heatmap.{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        print(f"Saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", action="store_true",
                   help="render a grid of depths to choose from")
    p.add_argument("--depth", type=float, default=DEFAULT_DEPTH,
                   help="slice depth in um")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep() if args.sweep else single(args.depth)
