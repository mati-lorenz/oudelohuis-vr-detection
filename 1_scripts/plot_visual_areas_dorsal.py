# -*- coding: utf-8 -*-
"""
plot_visual_areas_dorsal.py
===========================
Dorsal (top-down) view of the mouse cortex from the Allen atlas (CCFv3),
with V1, PM, AL and RSP highlighted.

Pure numpy + matplotlib: it projects the atlas annotation volume from above
instead of rendering it in 3D, so it needs no OpenGL, no GPU and no display.

Install once:
    pip install brainglobe-atlasapi matplotlib

The first run downloads the Allen atlas (a few hundred MB) to ~/.brainglobe.

Output:
    3_output/figures_lab_meeting/visual_areas_dorsal.png / .svg

Also importable: surface_labels() and region_mask() are reused by
plot_labeling_strategy.py.
"""
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patheffects as mpatheffects
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from brainglobe_atlasapi import BrainGlobeAtlas

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
ATLAS = "allen_mouse_25um"
OUT_DIR = Path(__file__).resolve().parent.parent / "3_output" / "figures_lab_meeting"

# Allen acronym -> (label on the slide, colour). Okabe-Ito, colour-blind safe.
AREAS = {
    "VISp":  ("V1",  "#0072B2"),
    "VISpm": ("PM",  "#E69F00"),
    "VISal": ("AL",  "#009E73"),
    "RSP":   ("RSP", "#CC79A7"),
}

CROP_TO_CORTEX = True   # trim the empty margins around the brain
AREA_ALPHA = 0.85


# ----------------------------------------------------------------------------
# Atlas helpers
# ----------------------------------------------------------------------------
def surface_labels(atlas):
    """Structure id of the topmost annotated voxel in each (AP, ML) column.

    The allen_mouse atlas is stored 'asr': axis 0 anterior->posterior,
    axis 1 superior->inferior, axis 2 left->right. Taking the first non-zero
    voxel down each column gives the dorsal surface as seen from above.
    """
    ann = atlas.annotation
    filled = ann > 0
    first = np.argmax(filled, axis=1)                   # (AP, ML)
    ap = np.arange(ann.shape[0])[:, None]
    ml = np.arange(ann.shape[2])[None, :]
    ids = ann[ap, first, ml]
    ids[~filled.any(axis=1)] = 0                        # columns with no brain
    return ids


def region_mask(ids, atlas, acronym):
    """Boolean mask of a region and all of its descendants in a label map."""
    target = atlas.structures[acronym]["id"]
    keep = []
    for sid in np.unique(ids):
        if sid == 0:
            continue
        try:
            path = atlas.structures[int(sid)]["structure_id_path"]
        except KeyError:
            continue
        if target in path:
            keep.append(sid)
    return np.isin(ids, keep)


def dorsal_extent(atlas, ids):
    """Image extent in mm, and the crop slices used."""
    res_ap, _, res_ml = atlas.resolution
    brain = ids > 0
    if CROP_TO_CORTEX:
        rows, cols = np.where(brain)
        pad = 10
        rsl = slice(max(rows.min() - pad, 0), rows.max() + pad)
        csl = slice(max(cols.min() - pad, 0), cols.max() + pad)
    else:
        rsl, csl = slice(None), slice(None)
    extent = (
        csl.start * res_ml / 1000 if csl.start else 0,
        (csl.stop if csl.stop else ids.shape[1]) * res_ml / 1000,
        (rsl.stop if rsl.stop else ids.shape[0]) * res_ap / 1000,
        rsl.start * res_ap / 1000 if rsl.start else 0,
    )
    return extent, rsl, csl


def draw_dorsal(ax, atlas, ids, areas=AREAS, alpha=AREA_ALPHA, label_areas=True):
    """Draw the brain outline and the filled regions. Returns the extent."""
    extent, rsl, csl = dorsal_extent(atlas, ids)
    ids = ids[rsl, csl]
    brain = ids > 0

    # whole-brain silhouette
    ax.imshow(np.where(brain, 1.0, np.nan), cmap="gray_r", vmin=0, vmax=3,
              extent=extent, interpolation="nearest")
    ax.contour(brain, levels=[0.5], colors="0.35", linewidths=1.0,
               extent=extent, origin="upper")

    for acronym, (label, colour) in areas.items():
        m = region_mask(ids, atlas, acronym)
        if not m.any():
            print(f"  warning: {acronym} not visible from the dorsal surface")
            continue
        rgba = np.zeros(m.shape + (4,))
        rgba[m] = list(mcolors.to_rgb(colour)) + [alpha]
        ax.imshow(rgba, extent=extent, interpolation="nearest")
        ax.contour(m, levels=[0.5], colors="white", linewidths=0.8,
                   extent=extent, origin="upper")
        if label_areas:
            rows, cols = np.where(m)
            y = extent[3] + (rows.mean() / m.shape[0]) * (extent[2] - extent[3])
            x = extent[0] + (cols.mean() / m.shape[1]) * (extent[1] - extent[0])
            ax.text(x, y, label, ha="center", va="center", fontsize=9,
                    fontweight="bold", color="white",
                    path_effects=[mpatheffects.withStroke(
                        linewidth=2, foreground="0.25")])

    ax.set_aspect("equal")
    ax.set_axis_off()
    return extent


def region_centroid_mm(ids, atlas, acronym, hemisphere="right"):
    """Centre of a region on the dorsal surface, in mm, for placing markers."""
    extent, rsl, csl = dorsal_extent(atlas, ids)
    m = region_mask(ids[rsl, csl], atlas, acronym)
    rows, cols = np.where(m)
    mid = m.shape[1] / 2
    sel = cols > mid if hemisphere == "right" else cols < mid
    if sel.any():
        rows, cols = rows[sel], cols[sel]
    y = extent[3] + (rows.mean() / m.shape[0]) * (extent[2] - extent[3])
    x = extent[0] + (cols.mean() / m.shape[1]) * (extent[1] - extent[0])
    return x, y


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    atlas = BrainGlobeAtlas(ATLAS)
    ids = surface_labels(atlas)

    fig, ax = plt.subplots(figsize=(5.5, 6.5), dpi=200)
    extent = draw_dorsal(ax, atlas, ids)

    handles = [mpatches.Patch(color=c, label=lbl) for lbl, c in AREAS.values()]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(1.0, 0.0),
              frameon=False, fontsize=11)

    ax.annotate("", xy=(0.5, 1.0), xytext=(0.5, 0.92),
                xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="0.4", lw=1.2))
    ax.text(0.5, 1.01, "anterior", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=9, color="0.4")

    # 1 mm scale bar
    x0 = extent[0] + 0.1 * (extent[1] - extent[0])
    y0 = extent[2] - 0.06 * (extent[2] - extent[3])
    ax.plot([x0, x0 + 1], [y0, y0], color="0.2", lw=2.5,
            solid_capstyle="butt", clip_on=False)
    ax.text(x0 + 0.5, y0 - 0.15, "1 mm", ha="center", va="bottom",
            fontsize=9, color="0.2", clip_on=False)

    for ext in ("png", "svg"):
        out = OUT_DIR / f"visual_areas_dorsal.{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
