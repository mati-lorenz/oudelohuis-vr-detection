# -*- coding: utf-8 -*-
"""
plot_topdown_areas.py
=====================
Top-down (dorsal) view of mouse cortex in the style of Steinmetz et al. 2019
Fig. 1i: every cortical area as a thin grey outline, the areas of interest
filled in colour, and labels set outside the brain on leader lines.

Built the same way those figures were: by projecting the Allen CCFv3
annotation volume from above. No OpenGL, no 3D rendering.

    pip install brainglobe-atlasapi scipy scikit-image matplotlib

The first run downloads the Allen atlas (a few hundred MB) to ~/.brainglobe.

Usage
-----
    python plot_topdown_areas.py
    python plot_topdown_areas.py --hemisphere left --dpi 600

Output (3_output/figures_lab_meeting/):
    topdown_areas.png   raster at --dpi (default 600)
    topdown_areas.svg   vector, for scaling without limit
    topdown_areas.pdf   vector, for print
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from brainglobe_atlasapi import BrainGlobeAtlas
from scipy import ndimage

try:
    from skimage import measure
    HAVE_SKIMAGE = True
except ImportError:                       # fall back to matplotlib contours
    HAVE_SKIMAGE = False

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
ATLAS = "allen_mouse_25um"
OUT_DIR = Path(__file__).resolve().parent.parent / "3_output" / "figures_lab_meeting"

# acronym -> (label, colour). Okabe-Ito, colour-blind safe.
AREAS = {
    "VISp":  ("V1",  "#0072B2"),
    "VISpm": ("PM",  "#E69F00"),
    "VISal": ("AL",  "#009E73"),
    "RSP":   ("RSP", "#CC79A7"),
}

# cortical subdivisions to merge up into their parent, as the papers do
MERGE_UP = {"SSp", "RSP", "VISC", "AUD", "ACA", "PL", "ILA", "ORB", "MO"}

OUTLINE_COLOUR = "#9aa3b0"
OUTLINE_LW = 0.45
AREA_EDGE_LW = 0.9
SMOOTH_SIGMA = 1.2        # boundary smoothing; 0 gives raw voxel staircase
LABEL_FONTSIZE = 9
LEADER_COLOUR = "#4a5160"

LAYER_RE = re.compile(r"(1|2/3|4|5|6a|6b)$")


# ----------------------------------------------------------------------------
# Atlas -> surface label map
# ----------------------------------------------------------------------------
def surface_labels(atlas):
    """Structure id of the topmost annotated voxel in each (AP, ML) column."""
    ann = atlas.annotation
    filled = ann > 0
    first = np.argmax(filled, axis=1)
    ap = np.arange(ann.shape[0])[:, None]
    ml = np.arange(ann.shape[2])[None, :]
    ids = ann[ap, first, ml]
    ids[~filled.any(axis=1)] = 0
    return ids


def area_of(atlas, sid, isocortex_id):
    """Map a voxel's structure id up to the cortical area it belongs to.

    Returns None for anything outside isocortex. Strips cortical layers
    (VISp1 -> VISp) and merges the subdivisions listed in MERGE_UP
    (SSp-bfd -> SSp, RSPagl -> RSP).
    """
    try:
        struct = atlas.structures[int(sid)]
    except KeyError:
        return None
    path = struct["structure_id_path"]
    if isocortex_id not in path:
        return None

    # walk from the deepest ancestor outwards, skipping layers
    for anc_id in reversed(path):
        acr = atlas.structures[anc_id]["acronym"]
        if LAYER_RE.search(acr):
            continue
        if anc_id == isocortex_id:
            return None
        # merge subdivisions into the parent when asked
        for parent in MERGE_UP:
            if acr != parent and acr.startswith(parent):
                return parent
        return acr
    return None


def build_area_map(atlas, ids):
    """(area-name array, set of names present)."""
    isocortex_id = atlas.structures["Isocortex"]["id"]
    lookup = {sid: area_of(atlas, sid, isocortex_id) for sid in np.unique(ids)}
    names = sorted({v for v in lookup.values() if v})
    index = {name: i + 1 for i, name in enumerate(names)}
    out = np.zeros_like(ids, dtype=np.int32)
    for sid, name in lookup.items():
        if name:
            out[ids == sid] = index[name]
    return out, names, index


# ----------------------------------------------------------------------------
# Boundaries
# ----------------------------------------------------------------------------
def contours_of(mask, sigma=SMOOTH_SIGMA):
    """Smoothed outline polylines of a binary mask, in (row, col) coords."""
    m = mask.astype(float)
    if sigma:
        m = ndimage.gaussian_filter(m, sigma)
    if HAVE_SKIMAGE:
        return measure.find_contours(m, 0.5)
    # fallback: matplotlib's contour generator
    fig = plt.figure()
    cs = plt.contour(m, levels=[0.5])
    polys = [p.vertices[:, ::-1] for coll in cs.allsegs for p in [] ] or \
            [np.column_stack((seg[:, 1], seg[:, 0]))
             for seg in cs.allsegs[0]]
    plt.close(fig)
    return polys


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
def main(hemisphere="both", dpi=600, sigma=SMOOTH_SIGMA):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    atlas = BrainGlobeAtlas(ATLAS)
    res_ap, _, res_ml = atlas.resolution

    ids = surface_labels(atlas)
    area_map, names, index = build_area_map(atlas, ids)

    missing = [a for a in AREAS if a not in index]
    if missing:
        print(f"  not found on the dorsal surface: {missing}")
        print(f"  available: {' '.join(names)}")

    # restrict to one hemisphere if asked
    mid = area_map.shape[1] // 2
    if hemisphere == "left":
        area_map[:, mid:] = 0
    elif hemisphere == "right":
        area_map[:, :mid] = 0

    # crop to the cortex, then work in mm
    rows, cols = np.where(area_map > 0)
    pad = 12
    rsl = slice(max(rows.min() - pad, 0), rows.max() + pad)
    csl = slice(max(cols.min() - pad, 0), cols.max() + pad)
    area_map = area_map[rsl, csl]

    def to_mm(rc):
        """(row, col) array -> (x_mm, y_mm) for plotting."""
        return np.column_stack((rc[:, 1] * res_ml / 1000.0,
                                rc[:, 0] * res_ap / 1000.0))

    fig, ax = plt.subplots(figsize=(7.0, 8.5), dpi=dpi)

    # --- every cortical area: thin grey outline --------------------------
    for name, i in index.items():
        if name in AREAS:
            continue
        mask = area_map == i
        if mask.sum() < 20:
            continue
        for c in contours_of(mask, sigma):
            xy = to_mm(c)
            ax.plot(xy[:, 0], xy[:, 1], c=OUTLINE_COLOUR, lw=OUTLINE_LW,
                    zorder=2, solid_joinstyle="round")

    # --- the whole cortex: one heavier silhouette ------------------------
    for c in contours_of(area_map > 0, sigma):
        xy = to_mm(c)
        ax.plot(xy[:, 0], xy[:, 1], c="#4a5160", lw=1.1, zorder=3)

    # --- areas of interest: filled ---------------------------------------
    centroids = {}
    for acronym, (label, colour) in AREAS.items():
        if acronym not in index:
            continue
        mask = area_map == index[acronym]
        for c in contours_of(mask, sigma):
            xy = to_mm(c)
            ax.fill(xy[:, 0], xy[:, 1], color=colour, alpha=0.9, zorder=4,
                    linewidth=0)
            ax.plot(xy[:, 0], xy[:, 1], c="white", lw=AREA_EDGE_LW, zorder=5)
        r, c_ = np.where(mask)
        centroids[acronym] = (c_.mean() * res_ml / 1000.0,
                              r.mean() * res_ap / 1000.0)

    ax.set_aspect("equal")
    ax.invert_yaxis()                 # anterior at the top
    ax.set_axis_off()

    # --- labels outside the brain, on leader lines -----------------------
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()        # note: inverted, so ymin > ymax
    label_x = xmin - 0.12 * (xmax - xmin)

    ordered = sorted(centroids.items(), key=lambda kv: kv[1][1])
    n = len(ordered)
    if n:
        top, bottom = ymax + 0.18 * (ymin - ymax), ymin - 0.18 * (ymin - ymax)
        ys = np.linspace(top, bottom, n) if n > 1 else [np.mean([top, bottom])]

    for (acronym, (cx, cy)), ly in zip(ordered, ys):
        label, colour = AREAS[acronym]
        ax.annotate(
            label,
            xy=(cx, cy), xycoords="data",
            xytext=(label_x, ly), textcoords="data",
            ha="right", va="center",
            fontsize=LABEL_FONTSIZE, fontweight="bold", color=colour,
            annotation_clip=False, zorder=6,
            arrowprops=dict(arrowstyle="-", color=LEADER_COLOUR, lw=0.7,
                            shrinkA=2, shrinkB=2,
                            connectionstyle="arc3,rad=0.0"),
        )

    # --- 1 mm scale bar ---------------------------------------------------
    bx = xmax - 0.22 * (xmax - xmin)
    by = ymin - 0.04 * (ymin - ymax)
    ax.plot([bx, bx + 1], [by, by], color="#3a4150", lw=2.0,
            solid_capstyle="butt", clip_on=False, zorder=6)
    ax.text(bx + 0.5, by - 0.12 * (ymin - ymax) * 0.4, "1 mm", ha="center",
            va="top", fontsize=8, color="#3a4150", clip_on=False)

    fig.subplots_adjust(left=0.18, right=0.98, top=0.98, bottom=0.04)

    base = OUT_DIR / "topdown_areas"
    fig.savefig(f"{base}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{base}.svg", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{base}.pdf", bbox_inches="tight", facecolor="white")
    w, h = fig.get_size_inches()
    print(f"Saved {base}.png  ({int(w * dpi)} x {int(h * dpi)} px at {dpi} dpi)")
    print(f"Saved {base}.svg and {base}.pdf (vector)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hemisphere", default="both",
                   choices=["both", "left", "right"])
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--smooth", type=float, default=SMOOTH_SIGMA,
                   help="boundary smoothing; 0 keeps the raw voxel edges")
    a = p.parse_args()
    main(hemisphere=a.hemisphere, dpi=a.dpi, sigma=a.smooth)
