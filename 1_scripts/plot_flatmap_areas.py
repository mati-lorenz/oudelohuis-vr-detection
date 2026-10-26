# -*- coding: utf-8 -*-
"""
plot_flatmap_areas.py
=====================
Flattened dorsal map of mouse isocortex with V1, PM, AL and RSP highlighted -
the style of figure used in Ciceri et al. 2024 and Steinmetz et al. 2019.

Unlike a horizontal slice, the flatmap unrolls the curved cortical sheet, so
lateral areas (AL) and midline areas (RSP) all appear at their true extent.

    pip install ccf-streamlines

Needs two small asset files (a few MB); the script downloads them on first run
into ./ccf_assets, or see https://ccf-streamlines.readthedocs.io/en/latest/data_files.html
to fetch them by hand. The large streamline .h5 files are NOT needed here,
because we only draw region boundaries, not projected data.

Usage
-----
    python plot_flatmap_areas.py
    python plot_flatmap_areas.py --list        # print available acronyms

Output:
    3_output/figures_lab_meeting/flatmap_areas.png / .svg
"""
import argparse
import urllib.request
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.patheffects as mpatheffects
import matplotlib.pyplot as plt
import ccf_streamlines.projection as ccfproj

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
ASSET_DIR = HERE / "ccf_assets"
OUT_DIR = HERE.parent / "3_output" / "figures_lab_meeting"

BASE_URL = ("https://download.alleninstitute.org/informatics-archive/"
            "current-release/mouse_ccf/cortical_coordinates/ccf_2017/"
            "ccf_streamlines_assets/master_updated/")
ASSETS = ["flatmap_dorsal.nrrd", "labelDescription_ITKSNAPColor.txt"]

# acronym -> (label on the figure, colour). Okabe-Ito, colour-blind safe.
# RSP is split into subdivisions in the atlas; all of them get the same colour.
AREAS = {
    "VISp":   ("V1",  "#0072B2"),
    "VISpm":  ("PM",  "#E69F00"),
    "VISal":  ("AL",  "#009E73"),
    "RSPagl": ("RSP", "#CC79A7"),
    "RSPd":   ("",    "#CC79A7"),
    "RSPv":   ("",    "#CC79A7"),
}

BACKDROP = "#9aa3b0"      # outline colour for all the other areas
BACKDROP_LW = 0.6
AREA_LW = 1.4


def fetch_assets():
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    for name in ASSETS:
        dest = ASSET_DIR / name
        if dest.exists():
            continue
        url = BASE_URL + name
        print(f"Downloading {name} ...")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as exc:
            raise SystemExit(
                f"Could not download {name} ({exc}).\n"
                f"Get it by hand from the Data Files page of the docs:\n"
                f"  https://ccf-streamlines.readthedocs.io/en/latest/data_files.html\n"
                f"and put it in {ASSET_DIR}"
            )
    return ASSET_DIR / ASSETS[0], ASSET_DIR / ASSETS[1]


def get_boundaries():
    atlas_file, labels_file = fetch_assets()
    bf = ccfproj.BoundaryFinder(
        projected_atlas_file=str(atlas_file),
        labels_file=str(labels_file),
    )
    return bf.region_boundaries()


def main(list_only=False, annotate=True):
    bounds = get_boundaries()

    if list_only:
        print(" ".join(sorted(bounds)))
        return

    missing = [a for a in AREAS if a not in bounds]
    if missing:
        print(f"  not in the atlas: {missing} (run --list to see the names)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 8), dpi=200)

    # every other area: thin grey outline, as a backdrop
    for acronym, coords in bounds.items():
        if acronym in AREAS:
            continue
        ax.plot(*coords.T, c=BACKDROP, lw=BACKDROP_LW, zorder=1)

    # our areas: filled and outlined
    for acronym, (label, colour) in AREAS.items():
        if acronym not in bounds:
            continue
        coords = bounds[acronym]
        ax.fill(*coords.T, color=colour, alpha=0.75, zorder=2)
        ax.plot(*coords.T, c="white", lw=AREA_LW, zorder=3)
        if annotate and label:
            cx, cy = coords[:, 0].mean(), coords[:, 1].mean()
            ax.text(cx, cy, label, ha="center", va="center", fontsize=11,
                    fontweight="bold", color="white", zorder=4,
                    path_effects=[mpatheffects.withStroke(
                        linewidth=2.5, foreground="#33383f")])

    ax.set_aspect("equal")
    ax.invert_yaxis()          # image convention: anterior ends up at the top
    ax.set_axis_off()

    handles = [mpatches.Patch(color=c, label=l)
               for l, c in dict(AREAS.values()).items() if l]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0),
              frameon=False, fontsize=11)

    fig.text(0.5, 0.02,
             "Flattened dorsal view of mouse isocortex (Allen CCFv3)",
             ha="center", fontsize=8, color="0.45")

    for ext in ("png", "svg"):
        out = OUT_DIR / f"flatmap_areas.{ext}"
        fig.savefig(out, bbox_inches="tight", facecolor="white")
        print(f"Saved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--list", action="store_true",
                   help="print the area acronyms available in the atlas")
    p.add_argument("--no-labels", dest="annotate", action="store_false")
    a = p.parse_args()
    main(list_only=a.list, annotate=a.annotate)
