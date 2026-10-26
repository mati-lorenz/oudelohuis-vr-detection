# -*- coding: utf-8 -*-
"""
plot_visual_areas_brainrender.py
================================
Dorsal (top-down) 3D rendering of the left hemisphere with V1, PM, AL and RSP,
drawn with brainrender. Companion to plot_visual_areas_heatmap.py, for
comparing the two looks before choosing one for the slides.

Needs a working OpenGL context (run it from the host terminal, not the
Flatpak VS Code terminal).

    pip install brainrender

Usage
-----
    python plot_visual_areas_brainrender.py              # top view, saved
    python plot_visual_areas_brainrender.py --interactive # rotate it yourself
    python plot_visual_areas_brainrender.py --camera three_quarters

Output:
    3_output/figures_lab_meeting/visual_areas_brainrender*.png
"""
import argparse
from pathlib import Path

from brainrender import Scene, settings

ATLAS = "allen_mouse_25um"
OUT_DIR = Path(__file__).resolve().parent.parent / "3_output" / "figures_lab_meeting"

AREAS = {
    "VISp":  ("V1",  "#0072B2"),
    "VISpm": ("PM",  "#E69F00"),
    "VISal": ("AL",  "#009E73"),
    "RSP":   ("RSP", "#CC79A7"),
}

HEMISPHERE = "left"
ROOT_ALPHA = 0.12
AREA_ALPHA = 0.95
SCALE = 3          # screenshot resolution multiplier

# a tilted view often reads better than straight-down, since the areas curve
CAMERAS = {
    "top": "top",
    "three_quarters": dict(position=(-16000, -12000, 9000),
                           focal_point=(7000, 4000, 5700),
                           viewup=(0, -1, 0)),
    "sagittal": "sagittal",
}


def main(camera="top", interactive=False, zoom=1.5, labels=True):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    settings.SHOW_AXES = False
    settings.SHADER_STYLE = "cartoon"   # try "plastic" or "glossy" too
    settings.BACKGROUND_COLOR = "white"
    settings.OFFSCREEN = not interactive

    scene = Scene(atlas_name=ATLAS, inset=False,
                  screenshots_folder=str(OUT_DIR))
    scene.root.alpha(ROOT_ALPHA)

    for acronym, (label, colour) in AREAS.items():
        region = scene.add_brain_region(
            acronym, color=colour, alpha=AREA_ALPHA,
            hemisphere=HEMISPHERE, silhouette=True,
        )
        if labels and region is not None:
            try:
                scene.add_label(region, label, size=320, color="black")
            except Exception as exc:        # label placement is finicky
                print(f"  could not label {acronym}: {exc}")

    name = f"visual_areas_brainrender_{camera}"
    if interactive:
        scene.render(camera=CAMERAS[camera], zoom=zoom)
        print("Close the window when done; press 's' in it to screenshot.")
    else:
        scene.render(camera=CAMERAS[camera], zoom=zoom, interactive=False)
        out = scene.screenshot(name=name, scale=SCALE)
        print(f"Saved {out or OUT_DIR / (name + '.png')}")

    try:
        scene.close()
    except Exception:
        pass    # brainrender sometimes throws on teardown; the file is saved


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--camera", default="top", choices=list(CAMERAS))
    p.add_argument("--interactive", action="store_true",
                   help="open a window you can rotate instead of saving")
    p.add_argument("--zoom", type=float, default=1.5)
    p.add_argument("--no-labels", dest="labels", action="store_false")
    a = p.parse_args()
    main(camera=a.camera, interactive=a.interactive, zoom=a.zoom,
         labels=a.labels)
