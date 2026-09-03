# -*- coding: utf-8 -*-
"""
plotting/save.py
==================
save_figure(): the ONE function every script should call instead of
fig.savefig(...) directly, so format/dpi/naming stay consistent across all
five step folders.

TODO(you):
- save_figure(fig, step_dir, name, formats=("png","pdf")) -> writes to
  <step_dir>/figures/<name>.<ext> for each format, creating the directory
  if needed. PNG for quick viewing/slides (see progress/ pipeline), PDF
  for anything that might end up in a paper figure (vector, not rasterized
  at a fixed DPI).
- Always bbox_inches="tight"; pick one DPI for PNG (300 is a safe default
  for a figure that might get zoomed in a slide deck) and don't silently
  vary it script-to-script.
- fig_path(step_dir, name, ext) -> the path-building logic alone, useful
  when something other than a matplotlib Figure needs the same convention
  (e.g. tables.py's image export below).
- Consider embedding the git commit hash or a run timestamp in a sidecar
  file (not the filename -- keep filenames stable for the progress-deck
  script to find "the latest X") if reproducibility tracking matters to
  you.
"""


def fig_path(step_dir, name, ext):
    raise NotImplementedError  # TODO: os.path.join(step_dir, "figures", f"{name}.{ext}")


def save_figure(fig, step_dir, name, formats=("png", "pdf"), dpi=300):
    raise NotImplementedError  # TODO: loop formats, fig.savefig(fig_path(...), dpi=dpi, bbox_inches="tight")
