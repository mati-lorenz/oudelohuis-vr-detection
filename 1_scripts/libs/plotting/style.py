# -*- coding: utf-8 -*-
"""
plotting/style.py
===================
One matplotlib style, applied once per script (`apply_style()` near the
top, before any figure is made), so every step's figures look consistent
without copy-pasting rcParams everywhere.

TODO(you): tune to taste, but keep it in ONE place. A few starting choices
that are almost always right for lab-meeting/paper figures:
- vector-friendly defaults (savefig.dpi high, but keep font sizes readable
  at slide scale, not just print scale)
- no top/right spines
- consistent area colors reused from celldata_utils.AREA_COLORS, so a
  reader learns "blue = V1" once and it holds across every figure in every
  step folder
"""

import matplotlib as mpl
from .. import celldata_utils  # reuse the SAME area color map everywhere


def apply_style():
    """Call once per script, before creating any figure.

    TODO: set mpl.rcParams for at least:
      - "figure.dpi" / "savefig.dpi" (150 on-screen, 300 for save.py's PDF path)
      - "font.size", "axes.titlesize", "axes.labelsize" (readable projected
        on a screen in a lab meeting, not just legible in a paper column)
      - "axes.spines.top" / "axes.spines.right" = False
      - "savefig.bbox" = "tight"
      - "font.family" (pin one so a machine without your preferred font
        doesn't silently substitute and shift layout, per the existing
        DejaVu Sans convention used across this project's scripts)
    """
    raise NotImplementedError
