# -*- coding: utf-8 -*-
"""
Small shared plotting helpers so every script doesn't redefine its own
color scheme. Kept intentionally minimal for now -- extend as more
scripts need shared plot conventions (this mirrors the role of the old
`utils.plot_lib` referenced by the pre-existing behavior scripts).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

OUTCOME_COLORS = {"HIT": "#2ca02c", "MISS": "#ff7f0e", "FA": "#d62728", "CR": "#1f77b4"}
PROTOCOL_COLORS = {"DM": "#7570b3", "DP": "#1b9e77", "DN": "#d95f02"}
ENGAGED_COLORS = {True: "#2ca02c", False: "#999999"}


def get_clr_outcome(outcomes) -> list[str]:
    return [OUTCOME_COLORS.get(o, "#333333") for o in outcomes]


def get_clr_protocol(protocols) -> list[str]:
    return [PROTOCOL_COLORS.get(p, "#333333") for p in protocols]


def set_style():
    plt.rcParams.update({
        "figure.dpi": 110,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    })


def clear_figures(folder: Path, pattern: str = "*.png"):
    """Delete existing figures in `folder` before a fresh run writes new
    ones. Without this, a figure whose script-side condition changes
    (e.g. a protocol that no longer gets a population-overlay plot once
    it's excluded from fitting) leaves its old, now-stale PNG behind
    forever -- call this once per run, before any save_fig calls."""
    folder = Path(folder)
    if folder.is_dir():
        for f in folder.glob(pattern):
            f.unlink()


def save_fig(fig, folder: Path, name: str, dpi: int = 150):
    """Save a figure as `<folder>/<name>.png`, creating `folder` if needed."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
