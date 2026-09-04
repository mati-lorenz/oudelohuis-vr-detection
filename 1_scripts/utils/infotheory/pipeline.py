# -*- coding: utf-8 -*-
"""
Resolve a script's own 2_pipeline/<script_name>/ folder automatically
from the script's file location, instead of hand-maintaining paths.

Usage (first thing a script in 1_scripts/ does):

    from infotheory.pipeline import get_pipeline_paths
    paths = get_pipeline_paths(__file__)

    paths.out    # 2_pipeline/<script_name>/out/   -- end-products for a LATER script
    paths.store  # 2_pipeline/<script_name>/store/ -- this script's own cache
    paths.tmp    # 2_pipeline/<script_name>/tmp/   -- throwaway inspection files
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelinePaths:
    root: Path         # project root (parent of 1_scripts/, 2_pipeline/, ...)
    script_name: str   # e.g. "1a_performance" (from "1a_performance.py")
    out: Path
    store: Path
    tmp: Path

    def out_from(self, other_script_name: str) -> Path:
        """Convenience: the `out/` folder of a DIFFERENT (earlier) script,
        e.g. `paths.out_from("1a_performance")` from within 1b_psychometric.
        Per project rule #1, only read from here -- never write."""
        return self.root / "2_pipeline" / other_script_name / "out"


def find_project_root(start: Path) -> Path:
    """Walk up from `start` until a folder containing both `1_scripts/`
    and `2_pipeline/` is found. Public so tools that aren't themselves a
    numbered pipeline script (e.g. the progress-report generator in
    b_progress/) can still locate the project root."""
    for candidate in (start, *start.parents):
        if (candidate / "1_scripts").is_dir() and (candidate / "2_pipeline").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate the project root (a folder containing both "
        "'1_scripts/' and '2_pipeline/') by walking up from %s." % start
    )


def get_pipeline_paths(script_file: str) -> PipelinePaths:
    """Always call this with `__file__` from the calling script.

    Creates 2_pipeline/<script_name>/{out,store,tmp} if they don't exist
    yet and returns their paths.
    """
    script_path = Path(script_file).resolve()
    script_name = script_path.stem  # "1a_performance.py" -> "1a_performance"
    root = find_project_root(script_path.parent)

    base = root / "2_pipeline" / script_name
    out, store, tmp = base / "out", base / "store", base / "tmp"
    for folder in (out, store, tmp):
        folder.mkdir(parents=True, exist_ok=True)

    return PipelinePaths(root=root, script_name=script_name, out=out, store=store, tmp=tmp)
