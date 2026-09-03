# -*- coding: utf-8 -*-
"""
pipeline.py
============
Implements the folder convention from Ties de Kok, "How to keep your
research projects organized, part 1: folder structure" (see this repo's
top-level README.md). Each script in 1_scripts/ gets its own pipeline
sub-folder (name-matched) under 2_pipeline/:

    out/   end-products meant to be loaded by a LATER script
    store/ intermediate results cached for reloading by THIS SAME script
    tmp/   throwaway inspection files, safe to delete anytime

`get_pipeline_paths(__file__)` derives both the script's name and its
project root from the script's own file path -- call it as the first line
of any script living directly in `1_scripts/` and it resolves
`2_pipeline/<script_name>/` automatically, creating it if missing.

NOTE: this assumes scripts live directly in `1_scripts/` (one level below
the project root) -- i.e. `<root>/1_scripts/x.py`, so `x.py`'s parent's
parent is `<root>`. If you ever nest scripts deeper, this needs a matching
depth change (see the R version, pipeline.R, for the equivalent).
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PipelinePaths:
    out: Path
    store: Path
    tmp: Path


def get_pipeline_paths(script_file, subdirs=("out", "store", "tmp")):
    script_path = Path(script_file).resolve()
    name = script_path.stem                # e.g. "1_behavior"
    root = script_path.parent.parent        # <root>/1_scripts/x.py -> <root>/
    pipeline_dir = root / "2_pipeline" / name

    paths = {}
    for sub in subdirs:
        p = pipeline_dir / sub
        p.mkdir(parents=True, exist_ok=True)
        paths[sub] = p

    if subdirs == ("out", "store", "tmp"):
        return PipelinePaths(out=paths["out"], store=paths["store"], tmp=paths["tmp"])
    return paths


def project_root(script_file, marker="pyproject.toml"):
    """Walk up from script_file until `marker` is found. Depth-agnostic
    (loop-based), unlike get_pipeline_paths -- use this whenever you need
    the true project root (e.g. to reach 0_data/)."""
    p = Path(script_file).resolve().parent
    while p != p.parent:
        if (p / marker).exists():
            return p
        p = p.parent
    raise FileNotFoundError(f"Could not find {marker} walking up from {script_file}")
