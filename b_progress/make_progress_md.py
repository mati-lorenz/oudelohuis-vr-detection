# -*- coding: utf-8 -*-
"""
Build a markdown progress summary for one or more pipeline steps, from
their `2_pipeline/<step>/out/` tables and figures. Output lands right
here in b_progress/, ready for the markdown -> slide-deck pipeline.

Usage
-----
    python make_progress_md.py 1a_performance
    python make_progress_md.py 1a_performance 1b_psychometric
    python make_progress_md.py --all              # every step with an out/ folder
    python make_progress_md.py 1a_performance --include-tmp
"""
from __future__ import annotations

import argparse
from pathlib import Path

from infotheory.pipeline import find_project_root
from infotheory.reporting import build_progress_markdown


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="*", help="Script names, e.g. 1a_performance")
    parser.add_argument("--all", action="store_true",
                         help="Build for every step under 2_pipeline/ that has an out/ folder")
    parser.add_argument("--include-tmp", action="store_true",
                         help="Also pull figures/tables from the step's tmp/ folder")
    args = parser.parse_args()

    root = find_project_root(Path(__file__).resolve().parent)

    if args.all:
        steps = sorted(p.name for p in (root / "2_pipeline").iterdir()
                        if p.is_dir() and (p / "out").is_dir())
        if not steps:
            parser.error("No steps with an out/ folder found under 2_pipeline/")
    elif args.steps:
        steps = args.steps
    else:
        parser.error("Give one or more step names, or pass --all")

    for step in steps:
        try:
            path = build_progress_markdown(step, project_root=root, include_tmp=args.include_tmp)
            print(f"Wrote {path}")
        except FileNotFoundError as exc:
            print(f"Skipping '{step}': {exc}")


if __name__ == "__main__":
    main()
