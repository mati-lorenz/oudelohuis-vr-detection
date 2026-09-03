# -*- coding: utf-8 -*-
"""
c_progress/make_deck.py
=========================
Assembles a DRAFT progress deck by pulling the most recent N figures per
step-prefix from the flat `3_output/` folder (e.g. all `1_behavior_*.png`
under one "## Behavior" heading). Still a draft -- edit before rendering.

Usage:
    python c_progress/make_deck.py --date 2026-09-08 --n-figures 1
    python c_progress/make_deck.py --date 2026-09-08 --render html
"""

import argparse
import datetime
import os

STEP_PREFIXES = {
    "1_behavior": "Behavior", "2_single": "Single cell", "3_pairs": "Pairwise",
    "4_area": "Single-area populations", "5_multi_area": "Inter-area populations",
}
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def latest_figures(prefix, n=1):
    raise NotImplementedError  # TODO: n most-recently-modified 3_output/<prefix>_*.png files


def build_draft(date, n_figures):
    raise NotImplementedError  # TODO: write c_progress/archive/<date>.md, one "## <Step>" per prefix


def render(md_path, fmt):
    raise NotImplementedError  # TODO: subprocess call to the pandoc commands in c_progress/README.md


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=datetime.date.today().isoformat())
    parser.add_argument("--n-figures", type=int, default=1)
    parser.add_argument("--render", choices=["html", "pptx"], default=None)
    args = parser.parse_args()

    md_path = os.path.join(REPO_ROOT, "c_progress", "archive", f"{args.date}.md")
    build_draft(args.date, args.n_figures)
    print(f"Draft written to {md_path} -- review/edit before rendering.")
    if args.render:
        render(md_path, args.render)
