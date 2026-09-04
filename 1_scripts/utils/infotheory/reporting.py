# -*- coding: utf-8 -*-
"""
Turn a pipeline step's `2_pipeline/<script_name>/out/` folder (tables +
figures) into one markdown file, written to `b_progress/<script_name>.md`.

This is meant as the FIRST stage feeding the existing markdown ->
slide-deck pipeline in b_progress/ -- it doesn't replace that, it just
saves hand-copying tables and figures into a deck every time a script's
output changes. Regenerate it whenever the script's `out/` changes;
don't hand-edit the generated .md (it says so at the top too).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline import find_project_root


def _format_cell(value) -> str:
    """Plain, compact formatting for one table cell -- avoids the raw
    float noise (`0.731428571429`) a straight str(df) would produce."""
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        return f"{value:.4g}"
    if pd.isna(value):
        return ""
    return str(value).replace("|", "\\|")  # escape pipes so they don't break the table


def _dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    """Hand-rolled pipe-table renderer -- no dependency on the optional
    `tabulate` package that `DataFrame.to_markdown()` needs, so a report
    never silently degrades to a code block just because that package
    isn't installed in a given environment."""
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(_format_cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, separator, *rows])


def _df_to_markdown(df: pd.DataFrame, max_rows: int = 30, max_cols: int = 15) -> str:
    """Render a dataframe as a markdown table, truncating long/wide
    tables so the summary stays readable (the full CSV is still linked
    below it)."""
    notes = []

    if len(df.columns) > max_cols:
        shown_cols = df.columns[:max_cols]
        notes.append(f"showing first {max_cols} of {len(df.columns)} columns")
    else:
        shown_cols = df.columns

    shown = df[shown_cols]
    if len(shown) > max_rows:
        shown = shown.head(max_rows)
        notes.append(f"showing first {max_rows} of {len(df)} rows")

    table = _dataframe_to_markdown_table(shown)
    if notes:
        table += f"\n\n*({'; '.join(notes)} -- see the linked CSV for the rest)*"
    return table


def _title_from_filename(path: Path) -> str:
    """'3_performance_trajectory.png' -> 'Performance trajectory'."""
    parts = path.stem.split("_")
    if parts and parts[0].isdigit():
        parts = parts[1:]
    title = " ".join(parts).strip()
    return title[:1].upper() + title[1:] if title else path.stem


def build_progress_markdown(script_name: str, project_root: Path | None = None,
                             include_tmp: bool = False) -> Path:
    """Gather every CSV and PNG under `2_pipeline/<script_name>/out/`
    (optionally also its `tmp/`) into one markdown report.

    Parameters
    ----------
    script_name : e.g. "1a_performance"
    project_root : pass explicitly if not calling this from inside the
        project tree (e.g. from a notebook); otherwise auto-detected.
    include_tmp : also pull in throwaway inspection figures/tables from
        `tmp/`, not just the hand-off `out/` folder.
    """
    root = project_root or find_project_root(Path.cwd())
    step_dir = root / "2_pipeline" / script_name
    out_dir = step_dir / "out"
    if not out_dir.is_dir():
        raise FileNotFoundError(f"No out/ folder for step '{script_name}' at {out_dir}")

    search_dirs = [out_dir] + ([step_dir / "tmp"] if include_tmp else [])
    csv_paths = sorted(p for d in search_dirs if d.is_dir() for p in d.rglob("*.csv"))
    png_paths = sorted(p for d in search_dirs if d.is_dir() for p in d.rglob("*.png"))

    progress_dir = root / "b_progress"
    progress_dir.mkdir(parents=True, exist_ok=True)
    report_path = progress_dir / f"{script_name}.md"

    lines = [
        f"# {script_name}",
        "",
        f"*Auto-generated from `2_pipeline/{script_name}/out/` on "
        f"{dt.datetime.now():%Y-%m-%d %H:%M}. Re-run `make_progress_md.py` "
        "after the script's output changes -- don't hand-edit this file.*",
    ]

    if csv_paths:
        lines += ["", "## Tables"]
        for csv_path in csv_paths:
            rel = Path("..") / csv_path.relative_to(root)
            lines += ["", f"### {_title_from_filename(csv_path)}", ""]
            try:
                df = pd.read_csv(csv_path)
                lines.append(_df_to_markdown(df) if len(df) else "*(empty -- no rows)*")
            except pd.errors.EmptyDataError:
                lines.append("*(empty -- no rows)*")
            except Exception as exc:
                lines.append(f"*(could not read {csv_path.name}: {exc})*")
            lines += ["", f"[{csv_path.name}]({rel.as_posix()})"]

    if png_paths:
        lines += ["", "## Figures"]
        for png_path in png_paths:
            rel = Path("..") / png_path.relative_to(root)
            lines += ["", f"### {_title_from_filename(png_path)}", "",
                      f"![{png_path.stem}]({rel.as_posix()})"]

    if not csv_paths and not png_paths:
        lines += ["", "*(out/ is empty -- nothing to report yet.)*"]

    report_path.write_text("\n".join(lines) + "\n")
    return report_path
