# -*- coding: utf-8 -*-
"""
plotting/tables.py
=====================
Two ways to make a pandas DataFrame presentable outside a Jupyter cell:
Markdown (cheap, diffable, drops straight into a step README or the
progress deck) and a rendered image (nicer for a slide, but needs the
`dataframe_image` package -- which itself needs a Chrome/Chromium install
or an orca/kaleido backend; test this early rather than at 5pm before a
lab meeting).

TODO(you):
- save_table_markdown(df, step_dir, name, float_format="%.3f") -> writes
  <step_dir>/tables/<name>.md via df.to_markdown() (needs `tabulate`,
  already in requirements.txt). This is the one to reach for by default --
  it's the same content a slide deck or README would want anyway.
- save_table_image(df, step_dir, name, styler_fn=None) -> writes
  <step_dir>/tables/<name>.png. `styler_fn` is an optional
  `df.style`-returning function so callers can add e.g.
  `.background_gradient()` / `.format()` before export via
  `dataframe_image.export(styled, path)`.
- Also save the RAW data as .csv alongside whichever presentation format
  you use -- the markdown/image is for humans, the CSV is for you six
  months from now.
"""


def save_table_markdown(df, step_dir, name, float_format="%.3f"):
    raise NotImplementedError


def save_table_image(df, step_dir, name, styler_fn=None):
    raise NotImplementedError
