"""Shared figure/table styling and saving, used by every step's scripts so
output looks consistent without each script reinventing it. See style.py,
save.py, tables.py."""

from .style import apply_style
from .save import save_figure, fig_path
from .tables import save_table_markdown, save_table_image
