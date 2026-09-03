# Minimal usage example for infotheory.plotting -- see save.py's TODOs for
# what to actually implement.

import matplotlib.pyplot as plt
from infotheory.plotting import apply_style, save_figure

apply_style()  # once per script, before any figure

fig, ax = plt.subplots(figsize=(5, 4))
# ... plot here ...
save_figure(fig, step_dir="area", name="dimensionality_by_group")
# -> writes area/figures/dimensionality_by_group.png AND .pdf
plt.close(fig)
