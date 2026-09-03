# -*- coding: utf-8 -*-
"""
<N>_<short_name>.py
=====================
<one-line description>

Copy into 1_scripts/, keep it directly there (not in a subfolder --
get_pipeline_paths relies on this). Only load from 0_data/ or an EARLIER
script's `2_pipeline/<name>/out/`.
"""

#%%
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from infotheory import InfoTheoryParams, get_pipeline_paths, project_root
from infotheory.plotting import apply_style, save_figure, save_table_markdown
from infotheory.data.loaders import YourLabSessionLoader  # TODO

apply_style()
paths = get_pipeline_paths(__file__)   # -> paths.out / paths.store / paths.tmp
root = project_root(__file__)           # -> repo root, e.g. for 0_data/

#%% Configuration
# TODO

#%% Load + compute
# TODO

#%% Save: end-products -> paths.out; promote the good ones to 3_output/ by hand
# TODO
