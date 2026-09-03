# -*- coding: utf-8 -*-
"""
utils
===========
Public API. Re-export the names scripts should import as
`from infotheory import X` rather than reaching into submodules directly,
so the internal layout can change without breaking every script.

TODO(you): as you implement each module, add its public names here. Keep
this list in sync with what scripts/*.py actually import -- it's the
contract between the package and the scripts.
"""

from .params import InfoTheoryParams, BinningParams, BiasCorrectionParams, ParallelParams
from .cache import cache_path, load_cache, save_cache, load_or_compute

# TODO: from .core import ...
# TODO: from .discretize import ...
# TODO: from .celldata_utils import ...
# TODO: from .singlecell import run_session_single_cell_analysis
# TODO: from .dimensionality import ...
# TODO: from .population.gpfa import ...
# TODO: from .population.jpca import ...
# TODO: from .population.tdr import ...
# TODO: from .population.information_dimred import ...
# TODO: from .data.session_utils import compute_respmat_for_session
# TODO: from .data.tensor_utils import compute_tensor_for_session
# TODO: from .data.behavior_signals import get_behavior_trace
# TODO: from .behavior_mi import compute_behavior_mi

from .pipeline import get_pipeline_paths, project_root
