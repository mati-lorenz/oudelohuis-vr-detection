# Pattern for "is this real" claims -- see docs/null_hypotheses_framework.md
# for choosing WHICH null actually targets your confound; this is just the
# shape of the comparison, not a specific null.

import numpy as np
from infotheory import nulls  # TODO: pick the right null constructor

rng = np.random.default_rng(0)
n_null_repeats = 200

observed_statistic = None  # TODO: your real statistic

null_values = np.array([
    None  # TODO: recompute the SAME statistic on nulls.<something>(data, rng)
    for _ in range(n_null_repeats)
])

p_value = (1 + np.sum(null_values >= observed_statistic)) / (1 + n_null_repeats)
# ^ the +1/+1 finite-sample correction -- never report p == 0
