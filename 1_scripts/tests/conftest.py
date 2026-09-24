import os
import sys

# Make `utils` importable when pytest is run from the project root or from
# 1_scripts/ itself.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(0)
