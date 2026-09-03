# -*- coding: utf-8 -*-
"""
data/session.py
=================
The one data container every other module depends on. Keep this generic --
no lab-specific file formats or attribute names here, only what the
analysis code actually needs.

TODO(you): flesh out NeuralSession's fields to match what you actually
have; the fields below are a reasonable starting minimum based on what
this pipeline's stages use (celldata for area/label grouping, trialdata
for stim/choice/timing, calciumdata+ts_F for continuous traces, protocol
to dispatch time-locked vs. spatial response-window logic).
"""

from dataclasses import dataclass, field
from typing import Optional, Protocol
import numpy as np
import pandas as pd


@dataclass
class NeuralSession:
    session_id: str
    protocol: str
    celldata: pd.DataFrame                     # per-neuron metadata (area, label, depth, ...)
    trialdata: pd.DataFrame                    # per-trial metadata (stim, choice, timing, ...)
    calciumdata: Optional[np.ndarray] = None   # (T, N) continuous trace, lazily loaded
    ts_F: Optional[np.ndarray] = None          # imaging-frame timestamps, same length as calciumdata
    behavior: dict = field(default_factory=dict)  # name -> 1D array aligned to ts_F (position, runspeed, ...)
    respmat: Optional[np.ndarray] = None       # (N, K) filled in by data/session_utils.py
    tensor: Optional[np.ndarray] = None        # (K, N, T) filled in by data/tensor_utils.py


class SessionLoader(Protocol):
    """Implement this against YOUR raw data format in loaders.py. Nothing
    else in the package should need to change when you add a new dataset
    or lab that implements this protocol."""

    def list_sessions(self, protocol: str) -> list[str]:
        ...

    def load(self, session_id: str, **kwargs) -> NeuralSession:
        ...
