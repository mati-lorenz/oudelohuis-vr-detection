# -*- coding: utf-8 -*-
"""

====================
Partial information decomposition (Williams & Beer; I_min redundancy, shuffle-corrected, as 3c / 3f) of each pair about a
BEHAVIORAL target (runspeed, pupil_area, motionenergy, lick; trial mean in the window, RESPONSE_BINS quantile bins):
redundancy, unique1, unique2, synergy, with I1, I2, I12. Definitions and caveats are in 3c's docstring. Windows, pairs and
filters as 3i; cost per (pair, window, behavior) about 9 ms with PID_SHUFFLES = 30.

Outputs
-------
2_pipeline/3l_pair_behavior_pid/
    out/    behavior_pid_moments.csv, behavior_pid_by_group[__<aggregate>].csv, figures/ (as 3i)
    store/  per-session caches
"""
import numpy as np

from infotheory import project_config as cfg
from infotheory.pid import pid_from_binned_responses, REDUNDANCY_MEASURES
from infotheory.behavior_pair_runner import run

R_BINS = cfg.RESPONSE_BINS
PID_SHUFFLES = 1
TERMS = ["redundancy", "unique1", "unique2", "synergy"]
STORE_VALUES = TERMS + ["I12", "I1", "I2"]


def add_args(parser):
    parser.add_argument("--pid-shuffles", type=int, default=PID_SHUFFLES)
    parser.add_argument("--redundancy", default="imin", choices=sorted(REDUNDANCY_MEASURES))


def compute(X, y, I, J, rng, args):
    res = [pid_from_binned_responses(X[:, i], X[:, j], y, bins=R_BINS, redundancy_measure=args.redundancy,
                                     n_shuffles=args.pid_shuffles, rng=rng) for i, j in zip(I, J)]
    return {v: np.array([getattr(r, v) for r in res]) for v in STORE_VALUES}


run(__file__, stem="behavior_pid", store_name="behavior_pid", version="3l-v1", what="Pairwise PID",
    components=[("redundancy", "tab:purple"), ("unique1", "tab:blue"), ("unique2", "tab:cyan"), ("synergy", "tab:red")],
    total="I12", compare=["I12", "redundancy", "synergy"], ylabel="information (bits)", compute=compute,
    store_values=STORE_VALUES, add_args=add_args,
    key_extra=lambda a: dict(pid_shuffles=a.pid_shuffles, redundancy=a.redundancy))
