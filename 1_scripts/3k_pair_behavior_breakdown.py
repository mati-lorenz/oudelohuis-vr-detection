# -*- coding: utf-8 -*-
"""
3k_pair_behavior_breakdown
==========================
Pola, Thiele, Hoffmann & Panzeri (2003) information breakdown of each pair about a BEHAVIORAL target (runspeed, pupil_area,
motionenergy, lick; trial mean in the window, RESPONSE_BINS quantile bins). Counterpart of 3b / 3e; definitions, signs, bias
treatment (I_cordep not floored) are those of 3b (formulas in infotheory/pair_information.py):
    I_total = I_lin + I_sigsim + I_cor,   I_cor = I_corind + I_cordep,   RSI = I_total - I_lin
    I_sigsim <= 0 (similar tuning to the behavior = redundancy), I_cordep unfloored (may be slightly negative after bias correction).
Reading it with a behavioral target: a pair whose cells both follow running speed has a strongly negative I_sigsim; I_cor
tells whether trial-to-trial co-fluctuations that are NOT explained by the behavior bin still carry information about it.
Windows, pairs, filters: as 3i. I_total / I_lin are identical to 3j's (same pairs, windows, rng seed and order of MI calls).

Outputs
-------
2_pipeline/3k_pair_behavior_breakdown/
    out/    behavior_breakdown_moments.csv, behavior_breakdown_by_group[__<aggregate>].csv, figures/ (as 3i)
    store/  per-session caches
"""
import numpy as np

from infotheory import project_config as cfg
from infotheory.pair_information import compute_pair_breakdown
from infotheory.behavior_pair_runner import run

R_BINS = cfg.RESPONSE_BINS
MI_SHUFFLES = 1
STORE_VALUES = ["I_total", "I_lin", "I_sigsim", "I_cor", "RSI", "I_corind", "I_cordep", "I_cordep_raw"]


def add_args(parser):
    parser.add_argument("--mi-shuffles", type=int, default=MI_SHUFFLES)


def compute(X, y, I, J, rng, args):
    res = [compute_pair_breakdown(X[:, i], X[:, j], y, r_bins=R_BINS, mi_shuffles=args.mi_shuffles, rng=rng)
           for i, j in zip(I, J)]
    # I_cordep_raw only exists in the current pair_information.py; NaN (left out of the means) with an older copy
    return {v: np.array([r.get(v, np.nan) for r in res]) for v in STORE_VALUES}


run(__file__, stem="behavior_breakdown", store_name="behavior_breakdown", version="3k-v2",
    what="Pairwise information breakdown",
    components=[("I_lin", "tab:blue"), ("I_sigsim", "tab:orange"), ("I_cor", "tab:green")], total="I_total",
    compare=["I_total", "RSI", "I_cor"], ylabel="information (bits)", compute=compute, store_values=STORE_VALUES,
    add_args=add_args, key_extra=lambda a: dict(mi_shuffles=a.mi_shuffles))
