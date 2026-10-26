# -*- coding: utf-8 -*-
"""
3j_pair_behavior_information
============================
Pair INFORMATION about a behavioral target (runspeed, pupil_area, motionenergy, lick; trial mean in the window, RESPONSE_BINS
quantile bins; windows, pairs and filters as 3i, see infotheory/behavior_pairs.py). Counterpart of 3b / 3e without the
decomposition, in the quantities one asks first:
    I_1, I_2    information of each cell alone (shuffle-corrected MI, histogram estimator, RESPONSE_BINS response bins)
    I_max       the better of the two cells (max of I_1, I_2 for each pair)
    I_total     information in the pair's JOINT response (r1, r2)
    I_lin       I_1 + I_2
    gain        I_total - I_max: what the second cell adds to the better one (> 0 the pair knows more than its best cell)
    RSI         I_total - I_lin: > 0 net synergy, < 0 net redundancy (Schneidman et al. 2003)
The estimators and the order of the MI calls are those of `pair_information.compute_pair_breakdown`, and the rng is seeded from
the session id, so I_total / I_1 / I_2 / I_lin are IDENTICAL to 3k's (same pairs, windows, behaviors): 3k only adds the Pola
terms. This script is the cheap one (no plug-in terms) -- use --mi-shuffles higher here if you want a better bias correction,
but then the equality with 3k is lost.

Project-wide rules: as 3i (selection, aggregate, memory, binning).

Outputs
-------
2_pipeline/3j_pair_behavior_information/
    out/    behavior_information_moments.csv, behavior_information_by_group[__<aggregate>].csv, figures/ (as 3i)
    store/  per-session caches
"""
import numpy as np

from infotheory import project_config as cfg
from infotheory.info_theory import mutual_information_shuffle
from infotheory.pair_information import digitize_quantile, joint_code
from infotheory.behavior_pair_runner import run

R_BINS = cfg.RESPONSE_BINS
MI_SHUFFLES = 1
VALUES = ["I_total", "I_1", "I_2", "I_lin", "I_max", "gain", "RSI"]


def add_args(parser):
    parser.add_argument("--mi-shuffles", type=int, default=MI_SHUFFLES)


def compute(X, y, I, J, rng, args):
    R = np.column_stack([digitize_quantile(X[:, c], R_BINS) for c in range(X.shape[1])])

    def single(c):                      # computed per pair like compute_pair_breakdown (same rng order)
        return mutual_information_shuffle(R[:, c], y, bins=R_BINS, n_shuffles=args.mi_shuffles, rng=rng).bits_corrected

    tot, i1, i2 = [], [], []
    for i, j in zip(I, J):
        a = single(i)
        b = single(j)
        t = mutual_information_shuffle(joint_code(R[:, i], R[:, j], R_BINS), y, bins=R_BINS * R_BINS,
                                       n_shuffles=args.mi_shuffles, rng=rng).bits_corrected
        i1.append(a), i2.append(b), tot.append(t)
    tot, i1, i2 = np.array(tot), np.array(i1), np.array(i2)
    imax = np.maximum(i1, i2)
    return {"I_total": tot, "I_1": i1, "I_2": i2, "I_lin": i1 + i2, "I_max": imax, "gain": tot - imax,
            "RSI": tot - (i1 + i2)}


run(__file__, stem="behavior_information", store_name="behavior_info", version="3j-v1", what="Pairwise information",
    components=[("I_1", "tab:blue"), ("I_2", "tab:cyan"), ("I_max", "tab:green")], total="I_total",
    compare=["I_total", "gain", "RSI"], ylabel="information (bits)", compute=compute, store_values=VALUES,
    add_args=add_args, key_extra=lambda a: dict(mi_shuffles=a.mi_shuffles))
