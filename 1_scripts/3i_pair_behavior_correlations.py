# -*- coding: utf-8 -*-
"""
3i_pair_behavior_correlations
=============================
Pair correlations with a BEHAVIORAL target: the counterpart of 3a / 3d / 3h in which the variable that defines "signal" and
"noise" is not the stimulus or the choice but a behavioral variable (runspeed, pupil_area, motionenergy, lick; as 2c / 2h).
In every window the trial-mean behavior is cut into RESPONSE_BINS equal-occupancy bins (the target) and, across the engaged
trials of the window, for each pair of simultaneously recorded cells:
    raw_corr    plain Pearson correlation of the two cells' trial-mean activity
    noise_corr  correlation of the residuals after removing each cell's mean in each behavior bin = the correlation that is
                NOT explained by the cells following the same behavior
    signal_corr correlation of the behavior-bin means of the two cells (needs >= 3 bins; NaN otherwise, e.g. a lick variable
                that has only two distinct values in a window). With 3 bins it is a coarse number (3 points).
    raw_minus_noise  = raw - noise: how much of the pair's correlation the shared behavior explains (it is not a significance
                test; a pair of cells that both follow running speed gets a positive value)
The same definitions as 3a (indicator-matrix label means) and the same code as 3d (`pair_temporal.corr_metrics`).

WINDOWS, PAIRS, FILTERS: see infotheory/behavior_pairs.py and behavior_pair_runner.py -- axes position (cm) and time (s); binnings
windows (3a-3c's three position windows), fixed, sliding; 150 pairs per (session, area, label) from the 2a + 2b included cells
(the same sample in 3i-3l); 1b-included sessions; engaged trials; each frame inside its own trial; the pair_trials rule.

Project-wide rules (infotheory/project_config.py) -- what applies here
    * selection : --animals / --sessions restrict the sessions; a restricted run writes to out/selection/<tag>/.
    * aggregate : --aggregate pooled|session|mouse (units with < MIN_PAIRS_PER_UNIT pairs dropped, band = SEM over units;
                  pooled: SEM over pairs). Always the six groups.
    * memory    : one session in RAM per worker; pairs reduced to per-(session, group) sums.
    * binning   : the SAME pairs, windows, behavior bins and trial rule as 3j / 3k / 3l, so correlations, information, breakdown
                  and PID agree before bias correction.

Outputs
-------
2_pipeline/3i_pair_behavior_correlations/
    out/    behavior_correlations_moments.csv, behavior_correlations_by_group[__<aggregate>].csv
            figures/<axis>/<binning>/<behavior>/*.png, figures/<axis>/<binning>/compare_behaviors_*.png,
            figures/position/windows/*_by_window.png
    store/  per-session caches
"""
import numpy as np

from infotheory import pair_temporal as ptt
from infotheory.behavior_pair_runner import run

VALUES = ["raw_corr", "noise_corr", "signal_corr", "raw_minus_noise"]     # raw first: defines n_pairs


def compute(X, y, I, J, rng, args):
    _, yi = np.unique(y, return_inverse=True)
    v = ptt.corr_metrics(X, yi, I, J)
    return {"raw_corr": v["raw_corr"], "noise_corr": v["noise_corr"], "signal_corr": v["signal_corr"],
            "raw_minus_noise": v["raw_corr"] - v["noise_corr"]}


run(__file__, stem="behavior_correlations", store_name="behavior_corr", version="3i-v1",
    what="Pairwise correlation", components=[("raw_corr", "k"), ("noise_corr", "tab:blue"), ("signal_corr", "tab:red")],
    compare=["raw_corr", "noise_corr", "signal_corr", "raw_minus_noise"], ylabel="correlation", compute=compute,
    store_values=VALUES)
