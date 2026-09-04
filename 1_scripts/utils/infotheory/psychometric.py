# -*- coding: utf-8 -*-
"""
Psychometric-curve fitting: cumulative-Gaussian (erf) parameterization,
Wichmann & Hill (2001) style -- response rate = guess + (1-guess-lapse)
* Phi((x-mu)/sigma). Ported from the old `fit_psycurve`/`psychometric_function`
(same bounds convention: guess/lapse rate constrained near the observed
response rates at the lowest/highest signal level, mu in [0,100],
sigma in [2,40]) and `get_idx_performing_sessions`'s z-scored threshold
check.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import special
from scipy.optimize import curve_fit


def _r2_score(y_true, y_pred) -> float:
    """Coefficient of determination, hand-rolled to avoid pulling in
    scikit-learn for one metric (see the earlier tabulate/reporting.py
    lesson -- fewer optional dependencies, fewer broken environments)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0
    return float(1 - ss_res / ss_tot)


def psychometric_function(x, mu, sigma, lapse_rate, guess_rate):
    """mu = threshold, sigma = slope, lapse_rate = 1 - response rate at
    ceiling, guess_rate = response rate at floor (the false-alarm rate,
    for a detection task)."""
    x = np.asarray(x, dtype=float)
    return guess_rate + (1 - guess_rate - lapse_rate) * 0.5 * (
        1 + special.erf((x - mu) / (np.sqrt(2) * sigma)))


@dataclass
class PsychometricFit:
    mu: float
    sigma: float
    lapse_rate: float
    guess_rate: float
    r2: float
    n_trials: int
    x: np.ndarray  # signal levels tested (condition means, for plotting the data points)
    y: np.ndarray  # observed response rate per signal level

    def predict(self, x):
        return psychometric_function(x, self.mu, self.sigma, self.lapse_rate, self.guess_rate)

    def threshold_z_range(self, intermediate_signal) -> tuple[float, float]:
        """z-scored range of the intermediate ("threshold") stimulus
        levels actually presented, relative to the fit -- the
        `noise_zmin`/`noise_zmax` from the original `noise_to_psy`. A
        range that straddles zero means the tested stimuli bracket the
        fitted threshold (interpolation); if it doesn't, mu was
        extrapolated beyond what was actually tested."""
        z = (np.asarray(intermediate_signal, dtype=float) - self.mu) / self.sigma
        return float(np.nanmin(z)), float(np.nanmax(z))

    def threshold_in_range(self, intermediate_signal) -> bool:
        zmin, zmax = self.threshold_z_range(intermediate_signal)
        return zmin <= 0 <= zmax


def fit_psychometric(trialdata: pd.DataFrame, signal_col: str = "signal",
                      response_col: str = "lickResponse", mu_bounds=(0, 100),
                      sigma_bounds=(2, 40)) -> PsychometricFit | None:
    """Fit on trial-by-trial responses (not condition means -- more
    data-efficient, and the original scripts' convention). Returns None
    if there aren't enough distinct signal levels to constrain the
    4-parameter fit, or the optimizer doesn't converge."""
    psydata = trialdata.groupby(signal_col)[response_col].mean()
    x = psydata.index.to_numpy(dtype=float)
    y = psydata.to_numpy()
    if len(x) < 3:
        return None

    X = trialdata[signal_col].to_numpy(dtype=float)
    Y = trialdata[response_col].to_numpy(dtype=float)

    guess_obs = y[0]        # response rate at the lowest signal level
    lapse_obs = 1 - y[-1]   # 1 - response rate at the highest signal level
    initial_guess = [20, 15, lapse_obs, guess_obs]

    guess_lower, guess_upper = max(guess_obs * 0.8 - 0.01, 0.0), min(guess_obs * 1.2, 1.0)
    lapse_lower, lapse_upper = max(lapse_obs * 0.8, 0.0), min(lapse_obs * 1.2 + 0.01, 1.0)
    # Bounds collapse to zero width when the observed rate is exactly 0 or 1
    # (e.g. a session with no false alarms at floor) -- curve_fit requires a
    # strictly positive gap, so pad it open a little in that case.
    if guess_upper <= guess_lower:
        guess_upper = guess_lower + 0.02
    if lapse_upper <= lapse_lower:
        lapse_upper = lapse_lower + 0.02

    bounds = (
        [mu_bounds[0], sigma_bounds[0], lapse_lower, guess_lower],
        [mu_bounds[1], sigma_bounds[1], lapse_upper, guess_upper],
    )

    try:
        params, _ = curve_fit(psychometric_function, X, Y, p0=initial_guess, bounds=bounds)
    except (RuntimeError, ValueError):
        return None

    r2 = _r2_score(y, psychometric_function(x, *params))
    return PsychometricFit(*params, r2=r2, n_trials=len(trialdata), x=x, y=y)
