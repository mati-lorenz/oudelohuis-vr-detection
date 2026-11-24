# -*- coding: utf-8 -*-
"""
noise_sdt.py
============
d' and criterion from NOISE trials only (0 % < signal < 100 %), i.e. without the 0 % / 100 % catch trials that the classical
signal-detection formulas need (hit rate at 100 %, false-alarm rate at 0 %).

Without the extremes there is no pure "signal" and no pure "noise" class, so the classical formula cannot be applied directly.
Four ways to get around it (all return the SAME sign convention as infotheory.behavior.compute_dprime: criterion > 0 = conservative,
the mouse licks less):

probit_pred   (recommended) per session and state bin, fit  P(lick | s) = g + (1 - g - l) * Phi(a + b * s / 100)  on the noise trials
              (s = signal in %). g (floor = false-alarm rate) and l (lapse) are FIXED to the session's 1b values, because they are
              set by the 0 % / 100 % trials that are not used here. (a, b) are fitted by penalised maximum likelihood; the prior is
              centred on the session's 1b curve (a0 = -mu / sigma, b0 = 100 / sigma), so a bin with few trials cannot run away.
              The 0 % and 100 % hit / FA rates are then PREDICTED from the fitted curve,
                  FA = g + (1 - g - l) * Phi(a),   H = g + (1 - g - l) * Phi(a + b),
              and the classical formulas are applied: d' = z(H) - z(FA), c = -(z(H) + z(FA)) / 2. These numbers are on the same scale as
              the classical 0 % / 100 % ones (they saturate at z(1 - l) - z(g)), so they can be compared with them directly.
probit_shift  scale-free changes relative to the session's own 1b curve, both INTERPOLATED (no extrapolation to 0 % / 100 %):
              criterion = (theta_bin - mu) / sigma, theta_bin = the fitted 50 %-point of the latent curve (-a / b * 100 %), i.e. the
              threshold shift in units of the session's sigma (> 0 = needs more signal before licking = conservative, 0 = the
              session's average curve); dprime = b / b0, the bin's slope relative to the session's 1b slope (1 = average sensitivity).
split         the weak vs strong halves of the noise trials (split at the session's median noise signal) are treated as the two
              classes: H = P(lick | strong), FA = P(lick | weak). Needs no model, but the two classes are close together, so d' is
              much smaller than the 0 % / 100 % d' (and depends on how far apart the noise levels are in that session).
classic       (reference, not computed here) d' and c on the 0 % / 100 % trials, infotheory.behavior.compute_dprime.

What is lost without the 0 % trials (read before interpreting a state effect)
--------------------------------------------------------------------------
* The floor g (false-alarm rate) and the lapse l are held at the session's average values. A state that changes the false-alarm rate
  (e.g. pupil -> licking at 0 %) cannot be seen as a floor change here; on the noise trials it shows up as a shift of a (criterion) and
  a flatter b (d'). So a criterion effect on noise trials is the sum of the threshold shift and any floor shift.
* The 0 % / 100 % rates are extrapolations along a cumulative Gaussian (equal-variance SDT); they are only as good as that shape and
  as the 1b fit (the session's noise levels should bracket mu).
* The classical criterion (0 % / 100 % rates) is dominated by the floor / lapse balance: with g and l fixed, shifting the whole curve
  left or right barely changes the predicted 0 % / 100 % rates, so `criterion_probit_pred` hardly moves with a threshold shift (checked on
  simulated data). `criterion_probit_shift` follows it. (A "lapse-free d' = slope x 100 %, c = - z at 50 %" version was dropped: with
  steep curves it extrapolates far beyond the tested levels and gives d' of ~20 and c of ~-9, dominated by the slope noise.)
* The prior pulls every bin towards the session's average curve, so state differences are, if anything, shrunk.
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, special, stats

NOISE_METHODS = ["probit_pred", "probit_shift", "split"]
NOISE_COLS = [f"{k}_{m}" for m in NOISE_METHODS for k in ("dprime", "criterion")]


def _ppf(p, clip):
    return stats.norm.ppf(np.clip(p, clip, 1.0 - clip))


def sdt_from_rates(hit, fa, clip=0.005):
    """Classical d' and criterion from a hit and a false-alarm rate (clipped away from 0 / 1)."""
    zh, zf = _ppf(hit, clip), _ppf(fa, clip)
    return float(zh - zf), float(-0.5 * (zh + zf))


def fit_probit_bin(signal, resp, guess, lapse, a0, b0, sd_a=1.0, sd_b_frac=0.75):
    """Penalised ML fit of (a, b) in P(lick) = guess + (1 - guess - lapse) * Phi(a + b * signal / 100) on noise trials.
    The Gaussian prior is centred on (a0, b0) (the session's 1b curve); b is kept >= 0 (a monotone curve)."""
    x = np.asarray(signal, float) / 100.0
    y = np.asarray(resp, float)
    scale = 1.0 - guess - lapse
    sd_b = max(sd_b_frac * abs(b0), 0.5)

    def nll(p):
        a, b = p
        P = np.clip(guess + scale * special.ndtr(a + b * x), 1e-6, 1.0 - 1e-6)
        return -(y * np.log(P) + (1.0 - y) * np.log1p(-P)).sum() + 0.5 * ((a - a0) / sd_a) ** 2 + 0.5 * ((b - b0) / sd_b) ** 2

    res = optimize.minimize(nll, [a0, max(b0, 0.0)], method="L-BFGS-B", bounds=[(-12.0, 12.0), (0.0, 60.0)])
    if not np.isfinite(res.fun):
        return np.nan, np.nan
    return float(res.x[0]), float(res.x[1])


def noise_sdt_session(signal, resp, mu, sigma, guess, lapse, split_at, *, min_n=20, sd_a=1.0, sd_b_frac=0.75,
                      rate_clip=0.005, min_per_class=3):
    """All noise-trial d' / criterion variants for ONE session and ONE state bin. `signal` / `resp` = the bin's noise trials
    (0 < signal < 100). guess / lapse NaN -> 0. Returns a dict with NOISE_COLS (NaN where not computable)."""
    out = {c: np.nan for c in NOISE_COLS}
    signal = np.asarray(signal, float)
    resp = np.asarray(resp, float)
    keep = np.isfinite(signal) & np.isfinite(resp)
    signal, resp = signal[keep], resp[keep]
    if len(signal) < min_n:
        return out
    g = float(guess) if guess is not None and np.isfinite(guess) else 0.0
    l_ = float(lapse) if lapse is not None and np.isfinite(lapse) else 0.0
    if g + l_ >= 0.98:
        g, l_ = 0.0, 0.0
    if mu is not None and sigma is not None and np.isfinite(mu) and np.isfinite(sigma) and sigma > 0:
        a0, b0 = -mu / sigma, 100.0 / sigma
        a, b = fit_probit_bin(signal, resp, g, l_, a0, b0, sd_a, sd_b_frac)
        if np.isfinite(a):
            sc = 1.0 - g - l_
            fa = g + sc * special.ndtr(a)
            hit = g + sc * special.ndtr(a + b)
            out["dprime_probit_pred"], out["criterion_probit_pred"] = sdt_from_rates(hit, fa, rate_clip)
            if b > 0.05 * b0:
                out["dprime_probit_shift"] = float(b / b0)
                out["criterion_probit_shift"] = float(np.clip((-a / b * 100.0 - mu) / sigma, -5.0, 5.0))
    if split_at is not None and np.isfinite(split_at):
        strong = signal >= split_at
        if min(strong.sum(), (~strong).sum()) >= min_per_class:
            n_s, n_w = int(strong.sum()), int((~strong).sum())
            h = np.clip(resp[strong].mean(), 0.5 / n_s, 1 - 0.5 / n_s)
            f = np.clip(resp[~strong].mean(), 0.5 / n_w, 1 - 0.5 / n_w)
            out["dprime_split"], out["criterion_split"] = sdt_from_rates(h, f, 1e-9)
    return out
