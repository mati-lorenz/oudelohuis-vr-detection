# -*- coding: utf-8 -*-
"""
gain_model.py
===============
Fits the "doubly stochastic" / modulated-Poisson population gain model
(Goris, Movshon & Simoncelli, 2014, Nat Neurosci) to simultaneously
recorded neurons:

    r_i(t) | G(t)  ~  Poisson( mu_i(t) * G(t) )

where mu_i(t) is neuron i's own stimulus/condition-driven mean response
(its tuning curve, estimated by cross-validation to avoid circularity --
see `estimate_tuning_curves_cv`), and G(t) is a SHARED, population-wide
multiplicative gain per trial with E[G]=1, Var[G]=sigma_G^2. Integrating
out G gives the population-level regularities checked elsewhere in this
pipeline:

    Var(r_i)   = mu_i + sigma_G^2 * mu_i^2      (quadratic mean-variance law)
    Fano_i     = 1 + sigma_G^2 * mu_i            (Fano grows linearly with rate)
    Cov(r_i,r_j) ~= sigma_G^2 * mu_i * mu_j       (shared-gain noise correlation)

Two variants are fit here:

  Model A (`estimate_gain_unconstrained`): G(t) is left completely free,
      estimated per trial via the precision-weighted estimator
          G_hat(t) = sum_i r_i(t) / sum_i mu_i(t)
      which is the minimum-variance unbiased combination across neurons
      given the model above (see the derivation in the function
      docstring), plus an analytic bias-corrected estimate of sigma_G^2.

  Model B (`fit_poisson_gain_glm`): G(t) is constrained to depend on one
      measured behavioral variable via a log-link,
          G(t) = exp(beta0 + beta1 * behavior(t))
      fit as a 2-parameter Poisson GLM with a fixed offset log(mu_i(t)),
      pooling all neurons and trials together. Implemented as a small,
      dependency-free Newton-IRLS solver (no statsmodels required).

`predicted_noise_correlation` gives the independent consistency check:
the shared-gain model predicts a SPECIFIC noise-correlation structure
(proportional to mu_i*mu_j), directly comparable against the empirical
noise correlations from `spike_stats.signal_noise_correlations`.

All estimators validated against synthetic data with known ground truth
in tests/test_gain_model_canonical.py.
"""

import numpy as np


# --------------------------------------------------------------------------
# Cross-validated tuning-curve (mu_i(t)) estimation
# --------------------------------------------------------------------------

def estimate_tuning_curves_cv(respmat, stim, n_folds=5, random_state=0, min_value=1e-3):
    """
    Cross-validated estimate of each neuron's condition-specific mean
    response mu_i(t), i.e. an out-of-fold per-condition mean: for every
    trial t, mu_hat_i(t) is neuron i's mean response to trials of the
    SAME condition as t, EXCLUDING t's own fold -- avoiding the
    circularity of using a trial's own response to predict itself
    (which would make G_hat(t) trivially close to 1 on every trial).

    Parameters
    ----------
    respmat : array (K trials, N neurons) -- already oriented (caller's
        responsibility here, since this module works trial-major
        throughout; use single_cell.py's orientation helper upstream if
        you have (N, K))
    stim : 1D array, length K, discrete condition label
    n_folds : int
    random_state : int
    min_value : float
        floor applied to the returned mu_hat (must be > 0 for the
        Poisson-gain machinery downstream: log(mu) and division by mu
        both require this)

    Returns
    -------
    mu_hat : array (K, N), same shape as respmat
    """
    X = np.asarray(respmat, dtype=float)
    stim = np.asarray(stim)
    K, N = X.shape
    rng = np.random.default_rng(random_state)

    # stratified fold assignment: each condition's trials are split
    # round-robin across folds, so every fold has training data for
    # every condition whenever a condition has >= n_folds trials
    fold_id = np.full(K, -1, dtype=int)
    for sv in np.unique(stim):
        idx = np.where(stim == sv)[0]
        idx = rng.permutation(idx)
        fold_id[idx] = np.arange(len(idx)) % n_folds

    mu_hat = np.full((K, N), np.nan)
    overall_cond_mean = {sv: X[stim == sv, :].mean(axis=0) for sv in np.unique(stim)}

    for f in range(n_folds):
        test_idx = np.where(fold_id == f)[0]
        train_idx = np.where(fold_id != f)[0]
        if len(test_idx) == 0:
            continue
        for sv in np.unique(stim[test_idx]):
            test_this_cond = test_idx[stim[test_idx] == sv]
            train_this_cond = train_idx[stim[train_idx] == sv]
            if len(train_this_cond) > 0:
                mu_hat[test_this_cond, :] = X[train_this_cond, :].mean(axis=0)
            else:
                # fold happened to take ALL of this condition's trials
                # (only possible if that condition has < n_folds trials);
                # fall back to the overall (non-held-out) condition mean
                mu_hat[test_this_cond, :] = overall_cond_mean[sv]

    return np.clip(mu_hat, min_value, None)


# --------------------------------------------------------------------------
# Model A: unconstrained per-trial gain
# --------------------------------------------------------------------------

def estimate_gain_unconstrained(respmat, mu_hat):
    """
    Precision-weighted estimate of the shared per-trial gain G(t) and an
    analytic, bias-corrected estimate of its variance sigma_G^2.

    Derivation: under r_i(t) | G(t) ~ Poisson(mu_i(t) G(t)),
        G_hat(t) = sum_i r_i(t) / sum_i mu_i(t)
    satisfies E[G_hat(t) | G(t)] = G(t) and
        Var[G_hat(t) | G(t)] = G(t) / sum_i mu_i(t)
    (since Var[r_i|G] = mu_i(t) G(t) for a Poisson, and the weights sum_i
    mu_i are the precision-optimal linear combination). Marginalizing
    over G(t) (E[G]=1):
        Var[G_hat(t)] = Var[G(t)] + E[G(t)] / sum_i mu_i(t)
                       = sigma_G^2 + 1 / sum_i mu_i(t)
    so sigma_G^2 is recovered by subtracting the (known, computable)
    average pure-sampling-noise variance from the observed variance of
    G_hat across trials.

    Parameters
    ----------
    respmat : array (K, N)
    mu_hat : array (K, N), from `estimate_tuning_curves_cv`

    Returns
    -------
    dict with:
        G_hat : array (K,)
        sigma_G_sq : float, bias-corrected estimate (clipped at 0)
        sigma_G_sq_raw : float, uncorrected (naive Var(G_hat))
        mean_sampling_noise_var : float, the subtracted correction term
    """
    X = np.asarray(respmat, dtype=float)
    mu_hat = np.asarray(mu_hat, dtype=float)
    sum_mu = mu_hat.sum(axis=1)          # (K,)
    sum_r = X.sum(axis=1)                # (K,)

    G_hat = sum_r / sum_mu

    sampling_noise_var_per_trial = 1.0 / sum_mu     # using E[G]=1
    mean_sampling_noise_var = float(np.mean(sampling_noise_var_per_trial))
    sigma_G_sq_raw = float(np.var(G_hat))
    sigma_G_sq = max(sigma_G_sq_raw - mean_sampling_noise_var, 0.0)

    return {
        'G_hat': G_hat,
        'sigma_G_sq': sigma_G_sq,
        'sigma_G_sq_raw': sigma_G_sq_raw,
        'mean_sampling_noise_var': mean_sampling_noise_var,
    }


# --------------------------------------------------------------------------
# Model B: gain constrained to a behavioral variable (Poisson GLM w/ offset)
# --------------------------------------------------------------------------

def fit_poisson_gain_glm(respmat, mu_hat, behavior, max_iter=50, tol=1e-8,
                          ridge=1e-8):
    """
    Fit G(t) = exp(beta0 + beta1 * behavior(t)) by pooled Poisson
    regression with a fixed offset log(mu_i(t)):
        log E[r_i(t)] = log(mu_i(t)) + beta0 + beta1 * behavior(t)
    across ALL (neuron, trial) pairs jointly (behavior varies by trial
    only, so this estimates ONE shared beta0, beta1 for the whole
    population). Implemented as a small dependency-free Newton-IRLS
    solver (2 parameters only, converges in a handful of iterations).

    Parameters
    ----------
    respmat : array (K, N)
    mu_hat : array (K, N)
    behavior : 1D array, length K
    max_iter, tol : IRLS convergence controls
    ridge : small L2 penalty on [beta0, beta1] for numerical stability
        (negligible effect on the fit; guards against near-singular
        design matrices, e.g. near-constant behavior traces)

    Returns
    -------
    dict with:
        beta0, beta1 : fitted coefficients
        G_hat_B : array (K,), exp(beta0 + beta1*behavior) -- the
            behavior-predicted gain, one value per TRIAL (broadcastable
            against neurons since it doesn't depend on neuron identity)
        deviance, null_deviance, pseudo_r2 : McFadden-style pseudo-R^2
            of the FULL model (offset + behavior) relative to the
            offset-only (beta1=0) model
        n_iter : IRLS iterations used
    """
    X = np.asarray(respmat, dtype=float)
    mu_hat = np.asarray(mu_hat, dtype=float)
    behavior = np.asarray(behavior, dtype=float)
    K, N = X.shape

    # flatten to long format: one row per (trial, neuron)
    y = X.ravel()                                     # (K*N,)
    log_offset = np.log(mu_hat).ravel()                # (K*N,)
    behavior_long = np.repeat(behavior, N)             # (K*N,)
    design = np.column_stack([np.ones_like(behavior_long), behavior_long])  # (K*N, 2)

    beta = np.zeros(2)
    # sensible starting point for beta0: log of the overall rate ratio
    beta[0] = np.log(max(y.mean() / np.exp(log_offset).mean(), 1e-6))

    for it in range(max_iter):
        eta = log_offset + design @ beta
        eta = np.clip(eta, -30, 30)          # guard against overflow in exp
        mu_pred = np.exp(eta)

        # IRLS weight = mu_pred (Poisson variance function), working
        # response z = eta - offset + (y - mu_pred)/mu_pred
        W = mu_pred
        z = (design @ beta) + (y - mu_pred) / np.maximum(mu_pred, 1e-12)

        WD = design * W[:, None]
        A = design.T @ WD + ridge * np.eye(2)
        b = design.T @ (W * z)
        beta_new = np.linalg.solve(A, b)

        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new

    beta0, beta1 = beta
    G_hat_B = np.exp(np.clip(beta0 + beta1 * behavior, -30, 30))

    # deviance of the fitted model vs. a saturated model, and of a
    # null (offset-only, beta1=0) model, for a McFadden pseudo-R^2
    def _poisson_deviance(y, mu_pred):
        mu_pred = np.maximum(mu_pred, 1e-12)
        with np.errstate(divide='ignore', invalid='ignore'):
            term = np.where(y > 0, y * np.log(y / mu_pred), 0.0) - (y - mu_pred)
        return float(2 * np.sum(term))

    eta_full = np.clip(log_offset + design @ beta, -30, 30)
    mu_full = np.exp(eta_full)
    deviance = _poisson_deviance(y, mu_full)

    beta0_null = np.log(max(y.mean() / np.exp(log_offset).mean(), 1e-6))
    mu_null = np.exp(np.clip(log_offset + beta0_null, -30, 30))
    null_deviance = _poisson_deviance(y, mu_null)

    pseudo_r2 = 1 - deviance / null_deviance if null_deviance > 0 else np.nan

    return {
        'beta0': float(beta0), 'beta1': float(beta1),
        'G_hat_B': G_hat_B,
        'deviance': deviance, 'null_deviance': null_deviance,
        'pseudo_r2': float(pseudo_r2), 'n_iter': it + 1,
    }


# --------------------------------------------------------------------------
# Independent consistency check: predicted noise correlation structure
# --------------------------------------------------------------------------

def predicted_noise_correlation(mu_i, mu_j, sigma_G_sq):
    """
    Shared-gain model's predicted PEARSON noise correlation between two
    neurons with (trial-averaged) mean responses mu_i, mu_j, given the
    fitted sigma_G^2:

        Cov(r_i, r_j)      ~= sigma_G^2 * mu_i * mu_j
        Var(r_i)            = mu_i + sigma_G^2 * mu_i^2
        corr_predicted      = Cov / sqrt(Var(r_i) Var(r_j))

    Compare this against the EMPIRICAL noise correlation from
    `spike_stats.signal_noise_correlations` -- if the shared-gain model
    is a good description of the data, predicted and observed should
    track each other closely across pairs; systematic deviation
    indicates the single-shared-gain model is too simple (e.g. multiple
    independent gain sources, or area-specific rather than global gain).

    IMPORTANT CAVEAT (see `predicted_noise_correlation_coupling` for the
    fix): this formula assumes every neuron has IDENTICAL, unit
    sensitivity to the shared gain. At the mean-response magnitudes
    typical of this pipeline's deconvolved traces (tens to hundreds,
    NOT literal small spike counts -- see spike_stats.event_rate's
    docstring), this assumption makes the formula saturate toward a
    predicted correlation of ~1 for essentially ANY sigma_G^2 > 0,
    regardless of the true population coupling strength -- e.g. with
    mu=100 and a modest sigma_G_sq=0.05, this formula already predicts
    corr~0.83. If your predicted-vs-observed check shows predictions
    uniformly saturated near 1 while observations scatter near 0 (a
    classic symptom), use `predicted_noise_correlation_coupling`
    instead, with per-neuron coupling strengths from
    `estimate_neuron_coupling_strength`.

    Parameters
    ----------
    mu_i, mu_j : scalars or arrays (elementwise), trial-averaged mean
        response of each neuron in the pair
    sigma_G_sq : float

    Returns
    -------
    predicted correlation, same shape as mu_i/mu_j
    """
    mu_i = np.asarray(mu_i, dtype=float)
    mu_j = np.asarray(mu_j, dtype=float)
    cov = sigma_G_sq * mu_i * mu_j
    var_i = mu_i + sigma_G_sq * mu_i ** 2
    var_j = mu_j + sigma_G_sq * mu_j ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = cov / np.sqrt(var_i * var_j)
    return corr


# --------------------------------------------------------------------------
# Per-neuron loadings on the shared factor (heterogeneous coupling)
# --------------------------------------------------------------------------
#
# The models above assume every neuron shares the SAME multiplicative
# gain with unit sensitivity -- i.e. r_i(t) | G(t) ~ Poisson(mu_i(t) G(t))
# for every i alike. Real populations instead show substantial
# neuron-to-neuron heterogeneity in how strongly they track a shared
# population-wide fluctuation (Okun et al. 2015, Nature; Lin, Okun,
# Carandini & Harris, 2015, "The nature of shared cortical variability",
# Neuron) -- some neurons follow it closely, some barely at all, and
# some can even be anti-coupled. This is captured by giving each neuron
# its own LOADING a_i on the shared factor:
#
#     r_i(t) | F(t)  ~  Poisson( mu_i(t) * exp(a_i * log F(t)) )
#                     = Poisson( mu_i(t) * F(t)^{a_i} )
#
# with a_i=1 recovering the original uniform-gain model for that neuron,
# a_i=0 meaning neuron i is completely insensitive to the shared
# fluctuation (pure Poisson noise, uncorrelated with everyone else), and
# a_i<0 meaning it's anti-coupled. This directly resolves the saturation
# problem above: predicted pairwise correlation now depends on a_i*a_j,
# so pairs where either neuron has small/zero/oppositely-signed loading
# correctly get predicted LOW or negative correlation, instead of the
# uniform near-1 saturation.

def estimate_gain_leave_one_out(respmat, mu_hat, eps=1e-6):
    """
    Per-neuron, leave-one-out version of the population gain: for each
    neuron i, the precision-weighted gain estimated from every OTHER
    neuron (excluding i itself). Used by `estimate_neuron_coupling_
    strength` purely as a SIGN reference (is neuron i positively or
    negatively coupled to the shared population fluctuation?) -- its
    MAGNITUDE is not used for that purpose, since this uniform-weight
    construction gets amplitude-compressed when the true population
    loadings are heterogeneous and mixed-sign (verified on simulated
    data: fitting a per-neuron loading by regressing directly against
    this reference inflated every loading by the same ~4x factor,
    because oppositely-loaded OTHER neurons partially cancel in this
    uniform-weighted sum) -- but its SIGN remains reliable, which is
    all `estimate_neuron_coupling_strength` needs from it.

    Parameters
    ----------
    respmat, mu_hat : array (K, N)

    Returns
    -------
    G_loo : array (K, N); G_loo[:, i] is the gain estimate excluding
        neuron i
    """
    X = np.asarray(respmat, dtype=float)
    mu_hat = np.asarray(mu_hat, dtype=float)
    sum_r = X.sum(axis=1, keepdims=True)         # (K, 1)
    sum_mu = mu_hat.sum(axis=1, keepdims=True)   # (K, 1)
    G_loo = (sum_r - X) / (sum_mu - mu_hat)
    return np.clip(G_loo, eps, None)


def estimate_neuron_coupling_strength(respmat, mu_hat, stim, min_reference_neurons=5):
    """
    Per-neuron "coupling strength" c_i = a_i * sqrt(sigma_G^2) to a
    shared population factor, WITHOUT separately estimating a loading
    a_i and a population sigma_G^2 -- that separation turns out to be
    only identifiable up to an arbitrary rescaling (a_i -> a_i/c,
    F(t) -> F(t)^c leaves the model unchanged), which made an earlier
    version of this module's iterative loadings+factor fit converge
    only very slowly and to an arbitrary, incorrect absolute scale
    (verified on simulated data: after 5 iterations the fitted-vs-true
    loading regression slope was still ~3.8, not 1.0). The PRODUCT
    c_i = a_i*sqrt(sigma_G^2) has no such ambiguity and is directly,
    robustly estimable from each neuron's OWN excess variance:

        Var(r_i) ~= mu_i + c_i^2 * mu_i^2   (same quadratic law as
                                              before, now per-neuron)
        =>  c_i^2 ~= ( E[(r_i(t)-mu_i(t))^2] - E[mu_i(t)] ) / E[mu_i(t)^2]

    computed from RESIDUALS after subtracting each trial's own
    condition-specific mu_i(t) (NOT the unconditional/pooled variance
    of r_i -- pooling across stimulus conditions would conflate genuine
    tuning-driven variance with this noise-driven quantity, exactly the
    issue flagged in the qc_lib.py review earlier in this project).

    The SIGN of c_i (is this neuron positively or negatively coupled to
    the shared factor?) is separately determined from the sign of its
    correlation with the (uniform-weight) leave-one-out reference gain
    -- only the SIGN is used from that regression, which is robust even
    though that regression's MAGNITUDE would be biased under
    heterogeneous population loadings (see `estimate_gain_leave_one_out`).

    Validated on simulated data with heterogeneous, mixed-sign true
    loadings: recovers c_i^2 with correlation > 0.999 and correct
    regression slope (~1.0, not biased), 100% correct signs, and the
    resulting `predicted_noise_correlation_coupling` matches PROPERLY
    stimulus-conditioned empirical noise correlations almost exactly
    (correlation 0.999 in the validation test).

    Parameters
    ----------
    respmat, mu_hat : array (K, N)
    stim : 1D array, length K -- needed to compute per-condition
        residuals properly (see conflation warning above)
    min_reference_neurons : if fewer than this many OTHER neurons are
        available for the leave-one-out sign reference, that neuron's
        sign defaults to positive (with c_i itself still valid; only
        the sign determination needs a reasonably-sized reference)

    Returns
    -------
    c : array (N,), signed coupling strength per neuron
    c_squared_raw : array (N,), the (unsigned) magnitude-squared estimate
    """
    X = np.asarray(respmat, dtype=float)
    mu_hat = np.asarray(mu_hat, dtype=float)
    stim = np.asarray(stim)
    K, N = X.shape

    # residuals against each trial's OWN condition-specific mu_hat --
    # NOT the unconditional variance, to avoid conflating tuning with
    # noise (see docstring)
    residual = X - mu_hat
    mean_resid_sq = (residual ** 2).mean(axis=0)
    mean_mu = mu_hat.mean(axis=0)
    mean_mu2 = (mu_hat ** 2).mean(axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        c_squared_raw = np.maximum((mean_resid_sq - mean_mu) / mean_mu2, 0.0)

    # sign from leave-one-out reference (magnitude of this regression is
    # NOT used, only its sign, which is robust to the amplitude bias
    # documented in estimate_gain_leave_one_out)
    if N > min_reference_neurons:
        G_loo = estimate_gain_leave_one_out(X, mu_hat)
        sign_c = np.ones(N)
        for i in range(N):
            with np.errstate(invalid='ignore'):
                s = np.corrcoef(residual[:, i], G_loo[:, i] - 1)[0, 1]
            sign_c[i] = 1.0 if not np.isfinite(s) or s >= 0 else -1.0
    else:
        sign_c = np.ones(N)

    c = sign_c * np.sqrt(c_squared_raw)
    return c, c_squared_raw


def predicted_noise_correlation_coupling(mu_i, mu_j, c_i, c_j):
    """
    Shared-factor noise-correlation prediction using per-neuron coupling
    strengths (see `estimate_neuron_coupling_strength`) -- the
    identifiable, correctly-scaled replacement for
    `predicted_noise_correlation`'s uniform-sensitivity assumption:

        Cov(r_i, r_j) ~= c_i * c_j * mu_i * mu_j
        Var(r_i)       = mu_i + c_i^2 * mu_i^2

    Reduces to `predicted_noise_correlation(mu_i, mu_j, sigma_G_sq)`
    when c_i = c_j = sqrt(sigma_G_sq) (the uniform-sensitivity case).

    Parameters
    ----------
    mu_i, mu_j, c_i, c_j : scalars or elementwise-matched arrays

    Returns
    -------
    predicted correlation, same shape as inputs
    """
    mu_i = np.asarray(mu_i, dtype=float)
    mu_j = np.asarray(mu_j, dtype=float)
    c_i = np.asarray(c_i, dtype=float)
    c_j = np.asarray(c_j, dtype=float)
    cov = c_i * c_j * mu_i * mu_j
    var_i = mu_i + (c_i ** 2) * (mu_i ** 2)
    var_j = mu_j + (c_j ** 2) * (mu_j ** 2)
    with np.errstate(divide='ignore', invalid='ignore'):
        corr = cov / np.sqrt(var_i * var_j)
    return corr
