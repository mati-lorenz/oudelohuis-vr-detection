# -*- coding: utf-8 -*-
"""
Dimensionality-reduction pipeline, STEP 1: cross-validated dimensionality
estimate, per (area, label) group, using Factor Analysis, raw PCA, and
z-scored PCA SEPARATELY (all three fit as probabilistic models --
sklearn's PCA.score() and FactorAnalysis.score() both return average
held-out log-likelihood under a linear-Gaussian generative model, so all
three are directly comparable, as n_components argmax, on identical CV
folds; FA additionally models private per-neuron noise, PCA assumes
isotropic noise). Raw PCA is deliberately run UNSTANDARDIZED against FA as
a diagnostic for heterogeneous per-neuron variance (see
compute_explained_variance's docstring); 'pca_zscored' is added alongside
it, not instead of it, to directly check whether standardizing PCA's
input closes the gap to FA -- confirming or challenging that explanation
rather than silently "fixing" PCA.

For each session and each requested group, sweeps n_components over a
candidate grid, K-fold cross-validates the held-out log-likelihood for
both methods, and reports the argmax n_components ("optimal"
dimensionality) for each -- the number of components a linear-Gaussian
model actually justifies for that population, before committing to a
heavier method (GPFA, dPCA, etc.) in later stages.

This uses the SAME group-selection convention as every other script in
this pipeline: filter_nearlabeled_layer23 (anatomical) + load_qc_pass
(QC) + filter_target_groups (which area/label groups you actually want --
default is everything, but e.g. target_labels=['lab'] restricts to
labeled cells only, target_areas=['PM'] to PM only, etc.)

Produces:
  - dimensionality_all_curves.csv: long-format CV score vs n_components,
    one row per (session, area, label, method, n_components)
  - dimensionality_optimal_summary.csv: one row per (session, area,
    label, method) with the argmax n_components and its CV score
  - dimensionality_curves.png: per-group grid of CV score curves
    (one line per session, FA and PCA in separate columns)
  - dimensionality_optimal_summary.png: per-group comparison of the
    optimal n_components distribution across sessions (FA vs PCA)
"""

#%%
import os, sys
os.chdir('/u/g/glorenz/Documents/Research/Code/external_repos/oudelohuis-vr-detection')
sys.path.insert(0, '/u/g/glorenz/Documents/Research/Code/external_repos/oudelohuis-vr-detection')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']  # avoid 'findfont: Arial not found' warning
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Methods compared throughout this script: raw PCA and FA (the original
# Step-1 comparison -- see compute_explained_variance docstring for why
# these are run UNSTANDARDIZED on purpose, as a diagnostic for
# heterogeneous per-neuron variance), plus z-scored variants of both
# ('pca_zscored', 'fa_zscored'), added so the raw-vs-standardized gap can
# be checked directly for each method rather than assumed. FA already has
# a per-neuron noise-variance term, so it's expected to be closer to
# scale-invariant than PCA (a poorly-scaled neuron's private noise just
# gets absorbed into its own noise parameter either way) -- fa_zscored is
# here to CHECK that expectation against this data, not because FA is
# assumed to need it the way raw PCA does. If pca_zscored's optimal
# n_components tracks FA closely while raw pca does not, that CONFIRMS the
# heterogeneous-variance explanation already in the comments below rather
# than replacing it -- z-scoring is a diagnostic comparison here, not a
# fix applied to the original 'pca'/'fa' entries.
METHODS = ('fa', 'fa_zscored', 'pca', 'pca_zscored')

from loaddata.session_info import filter_sessions, report_sessions
from infotheory import compute_tensor_for_session
from infotheory.celldata_utils import get_area_label, DEFAULT_LABEL_ORDER
from utils.cellselection_lib import filter_nearlabeled_layer23, filter_target_groups
from infotheory.qc_lib import load_qc_pass

#%% ------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
protocol = ['DN']
calciumversion = 'deconv'

# Which (area, label) groups to include. None = no restriction on that
# axis. Examples:
#   target_areas=None,        target_labels=None        -> everything
#   target_areas=None,        target_labels=['lab']      -> labeled cells, any area
#   target_areas=['PM'],      target_labels=None         -> PM, any label
#   target_areas=['V1','PM'], target_labels=['unl']      -> V1unl + PMunl only
target_areas = None
target_labels = None

# QC filter (in addition to the anatomical V1/PM filter): produced by
# run_qc_summary.py. Run that script first; set to None to skip QC
# filtering (e.g. for a quick look before QC has been run).
qc_csv_path = os.path.join('./output/infotheory_plots', 'qc_per_cell.csv')

# tensor window -- match Steps 2-4 if you want a directly comparable window
s_pre, s_post, binsize = -60, 80, 10
t_pre, t_post = -1, 2   # used instead of s_pre/s_post for time-locked protocols (IM/GR/GN)

# Fit dimensionality SEPARATELY within each of these windows (axis units:
# cm for position-locked protocols, seconds for time-locked), rather than
# collapsing the whole trial into one scalar per neuron. This matters:
# pooling prestimulus baseline together with stimulus/response-locked
# activity into one covariance estimate can inflate or dilute the apparent
# dimensionality, since different trial epochs can have genuinely
# different population structure. Defaults below are a starting point --
# 'response' matches the peak informative window found in Step 3's
# tdr_r2_over_time.png (positions 0-40cm); adjust per your own task timing.
# 'whole_trial' spans the full s_pre-s_post range, reproducing the
# original (pre-windowing) analysis for direct comparison.
window_ranges = {
    'whole_trial': (-60, 80),
    'prestimulus': (-60, -10),
    'stimulus':    (-10, 15),
    'response':    (15, 50),
}

# candidate n_components grid: capped per-group at runtime by
# min(n_neurons, n_trials_per_fold) - 1, see cv_dimensionality below
max_components_tested = 30
cv_folds = 5
n_null_repeats = 5   # independent-neuron-shuffle null repeats, see dimensionality_significance_test
random_state = 0
min_neurons_per_group = 10   # skip groups with fewer neurons than this (CV is unreliable below it)
max_nan_frac_per_trial = 0.5   # trials with more NaN than this within a window are dropped, see collapse_window

# n_jobs at the SESSION level (primary parallelism, one worker per session).
n_jobs_sessions = -1

output_dir = './output/dimred_plots'
os.makedirs(output_dir, exist_ok=True)


#%% ------------------------------------------------------------------
# Collapse a (K, N, T) tensor to a (K, N) response matrix within one
# window, dropping trials that are mostly NaN in that specific window
# (position-binned tensors leave NaN where a trial never reached that
# window -- same rationale as Steps 2-4's NaN handling)
# ----------------------------------------------------------------------
def collapse_window(tensor, window_mask, max_nan_frac=0.5):
    """
    tensor : (K trials, N neurons, T bins). window_mask : boolean, length T.
    Returns (X, trial_keep_idx): X is (K_kept, N), trial_keep_idx marks
    which original trials survived (< max_nan_frac NaN within the window).
    """
    sub = tensor[:, :, window_mask]  # (K, N, T_window)
    frac_nan_per_trial = np.mean(np.isnan(sub), axis=(1, 2))
    trial_keep_idx = frac_nan_per_trial <= max_nan_frac
    with np.errstate(invalid='ignore'):
        X = np.nanmean(sub[trial_keep_idx], axis=2)  # (K_kept, N)
    return X, trial_keep_idx


def compute_explained_variance(X, n_components, random_state=0):
    """
    Fits ONE full PCA and ONE full FactorAnalysis (not CV-limited -- this
    is a variance-decomposition diagnostic, not a generalization estimate)
    and returns per-component explained-variance-ratio arrays for both,
    directly comparable to each other and to a classic PCA scree plot.

    sklearn's FactorAnalysis has no explained_variance_ratio_ attribute
    (unlike PCA) -- built here from the loading matrix: variance explained
    by factor k, summed across neurons, as a fraction of total sample
    variance (sum of per-neuron variances -- same denominator convention
    PCA uses). FA doesn't guarantee factors are variance-ordered the way
    PCA is by construction, so sorted descending here for a meaningful
    scree-plot presentation. Validated during development against
    synthetic data with known decreasing factor variance -- FA's metric
    tracked PCA's closely (e.g. 0.65/0.16/0.05/0.02 vs 0.66/0.17/0.06/0.02).

    Also fits z-scored variants of both, 'pca_zscored' and 'fa_zscored'
    (StandardScaler, ddof=0 to match sklearn's convention). These are
    diagnostic comparisons, not replacements for raw 'pca'/'fa' -- see
    METHODS comment above. explained_variance_ratio_ (and the FA ratio
    built the same way) is already scale-normalized (fraction of total
    variance), so each raw-vs-zscored pair differs only in how variance is
    DISTRIBUTED across components, not in the ratios' scale.

    Returns dict: {'pca', 'pca_zscored', 'fa', 'fa_zscored'} -> array,
    each of length n_components
    """
    Xz = StandardScaler().fit_transform(X)

    pca = PCA(n_components=n_components, random_state=random_state).fit(X)
    pca_z = PCA(n_components=n_components, random_state=random_state).fit(Xz)
    fa = FactorAnalysis(n_components=n_components, random_state=random_state).fit(X)
    fa_z = FactorAnalysis(n_components=n_components, random_state=random_state).fit(Xz)

    def fa_ratio_from(fa_model, X_for_total_var):
        total_var = X_for_total_var.var(axis=0, ddof=1).sum()
        factor_var = (fa_model.components_ ** 2).sum(axis=1)
        return np.sort(factor_var / total_var)[::-1] if total_var > 0 else factor_var

    return {'pca': pca.explained_variance_ratio_, 'pca_zscored': pca_z.explained_variance_ratio_,
            'fa': fa_ratio_from(fa, X), 'fa_zscored': fa_ratio_from(fa_z, Xz)}


#%% ------------------------------------------------------------------
# Core cross-validated dimensionality estimator
# ----------------------------------------------------------------------
def cv_dimensionality(X, n_components_list, cv_folds=5, random_state=0):
    """
    Cross-validated held-out log-likelihood for FA, raw PCA, and their
    z-scored variants (see METHODS comment above for why all four are
    kept, rather than replacing the raw versions), all as probabilistic
    linear-Gaussian models, over a grid of n_components.

    For the '_zscored' variants, the StandardScaler is fit on X_train ONLY
    within each fold and applied to X_test with those train-fold
    statistics -- fitting it on the full X (train+test) before splitting
    would leak test-fold variance into the standardization and inflate the
    held-out score. Held-out log-likelihood under a '_zscored' variant is
    computed in z-scored units, so it is only comparable to its raw
    counterpart as an n_components (argmax) comparison, not as a raw score
    comparison -- they're on different denominated scales.

    Parameters
    ----------
    X : array (K trials, N neurons)
    n_components_list : list of int
    cv_folds : int
    random_state : int

    Returns
    -------
    DataFrame with columns: method, n_components, cv_score_mean, cv_score_sem
    """
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    fold_idx = list(kf.split(X))

    rows = []
    for method_name in METHODS:
        Model = FactorAnalysis if method_name.startswith('fa') else PCA
        standardize = method_name.endswith('_zscored')
        scores = np.full((len(n_components_list), cv_folds), np.nan)
        for ifold, (train_idx, test_idx) in enumerate(fold_idx):
            X_train, X_test = X[train_idx], X[test_idx]
            if standardize:
                scaler = StandardScaler().fit(X_train)  # train-fold stats only, no leakage
                X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)
            for ic, nc in enumerate(n_components_list):
                model = Model(n_components=nc, random_state=random_state)
                model.fit(X_train)
                scores[ic, ifold] = model.score(X_test)  # avg log-likelihood per sample
        for ic, nc in enumerate(n_components_list):
            rows.append({
                'method': method_name, 'n_components': nc,
                'cv_score_mean': np.nanmean(scores[ic, :]),
                'cv_score_sem': np.nanstd(scores[ic, :]) / np.sqrt(cv_folds),
            })
    return pd.DataFrame(rows)


def shuffle_neurons_independently(X, rng):
    """
    Null surrogate for the dimensionality-estimate significance check
    below (Elsayed & Cunningham 2017 style): shuffle each neuron's
    (column's) trial order INDEPENDENTLY. This preserves each neuron's own
    marginal trial-to-trial distribution exactly, while destroying any
    genuine cross-neuron trial-to-trial covariation -- the simple
    alternative explanation being ruled out is "neurons are independent;
    any apparent multi-dimensional structure is overfitting to few trials."

    X : (K trials, N neurons). Returns a shuffled copy, same shape.
    """
    X_shuf = X.copy()
    K = X.shape[0]
    for n in range(X.shape[1]):
        X_shuf[:, n] = X[rng.permutation(K), n]
    return X_shuf


def dimensionality_significance_test(X, n_components_list, cv_folds=5, n_null_repeats=5, random_state=0):
    """
    Compares the REAL cv_dimensionality fit against `n_null_repeats`
    independent-neuron-shuffle nulls (see shuffle_neurons_independently),
    for FA, raw PCA, and z-scored PCA. Cheap relative to a full permutation test (CV
    fitting is the expensive part) -- n_null_repeats=5 gives a rough but
    informative comparison, not a precise p-value; raise it if you want a
    tighter estimate.

    Returns
    -------
    df_real : same as cv_dimensionality(X, ...)
    df_null_summary : one row per method, with null_max_cv_score_mean/std
        (peak CV score across n_components, averaged over the null
        repeats) and exceeds_null (bool: does the real peak CV score beat
        the null's mean peak by more than 1 null SD)
    """
    df_real = cv_dimensionality(X, n_components_list, cv_folds=cv_folds, random_state=random_state)

    rng = np.random.default_rng(random_state)
    null_peak_scores = {method: [] for method in METHODS}
    for _ in range(n_null_repeats):
        X_null = shuffle_neurons_independently(X, rng)
        df_null = cv_dimensionality(X_null, n_components_list, cv_folds=cv_folds, random_state=random_state)
        for method in METHODS:
            null_peak_scores[method].append(df_null[df_null.method == method]['cv_score_mean'].max())

    rows = []
    for method in METHODS:
        real_peak = df_real[df_real.method == method]['cv_score_mean'].max()
        null_mean = float(np.mean(null_peak_scores[method]))
        null_std = float(np.std(null_peak_scores[method]))
        rows.append({
            'method': method, 'real_peak_cv_score': real_peak,
            'null_peak_cv_score_mean': null_mean, 'null_peak_cv_score_std': null_std,
            'exceeds_null': bool(real_peak > null_mean + null_std),
        })
    return df_real, pd.DataFrame(rows)


#%% ------------------------------------------------------------------
# Load sessions (shallow)
# ----------------------------------------------------------------------
sessions, nSessions = filter_sessions(
    protocols=protocol, load_behaviordata=False, load_calciumdata=False,
    load_videodata=False, min_trials=100)
report_sessions(sessions)


#%% ------------------------------------------------------------------
# Per session: filter to the requested groups, then fit CV FA/PCA per
# group PER WINDOW (parallelized across sessions)
# ----------------------------------------------------------------------
def process_session(ses, session_index, nSessions, calciumversion, qc_csv_path,
                     target_areas, target_labels, s_pre, s_post, binsize, t_pre, t_post,
                     window_ranges, max_components_tested, cv_folds, n_null_repeats,
                     max_nan_frac_per_trial, random_state, min_neurons_per_group):
    print(f'\n=== Session {session_index + 1}/{nSessions}: {ses.session_id} ===')

    tensor, axis, axis_label = compute_tensor_for_session(
        ses, calciumversion=calciumversion,
        t_pre=t_pre, t_post=t_post,
        s_pre=s_pre, s_post=s_post, binsize=binsize, keepraw=False)
    # tensor: (K trials, N neurons, T bins)

    area, label, _ = get_area_label(ses.celldata)
    areas_present = sorted(np.unique(area))  # BEFORE filtering, so zero-count groups still get reported below

    # V1/PM unl cells must be near a labeled cell (50um) and in layer 2/3
    # (depth<300um); labeled cells and cells outside V1/PM pass through
    # unaffected (see utils/cellselection_lib.py). Combined with the QC
    # pass/fail flag from run_qc_summary.py, and with the requested
    # (area, label) group restriction.
    idx_anat = filter_nearlabeled_layer23(ses, radius=50, depth_thr=300)
    idx_qc = load_qc_pass(ses, qc_csv_path=qc_csv_path)
    idx_target = filter_target_groups(area, label, target_areas, target_labels)
    idx_valid = idx_anat & idx_qc & idx_target
    print(f'  [{ses.session_id}] {np.sum(idx_valid)}/{len(idx_valid)} neurons kept '
          f'(anatomical: {np.sum(idx_anat)}, QC: {np.sum(idx_qc)}, target group: {np.sum(idx_target)})')

    tensor = tensor[:, idx_valid, :]
    area, label = area[idx_valid], label[idx_valid]

    curves_rows = []
    summary_rows = []
    explained_var_rows = []
    group_status_rows = []   # EVERY candidate group x window, whether run or skipped, and why
    for window_name, (win_lo, win_hi) in window_ranges.items():
        window_mask = (axis >= win_lo) & (axis <= win_hi)
        if not np.any(window_mask):
            print(f'  [{ses.session_id}] WARNING: window "{window_name}"={win_lo, win_hi} matches no bins '
                  f'on this session\'s axis ({axis.min():.1f} to {axis.max():.1f}) -- skipping this window.')
            continue

        for a in areas_present:
            for l in DEFAULT_LABEL_ORDER:
                idx_group = np.where((area == a) & (label == l))[0]
                n_neurons_group = len(idx_group)

                if n_neurons_group < min_neurons_per_group:
                    status = 'skipped_too_few_neurons'
                    group_status_rows.append({'window': window_name, 'area': a, 'label': l,
                                               'n_neurons_group': n_neurons_group, 'n_trials': np.nan,
                                               'status': status})
                    continue

                X, trial_keep_idx = collapse_window(tensor[:, idx_group, :], window_mask,
                                                     max_nan_frac=max_nan_frac_per_trial)
                n_trials = X.shape[0]
                n_dropped_trials = int(np.sum(~trial_keep_idx))
                if n_dropped_trials > 0:
                    print(f'    [{window_name}] {a}-{l}: dropped {n_dropped_trials} trials '
                          f'(>{max_nan_frac_per_trial:.0%} NaN within this window)')

                # cap candidate n_components at what's actually estimable: strictly
                # less than both the number of neurons and the smallest CV training
                # fold size, and no more than max_components_tested
                n_train_per_fold = n_trials - int(np.ceil(n_trials / cv_folds))
                max_nc = min(max_components_tested, n_neurons_group - 1, n_train_per_fold - 1)
                if max_nc < 1:
                    status = 'skipped_insufficient_trials'
                    print(f'    [{window_name}] {a}-{l}: n_neurons={n_neurons_group}, n_trials={n_trials} '
                          f'-- SKIPPED (not enough trials/neurons for CV)')
                    group_status_rows.append({'window': window_name, 'area': a, 'label': l,
                                               'n_neurons_group': n_neurons_group, 'n_trials': n_trials,
                                               'status': status})
                    continue
                n_components_list = list(range(1, max_nc + 1))

                print(f'    [{window_name}] {a}-{l}: n_neurons={n_neurons_group}, n_trials={n_trials}, '
                      f'testing n_components up to {max_nc} -- fitting')
                group_status_rows.append({'window': window_name, 'area': a, 'label': l,
                                           'n_neurons_group': n_neurons_group, 'n_trials': n_trials,
                                           'status': 'fit'})
                df_curve, df_null_summary = dimensionality_significance_test(
                    X, n_components_list, cv_folds=cv_folds, n_null_repeats=n_null_repeats,
                    random_state=random_state)
                df_curve['window'] = window_name
                df_curve['area'] = a
                df_curve['label'] = l
                df_curve['n_neurons_group'] = n_neurons_group
                df_curve['n_trials'] = n_trials
                curves_rows.append(df_curve)

                # explained-variance (scree plot) diagnostic -- separate
                # from the CV log-likelihood fit above: this is a variance-
                # decomposition question ("how many components to capture
                # X% of variance"), not a generalization question, so it's
                # computed on the full data (no CV split needed)
                exp_var = compute_explained_variance(X, n_components=max_nc, random_state=random_state)
                for method_name in METHODS:
                    cumulative = np.cumsum(exp_var[method_name])
                    for ic, ratio in enumerate(exp_var[method_name]):
                        explained_var_rows.append({
                            'window': window_name, 'area': a, 'label': l, 'method': method_name,
                            'component': ic + 1, 'explained_variance_ratio': float(ratio),
                            'cumulative_variance_ratio': float(cumulative[ic]),
                            'n_neurons_group': n_neurons_group, 'n_trials': n_trials,
                        })

                nc_opt_by_method = {}
                for method_name in METHODS:
                    sub = df_curve[df_curve['method'] == method_name]
                    i_opt = sub['cv_score_mean'].idxmax()
                    nc_opt = int(sub.loc[i_opt, 'n_components'])
                    nc_opt_by_method[method_name] = nc_opt
                    score_opt = float(sub.loc[i_opt, 'cv_score_mean'])
                    at_boundary = (nc_opt == max_nc)
                    null_row = df_null_summary[df_null_summary.method == method_name].iloc[0]
                    summary_rows.append({
                        'window': window_name, 'area': a, 'label': l, 'method': method_name,
                        'n_components_optimal': nc_opt, 'cv_score_optimal': score_opt,
                        'n_neurons_group': n_neurons_group, 'n_trials': n_trials,
                        'at_grid_boundary': at_boundary,
                        'null_peak_cv_score_mean': null_row['null_peak_cv_score_mean'],
                        'null_peak_cv_score_std': null_row['null_peak_cv_score_std'],
                        'exceeds_null': null_row['exceeds_null'],
                    })
                    if at_boundary:
                        print(f'    WARNING: [{window_name}] {a}-{l} {method_name} optimal n_components hit '
                              f'the grid boundary ({nc_opt}) -- its true optimum may be higher than tested; '
                              f'consider raising max_components_tested before trusting this number.')
                    if not null_row['exceeds_null']:
                        print(f'    NOTE: [{window_name}] {a}-{l} {method_name} does NOT exceed the '
                              f'independent-neuron-shuffle null (Elsayed & Cunningham-style check) -- little '
                              f'evidence of genuine shared population structure beyond independent '
                              f'per-neuron variability here.')

                # PCA needing many more components than FA is the expected
                # signature of PCA's isotropic-noise assumption being wrong
                # for this data (real neurons have very heterogeneous private
                # variance; FA's per-neuron noise term absorbs that directly,
                # PCA can only absorb it by spending shared components on it)
                # -- NOT evidence PCA is "worse at reducing" the data. A large
                # ratio here means trust FA's number, not PCA's, as the more
                # meaningful shared-dimensionality estimate.
                if nc_opt_by_method['fa'] > 0:
                    ratio = nc_opt_by_method['pca'] / nc_opt_by_method['fa']
                    if ratio > 2:
                        print(f'    NOTE: [{window_name}] {a}-{l} PCA needs {ratio:.1f}x more components than '
                              f'FA ({nc_opt_by_method["pca"]} vs {nc_opt_by_method["fa"]}) -- likely reflects '
                              f'heterogeneous private per-neuron variance that PCA (isotropic noise) can only '
                              f'absorb via extra shared components, not genuine extra shared dimensionality. '
                              f'Trust FA\'s number here.')

                    # Direct check of that explanation: if z-scoring PCA's
                    # input closes most of the raw-PCA-vs-FA gap, it
                    # confirms the gap was about per-neuron variance scale,
                    # not genuine extra shared dimensionality. If the gap
                    # persists even after z-scoring, that points to some
                    # other PCA/FA mismatch instead (e.g. genuinely
                    # non-isotropic shared noise structure).
                    ratio_z = nc_opt_by_method['pca_zscored'] / nc_opt_by_method['fa']
                    print(f'    NOTE: [{window_name}] {a}-{l} pca_zscored needs {ratio_z:.1f}x more components '
                          f'than FA ({nc_opt_by_method["pca_zscored"]} vs {nc_opt_by_method["fa"]}), vs raw PCA\'s '
                          f'{ratio:.1f}x -- '
                          + ('z-scoring closed most of the gap, consistent with heterogeneous per-neuron variance '
                             'being the explanation.' if ratio_z < ratio / 1.5 else
                             'z-scoring did NOT close the gap much, suggesting the raw PCA-vs-FA gap here is not '
                             'primarily about per-neuron variance scale.'))

                # FA already carries a per-neuron noise-variance term, so
                # it's expected to be closer to scale-invariant than PCA --
                # a poorly-scaled neuron's private noise just gets absorbed
                # into its own noise parameter either way. Check that
                # expectation directly rather than assuming it.
                if nc_opt_by_method['fa'] > 0:
                    fa_z_ratio = nc_opt_by_method['fa_zscored'] / nc_opt_by_method['fa']
                    if abs(fa_z_ratio - 1) > 0.5:
                        print(f'    NOTE: [{window_name}] {a}-{l} fa_zscored optimal n_components '
                              f'({nc_opt_by_method["fa_zscored"]}) differs notably from raw FA\'s '
                              f'({nc_opt_by_method["fa"]}), {fa_z_ratio:.1f}x -- FA was expected to be roughly '
                              f'scale-invariant here; worth a closer look if you\'re relying on the raw FA number.')
                    else:
                        print(f'    [{window_name}] {a}-{l} fa_zscored optimal n_components '
                              f'({nc_opt_by_method["fa_zscored"]}) tracks raw FA\'s ({nc_opt_by_method["fa"]}) as '
                              f'expected -- FA\'s per-neuron noise term is doing its job regardless of input scale.')

    df_curves = pd.concat(curves_rows, ignore_index=True) if curves_rows else pd.DataFrame()
    df_summary = pd.DataFrame(summary_rows)
    df_explained_var = pd.DataFrame(explained_var_rows)
    df_group_status = pd.DataFrame(group_status_rows)
    if len(df_curves):
        df_curves['session_id'] = ses.session_id
    if len(df_summary):
        df_summary['session_id'] = ses.session_id
    if len(df_explained_var):
        df_explained_var['session_id'] = ses.session_id
    if len(df_group_status):
        df_group_status['session_id'] = ses.session_id

    for attr in ('calciumdata', 'videodata', 'behaviordata'):
        if hasattr(ses, attr):
            delattr(ses, attr)

    return {'curves': df_curves, 'summary': df_summary, 'explained_var': df_explained_var,
            'group_status': df_group_status}


session_results = Parallel(n_jobs=n_jobs_sessions, backend='loky', verbose=10)(
    delayed(process_session)(
        ses, ises, nSessions, calciumversion, qc_csv_path,
        target_areas, target_labels, s_pre, s_post, binsize, t_pre, t_post,
        window_ranges, max_components_tested, cv_folds, n_null_repeats,
        max_nan_frac_per_trial, random_state, min_neurons_per_group)
    for ises, ses in enumerate(sessions)
)

df_curves_all = pd.concat([r['curves'] for r in session_results if len(r['curves'])], ignore_index=True)
df_summary_all = pd.concat([r['summary'] for r in session_results if len(r['summary'])], ignore_index=True)
df_explained_var_all = pd.concat([r['explained_var'] for r in session_results if len(r['explained_var'])],
                                  ignore_index=True)
df_group_status_all = pd.concat([r['group_status'] for r in session_results if len(r['group_status'])],
                                 ignore_index=True)

df_curves_all.to_csv(os.path.join(output_dir, 'dimensionality_all_curves.csv'), index=False)
df_summary_all.to_csv(os.path.join(output_dir, 'dimensionality_optimal_summary.csv'), index=False)
df_explained_var_all.to_csv(os.path.join(output_dir, 'dimensionality_explained_variance.csv'), index=False)
df_group_status_all.to_csv(os.path.join(output_dir, 'dimensionality_group_status.csv'), index=False)

# this is the table to check FIRST whenever a group unexpectedly doesn't
# show up in the curves/summary above -- it lists every (window, group) in
# every session, whether it ran or was skipped, and why:
print('\nGroup status per session (check this first if a group is missing from the results):')
print(df_group_status_all.sort_values(['window', 'area', 'label', 'session_id']).to_string(index=False))
print('\nFraction of sessions each (window, group) was actually fit in (rest were skipped -- see status above):')
print((df_group_status_all.groupby(['window', 'area', 'label'])['status']
       .apply(lambda s: (s == 'fit').mean()).rename('frac_sessions_fit')).to_string())

print('\nOptimal n_components per session x window x group x method:')
print(df_summary_all.sort_values(['window', 'area', 'label', 'method']).to_string(index=False))

#%% ------------------------------------------------------------------
# Figure 1: CV score curves per group (FA, raw PCA, and z-scored PCA in
# separate columns), one line per session -- ONE FIGURE PER WINDOW
# ----------------------------------------------------------------------
groups_present = sorted(df_curves_all.groupby(['area', 'label']).groups.keys())
n_groups = len(groups_present)
n_methods = len(METHODS)
windows_present = [w for w in window_ranges if w in df_curves_all['window'].unique()]

for window_name in windows_present:
    df_w = df_curves_all[df_curves_all.window == window_name]
    fig, axes = plt.subplots(n_groups, n_methods, figsize=(5 * n_methods, 3 * n_groups), squeeze=False)

    for i, (a, l) in enumerate(groups_present):
        for j, method_name in enumerate(METHODS):
            ax = axes[i, j]
            sub = df_w[(df_w.area == a) & (df_w.label == l) & (df_w.method == method_name)]
            for sid, sub_ses in sub.groupby('session_id'):
                sub_ses = sub_ses.sort_values('n_components')
                ax.plot(sub_ses['n_components'], sub_ses['cv_score_mean'], alpha=0.6, linewidth=1)
            ax.set_title(f'{a} - {l} ({method_name.upper()})', fontsize=9)
            ax.set_xlabel('n_components', fontsize=8)
            if j == 0:
                ax.set_ylabel('CV log-likelihood', fontsize=8)

    fig.suptitle(f'window: {window_name} ({window_ranges[window_name][0]} to {window_ranges[window_name][1]})', y=1.01)
    fig.tight_layout()
    fig_path = os.path.join(output_dir, f'dimensionality_curves_{window_name}.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'Saved CV score curves ({window_name}) to {fig_path}')

#%% ------------------------------------------------------------------
# Figure 1b: explained-variance scree plots (per-component and cumulative),
# PCA and FA overlaid for direct comparison -- ONE FIGURE PER WINDOW.
# This is a variance-decomposition diagnostic (how many components to
# capture X% of variance), distinct from Figure 1's CV log-likelihood
# generalization question, and from Figure 2's argmax-based "optimal"
# count -- a population can need very few CV-optimal components yet still
# have variance spread across many, or vice versa.
# ----------------------------------------------------------------------
for window_name in windows_present:
    df_w = df_explained_var_all[df_explained_var_all.window == window_name]
    if not len(df_w):
        continue
    fig, axes = plt.subplots(n_groups, 2, figsize=(10, 3 * n_groups), squeeze=False)

    for i, (a, l) in enumerate(groups_present):
        sub_group = df_w[(df_w.area == a) & (df_w.label == l)]
        for j, (metric, ylabel) in enumerate([('explained_variance_ratio', 'Per-component variance ratio'),
                                               ('cumulative_variance_ratio', 'Cumulative variance ratio')]):
            ax = axes[i, j]
            for method_name, color in (('pca', 'tab:orange'), ('pca_zscored', 'tab:green'),
                                       ('fa', 'tab:blue'), ('fa_zscored', 'tab:purple')):
                sub = sub_group[sub_group.method == method_name]
                for sid, sub_ses in sub.groupby('session_id'):
                    sub_ses = sub_ses.sort_values('component')
                    ax.plot(sub_ses['component'], sub_ses[metric], color=color, alpha=0.4, linewidth=1)
                # session-mean overlay for readability
                mean_curve = sub.groupby('component')[metric].mean()
                ax.plot(mean_curve.index, mean_curve.values, color=color, linewidth=2,
                        label=method_name.upper())
            ax.set_title(f'{a} - {l}', fontsize=9)
            ax.set_xlabel('component', fontsize=8)
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=8)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)

    fig.suptitle(f'Explained variance -- window: {window_name} '
                 f'({window_ranges[window_name][0]} to {window_ranges[window_name][1]})', y=1.01)
    fig.tight_layout()
    fig_path = os.path.join(output_dir, f'dimensionality_explained_variance_{window_name}.png')
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'Saved explained-variance scree plot ({window_name}) to {fig_path}')

#%% ------------------------------------------------------------------
# Figure 2: optimal n_components per group, FA vs PCA, distribution
# across sessions -- ONE SUBPLOT PER WINDOW, side by side for direct
# comparison (this is the headline plot: does dimensionality differ by
# trial epoch?)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, len(windows_present), figsize=(max(6, 1.2 * n_groups) * len(windows_present), 5),
                          squeeze=False)
axes = axes[0]
group_labels = [f'{a}\n{l}' for a, l in groups_present]
x = np.arange(n_groups)
width = 0.8 / len(METHODS)

for w_idx, window_name in enumerate(windows_present):
    ax = axes[w_idx]
    df_w = df_summary_all[df_summary_all.window == window_name]
    for k, method_name in enumerate(METHODS):
        means, sems = [], []
        for a, l in groups_present:
            vals = df_w[(df_w.area == a) & (df_w.label == l)
                        & (df_w.method == method_name)]['n_components_optimal']
            means.append(vals.mean() if len(vals) else np.nan)
            sems.append(vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0)
        offset = (k - (len(METHODS) - 1) / 2) * width
        ax.bar(x + offset, means, width=width, yerr=sems, capsize=3, label=method_name.upper())

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, fontsize=8)
    ax.set_title(f'{window_name} ({window_ranges[window_name][0]} to {window_ranges[window_name][1]})', fontsize=10)
    if w_idx == 0:
        ax.set_ylabel('Optimal n_components (mean +/- SEM across sessions)')
    ax.legend(fontsize=8)

fig.suptitle(f'Cross-validated dimensionality by group and window ({"+".join(protocol)}, {calciumversion})', y=1.03)
fig.tight_layout()
fig_path = os.path.join(output_dir, 'dimensionality_optimal_summary.png')
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f'Saved optimal-dimensionality summary to {fig_path}')

print(f'\nSaved curves table to {os.path.join(output_dir, "dimensionality_all_curves.csv")}')
print(f'Saved summary table to {os.path.join(output_dir, "dimensionality_optimal_summary.csv")}')
