# -*- coding: utf-8 -*-
"""
Heteroscedasticity scatter plots: per-neuron noise statistics (mean rate,
CV, Fano factor, std, skew) plotted against each other in a full pairplot
grid, POOLED across all sessions -- to visually assess whether
distinguishable neuron subpopulations exist. Directly motivated by
run_dimensionality_estimate.py's finding that PCA needs far more
components than FA to fit as well, which happens BECAUSE PCA assumes
uniform per-neuron noise while FA doesn't; if the population is a mixture
of neurons with very different noise statistics, that heterogeneity is
exactly what would show up here as separable point clouds.

Points can be colored by SESSION (categorical -- a necessary sanity check
when pooling: are apparent clusters real subpopulations, or just an
artifact of different sessions having different overall noise levels?)
or by each neuron's own loading on a chosen PCA/FA component from that
session's fit.

IMPORTANT on pooling loadings across sessions: raw loadings are session-
specific (arbitrary sign and scale -- session A's PC1 and session B's PC1
are not on the same numeric scale). To make a single pooled colorbar
meaningful, each session's loading is rescaled by its own max-absolute
value before pooling (validated: brings sessions with raw ranges like
[-0.29, 0.67] and [-5.11, 4.54] onto a comparable [-1, 1] scale). This
shows "how extreme is this neuron's loading RELATIVE TO other neurons in
its own session's fit" -- a relative-rank comparison, not a claim that
the underlying components are the same functional axis across sessions.

Uses the SAME group-selection convention as the rest of this pipeline:
filter_nearlabeled_layer23 (anatomical) + load_qc_pass (QC) +
filter_target_groups (which area/label groups you want), and the SAME
window-collapsing convention as Step 1 (a single trial-epoch window,
default matching Step 1's 'response' window).

Produces:
  - heteroscedasticity_stats.csv: per-neuron mean_rate, std, cv, fano,
    skew, and PCA/FA loadings (both raw and session-normalized) on the
    first few components, pooled across all sessions
  - heteroscedasticity_pairplot_<area>_<label>.png : full pairplot of all
    stat pairs, pooled across sessions, colored by session
  - heteroscedasticity_loadings_<area>_<label>.png : CV vs Fano factor
    specifically, pooled across sessions, one panel per component/method
    coloring (PC1, PC2, FA1, FA2), using session-normalized loadings
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
from scipy.stats import skew

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

target_areas = None
target_labels = None

qc_csv_path = os.path.join('./output/infotheory_plots', 'qc_per_cell.csv')

s_pre, s_post, binsize = -60, 80, 10
t_pre, t_post = -1, 2

# which trial-epoch window to compute stats/loadings within -- matches
# Step 1's 'response' window by default (see run_dimensionality_estimate.py)
window_range = (15, 50)
max_nan_frac_per_trial = 0.5

n_components_for_color = 2   # color by loadings on components 1..this
min_neurons_per_group = 10
random_state = 0
n_jobs_sessions = -1

output_dir = './output/dimred_plots'
os.makedirs(output_dir, exist_ok=True)


#%% ------------------------------------------------------------------
# Per-neuron statistics
# ----------------------------------------------------------------------
def compute_neuron_stats(X):
    """
    X : (K trials, N neurons). Returns a dict of length-N arrays:
    mean_rate, std, cv (std/mean), fano (var/mean), skew.
    """
    mean_rate = X.mean(axis=0)
    std = X.std(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        cv = std / mean_rate
        fano = (std ** 2) / mean_rate
    sk = skew(X, axis=0, nan_policy='omit')
    return {'mean_rate': mean_rate, 'std': std, 'cv': cv, 'fano': fano, 'skew': sk}


def collapse_window(tensor, window_mask, max_nan_frac=0.5):
    """Same as run_dimensionality_estimate.py's helper -- see that file for details."""
    sub = tensor[:, :, window_mask]
    frac_nan_per_trial = np.mean(np.isnan(sub), axis=(1, 2))
    trial_keep_idx = frac_nan_per_trial <= max_nan_frac
    with np.errstate(invalid='ignore'):
        X = np.nanmean(sub[trial_keep_idx], axis=2)
    return X, trial_keep_idx


def compute_loadings(X, n_components, random_state=0):
    """
    X : (K trials, N neurons), NOT yet z-scored. Z-scores internally (PCA/FA
    loadings need a common per-neuron scale, same convention as Steps 3-4).
    Returns dict: {'pca': (n_components, N), 'fa': (n_components, N),
    'pca_norm': same but each row rescaled by its own max-abs value (for
    pooling across sessions onto one colorbar -- see module docstring),
    'fa_norm': same for FA}.
    """
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    Xz = (X - mu) / sd

    n_components = min(n_components, X.shape[1] - 1, X.shape[0] - 1)
    pca = PCA(n_components=n_components, random_state=random_state).fit(Xz)
    fa = FactorAnalysis(n_components=n_components, random_state=random_state).fit(Xz)

    def normalize_rows(M):
        max_abs = np.max(np.abs(M), axis=1, keepdims=True)
        max_abs[max_abs == 0] = 1.0
        return M / max_abs

    return {'pca': pca.components_, 'fa': fa.components_,
            'pca_norm': normalize_rows(pca.components_), 'fa_norm': normalize_rows(fa.components_)}


#%% ------------------------------------------------------------------
# Load sessions (shallow)
# ----------------------------------------------------------------------
sessions, nSessions = filter_sessions(
    protocols=protocol, load_behaviordata=False, load_calciumdata=False,
    load_videodata=False, min_trials=100)
report_sessions(sessions)


#%% ------------------------------------------------------------------
# Per session: filter to requested groups, compute stats + loadings
# ----------------------------------------------------------------------
def process_session(ses, session_index, nSessions, calciumversion, qc_csv_path,
                     target_areas, target_labels, s_pre, s_post, binsize, t_pre, t_post,
                     window_range, max_nan_frac_per_trial, n_components_for_color,
                     min_neurons_per_group, random_state):
    print(f'\n=== Session {session_index + 1}/{nSessions}: {ses.session_id} ===')

    tensor, axis, axis_label = compute_tensor_for_session(
        ses, calciumversion=calciumversion,
        t_pre=t_pre, t_post=t_post,
        s_pre=s_pre, s_post=s_post, binsize=binsize, keepraw=False)

    area, label, _ = get_area_label(ses.celldata)

    idx_anat = filter_nearlabeled_layer23(ses, radius=50, depth_thr=300)
    idx_qc = load_qc_pass(ses, qc_csv_path=qc_csv_path)
    idx_target = filter_target_groups(area, label, target_areas, target_labels)
    idx_valid = idx_anat & idx_qc & idx_target
    print(f'  [{ses.session_id}] {np.sum(idx_valid)}/{len(idx_valid)} neurons kept '
          f'(anatomical: {np.sum(idx_anat)}, QC: {np.sum(idx_qc)}, target group: {np.sum(idx_target)})')

    tensor = tensor[:, idx_valid, :]
    area, label = area[idx_valid], label[idx_valid]
    cell_ids = ses.celldata['cell_id'].to_numpy()[idx_valid] if 'cell_id' in ses.celldata.columns else \
        np.arange(len(area))

    window_mask = (axis >= window_range[0]) & (axis <= window_range[1])
    if not np.any(window_mask):
        print(f'  [{ses.session_id}] WARNING: window_range={window_range} matches no bins on this session\'s '
              f'axis ({axis.min():.1f} to {axis.max():.1f}) -- skipping.')
        return {'session_id': ses.session_id, 'rows': []}

    rows = []
    for a in np.unique(area):
        for l in DEFAULT_LABEL_ORDER:
            idx_group = np.where((area == a) & (label == l))[0]
            n_neurons_group = len(idx_group)
            if n_neurons_group < min_neurons_per_group:
                continue

            X, trial_keep_idx = collapse_window(tensor[:, idx_group, :], window_mask,
                                                 max_nan_frac=max_nan_frac_per_trial)
            n_trials = X.shape[0]
            if n_trials < 10:
                print(f'    {a}-{l}: skipping, only {n_trials} trials survive NaN cleaning in this window')
                continue

            stats = compute_neuron_stats(X)
            loadings = compute_loadings(X, n_components=n_components_for_color, random_state=random_state)
            n_comp_fit = loadings['pca'].shape[0]

            print(f'    {a}-{l}: n_neurons={n_neurons_group}, n_trials={n_trials}, '
                  f'CV range={np.nanmin(stats["cv"]):.2f}-{np.nanmax(stats["cv"]):.2f}, '
                  f'Fano range={np.nanmin(stats["fano"]):.2f}-{np.nanmax(stats["fano"]):.2f}')

            group_ids = cell_ids[idx_group]
            for i in range(n_neurons_group):
                row = {
                    'session_id': ses.session_id, 'area': a, 'label': l, 'cell_id': group_ids[i],
                    'mean_rate': stats['mean_rate'][i], 'std': stats['std'][i],
                    'cv': stats['cv'][i], 'fano': stats['fano'][i], 'skew': stats['skew'][i],
                }
                for c in range(n_comp_fit):
                    row[f'pca_loading_{c + 1}'] = loadings['pca'][c, i]
                    row[f'fa_loading_{c + 1}'] = loadings['fa'][c, i]
                    row[f'pca_loading_{c + 1}_norm'] = loadings['pca_norm'][c, i]
                    row[f'fa_loading_{c + 1}_norm'] = loadings['fa_norm'][c, i]
                rows.append(row)

    for attr in ('calciumdata', 'videodata', 'behaviordata'):
        if hasattr(ses, attr):
            delattr(ses, attr)

    return {'session_id': ses.session_id, 'rows': rows}


session_results = Parallel(n_jobs=n_jobs_sessions, backend='loky', verbose=10)(
    delayed(process_session)(
        ses, ises, nSessions, calciumversion, qc_csv_path,
        target_areas, target_labels, s_pre, s_post, binsize, t_pre, t_post,
        window_range, max_nan_frac_per_trial, n_components_for_color,
        min_neurons_per_group, random_state)
    for ises, ses in enumerate(sessions)
)

all_rows = [row for res in session_results for row in res['rows']]
df = pd.DataFrame(all_rows)
if not len(df):
    print('\nNo groups had enough neurons/trials to compute statistics -- check group_status-style '
          'diagnostics from run_dimensionality_estimate.py for this dataset.')
else:
    df.to_csv(os.path.join(output_dir, 'heteroscedasticity_stats.csv'), index=False)
    print(f'\nSaved per-neuron statistics to {os.path.join(output_dir, "heteroscedasticity_stats.csv")}')

    #%% ------------------------------------------------------------------
    # Figure A: full pairplot of ALL stat pairs, POOLED across sessions,
    # colored by session -- both a richer view than just CV vs Fano, and a
    # necessary sanity check when pooling: apparent clusters should NOT
    # simply track session identity, or they're a session-level artifact
    # (different overall noise/SNR per session), not a real subpopulation.
    # Axis limits are clipped to the 1st-99th percentile per stat so a
    # handful of outlier neurons don't compress the rest of the cloud.
    # ----------------------------------------------------------------------
    stats_to_plot = ['mean_rate', 'std', 'cv', 'fano', 'skew']

    def _percentile_limits(values, lo=1, hi=99, pad=0.05):
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return (0, 1)
        p_lo, p_hi = np.percentile(values, [lo, hi])
        span = p_hi - p_lo
        return (p_lo - pad * span, p_hi + pad * span) if span > 0 else (p_lo - 1, p_hi + 1)

    for (a, l), df_group in df.groupby(['area', 'label']):
        n_stats = len(stats_to_plot)
        fig, axes = plt.subplots(n_stats, n_stats, figsize=(2.3 * n_stats, 2.3 * n_stats))
        session_ids = sorted(df_group['session_id'].unique())
        cmap_sessions = plt.cm.tab10(np.linspace(0, 1, max(len(session_ids), 2)))
        session_color = dict(zip(session_ids, cmap_sessions))

        limits = {s: _percentile_limits(df_group[s].to_numpy()) for s in stats_to_plot}

        for i, stat_y in enumerate(stats_to_plot):
            for j, stat_x in enumerate(stats_to_plot):
                ax = axes[i, j]
                if i == j:
                    ax.hist(df_group[stat_x].clip(*limits[stat_x]).dropna(), bins=30, color='tab:gray')
                else:
                    for sid in session_ids:
                        sub = df_group[df_group.session_id == sid]
                        ax.scatter(sub[stat_x], sub[stat_y], s=10, alpha=0.6,
                                   color=session_color[sid], label=sid if (i, j) == (1, 0) else None)
                    ax.set_xlim(limits[stat_x])
                    ax.set_ylim(limits[stat_y])
                if i == n_stats - 1:
                    ax.set_xlabel(stat_x, fontsize=8)
                if j == 0:
                    ax.set_ylabel(stat_y, fontsize=8)
                ax.tick_params(labelsize=6)

        handles, labels_ = axes[1, 0].get_legend_handles_labels()
        fig.legend(handles, labels_, loc='upper center', ncol=min(len(session_ids), 5),
                   bbox_to_anchor=(0.5, 1.02), fontsize=7)
        fig.suptitle(f'{a} - {l}: per-neuron statistics, pooled across sessions '
                     f'(window {window_range})', y=1.06)
        fig.tight_layout()
        fig_path = os.path.join(output_dir, f'heteroscedasticity_pairplot_{a}_{l}.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved pooled pairplot for {a}-{l} to {fig_path}')

    #%% ------------------------------------------------------------------
    # Figure B: CV vs Fano factor specifically, POOLED across sessions,
    # colored by session-normalized component loading (see module
    # docstring for why normalization is necessary before pooling loadings)
    # ----------------------------------------------------------------------
    color_options = [f'pca_loading_{c}_norm' for c in range(1, n_components_for_color + 1)] + \
        [f'fa_loading_{c}_norm' for c in range(1, n_components_for_color + 1)]
    color_labels = [f'PC{c} (normalized)' for c in range(1, n_components_for_color + 1)] + \
        [f'FA{c} (normalized)' for c in range(1, n_components_for_color + 1)]

    for (a, l), df_group in df.groupby(['area', 'label']):
        n_cols = len(color_options)
        fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 3.2), squeeze=False)
        axes = axes[0]
        cv_lim = _percentile_limits(df_group['cv'].to_numpy())
        fano_lim = _percentile_limits(df_group['fano'].to_numpy())

        for j, (color_col, color_label) in enumerate(zip(color_options, color_labels)):
            ax = axes[j]
            sca = ax.scatter(df_group['cv'], df_group['fano'], s=10, alpha=0.7,
                              c=df_group[color_col], cmap='RdBu_r', vmin=-1, vmax=1)
            plt.colorbar(sca, ax=ax, fraction=0.046, pad=0.04)
            ax.set_xlim(cv_lim)
            ax.set_ylim(fano_lim)
            ax.set_title(color_label, fontsize=9)
            ax.set_xlabel('CV', fontsize=8)
            if j == 0:
                ax.set_ylabel('Fano factor', fontsize=8)

        fig.suptitle(f'{a} - {l}: CV vs Fano, colored by session-normalized loading, '
                     f'pooled across {len(df_group["session_id"].unique())} sessions '
                     f'(window {window_range})', y=1.06)
        fig.tight_layout()
        fig_path = os.path.join(output_dir, f'heteroscedasticity_loadings_{a}_{l}.png')
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved pooled loading-colored scatter for {a}-{l} to {fig_path}')
