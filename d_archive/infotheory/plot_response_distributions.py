# -*- coding: utf-8 -*-
"""
Plot single-trial response distributions, split by brain area and
labeling status (unl / lab), to help choose the binning parameters
(`n_bins`, equipopulated vs equal-width) used throughout the
information-theoretic pipeline (infotheory/discretize.py).

For each (area, label) group, pools single-trial responses across all
neurons in that group (and across all loaded sessions), and plots:
  1. a histogram of the pooled response distribution, with the
     equipopulated bin edges for a few candidate `n_bins` overlaid
  2. a summary table (saved as CSV) with basic stats per group and a
     recommended `n_bins` based on trial counts (see
     infotheory.discretize.choose_n_bins)

Figures and the summary CSV are saved to `output_dir`.
"""

#%%
import os, sys
os.chdir('/u/g/glorenz/Documents/Research/Code/external_repos/oudelohuis-vr-detection')
sys.path.insert(0, '/u/g/glorenz/Documents/Research/Code/external_repos/oudelohuis-vr-detection')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from loaddata.session_info import filter_sessions, report_sessions
from infotheory import compute_respmat_for_session
from infotheory.celldata_utils import (
    get_area_label, ordered_groups, AREA_COLORS, DEFAULT_LABEL_ORDER)
from infotheory.discretize import equipopulated_edges, choose_n_bins
from utils.cellselection_lib import filter_nearlabeled_layer23
from qc_lib import load_qc_pass

#%% ------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
protocol = ['DN']
calciumversion = 'deconv'

candidate_n_bins = [2, 3, 4, 5]     # bin-edge overlays to compare
n_hist_bins = 100                    # resolution of the plotted histogram itself
max_samples_per_group = 300_000      # subsample cap per (area,label) group, for memory/speed
trials_per_bin_target = 8            # used only for the recommended-n_bins summary
random_state = 0

output_dir = './output/infotheory_plots'
os.makedirs(output_dir, exist_ok=True)

# n_jobs at the SESSION level (primary parallelism, one worker per session).
n_jobs_sessions = -1

# QC filter (in addition to the anatomical V1/PM filter): produced by
# run_qc_summary.py. Run that script first; set to None to skip QC
# filtering (e.g. for a quick look before QC has been run).
qc_csv_path = os.path.join(output_dir, 'qc_per_cell.csv')

#%% ------------------------------------------------------------------
# Load sessions and compute respmat (protocol-aware: time-locked window
# for IM/GR/GN, spatial window for VR/DM/DN/DP)
# ----------------------------------------------------------------------
sessions, nSessions = filter_sessions(
    protocols=protocol, load_behaviordata=False, load_calciumdata=False,
    load_videodata=False, min_trials=100)
report_sessions(sessions)

rng = np.random.default_rng(random_state)

#%% ------------------------------------------------------------------
# Per-session computation (parallelized across sessions)
# ----------------------------------------------------------------------
def process_session(ses, session_index, nSessions, calciumversion, qc_csv_path):
    print(f'\n=== Session {session_index + 1}/{nSessions}: {ses.session_id} ===')
    respmat = compute_respmat_for_session(ses, calciumversion=calciumversion, keepraw=False)
    respmat = np.asarray(respmat)  # (N neurons, K trials)

    area, label, group = get_area_label(ses.celldata)

    # V1/PM unl cells must be near a labeled cell (50um) and in layer 2/3
    # (depth<300um); labeled cells and cells outside V1/PM pass through
    # unaffected (see utils/cellselection_lib.py). Combined with the QC
    # pass/fail flag from run_qc_summary.py (silent/artifact/noisy cells).
    idx_anat = filter_nearlabeled_layer23(ses, radius=50, depth_thr=300)
    idx_qc = load_qc_pass(ses, qc_csv_path=qc_csv_path)
    idx_valid = idx_anat & idx_qc
    print(f'  [{ses.session_id}] {np.sum(idx_valid)}/{len(idx_valid)} neurons kept '
          f'(anatomical: {np.sum(idx_anat)}, QC: {np.sum(idx_qc)})')
    respmat = respmat[idx_valid, :]
    area, label, group = area[idx_valid], label[idx_valid], group[idx_valid]

    n_trials = respmat.shape[1]

    pooled_local = {}
    n_trials_local = {}
    for a in np.unique(area):
        for l in DEFAULT_LABEL_ORDER:
            idx = np.where((area == a) & (label == l))[0]
            if len(idx) == 0:
                continue
            vals = respmat[idx, :].ravel()
            vals = vals[~np.isnan(vals)]
            key = (a, l)
            pooled_local[key] = vals
            n_trials_local[key] = n_trials

    return {'pooled': pooled_local, 'n_trials': n_trials_local}


session_results = Parallel(n_jobs=n_jobs_sessions, backend='loky', verbose=10)(
    delayed(process_session)(ses, ises, nSessions, calciumversion, qc_csv_path)
    for ises, ses in enumerate(sessions)
)

#%% ------------------------------------------------------------------
# Pool across sessions per (area, label) group
# ----------------------------------------------------------------------
# pooled[(area, label)] -> list of 1D arrays of single-trial responses
pooled = {}
# per-neuron trial counts (for the recommended-n_bins summary), keyed the same way
n_trials_list = {}

for res in session_results:
    for key, vals in res['pooled'].items():
        pooled.setdefault(key, []).append(vals)
        n_trials_list.setdefault(key, []).append(res['n_trials'][key])

pooled = {k: np.concatenate(v) for k, v in pooled.items()}

# subsample very large groups for plotting speed / memory
for k, v in pooled.items():
    if len(v) > max_samples_per_group:
        pooled[k] = rng.choice(v, size=max_samples_per_group, replace=False)

#%% ------------------------------------------------------------------
# Summary table: basic stats + recommended n_bins per group
# ----------------------------------------------------------------------
summary_rows = []
for (a, l), vals in pooled.items():
    n_min_trials = int(np.min(n_trials_list[(a, l)]))
    rec_n_bins = choose_n_bins([n_min_trials], trials_per_bin_target=trials_per_bin_target)
    summary_rows.append({
        'area': a, 'label': l,
        'n_samples_pooled': len(vals),
        'frac_zero': float(np.mean(vals == 0)),
        'mean': float(np.mean(vals)),
        'std': float(np.std(vals)),
        'median': float(np.median(vals)),
        'p95': float(np.percentile(vals, 95)),
        'recommended_n_bins': rec_n_bins,
    })
df_summary = pd.DataFrame(summary_rows).sort_values(['area', 'label'])
df_summary.to_csv(os.path.join(output_dir, 'response_distribution_summary.csv'), index=False)
print(df_summary.to_string(index=False))

#%% ------------------------------------------------------------------
# Grid of histograms: rows = area, columns = label, with equipopulated
# bin-edge overlays for each candidate n_bins
# ----------------------------------------------------------------------
groups_present = list(pooled.keys())
areas_present = sorted(set(a for a, l in groups_present))
pairs = ordered_groups(areas_present)

n_areas = len(areas_present)
n_labels = len(DEFAULT_LABEL_ORDER)

fig, axes = plt.subplots(n_areas, n_labels, figsize=(5 * n_labels, 3 * n_areas),
                          squeeze=False, sharex=False)

edge_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(candidate_n_bins)))

for a, l in pairs:
    if (a, l) not in pooled:
        continue
    i = areas_present.index(a)
    j = DEFAULT_LABEL_ORDER.index(l)
    ax = axes[i, j]
    vals = pooled[(a, l)]

    ax.hist(vals, bins=n_hist_bins, color=AREA_COLORS.get(a, 'gray'),
            alpha=0.7, density=True)

    for nb, ec in zip(candidate_n_bins, edge_colors):
        edges, n_actual = equipopulated_edges(vals, nb)
        for e in edges[1:-1]:
            ax.axvline(e, color=ec, linestyle='--', linewidth=1, alpha=0.8)

    row = df_summary[(df_summary.area == a) & (df_summary.label == l)].iloc[0]
    ax.set_title(f'{a} - {l}  (n={row.n_samples_pooled:,}, '
                 f'%zero={row.frac_zero:.0%}, rec. n_bins={row.recommended_n_bins})',
                 fontsize=9)
    ax.set_xlabel('Response (a.u.)')
    ax.set_ylabel('Density')

# shared legend for the bin-edge overlays
legend_handles = [plt.Line2D([0], [0], color=ec, linestyle='--', label=f'n_bins={nb}')
                   for nb, ec in zip(candidate_n_bins, edge_colors)]
fig.legend(handles=legend_handles, loc='upper center', ncol=len(candidate_n_bins),
           bbox_to_anchor=(0.5, 1.02), fontsize=9)

fig.suptitle(f'Response distributions by area & label ({"+".join(protocol)}, '
             f'{calciumversion}, {nSessions} sessions)', y=1.06, fontsize=12)
fig.tight_layout()

fig_path = os.path.join(output_dir, 'response_distributions_by_area_label.png')
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f'\nSaved figure to {fig_path}')
print(f'Saved summary table to {os.path.join(output_dir, "response_distribution_summary.csv")}')
plt.close(fig)
