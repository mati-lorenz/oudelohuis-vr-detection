# -*- coding: utf-8 -*-
"""
Sanity-checks the neural recordings BEFORE they enter the information-
theoretic pipeline: flags silent neurons, likely artifacts (NaNs,
saturation/clipping, unstable trial-to-trial variance), and excessively
noisy cells, per utils/qc_lib.py -- and reports WHY each flagged cell was
flagged (which individual criterion tripped), broken down by session,
area, and label.

This is meant to be run first, alongside plot_spike_exploratory.py (that
script covers the richer activity-statistics diagnostics -- event rate,
Fano factor, reliability, etc. -- this one covers the more basic
recording-quality sanity check: is this session/plane/population usable
at all).

Produces:
  - qc_summary.png: pooled distributions of noise_level/rate/skew +
    fraction of cells passing per session
  - qc_reasons_by_session.png / qc_reasons_by_arealabel.png: stacked bar
    charts of how many cells were flagged and for which reason(s)
  - qc_by_arealabel.png: grid of per-(area,label) metric distributions,
    pass vs. flagged cells shown separately
  - qc_per_cell.csv: every qc_* column (including qc_fail_reason) for
    every cell
  - qc_group_summary.csv: pass/fail + per-reason counts, grouped by
    session x area x label
  - qc_per_session_summary.csv: same, collapsed to just session level
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
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

from loaddata.session_info import filter_sessions, report_sessions
from infotheory import compute_respmat_for_session
from qc_lib import (
    run_qc, summarize_qc, plot_qc_summary, plot_qc_reasons, plot_qc_by_group,
    RATE_THR, NOISE_THR, FANO_THR, SKEW_MIN, FLAT_STD_THR, REASON_COLS,
)

#%% ------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
protocol = ['DN']
calciumversion = 'deconv'

# QC thresholds -- see utils/qc_lib.py for what each one flags. Defaults
# imported above; override any of them here if this dataset needs
# different cutoffs (e.g. after inspecting a first pass of the figure).
qc_kwargs = dict(rate_thr=RATE_THR, noise_thr=NOISE_THR, fano_thr=FANO_THR,
                  skew_min=SKEW_MIN, flat_std_thr=FLAT_STD_THR)

output_dir = './output/infotheory_plots'
os.makedirs(output_dir, exist_ok=True)

#%% ------------------------------------------------------------------
# Load sessions (shallow), then load calcium data + response matrix per
# session -- qc_lib needs ses.calciumdata (for rate/std/skew/NaN checks)
# and ses.respmat (for the trial-to-trial Fano factor check)
# ----------------------------------------------------------------------
sessions, nSessions = filter_sessions(
    protocols=protocol, load_behaviordata=False, load_calciumdata=False,
    load_videodata=False, min_trials=100)
report_sessions(sessions)

for ises, ses in enumerate(sessions):
    print(f'\n=== Loading session {ises + 1}/{nSessions}: {ses.session_id} ===')
    ses.load_data(load_behaviordata=False, load_calciumdata=True,
                  load_videodata=False, calciumversion=calciumversion)
    compute_respmat_for_session(ses, calciumversion=calciumversion, keepraw=True)

#%% ------------------------------------------------------------------
# Run QC: adds qc_* columns + celldata['qc_pass'] to every session's
# celldata, and prints a one-line pass/fail summary per session
# ----------------------------------------------------------------------
sessions = run_qc(sessions, **qc_kwargs)

# free the raw traces now that the QC metrics have been computed off them
for ses in sessions:
    if hasattr(ses, 'calciumdata'):
        delattr(ses, 'calciumdata')

#%% ------------------------------------------------------------------
# Per-cell table (including WHY each flagged cell was flagged)
# ----------------------------------------------------------------------
celldata_all = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)
qc_cols = (['session_id', 'cell_id', 'roi_name', 'labeled', 'depth', 'noise_level',
            'qc_rate', 'qc_std', 'qc_skew', 'qc_fano', 'qc_nan_frac']
           + [f'qc_flag_{r}' for r in REASON_COLS]
           + ['qc_fail_reason', 'qc_silent', 'qc_artifact', 'qc_pass'])
qc_cols = [c for c in qc_cols if c in celldata_all.columns]
celldata_all[qc_cols].to_csv(os.path.join(output_dir, 'qc_per_cell.csv'), index=False)

#%% ------------------------------------------------------------------
# Detailed group summary: pass/fail + per-reason counts, by session x
# area x label (this is the main "why is X flagged" table)
# ----------------------------------------------------------------------
group_summary = summarize_qc(sessions, groupby=('session_id', 'roi_name', 'labeled'))
group_summary.to_csv(os.path.join(output_dir, 'qc_group_summary.csv'), index=False)

print('\nQC summary by session x area x label (worst first):')
print(group_summary.to_string(index=False))

# same thing collapsed to just session level, for a quick top-line check
session_summary = summarize_qc(sessions, groupby=('session_id',))
session_summary.to_csv(os.path.join(output_dir, 'qc_per_session_summary.csv'), index=False)

print('\nQC summary by session (worst first):')
print(session_summary.to_string(index=False))

flagged_sessions = session_summary[session_summary['frac_pass'] < 0.5]
if len(flagged_sessions):
    print(f'\nWARNING: {len(flagged_sessions)} session(s) have <50% of cells passing QC -- '
          f'consider excluding or re-checking these:')
    print(flagged_sessions['session_id'].to_string(index=False))

flagged_groups = group_summary[group_summary['frac_pass'] < 0.5]
if len(flagged_groups):
    print(f'\nWARNING: {len(flagged_groups)} session x area x label group(s) have <50% passing:')
    print(flagged_groups[['session_id', 'roi_name', 'labeled', 'n_total', 'frac_pass']]
          .to_string(index=False))

#%% ------------------------------------------------------------------
# Figures: pooled summary, failure-reason breakdown (by session and by
# area/label), and per-(area,label) metric distributions split by
# pass/flagged
# ----------------------------------------------------------------------
fig = plot_qc_summary(sessions)
fig_path = os.path.join(output_dir, 'qc_summary.png')
fig.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f'\nSaved QC summary figure to {fig_path}')

fig_reasons_session = plot_qc_reasons(sessions, groupby='session_id')
fig_reasons_session_path = os.path.join(output_dir, 'qc_reasons_by_session.png')
fig_reasons_session.savefig(fig_reasons_session_path, dpi=150, bbox_inches='tight')
print(f'Saved per-session failure-reason figure to {fig_reasons_session_path}')

fig_reasons_arealabel = plot_qc_reasons(sessions, groupby=['roi_name', 'labeled'])
fig_reasons_arealabel_path = os.path.join(output_dir, 'qc_reasons_by_arealabel.png')
fig_reasons_arealabel.savefig(fig_reasons_arealabel_path, dpi=150, bbox_inches='tight')
print(f'Saved per-area/label failure-reason figure to {fig_reasons_arealabel_path}')

fig_by_group = plot_qc_by_group(sessions, groupby=('roi_name', 'labeled'))
fig_by_group_path = os.path.join(output_dir, 'qc_by_arealabel.png')
fig_by_group.savefig(fig_by_group_path, dpi=150, bbox_inches='tight')
print(f'Saved per-area/label metric-distribution figure to {fig_by_group_path}')

print(f'\nSaved per-cell QC table to {os.path.join(output_dir, "qc_per_cell.csv")}')
print(f'Saved session x area x label QC table to {os.path.join(output_dir, "qc_group_summary.csv")}')
print(f'Saved per-session QC summary to {os.path.join(output_dir, "qc_per_session_summary.csv")}')
