# -*- coding: utf-8 -*-
"""
This script runs stage 1 of the information-theoretic analysis pipeline
(single-cell + cell-pair information about stimulus and choice) over one
or more sessions, and saves the results to disk.

Matthijs Oude Lohuis / G. Lorenz style loader usage, see
loaddata/just_load_a_session.py for the reference loading pattern.
"""

#%%
import os, sys
os.chdir('/u/g/glorenz/Documents/Research/Code/external_repos/oudelohuis-vr-detection')
sys.path.insert(0, '/u/g/glorenz/Documents/Research/Code/external_repos/oudelohuis-vr-detection')

import numpy as np
import pandas as pd

from loaddata.session_info import filter_sessions, report_sessions
from infotheory import (
    InfoTheoryParams, BinningParams, BiasCorrectionParams, ParallelParams,
    run_session_single_cell_analysis, compute_respmat_for_session,
)
from utils.cellselection_lib import filter_nearlabeled_layer23
from qc_lib import load_qc_pass

#%% ------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
protocol = ['DN']

params = InfoTheoryParams(
    binning=BinningParams(method='equipopulated', n_bins=3, auto_nbins=False),
    bias=BiasCorrectionParams(panzeri_treves=True, shuffle_bias_estimate=True,
                               n_shuffles=500, alternative='greater', random_state=0),
    parallel=ParallelParams(n_jobs=-1, backend='loky', verbose=5),
    stim_var='signal',        # e.g. 'S' (signal) vs 'N' (noise) trials
    stim_binarize_threshold=0,
    choice_var='lickResponse', # animal's binary choice
    trial_mask_var=None,       # e.g. 'engaged' if you want to restrict trials
    min_trials_per_class=5,
)

do_pairwise = True     # also run the Pola et al. (2003) pairwise breakdown
max_pairs = 2000        # cap on number of pairs per session (random subsample)

output_dir = './output/infotheory_stage1'
os.makedirs(output_dir, exist_ok=True)

# QC filter (in addition to the anatomical V1/PM filter): produced by
# run_qc_summary.py. Run that script first; set to None to skip QC
# filtering (e.g. for a quick look before QC has been run). Points at
# infotheory_plots (not infotheory_stage1) since that's where
# run_qc_summary.py writes it.
qc_csv_path = os.path.join('./output/infotheory_plots', 'qc_per_cell.csv')

#%% ------------------------------------------------------------------
# Find sessions (shallow: sessiondata/trialdata/celldata only, no
# calcium/behavior traces yet -- those are loaded per-session below via
# compute_respmat_for_session, which knows how to build the trial
# response matrix correctly whether the protocol is time-locked
# ('IM'/'GR'/'GN') or spatial/VR-based ('VR'/'DM'/'DN'/'DP').
# ----------------------------------------------------------------------
sessions, nSessions = filter_sessions(
    protocols=protocol,
    load_behaviordata=False, load_calciumdata=False, load_videodata=False,
    min_trials=100,
)
report_sessions(sessions)

#%% ------------------------------------------------------------------
# Compute the trial response matrix (N neurons x K trials) for each
# session and run the stage-1 information analysis
# ----------------------------------------------------------------------
all_single_cell = []
all_pairwise = []

for ises, ses in enumerate(sessions):
    print(f'\n=== Session {ises + 1}/{nSessions}: {ses.session_id} ===')

    # DN/DM/DP (and VR) are spatial-corridor protocols: response window
    # is defined in cm relative to the stimulus zone (s_resp_start=0,
    # s_resp_stop=20 matches loaddata.session_info.load_neural_performing_sessions).
    # IM/GR/GN would instead use Session.load_respmat's time-locked window.
    compute_respmat_for_session(
        ses, calciumversion='deconv',
        s_resp_start=0, s_resp_stop=20, keepraw=False)

    # V1/PM unl cells must be near a labeled cell (50um) and in layer 2/3
    # (depth<300um); labeled cells and cells outside V1/PM pass through
    # unaffected (see utils/cellselection_lib.py). Combined with the QC
    # pass/fail flag from run_qc_summary.py (silent/artifact/noisy cells).
    # run_session_single_cell_analysis reads directly from ses.celldata/
    # ses.respmat, so filtering has to happen by subsetting ses itself, in
    # place, before calling it -- both filters must be computed BEFORE
    # ses.celldata is overwritten below.
    idx_anat = filter_nearlabeled_layer23(ses, radius=50, depth_thr=300, lateral_only=True)
    idx_qc = load_qc_pass(ses, qc_csv_path=qc_csv_path)
    idx_valid = idx_anat & idx_qc
    print(f'  [{ses.session_id}] {np.sum(idx_valid)}/{len(idx_valid)} neurons kept '
          f'(anatomical: {np.sum(idx_anat)}, QC: {np.sum(idx_qc)})')
    ses.celldata = ses.celldata.loc[idx_valid].reset_index(drop=True)
    ses.respmat = np.asarray(ses.respmat)[idx_valid, :]

    results = run_session_single_cell_analysis(
        ses, params=params, do_pairwise=do_pairwise, max_pairs=max_pairs)

    all_single_cell.append(results['single_cell'])
    if do_pairwise:
        all_pairwise.append(results['pairwise'])

    # save per-session results as we go, in case a later session fails
    results['single_cell'].to_csv(
        os.path.join(output_dir, f'{ses.session_id}_singlecell_info.csv'), index=False)
    if do_pairwise:
        results['pairwise'].to_csv(
            os.path.join(output_dir, f'{ses.session_id}_pairwise_breakdown.csv'), index=False)

#%% ------------------------------------------------------------------
# Concatenate across sessions and save combined tables
# ----------------------------------------------------------------------
df_singlecell_all = pd.concat(all_single_cell, ignore_index=True)
df_singlecell_all.to_csv(os.path.join(output_dir, 'ALL_sessions_singlecell_info.csv'), index=False)

if do_pairwise:
    df_pairwise_all = pd.concat(all_pairwise, ignore_index=True)
    df_pairwise_all.to_csv(os.path.join(output_dir, 'ALL_sessions_pairwise_breakdown.csv'), index=False)

#%% ------------------------------------------------------------------
# Quick sanity-check summary
# ----------------------------------------------------------------------
frac_sig_stim = (df_singlecell_all['stim_p_value'] < 0.05).mean()
print(f'\nFraction of neurons with significant stimulus information (p<0.05): {frac_sig_stim:.2%}')

if 'choice_p_value' in df_singlecell_all:
    frac_sig_choice = (df_singlecell_all['choice_p_value'] < 0.05).mean()
    print(f'Fraction of neurons with significant choice information (p<0.05): {frac_sig_choice:.2%}')

print(f'\nMean single-cell stimulus MI (PT corrected): {df_singlecell_all["stim_I_pt"].mean():.4f} bits')
if 'choice_I_pt' in df_singlecell_all:
    print(f'Mean single-cell choice MI (PT corrected):    {df_singlecell_all["choice_I_pt"].mean():.4f} bits')

if do_pairwise:
    print('\nMean pairwise breakdown terms (bits):')
    print(df_pairwise_all[['I_full', 'I_lin', 'I_sig_sim', 'I_cor_indep', 'I_cor_dep']].mean())
