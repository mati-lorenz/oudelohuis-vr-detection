# -*- coding: utf-8 -*-
"""
This script analyzes the behavior of mice performing a virtual reality
navigation task while headfixed in a visual tunnel with landmarks. 
Matthijs Oude Lohuis, 2023, Champalimaud Center
"""

#%% Import packages
import math
import pandas as pd
import os, sys
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
os.chdir('/u/g/glorenz/Documents/Research/Code/oudelohuis-vr-detection/d_archive')
sys.path.insert(0, '/u/g/glorenz/Documents/Research/Code/oudelohuis-vr-detection/d_archive')
from loaddata.get_data_folder import get_local_drive
from loaddata.session_info import filter_sessions,load_sessions,report_sessions
from utils.psth import compute_tensor_space,compute_respmat_space
from utils.plot_lib import * #get all the plotting functions
from utils.behaviorlib import * # get support functions for beh analysis 

savedir = '/u/g/glorenz/Documents/Research/Code/oudelohuis-vr-detection/'
os.makedirs(savedir, exist_ok=True)

def save_fig(fig, path, **kwargs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, **kwargs)
    
#%% Load behavior of all protocols:
protocol                = ['DM','DP','DN']
sessions,nsessions      = filter_sessions(protocol)
sessiondata             = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
nanimals                = len(np.unique(sessiondata['animal_id']))

#%% Report on number of trials per session:
print('%3.1f +- %3.1f trials per session' % (np.mean(sessiondata['ntrials']),np.std(sessiondata['ntrials'])))
print(f"{len(sessiondata[sessiondata['protocol']=='DM'])} DM, \n"
      f"{len(sessiondata[sessiondata['protocol']=='DP'])} DP, \n"
      f"{len(sessiondata[sessiondata['protocol']=='DN'])} DN sessions\n")

#%% ###############################################################
### Show the overall dprime for each animal across sessions:
dp_ses      = np.zeros([nsessions])
dp_ses_eng  = np.zeros([nsessions])
cr_ses      = np.zeros([nsessions])
cr_ses_eng  = np.zeros([nsessions])
for i,ses in enumerate(sessions):
    idx = np.isin(ses.trialdata['signal'],[0,100])
    df = ses.trialdata[idx]
    dp_ses[i],cr_ses[i]     = compute_dprime(df['signal']>0,df['lickResponse'])

    #Engaged only:
    idx = np.isin(ses.trialdata['signal'],[0,100])
    idx = np.logical_and(idx,ses.trialdata['engaged']==1)
    df = ses.trialdata[idx]
    dp_ses_eng[i],cr_ses_eng[i]     = compute_dprime(df['signal']>0,df['lickResponse'])

#%% Show figure of dprime for all animals:
sessiondata = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
sessiondata['dprime']       = dp_ses #dp_target_eng
sessiondata['dprime_eng']   = dp_ses_eng #dp_target_eng

fig, ax = plt.subplots(1,1,figsize=(4,3))

sns.stripplot(data = sessiondata,x='animal_id',y='dprime_eng',hue='animal_id',palette='Dark2',size=6,ax=ax)
# sns.stripplot(data = sessiondata,x='animal_id',y='dprime_eng',hue='animal_id',palette='Dark2',size=6,ax=ax,legend=False)
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False,fontsize=8)
plt.subplots_adjust(right=0.7)
ax.axhline(y = 0, color = 'k', linestyle = ':')

# ax.set_title('Total Session', fontsize=11)
ax.set_xticks(range(nanimals))
ax.set_xticklabels(range(1,nanimals+1))
ax.errorbar(x=nanimals,y=sessiondata['dprime_eng'].mean(),yerr=sessiondata['dprime_eng'].std(),fmt='o',color='k',capsize=3)
ax.set_xticks(range(nanimals+1))
ax.set_xticklabels([str(x) for x in range(1,nanimals+1)] + [u'\u03BC'])
ax.set_ylabel('d-prime')
ax.set_xlabel('animal ID')
plt.tight_layout()
print('Mean Dprime: %.2f +/- %.2f' % (sessiondata['dprime_eng'].mean(),sessiondata['dprime_eng'].std()))
sns.despine(fig=fig, top=True, right=True, offset=3)

my_savefig(fig,savedir,'Dprime_%danimals' % nanimals)

#%% ########################## Load psychometric data #######################
protocol            = ['DP','DN']
sessions,nsessions  = filter_sessions(protocol,load_behaviordata=False)
sessions            = stim_remapping(sessions)
sessiondata         = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)

#%% ########################## Show psychometric curve for an example session #######################
# fig = plot_psycurve(sessions,filter_engaged=True)
example_session = 'LPE11998_2024_04_16'
# example_session = 'LPE11998_2024_04_18'
sesidx  = np.where(np.array([ses.sessiondata['session_id'][0] for ses in sessions]) == example_session)[0][0]
fig     = plot_psycurve([sessions[sesidx]],filter_engaged=True)
my_savefig(fig,savedir,'Psycurve_%s' % sessions[sesidx].sessiondata['session_id'][0])

#%% Get psychometric curves for all sessions:
sessions    = noise_to_psy(sessions,filter_engaged=True,bootstrap=True)

#%% Filter sessions with sufficient performance: 
idx_ses     = get_idx_performing_sessions(sessions,zmin_thr=0.5,zmax_thr=-0.5,guess_thr=0.4)

#%% ########################## Show psychometric curve for an example session #######################
filterses   = [sessions[ises] for ises in np.where(idx_ses)[0]]
fig         = plot_all_psycurve(filterses,filter_engaged=True)
# fig     = plot_all_psycurve(sessions,filter_engaged=True)
sns.despine(fig=fig, top=True, right=True, offset=3)
my_savefig(fig,savedir,'Psycurve_%dsessions' % len(filterses))

#%% For the different stimuli:
print('Number of sessions per stim:')
sessiondata     = pd.concat([ses.sessiondata for ses in filterses]).reset_index(drop=True)
print(sessiondata['stim'].value_counts())

#%% Show the threshold for each stim:

sessiondata     = pd.concat([ses.sessiondata for ses in filterses]).reset_index(drop=True)
# sessiondata     = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)
stims           = np.sort(sessiondata['stim'].unique())

fig,ax = plt.subplots(1,1,figsize=(3,3))
sns.stripplot(data=sessiondata,x='stim',y='mu',hue='stim',palette='tab10',ax=ax,
              legend=False,order=stims)
ax.set_ylabel('Threshold (% signal)')
ax.set_xlabel('Stim')

for istim,stim in enumerate(stims):
    stimdata = sessiondata[sessiondata['stim']==stim]
    mean = np.nanmean(stimdata['mu'])
    error = np.nanstd(stimdata['mu']) / np.sqrt(len(stimdata))
    error = np.nanstd(stimdata['mu'])
    ax.errorbar(x=istim+0.3,y=mean,yerr=error,fmt='o',color='k',capsize=0)
ax.set_yticks([0,10,20,30])
ax.set_ylim([0,25])
# ax.set_ylim([0,35])
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, offset=3)
my_savefig(fig,savedir,'Threshold_per_stim_%dsessions' % len(filterses))


#%%:
sessiondata     = pd.concat([ses.sessiondata for ses in filterses]).reset_index(drop=True)
df = pd.DataFrame(sessiondata[['mu','sigma','lapse_rate','guess_rate','psy_r2']])
for i,ses in enumerate(filterses):
    df.loc[i,'session_id'] = ses.sessiondata['session_id'][0]
    df.loc[i,'animal_id'] = ses.sessiondata['animal_id'][0]
# df = df[idx_ses].reset_index(drop=True)

#%% 
fig,axes = plt.subplots(1,5,figsize=(7,2.5),sharex=True)
for i,param in enumerate(df.columns[:5]):
    ax = axes[i]
    if i==4:
        sns.stripplot(data=df,y=param,ax=ax,hue='animal_id',palette='tab10',legend=True)
    else:
        sns.stripplot(data=df,y=param,ax=ax,hue='animal_id',palette='tab10',legend=False)

    # sns.stripplot(data=df,x=param,y='animal_id',ax=axes[i],palette='tab10',order=np.sort(sessiondata['animal_id'].unique()),hue='animal_id',legend=False)
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_title(param)
    # ax.set_xticks([])
    ax.set_ylim([0,np.nanmax(df[param])*1.2])
    if i==4:
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,frameon=False,fontsize=8)
plt.tight_layout()
sns.despine(fig=fig, top=True, right=True, bottom=True, offset=3)
my_savefig(fig,savedir,'Params_per_session_%danimals' % nanimals)

#%% Correlations between psy fit parameters:
fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(df[['mu','sigma','lapse_rate','guess_rate','psy_r2']].corr(), ax=ax, 
            cmap='coolwarm', 
            annot=True, 
            linewidths=0.5, 
            square=True,
            cbar_kws={'shrink': 0.5})
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.title('Correlation between psy fit parameters')
my_savefig(fig,savedir,'Corrmat_params_%danimals' % nanimals)
#%% ===== FINAL SESSIONS + CELLS (THEIR PIPELINE), v2 -- append at the END of task_perf.py =====
# Writes (same layout as our snippet, so they can be diffed):
#   final_sessions_theirs.csv : session_id, animal_id, protocol, kept, kept_frac, reason, ...
#   final_cells_theirs.csv    : session_id, roi_name, labeled, n_cells
#
# What the previous run showed:
#  * DP sessions have no noise trials -> noise_zmin/zmax are NaN -> the performance filter is meaningless for DP.
#    filter_sessions applies it to DN ONLY, so DP (like DM) is simply kept.
#  * The real DN filter is the one INSIDE filter_sessions (defaults zmin_thr=0, zmax_thr=0, guess_thr=0.4; strict, no 0.5
#    tolerance). The 0.5/-0.5 call in task_perf.py runs on the already-filtered list, so it removes nothing.
#  * That filter uses an UNSEEDED bootstrap: the same 36 DN sessions gave 23 kept in one call and 25 in another.
#    Here we repeat the fit with several seeds and report, per DN session, in what fraction of fits it passes.
from utils.cellselection_lib import filter_nearlabeled_layer23
from utils.behaviorlib import noise_to_psy, get_idx_performing_sessions

N_SEEDS = 500                                   # bootstrap repeats for DN (each = 100 curve fits per session)
ZMIN_THR, ZMAX_THR, GUESS_THR = 0, 0, 0.4      # the defaults filter_sessions really uses
RADIUS_UM, DEPTH_THR, LATERAL_ONLY = 50, 300, True

# ---------- sessions: load EVERYTHING that passes the non-performance criteria (filter_performing=False) ----------
all_ses, _ = filter_sessions(['DM', 'DP', 'DN'], load_behaviordata=False, filter_performing=False)
dn_idx = [i for i, s in enumerate(all_ses) if s.sessiondata['protocol'][0] == 'DN']
dn_ses = [all_ses[i] for i in dn_idx]

import io, contextlib, warnings
passes = np.zeros((N_SEEDS, len(dn_ses)), dtype=bool)
last = None
for seed in range(N_SEEDS):
    np.random.seed(seed)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        noise_to_psy(dn_ses, filter_engaged=True, bootstrap=True)
        passes[seed] = get_idx_performing_sessions(dn_ses, zmin_thr=ZMIN_THR, zmax_thr=ZMAX_THR, guess_thr=GUESS_THR)
    if seed == 0:
        last = pd.concat([s.sessiondata for s in dn_ses]).reset_index(drop=True)[
            ['noise_zmin', 'noise_zmax', 'guess_rate', 'mu', 'sigma']]
frac = passes.mean(axis=0)
print("DN sessions kept per seed:", passes.sum(axis=1).tolist(), "of", len(dn_ses))

rows = []
for ses in all_ses:
    r = ses.sessiondata.iloc[0]
    rows.append(dict(session_id=r['session_id'], animal_id=r['animal_id'], protocol=r['protocol'],
                     kept=True, kept_frac=np.nan, reason=''))
final_sessions = pd.DataFrame(rows)
for k, i in enumerate(dn_idx):
    final_sessions.loc[i, 'kept_frac'] = frac[k]
    final_sessions.loc[i, 'kept'] = bool(frac[k] >= 0.5)
    why = []
    if not last.loc[k, 'noise_zmin'] <= ZMIN_THR: why.append("zmin %.2f > %g" % (last.loc[k, 'noise_zmin'], ZMIN_THR))
    if not last.loc[k, 'noise_zmax'] >= ZMAX_THR: why.append("zmax %.2f < %g" % (last.loc[k, 'noise_zmax'], ZMAX_THR))
    if not last.loc[k, 'guess_rate'] <= GUESS_THR: why.append("guess %.2f > %g" % (last.loc[k, 'guess_rate'], GUESS_THR))
    final_sessions.loc[i, 'reason'] = "; ".join(why) + ("" if frac[k] in (0, 1) else "  [UNSTABLE across bootstrap seeds]")
    for c in last.columns:
        final_sessions.loc[i, c + '_seed0'] = last.loc[k, c]
final_sessions = final_sessions.sort_values(['protocol', 'session_id']).reset_index(drop=True)
final_sessions.to_csv(os.path.join(savedir, 'final_sessions_theirs.csv'), index=False)

print("\n===== FINAL SESSIONS (their filter) =====")
for proto, g in final_sessions.groupby('protocol'):
    print("%s: %d/%d kept" % (proto, g['kept'].sum(), len(g)))
unst = final_sessions[(final_sessions['kept_frac'] > 0) & (final_sessions['kept_frac'] < 1)]
print("\nDN sessions whose fate depends on the bootstrap seed:")
print(unst[['session_id', 'kept_frac']].to_string(index=False) if len(unst) else "  none")
print("\nDropped DN:")
print(final_sessions.loc[~final_sessions['kept'], ['protocol', 'session_id', 'kept_frac', 'reason']].to_string(index=False))
print("\nKept:")
print(final_sessions.loc[final_sessions['kept'], ['protocol', 'session_id']].to_string(index=False))

# ---------- cells of the kept sessions ----------
# celldata is loaded by load_data whenever celldata.csv exists; a session without it has no .celldata attribute
# (that was the AttributeError) -> it is listed and skipped.
kept_ids = final_sessions.loc[final_sessions['kept'], 'session_id'].tolist()
cell_sessions, _ = filter_sessions(['DM', 'DP', 'DN'], load_behaviordata=False, filter_performing=False,
                                   only_session_id=kept_ids)
crow, no_cells = [], []
for ses in cell_sessions:
    sid = ses.sessiondata['session_id'][0]
    if not hasattr(ses, 'celldata'):
        no_cells.append(sid)
        continue
    cd = ses.celldata.copy()
    cd['final'] = filter_nearlabeled_layer23(ses, radius=RADIUS_UM, depth_thr=DEPTH_THR, lateral_only=LATERAL_ONLY)
    # OPTIONAL QC -- adapt to your load_qc_pass signature, then uncomment:
    # cd['final'] &= load_qc_pass(ses)
    # cd['final'] &= (cd['noise_level'] < 20)            # filter_noiselevel
    cd = cd[cd['final']].copy()
    cd['labeled'] = np.where(cd['redcell'].astype(bool), 'lab', 'unl')
    for (area, lab), n in cd.groupby(['roi_name', 'labeled']).size().items():
        crow.append(dict(session_id=sid, roi_name=area, labeled=lab, n_cells=int(n)))
if no_cells:
    print("\nSessions WITHOUT celldata (skipped):", no_cells)
final_cells = pd.DataFrame(crow)
final_cells.to_csv(os.path.join(savedir, 'final_cells_theirs.csv'), index=False)
print("\n===== FINAL CELLS (their filter: proximity + layer 2/3 only unless QC lines are enabled) =====")
print(final_cells.pivot_table(index='roi_name', columns='labeled', values='n_cells', aggfunc='sum', fill_value=0))
