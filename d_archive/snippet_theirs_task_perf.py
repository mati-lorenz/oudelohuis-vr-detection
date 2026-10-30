#%% ===== FINAL SESSIONS + CELLS (THEIR PIPELINE) -- append at the END of task_perf.py =====
# Writes two CSVs with the same layout as the snippet for our pipeline, so they can be diffed:
#   final_sessions_theirs.csv : session_id, animal_id, protocol, kept, reason
#   final_cells_theirs.csv    : session_id, roi_name, labeled, n_cells   (cells passing proximity + layer 2/3 [+ QC])
# Uses variables already defined above in task_perf.py: sessions (DP+DN with psy fit), idx_ses, savedir.
from utils.cellselection_lib import filter_nearlabeled_layer23

ZMIN_THR, ZMAX_THR, GUESS_THR = 0.5, -0.5, 0.4      # same numbers passed to get_idx_performing_sessions above
RADIUS_UM, DEPTH_THR, LATERAL_ONLY = 50, 300, True    # filter_nearlabeled_layer23 defaults (signature default is lateral_only=True)

# ---------- sessions ----------
rows = []
sd_all = pd.concat([ses.sessiondata for ses in sessions]).reset_index(drop=True)      # DP + DN, psy params present
for i, r in sd_all.iterrows():
    why = []
    if not r['noise_zmin'] <= ZMIN_THR: why.append("zmin %.2f > %.1f" % (r['noise_zmin'], ZMIN_THR))
    if not r['noise_zmax'] >= ZMAX_THR: why.append("zmax %.2f < %.1f" % (r['noise_zmax'], ZMAX_THR))
    if not r['guess_rate'] <= GUESS_THR: why.append("guess %.2f > %.1f" % (r['guess_rate'], GUESS_THR))
    rows.append(dict(session_id=r['session_id'], animal_id=r['animal_id'], protocol=r['protocol'],
                     kept=bool(idx_ses[i]), reason="; ".join(why), mu=r['mu'], sigma=r['sigma'],
                     guess_rate=r['guess_rate'], noise_zmin=r['noise_zmin'], noise_zmax=r['noise_zmax']))

# DM sessions have no psychometric filter in this script: list them as kept (only filter_sessions' own criteria apply)
dm_sessions, _ = filter_sessions(['DM'], load_behaviordata=False)
for ses in dm_sessions:
    r = ses.sessiondata.iloc[0]
    rows.append(dict(session_id=r['session_id'], animal_id=r['animal_id'], protocol='DM', kept=True,
                     reason="no performance filter for DM"))

final_sessions = pd.DataFrame(rows).sort_values(['protocol', 'session_id']).reset_index(drop=True)
final_sessions.to_csv(os.path.join(savedir, 'final_sessions_theirs.csv'), index=False)

print("\n===== FINAL SESSIONS (their filter) =====")
for proto, g in final_sessions.groupby('protocol'):
    print("%s: %d/%d kept" % (proto, g['kept'].sum(), len(g)))
print("\nKept:")
print(final_sessions.loc[final_sessions['kept'], ['protocol', 'session_id']].to_string(index=False))
print("\nDropped:")
print(final_sessions.loc[~final_sessions['kept'], ['protocol', 'session_id', 'reason']].to_string(index=False))

# ---------- cells of the kept sessions ----------
# Needs celldata: reload the kept sessions WITHOUT behavior/calcium data (only celldata is used here).
kept_ids = set(final_sessions.loc[final_sessions['kept'], 'session_id'])
cell_sessions, _ = filter_sessions(['DM', 'DP', 'DN'], load_behaviordata=False, load_calciumdata=False)
crow = []
for ses in cell_sessions:
    sid = ses.sessiondata['session_id'][0]
    if sid not in kept_ids:
        continue
    cd = ses.celldata.copy()
    cd['pass_prox_l23'] = filter_nearlabeled_layer23(ses, radius=RADIUS_UM, depth_thr=DEPTH_THR,
                                                     lateral_only=LATERAL_ONLY)
    # OPTIONAL QC: uncomment and adapt to the signature of your load_qc_pass
    # cd['pass_qc'] = load_qc_pass(ses)
    # cd['pass_noise'] = cd['noise_level'] < 20          # filter_noiselevel, if you use it
    cd['final'] = cd['pass_prox_l23']                     # & cd['pass_qc'] & cd['pass_noise']
    cd = cd[cd['final']]
    cd['labeled'] = np.where(cd['redcell'].astype(bool), 'lab', 'unl')
    for (area, lab), n in cd.groupby(['roi_name', 'labeled']).size().items():
        crow.append(dict(session_id=sid, roi_name=area, labeled=lab, n_cells=int(n)))
final_cells = pd.DataFrame(crow)
final_cells.to_csv(os.path.join(savedir, 'final_cells_theirs.csv'), index=False)
print("\n===== FINAL CELLS (their filter), totals over kept sessions =====")
print(final_cells.pivot_table(index='roi_name', columns='labeled', values='n_cells', aggfunc='sum', fill_value=0))
