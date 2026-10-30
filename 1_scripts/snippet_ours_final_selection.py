#%% ===== FINAL SESSIONS + CELLS (OUR PIPELINE) -- append at the END of 2a_cell_distribution.py =====
# Reads only other steps' out/ (Rule #1). Needs 1b and 2b to have been run. Writes into 2a's own out/tables:
#   final_sessions_ours.csv : session_id, animal_id, protocol, kept, reason
#   final_cells_ours.csv    : session_id, roi_name, labeled, n_cells   (proximity AND 2b QC AND session kept by 1b)
# Same layout as final_*_theirs.csv, so:  pd.read_csv(a).merge(pd.read_csv(b), on=[...], how='outer', indicator=True)
import pandas as _pd

def _find(step, name):
    base = paths.out_from(step)
    for p in (base / name, base / "tables" / name):
        if p.exists():
            return p
    raise FileNotFoundError(f"{name} not found under {base} (run {step} first)")

# ---------- sessions (1b) ----------
fits = _pd.read_csv(_find("1b_psychometric", "psychometric_fits.csv"))
fits["kept"] = fits["included"].astype(bool)
fits["reason"] = fits["exclude_reasons"].fillna("")
sess_animal = celldata.groupby("session_id")["animal_id"].first()
fits["animal_id"] = fits["session_id"].map(sess_animal)
final_sessions = (fits[["session_id", "animal_id", "protocol", "kept", "reason", "dprime_engaged",
                        "fa_rate_engaged", "frac_engaged", "mu", "sigma", "guess_rate"]]
                  .sort_values(["protocol", "session_id"]).reset_index(drop=True))
final_sessions.to_csv(table_dir / "final_sessions_ours.csv", index=False)

print("\n===== FINAL SESSIONS (our filter, 1b) =====")
for proto, g in final_sessions.groupby("protocol"):
    print(f"{proto}: {int(g['kept'].sum())}/{len(g)} kept")
print("\nKept:")
print(final_sessions.loc[final_sessions["kept"], ["protocol", "session_id"]].to_string(index=False))
print("\nDropped:")
print(final_sessions.loc[~final_sessions["kept"], ["protocol", "session_id", "reason"]].to_string(index=False))

# ---------- cells (2a proximity + 2b QC), kept sessions only ----------
qc = _pd.read_csv(_find("2b_activity_statistics", "qc_per_cell.csv"))[["session_id", "cell_id", "qc_pass"]]
cells = celldata.merge(qc, on=["session_id", "cell_id"], how="left")
cells["qc_pass"] = cells["qc_pass"].fillna(False).astype(bool)             # cell without QC row = not included
kept_ids = set(final_sessions.loc[final_sessions["kept"], "session_id"])
cells["final"] = (cells["passes_proximity_filter"].astype(bool) & cells["qc_pass"]
                  & cells["session_id"].isin(kept_ids))
final_cells = (cells[cells["final"]].groupby(["session_id", "roi_name", "labeled"]).size()
               .rename("n_cells").reset_index())
final_cells.to_csv(table_dir / "final_cells_ours.csv", index=False)

print("\n===== FINAL CELLS (our filter: proximity + QC, kept sessions) =====")
print(final_cells.pivot_table(index="roi_name", columns="labeled", values="n_cells", aggfunc="sum", fill_value=0))
print(f"proximity only: {int(cells['passes_proximity_filter'].sum())}; + QC: "
      f"{int((cells['passes_proximity_filter'] & cells['qc_pass']).sum())}; + kept sessions: {int(cells['final'].sum())}")
