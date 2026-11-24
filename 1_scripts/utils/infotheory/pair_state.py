# -*- coding: utf-8 -*-
"""
pair_state.py
=============
Shared by the pair x state scripts (3n: pair INFORMATION by state tercile; 3m has its own copy of the same definitions): the pairs, the
pair-specific STATES and the leave-pair-out GLOBAL signal.

PAIRS      build_pair_list: within-group pairs = 3a's pair_list.csv (pair_rank < max_pairs: the SAME pairs as 3a / 3b / 3c), plus CROSS pairs
           (each of `cross_a_groups` x `source_group`, sampled per session with a seed from the session id, random pair_rank).
STATES     z-scored within the session (state window), then per pair:
               PM_lab / global   mean of the source-group cells / of all included cells outside it, WITHOUT the pair's own cells
               pupil, speed      cur_pupil_area / cur_runspeed of the state window
GLOBAL     G_loo = mean (z-scored response) of all included cells of that trial and window EXCEPT the pair's two cells; remove_global()
           regresses each cell on it (one intercept + slope per cell) and returns the residuals.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from . import pair_tables as pt
from .unit_stats import stable_seed


def zcols(M):
    M = np.asarray(M, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (M - np.nanmean(M, 0)) / np.nanstd(M, 0)


def zvec(v):
    v = np.asarray(v, float)
    sd = np.nanstd(v)
    return (v - np.nanmean(v)) / sd if sd > 0 else np.full_like(v, np.nan)


def resid_by_label(a, inv, k):
    cnt = np.bincount(inv, minlength=k)
    return a - (np.bincount(inv, weights=a, minlength=k) / np.maximum(cnt, 1))[inv]


def build_pair_list(session_ids, pair_list_path, max_pairs, incl_by_session, cell_group, source_group, cross_a_groups, base_seed=0):
    """DataFrame (session_id, animal_id, group, cell_i, cell_j, pair_rank, kind) of within + cross pairs of the sessions that have any."""
    within = pt.load_pair_list(pair_list_path, session_ids, max_pairs)
    if len(within):
        within = pt.finalize_pair_table(within)
        within["kind"] = "within"
        within = within[["session_id", "animal_id", "group", "cell_i", "cell_j", "pair_rank", "kind"]]
    rows = []
    for sid in session_ids:
        cells = incl_by_session.get(sid, [])
        tgt = [c for c in cells if cell_group.get(c) == source_group]
        for ga in cross_a_groups:
            src = [c for c in cells if cell_group.get(c) == ga]
            if not src or not tgt:
                continue
            ii, jj = np.meshgrid(np.arange(len(src)), np.arange(len(tgt)), indexing="ij")
            ii, jj = ii.ravel(), jj.ravel()
            rng = np.random.default_rng(stable_seed(sid, "cross", ga, source_group, base=base_seed))
            for rank, k in enumerate(rng.permutation(len(ii))[:max_pairs]):
                rows.append(dict(session_id=sid, animal_id=sid.split("_")[0], group=f"{ga}~{source_group}", cell_i=src[ii[k]],
                                 cell_j=tgt[jj[k]], pair_rank=rank, kind="cross"))
    cross = pd.DataFrame(rows)
    parts = [p for p in (within if len(within) else None, cross if len(cross) else None) if p is not None]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def pair_hash(pairs: pd.DataFrame) -> str:
    return hashlib.sha1(pairs.to_csv(index=False).encode()).hexdigest()[:10]


def session_state_context(st, cells, cell_group, source_group):
    """st = the session's table of the STATE window (build_session_tables; cell columns + cur_pupil_area / cur_runspeed)."""
    st = st.sort_values("trialNumber").reset_index(drop=True)
    grp = np.array([cell_group.get(c) for c in cells], dtype=object)
    src = grp == source_group
    Zs = zcols(st[cells].to_numpy(float))
    Zs0, Cs = np.nan_to_num(Zs), np.isfinite(Zs).astype(float)
    return dict(src=src, oth=~src, Zs0=Zs0, Cs=Cs,
                sums={"src": (Zs0[:, src].sum(1), Cs[:, src].sum(1)), "oth": (Zs0[:, ~src].sum(1), Cs[:, ~src].sum(1))},
                face={"pupil": zvec(st["cur_pupil_area"]) if "cur_pupil_area" in st else None,
                      "speed": zvec(st["cur_runspeed"]) if "cur_runspeed" in st else None},
                st_tn=st["trialNumber"].to_numpy())


def window_context(ctx, table, cells):
    """Per response window: trial alignment to the state table and the z-scored responses of all included cells."""
    tnum = table["trialNumber"].to_numpy()
    srow = pd.Index(ctx["st_tn"]).get_indexer(tnum)
    Z = zcols(table[cells].to_numpy(float))
    Z0, C = np.nan_to_num(Z), np.isfinite(Z).astype(float)
    return dict(tnum=tnum, tindex=pd.Index(tnum), srow=srow, has=srow >= 0, Z0=Z0, C=C, Sall=Z0.sum(1), Nall=C.sum(1))


def pair_global(wc, ci, cj, ix):
    """Leave-pair-out global signal at the trial rows `ix` of the window table."""
    G = (wc["Sall"] - wc["Z0"][:, ci] - wc["Z0"][:, cj]) / np.maximum(wc["Nall"] - wc["C"][:, ci] - wc["C"][:, cj], 1)
    return G[ix]


def pair_states(ctx, wc, ci, cj, ix, states, source_group, min_source):
    """{state: array over the pair's trials (rows ix)} -- NaN where undefined."""
    def at(v):
        out = np.full(len(wc["tnum"]), np.nan)
        out[wc["has"]] = v[wc["srow"][wc["has"]]]
        return out
    sv = {}
    for name, key, member in ((source_group, "src", ctx["src"]), ("global", "oth", ctx["oth"])):
        if name not in states:
            continue
        S_, N_ = ctx["sums"][key]
        own_s = sum(ctx["Zs0"][:, c] * member[c] for c in (ci, cj))
        own_n = sum(ctx["Cs"][:, c] * member[c] for c in (ci, cj))
        n_ = N_ - own_n
        v = np.where(n_ >= min_source, (S_ - own_s) / np.maximum(n_, 1), np.nan)
        sv[name] = at(zvec(v))[ix]
    for name in ("pupil", "speed"):
        if name in states:
            sv[name] = at(ctx["face"][name])[ix] if ctx["face"][name] is not None else np.full(len(ix), np.nan)
    return sv


def remove_global(a, G):
    """Residual of `a` after regressing it on [1, G] (trials where G is finite; others unchanged)."""
    fin = np.isfinite(G)
    out = np.asarray(a, float).copy()
    if fin.sum() >= 3:
        B = np.column_stack([np.ones(fin.sum()), G[fin]])
        out[fin] = out[fin] - B @ np.linalg.lstsq(B, out[fin], rcond=None)[0]
    return out
