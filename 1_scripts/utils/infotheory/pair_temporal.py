# -*- coding: utf-8 -*-
"""
pair_temporal.py
================
Shared by 3d (pair correlations over position), 3e (information breakdown over position) and 3f (PID over position):
the position-resolved counterparts of 3a / 3b / 3c. The three temporal scripts use EXACTLY the pairs of 3a / 3b / 3c
(3a's out/pair_list.csv, `--max-pairs` = first N per (session, area, label)), the same targets (`stim__<strategy>`,
choice), the same trial rule (pair_trials) and the same response bins, so a temporal curve evaluated in the stimulus
window can be compared with the window-based 3a-3c number (project rule 4).

TENSOR  one value per (trial, position bin) = the trial's OWN mean activity in that narrow bin (the project's "binmean"
        tensor, psth.build_position_binned_trial_tensor); different from 2f's frame pooling, which needed raw frames to
        scan lags. Only the cells that belong to a sampled pair enter the binning step.
CURVES  `position_curves`: per project group and position bin, the mean over analysis units of the unit's mean over its
        pairs (aggregate = session | mouse; units with fewer than `min_pairs` pairs at that bin are dropped; SEM over
        units). Pooled: the mean over all pairs and the SEM over pairs (pairs treated as independent -- pairs of one
        session share cells and animal, so this band is narrower than the session / mouse one).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import project_config as cfg
from . import pair_tables as pt
from . import cell_tables as ct
from .session import discover_sessions
from .neural_encoding import build_session_tables
from .continuous import build_calcium_continuous
from .psth import build_position_binned_trial_tensor
from .scope import select_sessions, data_stamp
from .unit_plots import mouse_palette
import hashlib
from scipy import sparse

from .groups import add_six_group_column, group_key
from .celldata_utils import SIX_GROUP_ORDER
from .stim_binning import CHOICE_TARGET
from .neural_encoding import build_session_tables as _bst
from . import position_info as pi
from .unit_stats import stable_seed


def pair_inputs(paths, scope, protocol: str, max_pairs: int, hint: str = "") -> SimpleNamespace:
    """Selected, included sessions with pairs from 3a's pair list. Exits with a message when an input is missing."""
    fits_path = paths.out_from("1b_psychometric") / "psychometric_fits.csv"
    pair_list_path = paths.out_from("3a_pair_activity_statistics") / "pair_list.csv"
    for f, h in [(fits_path, "1b_psychometric.py"), (pair_list_path, "3a_pair_activity_statistics.py (full dataset)")]:
        if not f.exists():
            raise SystemExit(f"Run {h} first -- expected {f}")
    fits = pd.read_csv(fits_path)
    included = set(fits[fits["included"] & (fits["protocol"] == protocol)]["session_id"])
    all_triples = discover_sessions([protocol])
    selected = {f"{a}_{d}": (p, a, d) for p, a, d in select_sessions(all_triples, scope)}
    session_ids = [s for s in selected if s in included]
    pairs_all = pt.load_pair_list(pair_list_path, session_ids, max_pairs)
    missing = sorted(set(session_ids) - set(pairs_all["session_id"]))
    if missing:
        print(f"WARNING: no pairs in 3a's list for {missing} (rerun 3a on the full dataset); skipped")
    session_ids = [s for s in session_ids if s in set(pairs_all["session_id"])]
    if not session_ids:
        raise SystemExit("No selected session has pairs in 3a's pair_list.csv.")
    by_session = {sid: sub.reset_index(drop=True) for sid, sub in pairs_all.groupby("session_id")}
    stamps = {s: f"{data_stamp(*selected[s])}-"
                 f"{hashlib.sha1(by_session[s].to_csv(index=False).encode()).hexdigest()[:10]}" for s in session_ids}
    print(f"{len(session_ids)} sessions, {len(pairs_all)} pairs (3a's list, pair_rank < {max_pairs})")
    return SimpleNamespace(
        fits_path=fits_path, mu_sigma_lookup=fits.set_index("session_id")[["mu", "sigma"]].to_dict("index"),
        session_ids=session_ids, mouse_colors=mouse_palette(a for _, a, _ in all_triples), pairs_by_session=by_session,
        cells_by_session={sid: sorted(set(s["cell_i"]) | set(s["cell_j"])) for sid, s in by_session.items()},
        stamps=stamps, n_pairs=len(pairs_all))


def session_position_table(ses, mu_sigma_lookup, cells, strategies, s_pre, s_post, binsize):
    """One session -> long table (one row per engaged trial x position bin `s_rel`) with a column per cell in `cells`,
    plus trialNumber, choice, signal/stim columns and one `stim__<strategy>` column per strategy; None if unusable."""
    cells = set(cells)
    tables = build_session_tables(ses, mu_sigma_lookup, {"stim": (0.0, 20.0)}, included_cell_ids=cells)
    if "stim" not in tables or tables["stim"].empty or "signal" not in tables["stim"].columns:
        return None
    tab = tables["stim"]
    meta = pt.add_target_columns(tab[[c for c in tab.columns if c not in cells]], strategies)
    td = ses.trialdata
    trials = td[td["engaged"] == 1] if "engaged" in td.columns else td
    cont = build_calcium_continuous(ses.calciumdata[sorted(cells)], ses.ts_F, ses.behaviordata)
    tensor = build_position_binned_trial_tensor(trials, cont, cell_cols=sorted(cells), s_pre=s_pre, s_post=s_post,
                                                binsize=binsize)
    return tensor.merge(meta, on="trialNumber", how="inner")


def position_curves(df, values, by_extra, aggregate, min_pairs):
    """(group, *by_extra, s_rel): mean and SEM over units of the unit means of each column in `values`; also n_units,
    n_pairs. Pooled = the plain mean over pairs and the SEM over pairs (n_units = n_pairs)."""
    by_extra = list(by_extra)
    if aggregate == "pooled":
        g = df.groupby(["group"] + by_extra + ["s_rel"], observed=True)
        out = g[values].mean()
        for v in values:
            out[f"{v}_sem"] = g[v].sem().fillna(0.0)
        out["n_units"] = g.size()
        out["n_pairs"] = out["n_units"]
        return out.reset_index()
    um = ct.unit_group_summary(df, ["s_rel"] + by_extra, aggregate, values)
    if aggregate != "pooled":
        um = um[um["n_cells"] >= min_pairs]
    g = um.groupby(["group"] + by_extra + ["s_rel"])
    out = g[values].mean()
    for v in values:
        out[f"{v}_sem"] = g[v].sem().fillna(0.0) if aggregate != "pooled" else 0.0
    out["n_units"] = g.size()
    out["n_pairs"] = g["n_cells"].sum()
    return out.reset_index()


def draw_lines(ax, curve, value, color, label=None, ls="-", lw=1.5, band=True, aggregate="pooled"):
    """One mean curve (+ SEM band when not pooled) from `position_curves` output already filtered to one series."""
    c = curve.sort_values("s_rel")
    if c.empty:
        return
    ax.plot(c["s_rel"], c[value], color=color, label=label, ls=ls, lw=lw)
    if band:
        ax.fill_between(c["s_rel"], c[value] - c[f"{value}_sem"], c[value] + c[f"{value}_sem"], color=color, alpha=0.18,
                        lw=0)


# ---------------------------------------------------------------------------------------------------------------------
# Large pair samples + fixed / sliding windows (3e, 3f; 3h has its own copy of the same logic for the time axis)
# ---------------------------------------------------------------------------------------------------------------------
GROUP_KEYS = [group_key(g) for g in SIX_GROUP_ORDER]
GCODE = {k: i for i, k in enumerate(GROUP_KEYS)}
NG = len(GROUP_KEYS)


def sample_group_pairs(ses, cells, max_pairs: int, tag: str, base_seed: int = 0):
    """`max_pairs` pairs per (area, label) of the session's INCLUDED cells (as 3g / 3h; not 3a's small list), with the
    project group of each pair in `group`; None if there is no pair."""
    celldata = ses.celldata[ses.celldata["cell_id"].isin(cells)].reset_index(drop=True)
    if len(celldata) < 2:
        return None
    pairs = pt.sample_session_pairs(celldata, ses.session_id,
                                    np.random.default_rng(stable_seed(ses.session_id, tag, base=base_seed)), max_pairs)
    if pairs.empty:
        return None
    pairs = add_six_group_column(pairs, area_var="roi_name", label_var="labeled_grouped")
    pairs = pairs[pairs["group"].isin(GCODE)].reset_index(drop=True)
    return None if pairs.empty else pairs


def valid_trials(lab, has_frames, min_trials=cfg.MIN_TRIALS_ANALYSIS, min_per_bin=cfg.MIN_TRIALS_PER_TARGET_BIN):
    """Boolean mask of the trials usable for a target in a bin: the cell-independent part of `pair_trials` (finite label,
    target bins with < min_per_bin trials dropped, >= min_trials trials, 2 labels); None if the bin is unusable."""
    ok = has_frames & np.isfinite(lab)
    if ok.any():
        vals, counts = np.unique(lab[ok], return_counts=True)
        small = vals[counts < min_per_bin]
        if len(small):
            ok &= ~np.isin(lab, small)
    if ok.sum() < min_trials or len(np.unique(lab[ok])) < 2:
        return None
    return ok


class SessionWindows:
    """Position windows of one session: trial x cell mean activity in every fixed bin / sliding window, relative to the
    stimulus onset, from the frames inside each ENGAGED trial's own [tStart, tEnd].
    `windows()` yields (binning, s_rel, TM, cnt): TM = (n_trials x cells) trial means, cnt = frames per trial.
    `lab[target]` = per-trial labels (rows of TM); `col_of` = cell -> TM column; `bad` = cells with non-finite activity."""

    def __init__(self, ses, mu_sigma_lookup, cells, targets, pre, post, binsize, window, step, binnings):
        self.cells = list(cells)
        self.col_of = {c: k for k, c in enumerate(self.cells)}
        self.ok = False
        self.pre, self.post, self.binsize, self.window, self.step, self.binnings = pre, post, binsize, window, step, binnings
        tabs = _bst(ses, mu_sigma_lookup, {"stim": (0.0, 20.0)}, included_cell_ids=set(self.cells))
        if "stim" not in tabs or tabs["stim"].empty or "signal" not in tabs["stim"].columns:
            return
        tt = tabs["stim"]
        cs = set(self.cells)
        meta = pt.add_target_columns(tt[[c for c in tt.columns if c not in cs]], [t for t in targets if t != CHOICE_TARGET])
        td = ses.trialdata
        trials = (td[td["engaged"] == 1] if "engaged" in td.columns else td).reset_index(drop=True)
        cont = build_calcium_continuous(ses.calciumdata[self.cells], ses.ts_F, ses.behaviordata)
        tmeta = trials[["trialNumber"]].merge(meta, on="trialNumber", how="left")
        self.lab = {t: tmeta[pt.target_column(t)].to_numpy(float) for t in targets}
        self.A = cont[self.cells].to_numpy(float)
        self.bad = ~np.isfinite(self.A).all(0)
        self.A = np.nan_to_num(self.A)
        self.ts = cont["ts"].to_numpy(float)
        self.zpos = cont["zpos"].to_numpy(float)
        self.stim = trials["stimStart"].to_numpy(float)
        self.t0, self.t1 = trials["tStart"].to_numpy(float), trials["tEnd"].to_numpy(float)
        self.nt = len(trials)
        self.ok = True

    def windows(self):
        for binning in self.binnings:
            base = self.binsize if binning == "fixed" else self.step
            m = 1 if binning == "fixed" else int(round(self.window / self.step))
            fi, tr, bb, centers = pi.frame_bin_index(self.stim, self.zpos, self.pre, self.post, base)
            if len(fi):
                own = (self.ts[fi] >= self.t0[tr]) & (self.ts[fi] <= self.t1[tr])
                fi, tr, bb = fi[own], tr[own], bb[own]
            if len(fi) == 0:
                continue
            for k in range(len(centers) - m + 1):
                s_rel = (centers[k] + centers[k + m - 1]) / 2.0
                sel = (bb >= k) & (bb < k + m)
                f_w, t_w = fi[sel], tr[sel]
                if len(f_w) == 0:
                    continue
                cnt = np.bincount(t_w, minlength=self.nt).astype(float)
                S = sparse.csr_matrix((np.ones(len(t_w)), (t_w, np.arange(len(t_w)))), shape=(self.nt, len(t_w)))
                yield binning, s_rel, (S @ self.A[f_w]) / np.maximum(cnt, 1)[:, None], cnt


def group_moments(values: dict, gcode):
    """values: name -> array over pairs. Returns name -> (n finite, sum, sum of squares) per project group."""
    out = {}
    for name, v in values.items():
        v = np.asarray(v, float)
        ok = np.isfinite(v) & (gcode >= 0)
        g, x = gcode[ok], v[ok]
        out[name] = (np.bincount(g, minlength=NG).astype(float), np.bincount(g, x, NG), np.bincount(g, x * x, NG))
    return out


def moment_records(base_rec: dict, mom: dict):
    """One record per project group from `group_moments` output (groups without a pair value are skipped)."""
    first = next(iter(mom.values()))[0]
    recs = []
    for g in range(NG):
        if first[g] == 0:
            continue
        r = dict(base_rec, group=GROUP_KEYS[g])
        for name, (n, s_, ss) in mom.items():
            r[f"{name}_n"], r[f"{name}_sum"], r[f"{name}_ss"] = n[g], s_[g], ss[g]
        recs.append(r)
    return recs


def moment_curves(m, values, keys, aggregate, min_pairs):
    """Curves from the per-(session, group, ...) moments written by the scripts: mean and SEM of each value per `keys`
    (must contain group and s_rel). pooled: over pairs; session / mouse: over units (units with < min_pairs pairs dropped)."""
    unit_col = {"pooled": None, "session": "session_id", "mouse": "animal_id"}[aggregate]
    cols = [f"{x}_{k}" for x in values for k in ("n", "sum", "ss")]
    first = values[0]
    if unit_col is None:
        g = m.groupby(keys)[cols].sum()
        out = pd.DataFrame(index=g.index)
        for x in values:
            n, s_, ss = g[f"{x}_n"], g[f"{x}_sum"], g[f"{x}_ss"]
            mean = s_ / n.where(n > 0)
            var = (ss / n.where(n > 0) - mean ** 2).clip(lower=0) * n / (n - 1).where(n > 1)
            out[x] = mean
            out[f"{x}_sem"] = np.sqrt(var / n.where(n > 0)).fillna(0.0)
        out["n_pairs"] = g[f"{first}_n"]
        out["n_units"] = g[f"{first}_n"]
        return out.reset_index()
    u = m.groupby(keys + [unit_col])[cols].sum().reset_index()
    res = None
    for x in values:
        v = u[f"{x}_sum"] / u[f"{x}_n"].where(u[f"{x}_n"] >= min_pairs)
        d = u[keys].assign(v=v, n=u[f"{x}_n"]).dropna(subset=["v"])
        g = d.groupby(keys)
        part = pd.DataFrame({x: g["v"].mean(), f"{x}_sem": g["v"].sem().fillna(0.0)})
        if x == first:
            part["n_units"] = g.size()
            part["n_pairs"] = g["n"].sum()
        res = part if res is None else res.join(part, how="outer")
    return res.reset_index()


def session_inputs(paths, scope, protocol: str) -> SimpleNamespace:
    """Selected 1b-included sessions with at least one cell passing 2a + 2b (as 3g / 3h), for the scripts that sample their
    own pairs: fits_path, mu_sigma_lookup, session_ids, cells_by_session, stamps."""
    from .neural_encoding import load_cell_inclusion
    fits_path = paths.out_from("1b_psychometric") / "psychometric_fits.csv"
    cell_dist_out = paths.out_from("2a_cell_distribution")
    activity_out = paths.out_from("2b_activity_statistics")
    for f, h in [(fits_path, "1b_psychometric.py"), (cell_dist_out / "celldata_combined.csv", "2a_cell_distribution.py"),
                 (activity_out / "qc_per_cell.csv", "2b_activity_statistics.py")]:
        if not f.exists():
            raise SystemExit(f"Run {h} first -- expected {f}")
    fits = pd.read_csv(fits_path)
    included = set(fits[fits["included"] & (fits["protocol"] == protocol)]["session_id"])
    all_triples = discover_sessions([protocol])
    selected = {f"{a}_{d}": (p, a, d) for p, a, d in select_sessions(all_triples, scope)}
    session_ids = [s for s in selected if s in included]
    if not session_ids:
        raise SystemExit("None of the selected sessions passed 1b_psychometric's inclusion criteria.")
    inc = load_cell_inclusion(cell_dist_out, activity_out)
    cells_by_session = {sid: set(sub.loc[sub["include"], "cell_id"]) for sid, sub in inc.groupby("session_id")
                        if sid in session_ids}
    session_ids = [s for s in session_ids if cells_by_session.get(s)]
    if not session_ids:
        raise SystemExit("No selected session has included cells.")
    print(f"{len(session_ids)} sessions (cells: 2a + 2b inclusion; pairs sampled per session)")
    return SimpleNamespace(fits_path=fits_path, mu_sigma_lookup=fits.set_index("session_id")[["mu", "sigma"]].to_dict("index"),
                           session_ids=session_ids, cells_by_session=cells_by_session,
                           stamps={s: data_stamp(*selected[s]) for s in session_ids})


# ---------------------------------------------------------------------------------------------------------------------
# Vectorised correlations of all pairs of a window (3d; 3h has its own copy of the same code for frame-level samples)
# ---------------------------------------------------------------------------------------------------------------------
def _zscore(X):
    sd = X.std(0)
    sd = np.where(sd < 1e-12, np.nan, sd)
    return (X - X.mean(0)) / sd


def _pair_corr(Z, I, J):
    return (Z[:, I] * Z[:, J]).mean(0)


def corr_metrics(X, y, I, J):
    """signal / noise / raw Pearson correlation of every pair (columns I[k], J[k] of X = trial x cell matrix) given integer
    labels y (0..L-1), as 3a: signal = correlation of the label means (needs >= 3 labels, else NaN), noise = correlation of
    the residuals after removing each cell's label mean, raw = plain correlation. Returns name -> array over pairs."""
    n_lab = int(y.max()) + 1
    M = sparse.csr_matrix((np.ones(len(y)), (y, np.arange(len(y)))), shape=(n_lab, len(y)))
    cnt = np.asarray(M.sum(1)).ravel()
    G = (M @ X) / np.maximum(cnt, 1)[:, None]
    out = {"raw_corr": _pair_corr(_zscore(X), I, J), "noise_corr": _pair_corr(_zscore(X - G[y]), I, J)}
    out["signal_corr"] = _pair_corr(_zscore(G), I, J) if n_lab >= 3 else np.full(len(I), np.nan)
    return out
