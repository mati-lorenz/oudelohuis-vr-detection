# -*- coding: utf-8 -*-
"""
Sanity checks for neural recordings: flags silent neurons, likely artifacts,
and excessively noisy cells, and gives detailed per-session / per-area /
per-label summaries and plots -- including WHICH criterion flagged each
cell -- so bad sessions/planes/populations can be spotted before running
downstream analyses.

Reuses ses.celldata['noise_level'] where already available (set upstream in
the preprocessing pipeline) rather than recomputing an SNR proxy from
scratch, and extends it with rate / variance / skew based checks on
ses.calciumdata and trial-to-trial Fano factor on ses.respmat, following
the same logic as the legacy ExcludeSilentNeurons() in fct_data.py but
adapted to this project's Session objects (celldata / calciumdata / respmat)
instead of the old .mat loader.

Matthijs Oude Lohuis, Champalimaud Research
"""

import functools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import skew

# ------------------------------------------------------------------------- #
# Default thresholds - tune these per dataset / imaging system
# ------------------------------------------------------------------------- #
RATE_THR        = 0.01   # min fraction of frames with non-trivial signal (silent neuron cutoff)
NOISE_THR       = 100     # max ses.celldata['noise_level'] (matches threshold already used in corr_lib.py/regress_lib.py)
FANO_THR        = 1e3    # max Fano factor across trials (flags unstable / artifact-driven neurons)
SKEW_MIN        = -1.0   # calcium traces should not be strongly left-skewed (saturation/clipping artifact)
FLAT_STD_THR    = 1e-6   # neurons with ~zero variance are dead/silent channels
NAN_FRAC_THR    = 0.01   # max fraction of NaN samples tolerated in the trace

# names of every individual failure reason, in the order they're checked --
# used consistently across flagging, summarizing, and plotting so nothing
# drifts out of sync
REASON_COLS = ['flat', 'lowrate', 'nan', 'lowskew', 'highfano', 'noisy']
REASON_LABELS = {
    'flat':     'flat trace (std too low)',
    'lowrate':  'low activity rate',
    'nan':      'too many NaNs',
    'lowskew':  'low/negative skew (saturation)',
    'highfano': 'unstable across trials (high Fano)',
    'noisy':    'high noise_level',
}
REASON_COLORS = {
    'flat': 'tab:gray', 'lowrate': 'tab:blue', 'nan': 'tab:red',
    'lowskew': 'tab:orange', 'highfano': 'tab:purple', 'noisy': 'tab:brown',
}


def compute_qc_metrics(ses):
    """
    Compute per-neuron QC metrics for one session and store them in
    ses.celldata. Requires ses.calciumdata (time x neurons) and, if
    available, ses.respmat (neurons x trials) for the Fano factor.

    Adds columns to ses.celldata:
        qc_rate      : fraction of frames with above-baseline signal
        qc_std       : std of the raw trace (flags flat/dead channels)
        qc_skew      : skewness of the raw trace
        qc_fano      : Fano factor of trial responses (var/mean), if respmat available
        qc_nan_frac  : fraction of NaN samples in the trace
    """
    data = ses.calciumdata
    if isinstance(data, pd.DataFrame):
        data = data.to_numpy()

    N = data.shape[1]
    nan_frac = np.mean(np.isnan(data), axis=0)

    # baseline-relative "activity" per neuron, ignoring nans:
    with np.errstate(invalid='ignore'):
        baseline = np.nanpercentile(data, 20, axis=0)
        thresholded = data > (baseline + 2 * np.nanstd(data, axis=0))
        qc_rate = np.nanmean(thresholded, axis=0)

    qc_std  = np.nanstd(data, axis=0)
    qc_skew = skew(data, axis=0, nan_policy='omit')

    ses.celldata['qc_rate']     = qc_rate
    ses.celldata['qc_std']      = qc_std
    ses.celldata['qc_skew']     = np.array(qc_skew)
    ses.celldata['qc_nan_frac'] = nan_frac

    if hasattr(ses, 'respmat') and ses.respmat is not None:
        resp = ses.respmat
        # Fano factor per neuron across trials (only defined for non-negative signals like events/spikes;
        # still informative as a stability metric for dF/F, just interpret loosely):
        with np.errstate(invalid='ignore', divide='ignore'):
            mean_resp = np.nanmean(resp, axis=1)
            var_resp  = np.nanvar(resp, axis=1)
            fano      = var_resp / np.where(mean_resp == 0, np.nan, mean_resp)
        ses.celldata['qc_fano'] = fano
    else:
        ses.celldata['qc_fano'] = np.nan

    return ses


def flag_bad_cells(ses, rate_thr=RATE_THR, noise_thr=NOISE_THR, fano_thr=FANO_THR,
                    skew_min=SKEW_MIN, flat_std_thr=FLAT_STD_THR, nan_frac_thr=NAN_FRAC_THR):
    """
    Combine QC metrics (computed via compute_qc_metrics) into pass/fail flags,
    keeping each individual reason separate so you can see WHY a cell failed.
    Requires compute_qc_metrics(ses) to have been run first.

    Adds to ses.celldata:
        qc_flag_<reason>  : one boolean column per entry in REASON_COLS
        qc_fail_reason    : comma-joined string of active reasons, or 'pass'
        qc_silent         : flat OR lowrate (kept for backward compatibility)
        qc_artifact       : nan OR lowskew OR highfano (kept for backward compatibility)
        qc_pass           : True iff no reason flag is active

    Returns
    -------
    idx_pass : boolean numpy array, len(ses.celldata)
    """
    for col in ['qc_rate', 'qc_std', 'qc_skew', 'qc_nan_frac']:
        assert col in ses.celldata, f"Run compute_qc_metrics(ses) first - missing '{col}'."

    fano_col = ses.celldata['qc_fano'] if 'qc_fano' in ses.celldata else \
        pd.Series(np.nan, index=ses.celldata.index)
    noise_col = ses.celldata['noise_level'] if 'noise_level' in ses.celldata else \
        pd.Series(np.nan, index=ses.celldata.index)

    flags = {
        'flat':     (ses.celldata['qc_std'] < flat_std_thr).to_numpy(),
        'lowrate':  (ses.celldata['qc_rate'] < rate_thr).to_numpy(),
        'nan':      (ses.celldata['qc_nan_frac'] > nan_frac_thr).to_numpy(),
        'lowskew':  (ses.celldata['qc_skew'] < skew_min).to_numpy(),
        'highfano': (fano_col > fano_thr).to_numpy(),
        'noisy':    (noise_col > noise_thr).to_numpy(),
    }

    for reason in REASON_COLS:
        ses.celldata[f'qc_flag_{reason}'] = flags[reason]

    reasons_df = pd.DataFrame({r: flags[r] for r in REASON_COLS}, index=ses.celldata.index)
    ses.celldata['qc_fail_reason'] = reasons_df.apply(
        lambda row: ','.join(row.index[row.to_numpy()]) if row.any() else 'pass', axis=1)

    ses.celldata['qc_silent']   = flags['flat'] | flags['lowrate']
    ses.celldata['qc_artifact'] = flags['nan'] | flags['lowskew'] | flags['highfano']
    ses.celldata['qc_pass']     = ~np.any(list(flags.values()), axis=0)

    return ses.celldata['qc_pass'].to_numpy()


def run_qc(sessions, **kwargs):
    """
    Convenience wrapper: run compute_qc_metrics + flag_bad_cells over a list
    of sessions, and print a one-line summary per session.

    Returns the (mutated) list of sessions; each ses.celldata now has the
    qc_* columns and a 'qc_pass' boolean column that can be combined with
    other selection filters (e.g. filter_nearlabeled_layer23).
    """
    for ses in sessions:
        compute_qc_metrics(ses)
        idx_pass = flag_bad_cells(ses, **kwargs)
        n_total, n_pass = len(ses.celldata), int(np.sum(idx_pass))
        sid = ses.celldata['session_id'].iloc[0] if 'session_id' in ses.celldata and n_total else 'unknown'
        reason_counts = ', '.join(
            f"{r}={int(ses.celldata[f'qc_flag_{r}'].sum())}" for r in REASON_COLS
            if ses.celldata[f'qc_flag_{r}'].sum() > 0)
        print(f"[QC] {sid}: {n_pass}/{n_total} cells pass" +
              (f" (failures by reason: {reason_counts})" if reason_counts else ""))
    return sessions


def summarize_qc(sessions, groupby=('session_id', 'roi_name', 'labeled')):
    """
    Detailed pass/fail + per-reason breakdown, grouped by any combination of
    celldata columns (default: session x area x label). Requires run_qc(sessions)
    to have been run first.

    Returns a DataFrame with one row per group: n_total, n_pass, frac_pass,
    and one column per failure reason (count of cells flagged for that
    reason within the group -- a cell with multiple reasons is counted in
    each, so these can sum to more than n_total - n_pass).
    """
    celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)
    assert 'qc_pass' in celldata, "Run run_qc(sessions) first."
    groupby = [g for g in groupby if g in celldata.columns]

    agg_dict = {'n_total': ('qc_pass', 'size'), 'n_pass': ('qc_pass', 'sum')}
    agg_dict.update({r: (f'qc_flag_{r}', 'sum') for r in REASON_COLS})
    summary = celldata.groupby(groupby).agg(**agg_dict).reset_index()
    summary['frac_pass'] = summary['n_pass'] / summary['n_total']
    return summary.sort_values('frac_pass')


def plot_qc_summary(sessions):
    """
    Diagnostic plots across ALL sessions/areas/labels pooled: distributions
    of the QC metrics and fraction of cells passing per session, to spot
    problematic sessions/planes at a glance. See plot_qc_by_group and
    plot_qc_reasons for the area/label- and reason-level breakdowns.
    """
    celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)
    assert 'qc_pass' in celldata, "Run run_qc(sessions) first."

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))

    axes[0].hist(celldata['noise_level'].dropna(), bins=50, color='grey')
    axes[0].axvline(NOISE_THR, color='r', linestyle='--')
    axes[0].set_xlabel('noise_level'); axes[0].set_title('Noise level')

    axes[1].hist(celldata['qc_rate'].dropna(), bins=50, color='grey')
    axes[1].axvline(RATE_THR, color='r', linestyle='--')
    axes[1].set_xlabel('qc_rate'); axes[1].set_title('Activity rate')

    axes[2].hist(celldata['qc_skew'].dropna(), bins=50, color='grey')
    axes[2].axvline(SKEW_MIN, color='r', linestyle='--')
    axes[2].set_xlabel('qc_skew'); axes[2].set_title('Trace skewness')

    frac_pass = celldata.groupby('session_id')['qc_pass'].mean().sort_values()
    axes[3].barh(np.arange(len(frac_pass)), frac_pass.values, color='grey')
    axes[3].set_yticks(np.arange(len(frac_pass)))
    axes[3].set_yticklabels(frac_pass.index, fontsize=6)
    axes[3].set_xlabel('Fraction cells passing QC'); axes[3].set_title('Per-session QC')

    fig.tight_layout()
    return fig


def plot_qc_reasons(sessions, groupby='session_id'):
    """
    Stacked bar chart of HOW MANY cells were flagged, and WHY, per group
    (default: per session; pass groupby='roi_name', ['roi_name','labeled'],
    etc. for other breakdowns). One bar per group, one color per reason
    (see REASON_COLORS/REASON_LABELS), stacked -- so both the total flagged
    count and its composition are visible at a glance.
    """
    celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)
    assert 'qc_fail_reason' in celldata, "Run run_qc(sessions) first."

    grouped = celldata.groupby(groupby)[[f'qc_flag_{r}' for r in REASON_COLS]].sum()
    grouped = grouped.loc[grouped.sum(axis=1).sort_values(ascending=False).index]
    group_labels = [' / '.join(map(str, g)) if isinstance(g, tuple) else str(g)
                    for g in grouped.index]

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(grouped)), 5))
    x = np.arange(len(grouped))
    bottom = np.zeros(len(grouped))
    for r in REASON_COLS:
        vals = grouped[f'qc_flag_{r}'].to_numpy()
        ax.bar(x, vals, bottom=bottom, color=REASON_COLORS[r], label=REASON_LABELS[r])
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Number of cells flagged')
    ax.set_title(f'QC failure reasons by {groupby}')
    ax.legend(fontsize=8, loc='upper right')
    fig.tight_layout()
    return fig


def plot_qc_by_group(sessions, groupby=('roi_name', 'labeled'), metrics=('noise_level', 'qc_rate', 'qc_skew')):
    """
    Grid of small-multiple histograms, one row per group (default: area x
    label), one column per metric, with pass/fail cells colored separately
    so you can see WHERE in the distribution the flagged cells sit (e.g. a
    tail of high noise_level vs. a cluster of near-zero qc_rate). Each row's
    title reports n_total / n_pass and the dominant failure reasons for that
    group.
    """
    celldata = pd.concat([ses.celldata for ses in sessions]).reset_index(drop=True)
    assert 'qc_pass' in celldata, "Run run_qc(sessions) first."
    groupby = [g for g in groupby if g in celldata.columns]
    metrics = [m for m in metrics if m in celldata.columns]

    groups = list(celldata.groupby(groupby))
    n_groups, n_metrics = len(groups), len(metrics)

    fig, axes = plt.subplots(n_groups, n_metrics, figsize=(4 * n_metrics, 2.5 * n_groups),
                              squeeze=False)

    for i, (g, sub) in enumerate(groups):
        n_total, n_pass = len(sub), int(sub['qc_pass'].sum())

        top_reasons = (sub.loc[~sub['qc_pass'], [f'qc_flag_{r}' for r in REASON_COLS]]
                       .sum().sort_values(ascending=False))
        top_reasons = top_reasons[top_reasons > 0]
        reason_str = ', '.join(f'{r.replace("qc_flag_", "")}={int(c)}' for r, c in top_reasons.items())

        glabel = ' / '.join(map(str, g)) if isinstance(g, tuple) else str(g)
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            ax.hist(sub.loc[sub['qc_pass'], metric].dropna(), bins=40, color='tab:green',
                    alpha=0.6, label='pass')
            ax.hist(sub.loc[~sub['qc_pass'], metric].dropna(), bins=40, color='tab:red',
                    alpha=0.6, label='flagged')
            if j == 0:
                ax.set_ylabel(f'{glabel}\n(n={n_total}, pass={n_pass})', fontsize=8)
            if i == 0:
                ax.set_title(metric, fontsize=9)
            if i == n_groups - 1:
                ax.set_xlabel(metric, fontsize=8)
        axes[i, 0].legend(fontsize=6, loc='upper right')
        axes[i, -1].annotate(reason_str, xy=(1.02, 0.5), xycoords='axes fraction',
                              fontsize=6, va='center', ha='left', wrap=True)

    fig.tight_layout()
    return fig


@functools.lru_cache(maxsize=8)
def _read_qc_table(qc_csv_path):
    """Cached CSV read so repeated calls (one per session) don't re-hit disk."""
    return pd.read_csv(qc_csv_path)


def load_qc_pass(ses, qc_csv_path, id_cols=('session_id', 'cell_id')):
    """
    Load a previously computed QC pass/fail flag (from run_qc_summary.py's
    qc_per_cell.csv) and return it as a boolean array aligned with
    ses.celldata, for use as a filter in downstream analysis scripts --
    this is the intended way to combine QC with e.g. the anatomical
    filter_nearlabeled_layer23 selection, without having to reload raw
    calcium traces / recompute QC metrics inside every analysis script.

    Parameters
    ----------
    ses : Session
        Must have ses.celldata with the id_cols present.
    qc_csv_path : str or None
        Path to qc_per_cell.csv (produced by run_qc_summary.py). If None,
        QC filtering is skipped entirely and everything passes (useful for
        quick exploration before QC has been run).
    id_cols : tuple of str
        Columns used to match cells between ses.celldata and the QC table.
        Default ('session_id', 'cell_id'), the celldata primary key
        convention used throughout this codebase.

    Returns
    -------
    idx_qc_pass : boolean numpy array, len(ses.celldata)
        True where the cell passed QC (or QC wasn't found / wasn't
        requested -- see warnings below for how missing entries are
        handled, so failures are visible rather than silent).
    """
    if qc_csv_path is None:
        return np.ones(len(ses.celldata), dtype=bool)

    import os
    if not os.path.exists(qc_csv_path):
        raise FileNotFoundError(
            f"QC table not found at '{qc_csv_path}'. Run run_qc_summary.py first "
            f"(it writes qc_per_cell.csv to its output_dir), or pass qc_csv_path=None "
            f"to skip QC filtering.")

    qc_df = _read_qc_table(qc_csv_path)

    id_cols = [c for c in id_cols if c in ses.celldata.columns and c in qc_df.columns]
    assert id_cols, ("No shared id column between ses.celldata and the QC table to merge "
                      "on -- check that both have matching 'session_id'/'cell_id' columns.")
    if 'cell_id' not in id_cols:
        raise ValueError(
            f"load_qc_pass is about to merge on {id_cols} only (no 'cell_id') -- this will "
            f"produce a cross-join within each session rather than a per-cell match. Check "
            f"that qc_per_cell.csv (written by run_qc_summary.py) actually has a 'cell_id' "
            f"column; if the column name differs in this codebase, pass the right name via "
            f"id_cols=(...).")

    merged = ses.celldata[id_cols].merge(qc_df[id_cols + ['qc_pass']], on=id_cols, how='left')

    if len(merged) != len(ses.celldata):
        raise ValueError(
            f"load_qc_pass merge produced {len(merged)} rows from {len(ses.celldata)} input "
            f"cells -- the join key {id_cols} isn't unique per cell in one of the two tables "
            f"(duplicate ids), so this would silently mismatch cells. Check for duplicate "
            f"'cell_id' values within a session in either ses.celldata or the QC table.")

    n_missing = int(merged['qc_pass'].isna().sum())
    if n_missing > 0:
        sid = ses.celldata['session_id'].iloc[0] if 'session_id' in ses.celldata.columns else '?'
        print(f"WARNING: {n_missing}/{len(merged)} cells in session {sid} have no matching "
              f"entry in {qc_csv_path} -- treating as QC PASS (QC table may be stale; "
              f"re-run run_qc_summary.py if this session/cell set has changed).")
        merged['qc_pass'] = merged['qc_pass'].fillna(True)

    return merged['qc_pass'].to_numpy().astype(bool)
