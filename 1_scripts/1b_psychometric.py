# -*- coding: utf-8 -*-
"""
1b_psychometric
================
Psychometric-curve fit and session-level QC filtering, across all three
protocols (DM, DP, DN). The curve fit itself only runs on DN -- DM has
no intermediate stimulus levels at all, and DP's aren't used for fitting
either (fit on engaged trials only when it does run). DM and DP still go
through the d'/FA/engagement exclusion criteria, just not the
threshold-range one, which only makes sense where a fit exists.

Session inclusion criteria:
    - d' (engaged trials) >= MIN_DPRIME
    - false-alarm rate (engaged trials) <= MAX_FA_RATE
    - engaged-trial fraction >= MIN_FRAC_ENGAGED (same threshold 1a uses
      to flag a session -- here it actually excludes)
    - DN only: the fitted threshold (mu) falls within the range of
      intermediate levels actually presented, i.e. interpolated rather
      than extrapolated -- checked via the z-scored range of those
      stimuli relative to the fit (same check as the old `noise_to_psy`'s
      noise_zmin/noise_zmax straddling zero)
    - DN only: fitted guess rate (floor of the curve) <= MAX_GUESS_RATE
  BORDERLINE zone: for the two DN fit criteria (threshold range, guess rate) a value within BORDERLINE_Z sigma /
  BORDERLINE_GUESS of the cutoff is NOT excluded: the session is kept and flagged `borderline` (+ `borderline_reasons`)
  in psychometric_fits.csv, because a different fit could flip it. Beyond the zone the criterion excludes as usual.
All five thresholds live in infotheory.criteria, shared with 1a's flags.

Project-wide rules (infotheory/project_config.py) -- what applies to 1b
    * selection : --animals / --sessions restrict the sessions; a restricted run writes to
                  out/selection/<tag>/ and NEVER overwrites psychometric_fits.csv in out/ (read by every
                  later step). The fit and the inclusion criteria are per session, so selecting
                  sessions does not change any session's result.
    * memory    : sessions are processed one at a time (trial tables in section 2, behavior for the window
                  timing in section 7), with per-session caches in store/ (--n-jobs sets the parallelism).
    * aggregate : --aggregate pooled|session|mouse changes the clean-data overview plots (section 6)
                  only; the psychometric fits (sections 3-5) and the window-timing histograms (section 7)
                  are inherently per session / per trial and are not affected.
    * binning   : not used here.

Reads: 2_pipeline/1a_performance/out/session_summary.csv for the
engaged-trial d'/FA/engagement stats (Rule #1: read from an earlier
script's out/, don't recompute what 1a already did).
Loads: raw trialdata itself for DM/DP/DN -- 1a's cache is off-limits
(Rule #1 again), and the trial-by-trial fit needs it anyway.

Outputs
-------
2_pipeline/1b_psychometric/
    out/    psychometric_fits.csv     one row per session (all 3
                                       protocols): fit params, r2,
                                       inclusion + reason
            excluded_sessions.csv     the excluded subset, reasons only
            window_timing.csv         per-trial stim/reward window timing
                                       (see section 7)
            figures/*.png             fit figures, the 1a overview plots
                                       redone on the clean subset (section
                                       6), and the stim/reward window
                                       timing figure (section 7)
    out/selection/<tag>/...   the same files for a restricted run
    store/  engaged_trials_<hash>.pkl   per-session cache of engaged trials
            window_timing_<hash>.pkl    per-session cache of window timing (section 7)
"""
from __future__ import annotations

import argparse
import zlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from infotheory.pipeline import get_pipeline_paths
from infotheory.session import load_sessions, PROTOCOLS
from infotheory.psychometric import fit_psychometric, psychometric_function
from infotheory.behavior import add_trial_outcome
from infotheory.psth import derive_onset_time
from infotheory.plotting import set_style, save_fig, clear_figures, OUTCOME_COLORS, PROTOCOL_COLORS
from infotheory.criteria import MIN_DPRIME, MAX_FA_RATE, MIN_FRAC_ENGAGED, MAX_GUESS_RATE, BORDERLINE_Z, BORDERLINE_GUESS
from infotheory.session import discover_sessions
from infotheory.scope import add_scope_args, scope_from_args, select_sessions, scoped_dirs, data_stamp, unit_noun
from infotheory.session_jobs import run_sessions
from infotheory.unit_stats import unit_summary
from infotheory.unit_plots import unit_bars, mouse_palette, mouse_legend, stacked_outcomes

MAX_PANELS_PER_FIG = 24      # per-session grid: paginate if more sessions than this
GRID_NCOLS = 4
STIM_WIDTH_CM = 20.0         # width of the stimulus zone in cm -- trialdata only stores stimStart
                              # (the zone's start position), not an end position, so this needs to be
                              # supplied; keep in sync with 1c_behavior.py's MI_WINDOWS stim=(0, 20)

N_BOOTSTRAP = 500      # DN psychometric fit: median over this many bootstrap refits (0 = single fit); --n-bootstrap
parser = argparse.ArgumentParser()
add_scope_args(parser)
parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP,
                    help="bootstrap resamples for the DN psychometric fit (median of the refits, seeded per session; "
                         "0 = single fit). Default %(default)s")
args = parser.parse_args()
scope = scope_from_args(args)
AGG = scope.aggregate
UNIT = unit_noun(AGG)

paths = get_pipeline_paths(__file__)
table_dir, figdir = scoped_dirs(paths, scope)
clear_figures(figdir)
set_style()
print(scope.describe())


# --------------------------------------------------------------------- #
# 1. Read 1a's session summary for the pre-computed engaged-trial
#    d'/FA/engagement stats, across all protocols.
# --------------------------------------------------------------------- #
summary_1a_path = paths.out_from("1a_performance") / "session_summary.csv"
if not summary_1a_path.exists():
    raise SystemExit(f"Run 1a_performance.py first -- expected {summary_1a_path}")

summary_1a = pd.read_csv(summary_1a_path)
summary_1a["fa_rate_engaged"] = summary_1a["n_FA_engaged"] / (summary_1a["n_FA_engaged"] + summary_1a["n_CR_engaged"])
summary_1a = summary_1a.set_index("session_id")


# --------------------------------------------------------------------- #
# 2. Load raw trial data, engaged trials only, all 3 protocols -- or
#    reuse the cache.
# --------------------------------------------------------------------- #
all_triples = discover_sessions(PROTOCOLS)
triples = select_sessions(all_triples, scope)
session_ids = [f"{a}_{d}" for _, a, d in triples]
stamps = {f"{a}_{d}": data_stamp(p, a, d) for p, a, d in triples}
mouse_colors = mouse_palette(a for _, a, _ in all_triples)
print(f"{len(triples)}/{len(all_triples)} sessions selected")


def engaged_trials_of_session(ses, seed):
    """Engaged trials of ONE session (the only thing section 3 needs from the raw data)."""
    if ses.trialdata is None:
        return None
    trial = ses.trialdata.copy()
    if "engaged" in trial.columns:
        trial = trial[trial["engaged"] == 1]
    trial["session_id"] = ses.session_id
    trial["animal_id"] = ses.animal_id
    trial["protocol"] = ses.protocol
    return trial


per_session = run_sessions(
    session_ids, engaged_trials_of_session, list(PROTOCOLS), paths.store, "engaged_trials",
    key_params=dict(version="1b-v2"), load_kwargs=dict(load_behaviordata=False, load_videodata=False),
    n_jobs=scope.n_jobs, recompute=args.recompute, extra_key_by_session=stamps, flatten=False)
rows = [t for t in per_session if t is not None]
if not rows:
    raise SystemExit("No sessions found under 0_data/.")
engaged_trialdata = pd.concat(rows, ignore_index=True)
del per_session, rows


# --------------------------------------------------------------------- #
# 3. Fit each session (where possible) and apply the inclusion criteria
# --------------------------------------------------------------------- #
records = []
fits = {}  # session_id -> PsychometricFit or None

for session_id, trial in engaged_trialdata.groupby("session_id"):
    protocol = trial["protocol"].iloc[0]
    reasons = []
    flags = []   # borderline notes: the session is kept (if reasons is empty) but flagged

    if session_id not in summary_1a.index:
        dprime_engaged = fa_rate_engaged = frac_engaged = np.nan
        reasons.append("session missing from 1a session_summary.csv")
    else:
        dprime_engaged = summary_1a.loc[session_id, "dprime_engaged"]
        fa_rate_engaged = summary_1a.loc[session_id, "fa_rate_engaged"]
        frac_engaged = summary_1a.loc[session_id, "eng_frac_engaged"]

    if np.isnan(dprime_engaged) or dprime_engaged < MIN_DPRIME:
        reasons.append(f"d' (engaged) = {dprime_engaged:.2f} < {MIN_DPRIME}")
    if np.isnan(fa_rate_engaged) or fa_rate_engaged > MAX_FA_RATE:
        reasons.append(f"FA rate (engaged) = {fa_rate_engaged:.0%} > {MAX_FA_RATE:.0%}")
    if np.isnan(frac_engaged) or frac_engaged < MIN_FRAC_ENGAGED:
        reasons.append(f"engaged fraction = {frac_engaged:.0%} < {MIN_FRAC_ENGAGED:.0%}")

    n_signal_levels = trial["signal"].nunique()
    fit_eligible = protocol == "DN"  # only DN's curve fit is used; DM has no intermediate
                                      # levels to fit at all, and DP's aren't fit either
    fit = None
    fit_status = "not_applicable"  # DM/DP -- expected, not a QC failure

    if fit_eligible:
        fit = fit_psychometric(trial, n_bootstrap=args.n_bootstrap,
                               seed=zlib.crc32(str(session_id).encode()))   # seeded per session: reproducible
        intermediate = trial.loc[(trial["signal"] > 0) & (trial["signal"] < 100), "signal"].to_numpy()
        if fit is None:
            fit_status = "failed_to_converge"
            reasons.append("psychometric fit failed to converge")
        else:
            # range check: margin = distance (sigma) of the nearer edge of the tested range from the threshold;
            # >= 0 means the range brackets the threshold, < 0 means it is extrapolated by that much.
            zmin, zmax = fit.threshold_z_range(intermediate)
            range_margin = min(-zmin, zmax)
            if range_margin < -BORDERLINE_Z:
                fit_status = "threshold_out_of_range"
                reasons.append(f"threshold (mu={fit.mu:.0f}%) outside tested stimulus range "
                                f"(z-range [{zmin:.2f}, {zmax:.2f}] doesn't bracket 0)")
            else:
                fit_status = "ok"
                if abs(range_margin) <= BORDERLINE_Z:
                    flags.append(f"threshold at the edge of the tested range (z-range [{zmin:.2f}, {zmax:.2f}], "
                                 f"margin {range_margin:+.2f} sigma)")
            # guess rate: independent of the range check (both reasons are listed when both fail)
            guess_margin = MAX_GUESS_RATE - fit.guess_rate
            if guess_margin < -BORDERLINE_GUESS:
                reasons.append(f"guess rate = {fit.guess_rate:.2f} > {MAX_GUESS_RATE}")
            elif abs(guess_margin) <= BORDERLINE_GUESS:
                flags.append(f"guess rate {fit.guess_rate:.3f} next to the {MAX_GUESS_RATE} limit")

    fits[session_id] = fit
    records.append({
        "session_id": session_id,
        "protocol": protocol,
        "n_trials_engaged": len(trial),
        "dprime_engaged": dprime_engaged,
        "fa_rate_engaged": fa_rate_engaged,
        "frac_engaged": frac_engaged,
        "n_signal_levels": n_signal_levels,
        "fit_status": fit_status,
        "mu": fit.mu if fit else np.nan,
        "sigma": fit.sigma if fit else np.nan,
        "lapse_rate": fit.lapse_rate if fit else np.nan,
        "guess_rate": fit.guess_rate if fit else np.nan,
        "r2": fit.r2 if fit else np.nan,
        "included": len(reasons) == 0,
        "exclude_reasons": "; ".join(reasons),
        "borderline": len(reasons) == 0 and len(flags) > 0,   # kept, but a different fit could flip it
        "borderline_reasons": "; ".join(flags) if len(reasons) == 0 else "",
    })

fits_df = pd.DataFrame(records).sort_values(["protocol", "session_id"]).reset_index(drop=True)
fits_df.to_csv(table_dir / "psychometric_fits.csv", index=False)
fits_df.loc[~fits_df["included"], ["session_id", "protocol", "exclude_reasons"]].to_csv(
    table_dir / "excluded_sessions.csv", index=False)

print(f"\nInclusion criteria: d' >= {MIN_DPRIME}, FA <= {MAX_FA_RATE:.0%}, "
      f"engaged fraction >= {MIN_FRAC_ENGAGED:.0%}, threshold within tested range and "
      f"guess rate <= {MAX_GUESS_RATE} (both DN only)")
for protocol in PROTOCOLS:
    sub = fits_df[fits_df["protocol"] == protocol]
    if len(sub):
        print(f"  {protocol}: {int(sub['included'].sum())}/{len(sub)} sessions included")
if (~fits_df["included"]).any():
    print(fits_df.loc[~fits_df["included"], ["session_id", "protocol", "exclude_reasons"]].to_string(index=False))
if fits_df["borderline"].any():
    print(f"\nBORDERLINE sessions (KEPT, but within {BORDERLINE_Z} sigma / {BORDERLINE_GUESS} guess of a cutoff):")
    print(fits_df.loc[fits_df["borderline"], ["session_id", "protocol", "borderline_reasons"]].to_string(index=False))


# --------------------------------------------------------------------- #
# 4. Per-session psychometric curves, one figure per protocol (data +
#    fit where a fit exists; DM sessions show data points only). Excluded
#    sessions get a grey curve/red title instead of being dropped from
#    the figure. Paginated for many sessions.
# --------------------------------------------------------------------- #
included_lookup = fits_df.set_index("session_id")["included"]

for protocol in PROTOCOLS:
    protocol_sessions = fits_df.loc[fits_df["protocol"] == protocol, "session_id"].to_numpy()
    if len(protocol_sessions) == 0:
        continue

    n_pages = int(np.ceil(len(protocol_sessions) / MAX_PANELS_PER_FIG))
    for page in range(n_pages):
        page_sessions = protocol_sessions[page * MAX_PANELS_PER_FIG:(page + 1) * MAX_PANELS_PER_FIG]
        n_panels = len(page_sessions)
        ncols = min(GRID_NCOLS, n_panels)
        nrows = int(np.ceil(n_panels / ncols))

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.8), squeeze=False)
        for i, session_id in enumerate(page_sessions):
            ax = axes[i // ncols][i % ncols]
            fit = fits[session_id]
            included = bool(included_lookup[session_id])
            trial = engaged_trialdata[engaged_trialdata["session_id"] == session_id]
            psydata = trial.groupby("signal")["lickResponse"].mean()

            ax.scatter(psydata.index.to_numpy(), psydata.to_numpy(), color="black", zorder=3, s=20)
            if fit is not None:
                x_hi = np.linspace(0, 100, 200)
                ax.plot(x_hi, fit.predict(x_hi), color="tab:blue" if included else "lightgrey",
                        linewidth=2, zorder=2)
                if included:
                    ax.axvline(fit.mu, linestyle="--", color="grey", linewidth=1)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlim(-2, 102)
            ax.set_title(session_id, fontsize=8, color="black" if included else "firebrick")
            ax.tick_params(labelsize=7)

        for j in range(n_panels, nrows * ncols):
            axes[j // ncols][j % ncols].set_visible(False)

        fig.supxlabel("signal strength (%)", fontsize=9)
        fig.supylabel("P(lick)", fontsize=9)
        page_suffix = f"_page{page + 1}" if n_pages > 1 else ""
        fig.suptitle(f"Psychometric fits -- {protocol}"
                     + (f" (page {page + 1}/{n_pages})" if n_pages > 1 else ""), y=1.02)
        fig.tight_layout()
        save_fig(fig, figdir, f"1_psychometric_fits_{protocol}{page_suffix}")


# --------------------------------------------------------------------- #
# 5. Population overlay per protocol (only where fits exist -- DM has
#    none, so it's skipped automatically).
# --------------------------------------------------------------------- #
for protocol in PROTOCOLS:
    protocol_sessions = fits_df.loc[fits_df["protocol"] == protocol, "session_id"].to_numpy()
    included_fits = [fits[sid] for sid in protocol_sessions
                      if included_lookup.get(sid, False) and fits[sid] is not None]
    if not included_fits:
        continue

    x_hi = np.linspace(0, 100, 200)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for fit in included_fits:
        ax.plot(x_hi, fit.predict(x_hi), color="grey", alpha=0.4, linewidth=1)

    median_params = np.median([[f.mu, f.sigma, f.lapse_rate, f.guess_rate] for f in included_fits], axis=0)
    ax.plot(x_hi, psychometric_function(x_hi, *median_params), color="black", linewidth=2.5,
            label=f"median (n={len(included_fits)})")
    ax.set_xlabel("signal strength (%)")
    ax.set_ylabel("P(lick)")
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 100)
    ax.set_title(f"Psychometric fits -- {protocol}, included sessions")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    save_fig(fig, figdir, f"2_psychometric_population_{protocol}")


# --------------------------------------------------------------------- #
# 6. Redo the 1a_performance overview plots, but on the CLEAN data only:
#    engaged trials (already true of engaged_trialdata) restricted to
#    sessions that passed all inclusion criteria above -- with DM/DP/DN
#    back as separate subplots/panels, same as the original 1a figures.
# --------------------------------------------------------------------- #
included_ids = fits_df.loc[fits_df["included"], "session_id"].to_numpy()

if len(included_ids) == 0:
    print("No included sessions -- skipping the clean-data overview plots.")
else:
    clean_trialdata = add_trial_outcome(
        engaged_trialdata[engaged_trialdata["session_id"].isin(included_ids)].copy())

    print(f"\nClean dataset: {len(included_ids)}/{len(fits_df)} sessions, "
          f"{len(clean_trialdata)}/{len(engaged_trialdata)} engaged trials kept after session exclusion")

    # Unit table of the clean sessions at the chosen aggregation level. Counts are ENGAGED trials only
    # (the 1a summary's *_engaged columns are mapped onto the columns unit_stats aggregates), so pooled /
    # per-mouse d' is recomputed from pooled engaged-trial counts.
    clean = summary_1a.reset_index()
    clean = clean[clean["session_id"].isin(included_ids)].copy()
    clean["n_trials"] = clean["eng_n_engaged"]
    for o in ("HIT", "MISS", "FA", "CR"):
        clean[f"n_{o}_all"] = clean[f"n_{o}_engaged"]
    units = unit_summary(clean, AGG, by=["protocol"])
    units.to_csv(table_dir / f"clean_unit_summary_{AGG}.csv", index=False)
    protocols_present = [p for p in PROTOCOLS if p in units["protocol"].unique()]
    pcol = {p: PROTOCOL_COLORS[p] for p in protocols_present}

    # 6a. Dataset overview: sessions per animal (always per mouse), engaged trials and duration per unit
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    cnt = clean.groupby(["animal_id", "protocol"]).size().unstack(fill_value=0).reindex(columns=protocols_present)
    cnt.plot(kind="bar", stacked=True, ax=axes[0], color=[PROTOCOL_COLORS[p] for p in cnt.columns], width=0.8)
    axes[0].set_title("Sessions per mouse (clean)")
    axes[0].set_ylabel("# sessions")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", labelsize=7, rotation=60)
    axes[0].legend(title=None, fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    unit_bars(axes[1], units, "protocol", "n_trials", protocols_present, AGG, colors=pcol, mouse_colors=mouse_colors,
              ylabel=f"engaged trials per {UNIT}")
    axes[1].set_title(f"Engaged trials per {UNIT} (clean)")
    unit_bars(axes[2], units, "protocol", "duration_min", protocols_present, AGG, colors=pcol,
              mouse_colors=mouse_colors, ylabel=f"duration per {UNIT} (min)")
    axes[2].set_title(f"Recording time per {UNIT} (clean)")
    mouse_legend(axes[2], mouse_colors, animals=clean["animal_id"].unique(), loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    save_fig(fig, figdir, "3_dataset_overview_clean")
    plt.close(fig)

    # 6b. Trial-outcome distribution (HIT/MISS/FA/CR), clean data only, one bar per unit
    fig, ax = plt.subplots(figsize=(max(5, 0.3 * len(units) + 2), 3.6))
    stacked_outcomes(ax, units, AGG, protocols_present, OUTCOME_COLORS)
    ax.set_title(f"Trial outcomes (clean) -- one bar per {UNIT}")
    fig.tight_layout()
    save_fig(fig, figdir, "4_trial_outcomes_clean")
    plt.close(fig)

    # 6c. d' (engaged trials, from the unit's pooled counts) and criterion, clean sessions
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    unit_bars(axes[0], units, "protocol", "dprime_all", protocols_present, AGG, colors=pcol, mouse_colors=mouse_colors,
              ylabel="d' (engaged)", hline=MIN_DPRIME)
    axes[0].set_title(f"d' per {UNIT} (clean)")
    for protocol in protocols_present:
        u = units[units["protocol"] == protocol]
        axes[1].scatter(u["dprime_all"], u["criterion_all"], s=28, color=PROTOCOL_COLORS[protocol],
                        edgecolor="black", linewidth=0.3, label=protocol)
    axes[1].axhline(0, color="grey", linewidth=0.8)
    axes[1].axvline(0, color="grey", linewidth=0.8)
    axes[1].set_xlabel("d'")
    axes[1].set_ylabel("criterion")
    axes[1].set_title(f"Bias vs. sensitivity (clean), per {UNIT}", fontsize=9)
    axes[1].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save_fig(fig, figdir, "5_dprime_criterion_clean")
    plt.close(fig)

    # 6d. Signal-strength trial counts per unit, clean data only
    ckeys = ["protocol"] + {"pooled": [], "session": ["animal_id", "session_id"], "mouse": ["animal_id"]}[AGG]
    counts = clean_trialdata.groupby(ckeys + ["signal"]).size().reset_index(name="n_trials")
    fig, axes = plt.subplots(1, len(protocols_present), figsize=(3.8 * len(protocols_present), 3.2), squeeze=False)
    for ax, protocol in zip(axes[0], protocols_present):
        c = counts[counts["protocol"] == protocol]
        unit_bars(ax, c, "signal", "n_trials", sorted(c["signal"].unique()), AGG, colors=PROTOCOL_COLORS[protocol],
                  mouse_colors=mouse_colors, ylabel=f"trials per {UNIT}")
        ax.set_title(protocol)
        ax.set_xlabel("signal strength (%)")
    fig.suptitle(f"Trials per signal level, per {UNIT} (clean)", y=1.03)
    fig.tight_layout()
    save_fig(fig, figdir, "6_signal_level_balance_clean")
    plt.close(fig)

# --------------------------------------------------------------------- #
# 7. Stim/reward windows are fixed in SPACE (stimStart/rewardZoneStart/
#    rewardZoneEnd are positions -- see psth.py's module docstring) --
#    how are they distributed in TIME? For each included/engaged trial,
#    derives the wall-clock time the animal's position crosses into and
#    out of each window (psth.derive_onset_time on the window's start
#    and end positions), then looks at: how long the animal spends in
#    each window (duration), when the window falls relative to the
#    trial's own start, and whether one trial's window ever overlaps in
#    TIME with the next trial's same window (gap < 0) -- which would be
#    a problem for any purely trial-locked, non-overlapping-window
#    analysis (like 1c_behavior.py's MI_WINDOWS).
# --------------------------------------------------------------------- #
if len(included_ids) == 0:
    print("No included sessions -- skipping the stim/reward timing figure.")
    timing = pd.DataFrame()
else:
    def window_timing_of_session(ses, seed):
        """Window timing of ONE included session (behavior trace is loaded for this session only)."""
        if ses.trialdata is None or ses.behaviordata is None:
            return None
        trial = (ses.trialdata[ses.trialdata["engaged"] == 1].copy()
                 if "engaged" in ses.trialdata.columns else ses.trialdata.copy())
        if trial.empty:
            return None
        trial["_stim_end_pos"] = trial["stimStart"] + STIM_WIDTH_CM

        continuous = ses.behaviordata.sort_values("ts")
        trial = derive_onset_time(trial, continuous, position_value_col="stimStart", out_col="stim_start_time")
        trial = derive_onset_time(trial, continuous, position_value_col="_stim_end_pos", out_col="stim_end_time")
        trial = derive_onset_time(trial, continuous, position_value_col="rewardZoneStart", out_col="reward_start_time")
        trial = derive_onset_time(trial, continuous, position_value_col="rewardZoneEnd", out_col="reward_end_time")

        trial["stim_duration"] = trial["stim_end_time"] - trial["stim_start_time"]
        trial["reward_duration"] = trial["reward_end_time"] - trial["reward_start_time"]
        trial["stim_start_rel_tStart"] = trial["stim_start_time"] - trial["tStart"]
        trial["reward_start_rel_tStart"] = trial["reward_start_time"] - trial["tStart"]

        # Gap (s) between this trial's window ending and the NEXT trial's same window starting --
        # negative means overlap. Requires trials sorted by number (== chronological order).
        trial = trial.sort_values("trialNumber")
        trial["stim_gap_to_next"] = trial["stim_start_time"].shift(-1) - trial["stim_end_time"]
        trial["reward_gap_to_next"] = trial["reward_start_time"].shift(-1) - trial["reward_end_time"]

        trial["session_id"] = ses.session_id
        trial["protocol"] = ses.protocol
        return trial[[
            "session_id", "protocol", "trialNumber",
            "stim_start_time", "stim_end_time", "stim_duration",
            "stim_start_rel_tStart", "stim_gap_to_next",
            "reward_start_time", "reward_end_time", "reward_duration",
            "reward_start_rel_tStart", "reward_gap_to_next",
        ]]

    timing_ids = [sid for sid in session_ids if sid in set(included_ids)]
    per_session_timing = run_sessions(
        timing_ids, window_timing_of_session, list(PROTOCOLS), paths.store, "window_timing",
        key_params=dict(version="1b-v2", stim_width=STIM_WIDTH_CM),
        load_kwargs=dict(load_behaviordata=True, load_videodata=False),
        n_jobs=scope.n_jobs, recompute=args.recompute,
        extra_key_by_session={k: v for k, v in stamps.items() if k in set(timing_ids)}, flatten=False)
    parts = [t for t in per_session_timing if t is not None]
    timing = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    del per_session_timing, parts

    if len(timing):
        timing.to_csv(table_dir / "window_timing.csv", index=False)

        n_stim_checked = timing["stim_gap_to_next"].notna().sum()
        n_stim_overlap = (timing["stim_gap_to_next"] < 0).sum()
        n_reward_checked = timing["reward_gap_to_next"].notna().sum()
        n_reward_overlap = (timing["reward_gap_to_next"] < 0).sum()
        print(f"\nStim-window time overlaps with the next trial: {n_stim_overlap}/{n_stim_checked} "
              f"({n_stim_overlap / max(n_stim_checked, 1):.1%})")
        print(f"Reward-window time overlaps with the next trial: {n_reward_overlap}/{n_reward_checked} "
              f"({n_reward_overlap / max(n_reward_checked, 1):.1%})")

        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for row, (window_name, dur_col, start_col, gap_col) in enumerate([
            ("stim", "stim_duration", "stim_start_rel_tStart", "stim_gap_to_next"),
            ("reward", "reward_duration", "reward_start_rel_tStart", "reward_gap_to_next"),
        ]):
            ax = axes[row][0]
            sns.histplot(data=timing, x=dur_col, hue="protocol", palette=PROTOCOL_COLORS, ax=ax,
                         element="step", stat="density", common_norm=False, legend=(row == 0))
            ax.set_xlabel(f"{window_name} window duration (s)")
            ax.set_title(f"{window_name.capitalize()} duration")

            ax = axes[row][1]
            sns.histplot(data=timing, x=start_col, hue="protocol", palette=PROTOCOL_COLORS, ax=ax,
                         element="step", stat="density", common_norm=False, legend=False)
            ax.set_xlabel(f"{window_name} start, time since trial start (s)")
            ax.set_title(f"{window_name.capitalize()} start time (rel. trial start)")

            ax = axes[row][2]
            gap_vals = timing[gap_col].dropna()
            n_overlap = int((gap_vals < 0).sum())
            sns.histplot(gap_vals, ax=ax, color="grey", element="step")
            ax.axvline(0, color="firebrick", linestyle="--", linewidth=1.5)
            ax.set_xlabel(f"gap to next trial's {window_name} window (s)")
            ax.set_title(f"{window_name.capitalize()} overlap check "
                         f"({n_overlap}/{len(gap_vals)} overlap)")

        fig.suptitle("Stim/reward window timing -- windows are fixed in position; "
                     "when do they actually occur in time?", y=1.02)
        fig.tight_layout()
        save_fig(fig, figdir, "7_window_timing")


print(f"\nWrote tables to {table_dir}, figures to {figdir}")
print("Run `python b_progress/make_progress_md.py 1b_psychometric` to build a markdown summary.")
