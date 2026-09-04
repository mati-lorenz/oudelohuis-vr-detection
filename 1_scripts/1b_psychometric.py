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
All four thresholds live in infotheory.criteria, shared with 1a's flags.

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
            figures/*.png             fit figures, plus the 1a overview
                                       plots redone on the clean subset,
                                       one subplot/figure per protocol
                                       where relevant (section 6)
    store/  engaged_trialdata.pkl     cached engaged-trial data for all
                                       3 protocols, reused unless
                                       --recompute is passed
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from infotheory.pipeline import get_pipeline_paths
from infotheory.session import load_sessions, PROTOCOLS
from infotheory.psychometric import fit_psychometric, psychometric_function
from infotheory.behavior import add_trial_outcome
from infotheory.plotting import set_style, save_fig, clear_figures, OUTCOME_COLORS, PROTOCOL_COLORS
from infotheory.criteria import MIN_DPRIME, MAX_FA_RATE, MIN_FRAC_ENGAGED

MAX_PANELS_PER_FIG = 24      # per-session grid: paginate if more sessions than this
GRID_NCOLS = 4

parser = argparse.ArgumentParser()
parser.add_argument("--recompute", action="store_true",
                     help="Reload sessions from 0_data/ instead of reusing the cached "
                          "store/engaged_trialdata.pkl")
args = parser.parse_args()

paths = get_pipeline_paths(__file__)
figdir = paths.out / "figures"
clear_figures(figdir)
set_style()


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
trialdata_cache = paths.store / "engaged_trialdata.pkl"

if not args.recompute and trialdata_cache.exists():
    print(f"Reusing cached engaged trial data from {paths.store} (pass --recompute to reload 0_data/)")
    engaged_trialdata = pd.read_pickle(trialdata_cache)
else:
    sessions = load_sessions(protocols=PROTOCOLS, load_behaviordata=False, load_videodata=False)
    rows = []
    for ses in sessions:
        if ses.trialdata is None:
            continue
        trial = ses.trialdata.copy()
        if "engaged" in trial.columns:
            trial = trial[trial["engaged"] == 1]
        trial["session_id"] = ses.session_id
        trial["protocol"] = ses.protocol
        rows.append(trial)
    if not rows:
        raise SystemExit("No sessions found under 0_data/.")
    engaged_trialdata = pd.concat(rows, ignore_index=True)
    engaged_trialdata.to_pickle(trialdata_cache)


# --------------------------------------------------------------------- #
# 3. Fit each session (where possible) and apply the inclusion criteria
# --------------------------------------------------------------------- #
records = []
fits = {}  # session_id -> PsychometricFit or None

for session_id, trial in engaged_trialdata.groupby("session_id"):
    protocol = trial["protocol"].iloc[0]
    reasons = []

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
        fit = fit_psychometric(trial)
        intermediate = trial.loc[(trial["signal"] > 0) & (trial["signal"] < 100), "signal"].to_numpy()
        if fit is None:
            fit_status = "failed_to_converge"
            reasons.append("psychometric fit failed to converge")
        elif not fit.threshold_in_range(intermediate):
            zmin, zmax = fit.threshold_z_range(intermediate)
            fit_status = "threshold_out_of_range"
            reasons.append(f"threshold (mu={fit.mu:.0f}%) outside tested stimulus range "
                            f"(z-range [{zmin:.2f}, {zmax:.2f}] doesn't bracket 0)")
        else:
            fit_status = "ok"

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
    })

fits_df = pd.DataFrame(records).sort_values(["protocol", "session_id"]).reset_index(drop=True)
fits_df.to_csv(paths.out / "psychometric_fits.csv", index=False)
fits_df.loc[~fits_df["included"], ["session_id", "protocol", "exclude_reasons"]].to_csv(
    paths.out / "excluded_sessions.csv", index=False)

print(f"\nInclusion criteria: d' >= {MIN_DPRIME}, FA <= {MAX_FA_RATE:.0%}, "
      f"engaged fraction >= {MIN_FRAC_ENGAGED:.0%}, threshold within tested range (DN only)")
for protocol in PROTOCOLS:
    sub = fits_df[fits_df["protocol"] == protocol]
    if len(sub):
        print(f"  {protocol}: {int(sub['included'].sum())}/{len(sub)} sessions included")
if (~fits_df["included"]).any():
    print(fits_df.loc[~fits_df["included"], ["session_id", "protocol", "exclude_reasons"]].to_string(index=False))


# --------------------------------------------------------------------- #
# 4. Per-session psychometric curves, one figure per protocol (data +
#    fit where a fit exists; DM sessions show data points only). Excluded
#    sessions get a grey curve/red title instead of being dropped from
#    the figure. Paginated for many sessions.
# --------------------------------------------------------------------- #
included_lookup = fits_df.set_index("session_id")["included"]

for protocol in PROTOCOLS[1:]:
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

    clean_summary = (fits_df[fits_df["included"]]
                      .merge(summary_1a[["animal_id", "sessiondate", "duration_s", "criterion_engaged"]]
                             .reset_index(),
                             on="session_id", how="left"))

    print(f"\nClean dataset: {len(included_ids)}/{len(fits_df)} sessions, "
          f"{len(clean_trialdata)}/{len(engaged_trialdata)} engaged trials kept after session exclusion")

    # 6a. Dataset overview: sessions per animal, engaged trials/session, duration
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    sns.countplot(data=clean_summary, x="protocol", hue="animal_id", ax=axes[0], legend=False,
                  order=PROTOCOLS)
    axes[0].set_title("Sessions per animal (clean)")
    axes[0].set_ylabel("# sessions")

    sns.boxplot(data=clean_summary, x="protocol", y="n_trials_engaged", order=PROTOCOLS, ax=axes[1],
                hue="protocol", palette=PROTOCOL_COLORS, legend=False)
    sns.stripplot(data=clean_summary, x="protocol", y="n_trials_engaged", order=PROTOCOLS, ax=axes[1],
                  color="black", alpha=0.5, size=3)
    axes[1].set_title("Engaged trials per session (clean)")

    sns.boxplot(data=clean_summary, x="protocol", y=clean_summary["duration_s"] / 60, order=PROTOCOLS,
                ax=axes[2], hue="protocol", palette=PROTOCOL_COLORS, legend=False)
    sns.stripplot(data=clean_summary, x="protocol", y=clean_summary["duration_s"] / 60, order=PROTOCOLS,
                  ax=axes[2], color="black", alpha=0.5, size=3)
    axes[2].set_ylabel("duration (min)")
    axes[2].set_title("Session duration (clean)")
    fig.tight_layout()
    save_fig(fig, figdir, "3_dataset_overview_clean")

    # 6b. Trial-outcome distribution (HIT/MISS/FA/CR) per protocol, clean data only
    catch = clean_trialdata.dropna(subset=["trialOutcome"])
    outcome_counts = catch.groupby(["protocol", "trialOutcome"]).size().unstack(fill_value=0)
    pivot = outcome_counts.div(outcome_counts.sum(axis=1), axis=0).reindex(
        index=[p for p in PROTOCOLS if p in outcome_counts.index], columns=["HIT", "CR", "MISS", "FA"])

    fig, ax = plt.subplots(figsize=(5, 3.2))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=[OUTCOME_COLORS[c] for c in pivot.columns])
    ax.set_ylabel("fraction of catch trials")
    ax.set_title("Trial outcomes (clean)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    save_fig(fig, figdir, "4_trial_outcomes_clean")

    # 6c. d' and criterion distribution across the clean, included sessions
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    sns.boxplot(data=clean_summary, x="protocol", y="dprime_engaged", order=PROTOCOLS, ax=axes[0],
                hue="protocol", palette=PROTOCOL_COLORS, legend=False)
    sns.stripplot(data=clean_summary, x="protocol", y="dprime_engaged", order=PROTOCOLS, ax=axes[0],
                  color="black", alpha=0.5, size=4)
    axes[0].axhline(MIN_DPRIME, color="grey", linestyle="--", linewidth=1)
    axes[0].set_ylabel("d' (engaged)")
    axes[0].set_title("d' by protocol (clean)")

    sns.scatterplot(data=clean_summary, x="dprime_engaged", y="criterion_engaged", hue="protocol",
                     hue_order=PROTOCOLS, palette=PROTOCOL_COLORS, ax=axes[1])
    axes[1].axhline(0, color="grey", linewidth=0.8)
    axes[1].axvline(1, color="grey", linewidth=0.8)
    axes[1].set_title("Bias vs. sensitivity (clean)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, figdir, "5_dprime_criterion_clean")

    # 6d. Signal-strength trial counts per protocol, clean data only
    fig, axes = plt.subplots(1, len(PROTOCOLS), figsize=(11, 3), sharey=False)
    for ax, protocol in zip(axes, PROTOCOLS):
        subset = clean_trialdata[clean_trialdata["protocol"] == protocol]
        if subset.empty:
            ax.set_visible(False)
            continue
        counts = subset.groupby(["session_id", "signal"]).size().reset_index(name="n_trials")
        sns.boxplot(data=counts, x="signal", y="n_trials", ax=ax, color=PROTOCOL_COLORS[protocol])
        sns.stripplot(data=counts, x="signal", y="n_trials", ax=ax, color="black", alpha=0.4, size=3)
        ax.set_title(protocol)
        ax.set_xlabel("signal strength (%)")
    fig.suptitle("Trials per signal level, per session (clean)", y=1.03)
    fig.tight_layout()
    save_fig(fig, figdir, "6_signal_level_balance_clean")

print(f"\nWrote tables to {paths.out}, figures to {figdir}")
print("Run `python b_progress/make_progress_md.py 1b_psychometric` to build a markdown summary.")
