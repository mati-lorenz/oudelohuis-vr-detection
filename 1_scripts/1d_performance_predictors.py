# -*- coding: utf-8 -*-
"""
1d_performance_predictors
==========================
Relationships between the animal's trial-by-trial PERFORMANCE and
behavioral variables -- both from the CURRENT trial and from the
PREVIOUS trial (classic "history effects": does yesterday's/last
trial's state or outcome predict today's/this trial's accuracy?).
Across all three protocols (DM, DP, DN).

Performance is CORRECTNESS (0/1), generalized to EVERY signal level, not
just the 0%/100% catch trials: "any signal present -> should lick,
no signal -> should withhold" is the task rule at every signal strength
in a detection task, so
    correct = (signal > 0 AND lickResponse == 1) OR (signal == 0 AND lickResponse == 0)
applies uniformly. (This subsumes the old catch-trial-only HIT/CR=1,
MISS/FA=0 definition -- those are exactly the signal==100/signal==0
special cases of the same rule.)

Stimulus strength itself is included as a predictor -- "stim", the
signal level binned into 4 groups. Where a psychometric fit exists (DN
only, from 1b_psychometric), stim uses the NORMALIZED/z-scored signal
(signal_psy = (signal-mu)/sigma, the same convention used in
1c_behavior.py's trial-aligned plots) so bins are comparable across
sessions with different thresholds; where no fit exists (DM, DP), it
falls back to the raw signal value. DM only ever has 2 signal levels
(0%/100%), so its "4 bins" naturally collapse to 2.

Behavioral predictors (speed, pupil area, motion energy, lick rate) are
taken from the PRE-STIMULUS window only ([-30cm, -10cm] relative to
stimStart, matching 1c_behavior.py's MI_WINDOWS convention) -- using a
window from AFTER stimulus onset would be circular, since the response
that determines "correct" happens in that window. This restriction only
matters for the CURRENT trial's predictors; the previous trial's
predictors (whichever window they're drawn from) are unambiguously
"past" relative to the current trial's outcome regardless.

Reads: 2_pipeline/1b_psychometric/out/psychometric_fits.csv for the
included-session list AND each session's fitted mu/sigma (Rule #1).
Loads: raw behaviordata/videodata for those sessions (1b/1c never
cached the merged continuous trace, and this step needs its own
pre-stimulus window extraction anyway).

What this does
--------------
1. Builds one row per (session, engaged trial): current-trial pre-stim
   behavioral state + binned stimulus strength, the SAME variables from
   the previous trial, the previous trial's outcome/choice, and this
   trial's correctness (now defined at every signal level -- see above).
2. Exploratory plots: accuracy by protocol, accuracy conditioned on the
   previous trial's outcome/choice (post-error/post-correct, win-stay/
   lose-shift), accuracy as a function of each behavioral predictor AND
   of stimulus strength (current AND previous trial, quantile-binned
   "tuning curves" -- the stim one is just the psychometric curve, as a
   sanity check that the generalized correctness rule behaves as
   expected), and a history kernel (how far back does previous-trial
   correctness predict current accuracy).
3. For every predictor: mutual information (histogram/quantile-based --
   appropriate here since the target, correctness, is already binary,
   unlike the continuous-vs-continuous case in 1c_behavior.py where the
   KSG estimator was used instead) AND a linear fit (r^2, a linear-
   probability-model since the target is 0/1). Compares the MI implied
   by r^2 against the actual (corrected) MI, same logic as 1c_behavior.py.

Parallelism: both the per-session feature extraction and the MI
computation loop are parallelized with joblib (matching the project's
existing convention, e.g. the uploaded plot_mi_behavior.py/
plot_temporal_information.py) -- see N_JOBS_SESSIONS/N_JOBS_MI below.

Outputs
-------
2_pipeline/1d_performance_predictors/
    out/    performance_mi_vs_linear.csv   one row per (protocol,
                                            predictor): r, r2, MI
                                            raw/corrected, MI implied by
                                            r2, whether linear looks
                                            sufficient
            performance_multivariate_mi_vs_linear.csv   one row per
                                            protocol: the same
                                            comparison but for ALL
                                            predictors fit jointly (see
                                            section 4b) -- includes
                                            adj_r2
            figures/*.png
    store/  trial_performance.pkl          per-(session, engaged trial)
                                            table: current/previous
                                            behavioral features + binned
                                            stim, previous outcome/choice,
                                            correctness (reused unless
                                            --recompute)
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed

from infotheory.pipeline import get_pipeline_paths
from infotheory.session import load_sessions, PROTOCOLS
from infotheory.continuous import merge_behavior_video, restrict_to_engaged, remove_pupil_outliers, \
    compute_trial_position_window_means
from infotheory.info_theory import mutual_information_shuffle, mutual_information_shuffle_ksg, \
    compare_mi_to_linear, linear_fit, multivariate_linear_fit, robust_qcut
from infotheory.plotting import set_style, save_fig, clear_figures, PROTOCOL_COLORS

BEHAVIOR_VARS = ["runspeed", "pupil_area", "motionenergy", "lick"]
RATE_VARIABLES = ["lick"]  # 0/1 event channel -- binned as counts/width (a rate), never averaged
VAR_LABELS = {"runspeed": "speed (cm/s)", "pupil_area": "pupil area (a.u.)",
              "motionenergy": "motion energy (a.u.)", "lick": "lick rate", "stim": "stimulus strength"}
PRESTIM_WINDOW = (-30.0, -10.0)  # cm relative to stimStart -- BEFORE the stimulus (see module
                                  # docstring for why this matters), matches 1c_behavior.py's
                                  # MI_WINDOWS pre_stim entry
N_STIM_BINS = 4          # bin count for the normalized-stimulus predictor (see module docstring)
N_QUANTILE_BINS = 10      # exploratory "tuning curve" plots: accuracy per quantile bin of a predictor
MAX_LAG_KERNEL = 5       # history-kernel plot: how many trials back to check previous-correctness
MI_BINS = 20             # histogram MI is appropriate here since the target (correctness) is
                          # already binary -- see module docstring for why this differs from
                          # 1c_behavior.py's KSG choice for continuous-vs-continuous pairs
N_SHUFFLES = 100
KSG_K = 5                # neighbors for the multivariate KSG estimator (section 4b) -- see
                          # info_theory.py; not used for the pairwise histogram MI above
N_SHUFFLES_MULTIVARIATE = 20  # fewer than N_SHUFFLES: each multivariate KSG call costs more
                               # (higher-dimensional joint space), and there's only one model
                               # per protocol here rather than one per predictor
LINEAR_SUFFICIENT_FRAC = 0.8

# Parallelism (joblib, matching the project's convention -- see e.g. the
# uploaded plot_mi_behavior.py/plot_temporal_information.py): -1 = use
# all available cores. Session feature-extraction is embarrassingly
# parallel; the MI computation loop is parallelized separately,
# coarse-grained over protocol -- see section 4.
N_JOBS_SESSIONS = -1
N_JOBS_MI = -1
JOBLIB_VERBOSE = 5

parser = argparse.ArgumentParser()
parser.add_argument("--recompute", action="store_true",
                     help="Reload sessions from 0_data/ instead of reusing the cached "
                          "store/trial_performance.pkl")
args = parser.parse_args()

paths = get_pipeline_paths(__file__)
figdir = paths.out / "figures"
clear_figures(figdir)
set_style()
rng = np.random.default_rng(0)


# --------------------------------------------------------------------- #
# 1. Included sessions from 1b, plus each session's fitted mu/sigma
#    (only ever non-NaN for DN -- see module docstring)
# --------------------------------------------------------------------- #
fits_path = paths.out_from("1b_psychometric") / "psychometric_fits.csv"
if not fits_path.exists():
    raise SystemExit(f"Run 1b_psychometric.py first -- expected {fits_path}")

fits_df = pd.read_csv(fits_path)
included = fits_df[fits_df["included"]]
included_ids = included["session_id"].tolist()
if not included_ids:
    raise SystemExit("No sessions passed 1b_psychometric's inclusion criteria -- nothing to analyze.")
print(f"Using {len(included_ids)} included sessions from 1b_psychometric "
      f"({included['protocol'].value_counts().to_dict()})")

mu_sigma_lookup = fits_df.set_index("session_id")[["mu", "sigma"]].to_dict("index")


# --------------------------------------------------------------------- #
# 2. Build the per-trial (current + previous trial features, outcome)
#    table -- or reuse the cache. Parallelized across sessions.
# --------------------------------------------------------------------- #
trial_performance_cache = paths.store / "trial_performance.pkl"

# Columns the rest of this script expects in trial_performance -- checked
# after loading so a cache built by an OLDER version of this script
# (e.g. before cur_stim/prev_stim existed) is treated as a miss and
# recomputed, rather than crashing with a bare KeyError once section 2b
# tries to bin a column that was never cached. Extend this whenever the
# cached schema changes.
REQUIRED_TRIAL_PERFORMANCE_COLUMNS = ["cur_stim", "prev_stim", "correct", "session_id", "protocol"]

trial_performance = None
if not args.recompute and trial_performance_cache.exists():
    try:
        candidate = pd.read_pickle(trial_performance_cache)
        missing = [c for c in REQUIRED_TRIAL_PERFORMANCE_COLUMNS if c not in candidate.columns]
        if missing:
            print(f"Cached trial-performance table is missing columns {missing} -- looks like it was "
                  f"built by an older version of this script. Recomputing from 0_data/ instead of using "
                  f"{trial_performance_cache} (safe to delete that file to silence this check in the future).")
        else:
            trial_performance = candidate
            print(f"Reusing cached trial-performance table from {paths.store} (pass --recompute to reload 0_data/)")
    except Exception as e:
        print(f"Could not load cached trial-performance table ({e}); recomputing from 0_data/ ...")

if trial_performance is None:
    sessions = load_sessions(protocols=PROTOCOLS, load_behaviordata=True, load_videodata=True,
                              only_session_ids=included_ids)

    def process_session(ses, mu_sigma_lookup):
        """All per-session work for one session, returned as a trial-
        level DataFrame (or None if the session had no usable data).
        Self-contained (no shared outer-scope state) so it can be
        dispatched via joblib.Parallel across sessions."""
        if ses.trialdata is None or ses.behaviordata is None:
            return None
        merged_full = merge_behavior_video(ses.behaviordata, ses.videodata)
        merged_full = remove_pupil_outliers(merged_full)
        if merged_full.empty:
            return None
        merged = restrict_to_engaged(merged_full, ses.trialdata)
        if merged.empty:
            return None

        available_vars = [v for v in BEHAVIOR_VARS if v in merged.columns]
        available_rate_vars = [v for v in RATE_VARIABLES if v in available_vars]

        engaged_trials = (ses.trialdata[ses.trialdata["engaged"] == 1]
                           if "engaged" in ses.trialdata.columns else ses.trialdata.copy())

        # Generalized correctness -- see module docstring: "any signal
        # present -> should lick" applies at every signal strength, not
        # just the 0%/100% catch levels.
        engaged_trials = engaged_trials.assign(
            correct=((engaged_trials["signal"] > 0) & (engaged_trials["lickResponse"] == 1))
            | ((engaged_trials["signal"] == 0) & (engaged_trials["lickResponse"] == 0)))
        engaged_trials["correct"] = engaged_trials["correct"].astype(int)

        # Normalized stimulus strength: z-scored via this session's own
        # fitted mu/sigma where a fit exists (DN), else the raw signal
        # value (DM/DP) -- binned into groups downstream (pooled across
        # a protocol's sessions, so binning happens after concatenation,
        # not here).
        mu_sigma = mu_sigma_lookup.get(ses.session_id, {})
        if mu_sigma and np.isfinite(mu_sigma.get("mu", np.nan)) and np.isfinite(mu_sigma.get("sigma", np.nan)):
            engaged_trials["stim"] = (engaged_trials["signal"] - mu_sigma["mu"]) / mu_sigma["sigma"]
        else:
            engaged_trials["stim"] = engaged_trials["signal"].astype(float)

        pre = compute_trial_position_window_means(engaged_trials, merged, columns=available_vars,
                                                    s_start=PRESTIM_WINDOW[0], s_end=PRESTIM_WINDOW[1],
                                                    rate_columns=available_rate_vars)
        pre = pre.rename(columns={v: f"cur_{v}" for v in available_vars})

        trial = engaged_trials[["trialNumber", "signal", "stim", "lickResponse", "correct"]].merge(
            pre, on="trialNumber", how="left")
        trial = trial.rename(columns={"stim": "cur_stim"})
        trial = trial.sort_values("trialNumber").reset_index(drop=True)

        # Previous-(analyzed, i.e. engaged)-trial features: the same
        # current-trial behavioral + stim columns shifted by one row,
        # plus the previous trial's own outcome and choice -- the
        # classic "history effect" predictors. A gap in trialNumber (a
        # disengaged trial in between) is NOT specially handled: "prev"
        # here means the previous ENGAGED trial actually analyzed, not
        # strictly trialNumber-1.
        lag_source_cols = [f"cur_{v}" for v in available_vars] + ["cur_stim", "correct", "lickResponse"]
        for col in lag_source_cols:
            out_name = col.replace("cur_", "prev_", 1) if col.startswith("cur_") else f"prev_{col}"
            trial[out_name] = trial[col].shift(1)

        for lag in range(1, MAX_LAG_KERNEL + 1):
            trial[f"correct_lag{lag}"] = trial["correct"].shift(lag)

        trial["session_id"] = ses.session_id
        trial["protocol"] = ses.protocol
        return trial

    print(f"Processing {len(sessions)} sessions (n_jobs={N_JOBS_SESSIONS}) ...")
    session_results = Parallel(n_jobs=N_JOBS_SESSIONS, backend="loky", verbose=JOBLIB_VERBOSE)(
        delayed(process_session)(ses, mu_sigma_lookup) for ses in sessions
    )
    session_rows = [r for r in session_results if r is not None]

    if not session_rows:
        raise SystemExit("No sessions had usable behaviordata -- nothing to analyze.")
    trial_performance = pd.concat(session_rows, ignore_index=True)
    trial_performance.to_pickle(trial_performance_cache)

print(f"Trials with a defined outcome: {len(trial_performance)} across "
      f"{trial_performance['session_id'].nunique()} sessions")


# --------------------------------------------------------------------- #
# 2b. Bin cur_stim/prev_stim into N_STIM_BINS groups, per protocol
#    (pooling across that protocol's sessions -- signal_psy is already
#    comparable across sessions for DN since it's z-scored per-session;
#    raw signal for DM/DP is on the same 0-100 scale for every session
#    of a given protocol already). Stored as the bin's numeric midpoint
#    (cur_stim_bin/prev_stim_bin) so it's usable as an ordinary numeric
#    predictor in the MI/linear analysis below, same as any other.
# --------------------------------------------------------------------- #
for col, out_col in [("cur_stim", "cur_stim_bin"), ("prev_stim", "prev_stim_bin")]:
    trial_performance[out_col] = np.nan
    for protocol in PROTOCOLS:
        mask = trial_performance["protocol"] == protocol
        vals = trial_performance.loc[mask, col]
        if vals.notna().sum() < N_STIM_BINS:
            continue
        try:
            binned = robust_qcut(vals, N_STIM_BINS)
        except ValueError:
            continue
        trial_performance.loc[mask, out_col] = binned.apply(
            lambda iv: round(iv.mid, 6) if pd.notna(iv) else np.nan)


# --------------------------------------------------------------------- #
# 3. Exploratory plots
# --------------------------------------------------------------------- #
current_vars = [v for v in BEHAVIOR_VARS if f"cur_{v}" in trial_performance.columns]
previous_vars = [v for v in BEHAVIOR_VARS if f"prev_{v}" in trial_performance.columns]

# 3a. Overview: accuracy by protocol, by previous outcome, by previous choice
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

sns.barplot(data=trial_performance, x="protocol", y="correct", order=PROTOCOLS, ax=axes[0],
            hue="protocol", palette=PROTOCOL_COLORS, legend=False, errorbar="se")
axes[0].set_ylabel("accuracy (fraction correct)")
axes[0].set_title("Accuracy by protocol")
axes[0].set_ylim(0, 1)

prev_outcome_df = trial_performance.dropna(subset=["prev_correct"]).assign(
    prev_outcome=lambda d: d["prev_correct"].map({1: "prev correct", 0: "prev error"}))
sns.barplot(data=prev_outcome_df, x="prev_outcome", y="correct", hue="protocol",
            hue_order=PROTOCOLS, palette=PROTOCOL_COLORS, ax=axes[1], errorbar="se")
axes[1].set_ylabel("accuracy")
axes[1].set_xlabel("")
axes[1].set_title("Post-error vs. post-correct")
axes[1].set_ylim(0, 1)
axes[1].legend(fontsize=7, frameon=False)

prev_choice_df = trial_performance.dropna(subset=["prev_lickResponse"]).assign(
    prev_choice=lambda d: d["prev_lickResponse"].map({1: "prev licked", 0: "prev no lick"}))
sns.barplot(data=prev_choice_df, x="prev_choice", y="correct", hue="protocol",
            hue_order=PROTOCOLS, palette=PROTOCOL_COLORS, ax=axes[2], errorbar="se")
axes[2].set_ylabel("accuracy")
axes[2].set_xlabel("")
axes[2].set_title("Accuracy by previous choice")
axes[2].set_ylim(0, 1)
axes[2].legend(fontsize=7, frameon=False)

fig.suptitle("Performance overview (all trials, correctness generalized to every signal level)", y=1.03)
fig.tight_layout()
save_fig(fig, figdir, "1_performance_overview")


def _plot_tuning_grid(df: pd.DataFrame, columns: list[tuple[str, str, bool]], title: str, filename: str):
    """Accuracy vs. quantile-binned predictor, one panel per (label,
    column) entry (protocol overlaid) -- the "does this predictor track
    performance" exploratory view. `columns` is a list of
    (label, column_name, already_binned) tuples; already_binned=True
    (used for cur_stim_bin/prev_stim_bin, pre-binned in section 2b)
    groups directly on the existing values instead of re-binning."""
    if not columns:
        return
    fig, axes = plt.subplots(1, len(columns), figsize=(4.2 * len(columns), 3.5), squeeze=False)
    axes = axes[0]
    for ax, (label, col, already_binned) in zip(axes, columns):
        sub = df.dropna(subset=[col, "correct"])
        for protocol in PROTOCOLS:
            psub = sub[sub["protocol"] == protocol]
            if len(psub) < N_QUANTILE_BINS * 2:
                continue
            if already_binned:
                agg = psub.groupby(col)["correct"].mean()
                sem = psub.groupby(col)["correct"].sem()
                x = agg.index.to_numpy()
            else:
                try:
                    bins = robust_qcut(psub[col], N_QUANTILE_BINS)
                except ValueError:
                    continue
                agg = psub.groupby(bins, observed=True)["correct"].mean()
                sem = psub.groupby(bins, observed=True)["correct"].sem()
                x = [interval.mid for interval in agg.index]
            ax.errorbar(x, agg.to_numpy(), yerr=sem.fillna(0).to_numpy(), color=PROTOCOL_COLORS[protocol],
                        marker="o", markersize=4, capsize=3, label=protocol)
        ax.set_xlabel(label)
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(title, y=1.03)
    fig.tight_layout()
    save_fig(fig, figdir, filename)


# 3b/3c. Accuracy vs. quantile-binned behavioral state (+ binned stim),
# current and previous trial.
current_columns = [(VAR_LABELS[v], f"cur_{v}", False) for v in current_vars]
current_columns.append((VAR_LABELS["stim"], "cur_stim_bin", True))
previous_columns = [(VAR_LABELS[v], f"prev_{v}", False) for v in previous_vars]
previous_columns.append((VAR_LABELS["stim"], "prev_stim_bin", True))

_plot_tuning_grid(trial_performance, current_columns,
                   "Accuracy vs. CURRENT-trial pre-stimulus state & stimulus strength",
                   "2_tuning_current_trial")
_plot_tuning_grid(trial_performance, previous_columns,
                   "Accuracy vs. PREVIOUS-trial pre-stimulus state & stimulus strength",
                   "3_tuning_previous_trial")

# 3d. History kernel: how far back does previous correctness predict
# current accuracy?
lag_cols = [f"correct_lag{lag}" for lag in range(1, MAX_LAG_KERNEL + 1)]
if all(c in trial_performance.columns for c in lag_cols):
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for protocol in PROTOCOLS:
        psub = trial_performance[trial_performance["protocol"] == protocol]
        means, sems = [], []
        for lag in range(1, MAX_LAG_KERNEL + 1):
            sub = psub.dropna(subset=[f"correct_lag{lag}", "correct"])
            same = sub[sub[f"correct_lag{lag}"] == 1]["correct"]
            diff = sub[sub[f"correct_lag{lag}"] == 0]["correct"]
            if len(same) < 5 or len(diff) < 5:
                means.append(np.nan)
                sems.append(np.nan)
                continue
            delta = same.mean() - diff.mean()
            se = np.sqrt(same.var(ddof=1) / len(same) + diff.var(ddof=1) / len(diff))
            means.append(delta)
            sems.append(se)
        ax.errorbar(range(1, MAX_LAG_KERNEL + 1), means, yerr=sems, color=PROTOCOL_COLORS[protocol],
                    marker="o", markersize=4, capsize=3, label=protocol)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("trials back")
    ax.set_ylabel("accuracy(prev correct) - accuracy(prev error)")
    ax.set_title("History kernel: how far back does correctness matter?")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save_fig(fig, figdir, "4_history_kernel")


# --------------------------------------------------------------------- #
# 4. Mutual information vs. linear regression, per predictor, per
#    protocol -- parallelized across protocols (coarse-grained: each
#    task computes every predictor for one protocol sequentially).
# --------------------------------------------------------------------- #
predictors = (["cur_stim_bin", "prev_stim_bin"] + [f"cur_{v}" for v in current_vars]
              + [f"prev_{v}" for v in previous_vars] + ["prev_correct", "prev_lickResponse"])
predictors = [p for p in predictors if p in trial_performance.columns]


def compute_mi_for_protocol(protocol, sub, predictors, mi_bins, n_shuffles, seed):
    rng_local = np.random.default_rng(seed)
    records = []
    for predictor in predictors:
        pair_data = sub[[predictor, "correct"]].dropna()
        if len(pair_data) < 10:
            continue
        mi = mutual_information_shuffle(pair_data[predictor], pair_data["correct"], bins=mi_bins,
                                         n_shuffles=n_shuffles, rng=rng_local)
        lin = linear_fit(pair_data[predictor], pair_data["correct"])
        cmp = compare_mi_to_linear(mi.bits_corrected, lin.r2)
        records.append({
            "protocol": protocol, "predictor": predictor, "n_trials": mi.n,
            "r": lin.r, "r2": lin.r2, "p_linear": lin.p_value,
            "mi_bits_raw": mi.bits_raw, "mi_bits_shuffle_mean": mi.bits_shuffle_mean,
            "mi_bits_corrected": mi.bits_corrected, "mi_p_value": mi.p_value,
            "mi_bits_from_r2": cmp.mi_from_r2_bits, "mi_reference_bits": cmp.mi_reference_bits,
            "frac_explained_linearly": cmp.frac_explained_linearly,
            "linear_sufficient": (cmp.frac_explained_linearly >= LINEAR_SUFFICIENT_FRAC)
            if not np.isnan(cmp.frac_explained_linearly) else None,
        })
    return records


protocol_groups = [(protocol, trial_performance[trial_performance["protocol"] == protocol])
                    for protocol in PROTOCOLS]
protocol_groups = [(p, sub) for p, sub in protocol_groups if not sub.empty]

print(f"\nComputing MI/linear fits for {len(protocol_groups)} protocols x {len(predictors)} "
      f"predictors (n_jobs={N_JOBS_MI}) ...")
nested_records = Parallel(n_jobs=N_JOBS_MI, backend="loky", verbose=JOBLIB_VERBOSE)(
    delayed(compute_mi_for_protocol)(protocol, sub, predictors, MI_BINS, N_SHUFFLES, seed=i)
    for i, (protocol, sub) in enumerate(protocol_groups)
)
records = [rec for group in nested_records for rec in group]

mi_table = pd.DataFrame(records)
mi_table.to_csv(paths.out / "performance_mi_vs_linear.csv", index=False)

if len(mi_table):
    mi_table_sorted = mi_table.sort_values("mi_bits_corrected", ascending=False)
    print("\nStrongest predictors of performance (by bias-corrected MI, top 10):")
    print(mi_table_sorted[["protocol", "predictor", "n_trials", "r2", "mi_bits_corrected", "mi_p_value"]]
          .head(10).to_string(index=False))

    n_insufficient = (mi_table["linear_sufficient"] == False).sum()  # noqa: E712 (None must not match)
    print(f"\n{n_insufficient}/{len(mi_table)} (protocol, predictor) combinations show a linear fit "
          f"missing >{100 * (1 - LINEAR_SUFFICIENT_FRAC):.0f}% of the (bias-corrected) mutual information:")
    if n_insufficient:
        print(mi_table.loc[mi_table["linear_sufficient"] == False,  # noqa: E712
                            ["protocol", "predictor", "r2", "mi_bits_corrected", "mi_bits_from_r2"]]
              .to_string(index=False))


# --------------------------------------------------------------------- #
# 4b. MULTIVARIATE regression + joint MI: not just pairwise -- one
#    model per protocol using EVERY predictor at once (correct ~ all
#    current/previous behavioral state + stim + history predictors
#    jointly), with the joint mutual information between that whole
#    predictor SET and correctness for comparison. A predictor can be
#    only weakly related to performance pairwise while the full set
#    together is much more predictive (or the reverse: mostly redundant
#    predictors that don't add much beyond the single best one) --
#    adj_r2 (not just r2) is reported specifically to catch the latter,
#    since r2 alone mechanically increases with every added predictor.
#    Uses KSG (not the histogram estimator the pairwise case above
#    uses) since multivariate histogram binning is impractical (curse
#    of dimensionality) -- KSG handles a binary target fine as one
#    dimension of the joint space, same as the pairwise KSG comparisons
#    in 1c_behavior.py.
# --------------------------------------------------------------------- #
def compute_multivariate_for_protocol(protocol, sub, predictors, ksg_k, n_shuffles, seed):
    if len(predictors) < 2:
        return []
    data = sub[predictors + ["correct"]].dropna()
    if len(data) < len(predictors) + 10:
        return []
    X = data[predictors].to_numpy()
    y = data["correct"].to_numpy()
    lin = multivariate_linear_fit(X, y, predictor_names=predictors)
    mi = mutual_information_shuffle_ksg(X, y, k=ksg_k, n_shuffles=n_shuffles,
                                         rng=np.random.default_rng(seed))
    cmp = compare_mi_to_linear(mi.bits_corrected, lin.r2)
    return [{
        "protocol": protocol, "predictors": ", ".join(predictors), "n_predictors": len(predictors),
        "n_trials": lin.n, "r2": lin.r2, "adj_r2": lin.adj_r2, "p_linear": lin.p_value,
        "mi_bits_raw": mi.bits_raw, "mi_bits_shuffle_mean": mi.bits_shuffle_mean,
        "mi_bits_corrected": mi.bits_corrected, "mi_p_value": mi.p_value,
        "mi_bits_from_r2": cmp.mi_from_r2_bits, "mi_reference_bits": cmp.mi_reference_bits,
        "frac_explained_linearly": cmp.frac_explained_linearly,
        "linear_sufficient": (cmp.frac_explained_linearly >= LINEAR_SUFFICIENT_FRAC)
        if not np.isnan(cmp.frac_explained_linearly) else None,
    }]


print(f"\nComputing multivariate (all predictors at once) MI/linear fit for {len(protocol_groups)} "
      f"protocols (n_jobs={N_JOBS_MI}) ...")
nested_multivariate = Parallel(n_jobs=N_JOBS_MI, backend="loky", verbose=JOBLIB_VERBOSE)(
    delayed(compute_multivariate_for_protocol)(protocol, sub, predictors, KSG_K, N_SHUFFLES_MULTIVARIATE,
                                                 seed=1000 + i)
    for i, (protocol, sub) in enumerate(protocol_groups)
)
multivariate_records = [rec for group in nested_multivariate for rec in group]
multivariate_table = pd.DataFrame(multivariate_records)
multivariate_table.to_csv(paths.out / "performance_multivariate_mi_vs_linear.csv", index=False)

if len(multivariate_table):
    print("\nMultivariate fit of performance from ALL predictors jointly, per protocol:")
    print(multivariate_table[["protocol", "n_trials", "r2", "adj_r2", "mi_bits_corrected", "mi_p_value"]]
          .to_string(index=False))


# --------------------------------------------------------------------- #
# 5. Figure: MI (best estimate) vs. MI-implied-by-r2, one point per
#    (protocol, predictor) -- color encodes the predictor (no shape
#    dimension needed here, unlike 1c_behavior.py's pairwise comparison,
#    since every point already shares the same "y" variable: performance).
# --------------------------------------------------------------------- #
if len(mi_table) and np.isfinite(mi_table["mi_reference_bits"]).any():
    finite = mi_table.replace([np.inf, -np.inf], np.nan).dropna(subset=["mi_reference_bits", "mi_bits_from_r2"])
    lims = [0, finite["mi_reference_bits"].max() * 1.1 + 1e-6] if len(finite) else [0, 1]

    color_palette = list(plt.cm.tab20(np.linspace(0, 1, 20)))
    predictor_color = {p: color_palette[i % len(color_palette)] for i, p in enumerate(predictors)}

    protocols_present = [p for p in PROTOCOLS if p in finite["protocol"].unique()]
    fig, axes = plt.subplots(1, len(protocols_present), figsize=(4.3 * len(protocols_present), 4.3), squeeze=False)
    axes = axes[0]
    handles = {}
    for ax, protocol in zip(axes, protocols_present):
        sub = finite[finite["protocol"] == protocol]
        for _, row in sub.iterrows():
            h = ax.scatter(row["mi_reference_bits"], row["mi_bits_from_r2"],
                            color=predictor_color[row["predictor"]], s=70, edgecolor="black", linewidth=0.4)
            handles.setdefault(row["predictor"], h)
        ax.plot(lims, lims, color="grey", linestyle="--", linewidth=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title(protocol)
        ax.set_xlabel("MI, best available estimate (bits)")
    axes[0].set_ylabel("MI implied by linear r\u00b2 (bits)")
    fig.suptitle("Does a linear model capture what predicts performance?\n"
                 "(dashed = linear matches MI exactly)", y=1.06)
    fig.legend(handles.values(), handles.keys(), loc="center left", bbox_to_anchor=(1.0, 0.5),
               fontsize=7, title="predictor", frameon=False)
    fig.tight_layout()
    save_fig(fig, figdir, "5_mi_vs_linear_summary")

# 5a. Same comparison, but as the FRACTION explained linearly directly
# (frac_explained_linearly = mi_from_r2 / max(mi_corrected, mi_from_r2),
# already in [0,1] by construction -- see compare_mi_to_linear) rather
# than reading it off the scatter's distance from the diagonal. One bar
# per predictor, sorted, one panel per protocol.
if len(mi_table):
    finite_frac = mi_table.dropna(subset=["frac_explained_linearly"])
    if len(finite_frac):
        protocols_present = [p for p in PROTOCOLS if p in finite_frac["protocol"].unique()]
        fig, axes = plt.subplots(1, len(protocols_present), figsize=(4.5 * len(protocols_present), 4.5), squeeze=False)
        for ax, protocol in zip(axes[0], protocols_present):
            sub = finite_frac[finite_frac["protocol"] == protocol].sort_values("frac_explained_linearly")
            colors = [predictor_color[p] for p in sub["predictor"]]
            ax.barh(sub["predictor"], sub["frac_explained_linearly"], color=colors, edgecolor="black", linewidth=0.5)
            ax.axvline(1.0, color="grey", linestyle="--", linewidth=1)
            ax.set_xlim(0, max(1.05, sub["frac_explained_linearly"].max() * 1.05))
            ax.set_xlabel("MI(linear) / MI(full)")
            ax.set_title(protocol)
            ax.tick_params(axis="y", labelsize=7)
        fig.suptitle("Fraction of mutual information a linear fit actually captures, per predictor", y=1.02)
        fig.tight_layout()
        save_fig(fig, figdir, "5a_frac_explained_linearly")


# --------------------------------------------------------------------- #
# 5b. Figure: multivariate MI vs. multivariate r2, one point per
#    protocol -- the multivariate analogue of the figure above. A point
#    below the diagonal means the joint predictor set has a real
#    nonlinear/interaction structure a purely linear-probability model
#    of correctness is missing.
# --------------------------------------------------------------------- #
if len(multivariate_table) and np.isfinite(multivariate_table["mi_reference_bits"]).any():
    finite_mv = multivariate_table.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["mi_reference_bits", "mi_bits_from_r2"])
    if len(finite_mv):
        lims_mv = [0, finite_mv["mi_reference_bits"].max() * 1.1 + 1e-6]
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        for _, row in finite_mv.iterrows():
            ax.scatter(row["mi_reference_bits"], row["mi_bits_from_r2"],
                       color=PROTOCOL_COLORS[row["protocol"]], s=90, edgecolor="black", linewidth=0.5,
                       label=row["protocol"])
        ax.plot(lims_mv, lims_mv, color="grey", linestyle="--", linewidth=1)
        ax.set_xlim(lims_mv)
        ax.set_ylim(lims_mv)
        ax.set_xlabel("joint MI, best available estimate (bits)")
        ax.set_ylabel("MI implied by multivariate r\u00b2 (bits)")
        ax.set_title("Multivariate: all predictors jointly vs. performance\n(dashed = linear matches MI exactly)")
        ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        save_fig(fig, figdir, "6_multivariate_mi_vs_linear_summary")

# 6a. Same, as a fraction: one bar per protocol.
if len(multivariate_table):
    finite_mv_frac = multivariate_table.dropna(subset=["frac_explained_linearly"])
    if len(finite_mv_frac):
        fig, ax = plt.subplots(figsize=(4, 4))
        colors = [PROTOCOL_COLORS[p] for p in finite_mv_frac["protocol"]]
        ax.bar(finite_mv_frac["protocol"], finite_mv_frac["frac_explained_linearly"],
               color=colors, edgecolor="black", linewidth=0.5)
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)
        ax.set_ylim(0, max(1.05, finite_mv_frac["frac_explained_linearly"].max() * 1.05))
        ax.set_ylabel("MI(linear) / MI(full)")
        ax.set_title("Fraction of the joint MI a\nmultivariate linear fit captures")
        fig.tight_layout()
        save_fig(fig, figdir, "6a_multivariate_frac_explained_linearly")

print(f"\nWrote tables to {paths.out}, figures to {figdir}")
print("Run `python b_progress/make_progress_md.py 1d_performance_predictors` to build a markdown summary.")
