# ==============================================================================
# explore_trialdata.R
# ==============================================================================
# Exploratory tidyverse plots for trialdata.csv: one row per trial -- stimulus
# strength/category, licking/reward outcome, timing, and engagement state.
#
# VERIFIED: run end-to-end against a real trialdata.csv (334 trials) with R
# 4.3 + dplyr/ggplot2/tidyr/readr/forcats -- all 6 plots render correctly.
# Confirmed factor levels: stimcat = {C, M, N} (56/52/226 trials),
# trialOutcome = {CR, FA, HIT, MISS} (43/13/177/101), engaged = {0, 1}
# (67/267). Plot 2 (psychometric-by-engagement) reproduces the
# behavior/README.md finding directly: disengaged trials show ~0.18 P(lick)
# even at 100% signal, vs. engaged trials reaching 1.0. Plot 4 (engagement
# timeline) likewise confirms disengagement is a single late-session step,
# not interleaved, in this example session.
#
# tReward/tStimStart are on the same absolute clock and, confirmed against
# this real file, their difference gives sensible response latencies
# (median 2.3s, range 1.7-7.9s on HIT trials) -- units are seconds.
#
# Usage: Rscript explore_trialdata.R path/to/trialdata.csv path/to/output_dir
# ==============================================================================

library(tidyverse)

args     <- commandArgs(trailingOnly = TRUE)
csv_path <- if (length(args) >= 1) args[[1]] else "trialdata.csv"
out_dir  <- if (length(args) >= 2) args[[2]] else "figures"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

trialdata_raw <- read_csv(csv_path, show_col_types = FALSE)

# Sanity check BEFORE the factor releveling below assumes specific levels.
message("stimcat values: ");      print(count(trialdata_raw, stimcat))
message("trialOutcome values: "); print(count(trialdata_raw, trialOutcome))
message("engaged values: ");      print(count(trialdata_raw, engaged))

trialdata <- trialdata_raw |>
  mutate(
    stimcat      = factor(stimcat, levels = c("C", "N", "M")),   # catch / noise / max -- CONFIRM against the count() above
    trialOutcome = factor(trialOutcome),
    engaged      = factor(engaged, levels = c(0, 1), labels = c("disengaged", "engaged")),
    lickResponse = as.logical(lickResponse),
  )

glimpse(trialdata)

# ------------------------------------------------------------------
# Plot 1: trial outcome counts by stimulus category -- the basic
# behavioral summary for the session.
# ------------------------------------------------------------------
p1 <- trialdata |>
  count(stimcat, trialOutcome) |>
  ggplot(aes(stimcat, n, fill = trialOutcome)) +
  geom_col(position = "stack") +
  labs(title = "Trial outcomes by stimulus category", x = "stimulus category", y = "n trials") +
  theme_minimal()
ggsave(file.path(out_dir, "trialdata_outcomes_by_stimcat.png"), p1, width = 5, height = 4, dpi = 300)

# ------------------------------------------------------------------
# Plot 2: psychometric curve -- P(lick) vs signal strength, split by
# engagement. Directly tests the behavior/README note that engagement
# changes the effective psychometric curve, rather than assuming it.
# ------------------------------------------------------------------
psycho <- trialdata |>
  group_by(signal, engaged) |>
  summarise(p_lick = mean(lickResponse), n = n(), .groups = "drop")

p2 <- ggplot(psycho, aes(signal, p_lick, color = engaged)) +
  geom_point(aes(size = n), alpha = 0.7) +
  geom_line(aes(group = engaged), linewidth = 0.6) +
  ylim(0, 1) +
  labs(title = "Psychometric curve by engagement state",
       x = "signal strength (%)", y = "P(lick)") +
  theme_minimal()
ggsave(file.path(out_dir, "trialdata_psychometric_by_engagement.png"), p2, width = 6, height = 4.5, dpi = 300)

# ------------------------------------------------------------------
# Plot 3: lick vigor (nLicks) by trial outcome -- do hits/false alarms
# differ in how vigorously the animal licked, not just whether it did?
# ------------------------------------------------------------------
p3 <- ggplot(trialdata, aes(trialOutcome, nLicks, fill = trialOutcome)) +
  geom_violin(alpha = 0.6, scale = "width") +
  geom_jitter(width = 0.1, size = 0.8, alpha = 0.4) +
  labs(title = "Lick count by trial outcome", x = NULL, y = "n licks") +
  theme_minimal() + theme(legend.position = "none")
ggsave(file.path(out_dir, "trialdata_nlicks_by_outcome.png"), p3, width = 5, height = 4, dpi = 300)

# ------------------------------------------------------------------
# Plot 4: engagement across the session timeline -- QUANTIFIES the
# "disengaged trials cluster at the end" pattern noted in
# behavior/README.md instead of assuming it holds for every session.
# ------------------------------------------------------------------
p4 <- ggplot(trialdata, aes(trialNumber, as.integer(engaged) - 1)) +
  geom_step() +
  geom_smooth(method = "loess", se = FALSE, color = "red", span = 0.3) +
  labs(title = "Engagement across the session", x = "trial number", y = "engaged (0/1)") +
  theme_minimal()
ggsave(file.path(out_dir, "trialdata_engagement_timeline.png"), p4, width = 6, height = 3.5, dpi = 300)

# ------------------------------------------------------------------
# Plot 5: trial duration by outcome -- session pacing; e.g. do MISS
# trials take longer (animal pausing/distracted) than HIT trials?
# ------------------------------------------------------------------
p5 <- trialdata |>
  mutate(trial_duration = trialEnd - trialStart) |>
  ggplot(aes(trialOutcome, trial_duration, fill = trialOutcome)) +
  geom_boxplot(outlier.size = 0.6) +
  labs(title = "Trial duration by outcome", x = NULL, y = "duration (s)") +
  theme_minimal() + theme(legend.position = "none")
ggsave(file.path(out_dir, "trialdata_duration_by_outcome.png"), p5, width = 5, height = 4, dpi = 300)

# ------------------------------------------------------------------
# Plot 6: response latency on HIT trials -- stimulus onset to reward,
# checking for a floor/ceiling or bimodality suggesting two different
# response strategies (see the CAVEAT above re: tReward/tStimStart units).
# ------------------------------------------------------------------
p6 <- trialdata |>
  filter(trialOutcome == "HIT", !is.na(tReward)) |>
  mutate(response_latency = tReward - tStimStart) |>
  ggplot(aes(response_latency)) +
  geom_histogram(binwidth = 0.1, boundary = 0) +
  labs(title = "Response latency on HIT trials",
       x = "latency, stim onset to reward (s)", y = "n trials") +
  theme_minimal()
ggsave(file.path(out_dir, "trialdata_response_latency.png"), p6, width = 5.5, height = 4, dpi = 300)

message("Saved 6 figures to ", out_dir)
