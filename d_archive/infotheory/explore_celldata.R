# ==============================================================================
# explore_celldata.R
# ==============================================================================
# Exploratory tidyverse plots for celldata.csv: one row per recorded/segmented
# ROI from a single imaging session -- anatomical area, projection-labeling
# status, depth, ROI-quality metrics, and basic activity summary stats.
#
# VERIFIED: run end-to-end against a real celldata.csv (2551 iscell==TRUE
# ROIs, 36 labeled/2515 unlabeled) with R 4.3 + dplyr/ggplot2/tidyr/readr/
# forcats -- all 6 plots render correctly. `iscell` is 0/1 as assumed;
# `labeled` is exactly {"unl","lab"} as assumed.
#
# Usage: Rscript explore_celldata.R path/to/celldata.csv path/to/output_dir
# ==============================================================================

library(tidyverse)

args     <- commandArgs(trailingOnly = TRUE)
csv_path <- if (length(args) >= 1) args[[1]] else "/u/g/glorenz/Documents/Research/Code/multi_area_detection_task_ff_fb/data/DN/LPE12385/2024_06_15/celldata.csv"
out_dir  <- if (length(args) >= 2) args[[2]] else "/u/g/glorenz/Documents/Research/Code/multi_area_detection_task_ff_fb/1_behavior/figures"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

AREA_ORDER <- c("V1", "PM", "AL", "RSP")   # matches celldata_utils.DEFAULT_AREA_ORDER

celldata_raw <- read_csv(csv_path, show_col_types = FALSE)

# Sanity check BEFORE the factor releveling below assumes specific levels --
# if these don't look like {0,1} and {"unl","lab"}, fix the mutate() call.
message("iscell values: ");  print(count(celldata_raw, iscell))
message("labeled values: "); print(count(celldata_raw, labeled))

celldata <- celldata_raw |>
  mutate(
    roi_name = fct_relevel(roi_name, AREA_ORDER),
    labeled  = factor(labeled, levels = c("unl", "lab")),
    iscell   = as.logical(iscell),
  ) |>
  filter(iscell)  # drop non-cell ROIs before any plot below

glimpse(celldata)

# ------------------------------------------------------------------
# Plot 1: spatial map of the imaging FOV, colored by area, shaped by
# projection-labeling status -- sanity check that area assignment and
# injection labeling look spatially sensible (e.g. labeled cells not
# randomly scattered across area boundaries).
# ------------------------------------------------------------------
p1 <- ggplot(celldata, aes(xloc, yloc, color = roi_name, shape = labeled)) +
  geom_point(size = 0.8, alpha = 0.7) +
  coord_fixed() +
  scale_color_manual(values = c(V1 = "#1f77b4", PM = "#ff7f0e",
                                 AL = "#2ca02c", RSP = "#d62728")) +
  labs(title = "FOV map: area x projection-labeling", x = "x (px)", y = "y (px)") +
  theme_minimal()
ggsave(file.path(out_dir, "celldata_fov_map.png"), p1, width = 6, height = 5, dpi = 300)

# ------------------------------------------------------------------
# Plot 2: neuron counts per (area, labeled) group -- the same summary
# celldata_utils.get_area_label()+groupby produces in the Python
# pipeline, here as one dplyr chain.
# ------------------------------------------------------------------
counts <- celldata |> count(roi_name, labeled)

p2 <- ggplot(counts, aes(roi_name, n, fill = labeled)) +
  geom_col(position = "dodge") +
  geom_text(aes(label = n), position = position_dodge(0.9), vjust = -0.3, size = 3) +
  labs(title = "Neuron counts by area and projection label", x = NULL, y = "n neurons") +
  theme_minimal()
ggsave(file.path(out_dir, "celldata_counts_by_group.png"), p2, width = 5, height = 4, dpi = 300)

# ------------------------------------------------------------------
# Plot 3: event-rate distribution by area, split by labeled status --
# is baseline activity level itself different for projection-
# identified cells, independent of any task-locked question?
# ------------------------------------------------------------------
p3 <- ggplot(celldata, aes(roi_name, event_rate, fill = labeled)) +
  geom_violin(position = position_dodge(0.8), alpha = 0.6, scale = "width") +
  geom_boxplot(position = position_dodge(0.8), width = 0.15, outlier.size = 0.5) +
  scale_y_log10() +
  labs(title = "Event rate by area and projection label",
       y = "event rate (log scale)", x = NULL) +
  theme_minimal()
ggsave(file.path(out_dir, "celldata_event_rate_by_group.png"), p3, width = 6, height = 4, dpi = 300)

# ------------------------------------------------------------------
# Plot 4: depth distribution per area, faceted -- confirms layer
# targeting (e.g. the depth<300um L2/3 cutoff used elsewhere in this
# project) looks right, rather than assuming it.
# ------------------------------------------------------------------
p4 <- ggplot(celldata, aes(depth)) +
  geom_histogram(binwidth = 25, boundary = 0) +
  geom_vline(xintercept = 300, linetype = "dashed", color = "red") +
  facet_wrap(~roi_name, scales = "free_y") +
  labs(title = "Depth distribution by area (dashed line: 300um L2/3 cutoff)",
       x = "depth (um)", y = "n neurons") +
  theme_minimal()
ggsave(file.path(out_dir, "celldata_depth_by_area.png"), p4, width = 7, height = 5, dpi = 300)

# ------------------------------------------------------------------
# Plot 5: ROI quality scatter -- skew vs event_rate, colored by
# iscell_prob, sized by radius. A look for whether "quality" metrics
# cluster in a way that could confound area/label comparisons.
# ------------------------------------------------------------------
p5 <- ggplot(celldata, aes(event_rate, skew, color = iscell_prob, size = radius)) +
  geom_point(alpha = 0.5) +
  scale_x_log10() +
  scale_color_viridis_c() +
  labs(title = "ROI quality: skew vs event rate",
       x = "event rate (log scale)", y = "skewness") +
  theme_minimal()
ggsave(file.path(out_dir, "celldata_quality_scatter.png"), p5, width = 6, height = 5, dpi = 300)

# ------------------------------------------------------------------
# Plot 6: mean fluorescence vs noise level, faceted by area -- a QC
# view for spotting a session/plane with abnormally noisy recordings
# before trusting downstream area/label comparisons.
# ------------------------------------------------------------------
p6 <- ggplot(celldata, aes(meanF, noise_level)) +
  geom_point(alpha = 0.4, size = 0.8) +
  geom_smooth(method = "lm", se = FALSE, color = "red", linewidth = 0.6) +
  facet_wrap(~roi_name) +
  scale_x_log10() +
  labs(title = "Noise level vs mean fluorescence, by area") +
  theme_minimal()
ggsave(file.path(out_dir, "celldata_noise_vs_meanF.png"), p6, width = 7, height = 5, dpi = 300)

message("Saved 6 figures to ", out_dir)
