
# ==============================================================================
# pipeline.R
# ==============================================================================
# R equivalent of infotheory/pipeline.py -- same out/store/tmp convention.
# Source this at the top of any script living directly in `1_scripts/` --
# see any of 1_behavior.R / 2_single.R for the exact one-line bootstrap to
# copy (it has to locate ITSELF relative to the running script first).
#
# NOTE: depth-dependent, same caveat as pipeline.py -- assumes scripts live
# directly in `1_scripts/` (one level below project root).
# ==============================================================================

# ==============================================================================
# pipeline.R
# ==============================================================================
# R has no reliable equivalent of Python's `__file__` -- one that works
# identically under `Rscript`, `source()`, and interactive/RStudio
# execution. Rather than keep patching heuristics for each invocation mode
# (which is what the previous version of this file did, and which still
# broke under `source()`/RStudio's Source button), this version follows
# the source article's own advice instead: always run with the WORKING
# DIRECTORY set to the project root, and use relative paths from there.
#
# Practically: open this repo as an RStudio Project (File -> Open Project,
# or double-click a .Rproj file at the repo root) and RStudio sets the
# working directory correctly every time, automatically. From a terminal,
# `cd` to the repo root before running `Rscript 1_scripts/1_behavior.R`.
#
# The tradeoff for this robustness: each script needs one extra line, a
# NAME constant -- matching the original de Kok generator's own template
# (see 1_scripts/1_behavior.R for the exact pattern to copy).
# ==============================================================================

check_project_root <- function() {
  ok <- dir.exists("1_scripts") && dir.exists("0_data")
  if (!ok) {
    stop(
      "Working directory doesn't look like the project root (expected to find ",
      "1_scripts/ and 0_data/ here). Current working directory: ", getwd(), "
",
      "Fix: setwd() to the repo root before running this script, or open the ",
      "repo as an RStudio Project so the working directory is set automatically."
    )
  }
  invisible(TRUE)
}

get_pipeline_paths <- function(name, subdirs = c("out", "store", "tmp")) {
  check_project_root()
  pipeline_dir <- file.path("2_pipeline", name)
  paths <- list()
  for (sub in subdirs) {
    p <- file.path(pipeline_dir, sub)
    dir.create(p, showWarnings = FALSE, recursive = TRUE)
    paths[[sub]] <- p
  }
  paths
}

project_root <- function() {
  check_project_root()
  normalizePath(getwd())
}
