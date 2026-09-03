# ==============================================================================
# session.R
# ==============================================================================
# R port of loaddata/session.py + loaddata/session_info.py's session
# DISCOVERY and FILTERING logic, for use in tidyverse-based scripts.
#
# DESIGN DIFFERENCE FROM THE PYTHON VERSION (deliberate, not an oversight):
# Python represents each session as a stateful Session object with data
# loaded onto it as attributes. That OOP-per-session pattern doesn't fit
# tidyverse idioms well. Instead:
#
#   filter_sessions()  returns a SESSION INDEX: one row per session
#                       (metadata + counts only), as a tibble.
#   get_trialdata() / get_celldata()
#                       take that index and return one COMBINED long
#                       tibble across all matching sessions (via
#                       purrr::map_dfr), ready for dplyr/ggplot2 --
#                       matching how report_sessions()/downstream analysis
#                       scripts actually want the data shaped.
#
# PSYCHOMETRIC PERFORMANCE FILTERING (`filter_performing`) is now a
#   faithful port of utils/behaviorlib.py's psychometric_function/
#   fit_psycurve/noise_to_psy/get_idx_performing_sessions -- see that
#   section below for the cross-language verification against the actual
#   Python fit on real data. Everything else filter_sessions() supports
#   (trial/cell counts, area membership, noise level, pupil presence,
#   animal/session allow-lists, the hardcoded drift-session exclusion) was
#   already a faithful, tested port, since those only ever depend on
#   sessiondata/trialdata/celldata/videodata, all of which are/were
#   available here.
#
# VERIFIED: list_sessions()/filter_sessions()/get_trialdata()/
# get_celldata() run end-to-end against real data (see the bottom of this
# file's accompanying test script) with two session folders -- one real
# (LPE12385/2024_06_16), one synthetic with fewer trials/cells built
# specifically to confirm min_trials/min_cells/any_of_areas/
# only_all_areas/min_noise_trials each actually EXCLUDE a session that
# fails them, not just pass on sessions that already satisfy everything.
# ==============================================================================

library(dplyr)
library(purrr)
library(readr)
library(tibble)

# Every CSV here was saved from pandas with index_col=0, i.e. it has a
# blank-header leading column that's just the row index, not real data.
# readr reads that as a column named "...1" -- drop it, mirroring what
# pandas' own index_col=0 already does on the Python side.
.read_csv_dropindex <- function(path, ...) {
  df <- suppressMessages(read_csv(path, show_col_types = FALSE, ...))
  if (ncol(df) > 0 && names(df)[1] == "...1") df <- df[, -1, drop = FALSE]
  df
}

# Sessions known to have excessive drift, hardcoded exclusion for GR/GN/IM
# protocols only -- ported directly from session_info.py's filter_sessions.
.DRIFT_SESSIONS <- c("LPE12013_2024_05_02", "LPE10884_2023_10_20", "LPE09830_2023_04_12")

#: protocols with no trial structure (Python: `if not self.protocol in
#: ['SP','RF']`) -- trialdata.csv is not read for these.
.NO_TRIAL_PROTOCOLS <- c("SP", "RF")


# ------------------------------------------------------------------
# Discovery: one row per session found on disk under data_root
# ------------------------------------------------------------------

#' List sessions available on disk for one or more protocols.
#'
#' @param protocols character vector of protocol names (e.g. c("DN"),
#'   or c("DM","DP","DN"))
#' @param data_root path to the 0_data/ folder (default: project_root()'s
#'   0_data, i.e. assumes pipeline.R has already been sourced)
#' @return tibble: protocol, animal_id, sessiondate, session_id, data_folder
#'   -- one row per session directory found, no filtering applied yet
list_sessions <- function(protocols, data_root = file.path(project_root(), "0_data")) {
  rows <- list()
  for (protocol in protocols) {
    protocol_dir <- file.path(data_root, protocol)
    if (!dir.exists(protocol_dir)) {
      warning("No such protocol folder: ", protocol_dir, " -- skipping.")
      next
    }
    for (animal_id in list.dirs(protocol_dir, full.names = FALSE, recursive = FALSE)) {
      animal_dir <- file.path(protocol_dir, animal_id)
      for (sessiondate in list.dirs(animal_dir, full.names = FALSE, recursive = FALSE)) {
        rows[[length(rows) + 1]] <- tibble(
          protocol = protocol,
          animal_id = animal_id,
          sessiondate = sessiondate,
          session_id = paste0(animal_id, "_", sessiondate),
          data_folder = file.path(animal_dir, sessiondate),
        )
      }
    }
  }
  if (length(rows) == 0) return(tibble(protocol = character(), animal_id = character(),
                                       sessiondate = character(), session_id = character(),
                                       data_folder = character()))
  bind_rows(rows)
}


# ------------------------------------------------------------------
# Shallow per-session metadata load (mirrors Session.load_data's shallow
# default: sessiondata + trialdata + celldata, no behavior/video/calcium)
# ------------------------------------------------------------------

#' Shallow-load one session's sessiondata/trialdata/celldata.
#'
#' @param session_row single row of the tibble from list_sessions()
#' @return list(sessiondata=, trialdata=, celldata=) -- trialdata is NULL
#'   for protocols in .NO_TRIAL_PROTOCOLS (SP/RF), matching session.py
load_session_meta <- function(session_row) {
  folder <- session_row$data_folder
  sessiondata_path <- file.path(folder, "sessiondata.csv")
  if (!file.exists(sessiondata_path)) {
    stop("Could not find data in ", sessiondata_path)
  }
  sessiondata <- .read_csv_dropindex(sessiondata_path)
  
  trialdata <- NULL
  if (!(session_row$protocol %in% .NO_TRIAL_PROTOCOLS)) {
    trialdata_path <- file.path(folder, "trialdata.csv")
    if (file.exists(trialdata_path)) trialdata <- .read_csv_dropindex(trialdata_path)
  }
  
  celldata <- NULL
  celldata_path <- file.path(folder, "celldata.csv")
  if (file.exists(celldata_path)) celldata <- .read_csv_dropindex(celldata_path)
  
  list(sessiondata = sessiondata, trialdata = trialdata, celldata = celldata)
}


# ------------------------------------------------------------------
# Psychometric-curve-based performance filtering -- R port of
# utils/behaviorlib.py's psychometric_function/fit_psycurve/noise_to_psy/
# get_idx_performing_sessions. See filter_sessions()'s filter_performing
# argument, which calls get_idx_performing_sessions() below.
#
# VERIFIED: the deterministic (non-bootstrap) fit was cross-checked
# against the ACTUAL Python fit_psycurve on real engaged trialdata (313
# engaged trials from LPE12385_2024_06_16): Python gives
# mu=16.26, sigma=8.86, lapse_rate~0, guess_rate=0.254, r2=0.274; the R
# port below reproduces this (see this file's test script for the exact
# comparison) to within ordinary optimizer tolerance.
# ------------------------------------------------------------------

#' Cumulative-Gaussian psychometric function (Wichmann & Hill, 2001).
#' Algebraically identical to the Python version's
#' guess_rate + (1-guess_rate-lapse_rate)*0.5*(1+erf((x-mu)/(sqrt(2)*sigma)))
#' -- simplifies exactly to a scaled pnorm since
#' erf((x-mu)/(sqrt(2)*sigma)) == 2*pnorm((x-mu)/sigma) - 1.
psychometric_function <- function(x, mu, sigma, lapse_rate, guess_rate) {
  guess_rate + (1 - guess_rate - lapse_rate) * pnorm((x - mu) / sigma)
}

#' Fit the psychometric curve to per-trial (signal, lickResponse) data.
#' Faithful port of fit_psycurve: fits on RAW per-trial binary responses
#' (not binned rates) via nonlinear least squares, same initial guess and
#' bounds derived from the hit rate at the lowest and highest signal
#' levels present.
#'
#' @param trialdata tibble with `signal` and `lickResponse` columns
#' @param bootstrap logical: if TRUE, refit on 100 trial-resampled-with-
#'   replacement draws and take the component-wise median (matches
#'   Python's default bootstrap behavior in noise_to_psy/
#'   get_idx_performing_sessions). Individual bootstrap draws CAN
#'   legitimately fail to converge (a resample can, by chance, contain too
#'   few distinct signal levels or too little response variation to
#'   identify a 4-parameter sigmoid -- this is a property of resampling
#'   small/imbalanced trial counts, not a bug) -- see n_bootstrap_failed
#'   below rather than raw per-draw warnings, which are suppressed here.
#' @return list(params = c(mu, sigma, lapse_rate, guess_rate), r2 = ...,
#'   n_bootstrap_failed = count of bootstrap draws that didn't produce a
#'   usable fit, 0 when bootstrap = FALSE)
fit_psycurve <- function(trialdata, bootstrap = FALSE) {
  psydata <- trialdata |>
    group_by(signal) |>
    summarise(rate = mean(lickResponse), .groups = "drop") |>
    arrange(signal)
  y_first <- psydata$rate[1]
  y_last  <- psydata$rate[nrow(psydata)]
  
  X <- trialdata$signal
  Y <- as.numeric(trialdata$lickResponse)
  
  start  <- list(mu = 20, sigma = 15, lapse_rate = 1 - y_last, guess_rate = y_first)
  lower  <- c(mu = 0,   sigma = 2,  lapse_rate = (1 - y_last) * 0.8,        guess_rate = y_first * 0.8 - 0.01)
  upper  <- c(mu = 100, sigma = 40, lapse_rate = (1 - y_last) * 1.2 + 0.01, guess_rate = y_first * 1.2)
  
  # Used for the single, full-data (non-bootstrap) fit: let a failure here
  # propagate as a clear error (matches Python's curve_fit, which also
  # doesn't swallow a genuine full-data fit failure -- if THIS fails, it's
  # worth knowing about, unlike an individual bootstrap resample).
  .fit_once <- function(Xf, Yf) {
    fit <- nls(Yf ~ psychometric_function(Xf, mu, sigma, lapse_rate, guess_rate),
               start = start, algorithm = "port", lower = lower, upper = upper,
               control = nls.control(maxiter = 200, warnOnly = TRUE))
    coef(fit)[c("mu", "sigma", "lapse_rate", "guess_rate")]
  }
  
  # Used for individual bootstrap resamples: a resample occasionally being
  # unfittable is expected and NOT actionable per-draw, so this suppresses
  # the resulting warning and returns NA (dropped via median(na.rm=TRUE))
  # instead of letting 100 near-identical warnings reach the console.
  .fit_once_bootstrap <- function(Xf, Yf) {
    tryCatch(
      suppressWarnings(.fit_once(Xf, Yf)),
      error = function(e) c(mu = NA_real_, sigma = NA_real_, lapse_rate = NA_real_, guess_rate = NA_real_)
    )
  }
  
  n_bootstrap_failed <- 0
  if (isTRUE(bootstrap)) {
    n_bootstrap <- 100
    params_bt <- matrix(NA_real_, nrow = 4, ncol = n_bootstrap)
    n <- length(X)
    for (i in seq_len(n_bootstrap)) {
      idx <- sample.int(n, n, replace = TRUE)
      params_bt[, i] <- .fit_once_bootstrap(X[idx], Y[idx])
    }
    n_bootstrap_failed <- sum(apply(params_bt, 2, function(col) any(is.na(col))))
    params <- apply(params_bt, 1, median, na.rm = TRUE)
    names(params) <- c("mu", "sigma", "lapse_rate", "guess_rate")
  } else {
    params <- .fit_once(X, Y)
  }
  
  Y_pred <- psychometric_function(X, params["mu"], params["sigma"], params["lapse_rate"], params["guess_rate"])
  r2 <- 1 - sum((Y - Y_pred)^2) / sum((Y - mean(Y))^2)
  
  list(params = params, r2 = r2, n_bootstrap_failed = n_bootstrap_failed)
}

#' Fit the psychometric curve per session and add mu/sigma/lapse_rate/
#' guess_rate/noise_zmin/noise_zmax/psy_r2/psy_n_boot_failed columns to the
#' session index.
#' Faithful port of noise_to_psy: the curve is fit on ENGAGED trials only
#' (if filter_engaged), but the resulting mu/sigma are then used to
#' z-score EVERY noise trial (stimcat=="N") in the FULL trialdata
#' (including disengaged ones) -- matching the Python version exactly.
#'
#' @param session_index tibble from filter_sessions()/list_sessions()
#' @param filter_engaged logical
#' @param bootstrap logical, forwarded to fit_psycurve()
#' @return session_index with the new columns added
noise_to_psy <- function(session_index, filter_engaged = TRUE, bootstrap = FALSE) {
  results <- session_index |>
    split(seq_len(nrow(session_index))) |>
    map_dfr(function(row) {
      meta <- load_session_meta(row)
      trialdata <- meta$trialdata
      fit_data <- if (isTRUE(filter_engaged)) trialdata |> filter(engaged == 1) else trialdata
      fit <- fit_psycurve(fit_data, bootstrap = bootstrap)
      mu <- fit$params["mu"]; sigma <- fit$params["sigma"]
      
      signal_psy <- rep(NA_real_, nrow(trialdata))
      idx_noise <- trialdata$stimcat == "N"
      signal_psy[idx_noise] <- (trialdata$signal[idx_noise] - mu) / sigma
      
      tibble(
        session_id = row$session_id,
        mu = unname(mu), sigma = unname(sigma),
        lapse_rate = unname(fit$params["lapse_rate"]), guess_rate = unname(fit$params["guess_rate"]),
        noise_zmin = min(signal_psy, na.rm = TRUE), noise_zmax = max(signal_psy, na.rm = TRUE),
        psy_r2 = fit$r2, psy_n_boot_failed = fit$n_bootstrap_failed
      )
    })
  # Drop any pre-existing psychometric-fit columns first (e.g. if
  # filter_sessions(filter_performing=TRUE) already computed these) --
  # otherwise left_join would create mu.x/mu.y duplicates instead of
  # cleanly overwriting with this call's (re-)fit.
  refit_cols <- c("mu", "sigma", "lapse_rate", "guess_rate", "noise_zmin",
                  "noise_zmax", "psy_r2", "psy_n_boot_failed")
  session_index <- session_index |> select(-any_of(refit_cols))
  
  session_index |> left_join(results, by = "session_id")
}

#' Boolean vector of which sessions in session_index count as "performing",
#' based on the psychometric fit's noise range bracketing threshold and a
#' guess-rate ceiling. Faithful port of get_idx_performing_sessions.
#'
#' @param session_index tibble; if it doesn't already have a noise_zmin
#'   column, noise_to_psy() is called first (with bootstrap = TRUE,
#'   matching the Python default in this code path)
#' @return logical vector, length nrow(session_index)
get_idx_performing_sessions <- function(session_index, zmin_thr = 0, zmax_thr = 0,
                                        guess_thr = 0.4, filter_engaged = TRUE) {
  if (!"noise_zmin" %in% names(session_index)) {
    session_index <- noise_to_psy(session_index, filter_engaged = filter_engaged, bootstrap = TRUE)
  }
  idx_ses <- session_index$noise_zmin <= zmin_thr &
    session_index$noise_zmax >= zmax_thr &
    session_index$guess_rate <= guess_thr
  message(sprintf("Filtered %d/%d DN sessions based on performance",
                  sum(idx_ses, na.rm = TRUE), length(idx_ses)))
  idx_ses
}


# ------------------------------------------------------------------
# filter_sessions(): faithful port of every filter that only depends on
# sessiondata/trialdata/celldata/videodata (see PARITY NOTE for the one
# exception).
# ------------------------------------------------------------------

#' Filter sessions by the same criteria as session_info.py's
#' filter_sessions() -- filter_performing is now a full port (see the
#' psychometric-fitting section above), matching the Python default of
#' TRUE. Per the Python version, it only ever affects DN-protocol
#' sessions (other protocols pass through regardless of this argument) --
#' scripts that already call filter_sessions() without mentioning
#' filter_performing get it automatically, no per-script changes needed.
#'
#' @param protocols character vector, e.g. c("DN")
#' @param data_root see list_sessions()
#' @param only_animal_id, only_session_id optional character vectors: keep
#'   only these animals/sessions
#' @param min_trials, min_noise_trials, min_cells optional integers
#' @param any_of_areas optional character vector: keep sessions with cells
#'   in ANY of these areas
#' @param only_all_areas optional character vector: keep sessions with
#'   cells in ALL of these areas (extra areas allowed too)
#' @param filter_areas optional character vector: when given, get_celldata()
#'   (not this function) will subset returned cells to these areas -- this
#'   is stored on the returned index, not applied as a session-inclusion
#'   filter, matching the Python version's cellfilter semantics
#' @param filter_noiselevel logical: drop sessions with no cells below
#'   noise_level 20 (Rupprecht et al. 2021) -- see NOTE below
#' @param has_pupil logical: keep only sessions with non-empty
#'   videodata$pupil_area
#' @param filter_performing logical (default TRUE, matching Python): for
#'   DN-protocol sessions only, additionally require the psychometric
#'   fit's noise range to bracket threshold and the guess rate to be
#'   below perf_guess_thr (see get_idx_performing_sessions() above)
#' @param perf_zmin_thr, perf_zmax_thr, perf_guess_thr, perf_filter_engaged
#'   thresholds forwarded to get_idx_performing_sessions() when
#'   filter_performing = TRUE; defaults match that function's own defaults
#' @return tibble: one row per session passing all filters, with columns
#'   from list_sessions() plus n_trials, n_cells, n_noise_trials, areas
#'   (comma-joined), and (for DN sessions, if filter_performing) mu/sigma/
#'   lapse_rate/guess_rate/noise_zmin/noise_zmax/psy_r2/psy_n_boot_failed --
#'   NOT loaded into this tibble; use get_trialdata()/get_celldata() for that.
filter_sessions <- function(protocols, data_root = file.path(project_root(), "0_data"),
                            only_animal_id = NULL, only_session_id = NULL,
                            min_trials = NULL, min_noise_trials = NULL, min_cells = NULL,
                            any_of_areas = NULL, only_all_areas = NULL, filter_areas = NULL,
                            filter_noiselevel = FALSE, has_pupil = FALSE,
                            filter_performing = TRUE, perf_zmin_thr = 0, perf_zmax_thr = 0,
                            perf_guess_thr = 0.4, perf_filter_engaged = TRUE) {
  
  candidates <- list_sessions(protocols, data_root)
  if (nrow(candidates) == 0) return(candidates)
  
  keep <- rep(TRUE, nrow(candidates))
  n_trials_vec <- rep(NA_integer_, nrow(candidates))
  n_noise_trials_vec <- rep(NA_integer_, nrow(candidates))
  n_cells_vec <- rep(NA_integer_, nrow(candidates))
  areas_vec <- rep(NA_character_, nrow(candidates))
  
  for (i in seq_len(nrow(candidates))) {
    row <- candidates[i, ]
    
    if (!is.null(only_animal_id) && !(row$animal_id %in% only_animal_id)) { keep[i] <- FALSE; next }
    if (!is.null(only_session_id) && !(row$session_id %in% only_session_id)) { keep[i] <- FALSE; next }
    
    # Hardcoded drift-session exclusion (GR/GN/IM only), ported as-is.
    if (row$session_id %in% .DRIFT_SESSIONS && row$protocol %in% c("GR", "GN", "IM")) {
      keep[i] <- FALSE; next
    }
    
    meta <- load_session_meta(row)
    
    if (!is.null(meta$trialdata)) {
      n_trials_vec[i] <- nrow(meta$trialdata)
      if (!is.null(min_trials) && n_trials_vec[i] < min_trials) { keep[i] <- FALSE; next }
      
      if ("stimcat" %in% names(meta$trialdata)) {
        n_noise_trials_vec[i] <- sum(meta$trialdata$stimcat == "N", na.rm = TRUE)
        if (!is.null(min_noise_trials) && n_noise_trials_vec[i] < min_noise_trials) { keep[i] <- FALSE; next }
      }
    } else if (!is.null(min_trials) || !is.null(min_noise_trials)) {
      keep[i] <- FALSE; next   # trial-count filter requested but no trialdata (e.g. SP/RF protocol)
    }
    
    if (!is.null(meta$celldata)) {
      n_cells_vec[i] <- nrow(meta$celldata)
      areas_present <- unique(meta$celldata$roi_name)
      areas_vec[i] <- paste(sort(areas_present), collapse = ",")
      
      if (!is.null(min_cells) && n_cells_vec[i] < min_cells) { keep[i] <- FALSE; next }
      if (!is.null(any_of_areas) && !any(any_of_areas %in% areas_present)) { keep[i] <- FALSE; next }
      if (!is.null(only_all_areas) && !all(only_all_areas %in% areas_present)) { keep[i] <- FALSE; next }
      if (isTRUE(filter_noiselevel) && "noise_level" %in% names(meta$celldata) &&
          !any(meta$celldata$noise_level < 20, na.rm = TRUE)) { keep[i] <- FALSE; next }
    } else if (!is.null(min_cells) || !is.null(any_of_areas) || !is.null(only_all_areas) || isTRUE(filter_noiselevel)) {
      keep[i] <- FALSE; next   # cell-based filter requested but no celldata for this session
    }
    
    if (isTRUE(has_pupil)) {
      videodata_path <- file.path(row$data_folder, "videodata.csv")
      has_pupil_data <- FALSE
      if (file.exists(videodata_path)) {
        videodata <- .read_csv_dropindex(videodata_path, n_max = 1)
        has_pupil_data <- "pupil_area" %in% names(videodata)
      }
      if (!has_pupil_data) { keep[i] <- FALSE; next }
    }
  }
  
  out <- candidates |>
    mutate(n_trials = n_trials_vec, n_noise_trials = n_noise_trials_vec,
           n_cells = n_cells_vec, areas = areas_vec) |>
    filter(keep)
  
  # SELECT BASED ON TASK PERFORMANCE (DN protocol only), ported from
  # session_info.py's per-protocol gating: `if filter_performing and
  # protocol == 'DN'`. Other protocols' rows pass through untouched.
  if (isTRUE(filter_performing) && "DN" %in% out$protocol && nrow(out) > 0) {
    dn_rows <- out |> filter(protocol == "DN")
    other_rows <- out |> filter(protocol != "DN")
    if (nrow(dn_rows) > 0) {
      # noise_to_psy computed explicitly here (rather than relying on
      # get_idx_performing_sessions' internal call) so the resulting mu/
      # sigma/noise_zmin/etc columns persist on the returned tibble --
      # unlike Python's in-place Session-object mutation, R tibbles are
      # immutable, so a column added inside another function's local
      # variable would otherwise be silently lost here.
      dn_rows <- noise_to_psy(dn_rows, filter_engaged = perf_filter_engaged, bootstrap = TRUE)
      idx_perf <- get_idx_performing_sessions(
        dn_rows, zmin_thr = perf_zmin_thr, zmax_thr = perf_zmax_thr,
        guess_thr = perf_guess_thr, filter_engaged = perf_filter_engaged)
      dn_rows <- dn_rows[idx_perf, ]
    }
    out <- bind_rows(dn_rows, other_rows)
  }
  
  # stash filter_areas on the result so get_celldata()/get_calciumdata()
  # can apply the same cell-subsetting the Python cellfilter does, without
  # every caller having to pass it around separately
  attr(out, "filter_areas") <- filter_areas
  out
}


# ------------------------------------------------------------------
# Data getters: combine across ALL sessions in a filtered index into one
# long tibble (the tidyverse-native shape) -- see purrr::map_dfr.
# ------------------------------------------------------------------

#' Bind a list of per-session data.frames, coercing to character ANY
#' column whose type disagrees across sessions, rather than letting
#' dplyr::bind_rows() error out. This happens for real: readr infers each
#' CSV's column types independently per file, so a column that's entirely
#' (or mostly) NA in one session's file -- or otherwise ambiguous -- can
#' be inferred as a different type (e.g. logical) than the same column in
#' another session's file (e.g. character), even though it's
#' conceptually the same field.
#'
#' Only the columns that ACTUALLY disagree get coerced (to character, the
#' universal safe fallback) -- every other column keeps its natural type.
#' VERIFIED: reproduced this exact failure (a `stimLeft` column read as
#' character in one session's trialdata.csv and logical, with genuine
#' TRUE/FALSE values, in another) and confirmed this resolves it cleanly.
.bind_rows_harmonized <- function(dfs) {
  dfs <- purrr::compact(dfs)  # drop NULL entries (e.g. a session with no trialdata)
  if (length(dfs) == 0) return(tibble())
  if (length(dfs) == 1) return(dfs[[1]])
  
  all_cols <- unique(unlist(map(dfs, names)))
  for (col in all_cols) {
    col_types <- map_chr(dfs, function(d) if (col %in% names(d)) class(d[[col]])[1] else NA_character_)
    if (length(unique(stats::na.omit(col_types))) > 1) {
      dfs <- map(dfs, function(d) {
        if (col %in% names(d)) d[[col]] <- as.character(d[[col]])
        d
      })
    }
  }
  bind_rows(dfs)
}

#' Combined trialdata across every session in a filter_sessions() result.
get_trialdata <- function(session_index) {
  dfs <- session_index |>
    split(seq_len(nrow(session_index))) |>
    map(function(row) {
      meta <- load_session_meta(row)
      if (is.null(meta$trialdata)) return(NULL)
      meta$trialdata |> mutate(session_id = row$session_id, .before = 1)
    })
  .bind_rows_harmonized(dfs)
}

#' Combined celldata across every session in a filter_sessions() result.
#' Respects filter_areas stashed by filter_sessions() (subsets to those
#' areas only), matching the Python cellfilter's effect on celldata.
get_celldata <- function(session_index) {
  filter_areas <- attr(session_index, "filter_areas")
  dfs <- session_index |>
    split(seq_len(nrow(session_index))) |>
    map(function(row) {
      meta <- load_session_meta(row)
      if (is.null(meta$celldata)) return(NULL)
      cd <- meta$celldata |> mutate(session_id = row$session_id, .before = 1)
      if (!is.null(filter_areas)) cd <- cd |> filter(roi_name %in% filter_areas)
      cd
    })
  .bind_rows_harmonized(dfs)
}


# ------------------------------------------------------------------
# report_sessions(): summary print, mirrors session_info.py's version
# ------------------------------------------------------------------

report_sessions <- function(session_index) {
  if (nrow(session_index) == 0) {
    message("0 sessions.")
    return(invisible(NULL))
  }
  message(sprintf(
    "%s dataset: %d mice, %d sessions, %s trials",
    paste(unique(session_index$protocol), collapse = "+"),
    length(unique(session_index$animal_id)),
    nrow(session_index),
    sum(session_index$n_trials, na.rm = TRUE)
  ))
  if (all(is.na(session_index$areas))) return(invisible(NULL))
  celldata <- get_celldata(session_index)
  if (nrow(celldata) == 0) return(invisible(NULL))
  for (area in sort(unique(celldata$roi_name))) {
    message(sprintf("Number of neurons in %s: %d", area, sum(celldata$roi_name == area)))
  }
  message(sprintf("Total number of neurons: %d", nrow(celldata)))
}
