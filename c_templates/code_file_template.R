# ==============================================================================
# <N>_<short_name>.R
# ==============================================================================
# <one-line description>
#
# Copy into 1_scripts/, keep it directly there. Only load from 0_data/ or
# an EARLIER script's 2_pipeline/<name>/out/.
# ==============================================================================

this_file <- normalizePath(sub("^--file=", "",
  grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)))
source(file.path(dirname(dirname(this_file)), "1_scripts", "utils", "pipeline.R"))
paths <- get_pipeline_paths()
root  <- project_root()

library(tidyverse)

csv_path <- file.path(root, "0_data", "DN", "TODO_mouse", "TODO_date", "TODO.csv")
df <- read_csv(csv_path, show_col_types = FALSE)
glimpse(df)

# p1 <- ggplot(df, aes(...)) + geom_...()
# ggsave(file.path(paths$out, "figure_name.png"), p1, width = 6, height = 4, dpi = 300)

message("Saved figures to ", paths$out)
