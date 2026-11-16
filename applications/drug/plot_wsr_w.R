# Same two-row WSR figure for the molecular-weight covariate-shift experiment.
# Usage: Rscript applications/drug/plot_wdr_w.R [input_directory] [output_file]
# Default: results_wdr_w/caco2_wang/diversity/estimated/drug_wdr_w_comparison.pdf

if (any(commandArgs(trailingOnly = TRUE) %in% c("-h", "--help"))) {
  cat("Usage: Rscript applications/drug/plot_wdr_w.R [input_directory] [output_file]\n")
  quit(status = 0)
}
wrapper_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
drug_plot_config <- list(prefix = "drug_wdr_w",
                         results = file.path("results_wdr_w", "caco2_wang", "diversity", "estimated"))
source(file.path(dirname(wrapper_path), "plot_wsr.R"), local = TRUE)
