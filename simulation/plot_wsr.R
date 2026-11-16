# Six-setting WSR comparison with shaded +/- 2 Monte Carlo standard errors.
# Usage: Rscript simulation/plot_wsr.R [input_directory] [output_file]
# Packages: install.packages(c("ggplot2", "dplyr", "gridExtra", "tikzDevice"))
# Uses pdflatex and the manuscript's lmodern fonts; exports PDF and TeX.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
})

args <- commandArgs(trailingOnly = TRUE)
if (any(args %in% c("-h", "--help"))) {
  cat("Usage: Rscript simulation/plot_wsr.R [input_directory] [output_file]\n",
      "Defaults: simulation/results_wsr and <input_directory>/wsr_comparison.pdf\n",
      "Output may be PDF or standalone TeX (requires pdflatex).\n", sep = "")
  quit(status = 0)
}
if (length(args) > 2) stop("Expected at most an input directory and output file.")
script_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
input <- if (length(args) >= 1) args[1] else file.path(dirname(script_path), "results_wsr")
output <- if (length(args) >= 2) args[2] else file.path(input, "wsr_comparison.pdf")

methods <- c("score_dtm", "score_hete", "score_homo",
             "binarized_half", "binarized_quarter")
palette <- c("#607466", "#DDAA00", "#E96056", "#1c84c6", "#93AACA")
method_labels <- c(
  "SCoRE: deterministic", "SCoRE: heterogeneous", "SCoRE: homogeneous",
  "Binarized: $u=\\alpha/2$", "Binarized: $u=\\alpha/4$"
)
panel_labels <- c(
  "Excess loss\nUnit cost", "Squared error\nUnit cost", "Sigmoid loss\nUnit cost",
  "Sigmoid loss\nIndependent cost", "Sigmoid loss\nPos. correlated cost",
  "Sigmoid loss\nNeg. correlated cost"
)

paths <- list.files(input, pattern = "^setting_[1-6]_seeds_[0-9]+_[0-9]+[.]csv$",
                    full.names = TRUE)
if (!length(paths)) stop("No raw setting_*_seeds_*.csv files found in: ", input)
raw <- bind_rows(lapply(paths, function(path) {
  frame <- read.csv(path)
  # Accept earlier results without changing the source CSVs.
  if (!"wsr" %in% names(frame)) names(frame)[names(frame) == "risk"] <- "wsr"
  if ("target" %in% names(frame)) frame$target <- "WSR"
  if (!"cost_design" %in% names(frame)) {
    frame$cost_design <- ifelse(frame$cost_mode == "unit", "unit",
                               ifelse(frame$cost_mode == "unrelated", "x10_linear", "mu_sigmoid"))
  }
  if (!"cost_features" %in% names(frame)) frame$cost_features <- FALSE
  frame
}))
required <- c("setting", "seed", "alpha", "method", "wsr", "fraction_selected")
if (!all(required %in% names(raw))) stop("Raw results are missing required columns.")
if (anyNA(raw[required])) stop("Raw results contain missing values.")
if (anyDuplicated(raw[c("setting", "seed", "alpha", "method")])) {
  stop("Overlapping seed ranges found; each replication must appear only once.")
}
if (!setequal(raw$setting, 1:6) || !setequal(raw$method, methods)) {
  stop("Expected all six settings and the five comparison methods.")
}
if (any(!is.finite(raw$wsr)) || any(!is.finite(raw$fraction_selected))) {
  stop("Metrics must be finite.")
}
# Prevent pooling runs with different configurations or incomplete comparisons.
config <- intersect(c("source_setting", "case", "dgp", "sigma", "dim", "tau",
                      "clip", "ntrain", "ncalib", "ntest", "trees", "cost_mode", "cost_design", "cost_features",
                      "score"), names(raw))
for (setting_data in split(raw, raw$setting)) {
  if (length(config) && nrow(unique(setting_data[config])) != 1) {
    stop("Multiple configurations found within a setting.")
  }
  expected <- length(unique(setting_data$alpha)) * length(methods)
  if (any(table(setting_data$seed) != expected)) {
    stop("A replication is missing an alpha/method comparison.")
  }
}

summary <- raw %>%
  group_by(setting, alpha, method) %>%
  summarise(
    nrep = n(),
    # Compute SEs before replacing raw columns with their means.
    wsr_se = sd(wsr) / sqrt(n()),
    selected_se = sd(fraction_selected) / sqrt(n()),
    wsr = mean(wsr), selected = mean(fraction_selected),
    .groups = "drop"
  ) %>%
  mutate(method = factor(method, levels = methods),
         panel = factor(setting, levels = 1:6, labels = panel_labels))
if (any(summary$nrep < 2)) stop("At least two replications are needed for error bands.")

plot_metric <- function(metric, ylabel) {
  ggplot(summary, aes(x = alpha, y = .data[[metric]], group = method)) +
    theme_bw(base_size = 14) +
    geom_ribbon(aes(
      ymin = .data[[metric]] - 2 * .data[[paste0(metric, "_se")]],
      ymax = .data[[metric]] + 2 * .data[[paste0(metric, "_se")]], fill = method
    ), alpha = 0.17, colour = NA, show.legend = FALSE) +
    geom_line(aes(colour = method), linewidth = 0.6) +
    geom_point(aes(colour = method), size = 0.9) +
    facet_wrap(vars(panel), nrow = 1, scales = "free_y") +
    scale_colour_manual(values = palette, labels = method_labels, name = NULL,
                        drop = FALSE) +
    scale_fill_manual(values = palette, guide = "none") +
    scale_x_continuous(breaks = c(0.1, 0.3, 0.5)) +
    guides(colour = guide_legend(nrow = 1)) +
    theme(
      legend.position = "top", legend.text = element_text(size = 13),
      legend.key.width = grid::unit(0.85, "lines"),
      legend.spacing.x = grid::unit(0.15, "lines"),
      strip.text = element_text(size = 14), axis.title = element_text(size = 16),
      axis.text = element_text(size = 13), panel.spacing = grid::unit(0.6, "lines"),
      plot.margin = margin(5, 6, 5, 6)
    ) +
    labs(y = ylabel, x = NULL)
}

wsr_plot <- plot_metric("wsr", "Realized WSR")
# Draw the target line behind the ribbons and curves.
wsr_plot$layers <- c(list(geom_abline(intercept = 0, slope = 1,
                                     colour = "grey", linewidth = 0.5)),
                    wsr_plot$layers)
selection_plot <- plot_metric("selected", "Fraction selected") +
  labs(x = "WSR target level $\\alpha$")

dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
project_root <- file.path(dirname(script_path), "..")
source(file.path(project_root, "plotting", "latex.R"), local = TRUE)
save_latex_plot(list(wsr_plot, selection_plot), output, project_root,
                 width = 14.5, height = 6.2, row_heights = c(1, 1.08))
cat("Saved", normalizePath(output, winslash = "/"), "\n")
