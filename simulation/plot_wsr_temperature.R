# Fixed-alpha temperature comparison, styled like plot_wsr.R. No summary CSV output.
# Usage: Rscript simulation/plot_wsr_temperature.R [input_directory] [output_file]
# Packages: install.packages(c("ggplot2", "dplyr", "gridExtra", "tikzDevice"))
# Uses pdflatex and the manuscript's lmodern fonts; exports PDF and TeX.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
})

args <- commandArgs(trailingOnly = TRUE)
if (any(args %in% c("-h", "--help"))) {
  cat("Usage: Rscript simulation/plot_wsr_temperature.R [input_directory] [output_file]\n",
      "Defaults: simulation/results_wsr_temperature and <input_directory>/wsr_temperature_comparison.pdf\n",
      sep = "")
  quit(status = 0)
}
if (length(args) > 2) stop("Expected at most an input directory and output file.")
script_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
input <- if (length(args) >= 1) args[1] else file.path(dirname(script_path), "results_wsr_temperature")
output <- if (length(args) >= 2) args[2] else file.path(input, "wsr_temperature_comparison.pdf")

methods <- c("score_dtm", "score_hete", "score_homo", "binarized_half", "binarized_quarter")
labels <- c("SCoRE: deterministic", "SCoRE: heterogeneous", "SCoRE: homogeneous",
            "Binarized: $u=\\alpha/2$", "Binarized: $u=\\alpha/4$")
palette <- c("#607466", "#DDAA00", "#E96056", "#1c84c6", "#93AACA")
paths <- list.files(input, pattern = "^temperature_setting_[34]_n[0-9]+_m[0-9]+_seeds_[0-9]+_[0-9]+[.]csv$",
                    full.names = TRUE)
if (!length(paths)) stop("No raw temperature_setting_* CSV files found in: ", input)
raw <- bind_rows(lapply(paths, read.csv))
required <- c("setting", "seed", "alpha", "tau", "method", "ncalib",
              "ntest", "pool_ncalib", "wsr", "fraction_selected", "target", "cost_mode",
              "loss_support", "training", "pairing", "source_setting", "cost_design")
if (!all(required %in% names(raw))) stop("Raw results are missing required columns.")
if (anyNA(raw[required])) stop("Raw results contain missing values.")
if (!all(vapply(raw[c("alpha", "tau", "wsr", "fraction_selected")],
                function(x) all(is.finite(x)), logical(1)))) stop("Nonfinite values found.")
if (!setequal(raw$method, methods) || !all(raw$setting %in% 3:4)) stop("Unknown setting or method.")
if (!all(raw$source_setting == 6) ||
    !all(raw$cost_mode == ifelse(raw$setting == 3, "unit", "unrelated")) ||
    !all(raw$cost_design == ifelse(raw$setting == 3, "unit", "interaction_sigmoid_v1"))) {
  stop("Expected the main-text sigmoid DGP with unit or independent costs.")
}
if (any(raw$tau <= 0) || !all(raw$target == "WSR") ||
    !all(raw$loss_support == "interval") || !all(raw$training == "refit_per_temperature") ||
    !all(raw$pairing == "same_xy_nested_calibration_v1")) stop("Unexpected temperature configuration.")
if (any(abs(raw$gamma[raw$method %in% methods[1:3]] - raw$alpha[raw$method %in% methods[1:3]]) > 1e-12)) stop("Expected gamma=alpha for SCoRE.")
if (anyDuplicated(raw[c("setting", "ncalib", "ntest", "seed", "tau", "method")])) {
  stop("Overlapping replications found.")
}
common <- intersect(c("alpha", "ntrain", "sigma", "dim", "trees", "score", "cost_features",
                      "target", "pool_ncalib", "ntest", "loss_support", "training", "pairing"), names(raw))
if (nrow(unique(raw[common])) != 1) stop("Different experiment configurations must be plotted separately.")
alpha <- unique(raw$alpha)
temperatures <- sort(unique(raw$tau))
config <- intersect(c(common, "source_setting", "case", "dgp", "clip", "cost_mode", "cost_design"), names(raw))
groups <- split(raw, interaction(raw$setting, raw$ncalib, raw$ntest, drop = TRUE))
for (panel_data in groups) {
  if (nrow(unique(panel_data[config])) != 1) stop("Multiple configurations found within a panel.")
  if (any(table(panel_data$seed) != length(temperatures) * length(methods))) {
    stop("A replication is missing a temperature/method comparison.")
  }
}

summary <- raw %>% group_by(setting, ncalib, ntest, tau, method) %>%
  summarise(nrep = n(), wsr_se = sd(wsr) / sqrt(n()),
            fraction_selected_se = sd(fraction_selected) / sqrt(n()),
            wsr = mean(wsr), fraction_selected = mean(fraction_selected), .groups = "drop")
if (any(summary$nrep < 2)) stop("At least two replications are needed for error bands.")
panels <- summary %>% distinct(setting, ncalib, ntest) %>% arrange(setting, ncalib, ntest) %>%
  mutate(panel = paste0(ifelse(setting == 3, "Unit cost", "Independent cost"),
                        "\n$n = ", ncalib, ",\\ m = ", ntest, "$"))
summary <- summary %>% left_join(panels, by = c("setting", "ncalib", "ntest")) %>%
  mutate(panel = factor(panel, levels = panels$panel), method = factor(method, levels = methods))

plot_metric <- function(metric, ylabel) {
  ggplot(summary, aes(x = tau, y = .data[[metric]], group = method)) +
    theme_bw(base_size = 14) +
    geom_ribbon(aes(ymin = .data[[metric]] - 2 * .data[[paste0(metric, "_se")]],
                    ymax = .data[[metric]] + 2 * .data[[paste0(metric, "_se")]], fill = method),
                alpha = 0.17, colour = NA, show.legend = FALSE) +
    geom_line(aes(colour = method), linewidth = 0.6) +
    geom_point(aes(colour = method), size = 0.7) +
    facet_wrap(vars(panel), nrow = 1, scales = "free_y") +
    scale_colour_manual(values = palette, labels = labels, name = NULL, drop = FALSE) +
    scale_fill_manual(values = palette, guide = "none") +
    scale_x_log10(breaks = temperatures, labels = temperatures) +
    guides(colour = guide_legend(nrow = 1)) +
    theme(legend.position = "top", legend.text = element_text(size = 13),
          legend.key.width = grid::unit(1.2, "lines"), strip.text = element_text(size = 13),
          axis.title = element_text(size = 16), axis.text = element_text(size = 12),
          panel.spacing = grid::unit(0.6, "lines"), plot.margin = margin(5, 6, 5, 6)) +
    labs(y = ylabel, x = NULL)
}
wsr_plot <- plot_metric("wsr", "Realized WSR")
wsr_plot$layers <- c(list(geom_hline(yintercept = alpha, colour = "grey55",
                                     linetype = "dotted", linewidth = 0.5)), wsr_plot$layers)
selection_plot <- plot_metric("fraction_selected", "Fraction selected") +
  labs(x = "Temperature $\\tau$ (log scale)")
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
project_root <- file.path(dirname(script_path), "..")
source(file.path(project_root, "plotting", "latex.R"), local = TRUE)
save_latex_plot(list(wsr_plot, selection_plot), output, project_root,
                 width = max(9, 14.5 * nrow(panels) / 6), height = 6.2, row_heights = c(1, 1.08))
cat("Fixed alpha:", alpha, "; panels:", nrow(panels), "; temperatures:", length(temperatures), "\n")
cat("Saved", normalizePath(output, winslash = "/"), "\n")
