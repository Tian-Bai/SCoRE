# Reward-score comparison with shaded +/- 2 Monte Carlo standard errors.
# Usage: Rscript simulation/plot_wsr_reward.R [input_directory] [output_file]
# Packages: install.packages(c("ggplot2", "dplyr", "gridExtra", "tikzDevice"))
# Uses pdflatex and the manuscript's lmodern fonts; exports PDF and TeX.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
})

args <- commandArgs(trailingOnly = TRUE)
if (any(args %in% c("-h", "--help"))) {
  cat("Usage: Rscript simulation/plot_wsr_reward.R [input_directory] [output_file]\n",
      "Defaults: simulation/results_wsr_reward and <input_directory>/wsr_reward_comparison.pdf\n",
      "Output may be PDF or standalone TeX (requires pdflatex).\n", sep = "")
  quit(status = 0)
}
if (length(args) > 2) stop("Expected at most an input directory and output file.")
script_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
input <- if (length(args) >= 1) args[1] else file.path(dirname(script_path), "results_wsr_reward")
output <- if (length(args) >= 2) args[2] else file.path(input, "wsr_reward_comparison.pdf")

scores <- c("loss", "cost", "reward", "cost_reward")
palette <- c("#607466", "#1c84c6", "#DDAA00", "#E96056")
score_labels <- c(
  "$\\hat{\\ell}$", "$\\hat{\\ell}-\\alpha c$",
  "$\\hat{\\ell}/\\hat{r}$", "$(\\hat{\\ell}-\\alpha c)/\\hat{r}$"
)
panel_labels <- c(
  "Excess loss\nUnit cost", "Squared error\nUnit cost", "Sigmoid loss\nUnit cost",
  "Sigmoid loss\nIndependent cost", "Sigmoid loss\nPos. correlated cost",
  "Sigmoid loss\nNeg. correlated cost"
)

required <- c("setting", "seed", "alpha", "score", "wsr", "percent_selected",
              "avg_reward", "cost_mode")
paths <- list.files(input, pattern = "^reward_setting_[1-6]_seeds_[0-9]+_[0-9]+[.]csv$",
                    full.names = TRUE)
if (!length(paths)) stop("No raw reward_setting_*_seeds_*.csv files found in: ", input)
raw <- bind_rows(lapply(paths, function(path) {
  frame <- read.csv(path)
  if (!all(required %in% names(frame))) stop("Missing required columns in: ", path)
  # Legacy metadata are normalized in memory; source CSVs are not changed.
  if (!"cost_design" %in% names(frame)) {
    frame$cost_design <- ifelse(frame$cost_mode == "unit", "unit",
                               ifelse(frame$cost_mode == "unrelated", "x10_linear", "mu_sigmoid"))
  }
  if (!"cost_features" %in% names(frame)) frame$cost_features <- FALSE
  frame$cost_features <- as.logical(frame$cost_features)
  frame$fraction_selected <- frame$percent_selected / 100
  frame
}))
if (anyNA(raw[c(required, "cost_design", "cost_features")])) {
  stop("Raw results contain missing values or invalid metadata.")
}
if (anyDuplicated(raw[c("setting", "seed", "alpha", "score")])) {
  stop("Overlapping seed ranges found; each replication must appear only once.")
}
if (!all(raw$setting %in% 1:6) || !setequal(raw$score, scores)) {
  stop("Expected settings in 1:6 and the four comparison scores.")
}
metrics <- c("wsr", "fraction_selected", "avg_reward")
if (!all(vapply(raw[metrics], function(x) all(is.finite(x)), logical(1)))) {
  stop("Metrics must be finite.")
}

# Reject incompatible runs and incomplete score comparisons before averaging.
common <- intersect(c("sigma", "dim", "tau", "ntrain", "ncalib", "ntest", "trees",
                      "reward", "reward_floor", "boosting", "cost_features"), names(raw))
if (nrow(unique(raw[common])) != 1) stop("Multiple experiment configurations found.")
config <- intersect(c(common, "source_setting", "case", "dgp", "clip", "cost_mode",
                      "cost_design"), names(raw))
for (setting_data in split(raw, raw$setting)) {
  if (nrow(unique(setting_data[config])) != 1) {
    stop("Multiple configurations found within a setting.")
  }
  expected <- length(unique(setting_data$alpha)) * length(scores)
  if (any(table(setting_data$seed) != expected)) {
    stop("A replication is missing an alpha/score comparison.")
  }
}

summary <- raw %>%
  group_by(setting, alpha, score) %>%
  summarise(
    nrep = n(),
    # Compute SEs before replacing the raw metric columns with their means.
    wsr_se = sd(wsr) / sqrt(n()),
    fraction_selected_se = sd(fraction_selected) / sqrt(n()),
    avg_reward_se = sd(avg_reward) / sqrt(n()),
    wsr = mean(wsr), fraction_selected = mean(fraction_selected),
    avg_reward = mean(avg_reward),
    .groups = "drop"
  ) %>%
  mutate(score = factor(score, levels = scores),
         panel = factor(setting, levels = 1:6, labels = panel_labels))
if (any(summary$nrep < 2)) stop("At least two replications are needed for error bands.")

plot_metric <- function(metric, ylabel) {
  ggplot(summary, aes(x = alpha, y = .data[[metric]], group = score)) +
    theme_bw(base_size = 14) +
    geom_ribbon(aes(
      ymin = .data[[metric]] - 2 * .data[[paste0(metric, "_se")]],
      ymax = .data[[metric]] + 2 * .data[[paste0(metric, "_se")]], fill = score
    ), alpha = 0.17, colour = NA, show.legend = FALSE) +
    geom_line(aes(colour = score, linetype = score), linewidth = 0.6) +
    geom_point(aes(colour = score), size = 0.9) +
    facet_wrap(vars(panel), nrow = 1, scales = "free_y") +
    scale_colour_manual(values = palette, labels = score_labels, name = NULL,
                        drop = FALSE) +
    scale_fill_manual(values = palette, guide = "none") +
    scale_linetype_manual(values = c("solid", "dashed", "solid", "dashed"),
                          labels = score_labels, name = NULL, drop = FALSE) +
    scale_x_continuous(breaks = c(0.1, 0.3, 0.5)) +
    guides(colour = guide_legend(nrow = 1), linetype = guide_legend(nrow = 1)) +
    theme(
      legend.position = "top", legend.text = element_text(size = 15),
      legend.key.width = grid::unit(2, "lines"),
      legend.spacing.x = grid::unit(0.4, "lines"),
      strip.text = element_text(size = 14), axis.title = element_text(size = 16),
      axis.text = element_text(size = 13), panel.spacing = grid::unit(0.6, "lines"),
      plot.margin = margin(5, 6, 5, 6)
    ) +
    labs(y = ylabel, x = NULL)
}

wsr_plot <- plot_metric("wsr", "Realized WSR")
wsr_plot$layers <- c(list(geom_abline(intercept = 0, slope = 1,
                                     colour = "grey", linewidth = 0.5)),
                    wsr_plot$layers)
selection_plot <- plot_metric("fraction_selected", "Fraction selected")
# avg_reward is already sum(selected rewards) / number of test units.
reward_plot <- plot_metric("avg_reward", "Average reward") +
  labs(x = "WSR target level $\\alpha$")

dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
settings <- sort(unique(raw$setting))
project_root <- file.path(dirname(script_path), "..")
source(file.path(project_root, "plotting", "latex.R"), local = TRUE)
save_latex_plot(list(wsr_plot, selection_plot, reward_plot), output, project_root,
                 width = max(10.5, 14.5 * length(settings) / 6), height = 8.2,
                 row_heights = c(1, 1, 1.08))
cat("Plotted settings:", paste(settings, collapse = ", "), "\n")
cat("Predictor inputs:", if (raw$cost_features[1]) "X and c(X)" else "X only", "\n")
cat("Saved", normalizePath(output, winslash = "/"), "\n")
