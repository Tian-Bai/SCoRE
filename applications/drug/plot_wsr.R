# Section 8.1: rows = count/SA-cost normalization; columns = WSR/selection/reward.
# Usage: Rscript applications/drug/plot_wsr.R [input_directory] [output_file]
# Defaults to caco2_wang with diversity reward and PDF output. No summary CSVs.
# Packages: install.packages(c("ggplot2", "dplyr", "gridExtra", "tikzDevice"))
# PDF/TeX output uses real LaTeX (pdflatex + lmodern), via plotting/latex.R.
# Figures are 10.2 x 7.2 inches (15% narrower).

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
})
args <- commandArgs(trailingOnly = TRUE)
if (!exists("drug_plot_config", inherits = FALSE)) {
  drug_plot_config <- list(prefix = "drug_wsr", results = file.path("results_wsr", "caco2_wang", "diversity"))
}
if (any(args %in% c("-h", "--help"))) {
  cat("Usage: Rscript applications/drug/plot_wsr.R [input_directory] [output_file]\n")
  quit(status = 0)
}
if (length(args) > 2) stop("Expected at most an input directory and output file.")
script_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
input <- if (length(args)) args[1] else file.path(dirname(script_path), drug_plot_config$results)
output <- if (length(args) > 1) args[2] else file.path(input, paste0(drug_plot_config$prefix, "_comparison.pdf"))
paths <- list.files(input, pattern = paste0("^", drug_plot_config$prefix, "_seeds_[0-9]+_[0-9]+[.]csv$"), full.names = TRUE)
if (!length(paths)) stop("No raw ", drug_plot_config$prefix, "_seeds_* CSV files found in: ", input)
raw <- bind_rows(lapply(paths, read.csv))
scores <- c("loss", "cost", "reward", "cost_reward")
metrics <- c("wsr", "fraction_selected", "avg_reward")
metadata <- c("dataset", "model", "reward", "activity_offset", "reward_floor", "shift", "loss",
              "cost_definition", "support", "prediction_seed", "predictions_sha256", "ncalib", "ntest", "method")
if (drug_plot_config$prefix == "drug_wdr_w") {
  metadata <- c(metadata, "sampling", "weights", "weight_seed", "weight_trees", "probability_clip")
}
required <- c(metadata, "seed", "normalization", "alpha", "gamma", "score", metrics)
if (!all(required %in% names(raw)) || anyNA(raw[required])) stop("Missing columns or values in raw results.")
if (!all(vapply(raw[c("alpha", "gamma", metrics)], function(x) all(is.finite(x)), logical(1)))) {
  stop("Nonfinite levels or metrics in raw results.")
}
if (nrow(unique(raw[metadata])) != 1) stop("Plot one dataset, reward, model, and boosting method at a time.")
if (!setequal(raw$normalization, c("count", "cost")) || !setequal(raw$score, scores)) {
  stop("Both normalizations and all four scores are required.")
}
if (any(raw$alpha <= 0) || any(abs(raw$alpha - raw$gamma) > 1e-12)) stop("Expected positive alpha and gamma=alpha.")
keys <- c("seed", "normalization", "alpha", "score")
if (anyDuplicated(raw[keys])) stop("Overlapping replications found; use one copy of each seed.")
seeds <- sort(unique(raw$seed))
for (normalization in c("count", "cost")) {
  panel <- raw[raw$normalization == normalization, ]
  if (!setequal(panel$seed, seeds) || any(table(panel$seed) != 4 * length(unique(panel$alpha)))) {
    stop("Incomplete paired score/alpha comparisons.")
  }
}
if (length(seeds) < 2) stop("At least two replications are needed for error bands.")

long <- bind_rows(lapply(metrics, function(metric) {
  raw %>% transmute(seed, normalization, alpha, score, metric = metric, value = .data[[metric]])
}))
summary <- long %>% group_by(normalization, alpha, score, metric) %>%
  summarise(se = sd(value) / sqrt(n()), value = mean(value), .groups = "drop") %>%
  mutate(score = factor(score, levels = scores),
         metric = factor(metric, levels = metrics, labels = c("WSR", "Fraction selected", "Average reward")))
palette <- c("#607466", "#1c84c6", "#DDAA00", "#E96056")
score_labels <- c("$\\hat{\\ell}$", "$\\hat{\\ell}-\\alpha c$",
               "$\\hat{\\ell}/\\hat{r}$", "$(\\hat{\\ell}-\\alpha c)/\\hat{r}$")

plot_row <- function(normalization, ylabel) {
  panel <- filter(summary, .data$normalization == .env$normalization)
  reference <- filter(panel, metric == "WSR") %>% distinct(alpha, metric)
  ggplot(panel, aes(alpha, value, group = score)) +
    theme_bw(base_size = 16) +
    geom_line(data = reference, aes(alpha, alpha), inherit.aes = FALSE,
              linetype = "dotted", colour = "black", linewidth = 0.6) +
    geom_ribbon(aes(ymin = value - 2*se, ymax = value + 2*se, fill = score),
                alpha = 0.15, colour = NA, show.legend = FALSE) +
    geom_line(aes(colour = score, linetype = score), linewidth = 0.75) +
    geom_point(aes(colour = score, shape = score), size = 1.7) +
    facet_wrap(vars(metric), nrow = 1, scales = "free_y") +
    scale_colour_manual(values = palette, labels = score_labels, name = NULL, drop = FALSE) +
    scale_fill_manual(values = palette, guide = "none") +
    scale_linetype_manual(values = c("solid", "dashed", "solid", "dashed"),
                          labels = score_labels, name = NULL, drop = FALSE) +
    scale_shape_manual(values = c(16, 15, 17, 18), labels = score_labels, name = NULL, drop = FALSE) +
    guides(colour = guide_legend(nrow = 1), linetype = guide_legend(nrow = 1),
           shape = guide_legend(nrow = 1)) +
    labs(x = "Target level $\\alpha$", y = ylabel) +
    theme(legend.position = "top", legend.text = element_text(size = 17),
          legend.key.width = grid::unit(1.8, "lines"), strip.text = element_text(size = 17),
          axis.title = element_text(size = 18), axis.text = element_text(size = 14),
          panel.spacing = grid::unit(1, "lines"), plot.margin = margin(6, 8, 6, 8))
}
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
project_root <- file.path(dirname(script_path), "..", "..")
source(file.path(project_root, "plotting", "latex.R"), local = TRUE)
save_latex_plot(list(plot_row("count", "Count-normalized"), plot_row("cost", "Cost-normalized")),
                 output, project_root, width = 10.2, height = 7.2)
cat("Dataset:", unique(raw$dataset), "; reward:", unique(raw$reward), "; replications:", length(seeds), "\n")
cat("Saved", normalizePath(output, winslash = "/"), "\n")
