# Count-normalized LLM WSR: one row, WSR / fraction selected / average reward.
# Usage: Rscript applications/llm/plot_wsr.R [input_directory] [output_file]
# Same LaTeX renderer and styling as the drug figures. No summary CSVs.
# Packages: install.packages(c("ggplot2", "dplyr", "gridExtra", "tikzDevice"))

suppressPackageStartupMessages({ library(ggplot2); library(dplyr) })
args <- commandArgs(trailingOnly = TRUE)
if (any(args %in% c("-h", "--help"))) {
  cat("Usage: Rscript applications/llm/plot_wsr.R [input_directory] [output_file]\n",
      "Defaults: results_wsr/confidence/llm_wsr_comparison.pdf beside this script.\n",
      "PDF or standalone TeX output uses pdflatex and lmodern.\n", sep = "")
  quit(status = 0)
}
if (length(args) > 2) stop("Expected at most an input directory and output file.")
script_path <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1])
input <- if (length(args)) args[1] else file.path(dirname(script_path), "results_wsr", "confidence")
output <- if (length(args) > 1) args[2] else file.path(input, "llm_wsr_comparison.pdf")
paths <- list.files(input, pattern = "^llm_wsr_seeds_[0-9]+_[0-9]+[.]csv$", full.names = TRUE)
if (!length(paths)) stop("No raw llm_wsr_seeds_* CSV files found in: ", input)
raw <- bind_rows(lapply(paths, read.csv))
scores <- c("loss", "cost_reward")
methods <- c("score_dtm", "score_hete", "score_homo")
metrics <- c("wsr", "fraction_selected", "avg_reward")
metadata <- c("ntrain", "ncalib", "ntest", "n_available", "n_features", "reward", "reward_floor",
              "reward_definition", "loss_definition", "normalization", "shift", "training", "model",
              "model_seed", "trees", "report_scale", "input_sha256",
              "loss_support", "label_count", "raw_loss_step", "loss_normalizer", "clip")
required <- c(metadata, "seed", "alpha", "alpha_normalized", "gamma", "gamma_normalized",
              "wsr_normalized", "score", "method", metrics)
if (!all(required %in% names(raw)) || anyNA(raw[required])) stop("Missing columns or values in raw results.")
# Omit obsolete scores if present in a saved run.
raw <- raw[!raw$score %in% c("cost", "reward"), ]
if (nrow(unique(raw[metadata])) != 1) stop("Different experiment configurations must be plotted separately.")
numeric <- c("alpha", "alpha_normalized", "gamma", "gamma_normalized", "wsr_normalized", metrics)
if (!all(vapply(raw[numeric], function(x) all(is.finite(x)), logical(1)))) stop("Nonfinite levels or metrics.")
if (!setequal(raw$score, scores) || !setequal(raw$method, methods) ||
    !all(raw$normalization == "count") || !all(raw$shift == "exchangeable")) stop("Unexpected score/method/normalization.")
if (anyDuplicated(raw[c("seed", "alpha", "score", "method")])) stop("Overlapping replications found.")
if (any(table(raw$seed) != length(scores) * length(methods) * length(unique(raw$alpha)))) {
  stop("Incomplete alpha/score/boosting comparisons; rerun applications/llm/wsr.py.")
}
if (length(unique(raw$seed)) < 2) stop("At least two replications are needed for error bands.")
if (!all(raw$loss_definition == "clipped_pos_err_plus_half_npos_err") ||
    !all(raw$loss_support == "finite") || !all(raw$label_count == 14) ||
    !all(raw$loss_normalizer == raw$clip & raw$clip > 0) ||
    !all(raw$report_scale == "normalized_clipped")) stop("Expected normalized discrete clipped weighted losses; rerun applications/llm/wsr.py.")
if (any(raw$alpha < 0 | raw$alpha > 1) ||
    any(abs(raw$alpha - raw$alpha_normalized) > 1e-12) ||
    any(abs(raw$wsr - raw$wsr_normalized) > 1e-12) ||
    any(abs(raw$gamma - raw$alpha) > 1e-12) ||
    any(abs(raw$gamma_normalized - raw$alpha_normalized) > 1e-12)) stop("Inconsistent normalized weighted-loss scales.")

long <- bind_rows(lapply(metrics, function(metric) {
  raw %>% transmute(seed, alpha, score, method, metric = metric, value = .data[[metric]])
}))
summary <- long %>% group_by(alpha, score, method, metric) %>%
  summarise(se = sd(value)/sqrt(n()), value = mean(value), .groups = "drop") %>%
  mutate(score = factor(score, levels = scores), method = factor(method, levels = methods),
         metric = factor(metric, levels = metrics, labels = c("WSR", "Fraction selected", "Average reward")))
palette <- c("#607466", "#DDAA00", "#E96056")
method_labels <- c("Deterministic", "Heterogeneous", "Homogeneous")
labels <- c("$\\hat{\\ell}$", "$(\\hat{\\ell}-\\alpha)/\\hat{r}$")
reference <- summary %>% filter(metric == "WSR") %>% distinct(alpha, metric)
comparison <- ggplot(summary, aes(alpha, value, group = interaction(method, score))) +
  theme_bw(base_size = 16) +
  geom_line(data = reference, aes(alpha, alpha), inherit.aes = FALSE,
            linetype = "dotted", colour = "black", linewidth = 0.6) +
  geom_ribbon(aes(ymin = value - 2*se, ymax = value + 2*se, fill = method),
              alpha = 0.15, colour = NA, show.legend = FALSE) +
  geom_line(aes(colour = method, linetype = score), linewidth = 0.75) +
  geom_point(aes(colour = method, shape = score), size = 1.7) +
  facet_wrap(vars(metric), nrow = 1, scales = "free_y") +
  scale_colour_manual(values = palette, labels = method_labels, name = NULL, drop = FALSE) +
  scale_fill_manual(values = palette, guide = "none") +
  scale_linetype_manual(values = c("solid", "dashed"), labels = labels, name = NULL, drop = FALSE) +
  scale_shape_manual(values = c(16, 18), labels = labels, name = NULL, drop = FALSE) +
  guides(colour = guide_legend(nrow = 1, order = 1),
         linetype = guide_legend(nrow = 1, order = 2), shape = guide_legend(nrow = 1, order = 2)) +
  labs(x = "Target level $\\alpha$", y = "Count-normalized") +
  theme(legend.position = "top", legend.box = "horizontal", legend.text = element_text(size = 17),
        legend.key.width = grid::unit(1.8, "lines"), strip.text = element_text(size = 17),
        axis.title = element_text(size = 18), axis.text = element_text(size = 14),
        panel.spacing = grid::unit(1, "lines"), plot.margin = margin(6, 8, 6, 8))

dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
project_root <- file.path(dirname(script_path), "..", "..")
source(file.path(project_root, "plotting", "latex.R"), local = TRUE)
save_latex_plot(list(comparison), output, project_root, width = 10.2, height = 4)
cat("Reward:", unique(raw$reward), "; replications:", length(unique(raw$seed)), "; loss: min(pos_err+0.5*npos_err,K)/K, K=", unique(raw$clip), "\n")
cat("Saved", normalizePath(output, winslash = "/"), "\n")
