# Shared LaTeX output for simulation and application figures. Formulas pass verbatim to
# TeX; lmodern/T1 match tex/main.tex. Text remains sans serif, math uses the
# manuscript's LaTeX math fonts. A standalone .tex accompanies each PDF.
# Dependencies: tikzDevice, gridExtra, and pdflatex with TikZ/preview/lmodern.

save_latex_plot <- function(plots, output, project_root, width, height,
                            row_heights = rep(1, length(plots))) {
  if (!tolower(tools::file_ext(output)) %in% c("pdf", "tex")) {
    stop("Use a .pdf or .tex output path for LaTeX figures.")
  }
  if (!length(plots) || length(row_heights) != length(plots) ||
      any(!is.finite(row_heights) | row_heights <= 0)) {
    stop("Provide one positive row height per plot.")
  }
  local_library <- file.path(project_root, ".R-library")
  if (dir.exists(local_library)) .libPaths(c(normalizePath(local_library), .libPaths()))
  if (!requireNamespace("tikzDevice", quietly = TRUE)) {
    stop("True LaTeX output requires tikzDevice. Install it with install.packages('tikzDevice').")
  }
  compiler <- Sys.which("pdflatex")
  if (!nzchar(compiler)) stop("True LaTeX output requires pdflatex on PATH (MiKTeX or TeX Live).")
  old_options <- options(tikzDocumentDeclaration = "\\documentclass[12pt]{article}",
                         tikzLatexPackages = c(getOption("tikzLatexPackages"),
                           "\\usepackage[T1]{fontenc}", "\\usepackage{lmodern}",
                           "\\usepackage{amsmath}", "\\renewcommand{\\familydefault}{\\sfdefault}"))
  on.exit(options(old_options), add = TRUE)
  # Reuse TeX font measurements across figures when a project library exists.
  if (dir.exists(local_library)) {
    previous_cache <- options(tikzMetricsDictionary = file.path(normalizePath(local_library, winslash = "/"),
                                                                "tikz-metrics-lmodern-12pt"))
    on.exit(options(previous_cache), add = TRUE)
  }
  output <- normalizePath(output, winslash = "/", mustWork = FALSE)
  work <- tempfile("score-latex-")
  dir.create(work)
  tex <- file.path(work, "figure.tex")
  tikzDevice::tikz(tex, width = width, height = height, pointsize = 12,
                   standAlone = TRUE, sanitize = FALSE, engine = "pdftex", bg = "white")
  device <- grDevices::dev.cur()
  on.exit(if (device %in% grDevices::dev.list()) grDevices::dev.off(device), add = TRUE)

  # Build grobs on the actual TikZ device so legend widths use TeX metrics.
  # ggarrange/cowplot otherwise measure raw TeX strings on a temporary PDF device.
  with_legend <- ggplot2::ggplotGrob(plots[[1]])
  legend <- with_legend$grobs[[which(with_legend$layout$name == "guide-box-top")]]
  rows <- lapply(plots, function(plot) ggplot2::ggplotGrob(plot + ggplot2::theme(legend.position = "none")))
  widths <- Reduce(grid::unit.pmax, lapply(rows, function(row) row$widths))
  rows <- lapply(rows, function(row) { row$widths <- widths; row })
  gridExtra::grid.arrange(grobs = c(list(legend), rows), nrow = length(rows) + 1,
                          heights = grid::unit.c(sum(legend$heights) + grid::unit(8, "pt"),
                                                 grid::unit(row_heights, "null")))
  grDevices::dev.off(device)
  tex_output <- paste0(tools::file_path_sans_ext(output), ".tex")
  if (tolower(tools::file_ext(output)) == "pdf") {
    log <- system2(compiler, c("-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape",
                               paste0("-output-directory=", shQuote(work)), shQuote(tex)),
                    stdout = TRUE, stderr = TRUE)
    status <- attr(log, "status")
    compiled <- file.path(work, "figure.pdf")
    if ((!is.null(status) && status != 0) || !file.exists(compiled)) {
      stop("LaTeX compilation failed:\n", paste(tail(log, 25), collapse = "\n"))
    }
    if (!file.copy(compiled, output, overwrite = TRUE)) stop("Cannot write PDF: ", output)
  }
  if (!file.copy(tex, tex_output, overwrite = TRUE)) stop("Cannot write LaTeX source: ", tex_output)
  cat("LaTeX source:", tex_output, "\n")
}
