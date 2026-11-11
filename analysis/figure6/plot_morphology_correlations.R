suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(scales))

output_dir <- Sys.getenv(
  "MORPH_CORRELATION_OUTPUT_DIR",
  unset = file.path(Sys.getenv("HALO_ANALYSIS_ROOT", unset = "."), "downstream/morphology/morphology_median_correlation_res1")
)
plot_dir <- file.path(output_dir, "individual_feature_barplots")
correlation_path <- file.path(output_dir, "morphology_feature_median_pearson_to_groundtruth.csv")

dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

method_fills <- c(
  "Halo" = "#5DADE2",
  "NE" = "#D98CB3",
  "Baysor" = "#D4AC0D",
  "Proseg" = "#58A65C",
  "RNA2seg" = "#9B59B6"
)

metric_levels <- c(
  "area",
  "Feret diameter",
  "eccentricity",
  "roundness",
  "circularity",
  "solidity",
  "aspect ratio"
)

safe_filename <- function(x) {
  x <- tolower(x)
  x <- gsub("[^a-z0-9]+", "_", x)
  gsub("^_|_$", "", x)
}

correlation_df <- readr::read_csv(correlation_path, show_col_types = FALSE) |>
  mutate(
    metric = factor(metric, levels = metric_levels, ordered = TRUE),
    method_label = as.character(method_label)
  )

sorted_df <- correlation_df |>
  arrange(metric, desc(pearson_r), method_label)

readr::write_csv(
  sorted_df,
  file.path(plot_dir, "morphology_feature_median_pearson_to_groundtruth_sorted_for_individual_barplots.csv")
)

for (metric_name in metric_levels) {
  metric_df <- correlation_df |>
    filter(as.character(metric) == metric_name) |>
    arrange(pearson_r) |>
    mutate(method_ordered = factor(method_label, levels = method_label, ordered = TRUE))

  x_min <- min(metric_df$pearson_r, na.rm = TRUE)
  x_max <- max(metric_df$pearson_r, na.rm = TRUE)
  x_lower <- if (x_min < 0) max(-1, floor((x_min - 0.05) * 10) / 10) else 0
  x_upper <- min(1, ceiling((x_max + 0.05) * 10) / 10)

  p <- ggplot(metric_df, aes(x = pearson_r, y = method_ordered, fill = method_label)) +
    geom_col(
      width = 0.68,
      color = "black",
      linewidth = 0.3
    ) +
    geom_vline(xintercept = 0, color = "grey35", linewidth = 0.45) +
    scale_fill_manual(values = method_fills, drop = FALSE, guide = "none") +
    scale_x_continuous(
      limits = c(x_lower, x_upper),
      breaks = scales::pretty_breaks(n = 5),
      expand = expansion(mult = c(0.02, 0.04))
    ) +
    labs(
      x = "PCC",
      y = NULL
    ) +
    theme_classic(base_size = 15) +
    theme(
      axis.text.y = element_text(size = 13),
      axis.text.x = element_text(size = 12),
      axis.title.x = element_text(size = 13),
      legend.position = "none",
      plot.margin = margin(8, 16, 8, 8)
    )

  filename_base <- paste0("morphology_feature_median_pearson_", safe_filename(metric_name), "_horizontal_barplot")

  ggsave(
    file.path(plot_dir, paste0(filename_base, ".png")),
    p,
    width = 4.8,
    height = 4.8,
    dpi = 600,
    bg = "white"
  )

  ggsave(
    file.path(plot_dir, paste0(filename_base, ".pdf")),
    p,
    width = 4.8,
    height = 4.8,
    bg = "white"
  )
}

message(sprintf("[DONE] Wrote individual horizontal barplots to %s", plot_dir))
