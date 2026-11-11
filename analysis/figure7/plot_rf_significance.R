suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(scales))
suppressPackageStartupMessages(library(tidyr))

output_dir <- Sys.getenv(
  "RF_SIGNIF_OUTPUT_DIR",
  unset = file.path(Sys.getenv("HALO_ANALYSIS_ROOT", unset = "."), "downstream/morphology/rf_five_methods_res1_seed707")
)
metrics_path <- file.path(output_dir, "rf_metrics_by_tissue_per_seed.csv")
seed_to_plot <- as.integer(Sys.getenv("RF_SIGNIF_SEED", unset = "707"))
sort_methods_by_metric_median <- tolower(Sys.getenv("RF_SIGNIF_SORT_METHODS", unset = "false")) %in% c("1", "true", "yes")
fixed_significance_y_top <- suppressWarnings(as.numeric(Sys.getenv("RF_SIGNIF_Y_TOP", unset = NA_character_)))
fixed_significance_step <- suppressWarnings(as.numeric(Sys.getenv("RF_SIGNIF_Y_STEP", unset = "0.045")))

method_labels <- c(
  cpsam = "Halo",
  ne = "NE",
  baysor = "Baysor",
  proseg = "Proseg",
  rna2seg = "RNA2seg"
)

method_order <- unname(method_labels)
significance_order <- rev(method_order[-1])

dataset_map <- c(
  human_brain = "Brain",
  human_breast = "Breast",
  human_lung = "Lung",
  human_ovarian = "Ovarian",
  human_pancreas = "Pancreas",
  human_skin = "Skin",
  human_cervical = "Cervical",
  human_prostate = "Prostate",
  human_lymph_node = "Lymph node",
  mouse_colon = "Mouse colon",
  mouse_brain_coronal = "Mouse brain coronal",
  mouse_pup = "Mouse pup"
)

method_fills <- c(
  "Halo" = scales::alpha("#A9D6FF", 0.45),
  "NE" = scales::alpha("#F8C8DC", 0.45),
  "Baysor" = scales::alpha("#F6D55C", 0.45),
  "Proseg" = scales::alpha("#88CC88", 0.45),
  "RNA2seg" = scales::alpha("#C9A0DC", 0.45)
)

p_to_stars <- function(p_value) {
  if (!is.finite(p_value)) {
    return("n.s.")
  }
  if (p_value < 1e-4) {
    return("****")
  }
  if (p_value < 1e-3) {
    return("***")
  }
  if (p_value < 1e-2) {
    return("**")
  }
  if (p_value < 0.05) {
    return("*")
  }
  "n.s."
}

metrics_df <- read_csv(metrics_path, show_col_types = FALSE) |>
  filter(seed == seed_to_plot) |>
  mutate(
    dataset = recode(tissue, !!!dataset_map, .default = tissue),
    method = recode(method, !!!method_labels)
  )

if (sort_methods_by_metric_median) {
  ranked_methods <- metrics_df |>
    filter(method != "Halo") |>
    group_by(method) |>
    summarise(
      score = mean(c(
        median(accuracy, na.rm = TRUE),
        median(macro_f1, na.rm = TRUE),
        median(macro_auc, na.rm = TRUE)
      )),
      .groups = "drop"
    ) |>
    arrange(desc(score)) |>
    pull(method)
  method_order <- c("Halo", ranked_methods)
  significance_order <- rev(ranked_methods)
}

summary_df <- metrics_df |>
  group_by(method) |>
  summarise(
    accuracy = mean(accuracy, na.rm = TRUE),
    macro_f1 = mean(macro_f1, na.rm = TRUE),
    macro_auc = mean(macro_auc, na.rm = TRUE),
    .groups = "drop"
  )

write_csv(
  summary_df,
  file.path(output_dir, sprintf("rf_metrics_res1_5methods_seed%d_summary.csv", seed_to_plot))
)

combined_plot <- metrics_df |>
  select(method, dataset, accuracy, macro_f1, macro_auc) |>
  pivot_longer(
    cols = c(accuracy, macro_f1, macro_auc),
    names_to = "metric",
    values_to = "value"
  ) |>
  mutate(
    metric = recode(
      metric,
      accuracy = "ACC",
      macro_f1 = "Macro-F1",
      macro_auc = "ROC-AUC"
    ),
    metric = factor(metric, levels = c("ACC", "Macro-F1", "ROC-AUC")),
    method = factor(method, levels = method_order, ordered = TRUE)
  )

significance_df <- lapply(levels(combined_plot$metric), function(metric_name) {
  metric_df <- combined_plot |> filter(metric == metric_name)
  y_values <- metric_df$value[is.finite(metric_df$value)]
  y_range <- diff(range(y_values, na.rm = TRUE))
  if (!is.finite(y_range) || y_range == 0) {
    y_range <- 0.05
  }
  if (is.finite(fixed_significance_y_top)) {
    y_top <- fixed_significance_y_top
    step <- fixed_significance_step
  } else {
    y_top <- max(y_values, na.rm = TRUE) + 0.30 * y_range
    step <- 0.075 * y_range
  }

  lapply(seq_along(significance_order), function(i) {
    comparison <- significance_order[[i]]
    paired <- metric_df |>
      filter(method %in% c("Halo", comparison)) |>
      select(dataset, method, value) |>
      pivot_wider(names_from = method, values_from = value) |>
      filter(!is.na(Halo), !is.na(.data[[comparison]]))

    p_value <- if (nrow(paired) >= 2) {
      tryCatch(
        wilcox.test(paired$Halo, paired[[comparison]], paired = TRUE, exact = FALSE)$p.value,
        error = function(e) NA_real_
      )
    } else {
      NA_real_
    }

    y <- y_top - (i - 1) * step
    tibble(
      metric = metric_name,
      comparison = comparison,
      x = 1,
      xend = match(comparison, method_order),
      xmid = (1 + match(comparison, method_order)) / 2,
      y = y,
      p_value = p_value,
      label = p_to_stars(p_value)
    )
  }) |> bind_rows()
}) |> bind_rows() |>
  mutate(metric = factor(metric, levels = levels(combined_plot$metric)))

metrics_plot <- ggplot(combined_plot, aes(x = method, y = value)) +
  geom_line(
    aes(group = dataset),
    color = "grey70",
    linewidth = 0.45,
    alpha = 0.75
  ) +
  geom_boxplot(
    aes(fill = method),
    color = "black",
    outlier.shape = NA,
    width = 0.58,
    linewidth = 0.55
  ) +
  geom_point(
    position = position_jitter(width = 0.08, height = 0),
    color = "grey35",
    size = 1.4,
    alpha = 0.9
  ) +
  geom_segment(
    data = significance_df,
    aes(x = x, xend = xend, y = y, yend = y),
    inherit.aes = FALSE,
    color = "black",
    linewidth = 0.4
  ) +
  geom_text(
    data = significance_df,
    aes(x = xmid, y = y, label = label),
    inherit.aes = FALSE,
    vjust = -0.25,
    size = 2.8
  ) +
  scale_fill_manual(values = method_fills, drop = FALSE) +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.12))) +
  facet_wrap(~metric, nrow = 1, scales = "free_y") +
  labs(x = NULL, y = NULL) +
  theme_classic(base_size = 15) +
  theme(
    strip.background = element_blank(),
    strip.text = element_text(size = 14, face = "bold"),
    legend.position = "none",
    panel.spacing = unit(1.2, "lines"),
    axis.text.x = element_text(angle = 35, hjust = 1)
  )

ggsave(
  filename = file.path(output_dir, sprintf("rf_metrics_res1_all_celltypes_5methods_seed%d_average.png", seed_to_plot)),
  plot = metrics_plot,
  width = 11.5,
  height = 4.35,
  units = "in",
  dpi = 600,
  bg = "white"
)

ggsave(
  filename = file.path(output_dir, sprintf("rf_metrics_res1_all_celltypes_5methods_seed%d_average.pdf", seed_to_plot)),
  plot = metrics_plot,
  width = 11.5,
  height = 4.35,
  units = "in",
  bg = "white"
)

write_csv(
  significance_df,
  file.path(output_dir, sprintf("rf_metrics_res1_5methods_seed%d_halo_paired_stats.csv", seed_to_plot))
)
