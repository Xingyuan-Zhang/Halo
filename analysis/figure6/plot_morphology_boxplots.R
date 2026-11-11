suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tidyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(scales))
suppressPackageStartupMessages(library(purrr))

project_root <- normalizePath(Sys.getenv("HALO_ANALYSIS_ROOT", unset = "."), mustWork = FALSE)
morphology_root <- file.path(project_root, "downstream/morphology/morphology_features")
celltype_root_original <- file.path(project_root, "downstream/morphology/celltype")
celltype_root_projected <- file.path(project_root, "downstream/morphology/celltype_groundtruth_projected")
celltype_root_all_methods <- Sys.getenv("CELLTYPE_ROOT_ALL_METHODS", unset = "")
output_dir <- Sys.getenv(
  "MORPH_BOXPLOT_OUTPUT_DIR",
  unset = file.path(project_root, "downstream/morphology/morphology_boxplot_six_methods_res1")
)

tissue_to_dataset <- c(
  "human_brain" = "glioblastoma",
  "human_breast" = "breast cancer",
  "human_lung" = "lung adenocarcinoma",
  "human_ovarian" = "ovarian cancer",
  "human_skin" = "melanoma",
  "human_pancreas" = "pancreatic cancer",
  "human_cervical" = "cervical cancer",
  "human_prostate" = "prostate cancer",
  "human_lymph_node" = "lymph node",
  "mouse_brain_coronal" = "mouse brain coronal",
  "mouse_colon" = "mouse colon",
  "mouse_pup" = "mouse pup"
)

methods <- c("tenx", "cpsam", "ne", "baysor", "proseg", "rna2seg")
projected_methods <- c("baysor", "proseg", "rna2seg")
method_labels <- c(
  "tenx" = "Ground truth",
  "cpsam" = "Halo",
  "ne" = "NE",
  "baysor" = "Baysor",
  "proseg" = "Proseg",
  "rna2seg" = "RNA2seg"
)

metrics <- c("area", "Feret", "eccentricity", "roundness", "circularity", "solidity", "AR")
metric_labels <- c(
  "area",
  "Feret diameter",
  "eccentricity",
  "roundness",
  "circularity",
  "solidity",
  "aspect ratio"
)
metric_display <- c(
  "area" = "area",
  "Feret" = "Feret diameter",
  "eccentricity" = "eccentricity",
  "roundness" = "roundness",
  "circularity" = "circularity",
  "solidity" = "solidity",
  "AR" = "aspect ratio"
)

tumor_labels <- c(
  "Brain cancer cells",
  "Breast cancer cells",
  "Ovarian cancer cells",
  "Lung cancer cells",
  "Pancreas cancer cells",
  "Skin cancer cells",
  "Prostate cancer cells",
  "Cervical cancer cells",
  "Proliferative Tumor Cells",
  "MT-High, Jun+/Fos+ Tumor Cells",
  "VEGFA+ Tumor Cells",
  "Tumor Cells",
  "Inflammatory Tumor Cells",
  "Malignant Cells Lining Cyst"
)

macro_colors <- c(
  "Cancer cells" = "#E67E22",
  "Endothelial cells" = "#003B75",
  "Epithelial cells" = "#00A6D6",
  "Fibroblasts" = "#009E73",
  "Smooth muscle cells" = "#D98CB3",
  "Myeloid cells" = "#6A3D9A",
  "T cells" = "#8B1E3F",
  "B cells" = "#9A8F00"
)

read_morphology_file <- function(method_name, tissue) {
  morph_file <- file.path(morphology_root, method_name, paste0(tissue, ".rds"))
  morph_csv <- file.path(morphology_root, method_name, paste0(tissue, ".csv"))

  if (file.exists(morph_file)) {
    readRDS(morph_file)
  } else if (file.exists(morph_csv)) {
    readr::read_csv(morph_csv, show_col_types = FALSE)
  } else {
    stop(sprintf("Missing morphology file: %s or %s", morph_file, morph_csv))
  }
}

read_celltype_file <- function(method_name, tissue) {
  root <- if (nzchar(celltype_root_all_methods)) {
    celltype_root_all_methods
  } else if (method_name %in% projected_methods) {
    celltype_root_projected
  } else {
    celltype_root_original
  }
  ct_file <- file.path(root, method_name, paste0(tissue, ".csv"))

  if (!file.exists(ct_file)) {
    stop(sprintf("Missing celltype file: %s", ct_file))
  }

  readr::read_csv(ct_file, show_col_types = FALSE) |>
    dplyr::select(cellid, celltype)
}

read_method_data <- function(method_name) {
  purrr::map_dfr(names(tissue_to_dataset), function(tissue) {
    morph <- read_morphology_file(method_name, tissue)
    ct <- read_celltype_file(method_name, tissue)

    missing_cols <- setdiff(c("cell_id", metrics), colnames(morph))
    if (length(missing_cols) > 0) {
      stop(sprintf("[%s/%s] missing columns: %s", method_name, tissue, paste(missing_cols, collapse = ", ")))
    }

    merged <- morph |>
      dplyr::inner_join(ct, by = c("cell_id" = "cellid")) |>
      dplyr::filter(area != 0) |>
      dplyr::mutate(
        dataset = unname(tissue_to_dataset[tissue]),
        tissue = tissue,
        method = method_name
      )

    message(sprintf(
      "[%s] %s: morphology=%d, celltype=%d, merged=%d",
      method_name,
      tissue,
      nrow(morph),
      nrow(ct),
      nrow(merged)
    ))

    merged
  })
}

all_methods_raw <- bind_rows(lapply(methods, read_method_data))

all_methods_avg <- all_methods_raw |>
  mutate(
    macro_celltype = ifelse(celltype %in% tumor_labels, "Cancer cells", celltype)
  ) |>
  filter(macro_celltype %in% names(macro_colors)) |>
  group_by(method, dataset, macro_celltype) |>
  summarise(
    across(all_of(metrics), ~ mean(.x, na.rm = TRUE)),
    n_cells = dplyr::n(),
    .groups = "drop"
  )

all_methods_long <- all_methods_avg |>
  pivot_longer(
    cols = all_of(metrics),
    names_to = "metric",
    values_to = "value"
  ) |>
  filter(is.finite(value)) |>
  mutate(
    metric = recode(metric, !!!metric_display),
    method = factor(method, levels = names(method_labels), labels = unname(method_labels), ordered = TRUE),
    macro_celltype = factor(macro_celltype, levels = names(macro_colors), ordered = TRUE)
  )

make_one_plot <- function(metric_name) {
  df_one <- all_methods_long |> filter(metric == metric_name)

  order_levels <- df_one |>
    filter(method == "Ground truth") |>
    group_by(macro_celltype) |>
    summarise(med = median(value, na.rm = TRUE), .groups = "drop") |>
    arrange(med) |>
    pull(macro_celltype) |>
    as.character()

  df_one <- df_one |>
    mutate(macro_celltype = factor(macro_celltype, levels = order_levels, ordered = TRUE))

  ggplot(
    df_one,
    aes(
      x = macro_celltype,
      y = value,
      fill = macro_celltype,
      color = macro_celltype
    )
    ) +
    geom_boxplot(outlier.shape = NA, linewidth = 0.45, width = 0.65) +
    geom_point(
      color = "grey25",
      alpha = 0.45,
      size = 0.9,
      position = position_jitter(width = 0.15, height = 0)
    ) +
    scale_y_continuous(expand = expansion(mult = c(0.06, 0.16))) +
    coord_flip() +
    facet_grid(. ~ method, drop = FALSE) +
    scale_fill_manual(values = alpha(macro_colors, 0.6), drop = FALSE) +
    scale_color_manual(values = macro_colors, drop = FALSE) +
    theme_classic(base_size = 12) +
    labs(title = metric_name, x = "", y = "") +
    theme(
      legend.position = "none",
      axis.text.x = element_text(size = 12),
      axis.text.y = element_text(size = 12),
      axis.title.x = element_text(size = 14),
      axis.title.y = element_text(size = 14),
      plot.title = element_text(face = "bold", size = 16),
      strip.text.x = element_text(face = "bold", size = 14, margin = margin(b = 6)),
      strip.background = element_blank(),
      panel.spacing.x = unit(1.15, "lines"),
      plot.margin = margin(10, 28, 10, 16)
    )
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
readr::write_csv(all_methods_avg, file.path(output_dir, "morphology_feature_dataset_macrocelltype_means.csv"))
readr::write_csv(all_methods_long, file.path(output_dir, "morphology_feature_long.csv"))

for (m in metric_labels) {
  fname <- gsub(" ", "_", m)
  plot <- make_one_plot(m)

  ggsave(
    filename = file.path(output_dir, paste0("boxplot_", fname, "_6methods.png")),
    plot = plot,
    width = 24,
    height = 5.8,
    dpi = 300,
    bg = "white"
  )
  ggsave(
    filename = file.path(output_dir, paste0("boxplot_", fname, "_6methods.pdf")),
    plot = plot,
    width = 24,
    height = 5.8,
    bg = "white"
  )
}

message(sprintf("[DONE] Wrote six-method plots to %s", output_dir))
