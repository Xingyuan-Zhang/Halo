suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(purrr))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(uwot))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(ggpubr))
suppressPackageStartupMessages(library(cowplot))

project_root <- normalizePath(Sys.getenv("HALO_ANALYSIS_ROOT", unset = "."), mustWork = FALSE)
morphology_root <- file.path(project_root, "downstream/morphology/morphology_features")
celltype_root_original <- file.path(project_root, "downstream/cluster/celltype_scanpy_res1")
celltype_root_projected <- file.path(project_root, "downstream/morphology/celltype_groundtruth_projected")
celltype_root_all_methods <- Sys.getenv("CELLTYPE_ROOT_ALL_METHODS", unset = "")
output_dir <- Sys.getenv(
  "MORPH_UMAP_OUTPUT_DIR",
  unset = file.path(project_root, "downstream/morphology/morphology_umap_six_methods_res1")
)
n_threads <- max(1L, as.integer(Sys.getenv("SLURM_CPUS_PER_TASK", "4")))

tissues <- c(
  "human_brain",
  "human_breast",
  "human_lung",
  "human_ovarian",
  "human_pancreas",
  "human_skin",
  "human_cervical",
  "human_prostate",
  "human_lymph_node",
  "mouse_colon",
  "mouse_brain_coronal",
  "mouse_pup"
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

dataset_levels <- c(
  "glioblastoma",
  "breast cancer",
  "lung adenocarcinoma",
  "ovarian cancer",
  "melanoma",
  "pancreatic cancer",
  "cervical cancer",
  "prostate cancer",
  "lymph node",
  "mouse brain coronal",
  "mouse colon",
  "mouse pup"
)

methods <- c("tenx", "cpsam", "ne", "baysor", "proseg", "rna2seg")
projected_methods <- c("baysor", "proseg", "rna2seg")
method_labels <- c(
  "tenx" = "Ground truth",
  "cpsam" = "Halo",
  "ne" = "Nuclear expansion",
  "baysor" = "Baysor",
  "proseg" = "Proseg",
  "rna2seg" = "RNA2seg"
)

feature_cols <- c(
  "area",
  "perimeter",
  "Feret",
  "eccentricity",
  "roundness",
  "circularity",
  "solidity",
  "AR"
)

celltype_levels <- c(
  "Cancer cells",
  "Endothelial cells",
  "Epithelial cells",
  "Fibroblasts",
  "Smooth muscle cells",
  "Myeloid cells",
  "T cells",
  "B cells"
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

celltype_palette <- c(
  "Cancer cells" = "#D55E00",
  "Endothelial cells" = "#004B87",
  "Epithelial cells" = "#00A6D6",
  "Fibroblasts" = "#009E73",
  "Smooth muscle cells" = "#CC79A7",
  "Myeloid cells" = "#7A52A3",
  "T cells" = "#8B1E3F",
  "B cells" = "#8A8F00"
)

dataset_shapes <- c(
  "glioblastoma" = 16,
  "breast cancer" = 15,
  "lung adenocarcinoma" = 17,
  "ovarian cancer" = 18,
  "melanoma" = 8,
  "pancreatic cancer" = 3,
  "cervical cancer" = 7,
  "prostate cancer" = 4,
  "lymph node" = 9,
  "mouse brain coronal" = 10,
  "mouse colon" = 12,
  "mouse pup" = 13
)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

read_morphology_file <- function(method_name, tissue) {
  morphology_file <- file.path(morphology_root, method_name, paste0(tissue, ".rds"))
  morphology_csv <- file.path(morphology_root, method_name, paste0(tissue, ".csv"))

  if (file.exists(morphology_file)) {
    readRDS(morphology_file)
  } else if (file.exists(morphology_csv)) {
    readr::read_csv(morphology_csv, show_col_types = FALSE)
  } else {
    stop(sprintf("Morphology file not found: %s or %s", morphology_file, morphology_csv))
  }
}

read_celltype_file <- function(method_name, tissue) {
  celltype_root <- if (nzchar(celltype_root_all_methods)) {
    celltype_root_all_methods
  } else if (method_name %in% projected_methods) {
    celltype_root_projected
  } else {
    celltype_root_original
  }
  celltype_file <- file.path(celltype_root, method_name, paste0(tissue, ".csv"))

  if (!file.exists(celltype_file)) {
    stop(sprintf("Cell-type file not found: %s", celltype_file))
  }

  readr::read_csv(celltype_file, show_col_types = FALSE) |>
    dplyr::rename(cell_id = cellid)
}

read_method_data <- function(method_name) {
  purrr::map_dfr(tissues, function(tissue) {
    morph <- read_morphology_file(method_name, tissue)
    ct <- read_celltype_file(method_name, tissue)

    missing_features <- setdiff(c("cell_id", feature_cols), colnames(morph))
    if (length(missing_features) > 0) {
      stop(sprintf(
        "%s/%s is missing column(s): %s",
        method_name,
        tissue,
        paste(missing_features, collapse = ", ")
      ))
    }

    merged <- morph |>
      dplyr::inner_join(ct, by = "cell_id") |>
      dplyr::filter(area != 0) |>
      dplyr::mutate(
        tissue = tissue,
        method = method_name
      )

    message(sprintf(
      "[%s] %s: morphology=%d, annotation=%d, merged=%d",
      method_name,
      tissue,
      nrow(morph),
      nrow(ct),
      nrow(merged)
    ))

    merged
  })
}

run_method_umap <- function(method_name, seed = 42) {
  combined_morphology <- read_method_data(method_name)

  combined <- combined_morphology |>
    dplyr::filter(celltype %in% c(celltype_levels, tumor_labels)) |>
    dplyr::mutate(
      celltype = dplyr::if_else(celltype %in% tumor_labels, "Cancer cells", celltype),
      celltype = factor(celltype, levels = celltype_levels),
      dataset = factor(unname(tissue_to_dataset[tissue]), levels = dataset_levels)
    ) |>
    dplyr::filter(if_all(all_of(feature_cols), is.finite))

  features <- combined |>
    dplyr::select(all_of(feature_cols))
  X <- scale(as.matrix(features))

  message(sprintf("[UMAP] %s: fitting %d cells", method_name, nrow(X)))
  set.seed(seed)
  um <- uwot::umap(
    X,
    n_neighbors = 30,
    min_dist = 0.01,
    metric = "euclidean",
    init = "random",
    n_threads = n_threads,
    verbose = TRUE
  )

  combined_umap_df <- data.frame(
    UMAP1 = um[, 1],
    UMAP2 = um[, 2],
    celltype = combined$celltype,
    dataset = combined$dataset
  )

  avg_umap <- combined_umap_df |>
    dplyr::group_by(dataset, celltype) |>
    dplyr::summarise(
      UMAP1 = mean(UMAP1, na.rm = TRUE),
      UMAP2 = mean(UMAP2, na.rm = TRUE),
      n = dplyr::n(),
      .groups = "drop"
    ) |>
    dplyr::mutate(
      dataset = factor(as.character(dataset), levels = dataset_levels),
      celltype = factor(as.character(celltype), levels = celltype_levels)
    )

  readr::write_csv(
    avg_umap,
    file.path(output_dir, paste0(method_name, "_umap_centroids.csv"))
  )

  avg_umap
}

make_centroid_plot <- function(centroids, title_text, show_legend = FALSE) {
  ggplot(
    centroids,
    aes(
      UMAP1,
      UMAP2,
      color = celltype,
      shape = dataset
    )
  ) +
    geom_point(size = 3.4, stroke = 1.2, alpha = 0.95) +
    scale_color_manual(values = celltype_palette, drop = FALSE) +
    scale_shape_manual(values = dataset_shapes, drop = FALSE) +
    guides(
      color = guide_legend(
        title = "cell type",
        override.aes = list(size = 4, alpha = 1, stroke = 1.2),
        ncol = 1,
        byrow = FALSE
      ),
      shape = guide_legend(
        title = "dataset",
        ncol = 1,
        byrow = FALSE,
        override.aes = list(size = 4, alpha = 1, stroke = 1.2, color = "black")
      )
    ) +
    coord_equal() +
    theme_classic() +
    theme(
      aspect.ratio = 1,
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      legend.position = if (show_legend) "bottom" else "none",
      legend.box = "horizontal",
      legend.background = element_rect(fill = "white", color = NA),
      legend.title = element_text(size = 14, face = "bold"),
      legend.text = element_text(size = 12),
      legend.key.size = unit(0.7, "lines"),
      legend.margin = margin(3, 3, 3, 3),
      plot.margin = margin(5.5, 5.5, 5.5, 5.5)
    ) +
    labs(title = title_text, color = "cell type", shape = "dataset")
}

centroid_list <- lapply(methods, run_method_umap)
names(centroid_list) <- methods

all_centroids <- bind_rows(
  lapply(methods, function(method_name) {
    centroid_list[[method_name]] |>
      mutate(method = method_name, .before = 1)
  })
)
readr::write_csv(all_centroids, file.path(output_dir, "six_methods_umap_centroids.csv"))

plots <- lapply(methods, function(method_name) {
  make_centroid_plot(centroid_list[[method_name]], unname(method_labels[method_name]))
})

legend_g <- ggpubr::get_legend(
  make_centroid_plot(centroid_list[["tenx"]], unname(method_labels["tenx"]), show_legend = TRUE)
)

g_panels <- ggpubr::ggarrange(
  plotlist = plots,
  ncol = 3,
  nrow = 2,
  align = "hv"
)

g_combined <- cowplot::plot_grid(
  g_panels,
  legend_g,
  ncol = 1,
  rel_heights = c(1, 0.22)
)

ggsave(
  file.path(output_dir, "morphology_umap_res1_6methods_panels.png"),
  g_panels,
  width = 12,
  height = 8,
  dpi = 300,
  bg = "white"
)

ggsave(
  file.path(output_dir, "morphology_umap_res1_6methods_panels.pdf"),
  g_panels,
  width = 12,
  height = 8,
  bg = "white"
)

ggsave(
  file.path(output_dir, "morphology_umap_res1_6methods_legend.png"),
  legend_g,
  width = 12,
  height = 5,
  dpi = 300,
  bg = "white"
)

ggsave(
  file.path(output_dir, "morphology_umap_res1_6methods_combined.png"),
  g_combined,
  width = 12,
  height = 11,
  dpi = 300,
  bg = "white"
)

ggsave(
  file.path(output_dir, "morphology_umap_res1_6methods_combined.pdf"),
  g_combined,
  width = 12,
  height = 11,
  bg = "white"
)

message(sprintf("[COMPLETE] UMAP results written to %s", output_dir))
