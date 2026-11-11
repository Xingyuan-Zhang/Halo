suppressPackageStartupMessages(library(randomForest))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(tidyr))
suppressPackageStartupMessages(library(ggplot2))
suppressPackageStartupMessages(library(purrr))
suppressPackageStartupMessages(library(tibble))
suppressPackageStartupMessages(library(readr))
suppressPackageStartupMessages(library(pROC))
suppressPackageStartupMessages(library(mltools))
suppressPackageStartupMessages(library(scales))

project_root <- normalizePath(Sys.getenv("HALO_ANALYSIS_ROOT", unset = "."), mustWork = FALSE)
morphology_root <- file.path(project_root, "downstream/morphology/morphology_features")
celltype_root_original <- file.path(project_root, "downstream/cluster/celltype_scanpy_res1")
celltype_root_projected <- file.path(project_root, "downstream/morphology/celltype_groundtruth_projected")
output_dir <- Sys.getenv(
  "MORPH_RF_OUTPUT_DIR",
  unset = file.path(project_root, "downstream/morphology/rf_five_methods_res1_5seeds")
)
run_label <- Sys.getenv("MORPH_RF_RUN_LABEL", unset = "")

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

dataset_map <- c(
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

methods <- c("cpsam", "ne", "baysor", "proseg", "rna2seg")
projected_methods <- c("baysor", "proseg", "rna2seg")
method_labels <- c(
  "cpsam" = "Halo",
  "ne" = "NE",
  "baysor" = "Baysor",
  "proseg" = "Proseg",
  "rna2seg" = "RNA2seg"
)

seeds_env <- Sys.getenv("MORPH_RF_SEEDS", unset = "")
seeds <- if (nzchar(seeds_env)) {
  as.integer(strsplit(seeds_env, "[,[:space:]]+")[[1]])
} else {
  c(101, 202, 303, 404, 505)
}
seeds <- seeds[!is.na(seeds)]
if (!length(seeds)) {
  stop("No valid MORPH_RF_SEEDS values were provided.")
}
if (!nzchar(run_label)) {
  run_label <- paste0(length(seeds), "seed")
}

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

unknown_levels <- c(
  "Unknown",
  "unknown",
  "UNK",
  "Unassigned",
  "unassigned"
)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

stratified_split <- function(df, y, test_frac = 0.2, seed = 42) {
  set.seed(seed)

  idx_test <- unlist(
    lapply(split(seq_len(nrow(df)), df[[y]]), function(ix) {
      n_test <- min(
        max(1L, floor(length(ix) * test_frac)),
        length(ix) - 1L
      )
      sample(ix, n_test)
    }),
    use.names = FALSE
  )

  list(
    train = df[-idx_test, , drop = FALSE],
    test = df[idx_test, , drop = FALSE]
  )
}

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
  celltype_root <- if (method_name %in% projected_methods) {
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
    morphology <- read_morphology_file(method_name, tissue)
    celltype <- read_celltype_file(method_name, tissue)

    missing_features <- setdiff(c("cell_id", feature_cols), colnames(morphology))
    if (length(missing_features) > 0) {
      stop(sprintf(
        "%s/%s is missing column(s): %s",
        method_name,
        tissue,
        paste(missing_features, collapse = ", ")
      ))
    }

    merged <- morphology |>
      inner_join(celltype, by = "cell_id") |>
      filter(area != 0) |>
      mutate(tissue = tissue, method = method_name)

    message(sprintf(
      "[%s] %s: morphology=%d, annotation=%d, merged=%d",
      method_name,
      tissue,
      nrow(morphology),
      nrow(celltype),
      nrow(merged)
    ))

    merged
  })
}

train_rf_celltype <- function(
  dat,
  ntree = 500,
  mtry = NULL,
  nodesize = 1,
  test_frac = 0.2,
  seed = 42,
  use_class_weights = TRUE
) {
  dat <- dat |>
    filter(!is.na(celltype)) |>
    filter(!celltype %in% unknown_levels) |>
    filter(if_all(all_of(feature_cols), is.finite))

  class_counts <- table(dat$celltype)
  usable_classes <- names(class_counts[class_counts >= 2])
  dat <- dat |> filter(celltype %in% usable_classes)

  if (nrow(dat) == 0 || length(usable_classes) < 2) {
    stop("Fewer than two usable cell types remain after filtering.")
  }

  dat$celltype <- droplevels(factor(dat$celltype))
  model_df <- dat |> select(all_of(feature_cols), celltype)

  split <- stratified_split(model_df, y = "celltype", test_frac = test_frac, seed = seed)
  train_df <- split$train
  test_df <- split$test

  train_counts <- table(train_df$celltype)
  classwt <- NULL
  if (use_class_weights) {
    classwt <- as.numeric(mean(train_counts) / train_counts)
    names(classwt) <- names(train_counts)
  }

  if (is.null(mtry)) {
    mtry <- max(1L, floor(sqrt(length(feature_cols))))
  }

  set.seed(seed)
  rf <- randomForest(
    celltype ~ .,
    data = train_df,
    ntree = ntree,
    mtry = mtry,
    nodesize = nodesize,
    importance = TRUE,
    classwt = classwt,
    keep.forest = TRUE
  )

  pred_lab <- predict(rf, newdata = test_df, type = "response")
  pred_prob <- predict(rf, newdata = test_df, type = "prob")
  truth <- factor(test_df$celltype, levels = levels(train_df$celltype))

  accuracy <- mean(pred_lab == truth)

  per_class <- lapply(levels(truth), function(cell_type) {
    tp <- sum(pred_lab == cell_type & truth == cell_type)
    fp <- sum(pred_lab == cell_type & truth != cell_type)
    fn <- sum(pred_lab != cell_type & truth == cell_type)

    precision <- if ((tp + fp) > 0) tp / (tp + fp) else NA_real_
    recall <- if ((tp + fn) > 0) tp / (tp + fn) else NA_real_
    f1 <- if (is.na(precision) || is.na(recall) || (precision + recall) == 0) {
      NA_real_
    } else {
      2 * precision * recall / (precision + recall)
    }

    tibble(
      celltype = cell_type,
      precision = precision,
      recall = recall,
      f1 = f1,
      test_cells = sum(truth == cell_type)
    )
  }) |> bind_rows()

  macro_f1 <- mean(per_class$f1, na.rm = TRUE)
  mcc <- mltools::mcc(preds = as.character(pred_lab), actuals = as.character(truth))

  probability_df <- as.data.frame(pred_prob)
  missing_probability_cols <- setdiff(levels(truth), colnames(probability_df))
  if (length(missing_probability_cols) > 0) {
    for (cell_type in missing_probability_cols) {
      probability_df[[cell_type]] <- 0
    }
  }
  probability_df <- probability_df[, levels(truth), drop = FALSE]

  auc_per_class <- sapply(levels(truth), function(cell_type) {
    truth_binary <- as.integer(truth == cell_type)
    if (length(unique(truth_binary)) < 2) {
      return(NA_real_)
    }

    roc_result <- tryCatch(
      pROC::roc(
        response = truth_binary,
        predictor = probability_df[[cell_type]],
        quiet = TRUE,
        direction = "<"
      ),
      error = function(e) NULL
    )

    if (is.null(roc_result)) {
      NA_real_
    } else {
      as.numeric(pROC::auc(roc_result))
    }
  })

  importance_df <- as.data.frame(importance(rf, type = 2))
  importance_df$feature <- rownames(importance_df)
  rownames(importance_df) <- NULL
  names(importance_df)[names(importance_df) == "MeanDecreaseGini"] <- "importance"

  list(
    accuracy = accuracy,
    macro_f1 = macro_f1,
    mcc = as.numeric(mcc),
    macro_auc = mean(auc_per_class, na.rm = TRUE),
    per_class = per_class,
    importance = importance_df,
    n_cells = nrow(dat),
    n_train = nrow(train_df),
    n_test = nrow(test_df),
    n_celltypes = nlevels(dat$celltype)
  )
}

run_method <- function(method_name) {
  method_data <- read_method_data(method_name)

  results <- lapply(seeds, function(seed_value) {
    lapply(tissues, function(tissue_name) {
      message(sprintf("[RF] method=%s seed=%s tissue=%s", method_name, seed_value, tissue_name))

      fit <- train_rf_celltype(
        dat = method_data |> filter(tissue == tissue_name),
        ntree = 500,
        mtry = NULL,
        nodesize = 1,
        test_frac = 0.2,
        seed = seed_value,
        use_class_weights = TRUE
      )

      metrics <- tibble(
        method = method_name,
        seed = seed_value,
        tissue = tissue_name,
        n_cells = fit$n_cells,
        n_train = fit$n_train,
        n_test = fit$n_test,
        n_celltypes = fit$n_celltypes,
        accuracy = fit$accuracy,
        macro_f1 = fit$macro_f1,
        mcc = fit$mcc,
        macro_auc = fit$macro_auc
      )

      per_class <- fit$per_class |>
        mutate(method = method_name, seed = seed_value, tissue = tissue_name, .before = 1)

      importance <- fit$importance |>
        mutate(method = method_name, seed = seed_value, tissue = tissue_name, .before = 1)

      list(metrics = metrics, per_class = per_class, importance = importance)
    })
  }) |> unlist(recursive = FALSE)

  list(
    metrics = bind_rows(lapply(results, `[[`, "metrics")),
    per_class = bind_rows(lapply(results, `[[`, "per_class")),
    importance = bind_rows(lapply(results, `[[`, "importance"))
  )
}

all_results <- lapply(methods, run_method)

metrics_seed_df <- bind_rows(lapply(all_results, `[[`, "metrics"))
per_class_seed_df <- bind_rows(lapply(all_results, `[[`, "per_class"))
importance_seed_df <- bind_rows(lapply(all_results, `[[`, "importance"))

metrics_avg_df <- metrics_seed_df |>
  group_by(method, tissue) |>
  summarise(
    n_seeds = n_distinct(seed),
    across(c(n_cells, n_train, n_test, n_celltypes), ~ mean(.x, na.rm = TRUE)),
    across(c(accuracy, macro_f1, mcc, macro_auc), ~ mean(.x, na.rm = TRUE)),
    .groups = "drop"
  )

per_class_avg_df <- per_class_seed_df |>
  group_by(method, tissue, celltype) |>
  summarise(
    n_seeds = n_distinct(seed),
    across(c(precision, recall, f1, test_cells), ~ mean(.x, na.rm = TRUE)),
    .groups = "drop"
  )

importance_avg_df <- importance_seed_df |>
  group_by(method, tissue, feature) |>
  summarise(
    n_seeds = n_distinct(seed),
    importance = mean(importance, na.rm = TRUE),
    .groups = "drop"
  )

write.csv(metrics_seed_df, file.path(output_dir, "rf_metrics_by_tissue_per_seed.csv"), row.names = FALSE)
write.csv(per_class_seed_df, file.path(output_dir, "rf_metrics_by_celltype_per_seed.csv"), row.names = FALSE)
write.csv(importance_seed_df, file.path(output_dir, "rf_feature_importance_per_seed.csv"), row.names = FALSE)
write.csv(metrics_avg_df, file.path(output_dir, "rf_metrics_by_tissue_seed_average.csv"), row.names = FALSE)
write.csv(per_class_avg_df, file.path(output_dir, "rf_metrics_by_celltype_seed_average.csv"), row.names = FALSE)
write.csv(importance_avg_df, file.path(output_dir, "rf_feature_importance_seed_average.csv"), row.names = FALSE)

combined_plot <- metrics_avg_df |>
  mutate(
    dataset = recode(tissue, !!!dataset_map, .default = tissue),
    method = recode(method, !!!method_labels)
  ) |>
  select(-tissue) |>
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
    method = factor(method, levels = unname(method_labels), ordered = TRUE)
  )

method_fills <- c(
  "Halo" = scales::alpha("#A9D6FF", 0.45),
  "NE" = scales::alpha("#F8C8DC", 0.45),
  "Baysor" = scales::alpha("#F6D55C", 0.45),
  "Proseg" = scales::alpha("#88CC88", 0.45),
  "RNA2seg" = scales::alpha("#C9A0DC", 0.45)
)

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
  scale_fill_manual(values = method_fills, drop = FALSE) +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.15))) +
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
  filename = file.path(output_dir, "rf_metrics_res1_all_celltypes_5methods_5seed_average.png"),
  plot = metrics_plot,
  width = 11.5,
  height = 4.9,
  units = "in",
  dpi = 600,
  bg = "white"
)

ggsave(
  filename = file.path(output_dir, "rf_metrics_res1_all_celltypes_5methods_5seed_average.pdf"),
  plot = metrics_plot,
  width = 11.5,
  height = 4.9,
  units = "in",
  bg = "white"
)

ggsave(
  filename = file.path(output_dir, sprintf("rf_metrics_res1_all_celltypes_5methods_%s_average.png", run_label)),
  plot = metrics_plot,
  width = 11.5,
  height = 4.9,
  units = "in",
  dpi = 600,
  bg = "white"
)

ggsave(
  filename = file.path(output_dir, sprintf("rf_metrics_res1_all_celltypes_5methods_%s_average.pdf", run_label)),
  plot = metrics_plot,
  width = 11.5,
  height = 4.9,
  units = "in",
  bg = "white"
)

message(sprintf("[COMPLETE] RF %s results written to %s", run_label, output_dir))
