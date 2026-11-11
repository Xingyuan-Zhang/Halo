# Published analysis

This directory contains the curated scripts that produced the principal analysis panels in the Halo article. They were copied from the final analysis workspace and grouped by published figure. Raw Xenium datasets, segmentation masks, intermediate objects, and cluster logs are not distributed here.

The exact panel mapping is recorded in [`figure_manifest.tsv`](figure_manifest.tsv). Some final multi-panel figures were assembled manually from the generated PDF/PNG panels; the manifest identifies those cases rather than implying a single unavailable assembly script.

## Data layout

Several scripts expect the original analysis layout below. Point `HALO_ANALYSIS_ROOT` at a compatible project directory:

```text
PROJECT_ROOT/
├── tiles_sub/
├── revision_benchmark_20260519/
└── downstream/
    ├── cluster/
    ├── cluster_methods_groundtruth/
    └── morphology/
```

```bash
export HALO_ANALYSIS_ROOT=/path/to/Xenium_CellSeg
export HALO_TILES_ROOT="$HALO_ANALYSIS_ROOT/tiles_sub"
```

Python plotting scripts with required path arguments show their interface with `--help`. R scripts use `HALO_ANALYSIS_ROOT` and optional output-specific environment variables documented in their source.

## Figure directories

- `figure1`: cell-count and transcript-within-cell summaries. Notebook outputs correspond to panels 1a and 1c; panels 1b and 1d were assembled as explanatory artwork.
- `figure2`: representative segmentation overlays. The notebook generates source panels; the published composition was assembled from selected crops.
- `figure3`: Image IoU and Gene IoU benchmarking, kidney benchmark, and held-out-tissue comparison.
- `figure4`: Scanpy clustering and comparison with ground truth using ARI, AMI, homogeneity, and completeness.
- `figure5`: lung adenocarcinoma and melanoma cell-type mask comparison across six segmentation methods.
- `figure6`: morphology feature extraction, six-method boxplots, and median Pearson correlation plots.
- `figure7`: morphology UMAP centroids and random-forest classification metrics.

## Environments

Python analysis dependencies are listed in `environment/analysis-requirements.txt`. The R scripts additionally use packages including `dplyr`, `tidyr`, `readr`, `ggplot2`, `purrr`, `uwot`, `randomForest`, `pROC`, `mltools`, `ggpubr`, and `cowplot`.

The published analysis used large non-public intermediate files. Re-running every panel therefore requires either the original analysis workspace or equivalently structured outputs from the referenced segmentation methods.
