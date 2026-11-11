# Halo model training

`train_halo.py` is the portable version of the Cellpose-SAM training workflow used for Halo. It expects preprocessed two-channel tiles and whole-cell instance masks.

Install the training dependencies from the repository root:

```bash
python -m pip install -e ".[training]"
```

## Input layout

```text
tiles/
├── human_brain/
│   ├── train/
│   │   ├── images/*.tiff
│   │   └── masks/*.npy
│   └── test/
│       ├── images/*.tiff
│       └── masks/*.npy
└── other_tissue/...
```

Each image must have shape `height × width × 2`:

1. normalized DAPI channel;
2. transcript-density pseudo-image.

Each mask must be a two-dimensional integer-labelled array with the same height and width. Image and mask filenames are paired by stem.

## Published-model configuration

The defaults reproduce the principal settings from the final Halo training program:

- initialization: Cellpose-SAM `cpsam`;
- epochs: 300;
- learning rate: `1e-3`;
- weight decay: `1e-5`;
- batch size: 4;
- checkpoint interval: 50 epochs;
- all available training tissues included.

Run a layout check before allocating a GPU:

```bash
python training/train_halo.py \
  --tiles-dir /path/to/tiles_sub \
  --cache-dir /scratch/halo-cache \
  --output-dir results/models \
  --dry-run
```

Train Halo:

```bash
python training/train_halo.py \
  --tiles-dir /path/to/tiles_sub \
  --cache-dir /scratch/halo-cache \
  --output-dir results/models \
  --model-name Halo
```

The cache directory stores large disk-backed arrays and should normally point to local scratch storage. Checkpoints, `training_losses.csv`, and `training_config.json` are written to the output directory.

## Held-out-tissue experiment

The held-out model used in the paper excluded `human_lung`, `human_lung2`, `human_pancreas`, and `mouse_brain_coronal`:

```bash
python training/train_halo.py \
  --tiles-dir /path/to/tiles_sub \
  --cache-dir /scratch/halo-heldout-cache \
  --output-dir results/heldout-model \
  --model-name halo_heldout \
  --exclude-tissue human_lung \
  --exclude-tissue human_lung2 \
  --exclude-tissue human_pancreas \
  --exclude-tissue mouse_brain_coronal
```

The original training data and trained checkpoints are intentionally not stored in GitHub. The published checkpoint remains available from the Hugging Face repository linked in the main README.
