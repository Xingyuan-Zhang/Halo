#!/usr/bin/env bash
set -euo pipefail

halo /path/to/xenium_dataset \
  --out-dir results/halo \
  --mask-format npy
