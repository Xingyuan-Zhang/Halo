#!/usr/bin/env bash
#SBATCH --job-name=halo-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G
#SBATCH --time=168:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

# Adjust these paths and activate an environment containing Halo and Cellpose.
python training/train_halo.py \
  --tiles-dir /path/to/tiles_sub \
  --cache-dir /scratch/halo-cache \
  --output-dir results/models \
  --model-name Halo \
  --workers "${SLURM_CPUS_PER_TASK:-8}"
