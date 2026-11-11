#!/usr/bin/env python3
"""Train Halo from paired two-channel image tiles and instance masks.

This is a portable version of the Cellpose-SAM training program used for the
published Halo model. It keeps large arrays in disk-backed NumPy memmaps.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile as tiff
from cellpose import io, models, train
from tqdm import tqdm


@dataclass(frozen=True)
class TilePair:
    tissue: str
    image: Path
    mask: Path


def discover_pairs(
    tiles_dir: Path,
    split: str,
    excluded_tissues: set[str],
) -> list[TilePair]:
    """Find images and masks using `<tissue>/<split>/{images,masks}`."""
    pairs: list[TilePair] = []
    for tissue_dir in sorted(tiles_dir.iterdir()):
        if not tissue_dir.is_dir() or tissue_dir.name in excluded_tissues:
            continue
        image_dir = tissue_dir / split / "images"
        mask_dir = tissue_dir / split / "masks"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            continue

        images: dict[str, Path] = {}
        for suffix in ("*.tif", "*.tiff"):
            for path in image_dir.glob(suffix):
                if path.stem in images:
                    raise ValueError(f"Duplicate image stem in {image_dir}: {path.stem}")
                images[path.stem] = path
        masks = {path.stem: path for path in mask_dir.glob("*.npy")}

        missing_masks = sorted(set(images) - set(masks))
        missing_images = sorted(set(masks) - set(images))
        if missing_masks or missing_images:
            raise ValueError(
                f"Unpaired files for {tissue_dir.name}/{split}: "
                f"{len(missing_masks)} images without masks and "
                f"{len(missing_images)} masks without images"
            )
        pairs.extend(
            TilePair(tissue_dir.name, images[stem], masks[stem])
            for stem in sorted(images)
        )

    if not pairs:
        raise RuntimeError(f"No paired tiles found for split={split!r} in {tiles_dir}")
    return pairs


def _load_pair(index: int, pair: TilePair, expected_shape: tuple[int, int, int]):
    image = tiff.imread(pair.image)
    if image.shape != expected_shape:
        raise ValueError(
            f"Image shape mismatch for {pair.image}: {image.shape} != {expected_shape}"
        )
    mask = np.load(pair.mask)
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.shape != expected_shape[:2]:
        raise ValueError(
            f"Mask shape mismatch for {pair.mask}: {mask.shape} != {expected_shape[:2]}"
        )
    image_chw = np.transpose(image, (2, 0, 1)).astype(np.float32, copy=False)
    return index, image_chw, mask.astype(np.uint32, copy=False)


def build_memmaps(
    pairs: list[TilePair],
    image_path: Path,
    mask_path: Path,
    workers: int,
    max_inflight: int,
) -> tuple[np.memmap, np.memmap]:
    """Load paired tiles into bounded, disk-backed arrays."""
    image_path.parent.mkdir(parents=True, exist_ok=True)
    sample = tiff.imread(pairs[0].image)
    if sample.ndim != 3 or sample.shape[-1] != 2:
        raise ValueError(
            f"Expected H x W x 2 input tiles; found {sample.shape} in {pairs[0].image}"
        )
    height, width, channels = sample.shape
    count = len(pairs)
    images = np.lib.format.open_memmap(
        image_path,
        mode="w+",
        dtype=np.float32,
        shape=(count, channels, height, width),
    )
    masks = np.lib.format.open_memmap(
        mask_path,
        mode="w+",
        dtype=np.uint32,
        shape=(count, height, width),
    )

    pair_iter = iter(enumerate(pairs))
    pending = set()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, pair in pair_iter:
            pending.add(executor.submit(_load_pair, index, pair, sample.shape))
            if len(pending) >= max_inflight:
                break

        with tqdm(total=count, desc=f"Building {image_path.stem}", unit="tile") as progress:
            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    index, image, mask = future.result()
                    images[index] = image
                    masks[index] = mask
                    progress.update(1)
                    try:
                        next_index, next_pair = next(pair_iter)
                    except StopIteration:
                        continue
                    pending.add(
                        executor.submit(_load_pair, next_index, next_pair, sample.shape)
                    )

    images.flush()
    masks.flush()
    return images, masks


def write_losses(path: Path, train_losses: Iterable[float], val_losses: Iterable[float]) -> None:
    train_values = list(np.asarray(train_losses).ravel())
    val_values = list(np.asarray(val_losses).ravel())
    rows = max(len(train_values), len(val_values))
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "validation_loss"])
        for index in range(rows):
            writer.writerow(
                [
                    index + 1,
                    train_values[index] if index < len(train_values) else "",
                    val_values[index] if index < len(val_values) else "",
                ]
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="Halo")
    parser.add_argument("--pretrained-model", default="cpsam")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="test")
    parser.add_argument("--exclude-tissue", action="append", default=[])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-inflight", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the directory layout and report tile counts without training.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    excluded = set(args.exclude_tissue)
    train_pairs = discover_pairs(args.tiles_dir, args.train_split, excluded)
    validation_pairs = discover_pairs(args.tiles_dir, args.validation_split, excluded)
    tissues = sorted({pair.tissue for pair in train_pairs})
    print(f"Training tissues ({len(tissues)}): {', '.join(tissues)}")
    print(f"Training tiles: {len(train_pairs)}")
    print(f"Validation tiles: {len(validation_pairs)}")
    if excluded:
        print(f"Excluded tissues: {', '.join(sorted(excluded))}")
    if args.dry_run:
        return

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_data, train_labels = build_memmaps(
        train_pairs,
        args.cache_dir / "train_images.npy",
        args.cache_dir / "train_masks.npy",
        args.workers,
        args.max_inflight,
    )
    validation_data, validation_labels = build_memmaps(
        validation_pairs,
        args.cache_dir / "validation_images.npy",
        args.cache_dir / "validation_masks.npy",
        args.workers,
        args.max_inflight,
    )

    io.logger_setup()
    model = models.CellposeModel(
        gpu=not args.cpu,
        pretrained_model=args.pretrained_model,
    )
    model_path, train_losses, validation_losses = train.train_seg(
        model.net,
        train_data=train_data,
        train_labels=train_labels,
        test_data=validation_data,
        test_labels=validation_labels,
        normalize=True,
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        nimg_per_epoch=None,
        model_name=args.model_name,
        save_path=str(args.output_dir),
        save_every=args.save_every,
        save_each=True,
    )
    print(f"Final checkpoint: {model_path}")
    write_losses(args.output_dir / "training_losses.csv", train_losses, validation_losses)
    with (args.output_dir / "training_config.json").open("w") as handle:
        config = vars(args).copy()
        config.update(
            training_tissues=tissues,
            training_tiles=len(train_pairs),
            validation_tiles=len(validation_pairs),
            checkpoint=str(model_path),
        )
        json.dump({key: str(value) if isinstance(value, Path) else value for key, value in config.items()}, handle, indent=2)

    del train_data, train_labels, validation_data, validation_labels, model
    gc.collect()


if __name__ == "__main__":
    main()
