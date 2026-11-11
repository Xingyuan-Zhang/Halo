#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Halo: Xenium preprocessing + Cellpose XeniumSeg pipeline"
    )
    p.add_argument(
        "xenium_dir",
        type=Path,
        help="Path to Xenium dataset root directory",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for outputs (default: current working directory)",
    )
    p.add_argument(
        "--dapi",
        type=Path,
        default=None,
        help="Optional path to DAPI image (overrides auto-detect)",
    )
    p.add_argument(
        "--processed-out",
        type=Path,
        default=None,
        help="Optional path for the processed 2-channel TIFF",
    )
    p.add_argument(
        "--mask-out",
        type=Path,
        default=None,
        help="Optional path for the output cell mask",
    )
    p.add_argument(
        "--mask-format",
        choices=["npy", "tiff"],
        default="npy",
        help="Output mask format (default: npy)",
    )
    p.add_argument(
        "--quantile",
        type=float,
        default=0.995,
        help="Upper quantile for clipping DAPI intensity",
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=2.5,
        help="Gaussian sigma for transcript density smoothing",
    )
    p.add_argument(
        "--pixel-size",
        type=float,
        default=0.2125,
        help="Microns per pixel for transcript binning",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Transcripts processed per chunk",
    )
    p.add_argument(
        "--qv-min",
        type=int,
        default=20,
        help="Minimum transcript QV to keep",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference (disable GPU)",
    )
    p.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local Halo checkpoint. By default it is downloaded from Hugging Face.",
    )
    p.add_argument(
        "--model-revision",
        default=None,
        help="Optional Hugging Face model revision, tag, or commit.",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    return p


def main() -> None:
    p = build_parser()
    args = p.parse_args()

    out_dir = args.out_dir if args.out_dir is not None else Path.cwd()

    mask_path = run_pipeline(
        xenium_dir=args.xenium_dir,
        out_dir=out_dir,
        dapi_path=args.dapi,
        processed_out=args.processed_out,
        mask_out=args.mask_out,
        mask_format=args.mask_format,
        quantile=args.quantile,
        sigma=args.sigma,
        pixel_size=args.pixel_size,
        chunk_size=args.chunk_size,
        qv_min=args.qv_min,
        use_gpu=not args.cpu,
        model_path=args.model_path,
        model_revision=args.model_revision,
        cache_dir=args.cache_dir,
    )

    print(f"✓ Saved cell mask to {mask_path}")


if __name__ == "__main__":
    main()
