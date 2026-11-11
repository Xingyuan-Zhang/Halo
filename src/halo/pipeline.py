#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
import tifffile as tiff
from spatialdata_io import xenium
from skimage.draw import polygon
from cellpose import models

from .model_download import download_halo_model

DEFAULT_PIXEL_SIZE = 0.2125  # microns per pixel
DEFAULT_CHUNK_SIZE = 1_000_000

BAD_PATTERNS = (
    "UnassignedCodeword",
    "NegControlCodeword",
    "NegControlProbe",
    "BLANK",
)


def find_dapi_image(xenium_dir: Path) -> Path:
    """
    Try to locate a Xenium DAPI image under the dataset directory.
    Preference:
      1) morphology_focus/ch0000_dapi.ome.tif(.tiff)
      2) morphology.ome.tif(.tiff)
    """
    mf = xenium_dir / "morphology_focus"
    for name in ["ch0000_dapi.ome.tif", "ch0000_dapi.ome.tiff"]:
        cand = mf / name
        if cand.exists():
            return cand
    for name in ["morphology.ome.tif", "morphology.ome.tiff"]:
        cand = xenium_dir / name
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "Could not find DAPI image. Tried morphology_focus/ch0000_dapi.ome.tif(f) "
        "and morphology.ome.tif(f) under the Xenium dataset root."
    )


def load_dapi(img_path: Path) -> np.ndarray:
    img = tiff.imread(str(img_path))
    if img.ndim == 2:
        return img
    if img.ndim == 3:
        return img[0, :, :]
    raise ValueError(f"Unexpected image ndim={img.ndim}; expected 2D or 3D")


def build_transcript_density(
    xenium_path: Path,
    shape_hw: Tuple[int, int],
    pixel_size: float = DEFAULT_PIXEL_SIZE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    qv_min: int = 20,
    sigma: float = 2.5,
) -> np.ndarray:
    h, w = shape_hw

    sdata = xenium(str(xenium_path), morphology_focus=False)
    trans = sdata.points["transcripts"]

    required_cols = {"x", "y", "qv", "feature_name"}
    missing = required_cols - set(trans.columns)
    if missing:
        raise KeyError(f"Missing expected columns in transcripts table: {missing}")

    bad_regex = "|".join(BAD_PATTERNS)

    trans_filt = trans[
        (trans["qv"] >= qv_min)
        & (~trans["feature_name"].astype(str).str.contains(bad_regex))
    ]

    trans_xy = trans_filt[["x", "y"]]
    dask_arr = trans_xy.to_dask_array(lengths=True)

    dens = np.zeros((h, w), dtype=np.uint32)

    n_rows = dask_arr.shape[0]
    for i in range(0, n_rows, chunk_size):
        xy_chunk = dask_arr[i:i + chunk_size].compute()
        x_pix = np.rint(xy_chunk[:, 0] / pixel_size).astype(np.int32)
        y_pix = np.rint(xy_chunk[:, 1] / pixel_size).astype(np.int32)

        mask = (x_pix >= 0) & (x_pix < w) & (y_pix >= 0) & (y_pix < h)
        np.add.at(dens, (y_pix[mask], x_pix[mask]), 1)

    dens = gaussian_filter(dens.astype(np.float32), sigma=sigma)
    return dens


def save_true_cell_mask(
    xenium_path: Path,
    shape_hw: Tuple[int, int],
    out_path: Path,
    pixel_size: float = DEFAULT_PIXEL_SIZE,
) -> None:
    h, w = shape_hw
    sdata = xenium(str(xenium_path), morphology_focus=False)
    true_boundary = sdata.shapes["cell_boundaries"]["geometry"]
    true_masks = np.zeros((h, w), dtype=np.int32)
    for idx, geom in enumerate(true_boundary):
        coords = np.array(geom.exterior.coords)
        rows = np.round(coords[:, 1] / pixel_size).astype(int)
        cols = np.round(coords[:, 0] / pixel_size).astype(int)
        rr, cc = polygon(rows, cols, shape=(h, w))
        true_masks[rr, cc] = idx
    np.save(str(out_path), true_masks)


def preprocess(
    img_path: Path,
    xenium_path: Path,
    out_path: Path,
    quantile: float = 0.995,
    sigma: float = 2.5,
    pixel_size: float = DEFAULT_PIXEL_SIZE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    qv_min: int = 20,
) -> np.ndarray:
    dapi = load_dapi(img_path)

    clip_val = np.quantile(dapi, quantile)
    dapi = dapi.clip(0, clip_val)
    h, w = dapi.shape

    dens = build_transcript_density(
        xenium_path,
        (h, w),
        pixel_size=pixel_size,
        chunk_size=chunk_size,
        qv_min=qv_min,
        sigma=sigma,
    )

    dapi_norm = dapi / dapi.max() if dapi.max() > 0 else dapi
    dens_norm = dens / dens.max() if dens.max() > 0 else dens

    stack = np.stack((dapi_norm, dens_norm), axis=-1).astype(np.float32)
    stack_u16 = (stack * 65535).astype(np.uint16)
    tiff.imwrite(str(out_path), stack_u16, photometric="minisblack", metadata=None)
    return stack


def run_cellpose(
    img: np.ndarray,
    use_gpu: bool = True,
    channels: Tuple[int, int] = (1, 0),
    model_path: Optional[Path] = None,
    model_revision: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> np.ndarray:
    if model_path is None:
        model_path = download_halo_model(
            revision=model_revision,
            cache_dir=cache_dir,
        )

    model = models.CellposeModel(gpu=use_gpu, pretrained_model=str(model_path))
    masks_pred, *_ = model.eval([img], channels=list(channels), normalize=True)
    return masks_pred[0]


def save_mask(mask: np.ndarray, out_path: Path, fmt: str) -> None:
    fmt = fmt.lower()
    if fmt == "npy":
        np.save(str(out_path), mask)
    elif fmt == "tiff" or fmt == "tif":
        # use uint32 to preserve labels
        tiff.imwrite(str(out_path), mask.astype(np.uint32), photometric="minisblack")
    else:
        raise ValueError("mask format must be 'npy' or 'tiff'")


def run_pipeline(
    xenium_dir: Path,
    out_dir: Path,
    dapi_path: Optional[Path] = None,
    processed_out: Optional[Path] = None,
    mask_out: Optional[Path] = None,
    mask_format: str = "npy",
    quantile: float = 0.995,
    sigma: float = 2.5,
    pixel_size: float = DEFAULT_PIXEL_SIZE,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    qv_min: int = 20,
    use_gpu: bool = True,
    model_path: Optional[Path] = None,
    model_revision: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> Path:
    xenium_dir = xenium_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if dapi_path is None:
        dapi_path = find_dapi_image(xenium_dir)

    if processed_out is None:
        processed_out = out_dir / "halo_processed.tiff"

    if mask_out is None:
        mask_out = out_dir / ("cell_masks.npy" if mask_format == "npy" else "cell_masks.tiff")

    img = preprocess(
        img_path=dapi_path,
        xenium_path=xenium_dir,
        out_path=processed_out,
        quantile=quantile,
        sigma=sigma,
        pixel_size=pixel_size,
        chunk_size=chunk_size,
        qv_min=qv_min,
    )

    masks = run_cellpose(
        img,
        use_gpu=use_gpu,
        channels=(1, 0),
        model_path=model_path,
        model_revision=model_revision,
        cache_dir=cache_dir,
    )
    save_mask(masks, mask_out, mask_format)
    return mask_out
