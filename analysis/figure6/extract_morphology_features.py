#!/usr/bin/env python3
"""Compute morphology features for tile-mask methods.

Feature definitions mirror the previous morphology workflow:
area, perimeter, Feret diameter, eccentricity, roundness, circularity,
solidity, and aspect ratio.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi


PROJECT_ROOT = Path(os.environ.get("HALO_ANALYSIS_ROOT", ".")).resolve()
BENCH_ROOT = PROJECT_ROOT / "revision_benchmark_20260519"
TILES_ROOT = PROJECT_ROOT / "tiles_sub"
OUT_ROOT = PROJECT_ROOT / "downstream/morphology/morphology_features"
BUNDLES = PROJECT_ROOT / "script/downstream/bundles.tsv"
METHODS = ["baysor", "proseg", "rna2seg"]
SCALE = 0.2125


def read_tissues() -> list[str]:
    tissues = []
    for line in BUNDLES.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tissues.append(line.split("\t")[0])
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task:
        idx = int(task) - 1
        return [tissues[idx]]
    return tissues


def region_features(mask: np.ndarray, label: int, scale: float) -> dict[str, float] | None:
    binary = (mask == label).astype(np.uint8)
    if int(binary.sum()) <= 0:
        return None
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = [c for c in contours if c.shape[0] >= 3]
    if not contours:
        return None

    pts = np.vstack([c.reshape(-1, 2) for c in contours]).astype(np.float64)
    if pts.shape[0] < 3:
        return None

    area_px = float(sum(abs(cv2.contourArea(c)) for c in contours))
    if not np.isfinite(area_px) or area_px <= 0:
        area_px = float(binary.sum())
    perimeter_px = float(sum(cv2.arcLength(c, True) for c in contours))

    hull = cv2.convexHull(pts.astype(np.float32))
    hull_pts = hull.reshape(-1, 2).astype(np.float64)
    if hull_pts.shape[0] >= 2:
        diff = hull_pts[:, None, :] - hull_pts[None, :, :]
        feret_px = float(np.sqrt((diff * diff).sum(axis=2)).max())
    else:
        feret_px = np.nan

    pts_centered = pts - pts.mean(axis=0, keepdims=True)
    eccentricity = np.nan
    if pts_centered.shape[0] >= 3:
        cov = np.cov(pts_centered.T)
        ev = np.linalg.eigvalsh(cov)
        ev = np.sort(ev)[::-1]
        if ev[0] > 0:
            eccentricity = float(1.0 - ev[1] / ev[0])

    (_, _), radius = cv2.minEnclosingCircle(pts.astype(np.float32))
    circle_area_px = float(np.pi * radius * radius)
    roundness = float(area_px / circle_area_px) if circle_area_px > 0 else np.nan

    circularity = float(4 * np.pi * area_px / (perimeter_px * perimeter_px)) if perimeter_px > 0 else np.nan

    convex_area_px = float(abs(cv2.contourArea(hull))) if hull.shape[0] >= 3 else np.nan
    solidity = float(area_px / convex_area_px) if np.isfinite(convex_area_px) and convex_area_px > 0 else np.nan

    rect = cv2.minAreaRect(pts.astype(np.float32))
    width, height = rect[1]
    short = min(width, height)
    long = max(width, height)
    ar = float(long / short) if short > 0 else np.nan

    return {
        "area": area_px * scale * scale,
        "perimeter": perimeter_px * scale,
        "Feret": feret_px * scale if np.isfinite(feret_px) else np.nan,
        "eccentricity": eccentricity,
        "roundness": roundness,
        "circularity": circularity,
        "solidity": solidity,
        "AR": ar,
    }


def compute_method_tissue(method: str, tissue: str) -> pd.DataFrame:
    tiles = pd.read_csv(TILES_ROOT / tissue / "test/tiles_meta.csv")
    mask_dir = BENCH_ROOT / "masks" / method / tissue
    rows = []
    for tile in tiles["tile_name"].astype(str):
        mask_path = mask_dir / f"{tile}.npy"
        if not mask_path.exists():
            continue
        mask = np.load(mask_path)
        objects = ndi.find_objects(mask)
        for label, bbox in enumerate(objects, start=1):
            if bbox is None:
                continue
            crop = mask[bbox]
            if not np.any(crop == label):
                continue
            feats = region_features(crop, int(label), SCALE)
            if feats is None:
                continue
            rows.append({"cell_id": f"{tile}__{int(label)}", **feats})
    return pd.DataFrame(rows)


def main() -> None:
    for tissue in read_tissues():
        for method in METHODS:
            out_dir = OUT_ROOT / method
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{tissue}.csv"
            if out_path.exists() and out_path.stat().st_size > 0:
                print(f"[SKIP] {out_path}", flush=True)
                continue
            print(f"[RUN ] {method} {tissue}", flush=True)
            df = compute_method_tissue(method, tissue)
            df.to_csv(out_path, index=False)
            print(f"[DONE] {out_path} rows={len(df)}", flush=True)


if __name__ == "__main__":
    main()
