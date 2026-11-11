#!/usr/bin/env python3
"""Compute instance-level Image IoU and Gene IoU for benchmark masks."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


TISSUE_TO_DATASET = {
    "human_brain": "glioblastoma",
    "human_breast": "breast cancer",
    "human_lung": "lung adenocarcinoma",
    "human_ovarian": "ovarian cancer",
    "human_skin": "melanoma",
    "human_pancreas": "pancreatic cancer",
    "human_cervical": "cervical cancer",
    "human_prostate": "prostate cancer",
    "human_lymph_node": "lymph node",
    "mouse_brain_coronal": "mouse brain coronal",
    "mouse_colon": "mouse colon",
    "mouse_pup": "mouse pup",
    "human_kidney": "kidney",
    "human_lung2": "lung",
    "human_ovarian_xe_prime": "ovarian cancer Xenium Prime",
}

ORIGINAL_12_TISSUES = [
    "human_brain",
    "human_breast",
    "human_cervical",
    "human_lung",
    "human_lymph_node",
    "human_ovarian",
    "human_pancreas",
    "human_prostate",
    "human_skin",
    "mouse_brain_coronal",
    "mouse_colon",
    "mouse_pup",
]


def dense_relabel(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ids = np.unique(arr)
    ids = ids[ids > 0]
    if ids.size == 0:
        return np.zeros(arr.shape, dtype=np.int32), ids
    out = np.zeros(arr.shape, dtype=np.int32)
    for i, lab in enumerate(ids, start=1):
        out[arr == lab] = i
    return out, ids


def rowmax_image_iou_match(true_mask: np.ndarray, pred_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t, _ = dense_relabel(true_mask)
    p, _ = dense_relabel(pred_mask)
    kt, kp = int(t.max()), int(p.max())
    if kp == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
    if kt == 0:
        return np.zeros(kp, dtype=np.int32), np.zeros(kp, dtype=np.float32)

    t_area = np.bincount(t.ravel())
    p_area = np.bincount(p.ravel())
    cols = kp + 1
    inter = np.bincount(t.ravel() * cols + p.ravel(), minlength=(kt + 1) * cols).reshape(kt + 1, cols)
    inter = inter[1:, 1:]
    t_area = t_area[1:]
    p_area = p_area[1:]
    union = t_area[:, None] + p_area[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    best = iou.argmax(axis=0).astype(np.int32) + 1
    vals = iou.max(axis=0).astype(np.float32)
    best[vals == 0] = 0
    return best, vals


def spots_to_labels(mask: np.ndarray, xy: np.ndarray) -> np.ndarray:
    if xy is None or xy.size == 0:
        return np.zeros((0,), dtype=np.int32)
    h, w = mask.shape
    x = np.clip(np.floor(xy[:, 0]).astype(np.int64), 0, w - 1)
    y = np.clip(np.floor(xy[:, 1]).astype(np.int64), 0, h - 1)
    return mask[y, x].astype(np.int32, copy=False)


def load_spots(spots_fp: Path | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    if spots_fp is None or not spots_fp.exists():
        return None, None
    z = np.load(spots_fp, allow_pickle=True)
    if "spots_xy_px" not in z or "spot_ids" not in z:
        return None, None
    xy = np.asarray(z["spots_xy_px"])
    spot_ids = np.asarray(z["spot_ids"])
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) != len(spot_ids):
        return None, None
    return xy, spot_ids


def dense_to_orig(ids: np.ndarray, dense_label: int) -> int:
    if dense_label <= 0:
        return 0
    idx = dense_label - 1
    return int(ids[idx]) if 0 <= idx < len(ids) else 0


def evaluate_tile(gt: np.ndarray, pred: np.ndarray, xy: np.ndarray | None, spot_ids: np.ndarray | None):
    t, t_ids = dense_relabel(gt)
    p, p_ids = dense_relabel(pred)
    kt, kp = int(t.max()), int(p.max())
    best_gt, image_iou = rowmax_image_iou_match(t, p)

    pred_sets = [set() for _ in range(kp + 1)]
    gt_sets = [set() for _ in range(kt + 1)]
    if xy is not None and spot_ids is not None:
        lab_p = spots_to_labels(p, xy)
        lab_t = spots_to_labels(t, xy)
        for sid, lp, lt in zip(spot_ids, lab_p, lab_t):
            if lp > 0:
                pred_sets[int(lp)].add(int(sid))
            if lt > 0:
                gt_sets[int(lt)].add(int(sid))

    rows = []
    for k in range(1, kp + 1):
        g = int(best_gt[k - 1]) if k - 1 < len(best_gt) else 0
        pred_spots = pred_sets[k]
        gt_spots = gt_sets[g] if g > 0 else set()
        if pred_spots:
            union = len(pred_spots | gt_spots)
            gene_iou = len(pred_spots & gt_spots) / union if union else 0.0
        else:
            gene_iou = math.nan
        rows.append(
            {
                "cell_label": dense_to_orig(p_ids, k),
                "matched_cell_label": dense_to_orig(t_ids, g),
                "image_iou": float(image_iou[k - 1]) if k - 1 < len(image_iou) else math.nan,
                "gene_iou": gene_iou,
                "n_spots_cell": len(pred_spots),
                "n_spots_matched_cell": len(gt_spots),
            }
        )
    return rows


def parse_method(s: str) -> tuple[str, str]:
    if "=" not in s:
        raise argparse.ArgumentTypeError("Method must be name=template")
    name, tmpl = s.split("=", 1)
    return name, tmpl


def default_methods(out_dir: Path) -> list[tuple[str, str]]:
    return [
        ("baysor", f"{out_dir}/masks/baysor/{{tissue}}/{{tile}}.npy"),
        ("proseg", f"{out_dir}/masks/proseg/{{tissue}}/{{tile}}.npy"),
        ("rna2seg", f"{out_dir}/masks/rna2seg/{{tissue}}/{{tile}}.npy"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tissues", nargs="*", default=None)
    ap.add_argument("--method", action="append", type=parse_method, default=None, help="name=mask_path_template")
    ap.add_argument(
        "--original-summary",
        type=Path,
        required=True,
        help="Existing trusted CellposeSAM/NE tissue-level result CSV.",
    )
    ap.add_argument(
        "--original-per-cell",
        type=Path,
        required=True,
        help="Existing trusted per-cell CellposeSAM/NE result CSV.",
    )
    ap.add_argument("--skip-original-baseline", action="store_true")
    args = ap.parse_args()

    methods = args.method if args.method else default_methods(args.out_dir)
    tissues = args.tissues
    if tissues is None:
        tissues = ORIGINAL_12_TISSUES

    all_rows = []
    missing_rows = []
    for tissue in tissues:
        root = args.tiles_dir / tissue / args.split
        meta = pd.read_csv(root / "tiles_meta.csv")
        for tile in meta["tile_name"].astype(str):
            gt_fp = root / "masks" / f"{tile}.npy"
            spots_fp = root / "spots" / f"{tile}.npz"
            if not gt_fp.exists():
                missing_rows.append({"tissue": tissue, "tile": tile, "method": "gt", "reason": "missing_gt"})
                continue
            gt = np.load(gt_fp)
            xy, spot_ids = load_spots(spots_fp)
            for method, tmpl in methods:
                mask_fp = Path(tmpl.format(tiles_dir=args.tiles_dir, out_dir=args.out_dir, tissue=tissue, tile=tile, method=method))
                if not mask_fp.exists():
                    missing_rows.append({"tissue": tissue, "tile": tile, "method": method, "reason": "missing_mask"})
                    continue
                pred = np.load(mask_fp)
                if pred.shape != gt.shape:
                    missing_rows.append({"tissue": tissue, "tile": tile, "method": method, "reason": f"shape_mismatch:{pred.shape}!={gt.shape}"})
                    continue
                for row in evaluate_tile(gt, pred, xy, spot_ids):
                    all_rows.append({"tissue": tissue, "dataset": TISSUE_TO_DATASET.get(tissue, tissue), "tile_id": tile, "method": method, **row})

    metrics_dir = args.out_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    per_cell = pd.DataFrame(all_rows)
    seeded_summary = []
    seeded_per_cell = []
    if not args.skip_original_baseline:
        if args.original_summary.exists():
            original = pd.read_csv(args.original_summary)
            original = original[original["tissue"].isin(tissues)].copy()
            for _, row in original.iterrows():
                tissue = row["tissue"]
                dataset = TISSUE_TO_DATASET.get(tissue, tissue)
                seeded_summary.extend(
                    [
                        {
                            "tissue": tissue,
                            "dataset": dataset,
                            "method": "cellposesam",
                            "image_iou_mean": row.get("image_iou_pred_mean", np.nan),
                            "gene_iou_mean": row.get("gene_iou_pred_mean", np.nan),
                            "n_pred_cells": np.nan,
                            "n_gene_cells": np.nan,
                            "n_tiles": 200,
                        },
                        {
                            "tissue": tissue,
                            "dataset": dataset,
                            "method": "ne",
                            "image_iou_mean": row.get("image_iou_xr_mean", np.nan),
                            "gene_iou_mean": row.get("gene_iou_xr_mean", np.nan),
                            "n_pred_cells": np.nan,
                            "n_gene_cells": np.nan,
                            "n_tiles": 200,
                        },
                    ]
                )
        else:
            missing_rows.append({"tissue": "", "tile": "", "method": "cellposesam/ne", "reason": f"missing_original_summary:{args.original_summary}"})

        if args.original_per_cell.exists():
            old_pc = pd.read_csv(args.original_per_cell)
            old_pc = old_pc[old_pc["tissue"].isin(tissues)].copy()
            old_pc["dataset"] = old_pc["tissue"].map(TISSUE_TO_DATASET).fillna(old_pc["tissue"])
            old_pc["method"] = old_pc["source"].map({"pred": "cellposesam", "xr": "ne"}).fillna(old_pc["source"])
            keep = [
                "tissue",
                "dataset",
                "tile_id",
                "method",
                "cell_label",
                "matched_cell_label",
                "image_iou",
                "gene_iou",
                "n_spots_cell",
                "n_spots_matched_cell",
            ]
            seeded_per_cell = old_pc[keep]
        else:
            missing_rows.append({"tissue": "", "tile": "", "method": "cellposesam/ne", "reason": f"missing_original_per_cell:{args.original_per_cell}"})

    if len(seeded_per_cell) > 0:
        per_cell = pd.concat([seeded_per_cell, per_cell], ignore_index=True)
    per_cell_fp = metrics_dir / "per_cell_iou_long.csv"
    per_cell.to_csv(per_cell_fp, index=False)

    if per_cell.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            per_cell.groupby(["tissue", "dataset", "method"], as_index=False)
            .agg(
                image_iou_mean=("image_iou", "mean"),
                gene_iou_mean=("gene_iou", "mean"),
                n_pred_cells=("image_iou", "size"),
                n_gene_cells=("gene_iou", lambda s: int(s.notna().sum())),
                n_tiles=("tile_id", "nunique"),
            )
            .sort_values(["tissue", "method"])
        )
    if seeded_summary:
        seeded_summary_df = pd.DataFrame(seeded_summary)
        if summary.empty:
            summary = seeded_summary_df
        else:
            summary = pd.concat([seeded_summary_df, summary], ignore_index=True)
            summary = summary.drop_duplicates(["tissue", "method"], keep="first")
        summary = summary.sort_values(["tissue", "method"])
    summary_fp = metrics_dir / "summary_by_tissue_method.csv"
    summary.to_csv(summary_fp, index=False)

    if not summary.empty:
        wide = summary.pivot(index=["tissue", "dataset"], columns="method", values=["image_iou_mean", "gene_iou_mean"])
        wide.columns = [f"{metric}_{method}" for metric, method in wide.columns]
        wide = wide.reset_index()
    else:
        wide = pd.DataFrame()
    wide_fp = metrics_dir / "summary_by_tissue_wide.csv"
    wide.to_csv(wide_fp, index=False)

    missing = pd.DataFrame(missing_rows)
    missing_fp = metrics_dir / "missing_masks.csv"
    missing.to_csv(missing_fp, index=False)
    print(f"Wrote {per_cell_fp}")
    print(f"Wrote {summary_fp}")
    print(f"Wrote {wide_fp}")
    print(f"Wrote {missing_fp}")


if __name__ == "__main__":
    main()
