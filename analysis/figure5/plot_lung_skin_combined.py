#!/usr/bin/env python3
from pathlib import Path
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image, ImageFilter


PROJECT = Path(os.environ.get("HALO_ANALYSIS_ROOT", ".")).resolve()
REV = PROJECT / "revision_benchmark_20260519"
FIG_DIR = REV / "figures"
OUT = Path(os.environ.get("HALO_FIGURE5_OUTPUT", "results/figure5")).resolve()

LUNG_SCRIPT = FIG_DIR / "figure004_style_tile_masks/figure004_style_tile_masks.py"
SKIN_PLOT_SCRIPT = FIG_DIR / "skin_fibroblast_immune_regions/plot_skin_fibroblast_immune_candidates.py"
SKIN_SCAN_SCRIPT = FIG_DIR / "skin_fibroblast_immune_regions/find_skin_fibroblast_immune_regions.py"

METHODS = [
    ("Ground truth", "tenx"),
    ("Halo", "halo"),
    ("Nuclear expansion", "ne"),
    ("Baysor", "baysor"),
    ("Proseg", "proseg"),
    ("RNA2seg", "rna2seg"),
]

PALETTE = {
    "Cancer cells": "#D55E00",
    "Endothelial cells": "#004B87",
    "Epithelial cells": "#00A6D6",
    "Fibroblasts": "#009E73",
    "Smooth muscle cells": "#CC79A7",
    "Myeloid cells": "#7A52A3",
    "T cells": "#8B1E3F",
    "B cells": "#8A8F00",
}
DISPLAY_TYPES = list(PALETTE)
UPSCALE = 8


def import_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return np.array([int(hex_color[i:i + 2], 16) for i in (0, 2, 4)], dtype=float) / 255.0


RGB = {k: hex_to_rgb(v) for k, v in PALETTE.items()}


def norm_type(value):
    if pd.isna(value):
        return "Unknown"
    value = str(value)
    if value in {"Lung cancer cells", "Skin cancer cells"}:
        return "Cancer cells"
    if value not in PALETTE:
        return "Unknown"
    return value


def smooth2d(arr, sigma=0.75):
    radius = max(1, int(round(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    padded = np.pad(arr, ((0, 0), (radius, radius)), mode="reflect")
    tmp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), 1, padded)
    padded = np.pad(tmp, ((radius, radius), (0, 0)), mode="reflect")
    return np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), 0, padded)


def shrink_cell(cell):
    # Kept for compatibility with older output logic; the current renderer uses
    # shrink_cell_contacts so gaps appear only between neighboring cells.
    return cell


def shrink_cell_contacts(mask, label):
    cell = mask == label
    remove = np.zeros(mask.shape, dtype=bool)
    remove[:, 1:] |= cell[:, 1:] & (mask[:, :-1] > label)
    remove[:, :-1] |= cell[:, :-1] & (mask[:, 1:] > label)
    remove[1:, :] |= cell[1:, :] & (mask[:-1, :] > label)
    remove[:-1, :] |= cell[:-1, :] & (mask[1:, :] > label)
    retained = cell & ~remove
    if retained.sum() >= max(4, cell.sum() * 0.55):
        return retained
    return cell


def colorize_mask(mask, label_to_type, bg, alpha=0.60, edge_sigma=0.65):
    rgb = np.repeat(bg[..., None], 3, axis=2)
    type_weights = {ct: np.zeros(mask.shape, dtype=float) for ct in DISPLAY_TYPES}
    for label in np.unique(mask):
        if label <= 0:
            continue
        ct = norm_type(label_to_type.get(int(label), "Unknown"))
        if ct == "Unknown":
            continue
        type_weights[ct][shrink_cell_contacts(mask, int(label))] = 1.0

    total = np.zeros(mask.shape, dtype=float)
    color_mix = np.zeros((*mask.shape, 3), dtype=float)
    for ct, weight in type_weights.items():
        if not np.any(weight):
            continue
        weight = smooth2d(weight, sigma=edge_sigma)
        total += weight
        color_mix += weight[..., None] * RGB[ct]

    visible = total > 1e-4
    color_mix[visible] /= total[visible, None]
    alpha_map = alpha * np.clip(total, 0, 1)
    rgb[visible] = (1 - alpha_map[visible, None]) * rgb[visible] + alpha_map[visible, None] * color_mix[visible]

    return rgb


def crop_array(arr, crop):
    x0, y0, x1, y1 = crop
    return arr[y0:y1, x0:x1]


def upscale_panel(rgb):
    arr = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    image = Image.fromarray(arr, mode="RGB")
    image = image.resize(
        (image.width * UPSCALE, image.height * UPSCALE),
        resample=Image.Resampling.LANCZOS,
    )
    image = image.filter(ImageFilter.UnsharpMask(radius=0.6, percent=25, threshold=2))
    return np.asarray(image)


def legend_handles(present_types):
    return [
        Patch(facecolor=PALETTE[ct], edgecolor="none", label=ct)
        for ct in DISPLAY_TYPES
        if ct in present_types
    ]


def present_celltypes(mask, label_to_type, crop):
    cropped = crop_array(mask, crop)
    out = set()
    for label in np.unique(cropped):
        if label <= 0:
            continue
        ct = norm_type(label_to_type.get(int(label), "Unknown"))
        if ct != "Unknown":
            out.add(ct)
    return out


def load_lung_panel(lung):
    tile = "tile_0109_base"
    return {
        "title": "",
        "tile": tile,
        "crop": lung.read_crop(lung.TILE_CONFIGS[tile]),
        "panel": lung.tile_data(tile),
    }


def load_skin_panel(skin_plot, skin_scan):
    tile = "tile_0045_base"
    meta = pd.read_csv(skin_scan.TILE_ROOT / "tiles_meta.csv")
    row = meta.loc[meta["tile_name"] == tile].iloc[0].to_dict()
    all_points = {
        "tenx": skin_scan.read_cell_points(skin_scan.TENX_CELLS, "tenx"),
        "cpsam": skin_scan.read_cell_points(skin_scan.CPSAM_CELLS, "cpsam"),
        "ne": skin_scan.read_cell_points(skin_scan.NE_CELLS, "ne"),
    }
    clusters_by_method = {m: skin_scan.load_method_clusters(m) for m in skin_scan.CLUSTER_METHODS}
    return {
        "title": "",
        "tile": tile,
        "crop": (206, 0, 396, 190),
        "panel": skin_plot.load_panel_data(tile, row, all_points, clusters_by_method),
    }


def render(panels, out_stem):
    nrows = len(METHODS)
    ncols = len(panels)
    present_types = set()
    for panel in panels:
        _, data = panel["panel"]
        for _, method_key in METHODS:
            mask, label_to_type = data[method_key]
            present_types.update(present_celltypes(mask, label_to_type, panel["crop"]))

    fig = plt.figure(figsize=(5.25 * ncols + 2.45, 3.30 * nrows + 1.20), dpi=600)
    gs = fig.add_gridspec(
        nrows=nrows,
        ncols=ncols + 1,
        width_ratios=[0.50] + [1.0] * ncols,
        hspace=0.055,
        wspace=0.08,
        left=0.035,
        right=0.985,
        top=0.965,
        bottom=0.095,
    )

    for row_idx, (method_name, method_key) in enumerate(METHODS):
        label_ax = fig.add_subplot(gs[row_idx, 0])
        label_ax.axis("off")
        label_ax.text(0.82, 0.50, method_name, ha="center", va="center", fontsize=18)

        for col_idx, panel in enumerate(panels):
            ax = fig.add_subplot(gs[row_idx, col_idx + 1])
            bg, data = panel["panel"]
            mask, label_to_type = data[method_key]
            rgb = colorize_mask(mask, label_to_type, bg)
            ax.imshow(upscale_panel(crop_array(rgb, panel["crop"])), interpolation="none", resample=False)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
    fig.legend(
        handles=legend_handles(present_types),
        loc="lower center",
        bbox_to_anchor=(0.60, 0.042),
        ncol=4,
        frameon=False,
        fontsize=11,
        handlelength=0.85,
        handleheight=0.85,
        columnspacing=1.25,
        handletextpad=0.35,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    png = OUT / f"{out_stem}.png"
    pdf = OUT / f"{out_stem}.pdf"
    fig.savefig(png, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("saved", png)
    print("saved", pdf)


def main():
    lung = import_from_path("lung_figure004_masks", LUNG_SCRIPT)
    sys.path.insert(0, str(SKIN_SCAN_SCRIPT.parent))
    skin_scan = import_from_path("skin_fibroblast_scan", SKIN_SCAN_SCRIPT)
    skin_plot = import_from_path("skin_fibroblast_plot", SKIN_PLOT_SCRIPT)
    panels = [load_lung_panel(lung), load_skin_panel(skin_plot, skin_scan)]
    render(panels, "lung_tile0109_skin_tile0045_combined_figure004_style_masks")


if __name__ == "__main__":
    main()
