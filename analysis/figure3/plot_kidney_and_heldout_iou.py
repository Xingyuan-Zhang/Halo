#!/usr/bin/env python3
"""Draw heldout3 and kidney IoU barplots with matched kidney-barplot styling."""

from __future__ import annotations

import argparse
from pathlib import Path

from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt
import pandas as pd


BAR_COLOR = "#17A0E8"
ALPHA_RANGE = (0.55, 0.85)

KIDNEY_FIG_DIR: Path
REVISION_FIG_DIR: Path
HELDOUT_CSV: Path
KIDNEY_VALUES: Path

METHOD_LABELS = {
    "xeniumseg": "Halo (all tissue)",
    "halo_heldout": "Halo (held out)",
    "proseg": "Proseg",
}

METHOD_COLORS = {
    "xeniumseg": "navy",
    "halo_heldout": "#5DADE2",
    "proseg": "#8C8C8C",
}

SAMPLE_LABELS = {
    "human_lung": "lung adenocarcinoma",
    "human_pancreas": "pancreatic cancer",
    "mouse_brain_coronal": "mouse brain coronal",
}
SAMPLE_ORDER = ["human_lung", "human_pancreas", "mouse_brain_coronal"]

PANEL_BOTTOM = 0.08
PANEL_HEIGHT = 0.35
PANEL_GAP = 0.10
PANEL_TOP_BOTTOM = PANEL_BOTTOM + PANEL_HEIGHT + PANEL_GAP

FONT_AXIS_LABEL = 22
FONT_TICK = 18
FONT_CATEGORY = 20
FONT_VALUE = 18
FONT_LEGEND = 18


def value_alphas(values: pd.Series) -> list[float]:
    low, high = ALPHA_RANGE
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax == vmin:
        return [(low + high) / 2] * len(values)
    return (low + (values - vmin) / (vmax - vmin) * (high - low)).tolist()


def style_axis(ax: plt.Axes, xlabel: str) -> None:
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(xlabel, fontsize=FONT_AXIS_LABEL)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=FONT_TICK)
    ax.tick_params(axis="y", labelsize=FONT_CATEGORY)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig: plt.Figure, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=360)
    fig.savefig(out_png.with_suffix(".pdf"))
    plt.close(fig)


def plot_kidney_combined() -> None:
    df = pd.read_csv(KIDNEY_VALUES)
    fig = plt.figure(figsize=(6, 12))
    axes = [
        fig.add_axes([0.36, PANEL_TOP_BOTTOM, 0.58, PANEL_HEIGHT]),
        fig.add_axes([0.36, PANEL_BOTTOM, 0.58, PANEL_HEIGHT]),
    ]
    for ax, value_col, xlabel in zip(
        axes,
        ["image_iou_mean", "gene_iou_mean"],
        ["Image IoU", "Gene IoU"],
    ):
        d = df[["method_label", value_col]].dropna().sort_values(value_col, ascending=True)
        colors = [to_rgba(BAR_COLOR, alpha) for alpha in value_alphas(d[value_col])]
        ax.barh(
            d["method_label"],
            d[value_col],
            color=colors,
            edgecolor=to_rgba(BAR_COLOR, 0.95),
            linewidth=0.8,
            height=0.82,
        )
        style_axis(ax, xlabel)
        for i, v in enumerate(d[value_col]):
            ax.text(float(v) + 0.01, i, f"{v:.3f}", va="center", ha="left", fontsize=FONT_VALUE)

    save_figure(fig, KIDNEY_FIG_DIR / "human_kidney_updated_iou_barplot_combined.png")


def plot_heldout_combined() -> None:
    df = pd.read_csv(HELDOUT_CSV)
    df = df[df["sample"].isin(SAMPLE_ORDER)].copy()
    rows = []
    for sample in SAMPLE_ORDER:
        row = df.loc[df["sample"] == sample].iloc[0]
        for method in ["xeniumseg", "halo_heldout", "proseg"]:
            rows.append(
                {
                    "sample": sample,
                    "sample_label": SAMPLE_LABELS[sample],
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "image_iou": row[f"{method}_image_iou"],
                    "gene_iou": row[f"{method}_gene_iou"],
                }
            )
    long_df = pd.DataFrame(rows)
    long_df.to_csv(
        REVISION_FIG_DIR / "barplot_heldout3_xeniumseg_halo_heldout_proseg_iou_values.csv",
        index=False,
    )

    fig = plt.figure(figsize=(14, 12))
    axes = [
        fig.add_axes([0.08, PANEL_TOP_BOTTOM, 0.68, PANEL_HEIGHT]),
        fig.add_axes([0.08, PANEL_BOTTOM, 0.68, PANEL_HEIGHT]),
    ]
    x_positions = list(range(len(SAMPLE_ORDER)))
    bar_width = 0.23
    offsets = {"xeniumseg": -bar_width, "halo_heldout": 0.0, "proseg": bar_width}
    handles = []

    for ax, value_col, title in zip(axes, ["image_iou", "gene_iou"], ["Image IoU", "Gene IoU"]):
        for method in ["xeniumseg", "halo_heldout", "proseg"]:
            d = long_df[long_df["method"] == method].copy()
            xs = [x_positions[SAMPLE_ORDER.index(sample)] + offsets[method] for sample in d["sample"]]
            bars = ax.bar(
                xs,
                d[value_col],
                color=METHOD_COLORS[method],
                edgecolor=to_rgba(METHOD_COLORS[method], 0.95),
                linewidth=0.8,
                width=bar_width,
                label=METHOD_LABELS[method],
            )
            if ax is axes[0]:
                handles.append(bars[0])
            for x, v in zip(xs, d[value_col]):
                ax.text(x, float(v) + 0.008, f"{v:.3f}", va="bottom", ha="center", fontsize=FONT_VALUE)

        ax.set_title(title, fontsize=FONT_AXIS_LABEL)
        ax.set_ylabel("IoU", fontsize=FONT_AXIS_LABEL)
        ax.set_ylim(0.50, 0.86)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([SAMPLE_LABELS[sample] for sample in SAMPLE_ORDER], fontsize=FONT_CATEGORY)
        ax.tick_params(axis="y", labelsize=FONT_TICK)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.35)

    fig.legend(
        handles=handles,
        labels=[METHOD_LABELS[m] for m in ["xeniumseg", "halo_heldout", "proseg"]],
        loc="center left",
        bbox_to_anchor=(0.77, 0.74),
        ncol=1,
        frameon=False,
        fontsize=FONT_LEGEND,
        handlelength=1.6,
    )
    save_figure(fig, REVISION_FIG_DIR / "barplot_heldout3_xeniumseg_halo_heldout_proseg_iou.png")


def main() -> None:
    global KIDNEY_FIG_DIR, REVISION_FIG_DIR, HELDOUT_CSV, KIDNEY_VALUES
    parser = argparse.ArgumentParser()
    parser.add_argument("--kidney-values", type=Path, required=True)
    parser.add_argument("--heldout-csv", type=Path, required=True)
    parser.add_argument("--kidney-out-dir", type=Path, required=True)
    parser.add_argument("--heldout-out-dir", type=Path, required=True)
    args = parser.parse_args()
    KIDNEY_VALUES = args.kidney_values
    HELDOUT_CSV = args.heldout_csv
    KIDNEY_FIG_DIR = args.kidney_out_dir
    REVISION_FIG_DIR = args.heldout_out_dir
    plot_kidney_combined()
    plot_heldout_combined()


if __name__ == "__main__":
    main()
