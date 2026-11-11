#!/usr/bin/env python3
"""Draw a multi-method paired boxplot for Image/Gene IoU."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon


METHODS = ["cellposesam", "proseg", "rna2seg", "baysor", "ne"]
SIGNIFICANCE_ORDER = ["baysor", "rna2seg", "ne", "proseg"]
LABELS = {
    "cellposesam": "Halo",
    "proseg": "Proseg",
    "rna2seg": "RNA2seg",
    "baysor": "Baysor",
    "ne": "NE",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 14,
            "legend.title_fontsize": 14,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def p_to_stars(p_val: float) -> str:
    if not np.isfinite(p_val):
        return "ns"
    if p_val < 1e-4:
        return "****"
    if p_val < 1e-3:
        return "***"
    if p_val < 1e-2:
        return "**"
    if p_val < 0.05:
        return "*"
    return "ns"


def paired_stats(df: pd.DataFrame, metric: str, baseline: str = "cellposesam") -> pd.DataFrame:
    rows = []
    base_col = f"{metric}_mean_{baseline}"
    for method in METHODS:
        if method == baseline:
            continue
        col = f"{metric}_mean_{method}"
        paired = df[[base_col, col]].dropna()
        if len(paired) >= 2:
            try:
                stat, p_val = wilcoxon(
                    paired[base_col].to_numpy(),
                    paired[col].to_numpy(),
                    zero_method="wilcox",
                    alternative="two-sided",
                )
            except ValueError:
                stat, p_val = np.nan, np.nan
        else:
            stat, p_val = np.nan, np.nan
        rows.append(
            {
                "metric": metric,
                "comparison": f"{baseline}_vs_{method}",
                "n": int(len(paired)),
                "baseline_mean": float(paired[base_col].mean()) if len(paired) else np.nan,
                "method_mean": float(paired[col].mean()) if len(paired) else np.nan,
                "mean_delta_method_minus_baseline": float((paired[col] - paired[base_col]).mean()) if len(paired) else np.nan,
                "wilcoxon_stat": float(stat) if not np.isnan(stat) else np.nan,
                "wilcoxon_p": float(p_val) if not np.isnan(p_val) else np.nan,
                "significance": p_to_stars(float(p_val)) if np.isfinite(p_val) else "ns",
            }
        )
    return pd.DataFrame(rows)


def add_significance_labels(
    ax,
    stats: pd.DataFrame,
    method_positions: dict[str, int],
    baseline: str = "cellposesam",
) -> None:
    if baseline not in method_positions:
        return
    x0 = method_positions[baseline]
    y_top = 1.12
    step = 0.070
    comparisons = [m for m in SIGNIFICANCE_ORDER if m != baseline and m in method_positions]
    for i, method in enumerate(comparisons):
        row = stats.loc[stats["comparison"] == f"{baseline}_vs_{method}"]
        if row.empty:
            continue
        stars = str(row.iloc[0]["significance"])
        x1 = method_positions[method]
        left, right = sorted([x0, x1])
        y = y_top - i * step
        ax.plot([left, right], [y, y], color="black", lw=1.1, clip_on=False)
        ax.text((left + right) / 2, y + 0.006, stars, ha="center", va="bottom", fontsize=9, color="black")


def draw_panel(ax, df: pd.DataFrame, metric: str, color_map: dict[str, object], seed: int) -> None:
    medians = {
        method: pd.to_numeric(df[f"{metric}_mean_{method}"], errors="coerce").median()
        for method in METHODS
    }
    methods = sorted(METHODS, key=lambda m: (-medians[m], METHODS.index(m)))
    cols = [f"{metric}_mean_{m}" for m in methods]
    vals = [pd.to_numeric(df[c], errors="coerce").dropna().to_numpy() for c in cols]
    bp = ax.boxplot(vals, positions=np.arange(len(methods)), patch_artist=False, showfliers=False, widths=0.42)
    for median in bp["medians"]:
        median.set(linewidth=2)

    rng = np.random.default_rng(seed)
    for _, row in df.iterrows():
        tissue = str(row["dataset"])
        color = color_map.get(tissue, "C0")
        xs = []
        ys = []
        for i, col in enumerate(cols):
            y = pd.to_numeric(row[col], errors="coerce")
            if pd.notna(y):
                x = i + rng.uniform(-0.045, 0.045)
                xs.append(x)
                ys.append(float(y))
                ax.scatter(x, y, s=38, alpha=0.92, edgecolor="none", c=[color], zorder=3)
        if len(xs) >= 2:
            ax.plot(xs, ys, color=color, alpha=0.22, lw=1.0, zorder=2)

    ax.set_xticks(np.arange(len(methods)), [LABELS[m] for m in methods])
    ax.set_ylabel("IoU")
    stats = paired_stats(df, metric)
    add_significance_labels(ax, stats, {m: i for i, m in enumerate(methods)})
    ax.set_ylim(0, 1.20)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def plot_four_method(df: pd.DataFrame, out_dir: Path) -> None:
    apply_style()
    missing = [f"{metric}_mean_{m}" for metric in ("image_iou", "gene_iou") for m in METHODS if f"{metric}_mean_{m}" not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    tissues = pd.unique(df["dataset"].astype(str))
    cmap = plt.get_cmap("tab20")
    color_map = {t: cmap(i % 20) for i, t in enumerate(tissues)}

    fig, axes = plt.subplots(2, 1, figsize=(7, 10), sharey=True)
    draw_panel(axes[0], df, "image_iou", color_map, 42)
    draw_panel(axes[1], df, "gene_iou", color_map, 43)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=6, markeredgewidth=0.3, markeredgecolor="black", color=color_map[t], label=t)
        for t in sorted(tissues, key=lambda x: str(x).lower())
    ]
    fig.legend(
        handles,
        [h.get_label() for h in handles],
        loc="center left",
        bbox_to_anchor=(0.98, 0.5),
        ncol=1,
        frameon=False,
        prop={"size": 20},
    )
    fig.subplots_adjust(hspace=0.35)
    fig.tight_layout(rect=(0, 0.08, 0.88, 0.98))

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "boxplot_methods_halo_proseg_rna2seg_baysor_ne.png"
    pdf = out_dir / "boxplot_methods_halo_proseg_rna2seg_baysor_ne.pdf"
    stats_fp = out_dir / "boxplot_methods_halo_proseg_rna2seg_baysor_ne_stats.csv"
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    stats = pd.concat([paired_stats(df, "image_iou"), paired_stats(df, "gene_iou")], ignore_index=True)
    stats.to_csv(stats_fp, index=False)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print(f"Wrote {stats_fp}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-wide", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    plot_four_method(pd.read_csv(args.summary_wide), args.out_dir)


if __name__ == "__main__":
    main()
