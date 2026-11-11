#!/usr/bin/env python3
"""Plot five-method clustering agreement against TenX ground truth at resolution 1.0."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import wilcoxon


METHOD_ORDER = ["cpsam", "ne", "proseg", "rna2seg", "baysor"]
SIGNIFICANCE_ORDER = ["baysor", "rna2seg", "proseg", "ne"]
METHOD_LABELS = {
    "cpsam": "Halo",
    "ne": "NE",
    "proseg": "Proseg",
    "rna2seg": "RNA2seg",
    "baysor": "Baysor",
}
METRICS = ["ARI", "AMI", "Homogeneity", "Completeness"]
TISSUE_LABELS = {
    "human_brain": "glioblastoma",
    "human_breast": "breast cancer",
    "human_cervical": "cervical cancer",
    "human_lung": "lung adenocarcinoma",
    "human_lymph_node": "lymph node",
    "human_ovarian": "ovarian cancer",
    "human_pancreas": "pancreatic cancer",
    "human_prostate": "prostate cancer",
    "human_skin": "melanoma",
    "mouse_brain_coronal": "mouse brain coronal",
    "mouse_colon": "mouse colon",
    "mouse_pup": "mouse pup",
}
TISSUE_ORDER = [
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


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 15,
            "legend.title_fontsize": 16,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def p_to_stars(p_val: float) -> str:
    if not np.isfinite(p_val):
        return "n.s."
    if p_val < 1e-4:
        return "****"
    if p_val < 1e-3:
        return "***"
    if p_val < 1e-2:
        return "**"
    if p_val < 0.05:
        return "*"
    return "n.s."


def load_tissues(path: Path) -> list[str]:
    tissues = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tissues.append(line.split("\t")[0])
    return tissues


def load_metrics(tissues: list[str], cluster_root: Path, method_gt_root: Path, res_label: str) -> pd.DataFrame:
    rows = []
    for tissue in tissues:
        for method in ["cpsam", "ne"]:
            p = cluster_root / tissue / f"precluster_res{res_label}" / method / "metrics.csv"
            if p.exists():
                rows.append(pd.read_csv(p).assign(method=method, tissue=tissue))
        for method in ["baysor", "proseg", "rna2seg"]:
            p = method_gt_root / tissue / f"precluster_res{res_label}" / method / "metrics.csv"
            if p.exists():
                rows.append(pd.read_csv(p).assign(method=method, tissue=tissue))
    if not rows:
        raise FileNotFoundError("No metrics.csv files found.")
    df = pd.concat(rows, ignore_index=True)
    rename = {
        "ari": "ARI",
        "ami": "AMI",
        "homogeneity": "Homogeneity",
        "completeness": "Completeness",
        "n_matches_total": "n_matches",
        "n_matches_with_clusters": "n_used",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    for col in METRICS + ["n_matches", "n_used"]:
        same = df.loc[:, df.columns == col]
        if same.shape[1] > 1:
            df = df.drop(columns=[col])
            df[col] = same.bfill(axis=1).iloc[:, 0]
    df["method"] = df["method"].astype(str)
    df["method_label"] = df["method"].map(METHOD_LABELS).fillna(df["method"])
    return df


def paired_stats(df: pd.DataFrame, baseline: str = "cpsam") -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        wide = df.pivot_table(index="tissue", columns="method", values=metric, aggfunc="first")
        for method in METHOD_ORDER:
            if method == baseline:
                continue
            paired = wide[[baseline, method]].dropna() if baseline in wide and method in wide else pd.DataFrame()
            if len(paired) >= 2:
                try:
                    stat, p_val = wilcoxon(
                        paired[baseline].to_numpy(),
                        paired[method].to_numpy(),
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
                    "baseline": baseline,
                    "method": method,
                    "n": int(len(paired)),
                    "baseline_mean": float(paired[baseline].mean()) if len(paired) else np.nan,
                    "method_mean": float(paired[method].mean()) if len(paired) else np.nan,
                    "mean_delta_method_minus_baseline": float((paired[method] - paired[baseline]).mean()) if len(paired) else np.nan,
                    "wilcoxon_stat": float(stat) if np.isfinite(stat) else np.nan,
                    "wilcoxon_p": float(p_val) if np.isfinite(p_val) else np.nan,
                    "significance": p_to_stars(float(p_val)) if np.isfinite(p_val) else "n.s.",
                }
            )
    return pd.DataFrame(rows)


def add_significance_labels(
    ax,
    stats: pd.DataFrame,
    metric: str,
    method_positions: dict[str, int],
    baseline: str = "cpsam",
) -> None:
    if baseline not in method_positions:
        return

    x0 = method_positions[baseline]
    y_top = 1.130
    step = 0.055
    comparisons = [m for m in SIGNIFICANCE_ORDER if m != baseline and m in method_positions]
    for i, method in enumerate(comparisons):
        row = stats.loc[
            (stats["metric"] == metric)
            & (stats["baseline"] == baseline)
            & (stats["method"] == method)
        ]
        if row.empty:
            continue
        stars = str(row.iloc[0]["significance"])
        x1 = method_positions[method]
        left, right = sorted([x0, x1])
        y = y_top - i * step
        ax.plot([left, right], [y, y], color="black", lw=1.1, clip_on=False)
        ax.text((left + right) / 2, y + 0.005, stars, ha="center", va="bottom", fontsize=8, color="black")


def draw_metric(ax, df: pd.DataFrame, metric: str, tissue_colors: dict[str, object], seed: int) -> None:
    methods = [m for m in METHOD_ORDER if m in set(df["method"])]
    data = [
        pd.to_numeric(df.loc[df["method"] == method, metric], errors="coerce").dropna().to_numpy()
        for method in methods
    ]
    bp = ax.boxplot(data, positions=np.arange(len(methods)), widths=0.46, patch_artist=True, showfliers=False)
    for box, method in zip(bp["boxes"], methods):
        box.set(facecolor="#f7f7f7", edgecolor="#303030", linewidth=1.2)
    for median in bp["medians"]:
        median.set(color="#111111", linewidth=2)
    for element in ["whiskers", "caps"]:
        for item in bp[element]:
            item.set(color="#303030", linewidth=1.1)

    rng = np.random.default_rng(seed)
    wide = df.pivot_table(index="tissue", columns="method", values=metric, aggfunc="first")
    for tissue, row in wide.iterrows():
        xs = []
        ys = []
        for i, method in enumerate(methods):
            val = row.get(method, np.nan)
            if pd.notna(val):
                x = i + rng.uniform(-0.055, 0.055)
                xs.append(x)
                ys.append(float(val))
                ax.scatter(
                    x,
                    float(val),
                    s=32,
                    color=tissue_colors.get(str(tissue), "C0"),
                    edgecolor="white",
                    linewidth=0.35,
                    zorder=3,
                )
        if len(xs) >= 2:
            ax.plot(xs, ys, color="#8a8a8a", alpha=0.23, linewidth=0.8, zorder=2)

    ax.set_title(metric)
    ax.set_xticks(np.arange(len(methods)), [METHOD_LABELS[m] for m in methods], rotation=22, ha="right")
    stats = paired_stats(df)
    add_significance_labels(ax, stats, metric, {m: i for i, m in enumerate(methods)})
    finite_vals = pd.to_numeric(df[metric], errors="coerce").dropna().to_numpy()
    if len(finite_vals):
        y_min = max(0.0, np.floor((float(np.nanmin(finite_vals)) - 0.04) / 0.05) * 0.05)
    else:
        y_min = 0.0
    ax.set_ylim(y_min, 1.20)
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color("#202020")


def plot(df: pd.DataFrame, out_dir: Path, prefix: str) -> None:
    apply_style()
    missing = sorted(set(METHOD_ORDER) - set(df["method"]))
    if missing:
        raise ValueError(f"Missing methods in metrics table: {missing}")

    present_tissues = set(df["tissue"].astype(str).unique())
    tissues = [t for t in TISSUE_ORDER if t in present_tissues]
    tissues.extend(sorted(present_tissues - set(tissues), key=lambda t: TISSUE_LABELS.get(t, t).lower()))
    cmap = plt.get_cmap("tab20")
    tissue_colors = {t: cmap(i % 20) for i, t in enumerate(TISSUE_ORDER)}
    for tissue in tissues:
        tissue_colors.setdefault(tissue, cmap(len(tissue_colors) % 20))

    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5), sharey=False)
    for ax, metric, seed in zip(axes.ravel(), METRICS, [11, 12, 13, 14]):
        draw_metric(ax, df, metric, tissue_colors, seed)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=7.0, color=tissue_colors[t], label=TISSUE_LABELS.get(t, t))
        for t in tissues
    ]
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.875, 0.5),
        frameon=False,
        fontsize=15,
        title_fontsize=16,
        handletextpad=0.45,
        labelspacing=0.68,
    )
    fig.tight_layout(rect=(0, 0, 0.84, 1), w_pad=3.2, h_pad=3.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{prefix}.png"
    pdf = out_dir / f"{prefix}.pdf"
    fig.savefig(png, dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    df.to_csv(out_dir / f"{prefix}_long_metrics.csv", index=False)
    summary = df.groupby("method", sort=False)[METRICS + ["n_matches", "n_used"]].mean().reset_index()
    summary["method_label"] = summary["method"].map(METHOD_LABELS)
    summary.to_csv(out_dir / f"{prefix}_method_means.csv", index=False)
    paired_stats(df).to_csv(out_dir / f"{prefix}_halo_paired_stats.csv", index=False)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundles", type=Path, required=True)
    ap.add_argument("--cluster-root", type=Path, required=True)
    ap.add_argument("--method-gt-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--resolution-label", default="1")
    ap.add_argument("--prefix", default="five_methods_vs_groundtruth_res1_2x2")
    args = ap.parse_args()

    tissues = load_tissues(args.bundles)
    df = load_metrics(tissues, args.cluster_root, args.method_gt_root, args.resolution_label)
    plot(df, args.out_dir, args.prefix)


if __name__ == "__main__":
    main()
