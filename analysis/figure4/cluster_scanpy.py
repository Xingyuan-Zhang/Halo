#!/usr/bin/env python3
"""
Matched Xenium analysis with Scanpy clustering, QC, and target #clusters.

Keeps:
- matched-pair filtering
- QC filtering
- target-k Leiden clustering (resolution tuning)
- ARI between A/B cluster labels
- per-gene A↔B correlations (Pearson + Spearman across matched cells)

Tile restriction:
- Restrict matched cells to those whose centroids fall inside any rectangular tile
  in $HALO_TILES_ROOT/{tissue}/test/tiles_meta.csv
  where tissue is inferred from --out path: .../downstream/cluster/<tissue>/<method>/result_scanpy

IMPORTANT:
- tiles_meta.csv stores pixel coordinates; multiply by 0.2125 to convert to microns.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import scanpy as sc
import anndata as ad
from scipy import sparse as sp
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from scipy.stats import spearmanr, pearsonr


# ---------------- Gene name normalization ----------------

def normalize_gene_series(s: pd.Series, case: str = "upper", strip_version: bool = True) -> pd.Series:
    s = s.astype(str).str.strip()
    if strip_version:
        s = s.str.replace(r"\.\d+$", "", regex=True)  # ENSG... .v
    if case == "upper":
        s = s.str.upper()
    elif case == "lower":
        s = s.str.lower()
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


# ---------------- I/O helpers ----------------

def read_csv_auto(path: Path) -> pd.DataFrame:
    if str(path).endswith(".gz"):
        return pd.read_csv(path, compression="infer")
    return pd.read_csv(path)

def to_csv_auto(df: pd.DataFrame, path: Path, **kw):
    if str(path).endswith(".gz"):
        df.to_csv(path, index=False, compression="infer", **kw)
    else:
        df.to_csv(path, index=False, **kw)

def find_transcripts_or_counts(folder: Path) -> dict:
    p = folder / "transcripts.parquet"
    if p.exists():
        return {"type": "transcripts_parquet", "path": p}
    for name in ("transcripts.csv.gz", "transcripts.csv"):
        p = folder / name
        if p.exists():
            return {"type": "transcripts_csv", "path": p}
    for name in ("gene_counts.csv.gz", "gene_counts.csv", "counts.csv.gz", "counts.csv"):
        p = folder / name
        if p.exists():
            return {"type": "counts", "path": p}
    raise FileNotFoundError(
        f"No transcript table found in {folder}. Expected transcripts.parquet or transcripts.csv(.gz)."
    )

def _normalize_transcripts_df(df: pd.DataFrame) -> pd.DataFrame:
    col = {c.lower(): c for c in df.columns}
    cid = col.get("cell_id") or col.get("cell") or col.get("cellid") or col.get("barcode")
    if cid is None:
        raise ValueError(f"Transcripts table missing cell_id column. Columns: {list(df.columns)}")
    gcol = col.get("feature_name") or col.get("gene") or col.get("target")
    if gcol is None:
        raise ValueError(f"Transcripts table missing gene/feature_name column. Columns: {list(df.columns)}")
    out = df[[cid, gcol]].copy()
    out.columns = ["cell_id", "gene"]
    out["cell_id"] = out["cell_id"].astype(str)
    out["gene"] = out["gene"].astype(str)
    return out

def build_counts_from_transcripts_parquet(path: Path, wanted_cells: set[str]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = _normalize_transcripts_df(df)
    if wanted_cells:
        df = df[df["cell_id"].isin(wanted_cells)]
    counts = (df.groupby(["cell_id", "gene"], sort=False).size()
              .rename("count").reset_index())
    return counts

def build_counts_from_transcripts_csv(path: Path, wanted_cells: set[str]) -> pd.DataFrame:
    df = read_csv_auto(path)
    df = _normalize_transcripts_df(df)
    if wanted_cells:
        df = df[df["cell_id"].isin(wanted_cells)]
    counts = (df.groupby(["cell_id", "gene"], sort=False).size()
              .rename("count").reset_index())
    return counts

def build_counts_from_generic(path: Path, wanted_cells: set[str]) -> pd.DataFrame:
    df = read_csv_auto(path)
    low = {c.lower(): c for c in df.columns}
    if {"cell_id", "gene"}.issubset(low) and ("count" in low or "value" in low):
        cid, gcol = low["cell_id"], low["gene"]
        ccol = low.get("count", low.get("value"))
        if wanted_cells:
            df = df[df[cid].astype(str).isin(wanted_cells)]
        out = df[[cid, gcol, ccol]].copy()
        out.columns = ["cell_id", "gene", "count"]
        out["cell_id"] = out["cell_id"].astype(str)
        out["gene"] = out["gene"].astype(str)
        return out
    if "gene" in low:  # wide → long
        gcol = low["gene"]
        df[gcol] = df[gcol].astype(str)
        df = df.set_index(gcol)
        if wanted_cells:
            keep = [c for c in df.columns if str(c) in wanted_cells]
            df = df[keep]
        out = df.stack().reset_index()
        out.columns = ["gene", "cell_id", "count"]
        out["cell_id"] = out["cell_id"].astype(str)
        out["gene"] = out["gene"].astype(str)
        return out
    else:  # first column is cell_id, wide → long
        first = df.columns[0]
        df[first] = df[first].astype(str)
        df = df.set_index(first)
        if wanted_cells:
            df = df[df.index.astype(str).isin(wanted_cells)]
        out = df.stack().reset_index()
        out.columns = ["cell_id", "gene", "count"]
        out["cell_id"] = out["cell_id"].astype(str)
        out["gene"] = out["gene"].astype(str)
        return out

def load_counts(folder: Path, wanted_cells: set[str]) -> pd.DataFrame:
    src = find_transcripts_or_counts(folder)
    if src["type"] == "transcripts_parquet":
        return build_counts_from_transcripts_parquet(src["path"], wanted_cells)
    if src["type"] == "transcripts_csv":
        return build_counts_from_transcripts_csv(src["path"], wanted_cells)
    if src["type"] == "counts":
        return build_counts_from_generic(src["path"], wanted_cells)
    raise RuntimeError("Unknown source type")


# ---------------- Tile restriction helpers ----------------

TILES_SUB_ROOT = Path(os.environ.get("HALO_TILES_ROOT", "tiles_sub"))
PX_TO_UM = 0.2125  # tiles are in pixels; convert to microns

def _infer_tissue_from_out(out_dir: Path) -> str:
    # expected: .../downstream/cluster/<tissue>/<method>/result_scanpy
    try:
        return out_dir.parent.parent.name
    except Exception:
        return ""

def _load_cells_centroids(folder: Path, wanted_cells: set[str]) -> pd.DataFrame:
    # Try common Xenium tables: cells.csv(.gz) with micron coordinates
    for name in ("cells.csv.gz", "cells.csv"):
        p = folder / name
        if p.exists():
            df = read_csv_auto(p)
            low = {c.lower(): c for c in df.columns}

            cid = low.get("cell_id") or low.get("cellid") or low.get("cell") or low.get("barcode")
            xcol = low.get("x_centroid") or low.get("x") or low.get("x_center") or low.get("xcentre") or low.get("xcenter")
            ycol = low.get("y_centroid") or low.get("y") or low.get("y_center") or low.get("ycentre") or low.get("ycenter")

            if cid is None or xcol is None or ycol is None:
                raise ValueError(
                    f"{p} missing required centroid columns. "
                    f"Need cell_id + x_centroid/y_centroid (or x/y). Columns: {list(df.columns)}"
                )

            out = df[[cid, xcol, ycol]].copy()
            out.columns = ["cell_id", "x", "y"]
            out["cell_id"] = out["cell_id"].astype(str)
            out["x"] = pd.to_numeric(out["x"], errors="coerce")
            out["y"] = pd.to_numeric(out["y"], errors="coerce")

            if wanted_cells:
                out = out[out["cell_id"].isin(wanted_cells)]

            out = out.dropna(subset=["x", "y"])
            return out

    raise FileNotFoundError(f"No cells.csv(.gz) found under: {folder}")

def _load_tiles_meta(tissue: str) -> pd.DataFrame:
    p = TILES_SUB_ROOT / tissue / "test" / "tiles_meta.csv"
    if not p.exists():
        raise FileNotFoundError(f"tiles_meta.csv not found: {p}")
    df = read_csv_auto(p)
    low = {c.lower(): c for c in df.columns}

    # accept several common naming conventions
    x0 = low.get("x0") or low.get("xmin") or low.get("x_min") or low.get("x_start") or low.get("left")
    x1 = low.get("x1") or low.get("xmax") or low.get("x_max") or low.get("x_end") or low.get("right")
    y0 = low.get("y0") or low.get("ymin") or low.get("y_min") or low.get("y_start") or low.get("top")
    y1 = low.get("y1") or low.get("ymax") or low.get("y_max") or low.get("y_end") or low.get("bottom")

    if x0 is None or x1 is None or y0 is None or y1 is None:
        raise ValueError(
            f"tiles_meta.csv missing rectangle columns. Need x0/x1/y0/y1 (or xmin/xmax/ymin/ymax etc). "
            f"Columns: {list(df.columns)}"
        )

    out = df[[x0, x1, y0, y1]].copy()
    out.columns = ["x0", "x1", "y0", "y1"]
    for c in ("x0", "x1", "y0", "y1"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["x0", "x1", "y0", "y1"])

    # ---- NEW: convert pixel coords -> microns ----
    out[["x0", "x1", "y0", "y1"]] = out[["x0", "x1", "y0", "y1"]] * float(PX_TO_UM)

    return out

def _cells_in_any_tile(centroids: pd.DataFrame, tiles: pd.DataFrame) -> set[str]:
    x = centroids["x"].to_numpy(dtype=float, copy=False)
    y = centroids["y"].to_numpy(dtype=float, copy=False)
    keep = np.zeros(x.shape[0], dtype=bool)

    # inclusive-left/top, exclusive-right/bottom (standard half-open box)
    for x0, x1, y0, y1 in tiles[["x0", "x1", "y0", "y1"]].itertuples(index=False, name=None):
        keep |= (x >= x0) & (x < x1) & (y >= y0) & (y < y1)

    return set(centroids.loc[keep, "cell_id"].astype(str).tolist())


# ---------------- Build AnnData from long counts ----------------

def counts_to_adata(counts: pd.DataFrame) -> ad.AnnData:
    counts["cell_id"] = counts["cell_id"].astype(str)
    counts["gene"] = counts["gene"].astype(str)
    cells = counts["cell_id"].unique().tolist()
    genes = counts["gene"].unique().tolist()
    cell_index = {c: i for i, c in enumerate(cells)}
    gene_index = {g: j for j, g in enumerate(genes)}
    rows = np.fromiter((cell_index[c] for c in counts["cell_id"]), dtype=np.int32, count=len(counts))
    cols = np.fromiter((gene_index[g] for g in counts["gene"]), dtype=np.int32, count=len(counts))
    data = counts["count"].astype(np.float32).to_numpy(copy=False)
    X = sp.csr_matrix((data, (rows, cols)), shape=(len(cells), len(genes)), dtype=np.float32)
    return ad.AnnData(X, obs=pd.DataFrame(index=cells), var=pd.DataFrame(index=genes))


# ---------------- Scanpy clustering with target-k via resolution tuning ----------------

def _leiden_with_resolution(adata: ad.AnnData, resolution: float, random_state: int,
                            flavor: str = "igraph", directed: bool = False, key: str = "cluster") -> pd.Series:
    sc.tl.leiden(adata, resolution=resolution, random_state=random_state,
                 flavor=flavor, directed=directed, n_iterations=2, key_added=key)
    return adata.obs[key]

def _labels_to_int(labels: pd.Series) -> np.ndarray:
    if labels.dtype.name == "category":
        cats = list(labels.cat.categories)
        mapping = {cat: i for i, cat in enumerate(cats)}
        return labels.astype(str).map(mapping).to_numpy(dtype=int)
    uniq = pd.Index(labels.astype(str).unique().tolist())
    mapping = {u: i for i, u in enumerate(uniq)}
    return labels.astype(str).map(mapping).to_numpy(dtype=int)

def _fit_resolution_for_k(adata_in: ad.AnnData, target_k: int, random_state: int = 0,
                          max_iter: int = 10, init_resolution: float = 1.0) -> Tuple[np.ndarray, float, int]:
    adata_base = adata_in.copy()
    best = None
    res = max(1e-3, float(init_resolution))
    for _ in range(max_iter):
        adata = adata_base.copy()
        try:
            labels = _leiden_with_resolution(adata, res, random_state)
        except Exception:
            sc.tl.leiden(adata, resolution=res, random_state=random_state, key_added="cluster")
            labels = adata.obs["cluster"]
        arr = _labels_to_int(labels)
        k_now = int(np.unique(arr).size)
        diff = abs(k_now - target_k)
        if (best is None) or (diff < best[0]):
            best = (diff, arr.copy(), res, k_now)
            if diff == 0:
                return arr, res, k_now
        factor = 2.0 if k_now == 0 else np.clip(target_k / max(k_now, 1), 0.25, 4.0)
        factor = 0.5 * factor + 0.5
        res = float(np.clip(res * factor, 1e-3, 50.0))
    return best[1], best[2], best[3]

def cluster_with_scanpy(adata_in: ad.AnnData, target_k: int, n_neighbors: int,
                        random_state: int = 0) -> Tuple[np.ndarray, int, float]:
    n_cells = adata_in.n_obs
    n_comp = int(min(50, max(2, n_cells - 1)))
    adata = adata_in.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.pca(adata, n_comps=n_comp, svd_solver="arpack")
    sc.pp.neighbors(adata, n_neighbors=min(n_neighbors, max(1, n_cells - 1)),
                    metric="euclidean", method="gauss")
    labels, res_used, k_now = _fit_resolution_for_k(adata, target_k=target_k,
                                                    random_state=random_state,
                                                    max_iter=10, init_resolution=1.0)
    if k_now != target_k:
        print(f"[WARN] Requested k={target_k}, achieved {k_now} clusters (resolution={res_used:.4f}).")
    else:
        print(f"[INFO] Achieved exactly k={k_now} clusters (resolution={res_used:.4f}).")
    return labels, k_now, res_used


# ---------------- Per-gene A↔B correlations ----------------

def _corr_chunk(args):
    j0, j1, A_csc, B_csc, genes = args
    out = []
    for j in range(j0, j1):
        a = A_csc.getcol(j).toarray().ravel()
        b = B_csc.getcol(j).toarray().ravel()
        if np.all(a == 0) and np.all(b == 0):
            out.append((genes[j], np.nan, np.nan, 0)); continue
        if (a.std() == 0) and (b.std() == 0):
            out.append((genes[j], np.nan, np.nan, int(len(a)))); continue
        pr = pearsonr(a, b)[0] if (a.std() > 0 or b.std() > 0) else np.nan
        sr = spearmanr(a, b).correlation
        out.append((genes[j], pr, sr, int(len(a))))
    return out

def per_gene_correlations_parallel(MA: sp.csr_matrix, MB: sp.csr_matrix, genes: List[str],
                                   workers: int = 4, chunk_cols: int = 512) -> pd.DataFrame:
    assert MA.shape == MB.shape
    A_csc = MA.tocsc(); B_csc = MB.tocsc()
    n_genes = A_csc.shape[1]
    chunks = [(j0, min(n_genes, j0 + chunk_cols), A_csc, B_csc, genes)
              for j0 in range(0, n_genes, chunk_cols)]
    out_rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_corr_chunk, c) for c in chunks]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Per-gene correlations"):
            out_rows.extend(fut.result())
    df = pd.DataFrame(out_rows, columns=["gene", "pearson_r", "spearman_r", "n_valid"])
    df.index = pd.CategoricalIndex(df["gene"], categories=genes, ordered=True)
    return df.sort_index().reset_index(drop=True)


# ---------------- QC summary helpers ----------------

def qc_summary_dict(args, cellsA_before, cellsA_after, cellsA_filtered,
                    cellsB_before, cellsB_after, cellsB_filtered,
                    genesA_before, genesA_after, genesA_filtered,
                    genesB_before, genesB_after, genesB_filtered,
                    n_pairs_csv, n_pairs_before, n_pairs_after,
                    shared_genes_before, shared_genes_after) -> Dict[str, Any]:
    pct = lambda num, den: (100.0 * num / den) if den > 0 else 0.0
    return {
        "min_counts_per_cell": args.min_counts_per_cell,
        "min_genes_per_cell": args.min_genes_per_cell,
        "min_cells_per_gene": args.min_cells_per_gene,
        "min_counts_per_gene": args.min_counts_per_gene,
        "cells_A_before": int(cellsA_before),
        "cells_A_after": int(cellsA_after),
        "cells_A_filtered": int(cellsA_filtered),
        "cells_A_filtered_pct": round(pct(cellsA_filtered, cellsA_before), 2),
        "cells_B_before": int(cellsB_before),
        "cells_B_after": int(cellsB_after),
        "cells_B_filtered": int(cellsB_filtered),
        "cells_B_filtered_pct": round(pct(cellsB_filtered, cellsB_before), 2),
        "genes_A_before": int(genesA_before),
        "genes_A_after": int(genesA_after),
        "genes_A_filtered": int(genesA_filtered),
        "genes_A_filtered_pct": round(pct(genesA_filtered, genesA_before), 2),
        "genes_B_before": int(genesB_before),
        "genes_B_after": int(genesB_after),
        "genes_B_filtered": int(genesB_filtered),
        "genes_B_filtered_pct": round(pct(genesB_filtered, genesB_before), 2),
        "matched_pairs_in_csv": int(n_pairs_csv),
        "matched_pairs_present_before_qc": int(n_pairs_before),
        "matched_pairs_after_qc": int(n_pairs_after),
        "shared_genes_before_qc": int(shared_genes_before),
        "shared_genes_after_qc": int(shared_genes_after),
    }

def write_qc_summary(out_dir: Path, qc: Dict[str, Any]):
    pd.DataFrame([qc]).to_csv(out_dir / "qc_summary.tsv", sep="\t", index=False)

def print_qc_summary(qc: Dict[str, Any]):
    print("\n=== QC summary ===")
    print(f"Cells A: {qc['cells_A_before']} → {qc['cells_A_after']}  (filtered {qc['cells_A_filtered']} = {qc['cells_A_filtered_pct']}%)")
    print(f"Cells B: {qc['cells_B_before']} → {qc['cells_B_after']}  (filtered {qc['cells_B_filtered']} = {qc['cells_B_filtered_pct']}%)")
    print(f"Genes A: {qc['genes_A_before']} → {qc['genes_A_after']}  (filtered {qc['genes_A_filtered']} = {qc['genes_A_filtered_pct']}%)")
    print(f"Genes B: {qc['genes_B_before']} → {qc['genes_B_after']}  (filtered {qc['genes_B_filtered']} = {qc['genes_B_filtered_pct']}%)")
    print(f"Matched pairs: CSV={qc['matched_pairs_in_csv']}, present before QC={qc['matched_pairs_present_before_qc']}, after QC={qc['matched_pairs_after_qc']}")
    print(f"Shared genes: before QC={qc['shared_genes_before_qc']}, after QC={qc['shared_genes_after_qc']}")


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(
        description="Scanpy clustering with QC + ARI + per-gene correlations on matched Xenium cells."
    )
    ap.add_argument("dataset_A", type=Path)
    ap.add_argument("dataset_B", type=Path)
    ap.add_argument("matches_csv", type=Path, help="CSV with columns a_id,b_id (string cell IDs)")

    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--corr-chunk", type=int, default=512)

    ap.add_argument("--min-counts-per-cell", type=int, default=0)
    ap.add_argument("--min-genes-per-cell", type=int, default=0)
    ap.add_argument("--min-cells-per-gene", type=int, default=0)
    ap.add_argument("--min-counts-per-gene", type=int, default=0)

    ap.add_argument("--out", type=Path, default=Path("results_scanpy_qc"))
    ap.add_argument("--save-matrix", action="store_true")
    ap.add_argument("--random-state", type=int, default=0)
    args = ap.parse_args()

    # Thread caps
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "debug").mkdir(parents=True, exist_ok=True)

    # Matches
    m = pd.read_csv(args.matches_csv)
    if not {"a_id", "b_id"}.issubset(set(m.columns)):
        raise ValueError("matches_csv must contain columns: a_id,b_id")
    m = m.drop_duplicates(subset=["a_id", "b_id"]).astype({"a_id": str, "b_id": str})

    # -------- Restrict to cells inside tile rectangles --------
    tissue = _infer_tissue_from_out(args.out)
    if tissue:
        try:
            tiles = _load_tiles_meta(tissue)

            matched_A_all0 = set(m["a_id"].astype(str))
            matched_B_all0 = set(m["b_id"].astype(str))

            centA = _load_cells_centroids(args.dataset_A, matched_A_all0)
            centB = _load_cells_centroids(args.dataset_B, matched_B_all0)

            keepA = _cells_in_any_tile(centA, tiles)
            keepB = _cells_in_any_tile(centB, tiles)

            m = m[m["a_id"].astype(str).isin(keepA) & m["b_id"].astype(str).isin(keepB)].copy()

        except Exception as e:
            print(f"[TILES] Skip tile restriction (reason: {e})")
    else:
        print("[TILES] Skip tile restriction (could not infer tissue from --out path).")
    # ---------------------------------------------------------

    matched_A_all = set(m["a_id"].astype(str))
    matched_B_all = set(m["b_id"].astype(str))
    n_pairs_csv = len(m)

    # Counts (restricted to matched)
    with ProcessPoolExecutor(max_workers=2) as ex:
        futA = ex.submit(load_counts, args.dataset_A, matched_A_all)
        futB = ex.submit(load_counts, args.dataset_B, matched_B_all)
        countsA = futA.result(); countsB = futB.result()

    countsA["gene"] = normalize_gene_series(countsA["gene"])
    countsB["gene"] = normalize_gene_series(countsB["gene"])

    # Before-QC stats
    cellsA_before = set(countsA["cell_id"].astype(str).unique())
    cellsB_before = set(countsB["cell_id"].astype(str).unique())
    genesA_before = set(countsA["gene"].unique())
    genesB_before = set(countsB["gene"].unique())
    shared_genes_before = genesA_before & genesB_before

    pairs_before = [(a, b) for a, b in m[["a_id", "b_id"]].itertuples(index=False)
                    if (a in cellsA_before) and (b in cellsB_before)]
    n_pairs_before = len(pairs_before)

    # Debug
    pd.Series(sorted(genesA_before)).to_csv(args.out / "debug" / "genes_A_matched_raw.txt", index=False, header=False)
    pd.Series(sorted(genesB_before)).to_csv(args.out / "debug" / "genes_B_matched_raw.txt", index=False, header=False)

    # AnnData
    adA = counts_to_adata(countsA); adB = counts_to_adata(countsB)
    adA.obs["cell_id"] = adA.obs_names; adB.obs["cell_id"] = adB.obs_names

    # QC
    cellsA_n_before = adA.n_obs; genesA_n_before = adA.n_vars
    cellsB_n_before = adB.n_obs; genesB_n_before = adB.n_vars
    if args.min_counts_per_cell > 0:
        sc.pp.filter_cells(adA, min_counts=args.min_counts_per_cell)
        sc.pp.filter_cells(adB, min_counts=args.min_counts_per_cell)
    if args.min_genes_per_cell > 0:
        sc.pp.filter_cells(adA, min_genes=args.min_genes_per_cell)
        sc.pp.filter_cells(adB, min_genes=args.min_genes_per_cell)
    if args.min_cells_per_gene > 0:
        sc.pp.filter_genes(adA, min_cells=args.min_cells_per_gene)
        sc.pp.filter_genes(adB, min_cells=args.min_cells_per_gene)
    if args.min_counts_per_gene > 0:
        sc.pp.filter_genes(adA, min_counts=args.min_counts_per_gene)
        sc.pp.filter_genes(adB, min_counts=args.min_counts_per_gene)

    cellsA_n_after = adA.n_obs; genesA_n_after = adA.n_vars
    cellsB_n_after = adB.n_obs; genesB_n_after = adB.n_vars
    cellsA_filtered = cellsA_n_before - cellsA_n_after
    cellsB_filtered = cellsB_n_before - cellsB_n_after
    genesA_filtered = genesA_n_before - genesA_n_after
    genesB_filtered = genesB_n_before - genesB_n_after

    # Pairs after QC
    cellsA_kept = set(adA.obs_names.astype(str))
    cellsB_kept = set(adB.obs_names.astype(str))
    order_pairs: List[Tuple[str, str]] = []
    for a, b in m[["a_id", "b_id"]].itertuples(index=False):
        if a in cellsA_kept and b in cellsB_kept:
            order_pairs.append((a, b))
    n_pairs_after = len(order_pairs)
    if not order_pairs:
        qc = qc_summary_dict(args, cellsA_n_before, cellsA_n_after, cellsA_filtered,
                             cellsB_n_before, cellsB_n_after, cellsB_filtered,
                             genesA_n_before, genesA_n_after, genesA_filtered,
                             genesB_n_before, genesB_n_after, genesB_filtered,
                             n_pairs_csv, n_pairs_before, n_pairs_after,
                             len(shared_genes_before), 0)
        write_qc_summary(args.out, qc)
        raise RuntimeError("All matched pairs were filtered out by QC thresholds.")

    # Shared genes after QC
    genesA_after = set(adA.var_names.astype(str))
    genesB_after = set(adB.var_names.astype(str))
    shared_genes_after = sorted(genesA_after & genesB_after)
    if not shared_genes_after:
        qc = qc_summary_dict(args, cellsA_n_before, cellsA_n_after, cellsA_filtered,
                             cellsB_n_before, cellsB_n_after, cellsB_filtered,
                             genesA_n_before, genesA_n_after, genesA_filtered,
                             genesB_n_before, genesB_n_after, genesB_filtered,
                             n_pairs_csv, n_pairs_before, n_pairs_after,
                             len(shared_genes_before), 0)
        write_qc_summary(args.out, qc)
        pd.Series(sorted(genesA_after)).to_csv(args.out / "debug" / "genes_A_from_transcripts.txt", index=False, header=False)
        pd.Series(sorted(genesB_after)).to_csv(args.out / "debug" / "genes_B_from_transcripts.txt", index=False, header=False)
        raise RuntimeError("No shared genes after QC filtering.")

    # Subset to shared genes & ordered pairs
    adA = adA[:, shared_genes_after].copy()
    adB = adB[:, shared_genes_after].copy()
    idxA = pd.Index(adA.obs_names.astype(str))
    idxB = pd.Index(adB.obs_names.astype(str))
    takeA = idxA.get_indexer([a for a, _ in order_pairs])
    takeB = idxB.get_indexer([b for _, b in order_pairs])
    keep_mask = (takeA >= 0) & (takeB >= 0)
    adA = adA[takeA[keep_mask], :].copy()
    adB = adB[takeB[keep_mask], :].copy()
    cellsA_ord = adA.obs_names.astype(str).to_list()
    cellsB_ord = adB.obs_names.astype(str).to_list()

    # Save final gene list
    pd.Series(shared_genes_after, name="gene").to_csv(args.out / "genes_common.txt", index=False)

    # Clustering
    labelsA, kA, resA = cluster_with_scanpy(adA, target_k=args.k, n_neighbors=args.n, random_state=args.random_state)
    labelsB, kB, resB = cluster_with_scanpy(adB, target_k=args.k, n_neighbors=args.n, random_state=args.random_state)
    (args.out / "clustering_summary.tsv").write_text(
        "dataset\ttarget_k\tachieved_k\tresolution\tn_neighbors\n"
        f"A\t{args.k}\t{kA}\t{resA:.6f}\t{args.n}\n"
        f"B\t{args.k}\t{kB}\t{resB:.6f}\t{args.n}\n"
    )

    # ARI + AMI
    ari = adjusted_rand_score(labelsA, labelsB)
    (args.out / "ari.txt").write_text(f"{ari:.6f}\n")
    ami = adjusted_mutual_info_score(labelsA, labelsB)
    (args.out / "ami.txt").write_text(f"{ami:.6f}\n")
    to_csv_auto(pd.DataFrame({"cell_id_A": cellsA_ord, "cluster": labelsA}), args.out / "cluster_labels_A.csv")
    to_csv_auto(pd.DataFrame({"cell_id_B": cellsB_ord, "cluster": labelsB}), args.out / "cluster_labels_B.csv")

    # Per-gene A↔B correlations (on filtered raw counts)
    MA = adA.X.tocsr() if sp.issparse(adA.X) else sp.csr_matrix(adA.X)
    MB = adB.X.tocsr() if sp.issparse(adB.X) else sp.csr_matrix(adB.X)
    corr_df = per_gene_correlations_parallel(MA, MB, list(adA.var_names.astype(str)),
                                             workers=args.workers, chunk_cols=args.corr_chunk)
    to_csv_auto(corr_df, args.out / "gene_correlation.csv")

    # Optional matrices
    if args.save_matrix:
        sp.save_npz(args.out / "counts_A_matched.npz", MA)
        sp.save_npz(args.out / "counts_B_matched.npz", MB)
        Path(args.out / "cell_ids_A.txt").write_text("\n".join(cellsA_ord) + "\n")
        Path(args.out / "cell_ids_B.txt").write_text("\n".join(cellsB_ord) + "\n")

    # QC summary
    qc = qc_summary_dict(
        args,
        cellsA_n_before, adA.n_obs, cellsA_filtered,
        cellsB_n_before, adB.n_obs, cellsB_filtered,
        genesA_n_before, adA.n_vars, genesA_filtered,
        genesB_n_before, adB.n_vars, genesB_filtered,
        n_pairs_csv, n_pairs_before, len(order_pairs),
        len(shared_genes_before), len(shared_genes_after),
    )
    write_qc_summary(args.out, qc)
    print_qc_summary(qc)

    # Summary
    print("=== Done ===")
    print(f"Matched pairs used: {adA.n_obs}")
    print(f"Shared genes after QC: {adA.n_vars}")
    print(f"Target clusters (k): {args.k} | n_neighbors: {args.n}")
    print(f"Workers (correlations): {args.workers}")
    print(f"ARI: {ari:.6f}")
    print(f"AMI: {ami:.6f}")
    print(f"Outputs -> {args.out}")

if __name__ == "__main__":
    main()
