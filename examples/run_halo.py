from pathlib import Path

from halo.pipeline import run_pipeline


mask_path = run_pipeline(
    xenium_dir=Path("/path/to/xenium_dataset"),
    out_dir=Path("results/halo"),
)
print(f"Saved cell mask to {mask_path}")
