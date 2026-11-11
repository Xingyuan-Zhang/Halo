"""Resolve the pretrained Halo checkpoint from the Hugging Face Hub."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from huggingface_hub import hf_hub_download

HF_REPO_ID = "XYZ1998/Halo"
HF_MODEL_FILENAME = "Halo"


def download_halo_model(
    revision: Optional[str] = None,
    cache_dir: Optional[Path] = None,
) -> Path:
    """Download Halo once and return the cached local checkpoint path."""
    try:
        path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_MODEL_FILENAME,
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not resolve the Halo checkpoint from "
            f"https://huggingface.co/{HF_REPO_ID}. Check the network connection "
            "or provide --model-path with an existing local checkpoint."
        ) from exc
    return Path(path)
