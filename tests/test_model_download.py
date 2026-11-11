from pathlib import Path

import halo.model_download as model_download


def test_download_uses_hugging_face_cache(monkeypatch, tmp_path):
    expected = tmp_path / "Halo"

    def fake_download(**kwargs):
        assert kwargs["repo_id"] == "XYZ1998/Halo"
        assert kwargs["filename"] == "Halo"
        return str(expected)

    monkeypatch.setattr(model_download, "hf_hub_download", fake_download)
    assert model_download.download_halo_model(cache_dir=tmp_path) == expected
