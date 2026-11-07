"""Pinned downloads are verified before publication; no network or model imports."""

import hashlib
from io import BytesIO

import pytest

from scripts import fetch_app_models as fetcher


def test_verified_download_is_idempotent_and_existing_files_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setattr(fetcher, "ROOT", tmp_path)
    monkeypatch.setattr(fetcher.urllib.request, "urlopen", lambda *a, **k: BytesIO(b"weights"))
    target = tmp_path / "models" / "weights"
    expected = hashlib.sha256(b"weights").hexdigest()
    fetcher.fetch(target, expected, "https://example.invalid/weights", {"size": 7})
    assert target.read_bytes() == b"weights"

    def unexpected_network(*args, **kwargs):
        pytest.fail("Verified file must not be downloaded again")

    monkeypatch.setattr(fetcher.urllib.request, "urlopen", unexpected_network)
    fetcher.fetch(target, expected, "https://example.invalid/weights", {"size": 7})
    target.write_bytes(b"other")
    with pytest.raises(ValueError, match="Existing file differs"):
        fetcher.fetch(target, expected, "https://example.invalid/weights", {"size": 7})
    assert target.read_bytes() == b"other"


@pytest.mark.parametrize("data,size", [(b"wrong", 7), (b"oversized", 7)])
def test_bad_download_never_becomes_a_model_file(tmp_path, monkeypatch, data, size):
    monkeypatch.setattr(fetcher, "ROOT", tmp_path)
    monkeypatch.setattr(fetcher.urllib.request, "urlopen", lambda *a, **k: BytesIO(data))
    target = tmp_path / "models" / "weights"
    with pytest.raises(ValueError):
        fetcher.fetch(
            target,
            hashlib.sha256(b"weights").hexdigest(),
            "https://example.invalid/x",
            {"size": size},
        )
    assert not target.exists()
    assert not list(target.parent.iterdir())


def test_sources_cover_application_models_and_translator():
    rows = list(fetcher.sources())
    assert {target.name for target, *_ in rows} >= {
        "rf-detr-small.pth",
        "label_mapping.json",
        "model.safetensors",
        "modeling_florence2.py",
    }
    assert {target.parent.name for target, *_ in rows} == {
        "rf_detr_small",
        "florence2_base_ft",
        "opus_mt_en_ru",
    }
    assert all(len(expected) == 64 for _, expected, *_ in rows)
    assert all(spec.get("size") or spec.get("generated") for *_, spec in rows)
