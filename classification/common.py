from __future__ import annotations

from pathlib import Path
from typing import Any

from detection.common import ROOT, object_hash, read_json, sha256

CONFIGS = ROOT / "classification/configs"
MODELS = ROOT / "classification/models"


def registry() -> dict[str, Any]:
    return read_json(CONFIGS / "models.json")


def model_spec(model_id: str) -> dict[str, Any]:
    return next(m for m in registry()["models"] if m["id"] == model_id)


def labels() -> list[dict[str, Any]]:
    rows = read_json(CONFIGS / "imagenet_labels.json")["classes"]
    synsets = [r["synset"] for r in rows]
    if len(rows) != 1000 or synsets != sorted(set(synsets)):
        raise ValueError("Expected 1000 unique sorted ImageNet synsets")
    if [r["index"] for r in rows] != list(range(1000)):
        raise ValueError("Invalid ImageNet class indices")
    return rows


def verify_model(model_id: str) -> dict[str, Any]:
    manifest = read_json(MODELS / "manifest.json")["models"][model_id]
    if manifest["status"] not in {"verified", "smoke_passed"}:
        raise RuntimeError(f"Model not verified: {model_id}")
    if manifest["spec_hash"] != object_hash(model_spec(model_id)):
        raise ValueError("Model specification changed")
    for item in manifest["files"]:
        path = MODELS / model_id / item["path"]
        if path.stat().st_size != item["size"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Model file hash/size mismatch: {path}")
    return manifest


def data_path(name: str) -> Path:
    path = (ROOT / name).resolve()
    if not path.is_relative_to((ROOT / "data").resolve()):
        raise ValueError("Images must be inside data/")
    return path
