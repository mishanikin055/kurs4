"""Artifact IO, hash checks, and the shared interprocess inference lock."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "detection/models"
REGISTRY = ROOT / "detection/configs/models.json"
CATEGORIES = ROOT / "detection/configs/coco_categories.json"


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_hash(value: Any) -> str:
    # JSON object keys become strings on disk; normalize before sorting so
    # integer-keyed counters have the same identity after serialization.
    normalized = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def registry() -> dict[str, Any]:
    return read_json(REGISTRY)


def model_spec(model_id: str) -> dict[str, Any]:
    return next(m for m in registry()["models"] if m["id"] == model_id)


@contextlib.contextmanager
def file_lock(path: Path) -> Iterator[int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Lock already held: {path}") from error
        try:
            yield handle.fileno()
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def verify_model(model_id: str) -> dict[str, Any]:
    manifest = read_json(MODELS / "manifest.json")["models"][model_id]
    if manifest["status"] not in {"verified", "smoke_passed"}:
        raise RuntimeError(f"Model is not verified: {model_id}")
    if manifest["spec_hash"] != object_hash(model_spec(model_id)):
        raise RuntimeError("Model registry changed; download and verify the pinned version again")
    for item in manifest["files"]:
        p = MODELS / model_id / item["path"]
        if not p.is_file() or p.stat().st_size != item["size"] or sha256(p) != item["sha256"]:
            raise RuntimeError(f"Model file checksum mismatch: {p}")
    return manifest
