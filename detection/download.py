"""Explicit streaming downloads. No library-managed download during inference."""

from __future__ import annotations

import hashlib
import shutil
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from detection.common import (
    CATEGORIES,
    MODELS,
    file_lock,
    model_spec,
    object_hash,
    read_json,
    registry,
    sha256,
    write_json,
)


def _download_file_attempt(
    url: str, path: Path, expected: dict[str, Any] | None = None
) -> dict[str, Any]:
    expected = expected or {}
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    if not path.exists():
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "kurs4-detection/1.0",
                **({"Range": f"bytes={offset}-"} if offset else {}),
            },
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            append = offset > 0 and response.status == 206
            if append and not response.headers.get("Content-Range", "").startswith(
                f"bytes {offset}-"
            ):
                raise RuntimeError("Server returned an unexpected resume range")
            with partial.open("ab" if append else "wb") as stream:
                shutil.copyfileobj(response, stream, length=1024 * 1024)
    target = path if path.exists() else partial
    digest = sha256(target)
    if expected.get("size") is not None and target.stat().st_size != expected["size"]:
        raise RuntimeError(f"Size mismatch: {path}: {target.stat().st_size} != {expected['size']}")
    if expected.get("sha256") and digest != expected["sha256"]:
        raise RuntimeError(f"SHA-256 mismatch: {path}")
    if expected.get("sha256_prefix") and not digest.startswith(expected["sha256_prefix"]):
        raise RuntimeError(f"SHA-256 prefix mismatch: {path}")
    if expected.get("md5"):
        md5 = hashlib.md5()  # Official RF-DETR integrity metadata, not a security primitive.
        with target.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                md5.update(block)
        if md5.hexdigest() != expected["md5"]:
            raise RuntimeError(f"MD5 mismatch: {path}")
    if target == partial:
        partial.replace(path)
    return {"path": path.name, "size": path.stat().st_size, "sha256": digest, "source": url}


def download_file(url: str, path: Path, expected: dict[str, Any] | None = None) -> dict[str, Any]:
    for attempt in range(3):
        try:
            print(f"Fetch {url} (attempt {attempt + 1}/3)", flush=True)
            return _download_file_attempt(url, path, expected)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if attempt == 2:
                raise
            print(f"Temporary download error: {error}; retrying partial download", flush=True)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("Download retries exhausted")


def download_models(model_id: str | None = None, dry_run: bool = False) -> None:
    chosen = [model_spec(model_id)] if model_id else registry()["models"]
    required = sum(f.get("size") or 0 for m in chosen for f in m["files"])
    print(
        f"Weight bytes (without metadata): {required}; free: {shutil.disk_usage(MODELS).free}",
        flush=True,
    )
    if dry_run:
        for m in chosen:
            print(m["id"], m["revision"], m["files"])
        return
    if shutil.disk_usage(MODELS).free < required * 2 + 100 * 1024**2:
        raise RuntimeError("Insufficient disk space for downloads and temporary files")
    with file_lock(MODELS / ".download.lock"):
        manifest_path = MODELS / "manifest.json"
        manifest = (
            read_json(manifest_path)
            if manifest_path.exists()
            else {"schema_version": "1.0", "models": {}}
        )
        for m in chosen:
            previous = manifest["models"].get(m["id"], {})
            if previous.get("spec_hash") == object_hash(m) and previous.get("status") in {
                "verified",
                "smoke_passed",
            }:
                from detection.common import verify_model

                verify_model(m["id"])
                print(m["id"], "already verified", flush=True)
                continue
            entry = {
                "spec_hash": object_hash(m),
                "revision": m["revision"],
                "license": m["license"],
                "status": "downloading",
                "files": [],
            }
            manifest["models"][m["id"]] = entry
            write_json(manifest_path, manifest)
            try:
                for f in m["files"]:
                    target = MODELS / m["id"] / f["path"]
                    print(f"Downloading {m['id']}/{f['path']}", flush=True)
                    if f.get("generated"):
                        write_json(
                            target, {"mapping": m["mapping"], "categories": read_json(CATEGORIES)}
                        )
                        result = {
                            "path": target.name,
                            "size": target.stat().st_size,
                            "sha256": sha256(target),
                            "source": "detection/configs/coco_categories.json",
                        }
                    else:
                        result = download_file(f["url"], target, f)
                    entry["files"].append(result)
                    write_json(manifest_path, manifest)
                write_json(MODELS / m["id"] / "model_spec.json", m)
                target = MODELS / m["id"] / "model_spec.json"
                entry["files"].append(
                    {
                        "path": target.name,
                        "size": target.stat().st_size,
                        "sha256": sha256(target),
                        "source": "detection/configs/models.json",
                    }
                )
                entry.update(
                    status="verified", downloaded_at=datetime.now(timezone.utc).isoformat()
                )
            except Exception as error:
                entry.update(status="failed", error=str(error))
                raise
            finally:
                write_json(manifest_path, manifest)
