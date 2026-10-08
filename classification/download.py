from __future__ import annotations

import shutil
from datetime import datetime, timezone
from typing import Any

from classification.common import CONFIGS, MODELS, labels, model_spec, registry
from detection.common import file_lock, object_hash, read_json, sha256, write_json
from detection.download import download_file


def download_models(model_id: str | None = None, dry_run: bool = False) -> None:
    chosen = [model_spec(model_id)] if model_id else registry()["models"]
    required = sum(f["size"] for m in chosen for f in m["files"])
    available = shutil.disk_usage(CONFIGS).free
    print(f"Files: {required} bytes; free: {available} bytes", flush=True)
    if available < required * 2:
        raise RuntimeError("Not enough free space for downloads and partial files")
    if dry_run:
        for m in chosen:
            for f in m["files"]:
                print(m["id"], f["path"], f["size"], f["url"])
        return
    from torchvision.models import get_model_weights

    with file_lock(MODELS / ".download.lock"):
        manifest_path = MODELS / "manifest.json"
        manifest: dict[str, Any] = (
            read_json(manifest_path)
            if manifest_path.exists()
            else {"schema_version": "1.0", "models": {}}
        )
        for spec in chosen:
            target = MODELS / spec["id"]
            target.mkdir(parents=True, exist_ok=True)
            entry = {
                "status": "downloading",
                "spec_hash": object_hash(spec),
                "revision": spec["revision"],
                "license": spec["license"],
                "files": [],
            }
            manifest["models"][spec["id"]] = entry
            write_json(manifest_path, manifest)
            try:
                entry["files"] = [
                    download_file(f["url"], target / f["path"], f) for f in spec["files"]
                ]
                weights = get_model_weights(spec["id"])[spec["weights"]]
                if weights.url != spec["files"][0]["url"]:
                    raise ValueError("Library weight URL differs from pinned artifact")
                if weights.meta["categories"] != [r["name"] for r in labels()]:
                    raise ValueError("ImageNet categories differ between models")
                t = weights.transforms()
                native = {
                    "schema_version": "1.0",
                    "weights": spec["weights"],
                    "crop_size": t.crop_size,
                    "resize_size": t.resize_size,
                    "mean": t.mean,
                    "std": t.std,
                    "interpolation": t.interpolation.value,
                    "antialias": t.antialias,
                    "class_mapping_hash": object_hash(labels()),
                }
                for name, value in [
                    ("native_config.json", native),
                    ("label_mapping.json", labels()),
                ]:
                    p = target / name
                    write_json(p, value)
                    entry["files"].append(
                        {
                            "path": name,
                            "size": p.stat().st_size,
                            "sha256": sha256(p),
                            "source": "generated from pinned torchvision and class mapping",
                        }
                    )
                entry.update(status="verified", verified_at=datetime.now(timezone.utc).isoformat())
            except BaseException as error:
                entry.update(status="failed", error=str(error))
                raise
            finally:
                write_json(manifest_path, manifest)
