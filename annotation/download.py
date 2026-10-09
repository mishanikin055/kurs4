"""Explicit official download; no runtime network access."""

from pathlib import Path

from annotation.common import MODELS, specs
from detection.common import file_lock, object_hash, read_json, sha256, write_json


def download(model_ids: list[str]) -> None:
    from huggingface_hub import hf_hub_download

    MODELS.mkdir(parents=True, exist_ok=True)
    manifest_path = MODELS / "manifest.json"
    with file_lock(MODELS / ".download.lock"):
        manifest = read_json(manifest_path) if manifest_path.exists() else {"models": {}}
        for model_id in model_ids:
            spec = specs()[model_id]
            records = []
            for f in spec["files"]:
                p = Path(
                    hf_hub_download(
                        spec["repo_id"],
                        f["path"],
                        revision=spec["revision"],
                        local_dir=MODELS / model_id,
                        token=False,
                    )
                )
                digest = sha256(p)
                if p.stat().st_size != f["size"] or (
                    f["expected_sha256"] and digest != f["expected_sha256"]
                ):
                    raise ValueError(f"Official size/hash mismatch: {p}")
                records.append({"path": f["path"], "size": p.stat().st_size, "sha256": digest})
            manifest["models"][model_id] = {
                "status": "verified",
                "spec_hash": object_hash(spec),
                "revision": spec["revision"],
                "files": records,
            }
            write_json(manifest_path, manifest)
            print(f"Downloaded and verified {model_id}", flush=True)
