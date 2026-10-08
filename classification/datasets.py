"""Explicit authenticated downloads and streaming import of ImageNet validation."""

from __future__ import annotations

import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from classification.common import CONFIGS, data_path, labels
from detection.common import ROOT, file_lock, object_hash, read_json, sha256, write_json


def freeze_manifest(path: Path, value: dict[str, Any]) -> None:
    if path.exists() and read_json(path) != value:
        raise ValueError(f"Frozen manifest differs: {path}; use a new dataset directory/version")
    write_json(path, value)


def fetch_validation(token_file: Path, dry_run: bool) -> None:
    source = read_json(CONFIGS / "dataset_source.json")
    output = ROOT / "data/imagenet/source"
    required = sum(f["size"] for f in source["files"])
    print(f"Validation parquet: {required} bytes; import needs additional image storage")
    if dry_run:
        print(json.dumps(source, indent=2))
        return
    if not token_file.is_file():
        raise RuntimeError(
            "Accept ImageNet terms on Hugging Face, then put a read token into .secrets/hf_token. "
            "Never send the token in chat or commit it."
        )
    token = token_file.read_text().strip()
    if not token:
        raise ValueError("Token file is empty")
    if shutil.disk_usage(ROOT / "data").free < required * 3:
        raise RuntimeError("Need space for parquet, original JPEGs and partial downloads")
    from huggingface_hub import HfApi, hf_hub_download

    output.mkdir(parents=True, exist_ok=True)
    with file_lock(output / ".download.lock"):
        api = HfApi(token=token)
        try:
            repo_files = api.list_repo_tree(
                source["repo_id"],
                path_in_repo="data",
                revision=source["revision"],
                repo_type="dataset",
                recursive=False,
            )
            remote = {f.path: f for f in repo_files if hasattr(f, "lfs")}
            plan = []
            for spec in source["files"]:
                f = remote[spec["path"]]
                digest = f.lfs.sha256 if f.lfs else None
                if not digest or not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ValueError("Authorized source did not expose a valid LFS SHA-256")
                if f.size != spec["size"] or f.blob_id != spec["git_oid"]:
                    raise ValueError("Frozen dataset source changed")
                plan.append({**spec, "sha256": digest})
            identity = {"schema_version": "1.0", "source": source, "files": plan}
            freeze_manifest(output / "download_plan.json", identity)
            for spec in plan:
                path = Path(
                    hf_hub_download(
                        source["repo_id"],
                        spec["path"],
                        repo_type="dataset",
                        revision=source["revision"],
                        local_dir=output,
                        token=token,
                    )
                )
                if path.stat().st_size != spec["size"] or sha256(path) != spec["sha256"]:
                    raise ValueError(f"Dataset shard checksum mismatch: {spec['path']}")
                print(f"Verified {spec['path']}", flush=True)
        except Exception as error:
            # Avoid HTTP tracebacks carrying Authorization headers or signed URLs.
            raise RuntimeError(
                f"ImageNet access/download failed ({type(error).__name__}); "
                "check dataset acceptance and token permissions, then retry."
            ) from None
        write_json(output / "download_manifest.json", {**identity, "status": "verified"})


def import_huggingface() -> None:
    import pyarrow.parquet as pq

    output = ROOT / "data/imagenet"
    download = read_json(output / "source/download_manifest.json")
    if download["status"] != "verified":
        raise ValueError("Dataset is not verified")
    classes = labels()
    with file_lock(output / ".prepare.lock"):
        for spec in download["files"]:
            path = output / "source" / spec["path"]
            if sha256(path) != spec["sha256"]:
                raise ValueError("Dataset shard changed")
            parquet = pq.ParquetFile(path)
            hf_meta = json.loads(parquet.schema_arrow.metadata[b"huggingface"])
            names = hf_meta["info"]["features"]["label"]["names"]
            validate_label_names(names, classes)
            for batch in parquet.iter_batches(batch_size=16, columns=["image", "label"]):
                for row in batch.to_pylist():
                    label = row["label"]
                    if type(label) is not int or not 0 <= label < 1000:
                        raise ValueError("Invalid ImageNet class index")
                    image = row["image"]
                    name = source_image_name(image["path"], classes[label]["synset"])
                    raw = image["bytes"]
                    if not isinstance(raw, bytes) or not raw:
                        raise ValueError("Missing original image bytes")
                    target = output / "val" / classes[label]["synset"] / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists():
                        if target.read_bytes() != raw:
                            raise ValueError("Conflicting imported image bytes")
                    else:
                        temporary = target.with_suffix(".part")
                        temporary.write_bytes(raw)
                        temporary.replace(target)
            print(f"Imported {spec['path']}", flush=True)
        write_json(output / "source_provenance.json", download)
    prepare_imagenet(output / "val", seed=42)


def validate_label_names(names: list[str], classes: list[dict[str, Any]]) -> None:
    if len(names) != 1000:
        raise ValueError("Expected 1000 reference class names")
    for i, (name, cls) in enumerate(zip(names, classes)):
        first = name.split(",")[0].strip().replace("_", " ").lower()
        expected = cls["name"].lower()
        # Three documented spelling differences in the pinned HF parquet metadata.
        aliases = {
            (134, "crane bird"): "crane",
            (517, "crane"): "crane2",
            (639, "maillot tank suit"): "maillot, tank suit",
        }
        alias = aliases.get((i, expected))
        if first != expected and name.lower() != alias:
            raise ValueError(f"Source/torchvision label ordering mismatch at {i}: {first}")


def source_image_name(name: str, target_synset: str) -> str:
    name = Path(name).name
    original = re.fullmatch(r"ILSVRC2012_val_(\d{8})\.JPEG", name)
    if original:
        return name
    # HF adds the true synset to original filenames. Validate it before removing it.
    tagged = re.fullmatch(r"ILSVRC2012_val_(\d{8})_(n\d{8})\.JPEG", name)
    if not tagged or tagged[2] != target_synset:
        raise ValueError("Unexpected ImageNet filename or source/label synset mismatch")
    return f"ILSVRC2012_val_{tagged[1]}.JPEG"


def balanced_selection(rows: list[dict[str, Any]], seed: int, per_class: int) -> list[dict]:
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["target_index"], []).append(row)
    randomizer = random.Random(seed)
    selected = []
    for target in sorted(grouped):
        group = sorted(grouped[target], key=lambda r: r["image_id"])
        randomizer.shuffle(group)
        if len(group) < per_class:
            raise ValueError("Insufficient images for balanced evaluation")
        selected.extend(group[:per_class])
    randomizer.shuffle(selected)
    return selected


def prepare_imagenet(images_dir: Path, seed: int) -> None:
    from PIL import Image

    images_dir = data_path(str(images_dir.resolve().relative_to(ROOT)))
    if not images_dir.is_dir():
        raise ValueError("Expected local ImageNet validation directory with synset subfolders")
    classes = labels()
    folders = sorted(p.name for p in images_dir.iterdir() if p.is_dir())
    if folders != [r["synset"] for r in classes]:
        raise ValueError("Validation directory must contain all 1000 ImageNet synsets")
    rows = []
    for cls in classes:
        files = sorted((images_dir / cls["synset"]).iterdir())
        if len(files) != 50:
            raise ValueError(f"Expected exactly 50 validation images: {cls['synset']}")
        for path in files:
            if not re.fullmatch(r"ILSVRC2012_val_\d{8}\.JPEG", path.name):
                raise ValueError("Only original validation filenames are accepted")
            path = data_path(str(path.relative_to(ROOT)))
            with Image.open(path) as image:
                if image.format != "JPEG" or image.width * image.height > 40_000_000:
                    raise ValueError("Unexpected ImageNet format or image dimensions")
                image.verify()
                width, height = image.size
            rows.append(
                {
                    "image_id": path.stem,
                    "path": str(path.relative_to(ROOT)),
                    "width": width,
                    "height": height,
                    "sha256": sha256(path),
                    "target_index": cls["index"],
                    "target_synset": cls["synset"],
                }
            )
    expected_ids = {f"ILSVRC2012_val_{i:08}" for i in range(1, 50001)}
    if {r["image_id"] for r in rows} != expected_ids or len(rows) != 50000:
        raise ValueError("Validation filenames do not cover exactly images 1..50000")
    assert set(Counter(r["target_index"] for r in rows).values()) == {50}
    base = {
        "schema_version": "1.0",
        "protocol_version": "classification-imagenet-v1",
        "dataset": "ImageNet-1K validation",
        "mode": "whole_image",
        "seed": seed,
        "class_mapping_hash": object_hash(classes),
        "n_classes": 1000,
        "exif_policy": "stored pixels; standard torchvision preprocessing, no EXIF transpose",
    }
    provenance_path = images_dir.parent / "source_provenance.json"
    if provenance_path.exists():
        base["source_provenance"] = read_json(provenance_path)
    else:
        base["source_provenance"] = {
            "kind": "user-provided local validation; acquisition terms accepted by user"
        }
    output = images_dir.parent
    with file_lock(output / ".prepare.lock"):
        freeze_manifest(output / "validation.json", {**base, "split": "validation", "images": rows})
        evaluation = balanced_selection(rows, seed, 5)
        freeze_manifest(
            output / "evaluation5000.json",
            {**base, "split": "evaluation5000", "images": evaluation},
        )
    print("Frozen validation: 50000; balanced evaluation: 5000 (5 per class)")
