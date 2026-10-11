"""Explicit preparation of pinned datasets and weights; never called by comparisons."""

from __future__ import annotations

import argparse
import collections
import contextlib
import fcntl
import hashlib
import json
import os
import random
import re
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ROOT / "detection/configs/coco_categories.json"
CONFIGS = ROOT / "classification/configs"


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
    normalized = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


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


def labels() -> list[dict[str, Any]]:
    rows = read_json(CONFIGS / "imagenet_labels.json")["classes"]
    synsets = [r["synset"] for r in rows]
    if len(rows) != 1000 or synsets != sorted(set(synsets)):
        raise ValueError("Expected 1000 unique sorted ImageNet synsets")
    if [r["index"] for r in rows] != list(range(1000)):
        raise ValueError("Invalid ImageNet class indices")
    return rows


def data_path(name: str) -> Path:
    path = (ROOT / name).resolve()
    if not path.is_relative_to((ROOT / "data").resolve()):
        raise ValueError("Images must be inside data/")
    return path


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
            if append and (
                not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
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
    if expected.get("sha256_prefix") and (not digest.startswith(expected["sha256_prefix"])):
        raise RuntimeError(f"SHA-256 prefix mismatch: {path}")
    if expected.get("md5"):
        md5 = hashlib.md5()
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


def download_detection_models(model_id: str | None = None, dry_run: bool = False) -> None:
    chosen = (
        [pinned_model("detection", model_id)]
        if model_id
        else read_json(ROOT / "detection/configs/models.json")["models"]
    )
    required = sum((f.get("size") or 0 for m in chosen for f in m["files"]))
    print(
        f"Weight bytes (without metadata): {required}; free: {shutil.disk_usage(ROOT / 'detection/models').free}",
        flush=True,
    )
    if dry_run:
        for m in chosen:
            print(m["id"], m["revision"], m["files"])
        return
    if shutil.disk_usage(ROOT / "detection/models").free < required * 2 + 100 * 1024**2:
        raise RuntimeError("Insufficient disk space for downloads and temporary files")
    with file_lock(ROOT / "detection/models" / ".download.lock"):
        manifest_path = ROOT / "detection/models" / "manifest.json"
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
                verify_weights("detection", m["id"])
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
                    target = ROOT / "detection/models" / m["id"] / f["path"]
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
                write_json(ROOT / "detection/models" / m["id"] / "model_spec.json", m)
                target = ROOT / "detection/models" / m["id"] / "model_spec.json"
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


def download_classification_models(model_id: str | None = None, dry_run: bool = False) -> None:
    chosen = (
        [pinned_model("classification", model_id)]
        if model_id
        else read_json(ROOT / "classification/configs/models.json")["models"]
    )
    required = sum((f["size"] for m in chosen for f in m["files"]))
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

    with file_lock(ROOT / "classification/models" / ".download.lock"):
        manifest_path = ROOT / "classification/models" / "manifest.json"
        manifest: dict[str, Any] = (
            read_json(manifest_path)
            if manifest_path.exists()
            else {"schema_version": "1.0", "models": {}}
        )
        for spec in chosen:
            if manifest["models"].get(spec["id"], {}).get("status") in {"verified", "smoke_passed"}:
                verify_weights("classification", spec["id"])
                continue
            target = ROOT / "classification/models" / spec["id"]
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


def download_annotation_models(model_ids: list[str]) -> None:
    from huggingface_hub import hf_hub_download

    (ROOT / "annotation/models").mkdir(parents=True, exist_ok=True)
    manifest_path = ROOT / "annotation/models" / "manifest.json"
    with file_lock(ROOT / "annotation/models" / ".download.lock"):
        manifest = read_json(manifest_path) if manifest_path.exists() else {"models": {}}
        for model_id in model_ids:
            if manifest["models"].get(model_id, {}).get("status") in {"verified", "smoke_passed"}:
                verify_weights("annotation", model_id)
                continue
            spec = {
                s["id"]: s for s in read_json(ROOT / "annotation/configs/models.json")["models"]
            }[model_id]
            records = []
            for f in spec["files"]:
                p = Path(
                    hf_hub_download(
                        spec["repo_id"],
                        f["path"],
                        revision=spec["revision"],
                        local_dir=ROOT / "annotation/models" / model_id,
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


def prepare_coco(download: bool, seed: int = 42) -> None:
    base = ROOT / "data/coco"
    base.mkdir(parents=True, exist_ok=True)
    annotation = base / "annotations/instances_val2017.json"
    images = base / "val2017"
    if download:
        if not annotation.exists():
            archive = base / "annotations_trainval2017.zip"
            download_file(
                "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip",
                archive,
                COCO_ARCHIVES[archive.name],
            )
            with zipfile.ZipFile(archive) as z:
                name = "annotations/instances_val2017.json"
                annotation.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as source, annotation.open("wb") as dest:
                    import shutil

                    shutil.copyfileobj(source, dest)
        if not images.exists() or len(list(images.glob("*.jpg"))) != 5000:
            archive = base / "val2017.zip"
            download_file(
                "https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip",
                archive,
                COCO_ARCHIVES[archive.name],
            )
            with zipfile.ZipFile(archive) as z:
                images.mkdir(exist_ok=True)
                for item in z.infolist():
                    p = Path(item.filename)
                    if len(p.parts) != 2 or p.parts[0] != "val2017" or p.suffix != ".jpg":
                        continue
                    with z.open(item) as source, (images / p.name).open("wb") as dest:
                        import shutil

                        shutil.copyfileobj(source, dest)
    gt = read_json(annotation)
    expected = {c["id"]: c["name"] for c in read_json(CATEGORIES)}
    actual = {c["id"]: c["name"] for c in gt["categories"]}
    if actual != expected or len(gt["images"]) != 5000:
        raise ValueError("Expected complete official COCO val2017 (5000 images, 80 categories)")
    ordered = sorted(gt["images"], key=lambda x: x["id"])
    random.Random(seed).shuffle(ordered)
    splits = {"dev": ordered[:500], "test": ordered[500:], "validation": ordered}
    for split, records in splits.items():
        rows = []
        for item in records:
            if Path(item["file_name"]).name != item["file_name"]:
                raise ValueError("Invalid COCO filename")
            path = images / item["file_name"]
            rows.append(
                {
                    "image_id": item["id"],
                    "path": str(path.relative_to(ROOT)),
                    "width": item["width"],
                    "height": item["height"],
                    "sha256": sha256(path),
                }
            )
        ids = {r["image_id"] for r in rows}
        annotations = [a for a in gt["annotations"] if a["image_id"] in ids]
        distribution = collections.Counter((a["category_id"] for a in annotations))
        sizes = collections.Counter(
            (
                "small" if a["area"] < 32**2 else "medium" if a["area"] < 96**2 else "large"
                for a in annotations
            )
        )
        counts = collections.Counter((a["image_id"] for a in annotations))
        manifest = {
            "schema_version": "1.0",
            "dataset": "COCO val2017",
            "split": split,
            "seed": seed,
            "annotations": str(annotation.relative_to(ROOT)),
            "annotations_sha256": sha256(annotation),
            "images": rows,
            "distribution": {
                "categories": dict(distribution),
                "sizes": dict(sizes),
                "objects_per_image": dict(collections.Counter((counts.get(i, 0) for i in ids))),
            },
        }
        frozen_path = ROOT / "detection/configs/dataset_split.json"
        if frozen_path.exists():
            frozen = read_json(frozen_path)
            if (
                seed != frozen["seed"]
                or object_hash(manifest) != frozen["splits"][split]["manifest_hash"]
            ):
                raise ValueError(
                    "Dataset differs from the pre-registered split; create a new protocol version"
                )
        target = base / f"{split}.json"
        if target.exists() and object_hash(read_json(target)) != object_hash(manifest):
            raise ValueError(f"Refusing to replace a different frozen split: {target}")
        freeze_manifest(target, manifest)
        print(f"{split}: {len(rows)} images; manifest hash {object_hash(manifest)}", flush=True)


def prepare_captions() -> None:
    captions_path = ROOT / "data/coco/annotations/captions_val2017.json"
    captions = read_json(captions_path)
    split_path = ROOT / "detection/configs/dataset_split.json"
    split = read_json(split_path)
    references = defaultdict(list)
    for annotation in captions["annotations"]:
        references[annotation["image_id"]].append(annotation["caption"])
    images = {i["id"]: i for i in captions["images"]}
    for name, count, parent in [("dev100", 100, "dev"), ("test500", 500, "test")]:
        ids = list(split["splits"][parent]["image_ids"])
        random.Random(42).shuffle(ids)
        selected = ids[:count]
        rows = []
        for image_id in selected:
            item = images[image_id]
            path = Path("data/coco/val2017") / item["file_name"]
            rows.append(
                {
                    "image_id": image_id,
                    "path": str(path),
                    "sha256": sha256(ROOT / path),
                    "width": item["width"],
                    "height": item["height"],
                    "references": references[image_id],
                }
            )
        manifest = {
            "name": name,
            "source": "COCO val2017; project-local detector partition",
            "source_url": "https://cocodataset.org/#download",
            "parent_partition": parent,
            "seed": 42,
            "captions_sha256": sha256(captions_path),
            "instances_sha256": sha256(ROOT / "data/coco/annotations/instances_val2017.json"),
            "detector_split_sha256": sha256(split_path),
            "images": rows,
        }
        manifest["hash"] = object_hash(manifest)
        freeze_manifest(ROOT / f"annotation/configs/{name}.json", manifest)


def freeze_manifest(path: Path, value: dict[str, Any]) -> None:
    if path.exists() and object_hash(read_json(path)) != object_hash(value):
        raise ValueError(f"Frozen manifest differs: {path}; use a new dataset directory/version")
    if not path.exists():
        write_json(path, value)


def fetch_validation(token_file: Path, dry_run: bool) -> None:
    source = read_json(CONFIGS / "dataset_source.json")
    output = ROOT / "data/imagenet/source"
    required = sum((f["size"] for f in source["files"]))
    print(f"Validation parquet: {required} bytes; import needs additional image storage")
    if dry_run:
        print(json.dumps(source, indent=2))
        return
    if not token_file.is_file():
        raise RuntimeError(
            "Accept ImageNet terms on Hugging Face, then put a read token into .secrets/hf_token. Never send the token in chat or commit it."
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
                if not digest or not re.fullmatch("[0-9a-f]{64}", digest):
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
            raise RuntimeError(
                f"ImageNet access/download failed ({type(error).__name__}); check dataset acceptance and token permissions, then retry."
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
    original = re.fullmatch("ILSVRC2012_val_(\\d{8})\\.JPEG", name)
    if original:
        return name
    tagged = re.fullmatch("ILSVRC2012_val_(\\d{8})_(n\\d{8})\\.JPEG", name)
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
    folders = sorted((p.name for p in images_dir.iterdir() if p.is_dir()))
    if folders != [r["synset"] for r in classes]:
        raise ValueError("Validation directory must contain all 1000 ImageNet synsets")
    rows = []
    for cls in classes:
        files = sorted((images_dir / cls["synset"]).iterdir())
        if len(files) != 50:
            raise ValueError(f"Expected exactly 50 validation images: {cls['synset']}")
        for path in files:
            if not re.fullmatch("ILSVRC2012_val_\\d{8}\\.JPEG", path.name):
                raise ValueError("Only original validation filenames are accepted")
            path = data_path(str(path.relative_to(ROOT)))
            with Image.open(path) as image:
                if image.format != "JPEG" or image.width * image.height > 40000000:
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
    assert set(Counter((r["target_index"] for r in rows)).values()) == {50}
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


def pinned_model(direction: str, model_id: str) -> dict[str, Any]:
    return next(
        m
        for m in read_json(ROOT / direction / "configs/models.json")["models"]
        if m["id"] == model_id
    )


def verify_weights(direction: str, model_id: str) -> dict[str, Any]:
    base = ROOT / direction / "models"
    record = read_json(base / "manifest.json")["models"][model_id]
    spec = pinned_model(direction, model_id)
    if record["status"] not in {"verified", "smoke_passed"} or record["spec_hash"] != object_hash(
        spec
    ):
        raise ValueError(f"Model is not verified against pinned sources: {model_id}")
    files = {f["path"]: f for f in record["files"]}
    if not {f["path"] for f in spec["files"]}.issubset(files):
        raise ValueError("Incomplete model manifest")
    for f in files.values():
        path = (base / model_id / f["path"]).resolve()
        if not path.is_relative_to((base / model_id).resolve()):
            raise ValueError("Invalid model path")
        if path.stat().st_size != f["size"] or sha256(path) != f["sha256"]:
            raise ValueError(f"Model checksum mismatch: {path}")
    for f in spec["files"]:
        expected = f.get("sha256") or f.get("expected_sha256")
        if expected and expected != files[f["path"]]["sha256"]:
            raise ValueError("Manifest differs from pinned artifact SHA-256")
    print(f"{direction}/{model_id}: verified", flush=True)
    return record


def prepare_annotation_resources(download: bool) -> None:
    base = ROOT / "data/coco"
    captions = base / "annotations/captions_val2017.json"
    if not captions.exists() and download:
        archive = base / "annotations_trainval2017.zip"
        download_file(
            "https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip",
            archive,
            COCO_ARCHIVES[archive.name],
        )
        with zipfile.ZipFile(archive) as z:
            name = "annotations/captions_val2017.json"
            captions.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, captions.open("wb") as dest:
                shutil.copyfileobj(src, dest)
    frozen = read_json(ROOT / "annotation/configs/test500.json")
    if sha256(captions) != frozen["captions_sha256"]:
        raise ValueError("COCO captions differ from pinned source")
    if sha256(base / "annotations/instances_val2017.json") != frozen["instances_sha256"]:
        raise ValueError("COCO instances differ from pinned source")
    import nltk

    nltk_dir = ROOT / "data/annotation_evaluator/nltk_data"
    if download:
        nltk.download("punkt_tab", download_dir=str(nltk_dir), raise_on_error=True, quiet=True)
    files = read_json(ROOT / "annotation/evaluation/sources.json")["nltk_punkt_tab"]["files"]
    for name, expected in files.items():
        if sha256(nltk_dir / "tokenizers/punkt_tab/english" / name) != expected:
            raise ValueError(f"NLTK resource SHA-256 mismatch: {name}")


def verify_dataset(direction: str) -> None:
    paths = {
        "detection": ROOT / "data/coco/test.json",
        "classification": ROOT / "data/imagenet/validation.json",
        "annotation": ROOT / "annotation/configs/test500.json",
    }
    dataset = read_json(paths[direction])
    expected = {"detection": 4500, "classification": 50000, "annotation": 500}[direction]
    if (
        len(dataset["images"]) != expected
        or len({r["image_id"] for r in dataset["images"]}) != expected
    ):
        raise ValueError("Unexpected or duplicate dataset image IDs")
    if direction == "detection":
        if (
            object_hash(dataset)
            != read_json(ROOT / "detection/configs/dataset_split.json")["splits"]["test"][
                "manifest_hash"
            ]
        ):
            raise ValueError("COCO manifest changed")
        if sha256(ROOT / dataset["annotations"]) != dataset["annotations_sha256"]:
            raise ValueError("COCO ground truth checksum mismatch")
    if direction == "classification":
        if dataset["class_mapping_hash"] != object_hash(labels()):
            raise ValueError("ImageNet class order changed")
        if Counter(r["target_index"] for r in dataset["images"]) != Counter(
            {i: 50 for i in range(1000)}
        ):
            raise ValueError("Expected 50 images per ImageNet class")
        classes = labels()
        for row in dataset["images"]:
            if classes[row["target_index"]]["synset"] != row["target_synset"]:
                raise ValueError("ImageNet synset/index mismatch")
    for row in dataset["images"]:
        path = data_path(row["path"])
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Image checksum mismatch: {path}")
    print(f"{direction}: {expected} images verified", flush=True)


COCO_ARCHIVES = {
    "annotations_trainval2017.zip": {
        "size": 252907541,
        "sha256": "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268",
    },
    "val2017.zip": {
        "size": 815585330,
        "sha256": "4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Явная подготовка данных и готовых весов")
    parser.add_argument("direction", choices=["detection", "classification", "annotation"])
    parser.add_argument("--download", action="store_true", help="Явно разрешить сеть для загрузки")
    parser.add_argument(
        "--dry-run", action="store_true", help="Показать закреплённые источники без записи"
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Только проверить уже подготовленные файлы"
    )
    args = parser.parse_args()
    if args.download and args.verify_only:
        parser.error("--download and --verify-only are mutually exclusive")
    specs = read_json(ROOT / args.direction / "configs/models.json")["models"]
    if args.dry_run:
        print(
            json.dumps(
                {"direction": args.direction, "models": specs, "network_enabled": args.download},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    # Do not modify weights while any model process is using them.
    with file_lock(ROOT / "storage/locks/inference.lock"):
        if args.download:
            os.environ["HF_HUB_OFFLINE"] = "0"
            os.environ["TRANSFORMERS_OFFLINE"] = "0"
            if args.direction == "detection":
                download_detection_models()
            elif args.direction == "classification":
                download_classification_models()
                if not (ROOT / "data/imagenet/validation.json").is_file():
                    fetch_validation(ROOT / ".secrets/hf_token", False)
                    import_huggingface()
            else:
                download_annotation_models([s["id"] for s in specs])
        for spec in specs:
            verify_weights(args.direction, spec["id"])
        if not args.verify_only:
            if args.direction == "detection":
                prepare_coco(args.download)
            elif args.direction == "classification":
                prepare_imagenet(ROOT / "data/imagenet/val", seed=42)
            else:
                prepare_annotation_resources(args.download)
                prepare_captions()
        elif args.direction == "annotation":
            prepare_annotation_resources(False)
        verify_dataset(args.direction)


if __name__ == "__main__":
    main()
