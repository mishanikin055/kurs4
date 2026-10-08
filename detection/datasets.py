"""Prepare immutable, disjoint COCO dev/test manifests before evaluation."""

from __future__ import annotations

import collections
import random
import zipfile
from pathlib import Path

from detection.common import CATEGORIES, ROOT, object_hash, read_json, sha256, write_json
from detection.download import download_file


def prepare_dataset(download: bool, seed: int = 42) -> None:
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
                "https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip", archive
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
        distribution = collections.Counter(a["category_id"] for a in annotations)
        sizes = collections.Counter(
            "small" if a["area"] < 32**2 else "medium" if a["area"] < 96**2 else "large"
            for a in annotations
        )
        counts = collections.Counter(a["image_id"] for a in annotations)
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
                "objects_per_image": dict(collections.Counter(counts.get(i, 0) for i in ids)),
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
        write_json(target, manifest)
        print(f"{split}: {len(rows)} images; manifest hash {object_hash(manifest)}", flush=True)
