"""COCO captions tied to the existing detector split, before model evaluation."""

import random
from collections import defaultdict
from pathlib import Path

from detection.common import ROOT, object_hash, read_json, sha256, write_json


def prepare() -> None:
    captions_path = ROOT / "data/coco/annotations/captions_val2017.json"
    captions = read_json(captions_path)
    split_path = ROOT / "detection/configs/dataset_split.json"
    split = read_json(split_path)
    references = defaultdict(list)
    for annotation in captions["annotations"]:
        references[annotation["image_id"]].append(annotation["caption"])
    images = {i["id"]: i for i in captions["images"]}
    # Stable disjoint selection from the already fixed detector partitions.
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
        write_json(ROOT / f"annotation/configs/{name}.json", manifest)
