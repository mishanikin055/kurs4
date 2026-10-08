from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from classification.datasets import freeze_manifest
from detection.common import ROOT, object_hash, read_json, sha256

CONFIGS = ROOT / "classification/crops/configs"


def mapping() -> dict[int, int | None]:
    return {
        r["index"]: r["coco_category_id"] for r in read_json(CONFIGS / "mapping.json")["classes"]
    }


def crop_box(box: list[float], width: int, height: int) -> list[int]:
    if len(box) != 4 or any(not math.isfinite(v) for v in box):
        raise ValueError("Invalid xyxy crop coordinates")
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Crop box has nonpositive area")
    clipped = [
        max(0, math.floor(x1)),
        max(0, math.floor(y1)),
        min(width, math.ceil(x2)),
        min(height, math.ceil(y2)),
    ]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError("Crop is outside the image")
    return clipped


def iou(a: list[float], b: list[float], crowd: bool = False) -> float:
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    denominator = area_a if crowd else area_a + area_b - intersection
    return intersection / denominator if denominator else 0.0


def match_proposals(predictions: list[dict], gt: list[dict], threshold: float) -> list[dict]:
    regular = [a for a in gt if not a.get("iscrowd")]
    crowds = [a for a in gt if a.get("iscrowd")]
    used = set()
    result = []
    for rank, pred in enumerate(predictions):
        candidates = [
            (iou(pred["bbox_xyxy"], a["bbox_xyxy"]), a["id"], a)
            for a in regular
            if a["id"] not in used
        ]
        overlap, _, annotation = (
            max(candidates, key=lambda t: (t[0], -t[1])) if candidates else (0.0, -1, None)
        )
        if overlap >= threshold:
            used.add(annotation["id"])
            result.append(
                {
                    "rank": rank,
                    "matched_annotation_id": annotation["id"],
                    "target_category_id": annotation["category_id"],
                    "matched_iou": overlap,
                    "ignored_crowd": False,
                }
            )
        else:
            ignored = any(
                iou(pred["bbox_xyxy"], a["bbox_xyxy"], crowd=True) >= threshold for a in crowds
            )
            result.append(
                {
                    "rank": rank,
                    "matched_annotation_id": None,
                    "target_category_id": None,
                    "matched_iou": overlap,
                    "ignored_crowd": ignored,
                }
            )
    return result


def prepare(split: str, detector_run: Path, output: Path) -> None:
    config = read_json(CONFIGS / "benchmark.json")
    category_map = read_json(CONFIGS / "mapping.json")
    supported = set(category_map["supported_categories"])
    source = read_json(ROOT / f"data/coco/{split}.json")
    if sha256(ROOT / source["annotations"]) != source["annotations_sha256"]:
        raise ValueError("COCO ground truth changed")
    images = sorted(source["images"], key=lambda r: r["image_id"])
    random.Random(config["scene_seed"]).shuffle(images)
    images = sorted(images[: config["test_scenes"]], key=lambda r: r["image_id"])
    wanted = {r["image_id"] for r in images}
    gt = read_json(ROOT / source["annotations"])
    by_image: dict[int, list[dict]] = defaultdict(list)
    for a in gt["annotations"]:
        if a["image_id"] in wanted and a["area"] > 0:
            x, y, width, height = a["bbox"]
            by_image[a["image_id"]].append({**a, "bbox_xyxy": [x, y, x + width, y + height]})
    run = read_json(detector_run / "run.json")
    if (
        run["status"] != "complete"
        or run["model_id"] != config["detector_model"]
        or run["repeat"] != config["detector_repeat"] - 1
    ):
        raise ValueError("Need the pinned completed detector run")
    detector_dataset = read_json(detector_run / "dataset_manifest.json")
    if object_hash(detector_dataset) != object_hash(source):
        raise ValueError("Detector and crop source COCO split differ")
    proposals = {}
    with (detector_run / "samples.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["image_id"] not in wanted:
                continue
            if row["image_id"] in proposals:
                raise ValueError("Duplicate detector image_id")
            proposals[row["image_id"]] = sorted(
                [p for p in row["predictions"] if p["score"] >= config["detector_threshold"]],
                key=lambda p: -p["score"],
            )[: config["detector_max_per_image"]]
    if set(proposals) != wanted:
        raise ValueError("Missing detector predictions")
    output.mkdir(parents=True, exist_ok=True)
    common = {
        "schema_version": "1.0",
        "protocol_version": config["protocol_version"],
        "split": split,
        "config": config,
        "mapping_hash": object_hash(category_map),
        "supported_categories": sorted(supported),
        "n_scenes": len(images),
        "scenes": [],
        "annotations_sha256": source["annotations_sha256"],
        "detector_source": {
            "run_path": str(detector_run),
            "samples_sha256": sha256(detector_run / "samples.jsonl"),
            "run_sha256": sha256(detector_run / "run.json"),
            "environment": read_json(detector_run / "environment.json"),
            "threshold": config["detector_threshold"],
            "max_per_image": config["detector_max_per_image"],
        },
    }
    gt_rows, det_rows = [], []
    regular_count = Counter()
    supported_count = Counter()
    for image in images:
        annotations = sorted(by_image[image["image_id"]], key=lambda a: a["id"])
        regular = [a for a in annotations if not a["iscrowd"]]
        image_gt_supported = [a for a in regular if a["category_id"] in supported]
        regular_count.update(a["category_id"] for a in regular)
        supported_count.update(a["category_id"] for a in image_gt_supported)
        common["scenes"].append(
            {
                **image,
                "regular_gt": len(regular),
                "supported_gt": len(image_gt_supported),
                "supported_gt_by_category": dict(
                    Counter(a["category_id"] for a in image_gt_supported)
                ),
            }
        )
        for a in image_gt_supported:
            gt_rows.append(
                {
                    **image,
                    "sample_id": f"{image['image_id']}:gt:{a['id']}",
                    "object_id": f"gt:{a['id']}",
                    "bbox_xyxy": a["bbox_xyxy"],
                    "crop_xyxy": crop_box(a["bbox_xyxy"], image["width"], image["height"]),
                    "target_category_id": a["category_id"],
                    "matched_annotation_id": a["id"],
                    "matched_iou": 1.0,
                    "ignored_crowd": False,
                    "detector_category_id": None,
                    "detector_score": None,
                }
            )
        prediction_rows = proposals[image["image_id"]]
        for pred, match in zip(
            prediction_rows, match_proposals(prediction_rows, annotations, config["match_iou"])
        ):
            det_rows.append(
                {
                    **image,
                    **match,
                    "sample_id": f"{image['image_id']}:det:{match['rank']}",
                    "object_id": f"det:{match['rank']}",
                    "bbox_xyxy": pred["bbox_xyxy"],
                    "crop_xyxy": crop_box(pred["bbox_xyxy"], image["width"], image["height"]),
                    "detector_category_id": pred["category_id"],
                    "detector_score": pred["score"],
                }
            )
    common.update(
        regular_gt_counts=dict(regular_count),
        supported_gt_counts=dict(supported_count),
        mapping_object_coverage=sum(supported_count.values()) / sum(regular_count.values()),
        unsupported_gt_counts={k: v for k, v in regular_count.items() if k not in supported},
    )
    for mode, rows in [("gt_crops", gt_rows), ("detector_crops", det_rows)]:
        freeze_manifest(output / f"{mode}.json", {**common, "mode": mode, "samples": rows})
    print(
        f"Frozen {len(images)} {split} scenes; GT crops {len(gt_rows)}, detector crops {len(det_rows)}; object coverage {common['mapping_object_coverage']:.4f}"
    )
