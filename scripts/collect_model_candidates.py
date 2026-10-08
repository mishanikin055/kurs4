"""Save official detection candidate metadata; no cross-channel popularity rank."""

from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

CANDIDATES = [
    ("YOLO26", "ultralytics/ultralytics", None, "https://docs.ultralytics.com/models/yolo26/"),
    ("YOLO11", "ultralytics/ultralytics", None, "https://docs.ultralytics.com/models/yolo11/"),
    ("YOLOv8", "ultralytics/ultralytics", None, "https://docs.ultralytics.com/models/yolov8/"),
    ("RT-DETR", "lyuwenyu/RT-DETR", "PekingU/rtdetr_r18vd", "https://github.com/lyuwenyu/RT-DETR"),
    (
        "RT-DETRv2",
        "lyuwenyu/RT-DETR",
        "PekingU/rtdetr_v2_r18vd",
        "https://huggingface.co/PekingU/rtdetr_v2_r18vd",
    ),
    ("RF-DETR", "roboflow/rf-detr", None, "https://github.com/roboflow/rf-detr"),
    ("Faster R-CNN", None, None, "https://docs.pytorch.org/vision/stable/models/faster_rcnn.html"),
    ("RetinaNet", None, None, "https://docs.pytorch.org/vision/stable/models/retinanet.html"),
    ("FCOS", None, None, "https://docs.pytorch.org/vision/stable/models/fcos.html"),
    ("SSD", None, None, "https://docs.pytorch.org/vision/stable/models/ssd.html"),
]


def collect() -> None:
    rows = []
    for family, repository, hub, source in CANDIDATES:
        row = {
            "family": family,
            "official_source": source,
            "repository": repository,
            "hub_checkpoint": hub,
            "repository_stars": None,
            "monthly_checkpoint_downloads": None,
            "latest_release": None,
            "errors": [],
        }
        for kind, url in [
            ("repository", f"https://api.github.com/repos/{repository}" if repository else None),
            ("hub", f"https://huggingface.co/api/models/{hub}" if hub else None),
            (
                "release",
                f"https://api.github.com/repos/{repository}/releases/latest"
                if repository
                else None,
            ),
        ]:
            if not url:
                continue
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "kurs4-detection"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    value = json.load(response)
                row[kind + "_snapshot"] = value
                if kind == "repository":
                    row["repository_stars"] = value.get("stargazers_count")
                elif kind == "hub":
                    row["monthly_checkpoint_downloads"] = value.get("downloads")
                else:
                    row["latest_release"] = {
                        "tag": value.get("tag_name"),
                        "published_at": value.get("published_at"),
                    }
            except Exception as error:
                row["errors"].append({"url": url, "error": str(error)})
        rows.append(row)
    output = (
        Path(__file__).resolve().parents[1] / "reports/model_selection/detection_candidates.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "scope": "official repositories and specified HF checkpoints; counters are not comparable across distribution channels",
                "selection": "architecture representatives from PROJECT_PLAN 3.2; not a proven top-four ranking",
                "candidates": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    print(output)


if __name__ == "__main__":
    collect()
