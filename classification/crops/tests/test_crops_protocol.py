from __future__ import annotations

import json

import pytest

from classification.crops.datasets import CONFIGS, crop_box, mapping, match_proposals
from classification.crops.metrics import quality, scene_arrays
from classification.crops.runner import interpret, recover
from detection.common import object_hash, read_json


def predictions(first: int = 207, score: float = 0.8) -> list[dict]:
    return [
        {"index": i, "score": s}
        for i, s in zip([first, 281, 282, 283, 284], [score, 0.1, 0.05, 0.03, 0.02])
    ]


def test_crop_rounding_clipping_and_invalid() -> None:
    assert crop_box([-2.1, 1.2, 20.1, 12.8], 10, 10) == [0, 1, 10, 10]
    for box in [[0, 0, 0, 1], [20, 20, 21, 21], [0, 0, float("nan"), 10]]:
        with pytest.raises(ValueError):
            crop_box(box, 10, 10)


def test_matching_is_class_agnostic_and_one_to_one() -> None:
    gt = [{"id": 4, "bbox_xyxy": [0, 0, 10, 10], "category_id": 18, "iscrowd": 0}]
    preds = [{"bbox_xyxy": [0, 0, 10, 10], "category_id": 17}] * 2
    rows = match_proposals(preds, gt, 0.5)
    assert rows[0]["target_category_id"] == 18
    assert rows[1]["matched_annotation_id"] is None


def test_crowd_ignores_contained_unmatched_proposal() -> None:
    gt = [{"id": 4, "bbox_xyxy": [0, 0, 100, 100], "category_id": 1, "iscrowd": 1}]
    assert match_proposals([{"bbox_xyxy": [1, 1, 10, 10]}], gt, 0.5)[0]["ignored_crowd"]


def test_sparse_category_mapping_and_nonforced_classification() -> None:
    assert mapping()[207] == 18  # golden retriever -> dog, sparse COCO category.
    assert mapping()[340] == 24  # zebra
    assert mapping()[0] is None  # fish has no COCO equivalent.
    config = read_json(CONFIGS / "benchmark.json")
    result = interpret(predictions(), 17, config)
    assert result["status"] == "conflict"
    assert result["effective_category_id"] == 17
    assert result["mapped_top1"] == 18
    assert interpret(predictions(score=0.3), 18, config)["status"] == "uncertain"
    assert interpret(predictions(first=0), 18, config)["status"] == "not_mappable"


def record(truth: int, detector: int, first: int, image_id: int = 1) -> dict:
    return {
        "image_id": image_id,
        "target_category_id": truth,
        "ignored_crowd": False,
        "matched_annotation_id": image_id,
        **interpret(predictions(first=first), detector, read_json(CONFIGS / "benchmark.json")),
    }


def test_quality_counts_corrected_and_introduced_errors() -> None:
    dataset = {
        "mode": "detector_crops",
        "supported_categories": [17, 18],
        "n_scenes": 2,
        "supported_gt_counts": {"17": 1, "18": 2},
        "mapping_object_coverage": 0.75,
    }
    rows = [record(18, 17, 207), record(17, 17, 207, 2)]
    result = quality(rows, dataset)
    assert result["helpful_changes"] == result["harmful_changes"] == 1
    assert (
        result["detector_only_matched_accuracy"]
        == result["hypothetical_gated_matched_accuracy"]
        == 0.5
    )
    assert result["missed_supported_gt"] == 1
    assert result["effective_label_changes"] == 0
    assert result["fine_grained_accuracy"] is None


def test_scene_cluster_keeps_multiple_crops_together() -> None:
    rows = [record(18, 17, 207), record(18, 17, 207), record(18, 18, 281, 2)]
    dataset = {
        "supported_categories": [18],
        "scenes": [{"image_id": 1}, {"image_id": 2}, {"image_id": 3}],
    }
    counts = scene_arrays(rows, dataset)
    assert counts[:, 0].tolist() == [2, 1, 0]
    assert counts[:, 1].tolist() == [2, 0, 0]


def test_resume_rejects_changed_crop_and_truncates_partial_line(tmp_path) -> None:
    source = {"sample_id": "1:gt:4", "crop_xyxy": [1, 2, 3, 4]}
    path = tmp_path / "samples.jsonl"
    row = {"sample_id": source["sample_id"], "source_hash": object_hash(source)}
    path.write_text(json.dumps(row) + "\n{")
    assert recover(path, {source["sample_id"]: source}) == {source["sample_id"]}
    assert path.read_text().endswith("\n")
    with pytest.raises(ValueError):
        recover(path, {source["sample_id"]: {**source, "crop_xyxy": [0, 0, 1, 1]}})
