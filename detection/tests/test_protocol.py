"""Protocol invariants and evaluator edge cases; no model downloads."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pycocotools.coco import COCO

from detection.adapters import normalize_predictions, to_coco_label
from detection.common import file_lock
from detection.metrics import evaluate_coco
from detection.runner import recover_samples, validate_config


def fixture_gt() -> COCO:
    gt = COCO()
    gt.dataset = {
        "info": {},
        "images": [{"id": 7, "width": 100, "height": 80}, {"id": 8, "width": 100, "height": 80}],
        "categories": [{"id": 18, "name": "dog"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 7,
                "category_id": 18,
                "bbox": [10, 10, 20, 20],
                "area": 400,
                "iscrowd": 0,
            },
            {
                "id": 2,
                "image_id": 8,
                "category_id": 18,
                "bbox": [40, 10, 20, 20],
                "area": 400,
                "iscrowd": 1,
            },
        ],
    }
    gt.createIndex()
    return gt


def test_coco_mapping_sparse_gaps() -> None:
    assert to_coco_label(11, "contiguous_0") == 13
    assert to_coco_label(79, "contiguous_0") == 90
    assert to_coco_label(18, "coco_sparse") == 18
    for value in (0, 12, 26, 91):
        with pytest.raises(ValueError):
            to_coco_label(value, "coco_sparse")


def test_bbox_clipping_and_xywh_non_square() -> None:
    rows = normalize_predictions(
        [[-5, 3, 120, 90]], [0.8], [16], "contiguous_0", 100, 80, 0.001, 300
    )
    assert rows[0]["bbox"] == [0, 3, 100, 77]
    assert rows[0]["bbox_xyxy"] == [0, 3, 100, 80]
    assert rows[0]["category_id"] == 18
    with pytest.raises(ValueError):
        normalize_predictions(
            [[0, 0, 5, 5]], [float("nan")], [16], "contiguous_0", 100, 80, 0.001, 300
        )


def test_rf_detr_unused_slots_are_excluded_without_remapping() -> None:
    boxes = [[0, 0, 10, 10]] * 4
    rows = normalize_predictions(
        boxes,
        [0.9, 0.8, 0.7, 0.6],
        [66, 0, 67, 90],
        "coco_sparse",
        100,
        80,
        0.001,
        300,
        exclude_coco_gaps=True,
    )
    assert [row["category_id"] for row in rows] == [67, 90]
    with pytest.raises(ValueError):
        normalize_predictions(boxes[:1], [0.9], [66], "coco_sparse", 100, 80, 0.001, 300)
    for invalid in (-1, 91):
        with pytest.raises(ValueError):
            normalize_predictions(
                boxes[:1],
                [0.9],
                [invalid],
                "coco_sparse",
                100,
                80,
                0.001,
                300,
                exclude_coco_gaps=True,
            )


def test_recovery_rejects_changes_outside_rf_detr_fix() -> None:
    from scripts.recover_detection import validate_adapter_change

    old = "def preprocess(x):\n    return x / 255\n"
    validate_adapter_change(old, old)
    with pytest.raises(ValueError):
        validate_adapter_change(old, old.replace("255", "256"))


def test_cached_bootstrap_matches_full_coco_with_ties_duplicates_and_crowd() -> None:
    import copy

    import numpy as np

    from detection.metrics import bootstrap_evaluator, cached_bootstrap_ap

    gt = fixture_gt()
    gt.dataset["images"].append({"id": 9, "width": 100, "height": 80})
    gt.dataset["annotations"].append(
        {
            "id": 3,
            "image_id": 9,
            "category_id": 18,
            "bbox": [0, 0, 10, 10],
            "area": 100,
            "iscrowd": 0,
        }
    )
    gt.createIndex()
    preds = [
        {"image_id": 7, "category_id": 18, "bbox": [0, 0, 5, 5], "score": 0.8},
        {"image_id": 7, "category_id": 18, "bbox": [10, 10, 20, 20], "score": 0.8},
        {"image_id": 8, "category_id": 18, "bbox": [40, 10, 20, 20], "score": 0.8},
        {"image_id": 9, "category_id": 18, "bbox": [0, 0, 9, 10], "score": 0.7},
    ]
    ids = [7, 8, 9]
    ev = bootstrap_evaluator(gt, preds, ids)
    cache = ev.evalImgs
    for indices in ([0, 0, 2], [2, 0, 2], [0, 1, 2], [2, 1, 0]):
        replay = COCO()
        replay.dataset = {
            "info": {},
            "categories": copy.deepcopy(gt.dataset["categories"]),
            "images": [],
            "annotations": [],
        }
        replay_predictions = []
        for new_id, index in enumerate(indices):
            old_id = ids[index]
            replay.dataset["images"].append({"id": new_id, "width": 100, "height": 80})
            for ann in gt.dataset["annotations"]:
                if ann["image_id"] == old_id:
                    replay.dataset["annotations"].append(
                        {**ann, "image_id": new_id, "id": len(replay.dataset["annotations"]) + 1}
                    )
            replay_predictions.extend(
                {**p, "image_id": new_id} for p in preds if p["image_id"] == old_id
            )
        replay.createIndex()
        direct = evaluate_coco(replay, replay_predictions, list(range(len(indices))))["quality"][
            "mAP"
        ]
        assert cached_bootstrap_ap(ev, cache, np.array(indices)) == pytest.approx(direct, abs=1e-12)


def test_empty_and_perfect_predictions_with_crowd() -> None:
    gt = fixture_gt()
    empty = evaluate_coco(gt, [], [7, 8])["quality"]
    assert empty["mAP"] == 0
    assert empty["FN"] == 1
    preds = [
        {"image_id": 7, "category_id": 18, "bbox": [10, 10, 20, 20], "score": 0.99},
        {"image_id": 8, "category_id": 18, "bbox": [40, 10, 20, 20], "score": 0.99},
    ]
    results = evaluate_coco(gt, preds, [7, 8])["quality"]
    assert results["mAP"] == pytest.approx(1)
    assert (results["TP"], results["FP"], results["FN"]) == (1, 0, 0)


def test_ap_not_cut_by_operating_threshold() -> None:
    preds = [{"image_id": 7, "category_id": 18, "bbox": [10, 10, 20, 20], "score": 0.2}]
    quality = evaluate_coco(fixture_gt(), preds, [7, 8], threshold=0.5)["quality"]
    assert quality["mAP"] == pytest.approx(1)
    assert quality["recall"] == 0


def test_crash_resume_truncates_only_incomplete_last_record(tmp_path: Path) -> None:
    p = tmp_path / "samples.jsonl"
    complete = json.dumps({"image_id": 7}) + "\n"
    p.write_text(complete + '{"image_id":')
    assert recover_samples(p) == {7}
    assert p.read_text() == complete
    p.write_text(complete + complete)
    with pytest.raises(ValueError):
        recover_samples(p)


def test_lock_held_by_child_after_parent_closes(tmp_path: Path) -> None:
    path = tmp_path / "inference.lock"
    # This simulates a dead parent's close without issuing flock(LOCK_UN).
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    import fcntl

    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    process = subprocess.Popen(
        [sys.executable, "-c", "import sys; print('ready',flush=True); sys.stdin.read()"],
        pass_fds=(fd,),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout.readline().strip() == "ready"
        os.close(fd)
        with pytest.raises(RuntimeError):
            with file_lock(path):
                pass
    finally:
        process.communicate("")
    with file_lock(path):
        pass


def test_incompatible_modes_rejected() -> None:
    config = {"batch_size": 2, "precision": "fp32"}
    with pytest.raises(ValueError):
        validate_config(config)


def test_false_positive_on_image_with_no_gt_for_category() -> None:
    gt = fixture_gt()
    gt.dataset["categories"].append({"id": 1, "name": "person"})
    gt.createIndex()
    predictions = [{"image_id": 7, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9}]
    quality = evaluate_coco(gt, predictions, [7, 8])["quality"]
    assert quality["FP"] == 1
    assert quality["FN"] == 1
    assert quality["mAP"] == 0


def test_hash_is_stable_after_json_roundtrip() -> None:
    from detection.common import object_hash

    value = {"counts": {2: 1, 10: 4}}
    assert object_hash(value) == object_hash(json.loads(json.dumps(value)))


def test_paired_bootstrap_reindexes_duplicate_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import detection.common as common
    import detection.metrics as metrics

    gt = fixture_gt()
    gt.dataset["annotations"][1]["iscrowd"] = 0
    annotation = tmp_path / "gt.json"
    annotation.write_text(json.dumps(gt.dataset))
    manifest = {"images": [{"image_id": 7}, {"image_id": 8}], "annotations": "gt.json"}
    runs = []
    for model in ("a", "b", "c", "d"):
        path = tmp_path / model
        path.mkdir()
        (path / "run.json").write_text(
            json.dumps({"repeat": 0, "model_id": model, "shortened": True})
        )
        (path / "dataset_manifest.json").write_text(json.dumps(manifest))
        runs.append(path)
    predictions = [
        {"image_id": 7, "category_id": 18, "bbox": [10, 10, 20, 20], "score": 0.99},
        {"image_id": 8, "category_id": 18, "bbox": [40, 10, 20, 20], "score": 0.99},
    ]
    for run in runs:
        (run / "samples.jsonl").write_text(
            "".join(
                json.dumps(
                    {"image_id": pred["image_id"], "predictions": [pred], "timing": {"total_ms": 1}}
                )
                + "\n"
                for pred in predictions
            )
        )
    monkeypatch.setattr(common, "ROOT", tmp_path)
    output = tmp_path / "bootstrap.json"
    metrics.bootstrap_runs(runs, 4, 42, output)
    result = json.loads(output.read_text())
    assert result["samples"] == 4
    assert all(interval == pytest.approx([1, 1]) for interval in result["mAP_ci95"].values())
    assert all(
        interval == pytest.approx([0, 0])
        for interval in result["paired_mAP_difference_ci95"].values()
    )
    metrics.bootstrap_runs(runs, 4, 42, output)
    assert json.loads(output.read_text()) == result
    with pytest.raises(ValueError, match="checkpoint differs"):
        metrics.bootstrap_runs(runs, 4, 43, output)
