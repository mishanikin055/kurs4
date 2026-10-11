"""COCO metric, mapping, coordinate and process-lifetime invariants, without weights."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from pycocotools.coco import COCO

from scripts.detection.common import evaluate_coco, file_lock, normalize_predictions, to_coco_label


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
    from scripts.detection.common import object_hash

    value = {"counts": {2: 1, 10: 4}}
    assert object_hash(value) == object_hash(json.loads(json.dumps(value)))


def test_normalize_rejects_fractional_and_invalid_labels() -> None:
    for label in (1.5, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            normalize_predictions([[0, 0, 3, 4]], [0.5], [label], "coco_sparse", 20, 30, 0.001, 300)


def create_comparison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple:
    from scripts.detection import common

    monkeypatch.setattr(common, "ROOT", tmp_path)
    manifest = {
        "annotations": "data/gt.json",
        "images": [
            {"image_id": i, "path": f"data/{i}.jpg", "sha256": "x", "width": 30, "height": 20}
            for i in range(4500)
        ],
    }
    common.write_json(tmp_path / "data/gt.json", fixture_gt().dataset)
    manifest["annotations_sha256"] = common.sha256(tmp_path / "data/gt.json")
    common.write_json(tmp_path / "data/coco/test.json", manifest)
    common.write_json(
        tmp_path / "detection/configs/dataset_split.json",
        {"splits": {"test": {"manifest_hash": common.object_hash(manifest)}}},
    )
    monkeypatch.setattr(common, "source_files", lambda: {"test": "fixed"})
    monkeypatch.setattr(common, "verify_model", lambda _: {"status": "verified"})
    output = tmp_path / "reports/runs/detection/unit"
    return common, output


def test_four_rows_failure_no_overwrite_and_incompatible_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import csv

    common, output = create_comparison(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["model.py", "--run", "unit", "--limit", "2"])
    monkeypatch.setattr(common, "run_child", lambda *_: 17)
    with pytest.raises(RuntimeError, match="failed"):
        common.main(object, "yolo26s")
    with (output / "comparison.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert rows[0]["status"] == "failed"
    assert [r["status"] for r in rows[1:]] == ["not_run"] * 3
    with pytest.raises(FileExistsError):
        common.main(object, "yolo26s")
    monkeypatch.setattr(sys, "argv", ["model.py", "--run", "unit", "--limit", "3"])
    with pytest.raises(ValueError, match="differ"):
        common.main(object, "rtdetr_v2_r18vd")


def test_report_only_uses_saved_samples_without_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import csv

    common, output = create_comparison(tmp_path, monkeypatch)
    common.write_json(output / "comparison.json", {"expected_images": 2, "shortened": True})
    run = output / "yolo26s"
    annotation = tmp_path / "data/gt.json"
    common.write_json(
        run / "run.json",
        {
            "status": "inference_complete",
            "expected_images": 2,
            "shortened": True,
            "dataset": {
                "annotations": "data/gt.json",
                "annotations_sha256": common.sha256(annotation),
                "images": [{"image_id": 7}, {"image_id": 8}],
            },
            "resources": {},
            "cold_load_ms": 2,
        },
    )
    samples = [
        {
            "image_id": 7,
            "predictions": [{"category_id": 18, "bbox": [10, 10, 20, 20], "score": 0.9}],
            "timing": {
                "total_ms": 1,
                "preprocess_ms": 0.2,
                "forward_ms": 0.6,
                "postprocess_ms": 0.2,
                "read_ms": 0.1,
            },
        },
        {
            "image_id": 8,
            "predictions": [],
            "timing": {
                "total_ms": 1,
                "preprocess_ms": 0.2,
                "forward_ms": 0.6,
                "postprocess_ms": 0.2,
                "read_ms": 0.1,
            },
        },
    ]
    (run / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in samples))
    monkeypatch.setattr(
        common, "verify_model", lambda _: pytest.fail("Model verification in report-only")
    )
    monkeypatch.setattr(common, "run_child", lambda *_: pytest.fail("Inference in report-only"))
    monkeypatch.setattr(sys, "argv", ["model.py", "--run", "unit", "--report-only"])
    common.main(object, "yolo26s")
    with (output / "comparison.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["status"] == "complete"
    assert float(rows[0]["mAP"]) == pytest.approx(1)


def test_incomplete_never_has_final_quality_and_preserves_truncated_samples(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common, output = create_comparison(tmp_path, monkeypatch)
    common.write_json(output / "comparison.json", {"expected_images": 2, "shortened": True})
    run = output / "rf_detr_small"
    common.write_json(
        run / "run.json",
        {
            "status": "running",
            "processed": 0,
            "dataset": {"images": [{"image_id": 7}, {"image_id": 8}]},
        },
    )
    raw = json.dumps({"image_id": 7, "predictions": []}) + '\n{"image_id":'
    (run / "samples.jsonl").write_text(raw)
    common.evaluate_run(run)
    assert common.read_json(run / "run.json")["status"] == "incomplete"
    assert common.read_json(run / "run.json")["processed"] == 1
    assert (run / "samples.jsonl").read_text() == raw
    common.build_report(output)
    assert "incomplete" in (output / "comparison.csv").read_text()


def test_oom_status_stays_explicit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    common, output = create_comparison(tmp_path, monkeypatch)

    def fake_child(_command: list, _fd: int, run: Path) -> int:
        meta = common.read_json(run / "run.json")
        meta.update(status="oom", error="CUDA out of memory")
        common.write_json(run / "run.json", meta)
        return 1

    monkeypatch.setattr(common, "run_child", fake_child)
    monkeypatch.setattr(sys, "argv", ["model.py", "--run", "unit", "--limit", "2"])
    with pytest.raises(RuntimeError, match="oom"):
        common.main(object, "yolo26s")
    assert common.read_json(output / "yolo26s/run.json")["status"] == "oom"


def test_missing_weights_are_recorded_as_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common, output = create_comparison(tmp_path, monkeypatch)

    def missing(_model: str) -> None:
        raise FileNotFoundError("Missing pinned checkpoint")

    monkeypatch.setattr(common, "verify_model", missing)
    monkeypatch.setattr(sys, "argv", ["model.py", "--run", "unit", "--limit", "2"])
    with pytest.raises(FileNotFoundError):
        common.main(object, "yolo26s")
    assert common.read_json(output / "yolo26s/run.json")["status"] == "failed"


def test_lock_contention_prevents_creation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    common, output = create_comparison(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", ["model.py", "--run", "unit", "--limit", "2"])
    with file_lock(tmp_path / "storage/locks/inference.lock"):
        with pytest.raises(RuntimeError, match="held"):
            common.main(object, "yolo26s")
    assert not output.exists()


def test_exif_rejected_before_any_coordinate_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image

    from scripts.detection import common

    monkeypatch.setattr(common, "ROOT", tmp_path)
    path = tmp_path / "data/image.jpg"
    path.parent.mkdir()
    image = Image.new("RGB", (40, 20))
    exif = image.getexif()
    exif[274] = 6
    image.save(path, exif=exif)
    with pytest.raises(ValueError, match="EXIF"):
        common.open_image(
            {"path": "data/image.jpg", "sha256": common.sha256(path), "width": 40, "height": 20}
        )


def test_sigterm_stops_model_process_group_and_releases_lock(tmp_path: Path) -> None:
    import signal
    import time

    from scripts.detection import common

    ready = tmp_path / "pids.txt"
    run = tmp_path / "run"
    run.mkdir()
    lock = tmp_path / "inference.lock"
    child_code = (
        "import os, subprocess, sys, time; from pathlib import Path; "
        "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
        f"Path({str(ready)!r}).write_text(str(os.getpid()) + ' ' + str(p.pid)); "
        "time.sleep(120)"
    )
    parent_code = (
        "import sys; from pathlib import Path; from scripts.detection import common; "
        f"lock = Path({str(lock)!r}); run = Path({str(run)!r}); "
        "handle = common.file_lock(lock); fd = handle.__enter__(); "
        f"common.run_child([sys.executable, '-c', {child_code!r}], fd, run)"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code],
        cwd=common.ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), "Mock model subprocess did not start"
        pids = [int(v) for v in ready.read_text().split()]
        parent.send_signal(signal.SIGTERM)
        parent.wait(timeout=15)
        for pid in pids:
            stat = Path(f"/proc/{pid}/stat")
            assert not stat.exists() or stat.read_text().split()[2] == "Z"
        with file_lock(lock):
            pass
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()


def test_report_limit_cannot_change_saved_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    common, output = create_comparison(tmp_path, monkeypatch)
    common.write_json(
        output / "comparison.json", {"expected_images": 2, "shortened": True, "limit": 2}
    )
    monkeypatch.setattr(sys, "argv", ["model.py", "--run", "unit", "--report-only", "--limit", "3"])
    with pytest.raises(ValueError, match="limit differs"):
        common.main(object, "yolo26s")
