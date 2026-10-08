from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from classification.datasets import (
    balanced_selection,
    freeze_manifest,
    source_image_name,
    validate_label_names,
)
from classification.metrics import bootstrap_runs, load_samples, quality
from classification.runner import recover_samples, validate_config
from detection.common import object_hash, read_json, sha256, write_json


def record(image_id: str, target: int | None, predictions: list[int]) -> dict:
    return {
        "schema_version": "1.0",
        "image_id": image_id,
        "model_id": "a",
        "target_index": target,
        "predictions": [{"index": i, "score": 0.9 - j * 0.1} for j, i in enumerate(predictions)],
    }


def test_balanced_selection_is_seeded_and_input_order_independent() -> None:
    rows = [{"image_id": f"{c}-{i}", "target_index": c} for c in range(4) for i in range(10)]
    a = balanced_selection(rows, 42, 5)
    assert a == balanced_selection(list(reversed(rows)), 42, 5)
    assert a != balanced_selection(rows, 43, 5)
    assert all(sum(r["target_index"] == c for r in a) == 5 for c in range(4))
    with pytest.raises(ValueError, match="Insufficient"):
        balanced_selection(rows, 42, 11)


def test_quality_macro_f1_and_sparse_confusion() -> None:
    rows = [
        record("a", 0, [0, 1, 2, 3, 4]),
        record("b", 1, [0, 1, 2, 3, 4]),
        record("c", 2, [2, 1, 0, 3, 4]),
    ]
    q = quality(rows, 5)
    assert q["top1"] == pytest.approx(2 / 3)
    assert q["top5"] == 1
    assert q["macro_f1"] == pytest.approx((2 / 3 + 1) / 5)
    assert q["per_class"][3]["accuracy"] is None
    assert q["confusion_sparse"] == [
        {"target": 0, "predicted": 0, "count": 1},
        {"target": 1, "predicted": 0, "count": 1},
        {"target": 2, "predicted": 2, "count": 1},
    ]


def test_smoke_without_gt_cannot_be_quality() -> None:
    with pytest.raises(ValueError, match="ground truth"):
        quality([record("smoke", None, [0, 1, 2, 3, 4])])


def test_frozen_manifest_cannot_be_replaced(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    freeze_manifest(path, {"seed": 42})
    freeze_manifest(path, {"seed": 42})
    with pytest.raises(ValueError, match="Frozen"):
        freeze_manifest(path, {"seed": 43})


def test_resume_discards_only_unfinished_line_and_checks_truth(tmp_path: Path) -> None:
    path = tmp_path / "samples.jsonl"
    good = json.dumps(record("a", 1, [0, 1, 2, 3, 4])) + "\n"
    path.write_text(good + '{"image_id":')
    assert recover_samples(path, {"a": {"target_index": 1}}) == {"a"}
    assert path.read_text() == good
    with pytest.raises(ValueError, match="ground truth"):
        recover_samples(path, {"a": {"target_index": 2}})
    path.write_text(good + good)
    with pytest.raises(ValueError, match="duplicate"):
        recover_samples(path, {"a": {"target_index": 1}})


def test_ground_truth_must_match_manifest(tmp_path: Path) -> None:
    write_json(tmp_path / "run.json", {"model_id": "a"})
    write_json(
        tmp_path / "dataset_manifest.json", {"images": [{"image_id": "a", "target_index": 1}]}
    )
    (tmp_path / "samples.jsonl").write_text(json.dumps(record("a", 0, [0, 1, 2, 3, 4])) + "\n")
    with pytest.raises(ValueError, match="reference label"):
        load_samples(tmp_path)


def test_invalid_top5_and_missing_rows_are_rejected(tmp_path: Path) -> None:
    write_json(tmp_path / "run.json", {"model_id": "a"})
    write_json(
        tmp_path / "dataset_manifest.json", {"images": [{"image_id": "a", "target_index": 1}]}
    )
    (tmp_path / "samples.jsonl").write_text(json.dumps(record("a", 1, [0, 0, 2, 3, 4])) + "\n")
    with pytest.raises(ValueError, match="five unique"):
        load_samples(tmp_path)
    (tmp_path / "samples.jsonl").write_text("")
    with pytest.raises(ValueError, match="Incomplete"):
        load_samples(tmp_path)


def test_synset_reference_label_order_must_match() -> None:
    classes = [{"name": f"class{i}"} for i in range(1000)]
    names = [f"class{i}, synonym" for i in range(1000)]
    validate_label_names(names, classes)
    names[0], names[1] = names[1], names[0]
    with pytest.raises(ValueError, match="ordering mismatch"):
        validate_label_names(names, classes)


def test_hf_source_filename_embedded_synset_is_verified() -> None:
    assert (
        source_image_name("ILSVRC2012_val_00027555_n01824575.JPEG", "n01824575")
        == "ILSVRC2012_val_00027555.JPEG"
    )
    assert (
        source_image_name("ILSVRC2012_val_00027555.JPEG", "n01824575")
        == "ILSVRC2012_val_00027555.JPEG"
    )
    with pytest.raises(ValueError, match="synset mismatch"):
        source_image_name("ILSVRC2012_val_00027555_n01824575.JPEG", "n00000000")


def test_hf_spelling_exceptions_are_limited_to_known_classes() -> None:
    classes = [{"name": f"class{i}"} for i in range(1000)]
    names = [f"class{i}, synonym" for i in range(1000)]
    for i, canonical, source in [
        (134, "crane bird", "crane"),
        (517, "crane", "crane2"),
        (639, "maillot tank suit", "maillot, tank suit"),
    ]:
        classes[i]["name"] = canonical
        names[i] = source
    validate_label_names(names, classes)
    names[517] = "crane bird"
    with pytest.raises(ValueError, match="ordering mismatch"):
        validate_label_names(names, classes)


def test_protocol_rejects_changed_precision_or_participants() -> None:
    from classification.common import CONFIGS

    config = read_json(CONFIGS / "benchmark.json")
    validate_config(config)
    with pytest.raises(ValueError, match="FP32"):
        validate_config({**config, "precision": "fp16"})
    with pytest.raises(ValueError, match="Participants"):
        validate_config({**config, "models": config["models"][::-1]})


def test_paired_bootstrap_preserves_class_design_and_identity(tmp_path: Path) -> None:
    models = ["a", "b", "c", "d"]
    write_json(tmp_path / "experiment.json", {"config": {"seed": 42}, "model_ids": models})
    rows = [record("a", 0, [0, 1, 2, 3, 4]), record("b", 1, [0, 1, 2, 3, 4])]
    dataset = {
        "images": [{"image_id": r["image_id"], "target_index": r["target_index"]} for r in rows]
    }
    for model in models:
        run = tmp_path / f"{model}_repeat1"
        run.mkdir()
        write_json(run / "dataset_manifest.json", dataset)
        (run / "samples.jsonl").write_text(
            "".join(json.dumps({**r, "model_id": model}) + "\n" for r in rows)
        )
        write_json(
            run / "run.json",
            {
                "model_id": model,
                "status": "complete",
                "shortened": False,
                "samples_sha256": sha256(run / "samples.jsonl"),
            },
        )
    bootstrap_runs(tmp_path, 5, 42)
    result = read_json(tmp_path / "bootstrap.json")
    assert result["ci95"]["a"]["top1"] == pytest.approx([0.5, 0.5])
    assert all(p["top1"] == [0.0, 0.0] for p in result["paired_difference_ci95"].values())
    bootstrap_runs(tmp_path, 5, 42)
    assert object_hash(read_json(tmp_path / "bootstrap.json")) == object_hash(result)
    with pytest.raises(ValueError, match="inputs changed"):
        bootstrap_runs(tmp_path, 5, 43)


def test_data_paths_cannot_escape_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import classification.common as common

    monkeypatch.setattr(common, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="inside data"):
        common.data_path("data/../../outside.jpg")
    (tmp_path / "data").mkdir()
    (tmp_path / "data/escape").symlink_to(tmp_path.parent)
    with pytest.raises(ValueError, match="inside data"):
        common.data_path("data/escape/outside.jpg")


def test_missing_runs_produce_four_empty_quality_rows(tmp_path: Path) -> None:
    import csv

    from classification.report import build_report

    build_report(tmp_path / "missing", tmp_path / "report")
    with (tmp_path / "report/classification.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert all(r["status"] == "not_run" and r["top1"] == r["top5"] == "" for r in rows)


def test_default_report_does_not_read_or_publish_intervals(tmp_path: Path) -> None:
    import csv

    from classification.report import build_report

    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "bootstrap.json").write_text("legacy artifact must not be read")
    output = tmp_path / "report"
    build_report(runs, output)
    with (output / "classification.csv").open(encoding="utf-8-sig") as handle:
        assert not any("ci95" in key for key in csv.DictReader(handle).fieldnames)
    assert not (output / "classification_paired.csv").exists()


def test_model_child_retains_shared_lock_when_parent_is_killed(tmp_path: Path) -> None:
    from detection.common import ROOT, file_lock

    lock = tmp_path / "inference.lock"
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    log = tmp_path / "process.log"
    child_code = (
        "import sys,time; from pathlib import Path; "
        "Path(sys.argv[1]).touch(); "
        "exec('while not Path(sys.argv[2]).exists():\\n time.sleep(0.02)')"
    )
    # Pass code as argv rather than interpolating it into another code string.
    parent_code = (
        "import sys; from pathlib import Path; "
        "from detection.common import file_lock; from detection.runner import child; "
        "exec('with file_lock(Path(sys.argv[1])) as fd:\\n child([sys.executable, \"-c\", sys.argv[5], sys.argv[2], sys.argv[3]], fd, Path(sys.argv[4]))')"
    )
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            parent_code,
            str(lock),
            str(ready),
            str(release),
            str(log),
            child_code,
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and parent.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), log.read_text() if log.exists() else "Child failed to start"
        parent.kill()
        parent.wait(timeout=5)
        with pytest.raises(RuntimeError, match="Lock already held"):
            with file_lock(lock):
                pass
    finally:
        release.touch()
        if parent.poll() is None:
            parent.terminate()
            parent.wait(timeout=5)
    deadline = time.monotonic() + 5
    while True:
        try:
            with file_lock(lock):
                break
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.02)
