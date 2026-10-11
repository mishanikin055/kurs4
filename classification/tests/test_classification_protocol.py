"""CPU invariants for the five-file, single-pass ImageNet comparison."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts.classification import common


def record(image_id: str, target: int | None, predictions: list[int]) -> dict:
    classes = common.labels()
    return {
        "image_id": image_id,
        "model_id": "resnet50",
        "target_index": target,
        "target_synset": classes[target]["synset"] if target is not None else None,
        "predictions": [
            {**classes[index], "score": score}
            for index, score in zip(predictions, [0.4, 0.2, 0.1, 0.08, 0.05], strict=True)
        ],
        "timing": {
            "preprocess_ms": 1.0,
            "inference_ms": 2.0,
            "postprocess_ms": 0.5,
            "total_ms": 3.5,
            "read_ms": 0.1,
        },
    }


def test_quality_macro_f1_and_sparse_confusion() -> None:
    rows = [
        record("a", 0, [0, 1, 2, 3, 4]),
        record("b", 1, [0, 1, 2, 3, 4]),
        record("c", 2, [2, 1, 0, 3, 4]),
    ]
    metrics = common.quality(rows, 5)
    assert metrics["top1"] == pytest.approx(2 / 3)
    assert metrics["top5"] == 1
    assert metrics["macro_f1"] == pytest.approx((2 / 3 + 1) / 5)
    assert metrics["per_class"][3]["accuracy"] is None
    assert metrics["confusion_sparse"] == [
        {"target": 0, "predicted": 0, "count": 1},
        {"target": 1, "predicted": 0, "count": 1},
        {"target": 2, "predicted": 2, "count": 1},
    ]
    assert len(common.quality(rows)["per_class"]) == 1000


def test_quality_requires_ground_truth() -> None:
    with pytest.raises(ValueError, match="ground truth"):
        common.quality([record("smoke", None, [0, 1, 2, 3, 4])])


@pytest.mark.parametrize(
    "corruption", ["synset", "name", "nan", "timing", "duplicate", "missing", "truth"]
)
def test_corrupt_predictions_rejected(tmp_path: Path, corruption: str) -> None:
    row = record("a", 1, [0, 1, 2, 3, 4])
    sources = [{key: row[key] for key in ("image_id", "target_index", "target_synset")}]
    common.write_json(tmp_path / "run.json", {"model_id": "resnet50"})
    if corruption == "synset":
        row["predictions"][0]["synset"] = "wrong"
    elif corruption == "name":
        row["predictions"][0]["name"] = "wrong"
    elif corruption == "nan":
        row["predictions"][0]["score"] = float("nan")
    elif corruption == "timing":
        row["timing"]["total_ms"] = 100.0
    elif corruption == "duplicate":
        row["predictions"][1] = row["predictions"][0]
    elif corruption == "truth":
        row["target_index"] = 0
    text = "" if corruption == "missing" else json.dumps(row) + "\n"
    (tmp_path / "samples.jsonl").write_text(text)
    with pytest.raises(ValueError):
        common.load_samples(tmp_path, sources)


def test_duplicate_and_truncated_sample_rows_rejected(tmp_path: Path) -> None:
    row = record("a", 1, [0, 1, 2, 3, 4])
    sources = [{key: row[key] for key in ("image_id", "target_index", "target_synset")}]
    common.write_json(tmp_path / "run.json", {"model_id": "resnet50"})
    line = json.dumps(row) + "\n"
    (tmp_path / "samples.jsonl").write_text(line + line)
    with pytest.raises(ValueError, match="Duplicate"):
        common.load_samples(tmp_path, sources)
    (tmp_path / "samples.jsonl").write_text(line + '{"unfinished":')
    with pytest.raises(ValueError, match="Incomplete final"):
        common.load_samples(tmp_path, sources)
    assert common.load_samples(tmp_path, sources, complete=False) == [row]


def test_data_paths_cannot_escape_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(common, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="inside data"):
        common.data_path("data/../../outside.jpg")
    (tmp_path / "data").mkdir()
    (tmp_path / "data/escape").symlink_to(tmp_path.parent)
    with pytest.raises(ValueError, match="inside data"):
        common.data_path("data/escape/outside.jpg")


@pytest.fixture
def comparison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict]:
    row = record("a", 1, [0, 1, 2, 3, 4])
    manifest = tmp_path / "data/validation.json"
    source = {key: row[key] for key in ("image_id", "target_index", "target_synset")}
    common.write_json(manifest, {"images": [source]})
    monkeypatch.setattr(common, "ROOT", tmp_path)
    identity = {
        "dataset": {"path": "data/validation.json", "sha256": common.sha256(manifest)},
        "labels_sha256": common.sha256(common.CONFIGS / "imagenet_labels.json"),
        "models_sha256": common.sha256(common.CONFIGS / "models.json"),
        "expected_images": 1,
        "shortened": True,
        "config": common.comparison_config(1),
    }
    output = tmp_path / "run"
    common.write_json(output / "comparison.json", identity)
    return output, row


def test_missing_runs_report_four_rows_without_intervals(
    comparison: tuple, capsys: pytest.CaptureFixture
) -> None:
    output, _ = comparison
    (output / "bootstrap.json").write_text("legacy artifact must not be read")
    common.build_report(output)
    with (output / "comparison.csv").open(encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        assert not any("ci" in key.lower() for key in reader.fieldnames)
        rows = list(reader)
    stdout = capsys.readouterr().out
    assert all(common.model_spec(model)["name"] in stdout for model in common.MODEL_IDS)
    assert "Top1" in stdout and "Top5" in stdout and "p50_ms" in stdout
    assert stdout.count("not_run") == 4
    assert len(rows) == 4
    assert all(row["status"] == "not_run" and row["top1"] == row["top5"] == "" for row in rows)


@pytest.mark.parametrize("status", ["complete", "failed", "oom", "running"])
def test_report_preserves_status_and_short_run_label(comparison: tuple, status: str) -> None:
    output, row = comparison
    run = output / "resnet50"
    run.mkdir()
    (run / "samples.jsonl").write_text(json.dumps(row) + "\n")
    common.write_json(
        run / "run.json",
        {
            "model_id": "resnet50",
            "status": status,
            "processed": 1,
            "comparison_hash": common.object_hash(common.read_json(output / "comparison.json")),
            "samples_sha256": common.sha256(run / "samples.jsonl"),
            "resources": {},
        },
    )
    common.build_report(output)
    with (output / "comparison.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["status"] == ("incomplete" if status == "running" else status)
    assert rows[0]["shortened"] == "True"
    if status == "complete":
        assert rows[0]["top1"] == "0.0"
        assert rows[0]["top5"] == "1.0"
        with (output / "per_class.csv").open(encoding="utf-8-sig") as handle:
            assert len(list(csv.DictReader(handle))) == 1000
    else:
        assert rows[0]["top1"] == rows[0]["top5"] == ""


def test_native_metadata_matches_all_fixed_checkpoints() -> None:
    from torchvision.models import (
        ConvNeXt_Tiny_Weights,
        EfficientNet_V2_S_Weights,
        ResNet50_Weights,
        ViT_B_16_Weights,
    )

    for model_id, weights in zip(
        common.MODEL_IDS,
        (
            ResNet50_Weights.IMAGENET1K_V2,
            EfficientNet_V2_S_Weights.IMAGENET1K_V1,
            ConvNeXt_Tiny_Weights.IMAGENET1K_V1,
            ViT_B_16_Weights.IMAGENET1K_V1,
        ),
        strict=True,
    ):
        assert common.validate_native(model_id, weights)[
            "class_mapping_hash"
        ] == common.object_hash(common.labels())


def test_inherited_shared_lock_survives_killed_parent(tmp_path: Path) -> None:
    lock, ready, release = (tmp_path / name for name in ("lock", "ready", "release"))
    child_code = (
        "import sys,time; from pathlib import Path; Path(sys.argv[1]).touch(); "
        "exec('while not Path(sys.argv[2]).exists():\\n time.sleep(0.02)')"
    )
    parent_code = (
        "import sys,os; from pathlib import Path; "
        "from scripts.classification.common import file_lock,child; "
        "exec('with file_lock(Path(sys.argv[1])) as fd:\\n "
        'child([sys.executable, "-c", sys.argv[4], sys.argv[2], sys.argv[3]], fd, os.environ.copy())\')'
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code, str(lock), str(ready), str(release), child_code],
        cwd=common.ROOT,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and parent.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        parent.kill()
        parent.wait(timeout=5)
        with pytest.raises(RuntimeError, match="Lock already held"):
            with common.file_lock(lock):
                pass
    finally:
        release.touch()
        if parent.poll() is None:
            parent.terminate()
            parent.wait(timeout=5)
    deadline = time.monotonic() + 5
    while True:
        try:
            with common.file_lock(lock):
                break
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.02)


def test_internal_worker_requires_correct_locked_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.delenv("KURS4_LOCK_FD", raising=False)
    with pytest.raises(RuntimeError, match="valid inherited"):
        common.require_inherited_lock()
    lock = tmp_path / "storage/locks/inference.lock"
    with common.file_lock(lock) as descriptor:
        monkeypatch.setenv("KURS4_LOCK_FD", str(descriptor))
        common.require_inherited_lock()
        with (tmp_path / "other").open("w") as other:
            monkeypatch.setenv("KURS4_LOCK_FD", str(other.fileno()))
            with pytest.raises(RuntimeError, match="different file"):
                common.require_inherited_lock()
    with lock.open("a+") as unlocked:
        monkeypatch.setenv("KURS4_LOCK_FD", str(unlocked.fileno()))
        with pytest.raises(RuntimeError, match="not held"):
            common.require_inherited_lock()


def test_cgroup_oom_is_based_on_new_kill_event() -> None:
    assert common.cgroup_oom({"oom_kill": 2}, {"oom_kill": 3})
    assert not common.cgroup_oom({"oom_kill": 2}, {"oom_kill": 2})
    assert not common.cgroup_oom({"oom": 2}, {"oom": 3})
    assert not common.cgroup_oom({}, {})


def test_sigterm_parent_waits_for_model_child_and_releases_lock(tmp_path: Path) -> None:
    import os
    import signal

    lock, ready = (tmp_path / name for name in ("lock", "ready"))
    child_code = (
        "import sys,time,os; from pathlib import Path; "
        "Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
    )
    parent_code = (
        "import sys,os; from pathlib import Path; "
        "from scripts.classification.common import file_lock,child; "
        "exec('with file_lock(Path(sys.argv[1])) as fd:\\n "
        'child([sys.executable, "-c", sys.argv[3], sys.argv[2]], fd, os.environ.copy())\')'
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code, str(lock), str(ready), child_code],
        cwd=common.ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        child_pid = int(ready.read_text())
        parent.send_signal(signal.SIGTERM)
        parent.wait(timeout=15)
        with common.file_lock(lock):
            pass
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if parent.poll() is None:
            parent.terminate()
            parent.wait(timeout=15)


def test_report_only_does_not_instantiate_model(
    comparison: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _ = comparison
    output_new = common.ROOT / "reports/runs/classification/report-test"
    output_new.parent.mkdir(parents=True)
    output.rename(output_new)
    monkeypatch.delenv("KURS4_MODEL_RUN", raising=False)
    monkeypatch.setattr(sys, "argv", ["resnet50.py", "--run", "report-test", "--report-only"])

    class ForbiddenModel:
        def __init__(self, device: str) -> None:
            raise AssertionError("report-only loaded a model")

    common.main(ForbiddenModel, "resnet50")
    assert (output_new / "comparison.csv").exists()


@pytest.fixture
def parent_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    row = record("a", 1, [0, 1, 2, 3, 4])
    source = {key: row[key] for key in ("image_id", "target_index", "target_synset")}
    common.write_json(tmp_path / "data/imagenet/validation.json", {"images": [source]})
    monkeypatch.setattr(common, "ROOT", tmp_path)
    monkeypatch.setattr(common, "validate_dataset", lambda value: None)
    monkeypatch.setattr(common, "source_fingerprint", lambda: {"test-source": "fixed"})
    monkeypatch.delenv("KURS4_MODEL_RUN", raising=False)
    monkeypatch.setattr(sys, "argv", ["resnet50.py", "--run", "parent-test", "--limit", "1"])
    return tmp_path / "reports/runs/classification/parent-test"


@pytest.mark.parametrize("oom", [False, True])
def test_abrupt_child_death_records_incomplete_or_real_cgroup_oom(
    parent_case: Path,
    monkeypatch: pytest.MonkeyPatch,
    oom: bool,
) -> None:
    events = iter([{"oom_kill": 2}, {"oom_kill": 3 if oom else 2}])
    monkeypatch.setattr(common, "memory_events", lambda: next(events))
    monkeypatch.setattr(common, "child", lambda command, descriptor, environment: -9)
    with pytest.raises(SystemExit) as result:
        common.main(object, "resnet50")
    assert result.value.code == 1
    meta = common.read_json(parent_case / "resnet50/run.json")
    assert meta["status"] == ("oom" if oom else "incomplete")
    assert meta["processed"] == 0
    assert meta["cgroup_memory_events"]["after"]["oom_kill"] == (3 if oom else 2)
    with (parent_case / "comparison.csv").open(encoding="utf-8-sig") as handle:
        assert list(csv.DictReader(handle))[0]["status"] == meta["status"]


def test_existing_model_and_changed_limit_cannot_be_overwritten(
    parent_case: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(common, "child", lambda command, descriptor, environment: 1)
    with pytest.raises(SystemExit):
        common.main(object, "resnet50")
    original = (parent_case / "resnet50/run.json").read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        common.main(object, "resnet50")
    monkeypatch.setattr(sys, "argv", ["resnet50.py", "--run", "parent-test"])
    with pytest.raises(ValueError, match="settings/data/source/container differ"):
        common.main(object, "efficientnet_v2_s")
    assert (parent_case / "resnet50/run.json").read_bytes() == original
