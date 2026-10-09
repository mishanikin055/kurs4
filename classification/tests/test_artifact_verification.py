"""Corruption checks for the independent completed-artifact verifier."""

from __future__ import annotations

import copy
import csv
import json
import tarfile
from pathlib import Path

import pytest

from classification.common import labels
from classification.metrics import quality
from detection.common import object_hash, read_json, sha256, write_json
from scripts.verify_classification import whole_quality


@pytest.fixture
def completed_run(tmp_path: Path) -> tuple[Path, dict, list[dict]]:
    classes = labels()
    predictions = [
        {**classes[i], "score": score}
        for i, score in zip(range(5), [0.5, 0.2, 0.1, 0.08, 0.05], strict=True)
    ]
    rows = [
        {
            "image_id": str(target),
            "model_id": "resnet50",
            "model_version_id": "test-checkpoint",
            "target_index": target,
            "target_synset": classes[target]["synset"],
            "predictions": copy.deepcopy(predictions),
            "timing": {
                "preprocess_ms": 1.0,
                "inference_ms": 2.0,
                "postprocess_ms": 0.5,
                "total_ms": 3.5,
            },
        }
        for target in [0, 1, 5]
    ]
    dataset = {
        "images": [
            {k: row[k] for k in ["image_id", "target_index", "target_synset"]} for row in rows
        ]
    }
    write_json(
        tmp_path / "run.json", {"model_id": "resnet50", "model_version_id": "test-checkpoint"}
    )
    write_json(tmp_path / "metrics.json", quality(rows))
    (tmp_path / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return tmp_path, dataset, rows


def test_verifier_recounts_quality_and_ignores_timing_in_parity(completed_run: tuple) -> None:
    run, dataset, rows = completed_run
    metrics, parity = whole_quality(run, dataset)
    assert metrics["correct_top1"] == 1
    assert metrics["correct_top5"] == 2
    for row in rows:
        row["timing"]["inference_ms"] += 1
        row["timing"]["total_ms"] += 1
    (run / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert whole_quality(run, dataset) == (metrics, parity)


@pytest.mark.parametrize("corruption", ["synset", "name", "nan", "timing", "duplicate", "missing"])
def test_verifier_rejects_corrupted_samples(completed_run: tuple, corruption: str) -> None:
    run, dataset, rows = completed_run
    if corruption == "synset":
        rows[0]["target_synset"] = "wrong"
    elif corruption == "name":
        rows[0]["predictions"][0]["name"] = "wrong"
    elif corruption == "nan":
        rows[0]["predictions"][0]["score"] = float("nan")
    elif corruption == "timing":
        rows[0]["timing"]["total_ms"] = 100
    elif corruption == "duplicate":
        rows.append(rows[0])
    else:
        rows.pop()
    (run / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError):
        whole_quality(run, dataset)


def test_two_repeat_verification_preserves_original_protocol(
    completed_run: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import verify_classification as verifier

    template, dataset, rows = completed_run
    root = template / "experiment"
    root.mkdir()
    models = ["resnet50", "efficientnet_v2_s", "convnext_tiny", "vit_b_16"]
    config = {"repeats": 3, "precision": "fp32", "batch_size": 1}
    experiment = {
        "model_ids": models,
        "config": config,
        "source_hash": object_hash({}),
        "dataset_hash": object_hash(dataset),
        "image_id": "test-container",
        "model_hashes": {m: object_hash({"id": m}) for m in models},
    }
    dataset["mode"] = "whole_image"
    experiment["dataset_hash"] = object_hash(dataset)
    write_json(root / "experiment.json", experiment)
    with tarfile.open(root / "source_snapshot.tar.gz", "w:gz"):
        pass
    monkeypatch.setattr(verifier, "verify_model", lambda model: {"id": model})
    for model in models:
        for repeat in [1, 2]:
            run = root / f"{model}_repeat{repeat}"
            run.mkdir()
            samples = copy.deepcopy(rows)
            for row in samples:
                row["model_id"] = model
            (run / "samples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in samples))
            write_json(run / "metrics.json", read_json(template / "metrics.json"))
            write_json(run / "dataset_manifest.json", dataset)
            write_json(run / "model_manifest.json", {"id": model})
            write_json(
                run / "environment.json",
                {"source_files": {}, "container_image_id": "test-container"},
            )
            write_json(
                run / "run.json",
                {
                    "status": "complete",
                    "shortened": False,
                    "config": config,
                    "repeat": repeat - 1,
                    "model_id": model,
                    "model_version_id": "test-checkpoint",
                    "samples_sha256": sha256(run / "samples.jsonl"),
                    "processed": 3,
                    "attempts": 1,
                },
            )
            (run / "errors.jsonl").touch()
    extra = root / "resnet50_repeat3/run.json"
    write_json(extra, {"status": "complete"})
    output = template / "verification.json"
    verifier.verify(root, output, repeats=2)
    result = read_json(output)
    assert result["reported_repeats"] == 2
    assert result["original_config_repeats"] == 3
    assert all(len(m["runs"]) == 2 for m in result["models"].values())
    assert result["excluded_completed_runs"] == [
        {"run": "resnet50_repeat3", "run_sha256": sha256(extra)}
    ]
    assert read_json(root / "experiment.json") == experiment
    with pytest.raises(ValueError, match="Invalid repeat selection"):
        verifier.verify(root, output, repeats=4)


def test_report_selects_equal_first_two_repeats_and_excludes_third(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from classification import report
    from classification.common import registry

    experiment = {"config": {"repeats": 3}, "source_hash": "test-source"}
    write_json(tmp_path / "experiment.json", experiment)
    for spec in registry()["models"]:
        for repeat, total in [(1, 1.0), (2, 3.0), (3, 100.0)]:
            run = tmp_path / f"{spec['id']}_repeat{repeat}"
            run.mkdir()
            write_json(run / "samples.jsonl", {"test_total_ms": total})
            write_json(
                run / "run.json",
                {
                    "status": "complete",
                    "shortened": False,
                    "repeat": repeat - 1,
                    "samples_sha256": sha256(run / "samples.jsonl"),
                    "parameters": 1,
                    "cold_load_ms": 1.0,
                    "config": {"device": "cpu", "precision": "fp32"},
                    "resources": {
                        "rss_peak_bytes": 100,
                        "cuda_allocated_peak_bytes": 0,
                        "cuda_reserved_peak_bytes": 0,
                    },
                },
            )
            write_json(
                run / "metrics.json",
                {
                    "n_images": 1,
                    "top1": 1.0,
                    "top5": 1.0,
                    "macro_f1": 0.001,
                    "per_class": [],
                    "confusion_sparse": [],
                },
            )
            write_json(run / "dataset_manifest.json", {"split": "validation"})
            write_json(
                run / "environment.json",
                {
                    "git_commit": "test",
                    "dirty": False,
                    "container_image_id": "test",
                },
            )
            write_json(
                run / "model_manifest.json",
                {
                    "files": [{"path": spec["weight_file"], "size": 1, "sha256": "test"}],
                },
            )
            write_json(run / "native_config.json", {})
            (run / "write_timings.jsonl").write_text('{"image_id":"test","write_ms":1.0}\n')

    def load(path: Path) -> list[dict]:
        total = read_json(path / "samples.jsonl")["test_total_ms"]
        return [
            {
                "timing": {
                    "total_ms": total,
                    "preprocess_ms": 0.0,
                    "inference_ms": total,
                    "postprocess_ms": 0.0,
                    "read_ms": 0.0,
                }
            }
        ]

    monkeypatch.setattr(report, "load_samples", load)
    output = tmp_path / "report"
    report.build_report(tmp_path, output, repeats=2)
    with (output / "classification.csv").open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert all(r["completed_repeats"] == "2" and r["status"] == "complete" for r in rows)
    assert all(float(r["p50_ms"]) == 2.0 for r in rows)
    selection = read_json(output / "classification_report_manifest.json")["repeat_selection"]
    assert selection["original_config_repeats"] == 3
    assert len(selection["excluded_runs"]) == 4
    assert read_json(tmp_path / "experiment.json") == experiment
    with pytest.raises(ValueError, match="Invalid report repeat selection"):
        report.build_report(tmp_path, output, repeats=4)
