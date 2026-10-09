"""Corruption checks for the independent completed-artifact verifier."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from classification.common import labels
from classification.metrics import quality
from detection.common import write_json
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
