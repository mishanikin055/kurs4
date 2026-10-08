from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from detection.common import object_hash, read_json, sha256, write_json


def load_samples(run: Path) -> list[dict[str, Any]]:
    meta = read_json(run / "run.json")
    dataset = read_json(run / "dataset_manifest.json")
    expected = {r["image_id"]: r for r in dataset["images"]}
    records = []
    seen = set()
    with (run / "samples.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            image_id = row["image_id"]
            if image_id not in expected or image_id in seen or row["model_id"] != meta["model_id"]:
                raise ValueError("Duplicate, unexpected image or incorrect model in predictions")
            source = expected[image_id]
            if row["target_index"] != source.get("target_index"):
                raise ValueError("Prediction reference label differs from frozen dataset")
            top = row["predictions"]
            indices = [p["index"] for p in top]
            if len(top) != 5 or len(set(indices)) != 5 or any(not 0 <= i < 1000 for i in indices):
                raise ValueError("Expected five unique ImageNet class indices")
            scores = [p["score"] for p in top]
            if any(not 0 <= s <= 1 for s in scores) or scores != sorted(scores, reverse=True):
                raise ValueError("Invalid top-5 probabilities")
            seen.add(image_id)
            records.append(row)
    if seen != set(expected):
        raise ValueError("Incomplete predictions; quality cannot be evaluated")
    return sorted(records, key=lambda r: r["image_id"])


def quality(records: list[dict[str, Any]], n_classes: int = 1000) -> dict[str, Any]:
    if not records or any(r["target_index"] is None for r in records):
        raise ValueError("Quality requires ground truth; smoke is not an accuracy evaluation")
    truth = np.array([r["target_index"] for r in records])
    predictions = np.array([r["predictions"][0]["index"] for r in records])
    if (truth < 0).any() or (truth >= n_classes).any():
        raise ValueError("Invalid target labels")
    support = np.bincount(truth, minlength=n_classes)
    predicted = np.bincount(predictions, minlength=n_classes)
    tp = np.bincount(truth[truth == predictions], minlength=n_classes)
    denom = support + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros(n_classes, dtype=float), where=denom > 0)
    top5 = [r["target_index"] in [p["index"] for p in r["predictions"]] for r in records]
    confusion = Counter(zip(truth.tolist(), predictions.tolist()))
    return {
        "schema_version": "1.0",
        "n_images": len(records),
        "n_classes": n_classes,
        "top1": float(np.mean(truth == predictions)),
        "top5": float(np.mean(top5)),
        "macro_f1": float(f1.mean()),
        "per_class": [
            {
                "index": i,
                "support": int(support[i]),
                "correct": int(tp[i]),
                "accuracy": float(tp[i] / support[i]) if support[i] else None,
                "f1": float(f1[i]),
            }
            for i in range(n_classes)
        ],
        "confusion_sparse": [
            {"target": t, "predicted": p, "count": n} for (t, p), n in sorted(confusion.items())
        ],
    }


def evaluate_run(run: Path) -> None:
    meta = read_json(run / "run.json")
    if meta.get("smoke"):
        raise ValueError("Smoke cannot be evaluated as ImageNet")
    records = load_samples(run)
    write_json(run / "metrics.json", quality(records))
    for name, key in [("predictions.jsonl", "predictions"), ("timings.jsonl", "timing")]:
        with (run / name).open("w") as handle:
            for record in records:
                handle.write(
                    json.dumps(
                        {"schema_version": "1.0", "image_id": record["image_id"], key: record[key]},
                        allow_nan=False,
                    )
                    + "\n"
                )
    meta.update(
        status="complete", processed=len(records), samples_sha256=sha256(run / "samples.jsonl")
    )
    write_json(run / "run.json", meta)


def bootstrap_runs(output: Path, samples: int, seed: int) -> None:
    experiment = read_json(output / "experiment.json")
    config = experiment["config"]
    selected = experiment["model_ids"]
    if len(selected) != 4 or samples < 1:
        raise ValueError("Bootstrap requires exactly four models and a positive sample count")
    runs = {m: output / f"{m}_repeat1" for m in selected}
    identity = {
        "experiment_hash": object_hash(experiment),
        "samples": samples,
        "seed": seed,
        "source_sha256": sha256(Path(__file__)),
        "predictions_sha256": {m: sha256(p / "samples.jsonl") for m, p in runs.items()},
    }
    if (output / "bootstrap.json").exists():
        if read_json(output / "bootstrap.json")["identity"] != identity:
            raise ValueError("Bootstrap inputs changed; use a new experiment")
        return
    records = {}
    for model, run in runs.items():
        meta = read_json(run / "run.json")
        if meta["status"] != "complete" or meta.get("smoke") or meta["shortened"]:
            raise ValueError("Bootstrap requires complete full-size evaluation runs")
        if meta["samples_sha256"] != identity["predictions_sha256"][model]:
            raise ValueError("Predictions changed since evaluation")
        records[model] = load_samples(run)
    first = records[selected[0]]
    ids = [r["image_id"] for r in first]
    truth = np.array([r["target_index"] for r in first])
    if any([r["image_id"] for r in rows] != ids for rows in records.values()):
        raise ValueError("Paired models must have identical image order")
    arrays = {}
    for model, rows in records.items():
        if [r["target_index"] for r in rows] != truth.tolist():
            raise ValueError("Paired ground-truth labels differ")
        arrays[model] = np.array(
            [
                [
                    r["target_index"] == r["predictions"][0]["index"],
                    r["target_index"] in [p["index"] for p in r["predictions"]],
                ]
                for r in rows
            ],
            dtype=float,
        )
    # Balanced class design: paired, stratified bootstrap resamples within each class.
    groups = [np.flatnonzero(truth == target) for target in sorted(set(truth.tolist()))]
    rng = np.random.default_rng(seed)
    raw: dict[str, list] = {model: [] for model in selected}
    for _ in range(samples):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        for model in selected:
            raw[model].append(arrays[model][indices].mean(0).tolist())
    values = {m: np.array(v) for m, v in raw.items()}
    pairs = {
        f"{a} - {b}": {
            metric: np.percentile(values[a][:, j] - values[b][:, j], [2.5, 97.5]).tolist()
            for j, metric in enumerate(["top1", "top5"])
        }
        for a, b in combinations(selected, 2)
    }
    write_json(
        output / "bootstrap.json",
        {
            "schema_version": "1.0",
            "identity": identity,
            "method": "paired class-stratified image bootstrap, percentile CI95; quality from repeat1",
            "config_seed": config["seed"],
            "samples": samples,
            "n_images": len(ids),
            "raw_top1_top5": raw,
            "ci95": {
                m: {
                    metric: np.percentile(values[m][:, j], [2.5, 97.5]).tolist()
                    for j, metric in enumerate(["top1", "top5"])
                }
                for m in selected
            },
            "paired_difference_ci95": pairs,
        },
    )
