"""Coarse COCO quality, hypothetical changes, and paired scene bootstrap."""

from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

from classification.common import labels
from classification.crops.runner import interpret
from detection.common import object_hash, read_json, sha256, write_json


def load(run: Path) -> tuple[list[dict], dict]:
    meta = read_json(run / "run.json")
    dataset = read_json(run / "dataset_manifest.json")
    expected = {r["sample_id"]: r for r in dataset["samples"]}
    result = []
    seen = set()
    classes = labels()
    for line in (run / "samples.jsonl").read_text().splitlines():
        row = json.loads(line)
        key = row["sample_id"]
        if key not in expected or key in seen or row["source_hash"] != object_hash(expected[key]):
            raise ValueError("Prediction crop identity changed")
        if (
            row["model_id"] != meta["model_id"]
            or row["model_version_id"] != meta["model_version_id"]
        ):
            raise ValueError("Prediction model differs")
        if any(row[k] != v for k, v in expected[key].items()):
            raise ValueError("Prediction ground truth differs")
        top = row["predictions"]
        indices = [p["index"] for p in top]
        scores = [p["score"] for p in top]
        if (
            len(top) != 5
            or len(set(indices)) != 5
            or any(not 0 <= i < 1000 for i in indices)
            or any(not 0 <= s <= 1 for s in scores)
            or scores != sorted(scores, reverse=True)
        ):
            raise ValueError("Invalid top5 predictions")
        if any(
            p["synset"] != classes[p["index"]]["synset"] or p["name"] != classes[p["index"]]["name"]
            for p in top
        ):
            raise ValueError("Prediction synset differs")
        interpretation = interpret(top, row["detector_category_id"], meta["config"])
        if any(row[k] != v for k, v in interpretation.items()):
            raise ValueError("Prediction mapping/decision differs")
        seen.add(key)
        result.append(row)
    if seen != set(expected):
        raise ValueError("Incomplete crops; quality not evaluated")
    return sorted(result, key=lambda r: r["sample_id"]), dataset


def flags(row: dict, supported: set[int]) -> list[int]:
    eligible = row["target_category_id"] in supported and not row["ignored_crowd"]
    truth = row["target_category_id"]
    correct = eligible and row["mapped_top1"] == truth
    top5 = eligible and truth in {p["coco_category_id"] for p in row["predictions"]}
    baseline = eligible and row["detector_category_id"] == truth
    proposed = row["mapped_top1"] if row["accepted"] else row["detector_category_id"]
    gated = eligible and proposed == truth
    # Counts retain all proposals, including unsupported and unmatched objects.
    return [
        int(eligible),
        int(correct),
        int(top5),
        int(baseline),
        int(gated),
        int(eligible and not baseline and gated),
        int(eligible and baseline and not gated),
        int(not row["ignored_crowd"]),
        int(not row["ignored_crowd"] and row["accepted"]),
        int(eligible and row["accepted"] and correct),
    ]


def quality(records: list[dict], dataset: dict) -> dict:
    supported = set(dataset["supported_categories"])
    counts = np.sum([flags(r, supported) for r in records], axis=0).astype(int)
    n, correct, top5, baseline, gated, helpful, harmful, proposals, accepted, accepted_correct = (
        counts.tolist()
    )
    chosen = [r for r in records if r["target_category_id"] in supported and not r["ignored_crowd"]]
    per_class = []
    for category in sorted(supported):
        targets = [r for r in chosen if r["target_category_id"] == category]
        tp = sum(r["mapped_top1"] == category for r in targets)
        predicted = sum(r["mapped_top1"] == category for r in chosen)
        denominator = len(targets) + predicted
        per_class.append(
            {
                "category_id": category,
                "support": len(targets),
                "correct": tp,
                "accuracy": tp / len(targets) if targets else None,
                "f1": 2 * tp / denominator if denominator else 0.0,
            }
        )
    present = [r for r in per_class if r["support"]]
    total_gt = sum(int(v) for v in dataset["supported_gt_counts"].values())
    result = {
        "schema_version": "1.0",
        "mode": dataset["mode"],
        "n_scenes": dataset["n_scenes"],
        "n_crops": len(records),
        "n_eligible_crops": n,
        "n_supported_gt": total_gt,
        "mapping_object_coverage": dataset["mapping_object_coverage"],
        "supported_categories": len(supported),
        "observed_categories": len(present),
        "top1_coarse": correct / n if n else None,
        "top5_coarse": top5 / n if n else None,
        "macro_f1_observed": float(np.mean([r["f1"] for r in present])) if present else None,
        "acceptance_rate": accepted / proposals if proposals else None,
        "accepted_top1_accuracy": accepted_correct / sum(r["accepted"] for r in chosen)
        if any(r["accepted"] for r in chosen)
        else None,
        "status_counts": dict(Counter(r["status"] for r in records)),
        "per_class": per_class,
        "fine_grained_accuracy": None,
        "fine_grained_limitation": "COCO has no ImageNet subtype ground truth",
    }
    if dataset["mode"] == "detector_crops":
        result.update(
            detector_only_matched_accuracy=baseline / n if n else None,
            hypothetical_gated_matched_accuracy=gated / n if n else None,
            helpful_changes=helpful,
            harmful_changes=harmful,
            matched_supported_gt=n,
            missed_supported_gt=total_gt - n,
            unmatched_proposals=sum(
                r["matched_annotation_id"] is None and not r["ignored_crowd"] for r in records
            ),
            matched_unsupported_proposals=sum(
                r["matched_annotation_id"] is not None and r["target_category_id"] not in supported
                for r in records
            ),
            crowd_ignored_proposals=sum(r["ignored_crowd"] for r in records),
            detector_only_supported_recall=baseline / total_gt if total_gt else None,
            hypothetical_gated_supported_recall=gated / total_gt if total_gt else None,
            effective_label_changes=0,
        )
    return result


def evaluate(run: Path) -> None:
    rows, dataset = load(run)
    meta = read_json(run / "run.json")
    if meta["shortened"]:
        write_json(
            run / "metrics.json",
            {
                "schema_version": "1.0",
                "mode": dataset["mode"],
                "status": "smoke_only",
                "processed_crops": len(rows),
                "quality": None,
            },
        )
    else:
        write_json(run / "metrics.json", quality(rows, dataset))
    meta.update(
        status="complete", processed=len(rows), samples_sha256=sha256(run / "samples.jsonl")
    )
    write_json(run / "run.json", meta)


def scene_arrays(records: list[dict], dataset: dict) -> np.ndarray:
    ids = [r["image_id"] for r in dataset["scenes"]]
    indices = {key: i for i, key in enumerate(ids)}
    counts = np.zeros((len(ids), 10), dtype=float)
    supported = set(dataset["supported_categories"])
    for row in records:
        counts[indices[row["image_id"]]] += flags(row, supported)
    return counts


def bootstrap(output: Path, samples: int = 1000, seed: int = 42) -> None:
    experiment = read_json(output / "experiment.json")
    if experiment["shortened"] or samples < 1:
        raise ValueError("Bootstrap needs complete, full-size crops")
    models = experiment["model_ids"]
    arrays = {}
    identities = {}
    dataset_hash = None
    for model in models:
        run = output / f"{model}_repeat1"
        meta = read_json(run / "run.json")
        if (
            meta["status"] != "complete"
            or meta["shortened"]
            or meta["samples_sha256"] != sha256(run / "samples.jsonl")
        ):
            raise ValueError("Bootstrap needs completed unchanged runs")
        rows, dataset = load(run)
        current_hash = object_hash(dataset)
        if dataset_hash is not None and current_hash != dataset_hash:
            raise ValueError("Paired crop manifests differ")
        dataset_hash = current_hash
        arrays[model] = scene_arrays(rows, dataset)
        identities[model] = meta["samples_sha256"]
    identity = {
        "experiment_hash": object_hash(experiment),
        "predictions_sha256": identities,
        "samples": samples,
        "seed": seed,
        "source_sha256": sha256(Path(__file__)),
    }
    if (output / "bootstrap.json").exists():
        if read_json(output / "bootstrap.json")["identity"] != identity:
            raise ValueError("Bootstrap inputs changed")
        return
    rng = np.random.default_rng(seed)
    n_scenes = len(next(iter(arrays.values())))
    raw = {m: [] for m in models}
    for _ in range(samples):
        indices = rng.integers(0, n_scenes, n_scenes)
        for m in models:
            sums = arrays[m][indices].sum(axis=0)
            raw[m].append([float(sums[k] / sums[0]) if sums[0] else None for k in [1, 2, 4]])
    intervals = {
        m: {
            name: np.nanpercentile(np.array(raw[m], dtype=float)[:, i], [2.5, 97.5]).tolist()
            for i, name in enumerate(
                ["top1_coarse", "top5_coarse", "hypothetical_gated_matched_accuracy"]
            )
        }
        for m in models
    }
    paired = []
    for a, b in combinations(models, 2):
        differences = np.array(raw[a], dtype=float) - np.array(raw[b], dtype=float)
        for i, name in enumerate(
            ["top1_coarse", "top5_coarse", "hypothetical_gated_matched_accuracy"]
        ):
            lo, hi = np.nanpercentile(differences[:, i], [2.5, 97.5])
            paired.append(
                {
                    "model_a": a,
                    "model_b": b,
                    "metric": name,
                    "ci95_low": float(lo),
                    "ci95_high": float(hi),
                }
            )
    write_json(
        output / "bootstrap.json",
        {
            "schema_version": "1.0",
            "identity": identity,
            "unit": "COCO image; all its crops move together",
            "intervals": intervals,
            "paired_differences": paired,
            "raw_samples": raw,
        },
    )
