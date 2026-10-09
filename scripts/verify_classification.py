"""Verify completed classification artifacts without loading models or calculating CI."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classification.common import labels, verify_model
from detection.common import object_hash, read_json, sha256, write_json


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def source_files(snapshot: Path) -> dict[str, str]:
    with tarfile.open(snapshot) as archive:
        result = {}
        for member in archive.getmembers():
            if member.isfile():
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError(f"Unreadable snapshot member: {member.name}")
                require(member.name not in result, "Duplicate snapshot source")
                result[member.name] = hashlib.sha256(handle.read()).hexdigest()
        return result


def check_timing(row: dict[str, Any]) -> None:
    timing = row["timing"]
    require(
        all(math.isfinite(t) and t >= 0 for t in timing.values()),
        "Invalid sample timing",
    )
    total = sum(timing[f"{stage}_ms"] for stage in ["preprocess", "inference", "postprocess"])
    require(math.isclose(total, timing["total_ms"], abs_tol=1e-6), "Timing stages differ")


def whole_quality(run: Path, dataset: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Independently recount whole-image quality while keeping memory bounded."""
    expected = {r["image_id"]: r for r in dataset["images"]}
    classes = labels()
    seen: set[str] = set()
    support: Counter[int] = Counter()
    predicted: Counter[int] = Counter()
    correct: Counter[int] = Counter()
    confusion: Counter[tuple[int, int]] = Counter()
    top5_correct = 0
    parity = hashlib.sha256()
    meta = read_json(run / "run.json")
    with (run / "samples.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            key = row["image_id"]
            require(key in expected and key not in seen, "Duplicate or unexpected image_id")
            seen.add(key)
            source = expected[key]
            target = source["target_index"]
            require(
                row["target_index"] == target and row["target_synset"] == source["target_synset"],
                "Sample ground truth differs",
            )
            require(
                row["model_id"] == meta["model_id"]
                and row["model_version_id"] == meta["model_version_id"],
                "Sample model differs",
            )
            top = row["predictions"]
            indices = [p["index"] for p in top]
            scores = [p["score"] for p in top]
            require(
                len(top) == len(set(indices)) == 5 and all(0 <= i < 1000 for i in indices),
                "Invalid top5 indices",
            )
            require(
                all(math.isfinite(s) and 0 <= s <= 1 for s in scores)
                and scores == sorted(scores, reverse=True),
                "Invalid top5 probabilities",
            )
            require(
                all(
                    p["synset"] == classes[p["index"]]["synset"]
                    and p["name"] == classes[p["index"]]["name"]
                    for p in top
                ),
                "Sample label order differs",
            )
            check_timing(row)
            support[target] += 1
            predicted[indices[0]] += 1
            correct[target] += target == indices[0]
            top5_correct += target in indices
            confusion[target, indices[0]] += 1
            parity.update((object_hash([key, top]) + "\n").encode())
    require(seen == set(expected), "Incomplete whole-image results")
    n = len(seen)
    per_class = [
        {
            "index": i,
            "support": support[i],
            "correct": correct[i],
            "accuracy": correct[i] / support[i] if support[i] else None,
            "f1": 2 * correct[i] / (support[i] + predicted[i])
            if support[i] + predicted[i]
            else 0.0,
        }
        for i in range(1000)
    ]
    metrics = read_json(run / "metrics.json")
    require(metrics["n_images"] == n and metrics["n_classes"] == 1000, "Quality N differs")
    require(metrics["top1"] == sum(correct.values()) / n, "Top1 recount differs")
    require(metrics["top5"] == top5_correct / n, "Top5 recount differs")
    require(metrics["per_class"] == per_class, "Per-class recount differs")
    require(
        math.isclose(metrics["macro_f1"], sum(r["f1"] for r in per_class) / 1000, abs_tol=1e-14),
        "Macro-F1 recount differs",
    )
    require(
        metrics["confusion_sparse"]
        == [
            {"target": t, "predicted": p, "count": count}
            for (t, p), count in sorted(confusion.items())
        ],
        "Confusion recount differs",
    )
    return {
        "n_images": n,
        "correct_top1": sum(correct.values()),
        "correct_top5": top5_correct,
        "top1": metrics["top1"],
        "top5": metrics["top5"],
        "macro_f1": metrics["macro_f1"],
    }, parity.hexdigest()


def crop_quality(run: Path) -> tuple[dict[str, Any], str]:
    from classification.crops.metrics import load, quality

    rows, dataset = load(run)
    recomputed = quality(rows, dataset)
    require(read_json(run / "metrics.json") == recomputed, "Crop quality recount differs")
    parity = hashlib.sha256()
    for row in rows:
        check_timing(row)
        require(
            row["effective_category_id"] == row["detector_category_id"],
            "Original detector label was changed",
        )
        parity.update(
            (object_hash([row["sample_id"], row["predictions"], row["status"]]) + "\n").encode()
        )
    return {k: v for k, v in recomputed.items() if k != "per_class"}, parity.hexdigest()


def verify(runs: Path, output: Path, repeats: int | None = None) -> None:
    experiment = read_json(runs / "experiment.json")
    require(
        experiment["model_ids"] == ["resnet50", "efficientnet_v2_s", "convnext_tiny", "vit_b_16"]
        and experiment["config"]["repeats"] == 3
        and experiment["config"]["precision"] == "fp32"
        and experiment["config"]["batch_size"] == 1,
        "Expected the frozen four-model, three-repeat FP32/batch1 protocol",
    )
    sources = source_files(runs / "source_snapshot.tar.gz")
    require(object_hash(sources) == experiment["source_hash"], "Source snapshot differs")
    require(not experiment.get("shortened") and not experiment.get("smoke"), "Not a full run")
    require(not (runs / "bootstrap.json").exists(), "Unexpected CI artifact in this experiment")
    selected_repeats = repeats if repeats is not None else experiment["config"]["repeats"]
    require(1 <= selected_repeats <= experiment["config"]["repeats"], "Invalid repeat selection")
    summary = {
        "schema_version": "1.0",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "verifier_sha256": sha256(Path(__file__)),
        "runs_dir": str(runs),
        "experiment_sha256": sha256(runs / "experiment.json"),
        "source_snapshot_sha256": sha256(runs / "source_snapshot.tar.gz"),
        "dataset_hash": experiment["dataset_hash"],
        "source_hash": experiment["source_hash"],
        "container_image_id": experiment["image_id"],
        "confidence_intervals": "not calculated",
        "reported_repeats": selected_repeats,
        "original_config_repeats": experiment["config"]["repeats"],
        "excluded_completed_runs": [],
        "models": {},
    }
    for model in experiment["model_ids"]:
        require(
            object_hash(verify_model(model)) == experiment["model_hashes"][model],
            "Local model manifest differs",
        )
        verified = []
        parities = []
        for repeat in range(selected_repeats):
            run = runs / f"{model}_repeat{repeat + 1}"
            meta = read_json(run / "run.json")
            require(meta["status"] == "complete" and not meta["shortened"], "Incomplete run")
            require(meta["config"] == experiment["config"], "Run config differs")
            require(meta["repeat"] == repeat and meta["model_id"] == model, "Run identity differs")
            require(sha256(run / "samples.jsonl") == meta["samples_sha256"], "Samples SHA differs")
            dataset = read_json(run / "dataset_manifest.json")
            require(object_hash(dataset) == experiment["dataset_hash"], "Dataset identity differs")
            require(
                object_hash(read_json(run / "model_manifest.json"))
                == experiment["model_hashes"][model],
                "Run model manifest differs",
            )
            env = read_json(run / "environment.json")
            require(env["source_files"] == sources, "Run source files differ from snapshot")
            require(env["container_image_id"] == experiment["image_id"], "Container ID differs")
            metrics, parity = (
                whole_quality(run, dataset)
                if dataset["mode"] == "whole_image"
                else crop_quality(run)
            )
            n = metrics.get("n_images", metrics.get("n_crops"))
            require(meta["processed"] == n, "Processed count differs")
            require((run / "errors.jsonl").stat().st_size == 0, "Run contains errors")
            parities.append(parity)
            verified.append(
                {
                    "run": run.name,
                    "status": meta["status"],
                    "attempts": meta["attempts"],
                    "errors_bytes": 0,
                    **metrics,
                    "prediction_content_sha256": parity,
                    "artifacts": {
                        p.name: sha256(p)
                        for p in sorted(run.iterdir())
                        if p.is_file() and p.suffix in {".json", ".jsonl", ".log"}
                    },
                }
            )
            print(f"Verified {run.name}: {n}", flush=True)
        require(len(set(parities)) == 1, f"Predictions differ between repeats: {model}")
        summary["models"][model] = {
            "predictions_identical_between_repeats": True,
            "processed_total": sum(r.get("n_images", r.get("n_crops", 0)) for r in verified),
            "runs": verified,
        }
        for repeat in range(selected_repeats, experiment["config"]["repeats"]):
            path = runs / f"{model}_repeat{repeat + 1}" / "run.json"
            if path.exists() and read_json(path)["status"] == "complete":
                summary["excluded_completed_runs"].append(
                    {"run": path.parent.name, "run_sha256": sha256(path)}
                )
    write_json(output, summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Проверка полных результатов классификации без CI")
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repeats", type=int, help="Проверить первые N повторов без изменения identity"
    )
    args = parser.parse_args()
    verify(args.runs_dir, args.output, args.repeats)


if __name__ == "__main__":
    main()
