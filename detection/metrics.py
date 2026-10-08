"""COCO evaluation and paired, dataset-level AP bootstrap."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path
from typing import Any

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from detection.common import read_json, write_json

METRIC_NAMES = [
    "mAP",
    "AP50",
    "AP75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR1",
    "AR10",
    "AR100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def empty_detections(gt: COCO) -> COCO:
    dt = COCO()
    dt.dataset = {
        "images": gt.dataset["images"],
        "categories": gt.dataset["categories"],
        "annotations": [],
    }
    dt.createIndex()
    return dt


def evaluate_coco(
    gt: COCO,
    predictions: list[dict[str, Any]],
    ids: list[int],
    threshold: float = 0.5,
    iou: float = 0.5,
) -> dict[str, Any]:
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(predictions) if predictions else empty_detections(gt)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = sorted(ids)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
        quality = {k: (float(v) if v >= 0 else None) for k, v in zip(METRIC_NAMES, ev.stats)}
        per_class = {}
        for index, category in enumerate(ev.params.catIds):
            p = ev.eval["precision"][:, :, index, 0, -1]
            valid = p[p >= 0]
            per_class[str(category)] = {
                "name": gt.cats[category]["name"],
                "AP": float(valid.mean()) if valid.size else None,
            }
        # COCO matching handles crowds and ignore flags; no bespoke IoU matching.
        fixed_dt = [p for p in predictions if p["score"] >= threshold]
        fixed = COCOeval(gt, gt.loadRes(fixed_dt) if fixed_dt else empty_detections(gt), "bbox")
        fixed.params.imgIds = sorted(ids)
        fixed.params.iouThrs = np.array([iou])
        fixed.params.areaRng = [[0, 1e10]]
        fixed.params.areaRngLbl = ["all"]
        fixed.params.maxDets = [100]
        fixed.evaluate()
    tp = fp = fn = 0
    class_counts: dict[int, list[int]] = {}
    for row in fixed.evalImgs:
        if row is None:
            continue
        match = row["dtMatches"][0] > 0
        ignore = row["dtIgnore"][0]
        values = [
            int(np.sum(match & ~ignore)),
            int(np.sum(~match & ~ignore)),
            int(np.sum((row["gtMatches"][0] == 0) & np.logical_not(row["gtIgnore"].astype(bool)))),
        ]
        counts = class_counts.setdefault(row["category_id"], [0, 0, 0])
        for i, val in enumerate(values):
            counts[i] += val
        tp += values[0]
        fp += values[1]
        fn += values[2]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    quality.update(
        precision=precision,
        recall=recall,
        F1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        TP=tp,
        FP=fp,
        FN=fn,
        operating_threshold=threshold,
        operating_iou=iou,
        operating_max_detections=100,
    )
    for cat, counts in class_counts.items():
        per_class[str(cat)].update(zip(["TP", "FP", "FN"], counts))
    return {"quality": quality, "per_class": per_class}


def load_predictions(run: Path) -> tuple[list[int], list[dict[str, Any]], list[dict[str, Any]]]:
    import json

    ids, predictions, timings = [], [], []
    with (run / "samples.jsonl").open() as stream:
        for line in stream:
            row = json.loads(line)
            ids.append(row["image_id"])
            predictions.extend(
                {"image_id": row["image_id"], **{k: p[k] for k in ("category_id", "bbox", "score")}}
                for p in row["predictions"]
            )
            timings.append({"image_id": row["image_id"], **row["timing"]})
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate predictions for an image")
    write_path = run / "write_timings.jsonl"
    writes = {}
    if write_path.exists():
        with write_path.open() as stream:
            for line in stream:
                if line.endswith("\n"):
                    row = json.loads(line)
                    writes[row["image_id"]] = row["write_ms"]
    for timing in timings:
        timing["write_ms"] = writes.get(timing["image_id"])
    return ids, predictions, timings


def evaluate_run(run: Path) -> None:
    import json

    meta = read_json(run / "run.json")
    if meta["status"] not in {"inference_complete", "complete"}:
        raise RuntimeError("Quality metrics require a successfully finished inference run")
    ids, predictions, timings = load_predictions(run)
    expected = read_json(run / "dataset_manifest.json")["images"]
    if set(ids) != {r["image_id"] for r in expected}:
        raise ValueError("Predictions do not cover the full selected manifest")
    manifest = read_json(run / "dataset_manifest.json")
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(Path(__file__).resolve().parent.parent / manifest["annotations"]))
    results = evaluate_coco(
        gt, predictions, ids, meta["config"]["operating_threshold"], meta["config"]["operating_iou"]
    )
    latency = np.array([t["total_ms"] for t in timings])
    results.update(
        schema_version="1.0",
        run_id=run.name,
        n_images=len(ids),
        shortened=meta["shortened"],
        latency={
            "p50_ms": float(np.percentile(latency, 50)),
            "p95_ms": float(np.percentile(latency, 95)),
            "images_per_second": float(1000 / latency.mean()),
            **{
                f"{stage}_mean_ms": float(
                    np.mean([t[f"{stage}_ms"] for t in timings if t[f"{stage}_ms"] is not None])
                )
                for stage in ("preprocess", "forward", "postprocess", "read", "write")
            },
        },
        resources=meta["resources"],
        cold_load_ms=meta["cold_load_ms"],
        parameters=meta["parameters"],
    )
    write_json(run / "metrics.json", results)
    write_json(run / "predictions_coco.json", predictions)
    with (run / "timings.jsonl").open("w") as stream:
        for t in timings:
            stream.write(json.dumps(t) + "\n")
    with (run / "predictions.jsonl").open("w") as output, (run / "samples.jsonl").open() as source:
        for line in source:
            row = json.loads(line)
            output.write(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "image_id": row["image_id"],
                        "predictions": row["predictions"],
                    }
                )
                + "\n"
            )
    meta["status"] = "complete"
    write_json(run / "run.json", meta)
    print(
        run.name, "mAP", results["quality"]["mAP"], "p50", results["latency"]["p50_ms"], flush=True
    )


def bootstrap_evaluator(gt: COCO, predictions: list[dict[str, Any]], ids: list[int]) -> COCOeval:
    """Cache per-image matches; bootstrap changes only dataset-level accumulation."""
    with contextlib.redirect_stdout(io.StringIO()):
        dt = gt.loadRes(predictions) if predictions else empty_detections(gt)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = sorted(ids)
        ev.params.areaRng = [[0, 1e10]]
        ev.params.areaRngLbl = ["all"]
        ev.params.maxDets = [100]
        ev.evaluate()
    return ev


def cached_bootstrap_ap(ev: COCOeval, cached: list[Any], indices: np.ndarray) -> float:
    """Duplicate match records by sampled image position, then use official COCO AP.

    Matching is independent per image/category. Duplicate IDs are represented by
    distinct positions; no prediction scores or per-image APs are averaged.
    The sampled order preserves COCO's stable sorting for equal-score ties.
    """
    count = len(indices)
    ev.evalImgs = [
        cached[category * count + int(index)]
        for category in range(len(ev.params.catIds))
        for index in indices
    ]
    ev.params.imgIds = list(range(count))
    ev._paramsEval.imgIds = list(range(count))
    with contextlib.redirect_stdout(io.StringIO()):
        ev.accumulate()
    precision = ev.eval["precision"]
    valid = precision[precision >= 0]
    if not valid.size:
        raise ValueError("Undefined AP in a bootstrap draw")
    return float(valid.mean())


def bootstrap_runs(runs: list[Path], samples: int, seed: int, output: Path) -> None:
    """Paired image sampling, recomputing dataset-level COCO AP.

    Process one model's predictions at a time to fit the 6 GiB container.
    Every model receives the identical precomputed matrix of sampled image IDs.
    """
    runs = [r for r in runs if read_json(r / "run.json")["repeat"] == 0]
    model_ids = [read_json(r / "run.json")["model_id"] for r in runs]
    if len(runs) != 4 or len(set(model_ids)) != 4:
        raise ValueError("Paired bootstrap requires four distinct completed first-repeat runs")
    from detection.common import ROOT, object_hash, sha256

    dataset = read_json(runs[0] / "dataset_manifest.json")
    if any(
        object_hash(read_json(r / "dataset_manifest.json")) != object_hash(dataset) for r in runs
    ):
        raise ValueError("Bootstrap dataset manifests differ")
    ids = sorted(r["image_id"] for r in dataset["images"])
    gt_source = read_json(ROOT / dataset["annotations"])
    rng = np.random.default_rng(seed)
    draws = rng.choice(np.arange(len(ids), dtype=np.int64), size=(samples, len(ids)), replace=True)
    identity = {
        "samples": samples,
        "seed": seed,
        "dataset_hash": object_hash(dataset),
        "annotation_hash": sha256(ROOT / dataset["annotations"]),
        "metrics_source_hash": sha256(Path(__file__)),
        "prediction_hashes": {
            read_json(r / "run.json")["model_id"]: sha256(r / "samples.jsonl") for r in runs
        },
    }
    if (
        dataset.get("annotations_sha256")
        and identity["annotation_hash"] != dataset["annotations_sha256"]
    ):
        raise ValueError("Bootstrap annotation hash changed")
    checkpoint = output.with_suffix(".progress.json")
    progress = (
        read_json(checkpoint) if checkpoint.exists() else {"identity": identity, "raw_mAP": {}}
    )
    if progress["identity"] != identity:
        raise ValueError("Bootstrap checkpoint differs from config/data/predictions/source")
    values = []
    for model_index, run in enumerate(runs):
        model = model_ids[model_index]
        model_values = progress["raw_mAP"].setdefault(model, [])
        if len(model_values) == samples:
            values.append(model_values)
            continue
        run_ids, raw_predictions, _ = load_predictions(run)
        if sorted(run_ids) != ids:
            raise ValueError("Bootstrap prediction image sets differ")
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO()
            gt.dataset = gt_source
            gt.createIndex()
        ev = bootstrap_evaluator(gt, raw_predictions, ids)
        cached = ev.evalImgs
        del raw_predictions
        for iteration in range(len(model_values), samples):
            ap = cached_bootstrap_ap(ev, cached, draws[iteration])
            model_values.append(ap)
            if (iteration + 1) % 10 == 0 or iteration == 0 or iteration + 1 == samples:
                write_json(checkpoint, progress)
                print(f"Bootstrap {model}: {iteration + 1}/{samples}", flush=True)
        values.append(model_values)
        del ev, cached, gt
        import gc

        gc.collect()
    intervals = {name: np.percentile(v, [2.5, 97.5]).tolist() for name, v in zip(model_ids, values)}
    differences = {
        f"{model_ids[i]} - {model_ids[j]}": np.percentile(
            np.array(values[i]) - np.array(values[j]), [2.5, 97.5]
        ).tolist()
        for i in range(4)
        for j in range(i + 1, 4)
    }
    write_json(
        output,
        {
            "schema_version": "1.0",
            "samples": samples,
            "seed": seed,
            "shortened": any(read_json(r / "run.json")["shortened"] for r in runs),
            "method": "paired image bootstrap; dataset-level COCO AP; percentile 95%",
            "implementation": "cached per-image COCO matches + official COCOeval.accumulate",
            "identity": identity,
            "run_ids": [r.name for r in runs],
            "mAP_ci95": intervals,
            "paired_mAP_difference_ci95": differences,
            "raw_mAP": dict(zip(model_ids, values)),
        },
    )
