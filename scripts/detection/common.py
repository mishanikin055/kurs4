"""COCO data, staged timings, official metrics and a single-process-model launch."""

from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import resource
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "detection/models"
CATEGORIES = ROOT / "detection/configs/coco_categories.json"
MODEL_IDS = ("yolo26s", "rtdetr_v2_r18vd", "rf_detr_small", "fasterrcnn_resnet50_fpn_v2")
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


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def object_hash(value: Any) -> str:
    # JSON object keys become strings on disk; normalize before sorting so
    # integer-keyed counters have the same identity after serialization.
    normalized = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def verify_model(model_id: str) -> dict[str, Any]:
    manifest = read_json(MODELS / "manifest.json")["models"][model_id]
    if manifest["status"] not in {"verified", "smoke_passed"}:
        raise RuntimeError(f"Model is not verified: {model_id}")
    if manifest["spec_hash"] != object_hash(model_spec(model_id)):
        raise RuntimeError("Model registry changed; download and verify the pinned version again")
    for item in manifest["files"]:
        p = MODELS / model_id / item["path"]
        if not p.is_file() or p.stat().st_size != item["size"] or sha256(p) != item["sha256"]:
            raise RuntimeError(f"Model file checksum mismatch: {p}")
    return manifest


def to_coco_label(label: int, mapping: str) -> int:
    if mapping == "contiguous_0":
        if not 0 <= label < len(COCO_IDS):
            raise ValueError(f"Invalid continuous class index: {label}")
        return COCO_IDS[label]
    if mapping == "coco_sparse" and label in COCO_NAMES:
        return label
    raise ValueError(f"Unsupported COCO category: {label}, mapping={mapping}")


def normalize_predictions(
    boxes: Any,
    scores: Any,
    labels: Any,
    mapping: str,
    width: int,
    height: int,
    floor: float,
    max_det: int,
    *,
    exclude_coco_gaps: bool = False,
) -> list[dict[str, Any]]:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if not np.isfinite(labels).all() or (labels != np.floor(labels)).any():
        raise ValueError("Invalid prediction class indices")
    if len(boxes) != len(scores) or len(scores) != len(labels):
        raise ValueError("Prediction fields have different lengths")
    if (
        not np.isfinite(boxes).all()
        or not np.isfinite(scores).all()
        or ((scores < 0) | (scores > 1)).any()
    ):
        raise ValueError("Invalid prediction values")
    rows = []
    for index in np.argsort(-scores, kind="stable"):
        if scores[index] < floor:
            continue
        label = int(labels[index])
        # RF-DETR's pretrained head retains unused COCO slots (including 0).
        # These slots have no dataset category and must never be remapped.
        if exclude_coco_gaps and mapping == "coco_sparse" and 0 <= label <= 90:
            if label not in COCO_NAMES:
                continue
        category = to_coco_label(label, mapping)
        x1, y1, x2, y2 = boxes[index]
        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, height)
        if x2 <= x1 or y2 <= y1:
            continue
        rows.append(
            {
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(scores[index]),
                "category_id": category,
                "original_label": label,
            }
        )
        if len(rows) == max_det:
            break
    return rows


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


CATEGORY_LIST = read_json(CATEGORIES)
COCO_IDS = [c["id"] for c in CATEGORY_LIST]
COCO_NAMES = {c["id"]: c["name"] for c in CATEGORY_LIST}


def model_spec(model_id: str) -> dict[str, Any]:
    return next(
        m
        for m in read_json(ROOT / "detection/configs/models.json")["models"]
        if m["id"] == model_id
    )


@contextlib.contextmanager
def file_lock(path: Path) -> Iterator[int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Inference lock already held: {path}") from error
        # Close, do not explicitly unlock: an orphan child retains this flock.
        yield handle.fileno()


def source_files() -> dict[str, str]:
    paths = list((ROOT / "scripts/detection").glob("*.py")) + [
        ROOT / "detection/requirements.lock.txt",
        ROOT / "detection/requirements.in",
        ROOT / "detection/Dockerfile",
        ROOT / "compose.yaml",
        ROOT / "compose.gpu.yaml",
        CATEGORIES,
        ROOT / "detection/configs/models.json",
    ]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths)}


def environment() -> dict[str, Any]:
    import psutil
    import torch

    def git(*args: str) -> str | None:
        result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    result = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "torchvision",
                "ultralytics",
                "rfdetr",
                "transformers",
                "pycocotools",
                "numpy",
                "Pillow",
                "psutil",
            )
        },
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain"),
        "source_files": source_files(),
        "container_image_id": os.environ.get("CONTAINER_IMAGE_ID"),
        "ram_available_bytes": psutil.virtual_memory().available,
        "ram_total_bytes": psutil.virtual_memory().total,
        "cpu": next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            platform.processor(),
        ),
        "cgroup_memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if Path("/sys/fs/cgroup/memory.max").exists()
        else None,
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        result.update(
            gpu=torch.cuda.get_device_name(0),
            cuda_free_bytes=torch.cuda.mem_get_info()[0],
            vram_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        )
    return result


def predict(model: Any, image: Any, config: dict[str, Any]) -> tuple[list, dict]:
    """Measure native stages with CUDA synchronization; boxes are original pixels."""
    import torch

    def sync() -> None:
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)

    with torch.inference_mode():
        sync()
        start = time.perf_counter()
        prepared = model.preprocess(image)
        sync()
        pre = time.perf_counter()
        raw = model.forward(prepared)
        sync()
        forward = time.perf_counter()
        boxes, scores, labels = model.postprocess(raw, prepared, image)
        rows = normalize_predictions(
            boxes,
            scores,
            labels,
            model.mapping,
            image.width,
            image.height,
            config["score_floor"],
            config["max_detections"],
            exclude_coco_gaps=model.exclude_coco_gaps,
        )
        sync()
        end = time.perf_counter()
    return rows, {
        "preprocess_ms": (pre - start) * 1000,
        "forward_ms": (forward - pre) * 1000,
        "postprocess_ms": (end - forward) * 1000,
        "total_ms": (end - start) * 1000,
        "excluded_coco_gap_predictions": int(
            np.sum((np.asarray(scores) >= config["score_floor"]) & ~np.isin(labels, COCO_IDS))
        )
        if model.exclude_coco_gaps
        else 0,
    }


def validate_native(model: Any, image: Any, rows: list, config: dict) -> dict:
    import torch

    with torch.inference_mode():
        boxes, scores, labels = model.native_predict(image)
    native = normalize_predictions(
        boxes,
        scores,
        labels,
        model.mapping,
        image.width,
        image.height,
        config["score_floor"],
        config["max_detections"],
        exclude_coco_gaps=model.exclude_coco_gaps,
    )
    if [r["category_id"] for r in rows] != [r["category_id"] for r in native]:
        raise ValueError("Native/staged prediction counts or category order differ")
    np.testing.assert_allclose(
        [r["bbox"] for r in rows], [r["bbox"] for r in native], rtol=1e-4, atol=0.03
    )
    np.testing.assert_allclose(
        [r["score"] for r in rows], [r["score"] for r in native], rtol=1e-4, atol=1e-5
    )
    return {
        "status": "passed",
        "image_size": list(image.size),
        "n_predictions": len(rows),
        "bbox_atol_pixels": 0.03,
        "score_atol": 1e-5,
    }


def open_image(row: dict) -> Any:
    from PIL import Image

    path = (ROOT / row["path"]).resolve()
    if not path.is_relative_to((ROOT / "data").resolve()) or sha256(path) != row["sha256"]:
        raise ValueError(f"Image path/checksum mismatch: {path}")
    with Image.open(path) as source:
        if (
            source.size != (row["width"], row["height"])
            or source.width * source.height > 40_000_000
            or source.getexif().get(274, 1) != 1
        ):
            raise ValueError(f"Image dimensions/EXIF mismatch: {path}")
        return source.convert("RGB")


def resources(device: Any) -> dict:
    import torch

    return {
        "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "cuda_allocated_peak_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "cuda_reserved_peak_bytes": torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else 0,
    }


def worker(model_class: type, run: Path) -> None:
    import random

    import torch
    from PIL import ImageDraw

    meta = read_json(run / "run.json")
    config = meta["config"]
    device = torch.device(config["device"])
    model = None
    try:
        # The parent already verified files before creating the model subprocess.
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.set_num_threads(config["cpu_threads"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        meta.update(status="running", environment=environment())
        write_json(run / "run.json", meta)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        model = model_class(config)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        meta.update(
            cold_load_ms=(time.perf_counter() - start) * 1000,
            parameters=getattr(
                model, "parameters", sum(p.numel() for p in model.module.parameters())
            ),
            native_config=model.native_config(),
        )
        first = open_image(meta["dataset"]["images"][0])
        for _ in range(config["warmup"]):
            predict(model, first, config)
        write_json(run / "run.json", meta)
        with (run / "samples.jsonl").open("x", encoding="utf-8") as output:
            for index, row in enumerate(meta["dataset"]["images"]):
                start = time.perf_counter()
                image = open_image(row)
                read_ms = (time.perf_counter() - start) * 1000
                predictions, timing = predict(model, image, config)
                timing["read_ms"] = read_ms
                if index == 0:
                    meta["native_parity"] = validate_native(model, image, predictions, config)
                    image.save(run / "original.png")
                    overlay = image.copy()
                    draw = ImageDraw.Draw(overlay)
                    for pred in predictions:
                        if pred["score"] >= config["operating_threshold"]:
                            draw.rectangle(pred["bbox_xyxy"], outline="red", width=2)
                            draw.text(
                                pred["bbox_xyxy"][:2],
                                f"{COCO_NAMES[pred['category_id']]} {pred['score']:.2f}",
                                fill="red",
                            )
                    overlay.save(run / "overlay.png")
                if image.width != image.height and "native_parity_non_square" not in meta:
                    meta["native_parity_non_square"] = (
                        meta["native_parity"]
                        if index == 0
                        else validate_native(model, image, predictions, config)
                    )
                record = {"image_id": row["image_id"], "predictions": predictions, "timing": timing}
                start = time.perf_counter()
                output.write(json.dumps(record, allow_nan=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
                # fsync is complete before recording its cost in run.json.
                meta.setdefault("write_timings", []).append(
                    {
                        "image_id": row["image_id"],
                        "write_ms": (time.perf_counter() - start) * 1000,
                    }
                )
                meta.update(processed=index + 1, resources=resources(device))
                if (index + 1) % 25 == 0 or index + 1 == meta["expected_images"]:
                    write_json(run / "run.json", meta)
                    print(f"{meta['model_id']}: {index + 1}/{meta['expected_images']}", flush=True)
        meta["status"] = "inference_complete"
    except BaseException as error:
        meta.update(
            status="oom"
            if isinstance(error, (torch.cuda.OutOfMemoryError, MemoryError))
            else "failed",
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        try:
            meta["resources"] = resources(device)
        finally:
            write_json(run / "run.json", meta)


def read_samples(run: Path) -> list[dict]:
    rows = []
    path = run / "samples.jsonl"
    if path.exists():
        with path.open() as stream:
            for line in stream:
                if not line.endswith("\n"):
                    break
                rows.append(json.loads(line))
    ids = [r["image_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate image_id in {path}")
    return rows


def evaluate_run(run: Path) -> None:
    meta = read_json(run / "run.json")
    rows = read_samples(run)
    expected = [r["image_id"] for r in meta["dataset"]["images"]]
    actual = [r["image_id"] for r in rows]
    if actual != expected or meta["status"] not in {"inference_complete", "complete"}:
        if meta["status"] in {"running", "planned", "inference_complete", "complete"}:
            meta.update(status="incomplete", processed=len(rows))
            write_json(run / "run.json", meta)
        return
    annotation = ROOT / meta["dataset"]["annotations"]
    if sha256(annotation) != meta["dataset"]["annotations_sha256"]:
        raise ValueError("Annotation checksum changed")
    predictions = [
        {"image_id": row["image_id"], **{k: p[k] for k in ("bbox", "score", "category_id")}}
        for row in rows
        for p in row["predictions"]
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(annotation))
    meta["metrics"] = evaluate_coco(gt, predictions, actual)
    timings = [row["timing"] for row in rows]
    latency = np.array([t["total_ms"] for t in timings])
    meta["latency"] = {
        "p50_ms": float(np.percentile(latency, 50)),
        "p95_ms": float(np.percentile(latency, 95)),
        "images_per_second": float(1000 / latency.mean()),
        **{
            f"{stage}_mean_ms": float(np.mean([t[f"{stage}_ms"] for t in timings]))
            for stage in ("preprocess", "forward", "postprocess", "read")
        },
        "write_mean_ms": sum(t["write_ms"] for t in meta.get("write_timings", [])) / len(rows),
    }
    meta.update(
        status="complete", processed=len(rows), samples_sha256=sha256(run / "samples.jsonl")
    )
    write_json(run / "run.json", meta)


def build_report(output: Path) -> None:
    if not (output / "comparison.json").is_file():
        raise FileNotFoundError(f"No saved comparison: {output}")
    columns = [
        "model_id",
        "status",
        "processed",
        "expected_images",
        "shortened",
        *METRIC_NAMES,
        "precision",
        "recall",
        "TP",
        "FP",
        "FN",
        "p50_ms",
        "p95_ms",
        "images_per_second",
        "cold_load_ms",
        "rss_peak_GiB",
        "cuda_allocated_peak_GiB",
        "cuda_reserved_peak_GiB",
        "error",
    ]
    rows = []
    identity = read_json(output / "comparison.json")
    for model_id in MODEL_IDS:
        path = output / model_id / "run.json"
        row = dict.fromkeys(columns)
        row.update(
            model_id=model_id,
            status="not_run",
            processed=0,
            expected_images=identity["expected_images"],
            shortened=identity["shortened"],
        )
        if path.exists():
            meta = read_json(path)
            quality = (
                meta.get("metrics", {}).get("quality", {}) if meta["status"] == "complete" else {}
            )
            row.update(
                status=meta["status"],
                processed=meta.get("processed", 0),
                cold_load_ms=meta.get("cold_load_ms"),
                error=meta.get("error"),
            )
            for key in METRIC_NAMES + ["precision", "recall", "TP", "FP", "FN"]:
                row[key] = quality.get(key)
            row.update(
                {
                    k: meta.get("latency", {}).get(k)
                    for k in ("p50_ms", "p95_ms", "images_per_second")
                }
            )
            for field in ("rss", "cuda_allocated", "cuda_reserved"):
                value = meta.get("resources", {}).get(f"{field}_peak_bytes")
                row[f"{field}_peak_GiB"] = value / 2**30 if value is not None else None
        rows.append(row)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    text = [
        "# Сравнение детекторов",
        "",
        "Один проход, batch 1, FP32. "
        + (
            "Сокращённая проверка; не итог полного COCO test."
            if identity["shortened"]
            else "Полный закреплённый COCO test."
        ),
        "",
        "AP/AR в диапазоне 0–1; precision/recall: score 0.5, IoU 0.5. "
        "Время модели исключает чтение, запись и загрузку. RAM/VRAM — GiB; VRAM "
        "PyTorch allocated/reserved не включает всю память драйвера. Новые CI не считаются.",
        "",
    ]
    display = [
        "model_id",
        "status",
        "processed",
        "expected_images",
        "mAP",
        "AP50",
        "AR100",
        "precision",
        "recall",
        "p50_ms",
        "p95_ms",
        "images_per_second",
        "cold_load_ms",
        "rss_peak_GiB",
        "cuda_allocated_peak_GiB",
        "cuda_reserved_peak_GiB",
    ]
    text += ["| " + " | ".join(display) + " |", "| " + " | ".join(["---"] * len(display)) + " |"]

    def show(value: Any) -> str:
        return "—" if value is None else f"{value:.6g}" if isinstance(value, float) else str(value)

    for row in rows:
        text.append("| " + " | ".join(show(row[k]) for k in display) + " |")
    per_class = []
    for model_id in MODEL_IDS:
        path = output / model_id / "run.json"
        if path.exists():
            meta = read_json(path)
            if meta["status"] == "complete":
                for category, values in meta["metrics"]["per_class"].items():
                    per_class.append({"model_id": model_id, "category_id": category, **values})
    with (output / "per_class.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["model_id", "category_id", "name", "AP", "TP", "FP", "FN"]
        )
        writer.writeheader()
        writer.writerows(per_class)
    (output / "comparison.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    console_columns = ["model_id", "status", "mAP", "AP50", "p50_ms"]
    widths = {key: max(len(key), *(len(show(row[key])) for row in rows)) for key in console_columns}
    print("\nДетекторы: " + output.name, flush=True)
    print(" | ".join(key.ljust(widths[key]) for key in console_columns), flush=True)
    print("-+-".join("-" * widths[key] for key in console_columns), flush=True)
    for row in rows:
        print(" | ".join(show(row[key]).ljust(widths[key]) for key in console_columns), flush=True)


def run_child(command: list[str], lock_fd: int, run: Path) -> int:
    env = {
        **os.environ,
        "KURS4_MODEL_RUN": str(run),
        "KURS4_LOCK_FD": str(lock_fd),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    with (run / "process.log").open("x") as log:
        process = subprocess.Popen(
            command,
            pass_fds=(lock_fd,),
            start_new_session=True,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        previous = signal.getsignal(signal.SIGTERM)

        def stopped(_signum: int, _frame: Any) -> None:
            raise KeyboardInterrupt("SIGTERM received")

        signal.signal(signal.SIGTERM, stopped)
        try:
            return process.wait()
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        finally:
            signal.signal(signal.SIGTERM, previous)


def oom_kills() -> int:
    path = Path("/sys/fs/cgroup/memory.events")
    if not path.exists():
        return 0
    return int(dict(line.split() for line in path.read_text().splitlines()).get("oom_kill", 0))


def main(model_class: type, model_id: str) -> None:
    child_path = os.environ.get("KURS4_MODEL_RUN")
    if child_path:
        # Refuse the internal entry point without an inherited, held flock.
        fd = int(os.environ["KURS4_LOCK_FD"])
        os.fstat(fd)
        worker(model_class, Path(child_path))
        return
    parser = argparse.ArgumentParser(description=f"Один офлайн проход COCO: {model_id}")
    parser.add_argument("--run", required=True, help="Имя нового сравнения")
    parser.add_argument(
        "--limit", type=int, help="Положительное число изображений для короткой проверки"
    )
    parser.add_argument(
        "--report-only", action="store_true", help="Восстановить таблицы без моделей"
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", args.run):
        parser.error("Недопустимое имя сравнения")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit должен быть положительным")
    output = ROOT / "reports/runs/detection" / args.run
    with file_lock(ROOT / "storage/locks/inference.lock") as lock_fd:
        if args.report_only:
            if not (output / "comparison.json").is_file():
                raise FileNotFoundError(f"No saved comparison: {output}")
            saved = read_json(output / "comparison.json")
            if args.limit is not None and args.limit != saved.get("limit"):
                raise ValueError("Report --limit differs from the saved comparison")
            for item in MODEL_IDS:
                path = output / item
                if (path / "run.json").exists():
                    evaluate_run(path)
            build_report(output)
            return
        config = {
            "protocol": "detection-simple-v1",
            "device": os.environ.get("KURS4_DEVICE", "cuda:0"),
            "precision": "fp32",
            "batch_size": 1,
            "seed": 42,
            "warmup": 20,
            "repeats": 1,
            "score_floor": 0.001,
            "operating_threshold": 0.5,
            "operating_iou": 0.5,
            "max_detections": 300,
            "cpu_threads": 4,
        }
        if config["device"] != "cpu" and not re.fullmatch(r"cuda(:[0-9]+)?", config["device"]):
            raise ValueError("KURS4_DEVICE must be cpu or cuda:N")
        manifest_path = ROOT / "data/coco/test.json"
        dataset = read_json(manifest_path)
        total = len(dataset["images"])
        if total != 4500:
            raise ValueError("Expected frozen 4500-image COCO test manifest")
        frozen = read_json(ROOT / "detection/configs/dataset_split.json")
        if object_hash(dataset) != frozen["splits"]["test"]["manifest_hash"]:
            raise ValueError("COCO test manifest differs from its pinned split")
        if args.limit is not None:
            dataset["images"] = dataset["images"][: args.limit]
        if sha256(ROOT / dataset["annotations"]) != dataset["annotations_sha256"]:
            raise ValueError("Annotation checksum changed")
        identity = {
            "config": config,
            "manifest_sha256": sha256(manifest_path),
            "selected_images_sha256": object_hash(dataset["images"]),
            "limit": args.limit,
            "expected_images": len(dataset["images"]),
            "shortened": args.limit is not None,
            "source_files": source_files(),
            "container_image_id": os.environ.get("CONTAINER_IMAGE_ID"),
        }
        output.mkdir(parents=True, exist_ok=True)
        comparison_path = output / "comparison.json"
        if comparison_path.exists():
            if read_json(comparison_path) != identity:
                raise ValueError("Comparison config/data/source/container differ; use a new --run")
        else:
            if any(output.iterdir()):
                raise ValueError("Comparison directory is not empty; use a new --run")
            write_json(comparison_path, identity)
        run = output / model_id
        if run.exists():
            raise FileExistsError(f"Refusing to overwrite {run}; use a new --run")
        run.mkdir()
        meta = {
            "model_id": model_id,
            "status": "planned",
            "processed": 0,
            "expected_images": len(dataset["images"]),
            "shortened": args.limit is not None,
            "config": config,
            "dataset": dataset,
            "resources": {},
            "cold_load_ms": None,
            "command": sys.argv,
        }
        write_json(run / "run.json", meta)
        killed_before = oom_kills()
        try:
            meta["model"] = verify_model(model_id)
            write_json(run / "run.json", meta)
            print(f"Starting {model_id}; log: {run / 'process.log'}", flush=True)
            code = run_child([sys.executable, str(Path(sys.argv[0]).resolve())], lock_fd, run)
            meta = read_json(run / "run.json")
            if code:
                if meta["status"] not in {"failed", "oom"}:
                    meta.update(
                        status="oom"
                        if oom_kills() > killed_before
                        else "incomplete"
                        if code < 0
                        else "failed",
                        error=f"Model process exited with code {code}; see process.log",
                        processed=len(read_samples(run)),
                    )
                    write_json(run / "run.json", meta)
                raise RuntimeError(f"Model {model_id}: {meta['status']}; see {run / 'process.log'}")
            evaluate_run(run)
        except BaseException as error:
            meta = read_json(run / "run.json")
            if meta["status"] not in {"failed", "oom"}:
                meta.update(
                    status="incomplete" if isinstance(error, KeyboardInterrupt) else "failed",
                    error=str(error),
                    processed=len(read_samples(run)),
                )
                write_json(run / "run.json", meta)
            raise
        finally:
            build_report(output)
        print(f"Finished {model_id}; table: {output / 'comparison.csv'}", flush=True)
