"""ImageNet data checks, one-process measurements and four-row comparison tables."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import resource
import signal
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "classification/configs"
MODELS = ROOT / "classification/models"
MODEL_IDS = ("resnet50", "efficientnet_v2_s", "convnext_tiny", "vit_b_16")
PROTOCOL = "classification-imagenet-single-pass-v2"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def model_spec(model_id: str) -> dict:
    return next(m for m in read_json(CONFIGS / "models.json")["models"] if m["id"] == model_id)


@lru_cache(maxsize=1)
def labels() -> list[dict]:
    classes = read_json(CONFIGS / "imagenet_labels.json")["classes"]
    if (
        len(classes) != 1000
        or [r["index"] for r in classes] != list(range(1000))
        or [r["synset"] for r in classes] != sorted({r["synset"] for r in classes})
    ):
        raise ValueError("Expected 1000 ordered ImageNet indices and unique sorted synsets")
    return classes


def data_path(name: str) -> Path:
    path = (ROOT / name).resolve()
    if not path.is_relative_to((ROOT / "data").resolve()):
        raise ValueError("Images must be inside data/")
    return path


def verify_model(model_id: str) -> dict:
    spec = model_spec(model_id)
    manifest = read_json(MODELS / "manifest.json")["models"][model_id]
    if manifest["status"] not in {"verified", "smoke_passed"}:
        raise ValueError(f"Local weights are not verified: {model_id}; run prepare.py explicitly")
    if manifest["spec_hash"] != object_hash(spec):
        raise ValueError("Model specification changed")
    pinned = {item["path"]: item for item in manifest["files"]}
    for item in spec["files"]:
        if pinned[item["path"]]["sha256"] != item["sha256"]:
            raise ValueError("Model manifest differs from pinned official SHA-256")
    for item in manifest["files"]:
        path = MODELS / model_id / item["path"]
        if path.stat().st_size != item["size"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Model file hash/size mismatch: {path}")
    if read_json(MODELS / model_id / "label_mapping.json") != labels():
        raise ValueError("Checkpoint synset mapping differs from frozen ImageNet mapping")
    return manifest


def validate_native(model_id: str, weights: Any) -> dict:
    classes = labels()
    if weights.meta["categories"] != [r["name"] for r in classes]:
        raise ValueError("Checkpoint class ordering differs from frozen mapping")
    transform = weights.transforms()
    native = {
        "schema_version": "1.0",
        "weights": model_spec(model_id)["weights"],
        "crop_size": transform.crop_size,
        "resize_size": transform.resize_size,
        "mean": transform.mean,
        "std": transform.std,
        "interpolation": transform.interpolation.value,
        "antialias": transform.antialias,
        "class_mapping_hash": object_hash(classes),
    }
    if native != read_json(MODELS / model_id / "native_config.json"):
        raise ValueError("Native preprocessing differs from frozen checkpoint metadata")
    return native


def validate_dataset(dataset: dict) -> None:
    rows = dataset["images"]
    classes = labels()
    if (
        dataset["mode"] != "whole_image"
        or dataset["split"] != "validation"
        or dataset["class_mapping_hash"] != object_hash(classes)
        or len(rows) != 50000
        or len({row["image_id"] for row in rows}) != 50000
        or Counter(row["target_index"] for row in rows) != dict.fromkeys(range(1000), 50)
    ):
        raise ValueError("Expected complete balanced ImageNet validation with frozen mapping")
    for row in rows:
        if classes[row["target_index"]]["synset"] != row["target_synset"]:
            raise ValueError("Target synset and index differ")
        data_path(row["path"])


def validate_prediction(row: dict, source: dict, model_id: str) -> None:
    if (
        row["model_id"] != model_id
        or row["target_index"] != source["target_index"]
        or row["target_synset"] != source["target_synset"]
    ):
        raise ValueError("Prediction reference label or model differs from dataset")
    top = row["predictions"]
    indices = [p["index"] for p in top]
    if (
        len(top) != 5
        or len(set(indices)) != 5
        or any(type(index) is not int or not 0 <= index < 1000 for index in indices)
    ):
        raise ValueError("Expected five unique ImageNet class indices")
    classes = labels()
    for item in top:
        if any(item[key] != classes[item["index"]][key] for key in ("name", "synset")):
            raise ValueError("Prediction name/synset differs from frozen mapping")
    scores = [p["score"] for p in top]
    if (
        any(not math.isfinite(s) or not 0 <= s <= 1 for s in scores)
        or scores != sorted(scores, reverse=True)
        or sum(scores) > 1 + 1e-6
    ):
        raise ValueError("Invalid top-5 probabilities")
    timing = row["timing"]
    if any(not math.isfinite(v) or v < 0 for v in timing.values()):
        raise ValueError("Invalid measurement")
    total = sum(timing[f"{stage}_ms"] for stage in ("preprocess", "inference", "postprocess"))
    if not math.isclose(total, timing["total_ms"], rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("Total timing differs from its stages")


def load_samples(run: Path, sources: list[dict], *, complete: bool = True) -> list[dict]:
    meta = read_json(run / "run.json")
    expected = {row["image_id"]: row for row in sources}
    seen = set()
    records = []
    with (run / "samples.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.endswith("\n"):
                if complete:
                    raise ValueError("Incomplete final JSONL record")
                break
            row = json.loads(line)
            image_id = row["image_id"]
            if image_id in seen or image_id not in expected:
                raise ValueError("Duplicate or unexpected image_id")
            validate_prediction(row, expected[image_id], meta["model_id"])
            seen.add(image_id)
            records.append(row)
    if complete and seen != set(expected):
        raise ValueError("Incomplete predictions; quality cannot be evaluated")
    return records


def quality(records: list[dict], n_classes: int = 1000) -> dict:
    if not records or any(r["target_index"] is None for r in records):
        raise ValueError("Quality requires ground truth")
    truth = [r["target_index"] for r in records]
    predictions = [r["predictions"][0]["index"] for r in records]
    if any(not 0 <= i < n_classes for i in truth + predictions):
        raise ValueError("Invalid target or predicted labels")
    support = Counter(truth)
    predicted = Counter(predictions)
    correct = Counter(t for t, p in zip(truth, predictions, strict=True) if t == p)
    per_class = []
    for index in range(n_classes):
        denominator = support[index] + predicted[index]
        per_class.append(
            {
                "index": index,
                "support": support[index],
                "correct": correct[index],
                "accuracy": correct[index] / support[index] if support[index] else None,
                "f1": 2 * correct[index] / denominator if denominator else 0.0,
            }
        )
    return {
        "n_images": len(records),
        "n_classes": n_classes,
        "top1": sum(correct.values()) / len(records),
        "top5": sum(r["target_index"] in [p["index"] for p in r["predictions"]] for r in records)
        / len(records),
        "macro_f1": sum(r["f1"] for r in per_class) / n_classes,
        "per_class": per_class,
        "confusion_sparse": [
            {"target": t, "predicted": p, "count": count}
            for (t, p), count in sorted(Counter(zip(truth, predictions, strict=True)).items())
        ],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: list[float], percent: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * percent / 100
    low = math.floor(position)
    high = math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def build_report(output: Path) -> None:
    comparison = read_json(output / "comparison.json")
    if (
        sha256(CONFIGS / "imagenet_labels.json") != comparison["labels_sha256"]
        or sha256(CONFIGS / "models.json") != comparison["models_sha256"]
    ):
        raise ValueError("Frozen model/class specifications changed")
    manifest = ROOT / comparison["dataset"]["path"]
    if sha256(manifest) != comparison["dataset"]["sha256"]:
        raise ValueError("Dataset manifest changed since comparison")
    sources = read_json(manifest)["images"][: comparison["expected_images"]]
    rows, per_class = [], []
    for model_id in MODEL_IDS:
        row = dict.fromkeys(
            (
                "model",
                "model_id",
                "status",
                "shortened",
                "processed",
                "expected_images",
                "top1",
                "top5",
                "macro_f1",
                "p50_ms",
                "p95_ms",
                "images_per_second",
                "cold_load_ms",
                "RAM_peak_GiB",
                "VRAM_allocated_peak_GiB",
                "VRAM_reserved_peak_GiB",
                "error",
            )
        )
        row.update(
            model=model_spec(model_id)["name"],
            model_id=model_id,
            status="not_run",
            shortened=comparison["shortened"],
            processed=0,
            expected_images=comparison["expected_images"],
        )
        run = output / model_id
        if (run / "run.json").exists():
            meta = read_json(run / "run.json")
            if meta["comparison_hash"] != object_hash(comparison):
                raise ValueError("Run belongs to a different comparison")
            status = meta["status"]
            row.update(
                status=status if status in {"complete", "failed", "oom"} else "incomplete",
                processed=meta["processed"],
                cold_load_ms=meta.get("cold_load_ms"),
                error=meta.get("error"),
            )
            for label, key in (
                ("RAM", "rss"),
                ("VRAM_allocated", "cuda_allocated"),
                ("VRAM_reserved", "cuda_reserved"),
            ):
                value = meta.get("resources", {}).get(f"{key}_peak_bytes")
                row[f"{label}_peak_GiB"] = value / 1024**3 if value is not None else None
            if status == "complete":
                if sha256(run / "samples.jsonl") != meta["samples_sha256"]:
                    raise ValueError("Completed samples changed")
                records = load_samples(run, sources)
                if len(records) != meta["processed"]:
                    raise ValueError("Processed count differs from predictions")
                metrics = quality(records)
                row.update({key: metrics[key] for key in ("top1", "top5", "macro_f1")})
                times = [r["timing"]["total_ms"] for r in records]
                row.update(
                    p50_ms=percentile(times, 50),
                    p95_ms=percentile(times, 95),
                    images_per_second=1000 * len(times) / sum(times),
                )
                per_class.extend(
                    {"model_id": model_id, **labels()[r["index"]], **r}
                    for r in metrics["per_class"]
                )
        rows.append(row)
    print("\nСравнение ImageNet (Top-1/Top-5: доли, p50: ms):", flush=True)
    print(f"{'model':<23} {'status':<12} {'Top1':>9} {'Top5':>9} {'p50_ms':>10}", flush=True)
    for row in rows:
        values = [
            "—" if row[key] is None else f"{row[key]:.5f}" for key in ("top1", "top5", "p50_ms")
        ]
        print(
            f"{row['model']:<23} {row['status']:<12} {values[0]:>9} {values[1]:>9} {values[2]:>10}",
            flush=True,
        )
    write_csv(output / "comparison.csv", rows)
    if per_class:
        write_csv(output / "per_class.csv", per_class)
    columns = (
        "model",
        "status",
        "shortened",
        "processed",
        "top1",
        "top5",
        "macro_f1",
        "p50_ms",
        "p95_ms",
        "RAM_peak_GiB",
        "VRAM_allocated_peak_GiB",
        "VRAM_reserved_peak_GiB",
    )
    lines = [
        "# Классификация ImageNet: один проход",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                "—"
                if row[key] is None
                else f"{row[key]:.6f}"
                if isinstance(row[key], float)
                else str(row[key])
                for key in columns
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Доли качества: 0–1; Macro-F1 включает все 1000 классов.",
            "shortened=True означает проверку части набора, а не полный ImageNet.",
            "p50/p95: preprocessing + transfer + forward + postprocessing; чтение/запись отдельно в JSONL.",
            "Скорость = 1000 / средняя задержка, без чтения, записи, прогрева и загрузки.",
            "RAM: peak RSS процесса. VRAM: пики PyTorch allocated/reserved с загрузкой модели; не вся память GPU.",
            "Холодная загрузка выполняется в новом процессе без очистки page cache. Новые CI не рассчитываются.",
        ]
    )
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


@contextmanager
def file_lock(path: Path) -> Iterator[int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Lock already held: {path}") from error
        # Close only: an inherited child descriptor must retain this flock if the parent dies.
        yield handle.fileno()


def require_inherited_lock() -> None:
    """The private worker path must hold the very same system inference lock."""
    try:
        descriptor = int(os.environ["KURS4_LOCK_FD"])
        inherited = os.fstat(descriptor)
        path = ROOT / "storage/locks/inference.lock"
        actual = path.stat()
        if (inherited.st_dev, inherited.st_ino) != (actual.st_dev, actual.st_ino):
            raise ValueError("Inherited lock descriptor references a different file")
        with path.open("a+") as probe:
            try:
                fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return
        raise ValueError("Inference lock is not held")
    except (KeyError, OSError, ValueError) as error:
        raise RuntimeError(
            f"Model worker requires a valid inherited inference lock: {error}"
        ) from error


def memory_events() -> dict[str, int]:
    path = Path("/sys/fs/cgroup/memory.events")
    if not path.exists():
        return {}
    return {
        key: int(value) for key, value in (line.split() for line in path.read_text().splitlines())
    }


def cgroup_oom(before: dict[str, int], after: dict[str, int]) -> bool:
    return after.get("oom_kill", 0) > before.get("oom_kill", 0)


def source_fingerprint() -> dict[str, str]:
    sources = list(Path(__file__).parent.glob("*.py")) + list(CONFIGS.glob("*.json"))
    sources += [
        ROOT / name
        for name in (
            "classification/requirements.in",
            "classification/requirements.lock.txt",
            "classification/Dockerfile",
            "compose.yaml",
            "compose.gpu.yaml",
        )
    ]
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(sources) if path.is_file()}


def child(command: list[str], lock_fd: int, environment: dict[str, str]) -> int:
    process = subprocess.Popen(
        command, env=environment, pass_fds=(lock_fd,), start_new_session=True
    )
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}

    def terminate(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt(f"Received signal {signum}")

    for sig in previous:
        signal.signal(sig, terminate)
    try:
        return process.wait()
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def environment_info() -> dict:
    import psutil
    import torch

    def git(*args: str) -> str | None:
        result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    cgroup = Path("/sys/fs/cgroup/memory.max")
    result = {
        "python": sys.version,
        "os": platform.platform(),
        "ram_total_bytes": psutil.virtual_memory().total,
        "ram_available_bytes": psutil.virtual_memory().available,
        "cgroup_memory_max": cgroup.read_text().strip() if cgroup.exists() else None,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchvision", "numpy", "Pillow", "psutil")
        },
        "git_commit": os.environ.get("PROJECT_GIT_COMMIT") or git("rev-parse", "HEAD"),
        "git_status": os.environ.get("PROJECT_GIT_STATUS", git("status", "--porcelain")),
        "container_image_id": os.environ.get("CONTAINER_IMAGE_ID"),
        "source_sha256": source_fingerprint(),
        "torch_cuda": torch.version.cuda,
    }
    if torch.cuda.is_available():
        result.update(
            gpu=torch.cuda.get_device_name(0),
            vram_total_bytes=torch.cuda.get_device_properties(0).total_memory,
            cuda_free_bytes=torch.cuda.mem_get_info()[0],
        )
    return result


def resources(device: Any) -> dict:
    import torch

    cuda = device.type == "cuda" and torch.cuda.is_available()
    return {
        "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "cuda_allocated_peak_bytes": torch.cuda.max_memory_allocated(device) if cuda else 0,
        "cuda_reserved_peak_bytes": torch.cuda.max_memory_reserved(device) if cuda else 0,
    }


def worker(model_class: Any, model_id: str, run: Path) -> None:
    import numpy as np
    import torch
    from PIL import Image

    def terminate(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    meta = read_json(run / "run.json")
    config = meta["config"]
    comparison = read_json(run.parent / "comparison.json")
    manifest = ROOT / comparison["dataset"]["path"]
    done = 0
    adapter = None
    meta["status"] = "running"
    try:
        if sha256(manifest) != comparison["dataset"]["sha256"]:
            raise ValueError("Dataset manifest changed")
        sources = read_json(manifest)["images"][: meta["expected_images"]]
        meta["model_manifest"] = verify_model(model_id)
        meta["environment"] = environment_info()
        if not re.fullmatch(
            r"sha256:[0-9a-f]{64}", meta["environment"]["container_image_id"] or ""
        ):
            raise ValueError("CONTAINER_IMAGE_ID must contain the actual Docker image ID")
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.set_num_threads(config["cpu_threads"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        start = time.perf_counter()
        adapter = model_class(config["device"])
        adapter.sync()
        meta.update(
            cold_load_ms=(time.perf_counter() - start) * 1000,
            native_config=adapter.native,
            parameters=adapter.parameters,
        )
        first = sources[0]
        if sha256(data_path(first["path"])) != first["sha256"]:
            raise ValueError("Image content changed")
        with Image.open(data_path(first["path"])) as original:
            image = original.convert("RGB")
            for _ in range(config["warmup"]):
                adapter.predict(image)
        write_json(run / "run.json", meta)
        with (run / "samples.jsonl").open("x", encoding="utf-8") as handle:
            for index, source in enumerate(sources):
                path = data_path(source["path"])
                if sha256(path) != source["sha256"]:
                    raise ValueError("Image content changed")
                start = time.perf_counter()
                with Image.open(path) as original:
                    if (
                        original.size != (source["width"], source["height"])
                        or original.width * original.height > 40_000_000
                    ):
                        raise ValueError("Image dimensions changed")
                    image = original.convert("RGB")
                read_ms = (time.perf_counter() - start) * 1000
                predictions, timing = adapter.predict(image)
                if index == 0:
                    meta["native_parity"] = adapter.validate_native(image, predictions)
                timing["read_ms"] = read_ms
                record = {
                    "image_id": source["image_id"],
                    "model_id": model_id,
                    "target_index": source["target_index"],
                    "target_synset": source["target_synset"],
                    "predictions": predictions,
                    "timing": timing,
                }
                validate_prediction(record, source, model_id)
                start = time.perf_counter()
                # Encoding time is retained in this row. The following row also measures actual
                # flush/fsync of the previous record, without rewriting durable predictions.
                json.dumps(record, ensure_ascii=False, allow_nan=False)
                record["timing"]["encode_ms"] = (time.perf_counter() - start) * 1000
                record["timing"]["previous_write_ms"] = meta.pop("last_write_ms", 0.0)
                record["previous_write_image_id"] = (
                    sources[index - 1]["image_id"] if index else None
                )
                start = time.perf_counter()
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                meta["last_write_ms"] = (time.perf_counter() - start) * 1000
                done += 1
                meta.update(processed=done, resources=resources(adapter.device))
                if done % 100 == 0:
                    write_json(run / "run.json", meta)
                    print(f"{model_id}: {done}/{len(sources)}", flush=True)
        meta.update(status="inference_complete", samples_sha256=sha256(run / "samples.jsonl"))
    except BaseException as error:
        meta.update(
            status="oom"
            if isinstance(error, torch.cuda.OutOfMemoryError)
            else "incomplete"
            if isinstance(error, KeyboardInterrupt)
            else "failed",
            processed=done,
            error=f"{type(error).__name__}: {error}",
        )
        raise
    finally:
        meta["resources"] = resources(
            adapter.device if adapter is not None else torch.device(config["device"])
        )
        write_json(run / "run.json", meta)


def comparison_config(limit: int | None) -> dict:
    return {
        "protocol": PROTOCOL,
        "device": "cuda:0",
        "precision": "fp32",
        "batch_size": 1,
        "seed": 42,
        "warmup": 20,
        "cpu_threads": 4,
        "top_k": 5,
        "passes": 1,
        "limit": limit,
    }


def main(model_class: Any, model_id: str) -> None:
    internal = os.environ.get("KURS4_MODEL_RUN")
    if internal:
        require_inherited_lock()
        worker(model_class, model_id, Path(internal))
        return
    parser = argparse.ArgumentParser(
        description=f"{model_id}: ImageNet, один проход, локальные веса"
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", args.run):
        parser.error("--run must be a simple new comparison name (letters, digits, _, -, .)")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    output = ROOT / "reports/runs/classification" / args.run
    if args.report_only:
        if not (output / "comparison.json").exists():
            parser.error("comparison does not exist")
        if (
            args.limit is not None
            and args.limit != read_json(output / "comparison.json")["config"]["limit"]
        ):
            parser.error("--limit differs from saved comparison")
        with file_lock(ROOT / "storage/locks/inference.lock"):
            build_report(output)
        return
    dataset_path = ROOT / "data/imagenet/validation.json"
    dataset = read_json(dataset_path)
    validate_dataset(dataset)
    if args.limit is not None and args.limit > len(dataset["images"]):
        parser.error("--limit exceeds the dataset")
    config = comparison_config(args.limit)
    comparison = {
        "protocol": PROTOCOL,
        "config": config,
        "dataset": {"path": str(dataset_path.relative_to(ROOT)), "sha256": sha256(dataset_path)},
        "labels_sha256": sha256(CONFIGS / "imagenet_labels.json"),
        "models_sha256": sha256(CONFIGS / "models.json"),
        "source_sha256": source_fingerprint(),
        "container_image_id": os.environ.get("CONTAINER_IMAGE_ID"),
        "expected_images": min(args.limit or len(dataset["images"]), len(dataset["images"])),
        "full_images": len(dataset["images"]),
        "shortened": args.limit is not None and args.limit < len(dataset["images"]),
    }
    with file_lock(ROOT / "storage/locks/inference.lock") as lock_fd:
        output.mkdir(parents=True, exist_ok=True)
        identity_path = output / "comparison.json"
        if identity_path.exists():
            if read_json(identity_path) != comparison:
                raise ValueError(
                    "Comparison settings/data/source/container differ; use a new --run"
                )
        else:
            write_json(identity_path, comparison)
        run = output / model_id
        if run.exists():
            raise FileExistsError("Model result already exists; use a new --run after interruption")
        run.mkdir()
        meta = {
            "model_id": model_id,
            "status": "planned",
            "processed": 0,
            "expected_images": comparison["expected_images"],
            "config": config,
            "shortened": comparison["shortened"],
            "comparison_hash": object_hash(comparison),
            "resources": {},
            "cold_load_ms": None,
        }
        write_json(run / "run.json", meta)
        environment = {**os.environ, "KURS4_MODEL_RUN": str(run), "KURS4_LOCK_FD": str(lock_fd)}
        events_before = memory_events()
        code = 1
        interrupted = False
        try:
            code = child([sys.executable, str(Path(sys.argv[0]).resolve())], lock_fd, environment)
            meta = read_json(run / "run.json")
            if code == 0 and meta["status"] == "inference_complete":
                sources = dataset["images"][: comparison["expected_images"]]
                records = load_samples(run, sources)
                metrics = quality(records)
                meta.update(
                    status="complete",
                    metrics={
                        k: v
                        for k, v in metrics.items()
                        if k not in {"per_class", "confusion_sparse"}
                    },
                )
            elif meta["status"] not in {"failed", "oom", "incomplete"}:
                meta.update(status="incomplete", error=f"Model process exit {code}")
        except KeyboardInterrupt as error:
            interrupted = True
            meta = read_json(run / "run.json")
            if meta["status"] not in {"failed", "oom"}:
                meta.update(status="incomplete", error=str(error))
        except Exception as error:
            meta = read_json(run / "run.json")
            meta.update(status="failed", error=f"{type(error).__name__}: {error}")
        finally:
            events_after = memory_events()
            meta["cgroup_memory_events"] = {"before": events_before, "after": events_after}
            if cgroup_oom(events_before, events_after):
                meta.update(status="oom", error=f"cgroup OOM kill; model process exit {code}")
            if meta["status"] != "complete" and (run / "samples.jsonl").exists():
                try:
                    meta["processed"] = len(
                        load_samples(
                            run, dataset["images"][: comparison["expected_images"]], complete=False
                        )
                    )
                except (ValueError, KeyError) as error:
                    meta["error"] = f"{meta.get('error', '')}; invalid partial samples: {error}"
            write_json(run / "run.json", meta)
            build_report(output)
        if interrupted:
            raise SystemExit(130)
        if meta["status"] != "complete":
            raise SystemExit(1)
    print(f"{model_id}: complete; {output / 'comparison.md'}", flush=True)
