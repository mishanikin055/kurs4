"""Sequential subprocess runner; the inference lock survives parent crashes."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import resource
import signal
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path
from typing import Any

from detection.common import (
    ROOT,
    file_lock,
    object_hash,
    read_json,
    sha256,
    verify_model,
    write_json,
)


def source_fingerprint() -> dict[str, str]:
    paths = sorted(
        list((ROOT / "detection").glob("*.py"))
        + list((ROOT / "detection/configs").glob("*.json"))
        + list((ROOT / "scripts").glob("*.py"))
        + list((ROOT / "scripts").glob("*.sh"))
        + list((ROOT / "detection").glob("*requirements*"))
        + [
            ROOT / "detection/Dockerfile",
            ROOT / "compose.detection.yaml",
            ROOT / "compose.detection.wsl.yaml",
        ]
    )
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths if p.is_file()}


def environment() -> dict[str, Any]:
    import psutil
    import torch

    packages = {}
    for name in [
        "torch",
        "torchvision",
        "ultralytics",
        "rfdetr",
        "transformers",
        "numpy",
        "pycocotools",
        "Pillow",
        "psutil",
    ]:
        packages[name] = importlib.metadata.version(name)

    def git(*args: str) -> str | None:
        result = subprocess.run(
            ["git", "-C", str(ROOT), *args], text=True, capture_output=True, check=False
        )
        return result.stdout.strip() if result.returncode == 0 else None

    cpu = next(
        (
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        platform.processor(),
    )
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "python": sys.version,
        "os": platform.platform(),
        "cpu": cpu,
        "ram_total_bytes": psutil.virtual_memory().total,
        "ram_available_bytes": psutil.virtual_memory().available,
        "cgroup_memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if Path("/sys/fs/cgroup/memory.max").exists()
        else None,
        "packages": packages,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "container_image_id": os.environ.get("CONTAINER_IMAGE_ID"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_status": git("status", "--porcelain"),
        "source_files": source_fingerprint(),
    }
    result["dirty"] = bool(result["git_status"]) or result["git_commit"] is None
    if torch.cuda.is_available():
        result.update(
            gpu=torch.cuda.get_device_name(0),
            vram_total_bytes=torch.cuda.get_device_properties(0).total_memory,
            cuda_free_bytes=torch.cuda.mem_get_info()[0],
        )
    return result


def validate_config(config: dict[str, Any]) -> None:
    if config["batch_size"] != 1 or config["precision"] != "fp32":
        raise ValueError(
            "This protocol supports batch=1 FP32 only; other modes need a new config/version"
        )
    if config["device"] != "cpu" and not config["device"].startswith("cuda"):
        raise ValueError("Device must be cpu or cuda:N")
    if config["warmup"] < 1 or config["repeats"] < 1 or config["max_detections"] < 100:
        raise ValueError("Invalid warmup/repeat/max_detections")
    if not 0 <= config["score_floor"] < config["operating_threshold"] <= 1:
        raise ValueError("AP score floor must be lower than the operating threshold")


def recover_samples(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done = set()
    last_valid = 0
    with path.open("rb") as stream:
        while line := stream.readline():
            if not line.endswith(b"\n"):
                break
            row = json.loads(line)
            if row["image_id"] in done:
                raise ValueError("Duplicate image_id in resume artifacts")
            done.add(row["image_id"])
            last_valid = stream.tell()
    with path.open("r+b") as stream:
        stream.truncate(last_valid)
    return done


def worker(run: Path, resume: bool) -> None:
    import random

    import numpy as np
    import psutil
    import torch
    from PIL import Image, ImageDraw

    from detection.adapters import COCO_NAMES, Adapter

    meta = read_json(run / "run.json")
    config = meta["config"]
    verify_model(meta["model_id"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["cpu_threads"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    dataset = read_json(run / "dataset_manifest.json")
    rows = dataset["images"]
    done = recover_samples(run / "samples.jsonl") if resume else set()
    if not done.issubset({r["image_id"] for r in rows}):
        raise ValueError("Resume samples are outside the frozen dataset")
    write_json(run / "environment.json", environment())
    errors_path = run / "errors.jsonl"
    errors_path.touch(exist_ok=True)
    meta["status"] = "running"
    meta["attempts"] = meta.get("attempts", 0) + 1
    write_json(run / "run.json", meta)
    try:
        start = time.perf_counter()
        adapter = Adapter(meta["model_id"], config)
        adapter.sync()
        load_ms = (time.perf_counter() - start) * 1000
        meta.setdefault("cold_load_ms", load_ms)
        meta.setdefault("load_attempts_ms", []).append(load_ms)
        meta["parameters"] = adapter.parameters
        write_json(run / "native_config.json", adapter.native_config())
        if adapter.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(adapter.device)
        first = rows[0]
        with Image.open(ROOT / first["path"]) as image:
            if image.getexif().get(274, 1) != 1:
                raise ValueError("EXIF orientation requires a new image/GT coordinate version")
            image = image.convert("RGB")
            for _ in range(config["warmup"]):
                adapter.predict(image)
        resources = meta.get("resources", {})
        rss = psutil.Process()
        resources["rss_peak_bytes"] = max(
            resources.get("rss_peak_bytes", 0),
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        )
        if adapter.device.type == "cuda":
            resources["cuda_allocated_peak_bytes"] = max(
                resources.get("cuda_allocated_peak_bytes", 0),
                torch.cuda.max_memory_allocated(adapter.device),
            )
            resources["cuda_reserved_peak_bytes"] = max(
                resources.get("cuda_reserved_peak_bytes", 0),
                torch.cuda.max_memory_reserved(adapter.device),
            )
        for index, row in enumerate(rows):
            if row["image_id"] in done:
                continue
            path = (ROOT / row["path"]).resolve()
            if not path.is_relative_to((ROOT / "data").resolve()) or sha256(path) != row["sha256"]:
                raise ValueError(f"Image path/hash mismatch: {path}")
            start = time.perf_counter()
            with Image.open(path) as image:
                if (
                    image.size != (row["width"], row["height"])
                    or image.width * image.height > 40_000_000
                    or image.getexif().get(274, 1) != 1
                ):
                    raise ValueError(f"Image dimensions/EXIF mismatch: {path}")
                image = image.convert("RGB")
                read_ms = (time.perf_counter() - start) * 1000
                predictions, timing = adapter.predict(image)
                if meta.get("smoke"):
                    write_json(
                        run / "native_parity.json", adapter.validate_native(image, predictions)
                    )
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
            timing["read_ms"] = read_ms
            record = {
                "schema_version": "1.0",
                "image_id": row["image_id"],
                "model_id": meta["model_id"],
                "predictions": predictions,
                "timing": timing,
            }
            # One durable per-image record is authoritative for resume and report rebuilds.
            start_write = time.perf_counter()
            record["timing"]["write_ms"] = 0.0
            json.dumps(record, allow_nan=False)
            record["timing"]["write_ms"] = (time.perf_counter() - start_write) * 1000
            with (run / "samples.jsonl").open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, allow_nan=False) + "\n")
                output.flush()
                os.fsync(output.fileno())
            # Disk flush is separate from serialized timing (stored in write_timings.jsonl).
            with (run / "write_timings.jsonl").open("a") as output:
                output.write(
                    json.dumps(
                        {
                            "image_id": row["image_id"],
                            "write_ms": (time.perf_counter() - start_write) * 1000,
                        }
                    )
                    + "\n"
                )
            resources["rss_peak_bytes"] = max(
                resources.get("rss_peak_bytes", 0),
                rss.memory_info().rss,
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            )
            if adapter.device.type == "cuda":
                resources["cuda_allocated_peak_bytes"] = max(
                    resources.get("cuda_allocated_peak_bytes", 0),
                    torch.cuda.max_memory_allocated(adapter.device),
                )
                resources["cuda_reserved_peak_bytes"] = max(
                    resources.get("cuda_reserved_peak_bytes", 0),
                    torch.cuda.max_memory_reserved(adapter.device),
                )
            done.add(row["image_id"])
            meta.update(processed=len(done), resources=resources)
            if index % 25 == 0:
                write_json(run / "run.json", meta)
                print(f"{meta['model_id']}: {len(done)}/{len(rows)}", flush=True)
        meta.update(
            status="smoke_passed" if meta.get("smoke") else "inference_complete",
            resources=resources,
            processed=len(done),
        )
    except BaseException as error:
        meta.update(
            status="oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "failed",
            processed=len(done),
            error=str(error),
        )
        with errors_path.open("a") as output:
            output.write(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
                + "\n"
            )
        raise
    finally:
        write_json(run / "run.json", meta)
        (run / "summary.md").write_text(
            f"# {meta['model_id']}\n\nСтатус: `{meta['status']}`; обработано {meta.get('processed', 0)}/{len(rows)} изображений. Сокращённый запуск: {meta['shortened']}. Ошибка: {meta.get('error', 'нет')}.\n",
            encoding="utf-8",
        )


def child(command: list[str], lock_fd: int, log: Path) -> int:
    # Inherit the same flock open-file-description: SIGKILL of the parent cannot
    # release exclusivity while a model child is still alive.
    with log.open("a") as output:
        process = subprocess.Popen(
            command,
            pass_fds=(lock_fd,),
            start_new_session=True,
            stdout=output,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
        )
        try:
            return process.wait()
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise


def run_benchmark(
    config_path: Path,
    manifest_path: Path,
    output: Path,
    limit: int | None,
    resume: bool,
    model_id: str | None = None,
    smoke: bool = False,
) -> None:
    config = read_json(config_path)
    validate_config(config)
    if limit is not None and limit < 1:
        raise ValueError("Limit must be positive")
    dataset = read_json(manifest_path)
    if limit:
        dataset["images"] = dataset["images"][:limit]
    if not dataset["images"]:
        raise ValueError("Empty dataset")
    if not smoke and sha256(ROOT / dataset["annotations"]) != dataset["annotations_sha256"]:
        raise ValueError("Annotation hash changed")
    if smoke:
        config.update(warmup=1, repeats=1, bootstrap_samples=0)
    selected = [model_id] if model_id else config["models"]
    if not model_id and len(selected) != 4:
        raise ValueError("The comparison requires exactly four model IDs")
    output.mkdir(parents=True, exist_ok=True)
    source = source_fingerprint()
    identity = {
        "config": config,
        "dataset_hash": object_hash(dataset),
        "source_hash": object_hash(source),
        "model_ids": selected,
        "model_hashes": {m: object_hash(verify_model(m)) for m in selected},
        "image_id": os.environ.get("CONTAINER_IMAGE_ID"),
    }
    experiment_path = output / "experiment.json"
    if experiment_path.exists():
        if not resume or read_json(experiment_path) != identity:
            raise ValueError(
                "Existing experiment differs (config/model/data/source/container); choose a new output-dir"
            )
    else:
        write_json(experiment_path, identity)
        with tarfile.open(output / "source_snapshot.tar.gz", "w:gz") as tar:
            for name in source:
                tar.add(ROOT / name, arcname=name)
    failures = []
    with file_lock(ROOT / "storage/locks/inference.lock") as lock_fd:
        for repeat in range(config["repeats"]):
            order = selected[repeat % len(selected) :] + selected[: repeat % len(selected)]
            for model in order:
                run = output / f"{model}_repeat{repeat + 1}"
                run.mkdir(exist_ok=True)
                status = (
                    read_json(run / "run.json").get("status")
                    if (run / "run.json").exists()
                    else None
                )
                if resume and status in {"complete", "smoke_passed"}:
                    continue
                if not status:
                    write_json(run / "config.json", config)
                    write_json(run / "model_manifest.json", verify_model(model))
                    write_json(run / "dataset_manifest.json", dataset)
                    write_json(
                        run / "run.json",
                        {
                            "schema_version": "1.0",
                            "model_id": model,
                            "repeat": repeat,
                            "config": config,
                            "status": "planned",
                            "shortened": bool(limit) or smoke,
                            "smoke": smoke,
                        },
                    )
                print(
                    f"Starting {model}, repeat {repeat + 1}; log: {run / 'process.log'}", flush=True
                )
                if status != "inference_complete":
                    command = [
                        sys.executable,
                        "-m",
                        "detection.cli",
                        "_worker",
                        "--run-dir",
                        str(run),
                    ]
                    if resume:
                        command.append("--resume")
                    code = child(command, lock_fd, run / "process.log")
                    if code:
                        meta = read_json(run / "run.json")
                        if meta["status"] in {"running", "planned"}:
                            meta.update(
                                status="failed",
                                error=f"Process exited with code {code}; see process.log",
                            )
                            write_json(run / "run.json", meta)
                        failures.append(model)
                        print(f"FAILED {model}: exit={code}; inspect process.log", flush=True)
                        continue
                if not smoke:
                    code = child(
                        [sys.executable, "-m", "detection.cli", "_evaluate", "--run-dir", str(run)],
                        lock_fd,
                        run / "process.log",
                    )
                    if code:
                        failures.append(model)
                        meta = read_json(run / "run.json")
                        meta.update(
                            status="evaluation_failed",
                            error="COCO evaluation failed; see process.log",
                        )
                        write_json(run / "run.json", meta)
                print(f"Finished {model}, repeat {repeat + 1}", flush=True)
    if not smoke:
        from detection.report import build_report

        build_report(output, output / "comparison")
    if failures:
        raise RuntimeError(f"Failed models: {failures}; partial artifacts retained")
