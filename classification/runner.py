"""One model child at a time, with the same inherited inference lock as detection."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import resource
import subprocess
import sys
import tarfile
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

from classification.common import CONFIGS, data_path, labels, verify_model
from detection.common import ROOT, file_lock, object_hash, read_json, sha256, write_json
from detection.runner import child


def source_fingerprint() -> dict[str, str]:
    paths = sorted(
        list((ROOT / "classification").glob("*.py"))
        + list(CONFIGS.glob("*.json"))
        + list((ROOT / "classification").glob("*requirements*"))
        + [
            ROOT / p
            for p in [
                "classification/Dockerfile",
                "compose.classification.yaml",
                "compose.classification.wsl.yaml",
                "scripts/classification.sh",
                "detection/common.py",
                "detection/download.py",
                "detection/runner.py",
            ]
        ]
    )
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths if p.is_file()}


def environment() -> dict[str, Any]:
    import psutil
    import torch

    def git(*args: str) -> str | None:
        result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    result = {
        "schema_version": "1.0",
        "python": sys.version,
        "os": platform.platform(),
        "cpu": next(
            (
                s.split(":", 1)[1].strip()
                for s in Path("/proc/cpuinfo").read_text().splitlines()
                if s.startswith("model name")
            ),
            platform.processor(),
        ),
        "host_ram_declared_bytes": 16 * 1024**3,
        "ram_total_bytes": psutil.virtual_memory().total,
        "ram_available_bytes": psutil.virtual_memory().available,
        "cgroup_memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
        "packages": {
            n: importlib.metadata.version(n)
            for n in [
                "torch",
                "torchvision",
                "numpy",
                "Pillow",
                "psutil",
                "huggingface_hub",
                "pyarrow",
            ]
        },
        "git_commit": os.environ.get("PROJECT_GIT_COMMIT") or git("rev-parse", "HEAD"),
        "git_status": os.environ.get("PROJECT_GIT_STATUS", git("status", "--porcelain")),
        "git_state_source": "host wrapper"
        if "PROJECT_GIT_STATUS" in os.environ
        else "container checkout",
        "container_image_id": os.environ.get("CONTAINER_IMAGE_ID"),
        "source_files": source_fingerprint(),
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
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
    if config["batch_size"] != 1 or config["precision"] != "fp32" or config["top_k"] != 5:
        raise ValueError("Only batch=1, FP32, top5 are frozen in this protocol")
    if config["device"] != "cpu" and not config["device"].startswith("cuda:"):
        raise ValueError("Use cpu or cuda:N")
    if min(config["warmup"], config["repeats"], config["cpu_threads"]) < 1:
        raise ValueError("Warmup, repeats and CPU threads must be positive")
    expected = ["resnet50", "efficientnet_v2_s", "convnext_tiny", "vit_b_16"]
    if config["models"] != expected:
        raise ValueError("Participants are fixed by the user")


def recover_samples(path: Path, expected: dict[str, dict]) -> set[str]:
    if not path.exists():
        return set()
    seen = set()
    valid_end = 0
    with path.open("rb") as handle:
        while line := handle.readline():
            if not line.endswith(b"\n"):
                break
            row = json.loads(line)
            image_id = row["image_id"]
            if image_id not in expected or image_id in seen:
                raise ValueError("Resume contains duplicate or unexpected image_id")
            if row["target_index"] != expected[image_id].get("target_index"):
                raise ValueError("Resume ground truth differs")
            seen.add(image_id)
            valid_end = handle.tell()
    with path.open("r+b") as handle:
        handle.truncate(valid_end)
    return seen


def worker(run: Path, resume: bool) -> None:
    import random

    import numpy as np
    import torch
    from PIL import Image

    from classification.adapters import Adapter

    meta = read_json(run / "run.json")
    config = meta["config"]
    dataset = read_json(run / "dataset_manifest.json")
    rows = dataset["images"]
    done: set[str] = set()
    (run / "errors.jsonl").touch(exist_ok=True)
    meta.update(status="running", attempts=meta.get("attempts", 0) + 1)
    write_json(run / "run.json", meta)
    try:
        verify_model(meta["model_id"])
        if resume:
            done = recover_samples(run / "samples.jsonl", {r["image_id"]: r for r in rows})
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.set_num_threads(config["cpu_threads"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        write_json(run / "environment.json", environment())
        start = time.perf_counter()
        adapter = Adapter(meta["model_id"], config)
        adapter.sync()
        load_ms = (time.perf_counter() - start) * 1000
        meta.setdefault("load_attempts_ms", []).append(load_ms)
        meta.setdefault("cold_load_ms", load_ms)
        meta["parameters"] = adapter.parameters
        write_json(run / "native_config.json", adapter.native_config())
        # Peaks include model loading; do not reset the allocator after loading.
        with Image.open(data_path(rows[0]["path"])) as image:
            for _ in range(config["warmup"]):
                adapter.predict(image.convert("RGB"))
        resources = meta.get("resources", {})
        for index, source in enumerate(rows):
            path = data_path(source["path"])
            if sha256(path) != source["sha256"]:
                raise ValueError("Image content changed")
            if source["image_id"] in done:
                continue
            start = time.perf_counter()
            with Image.open(path) as image:
                if (
                    image.size != (source["width"], source["height"])
                    or image.width * image.height > 40_000_000
                ):
                    raise ValueError("Image dimensions changed")
                image = image.convert("RGB")
                read_ms = (time.perf_counter() - start) * 1000
                predictions, timing = adapter.predict(image)
                if meta.get("smoke"):
                    write_json(
                        run / "native_parity.json", adapter.validate_native(image, predictions)
                    )
                    image.save(run / "original.png")
            timing["read_ms"] = read_ms
            record = {
                "schema_version": "1.0",
                "image_id": source["image_id"],
                "model_id": meta["model_id"],
                "model_version_id": meta["model_version_id"],
                "target_index": source.get("target_index"),
                "target_synset": source.get("target_synset"),
                "predictions": predictions,
                "timing": timing,
            }
            start = time.perf_counter()
            with (run / "samples.jsonl").open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            with (run / "write_timings.jsonl").open("a") as handle:
                handle.write(
                    json.dumps(
                        {
                            "image_id": source["image_id"],
                            "write_ms": (time.perf_counter() - start) * 1000,
                        }
                    )
                    + "\n"
                )
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
            done.add(source["image_id"])
            meta.update(processed=len(done), resources=resources)
            if index % 100 == 0:
                write_json(run / "run.json", meta)
                print(f"{meta['model_id']}: {len(done)}/{len(rows)}", flush=True)
        meta.update(
            status="smoke_passed" if meta.get("smoke") else "inference_complete",
            processed=len(done),
            resources=resources,
        )
        meta.pop("error", None)
    except BaseException as error:
        meta.update(
            status="oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "failed",
            processed=len(done),
            error=str(error),
        )
        with (run / "errors.jsonl").open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "attempt": meta["attempts"],
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
                + "\n"
            )
        raise
    finally:
        write_json(run / "run.json", meta)


def validate_dataset(dataset: dict, smoke: bool) -> None:
    rows = dataset["images"]
    if not rows or len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Empty dataset or duplicate image_id")
    if not smoke and (
        dataset["mode"] != "whole_image" or dataset["class_mapping_hash"] != object_hash(labels())
    ):
        raise ValueError(
            "This protocol requires whole-image ImageNet with the frozen class mapping"
        )
    classes = labels()
    if not smoke:
        per_class = {"evaluation5000": 5, "validation": 50}.get(dataset["split"])
        counts = Counter(r["target_index"] for r in rows)
        if (
            per_class is None
            or len(rows) != 1000 * per_class
            or set(counts) != set(range(1000))
            or set(counts.values()) != {per_class}
        ):
            raise ValueError("Expected balanced ImageNet evaluation5000 or complete validation")
    for row in rows:
        path = data_path(row["path"])
        if sha256(path) != row["sha256"]:
            raise ValueError("Dataset image hashes differ")
        if not smoke and classes[row["target_index"]]["synset"] != row["target_synset"]:
            raise ValueError("Target synset and index differ")


def run_benchmark(
    config_path: Path,
    manifest_path: Path,
    output: Path,
    limit: int | None,
    resume: bool,
    smoke: bool = False,
) -> None:
    config = read_json(config_path)
    validate_config(config)
    dataset = read_json(manifest_path)
    validate_dataset(dataset, smoke)
    full_size = len(dataset["images"])
    if limit is not None:
        if limit < 1:
            raise ValueError("Limit must be positive")
        dataset["images"] = dataset["images"][:limit]
    if smoke:
        config.update(warmup=1, repeats=1, bootstrap_samples=0)
    selected = config["models"]
    source = source_fingerprint()
    manifests = {m: verify_model(m) for m in selected}
    identity = {
        "schema_version": "1.0",
        "config": config,
        "dataset_hash": object_hash(dataset),
        "source_hash": object_hash(source),
        "model_ids": selected,
        "model_hashes": {m: object_hash(manifests[m]) for m in selected},
        "image_id": os.environ.get("CONTAINER_IMAGE_ID"),
        "smoke": smoke,
    }
    if not identity["image_id"] or identity["image_id"] == "unrecorded":
        raise ValueError("Record the actual Docker image ID before benchmarking")
    output.mkdir(parents=True, exist_ok=True)
    experiment_path = output / "experiment.json"
    if experiment_path.exists():
        if not resume or read_json(experiment_path) != identity:
            raise ValueError(
                "Experiment differs; config/data/model/source/container must match for resume"
            )
    else:
        write_json(experiment_path, identity)
        with tarfile.open(output / "source_snapshot.tar.gz", "w:gz") as tar:
            for name in source:
                tar.add(ROOT / name, arcname=name)
    failures = []
    with file_lock(ROOT / "storage/locks/inference.lock") as lock_fd:
        for repeat in range(config["repeats"]):
            order = selected[repeat % 4 :] + selected[: repeat % 4]
            for model in order:
                run = output / f"{model}_repeat{repeat + 1}"
                run.mkdir(exist_ok=True)
                meta_path = run / "run.json"
                status = read_json(meta_path)["status"] if meta_path.exists() else None
                if resume and status in {"complete", "smoke_passed"}:
                    if sha256(run / "samples.jsonl") != read_json(meta_path)["samples_sha256"]:
                        raise ValueError("Completed samples changed")
                    continue
                if status is None:
                    write_json(run / "config.json", config)
                    write_json(run / "dataset_manifest.json", dataset)
                    write_json(run / "model_manifest.json", manifests[model])
                    write_json(
                        meta_path,
                        {
                            "schema_version": "1.0",
                            "model_id": model,
                            "model_version_id": manifests[model]["revision"],
                            "repeat": repeat,
                            "config": config,
                            "status": "planned",
                            "smoke": smoke,
                            "shortened": smoke or len(dataset["images"]) < full_size,
                        },
                    )
                print(
                    f"Starting {model}, repeat {repeat + 1}; log: {run / 'process.log'}", flush=True
                )
                command = [
                    sys.executable,
                    "-m",
                    "classification.cli",
                    "_worker",
                    "--run-dir",
                    str(run),
                ]
                if resume:
                    command.append("--resume")
                code = (
                    0
                    if status == "inference_complete"
                    else child(command, lock_fd, run / "process.log")
                )
                if not code and not smoke:
                    code = child(
                        [
                            sys.executable,
                            "-m",
                            "classification.cli",
                            "_evaluate",
                            "--run-dir",
                            str(run),
                        ],
                        lock_fd,
                        run / "process.log",
                    )
                meta = read_json(meta_path)
                if code:
                    if meta["status"] not in {"failed", "oom"}:
                        meta.update(
                            status="evaluation_failed"
                            if meta["status"] == "inference_complete"
                            else "failed",
                            error=f"Exit {code}; see process.log",
                        )
                    write_json(meta_path, meta)
                    failures.append(model)
                elif smoke:
                    meta["samples_sha256"] = sha256(run / "samples.jsonl")
                    write_json(meta_path, meta)
                print(f"Finished {model}, repeat {repeat + 1}: {meta['status']}", flush=True)
    if not smoke:
        from classification.report import build_report

        build_report(output, output / "comparison")
    if failures:
        raise RuntimeError(f"Failed models: {failures}; artifacts retained")
