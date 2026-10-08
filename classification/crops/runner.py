"""Sequential crop inference with immutable inputs and an inherited GPU lock."""

from __future__ import annotations

import json
import os
import resource
import sys
import tarfile
import time
import traceback
from pathlib import Path

from classification.common import data_path, verify_model
from classification.crops.datasets import CONFIGS, crop_box, mapping
from classification.runner import environment, source_fingerprint, validate_config
from detection.common import ROOT, file_lock, object_hash, read_json, sha256, write_json
from detection.runner import child


def sources() -> dict[str, str]:
    result = source_fingerprint()
    for name in ["scripts/classification_crops.sh", "compose.classification.crops.yaml"]:
        result[name] = sha256(ROOT / name)
    result.update(
        {
            str(p.relative_to(ROOT)): sha256(p)
            for p in sorted((ROOT / "classification/crops").rglob("*"))
            if p.suffix in {".py", ".json"}
        }
    )
    return result


def interpret(predictions: list[dict], detector_category: int | None, config: dict) -> dict:
    category_map = mapping()
    predictions = [{**p, "coco_category_id": category_map[p["index"]]} for p in predictions]
    category = predictions[0]["coco_category_id"]
    margin = predictions[0]["score"] - predictions[1]["score"]
    accepted = (
        category is not None
        and predictions[0]["score"] >= config["accept_score"]
        and margin >= config["accept_margin"]
    )
    status = (
        "not_mappable"
        if category is None
        else "uncertain"
        if not accepted
        else "classified"
        if detector_category is None
        else "agreement"
        if category == detector_category
        else "conflict"
    )
    return {
        "predictions": predictions,
        "mapped_top1": category,
        "accepted": accepted,
        "margin": margin,
        "status": status,
        "detector_category_id": detector_category,
        "effective_category_id": detector_category,
    }


def recover(path: Path, expected: dict[str, dict]) -> set[str]:
    seen = set()
    if not path.exists():
        return seen
    valid_end = 0
    with path.open("rb") as handle:
        while line := handle.readline():
            if not line.endswith(b"\n"):
                break
            row = json.loads(line)
            key = row["sample_id"]
            if (
                key not in expected
                or key in seen
                or row["source_hash"] != object_hash(expected[key])
            ):
                raise ValueError("Resume crop identity changed or duplicated")
            seen.add(key)
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
    rows = dataset["samples"]
    done = set()
    meta.update(status="running", attempts=meta.get("attempts", 0) + 1)
    write_json(run / "run.json", meta)
    (run / "errors.jsonl").touch(exist_ok=True)
    try:
        verify_model(meta["model_id"])
        if resume:
            done = recover(run / "samples.jsonl", {r["sample_id"]: r for r in rows})
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.set_num_threads(config["cpu_threads"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        env = environment()
        env["source_files"] = sources()
        write_json(run / "environment.json", env)
        start = time.perf_counter()
        adapter = Adapter(meta["model_id"], config)
        adapter.sync()
        load_ms = (time.perf_counter() - start) * 1000
        meta.setdefault("load_attempts_ms", []).append(load_ms)
        meta.setdefault("cold_load_ms", load_ms)
        meta["parameters"] = adapter.parameters
        write_json(run / "native_config.json", adapter.native_config())
        with Image.open(data_path(rows[0]["path"])) as image:
            warm_image = image.convert("RGB").crop(rows[0]["crop_xyxy"])
            for _ in range(config["warmup"]):
                adapter.predict(warm_image)
        del warm_image
        resources = meta.get("resources", {})
        checked = set()
        for index, source in enumerate(rows):
            path = data_path(source["path"])
            if path not in checked:
                if sha256(path) != source["sha256"]:
                    raise ValueError("Image content changed")
                checked.add(path)
            if source["sample_id"] in done:
                continue
            start = time.perf_counter()
            with Image.open(path) as image:
                if (
                    image.size != (source["width"], source["height"])
                    or image.width * image.height > 40_000_000
                ):
                    raise ValueError("Image dimensions changed")
                rgb = image.convert("RGB")
            read_ms = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            crop = rgb.crop(source["crop_xyxy"])
            del rgb
            crop_ms = (time.perf_counter() - start) * 1000
            predictions, timing = adapter.predict(crop)
            del crop
            timing.update(read_ms=read_ms, crop_ms=crop_ms)
            record = {
                "schema_version": "1.0",
                **source,
                "source_hash": object_hash(source),
                "model_id": meta["model_id"],
                "model_version_id": meta["model_version_id"],
                "timing": timing,
                **interpret(predictions, source["detector_category_id"], config),
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
                            "sample_id": source["sample_id"],
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
            done.add(source["sample_id"])
            meta.update(processed=len(done), resources=resources)
            if index % 100 == 0:
                write_json(run / "run.json", meta)
                print(f"{meta['model_id']}: {len(done)}/{len(rows)}", flush=True)
        meta.update(status="inference_complete", processed=len(done), resources=resources)
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


def validate(dataset: dict, config: dict) -> None:
    if (
        dataset["mapping_hash"] != object_hash(read_json(CONFIGS / "mapping.json"))
        or dataset["config"] != config
    ):
        raise ValueError("Crop mapping or frozen protocol changed")
    if dataset["mode"] not in {"gt_crops", "detector_crops"} or not dataset["samples"]:
        raise ValueError("Expected nonempty crop manifest")
    ids = set()
    images = {}
    for row in dataset["samples"]:
        if row["sample_id"] in ids:
            raise ValueError("Duplicate crop ID")
        ids.add(row["sample_id"])
        if crop_box(row["bbox_xyxy"], row["width"], row["height"]) != row["crop_xyxy"]:
            raise ValueError("Crop geometry changed")
        path = data_path(row["path"])
        if path in images and images[path] != row["sha256"]:
            raise ValueError("Inconsistent scene image hash")
        if path not in images and sha256(path) != row["sha256"]:
            raise ValueError("Scene image changed")
        images[path] = row["sha256"]


def benchmark(manifest: Path, output: Path, resume: bool, limit: int | None = None) -> None:
    config = read_json(CONFIGS / "benchmark.json")
    validate_config(config)
    dataset = read_json(manifest)
    validate(dataset, config)
    if limit is not None:
        if limit < 1:
            raise ValueError("Limit must be positive")
        dataset["samples"] = dataset["samples"][:limit]
        config = {**config, "warmup": 1, "repeats": 1}
    source = sources()
    manifests = {m: verify_model(m) for m in config["models"]}
    identity = {
        "schema_version": "1.0",
        "config": config,
        "dataset_hash": object_hash(dataset),
        "source_hash": object_hash(source),
        "model_ids": config["models"],
        "model_hashes": {m: object_hash(v) for m, v in manifests.items()},
        "image_id": os.environ.get("CONTAINER_IMAGE_ID"),
        "shortened": limit is not None,
    }
    if not identity["image_id"] or identity["image_id"] == "unrecorded":
        raise ValueError("Record container image ID")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "experiment.json").exists():
        if not resume or read_json(output / "experiment.json") != identity:
            raise ValueError("Experiment identity differs")
    else:
        write_json(output / "experiment.json", identity)
        with tarfile.open(output / "source_snapshot.tar.gz", "w:gz") as tar:
            for name in source:
                tar.add(ROOT / name, arcname=name)
    failures = []
    with file_lock(ROOT / "storage/locks/inference.lock") as lock_fd:
        for repeat in range(config["repeats"]):
            models = config["models"]
            for model in models[repeat % 4 :] + models[: repeat % 4]:
                run = output / f"{model}_repeat{repeat + 1}"
                run.mkdir(exist_ok=True)
                status = (
                    read_json(run / "run.json")["status"] if (run / "run.json").exists() else None
                )
                if resume and status == "complete":
                    if (
                        sha256(run / "samples.jsonl")
                        != read_json(run / "run.json")["samples_sha256"]
                    ):
                        raise ValueError("Completed predictions changed")
                    continue
                if status is None:
                    write_json(run / "dataset_manifest.json", dataset)
                    write_json(run / "config.json", config)
                    write_json(run / "model_manifest.json", manifests[model])
                    write_json(
                        run / "run.json",
                        {
                            "schema_version": "1.0",
                            "model_id": model,
                            "model_version_id": manifests[model]["revision"],
                            "repeat": repeat,
                            "status": "planned",
                            "shortened": limit is not None,
                            "config": config,
                            "mode": dataset["mode"],
                        },
                    )
                print(f"Starting {model}, repeat {repeat + 1}: {run}", flush=True)
                command = [
                    sys.executable,
                    "-m",
                    "classification.crops.cli",
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
                if not code:
                    code = child(
                        [
                            sys.executable,
                            "-m",
                            "classification.crops.cli",
                            "_evaluate",
                            "--run-dir",
                            str(run),
                        ],
                        lock_fd,
                        run / "process.log",
                    )
                meta = read_json(run / "run.json")
                if code:
                    if meta["status"] not in {"failed", "oom"}:
                        meta.update(
                            status="evaluation_failed", error=f"Exit {code}; see process.log"
                        )
                        write_json(run / "run.json", meta)
                    failures.append(model)
                print(f"Finished {model}: {meta['status']}", flush=True)
    if failures:
        raise RuntimeError(f"Failed models: {failures}; artifacts retained")
