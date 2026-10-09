"""Sequential isolated workers retaining the shared inference lock."""

import importlib.metadata
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from annotation.common import CONFIG, validate_config, verify_model
from detection.common import ROOT, file_lock, object_hash, read_json, sha256, write_json
from detection.runner import child


def source_fingerprint() -> dict[str, str]:
    paths = list((ROOT / "annotation").rglob("*.py")) + list(
        (ROOT / "annotation/configs").glob("*.json")
    )
    paths += list((ROOT / "annotation/evaluation").glob("*.json"))
    paths += list((ROOT / "annotation/evaluation").glob("*.txt"))
    paths += [
        ROOT / p
        for p in [
            "annotation/Dockerfile",
            "annotation/requirements.in",
            "annotation/requirements.lock.txt",
            "scripts/annotation.sh",
            "compose.annotation.yaml",
            "compose.annotation.wsl.yaml",
            "detection/common.py",
            "detection/runner.py",
        ]
    ]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths) if p.exists()}


def environment() -> dict[str, Any]:
    import psutil
    import torch

    return {
        "packages": {
            n: importlib.metadata.version(n)
            for n in ["torch", "transformers", "accelerate", "Pillow", "psutil", "pycocoevalcap"]
        },
        "git_commit": os.getenv("PROJECT_GIT_COMMIT"),
        "git_status": os.getenv("PROJECT_GIT_STATUS"),
        "container_image_id": os.getenv("CONTAINER_IMAGE_ID"),
        "source_files": source_fingerprint(),
        "wsl_ram_total_bytes": psutil.virtual_memory().total,
        "wsl_ram_available_bytes": psutil.virtual_memory().available,
        "cgroup_memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
        "gpu": torch.cuda.get_device_name(0),
        "vram_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "cuda_free_bytes": torch.cuda.mem_get_info()[0],
        "torch_cuda": torch.version.cuda,
    }


def worker(run: Path) -> None:
    import random

    import torch
    from PIL import Image

    from annotation.adapters import Adapter

    meta = read_json(run / "run.json")
    config = meta["config"]
    rows = read_json(run / "dataset.json")["images"]
    samples_path = run / "samples.jsonl"
    errors_path = run / "errors.jsonl"
    samples_path.touch(exist_ok=True)
    errors_path.touch(exist_ok=True)
    try:
        verify_model(meta["model_id"])
        torch.set_num_threads(config["cpu_threads"])
        torch.manual_seed(config["seed"])
        random.seed(config["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        write_json(run / "environment.json", environment())
        start = time.perf_counter()
        adapter = Adapter(meta["model_id"], config)
        adapter.sync()
        meta.update(
            cold_load_ms=(time.perf_counter() - start) * 1000,
            parameters=adapter.parameters,
            status="running",
            decoding={
                **adapter.model.generation_config.to_dict(),
                "max_new_tokens": config["max_new_tokens"],
                "do_sample": False,
                "num_beams": 1,
                "use_cache": True,
            },
        )
        write_json(run / "native_model_config.json", adapter.model.config.to_dict())
        write_json(run / "run.json", meta)
        # Warmup is always a disjoint dev image, including in the test run.
        warmup_row = read_json(ROOT / "annotation/configs/dev100.json")["images"][0]
        if sha256(ROOT / warmup_row["path"]) != warmup_row["sha256"]:
            raise ValueError("Warmup checksum changed")
        with Image.open(ROOT / warmup_row["path"]) as image:
            adapter.predict(image.convert("RGB"))
        for index, source in enumerate(rows):
            if sha256(ROOT / source["path"]) != source["sha256"]:
                raise ValueError("Input checksum changed")
            start = time.perf_counter()
            with Image.open(ROOT / source["path"]) as image:
                if image.size != (source["width"], source["height"]):
                    raise ValueError("Input dimensions changed")
                image = image.convert("RGB")
                read_ms = (time.perf_counter() - start) * 1000
                result = adapter.predict(image)
            result.update(image_id=source["image_id"], read_ms=read_ms)
            result["timing"]["total_with_read_ms"] = result["timing"]["total_ms"] + read_ms
            with samples_path.open("a") as handle:
                handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
            meta["completed_images"] = index + 1
            meta["resources"] = {
                "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "vram_allocated_peak_bytes": torch.cuda.max_memory_allocated(),
                "vram_reserved_peak_bytes": torch.cuda.max_memory_reserved(),
            }
            if index % 10 == 0:
                write_json(run / "run.json", meta)
                print(meta["model_id"], index + 1, "/", len(rows), flush=True)
        meta["status"] = "complete"
    except BaseException as error:
        meta["status"] = "oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "failed"
        meta["error"] = str(error)
        with errors_path.open("a") as handle:
            handle.write(
                json.dumps(
                    {
                        "type": type(error).__name__,
                        "message": str(error),
                        "traceback": traceback.format_exc(),
                    }
                )
                + "\n"
            )
        raise
    finally:
        write_json(run / "run.json", meta)


def benchmark(
    dataset_path: Path, output: Path, model_id: str | None, limit: int | None, mode: str
) -> None:
    config = read_json(CONFIG)
    validate_config(config)
    dataset = read_json(dataset_path)
    if limit:
        if mode == "test":
            raise ValueError("Main test cannot be limited")
        dataset["images"] = dataset["images"][:limit]
    selected = [model_id] if model_id else config["models"]
    records = {m: verify_model(m) for m in selected}
    source = source_fingerprint()
    identity = {
        "config": config,
        "dataset_hash": object_hash(dataset),
        "source_hash": object_hash(source),
        "model_hashes": {m: object_hash(v) for m, v in records.items()},
        "container_image_id": os.getenv("CONTAINER_IMAGE_ID"),
        "mode": mode,
    }
    if output.exists():
        raise ValueError("Output already exists; never rerun or overwrite a main test")
    output.mkdir(parents=True)
    write_json(output / "experiment.json", identity)
    # Freeze the actual executable source with each experiment.
    import tarfile

    with tarfile.open(output / "source_snapshot.tar.gz", "w:gz") as archive:
        for path in source:
            archive.add(ROOT / path, arcname=path)
    failed = []
    with file_lock(ROOT / "storage/locks/inference.lock") as lock_fd:
        for model in selected:
            run = output / model
            run.mkdir()
            write_json(run / "dataset.json", dataset)
            write_json(
                run / "run.json",
                {
                    "model_id": model,
                    "config": config,
                    "identity": identity,
                    "mode": mode,
                    "status": "pending",
                    "expected_images": len(dataset["images"]),
                    "completed_images": 0,
                },
            )
            returncode = child(
                [sys.executable, "-m", "annotation.cli", "_worker", str(run)],
                lock_fd,
                run / "worker.log",
            )
            if returncode:
                meta = read_json(run / "run.json")
                if meta["status"] in {"pending", "running"}:
                    meta.update(
                        status="killed" if returncode == -9 else "failed", returncode=returncode
                    )
                    write_json(run / "run.json", meta)
                failed.append(model)
            print(model, read_json(run / "run.json")["status"], flush=True)
    if failed:
        raise RuntimeError("Incomplete participants: " + ", ".join(failed))
