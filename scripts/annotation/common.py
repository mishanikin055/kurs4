"""Общий протокол аннотаторов: набор, измерения, метрики и четыре строки отчёта."""

import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
import random
import re
import resource
import signal
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
# Direct execution of scripts/annotation/model.py also needs the vendored evaluator.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODELS = ROOT / "annotation/models"
CONFIG = ROOT / "annotation/configs/benchmark.json"
MODEL_IDS = ("blip_base", "florence2_base_ft", "qwen3_vl_2b", "smolvlm2_500m")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def object_hash(value: Any) -> str:
    normalized = json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    return hashlib.sha256(
        json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def specs() -> dict[str, dict[str, Any]]:
    return {s["id"]: s for s in read_json(ROOT / "annotation/configs/models.json")["models"]}


def verify_model(model_id: str) -> dict[str, Any]:
    spec = specs()[model_id]
    record = read_json(MODELS / "manifest.json")["models"][model_id]
    if record["spec_hash"] != object_hash(spec) or record["status"] != "verified":
        raise ValueError("Model is not verified against pinned registry")
    for item in record["files"]:
        path = MODELS / model_id / item["path"]
        if path.stat().st_size != item["size"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Model checksum mismatch: {path}")
    return record


def validate_config(config: dict[str, Any]) -> None:
    if config["repeats"] != 1 or config["batch_size"] != 1:
        raise ValueError("Exactly one pass and batch size 1 are required")
    if config.get("bootstrap_samples", 0) or config.get("confidence_intervals", False):
        raise ValueError("New confidence intervals are prohibited")
    if config["precision"] != "fp16" or config["do_sample"] or config["num_beams"] != 1:
        raise ValueError("Frozen FP16 greedy decoding required")
    if config["models"] != list(MODEL_IDS) or config["warmup"] != 1:
        raise ValueError("Participants and one warmup are frozen")
    if config["max_new_tokens"] != 96:
        raise ValueError("Frozen token budget required")


@contextmanager
def file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another model owns storage/locks/inference.lock") from error
        # Do not explicitly LOCK_UN: descendants share the open file description.
        # Even SIGKILL of this parent cannot release a living model's inherited lock.
        yield handle.fileno()


def source_fingerprint() -> dict[str, str]:
    paths = list((ROOT / "scripts/annotation").glob("*.py"))
    paths += list((ROOT / "annotation/evaluation").glob("*"))
    paths += [
        ROOT / name
        for name in (
            "annotation/requirements.lock.txt",
            "annotation/requirements.in",
            "annotation/Dockerfile",
            "compose.yaml",
            "compose.gpu.yaml",
        )
    ]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(paths) if p.is_file()}


def comparison_identity(limit: int | None) -> dict[str, Any]:
    config = read_json(CONFIG)
    validate_config(config)
    dataset = read_json(ROOT / "annotation/configs/test500.json")
    if dataset["hash"] != object_hash({k: v for k, v in dataset.items() if k != "hash"}):
        raise ValueError("Dataset manifest hash mismatch")
    rows = dataset["images"][:limit] if limit else dataset["images"]
    if not rows or len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Dataset must contain unique images")
    return {
        "protocol": "annotation-single-pass-v2",
        "config": config,
        "dataset_manifest_sha256": sha256(ROOT / "annotation/configs/test500.json"),
        "dataset_hash": dataset["hash"],
        "image_ids": [r["image_id"] for r in rows],
        "expected_images": len(rows),
        "full_dataset_images": len(dataset["images"]),
        "limit": limit,
        "shortened": limit is not None,
        "model_registry_sha256": sha256(ROOT / "annotation/configs/models.json"),
        "source_files": source_fingerprint(),
        "container_image_id": os.getenv("CONTAINER_IMAGE_ID", "unrecorded"),
    }


def dataset_rows(identity: dict[str, Any]) -> list[dict[str, Any]]:
    path = ROOT / "annotation/configs/test500.json"
    if sha256(path) != identity["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest changed since inference")
    dataset = read_json(path)
    rows = dataset["images"][: identity["expected_images"]]
    if [r["image_id"] for r in rows] != identity["image_ids"]:
        raise ValueError("Dataset order changed")
    for name in ("instances", "captions"):
        if sha256(ROOT / f"data/coco/annotations/{name}_val2017.json") != dataset[f"{name}_sha256"]:
            raise ValueError(f"COCO {name} ground truth changed")
    return rows


def image_path(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if not path.is_relative_to((ROOT / "data").resolve()):
        raise ValueError("Image path escapes data/")
    return path


def memory_events() -> dict[str, int]:
    path = Path("/sys/fs/cgroup/memory.events")
    if not path.exists():
        return {}
    return {
        name: int(value) for name, value in (line.split() for line in path.read_text().splitlines())
    }


def child_status(returncode: int | None, before: dict[str, int], after: dict[str, int]) -> str:
    # A signal exit alone does not prove OOM: require the cgroup's actual OOM counter.
    if returncode == -signal.SIGKILL and after.get("oom_kill", 0) > before.get("oom_kill", 0):
        return "oom"
    return "incomplete"


def sync(torch: Any) -> None:
    torch.cuda.synchronize()


def move_inputs(inputs: Any, device: str, torch: Any) -> Any:
    inputs = inputs.to(device)
    for key, value in inputs.items():
        if value.is_floating_point():
            inputs[key] = value.to(torch.float16)
    return inputs


def decode_result(
    processor: Any,
    ids: Any,
    prompt: str | None,
    rendered: str | None,
    input_tokens: int | None,
    pre_ms: float,
    gen_ms: float,
    torch: Any,
) -> dict[str, Any]:
    start = time.perf_counter()
    raw = processor.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    token_ids = ids.tolist()
    sync(torch)
    post_ms = (time.perf_counter() - start) * 1000
    return {
        "raw_caption": raw,
        "output_token_ids": token_ids,
        "output_tokens": len(token_ids),
        "prompt": prompt,
        "rendered_prompt": rendered,
        "input_tokens": input_tokens,
        "timing": {
            "preprocess_ms": pre_ms,
            "generate_ms": gen_ms,
            "postprocess_ms": post_ms,
            "total_ms": pre_ms + gen_ms + post_ms,
            "tokens_per_second": len(token_ids) / (gen_ms / 1000),
        },
    }


def environment(torch: Any) -> dict[str, Any]:
    import psutil

    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "accelerate", "Pillow", "psutil", "pycocoevalcap")
        },
        "git_commit": os.getenv("PROJECT_GIT_COMMIT") or git("rev-parse", "HEAD"),
        "git_status": os.getenv("PROJECT_GIT_STATUS") or git("status", "--short"),
        "container_image_id": os.getenv("CONTAINER_IMAGE_ID"),
        "container_hostname": os.getenv("HOSTNAME"),
        "wsl_ram_available_bytes": psutil.virtual_memory().available,
        "wsl_ram_total_bytes": psutil.virtual_memory().total,
        "cgroup_memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip(),
        "gpu": torch.cuda.get_device_name(0),
        "torch_cuda": torch.version.cuda,
        "vram_total_bytes": torch.cuda.get_device_properties(0).total_memory,
    }


def worker(model_class: Any, run: Path) -> None:
    import torch
    from PIL import Image

    fd = int(os.environ["KURS4_LOCK_FD"])
    lock_stat = (ROOT / "storage/locks/inference.lock").stat()
    if (os.fstat(fd).st_dev, os.fstat(fd).st_ino) != (lock_stat.st_dev, lock_stat.st_ino):
        raise RuntimeError("Model worker must inherit the actual inference lock")
    meta = read_json(run / "run.json")
    try:
        rows = dataset_rows(meta["identity"])
        meta["model_artifact"] = verify_model(meta["model_id"])
        config = meta["config"]
        torch.set_num_threads(config["cpu_threads"])
        torch.manual_seed(config["seed"])
        random.seed(config["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        meta["environment"] = environment(torch)
        sync(torch)
        started = time.perf_counter()
        model = model_class(config)
        sync(torch)
        meta.update(
            cold_load_ms=(time.perf_counter() - started) * 1000,
            parameters=sum(p.numel() for p in model.model.parameters()),
            status="running",
            native_model_config=model.model.config.to_dict(),
            decoding={
                **model.model.generation_config.to_dict(),
                "max_new_tokens": 96,
                "do_sample": False,
                "num_beams": 1,
                "use_cache": True,
            },
        )
        write_json(run / "run.json", meta)
        warmup = read_json(ROOT / "annotation/configs/dev100.json")["images"][0]
        if warmup["image_id"] in meta["identity"]["image_ids"]:
            raise ValueError("Warmup must be disjoint from evaluated images")
        if sha256(image_path(warmup["path"])) != warmup["sha256"]:
            raise ValueError("Warmup checksum changed")
        with Image.open(image_path(warmup["path"])) as image:
            model.predict(image.convert("RGB"))
        with (run / "samples.jsonl").open("a", encoding="utf-8") as handle:
            for source in rows:
                if sha256(image_path(source["path"])) != source["sha256"]:
                    raise ValueError("Input checksum changed")
                started = time.perf_counter()
                with Image.open(image_path(source["path"])) as image:
                    if image.size != (source["width"], source["height"]):
                        raise ValueError("Input dimensions changed")
                    image = image.convert("RGB")
                    read_ms = (time.perf_counter() - started) * 1000
                    result = model.predict(image)
                result.update(image_id=source["image_id"], read_ms=read_ms)
                result["timing"]["total_with_read_ms"] = result["timing"]["total_ms"] + read_ms
                handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                meta["processed"] += 1
                meta["resources"] = {
                    "rss_peak_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    "vram_allocated_peak_bytes": torch.cuda.max_memory_allocated(),
                    "vram_reserved_peak_bytes": torch.cuda.max_memory_reserved(),
                }
                write_json(run / "run.json", meta)
                print(meta["model_id"], meta["processed"], "/", len(rows), flush=True)
        meta["status"] = "complete"
    except BaseException as error:
        meta.update(
            status="oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "failed",
            error={
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        meta["resources"]["rss_peak_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        if torch.cuda.is_initialized():
            meta["resources"].update(
                vram_allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                vram_reserved_peak_bytes=torch.cuda.max_memory_reserved(),
            )
        write_json(run / "run.json", meta)


def run_child(script: Path, run: Path, lock_fd: int) -> int:
    env = {
        **os.environ,
        "KURS4_MODEL_RUN": str(run),
        "KURS4_LOCK_FD": str(lock_fd),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "NLTK_DATA": str(ROOT / "data/annotation_evaluator/nltk_data"),
    }
    child = subprocess.Popen(
        [sys.executable, str(script)], env=env, pass_fds=(lock_fd,), start_new_session=True
    )
    interrupted: list[int] = []
    stop_started: list[float] = []

    def stop(signum: int, _frame: Any) -> None:
        interrupted.append(signum)
        if not stop_started:
            stop_started.append(time.monotonic())
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        # Never wait in a signal handler: Popen.wait can already own its waitpid lock.

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        while True:
            try:
                returncode = child.wait(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                if stop_started and time.monotonic() - stop_started[0] >= 10:
                    os.killpg(child.pid, signal.SIGKILL)
        if interrupted:
            # Also stop descendants that ignored SIGTERM after their leader exited.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        return returncode or (-interrupted[0] if interrupted else 0)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def read_samples(run: Path, expected_ids: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    meta = read_json(run / "run.json")
    samples = [json.loads(s) for s in (run / "samples.jsonl").read_text().splitlines()]
    if [r["image_id"] for r in samples] != expected_ids or len(set(expected_ids)) != len(
        expected_ids
    ):
        raise ValueError("Missing, duplicate or reordered samples")
    if meta["status"] not in {"complete", "evaluation_failed"} or meta["processed"] != len(samples):
        raise ValueError("Participant incomplete")
    if meta["expected_images"] != len(expected_ids):
        raise ValueError("Expected image count mismatch")
    for sample in samples:
        if not isinstance(sample["raw_caption"], str) or not sample["raw_caption"].strip():
            raise ValueError("Empty/invalid caption")
        if sample["output_tokens"] != len(sample["output_token_ids"]):
            raise ValueError("Token contract mismatch")
    return meta, samples


def quality(
    samples: list[dict[str, Any]], images: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[float]]:
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    gt = {r["image_id"]: [{"caption": c} for c in r["references"]] for r in images}
    predictions = {r["image_id"]: [{"caption": r["raw_caption"]}] for r in samples}
    tokenizer = PTBTokenizer()
    gt_tokens, pred_tokens = tokenizer.tokenize(gt), tokenizer.tokenize(predictions)
    bleu, _ = Bleu(4).compute_score(gt_tokens, pred_tokens)
    cider, per_cider = Cider().compute_score(gt_tokens, pred_tokens)
    rouge, _ = Rouge().compute_score(gt_tokens, pred_tokens)
    chair, details = chair_score(samples, images)
    return (
        {"CIDEr": cider, "BLEU_4": bleu[3], "ROUGE_L": rouge, **chair},
        details,
        per_cider.tolist(),
    )


def chair_score(
    samples: list[dict[str, Any]], images: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import nltk

    from annotation.evaluation.chair_words import CHAIR

    nltk.data.path.insert(0, str(ROOT / "data/annotation_evaluator/nltk_data"))

    # Same word-tokenization, pattern singularization, synonyms and special phrase
    # rules as the authors. val2017 replaces their 2014 train/val loader because
    # every evaluated image and all its GT are contained in this single partition.
    tokenizer_source = read_json(ROOT / "annotation/evaluation/sources.json")["nltk_punkt_tab"][
        "files"
    ]
    for name, digest in tokenizer_source.items():
        if (
            sha256(ROOT / "data/annotation_evaluator/nltk_data/tokenizers/punkt_tab/english" / name)
            != digest
        ):
            raise ValueError("NLTK tokenizer resource changed")
    evaluator = CHAIR([r["image_id"] for r in images], None)
    gt = defaultdict(set)
    instances = read_json(ROOT / "data/coco/annotations/instances_val2017.json")
    categories = {r["id"]: r["name"] for r in instances["categories"]}
    for a in instances["annotations"]:
        gt[a["image_id"]].add(evaluator.inverse_synonym_dict[categories[a["category_id"]]])
    for image in images:
        for caption in image["references"]:
            gt[image["image_id"]].update(evaluator.caption_to_words(caption)[1])
    details = []
    total = hallucinated = sentences = 0
    recalled, gt_count = 0, 0
    for sample in samples:
        objects = evaluator.caption_to_words(sample["raw_caption"])[1]
        incorrect = [o for o in objects if o not in gt[sample["image_id"]]]
        total += len(objects)
        hallucinated += len(incorrect)
        sentences += bool(incorrect)
        recalled += len(set(objects) & gt[sample["image_id"]])
        gt_count += len(gt[sample["image_id"]])
        details.append(
            {
                "image_id": sample["image_id"],
                "objects": objects,
                "gt_objects": sorted(gt[sample["image_id"]]),
                "hallucinated": incorrect,
            }
        )
    return {
        "CHAIRs": sentences / len(samples),
        "CHAIRi": hallucinated / total if total else None,
        "object_recall_micro": recalled / gt_count if gt_count else None,
    }, details


def report(output: Path) -> None:
    import numpy as np

    identity = read_json(output / "comparison.json")
    images = None
    summary = []
    for model_id in MODEL_IDS:
        run = output / model_id
        row = {
            "model": model_id,
            "status": "not_run",
            "processed": 0,
            "expected_images": identity["expected_images"],
            "shortened": identity["shortened"],
        }
        if (run / "run.json").exists():
            meta = read_json(run / "run.json")
            if meta["identity"] != identity:
                raise ValueError("Participants have different dataset/config/source identities")
            row.update(status=meta["status"], processed=meta["processed"])
            if meta.get("error"):
                row["error"] = meta["error"]["message"]
            if meta["status"] in {"pending", "running"}:
                row["status"] = "incomplete"
            if meta["status"] in {"complete", "evaluation_failed"}:
                try:
                    _, samples = read_samples(run, identity["image_ids"])
                except (ValueError, FileNotFoundError) as error:
                    row.update(status="incomplete", error=str(error))
                else:
                    try:
                        if images is None:
                            images = dataset_rows(identity)
                        scores, _, _ = quality(samples, images)
                        measured = dict(scores)
                        for field in (
                            "preprocess_ms",
                            "generate_ms",
                            "postprocess_ms",
                            "total_ms",
                            "total_with_read_ms",
                            "tokens_per_second",
                        ):
                            values = [s["timing"][field] for s in samples]
                            measured[field + "_mean"] = float(np.mean(values))
                            measured[field + "_p50"] = float(np.percentile(values, 50))
                            measured[field + "_p95"] = float(np.percentile(values, 95))
                        measured.update(
                            images_per_second=1000 / measured["total_with_read_ms_mean"],
                            output_tokens_mean=float(
                                np.mean([s["output_tokens"] for s in samples])
                            ),
                            cold_load_ms=meta["cold_load_ms"],
                            **meta["resources"],
                        )
                    except Exception as error:
                        row.update(status="evaluation_failed", error=str(error))
                        meta.update(status="evaluation_failed", evaluation_error=str(error))
                        write_json(run / "run.json", meta)
                    else:
                        row.update(status="complete", **measured)
                        if meta["status"] == "evaluation_failed":
                            meta["status"] = "complete"
                            meta.pop("evaluation_error", None)
                            write_json(run / "run.json", meta)

        summary.append(row)
    fields = list(dict.fromkeys(k for row in summary for k in row))
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    columns = (
        "model",
        "status",
        "processed",
        "CIDEr",
        "BLEU_4",
        "ROUGE_L",
        "CHAIRs",
        "CHAIRi",
        "total_ms_p50",
        "total_ms_p95",
        "images_per_second",
        "cold_load_ms",
        "rss_peak_bytes",
        "vram_allocated_peak_bytes",
        "vram_reserved_peak_bytes",
    )
    lines = [
        "# Сравнение аннотаторов: один проход",
        "",
        f"Набор: {identity['expected_images']} из {identity['full_dataset_images']} изображений. "
        + (
            "**Короткая проверка; не полный benchmark.**"
            if identity["shortened"]
            else "Полный test500."
        ),
        "",
        "| " + " | ".join(columns) + " |",
        "|" + "---|" * len(columns),
    ]
    for row in summary:
        lines.append(
            "| "
            + " | ".join(
                str(round(row[k], 6)) if isinstance(row.get(k), float) else str(row.get(k, "—"))
                for k in columns
            )
            + " |"
        )
    lines += [
        "",
        "Время — мс; память — bytes. RAM — peak RSS модельного процесса; "
        "VRAM — пики PyTorch allocated/reserved, включая загрузку и прогрев, "
        "без памяти драйвера и других процессов. p50/p95 относятся к одному проходу. "
        "images_per_second учитывает чтение, preprocess/generate/postprocess, но не запись JSONL. "
        "Cold load: загрузка локальной модели после проверки SHA и инициализации CUDA; "
        "кэш файлов ОС не сбрасывается.",
        "",
        "CIDEr — исходная шкала evaluator; CHAIRs/CHAIRi — доли описаний/упоминаний. "
        "CHAIR проверяет только словарь объектов COCO. Сравниваются готовые модели, "
        "известно пересечение с COCO при предобучении. Внешний набор и SPICE не использованы. "
        "Человеческая оценка исключена пользователем. Новые CI не вычисляются.",
    ]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    console_columns = (
        "model",
        "status",
        "processed",
        "CIDEr",
        "BLEU_4",
        "ROUGE_L",
        "CHAIRs",
        "CHAIRi",
    )
    print("\n" + " | ".join(console_columns), flush=True)
    for row in summary:
        print(
            " | ".join(
                f"{row[key]:.4f}" if isinstance(row.get(key), float) else str(row.get(key, "—"))
                for key in console_columns
            ),
            flush=True,
        )


def main(model_class: Any, model_id: str) -> None:
    if os.getenv("KURS4_MODEL_RUN"):
        worker(model_class, Path(os.environ["KURS4_MODEL_RUN"]))
        return
    parser = argparse.ArgumentParser(
        description=f"Один проход {model_id}, исходное RGB-изображение"
    )
    parser.add_argument("--run", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", args.run) or args.run in {".", ".."}:
        parser.error("--run must be a simple name (letters/digits/._-)")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    output = ROOT / "reports/runs/annotation" / args.run
    if args.report_only:
        saved = read_json(output / "comparison.json")
        if args.limit is not None and args.limit != saved["limit"]:
            raise ValueError("--limit differs from saved comparison")
        with file_lock(ROOT / "storage/locks/inference.lock"):
            report(output)
        return
    with file_lock(ROOT / "storage/locks/inference.lock") as lock_fd:
        identity = comparison_identity(args.limit)
        if output.exists():
            if read_json(output / "comparison.json") != identity:
                raise ValueError("Run name already belongs to different data/config/limit/source")
        else:
            output.mkdir(parents=True)
            write_json(output / "comparison.json", identity)
        run = output / model_id
        if run.exists():
            raise ValueError("Model result already exists; use a new --run name after interruption")
        run.mkdir()
        (run / "samples.jsonl").touch(exist_ok=False)
        write_json(
            run / "run.json",
            {
                "model_id": model_id,
                "identity": identity,
                "config": identity["config"],
                "status": "pending",
                "processed": 0,
                "expected_images": identity["expected_images"],
                "cold_load_ms": None,
                "resources": {},
            },
        )
        before = memory_events()
        returncode = None
        try:
            returncode = run_child(Path(sys.argv[0]).resolve(), run, lock_fd)
        finally:
            after = memory_events()
            meta = read_json(run / "run.json")
            meta["cgroup_memory_events_before"] = before
            meta["cgroup_memory_events_after"] = after
            if meta["status"] in {"pending", "running"} or (
                returncode and meta["status"] == "complete"
            ):
                meta.update(status=child_status(returncode, before, after), returncode=returncode)
            write_json(run / "run.json", meta)
            # Child exited before evaluation, including on parent interruption.
            report(output)
        if returncode:
            raise SystemExit(abs(returncode) if returncode > 0 else 128 - returncode)
        if read_json(run / "run.json")["status"] == "evaluation_failed":
            raise SystemExit(1)
