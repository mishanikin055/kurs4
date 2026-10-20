"""Parallel annotation and detection/classification, then files and one DB write."""

import fcntl
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager, nullcontext
from pathlib import Path
from threading import Event
from uuid import UUID

from PIL import Image

from backend.repository import ImageRepository
from backend.schemas import AnnotationPayload, ClassificationPayload, DetectionPayload, ImageResult
from backend.settings import Settings, model_specs
from backend.storage import result_directory, save_result
from backend.tables import STAGES


class ProcessingInterrupted(BaseException):
    pass


class LockTimeout(Exception):
    pass


def interrupt(signum, frame):
    raise ProcessingInterrupted


@contextmanager
def inference_lock(path: Path, deadline: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise LockTimeout
                time.sleep(0.1)
        # Never LOCK_UN: a child surviving a killed coordinator must retain the lock.
        yield handle.fileno()


def stop_process_group(process: subprocess.Popen) -> None:
    # Kill descendants even if the direct child already exited.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_stage(
    request: dict,
    settings: Settings,
    lock_fd: int,
    timeout: float,
    *,
    module: str = "backend.models",
    cancelled: Event | None = None,
) -> dict:
    executable = getattr(settings, request["stage"] + "_python")
    with tempfile.TemporaryDirectory(prefix="kurs4-inference-") as directory:
        input_path, output_path = Path(directory) / "input.json", Path(directory) / "output.json"
        input_path.write_text(json.dumps(request, allow_nan=False))
        environment = {
            **os.environ,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OMP_NUM_THREADS": "4",
        }
        # Model logs can contain paths and huge tracebacks; don't persist them in API results.
        process = subprocess.Popen(
            [executable, "-m", module, str(input_path), str(output_path)],
            start_new_session=True,
            pass_fds=(lock_fd,),
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            deadline = time.monotonic() + timeout
            while True:
                if cancelled is not None and cancelled.is_set():
                    raise ProcessingInterrupted
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {"error": {"code": "stage_timeout"}}
                try:
                    process.wait(timeout=min(0.1, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue
            if process.returncode != 0:
                return {"error": {"code": "model_process_failed", "exit_code": process.returncode}}
            if output_path.stat().st_size > 5 * 1024 * 1024:
                return {"error": {"code": "model_output_too_large"}}
            return json.loads(output_path.read_text())
        finally:
            stop_process_group(process)


def validate_payload(
    stage: str, payload: dict, size: tuple[int, int] | None, detection: dict | None
) -> tuple[str, dict]:
    if stage == "detection":
        parsed = DetectionPayload.model_validate(payload)
        if any(obj.bbox[2] > size[0] or obj.bbox[3] > size[1] for obj in parsed.objects):
            raise ValueError("Box outside normalized image")
        return "succeeded", parsed.model_dump(mode="json")
    if stage == "annotation":
        parsed = AnnotationPayload.model_validate(payload)
        if not parsed.raw_caption.strip():
            raise ValueError("Empty caption")
        return "succeeded", parsed.model_dump(mode="json")
    parsed = ClassificationPayload.model_validate(payload)
    expected = {obj["object_id"]: obj for obj in detection["objects"]}
    ids = [str(obj.object_id) for obj in parsed.objects]
    if set(ids) != set(expected) or len(ids) != len(expected):
        raise ValueError("Classification object IDs do not match detection")
    count = sum(obj.label is not None for obj in parsed.objects)
    status = "succeeded" if count == len(parsed.objects) else "partial" if count else "failed"
    return status, parsed.model_dump(mode="json", exclude_none=True)


def process_image(
    image_id: UUID,
    sessions,
    settings: Settings,
    *,
    runner=run_stage,
    lock_fd: int | None = None,
) -> None:
    deadline = time.monotonic() + settings.job_timeout
    destination = result_directory(settings, image_id)
    source = settings.image_root / str(image_id)
    lock = inference_lock(settings.lock_path, deadline) if lock_fd is None else nullcontext(lock_fd)
    with lock as lock_fd:
        with sessions() as session:
            existing = ImageRepository(session).get(image_id)
            if existing is not None:
                # Duplicates must never rerun models or overwrite a completed image.
                if Path(existing.result_path).parent != destination:
                    raise ValueError("Conflicting volume path for completed image")
                shutil.rmtree(source, ignore_errors=True)
                return
        manifest = destination / "results.json"
        if manifest.is_file():
            # If the DB was unavailable after file publication, retry only its write.
            cached_manifest = json.loads(manifest.read_text(encoding="utf-8"))
            result = ImageResult.model_validate(cached_manifest)
            if result.id != image_id or not all(
                (destination / name).is_file()
                for name in ("original", "normalized.png", "detection.png")
            ):
                raise ValueError("Incomplete result bundle")
            cached_crops = cached_manifest.get("crops", [])
            expected_crops = [f"{obj.object_id}.png" for obj in result.detection.objects]
            if "crops" in cached_manifest and (
                cached_crops != expected_crops
                or not all((destination / "crops" / name).is_file() for name in cached_crops)
            ):
                raise ValueError("Incomplete crop bundle")
        else:
            specs = model_specs(settings)
            with Image.open(source / "normalized.png") as image:
                size = image.size
            cancelled = Event()

            def infer(stage: str, detection: dict | None = None) -> dict:
                if cancelled.is_set():
                    raise ProcessingInterrupted
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Inference deadline exceeded")
                request = dict(
                    stage=stage,
                    image_path=str(source / "normalized.png"),
                    model_root=str(settings.model_root),
                    **specs[stage],
                )
                if stage == "classification":
                    request["detection"] = detection
                response = runner(
                    request,
                    settings,
                    lock_fd,
                    min(settings.stage_timeout, remaining),
                    cancelled=cancelled,
                )
                if response.get("error"):
                    raise RuntimeError(f"{stage}: {response['error'].get('code', 'model_error')}")
                status, payload = validate_payload(stage, response["payload"], size, detection)
                if status != "succeeded":
                    raise RuntimeError(f"{stage}: incomplete result")
                return payload

            def detect_and_classify() -> dict:
                detection = infer("detection")
                classification = (
                    infer("classification", detection) if detection["objects"] else {"objects": []}
                )
                return {"detection": detection, "classification": classification}

            payloads = {}
            errors = []
            # Threads only coordinate IPC; CUDA stays in the three model processes.
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="inference") as executor:
                try:
                    pending = {
                        executor.submit(detect_and_classify): None,
                        executor.submit(infer, "annotation"): "annotation",
                    }
                    while pending:
                        done, _ = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                        for future in done:
                            stage = pending.pop(future)
                            try:
                                payload = future.result()
                            except Exception as exc:
                                # Drain the other branch before reusing resident IPC channels.
                                errors.append(exc)
                            else:
                                if stage is None:
                                    payloads.update(payload)
                                else:
                                    payloads[stage] = payload
                finally:
                    # Interrupt/IPC failure must unblock both branches before releasing the lock.
                    cancelled.set()
            if errors:
                raise errors[0]
            result = ImageResult(
                id=image_id,
                width=size[0],
                height=size[1],
                models={
                    stage: {key: specs[stage][key] for key in ("model_name", "model_revision")}
                    for stage in STAGES
                },
                **payloads,
            )
            save_result(
                result,
                source,
                destination,
                crop_padding=specs["classification"]["parameters"].get(
                    "crop_padding_fraction", 0.1
                ),
            )
        # No image, job, status, error or intermediate stage is written before this point.
        with sessions.begin() as session:
            ImageRepository(session).save(result, destination)
        shutil.rmtree(source, ignore_errors=True)
