"""Sequential subprocess inference with durable stage results and an inherited flock."""

import fcntl
import json
import os
import signal
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError

from backend.repository import JobRepository
from backend.schemas import AnnotationPayload, ClassificationPayload, DetectionPayload
from backend.settings import Settings, model_specs
from backend.tables import STAGES, utcnow

FINAL_RESULTS = {"succeeded", "partial", "failed", "skipped"}
FINAL_JOBS = {"succeeded", "partial", "failed"}


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
            process.wait(timeout=max(0.01, timeout))
            if process.returncode != 0:
                return {"error": {"code": "model_process_failed", "exit_code": process.returncode}}
            if output_path.stat().st_size > 5 * 1024 * 1024:
                return {"error": {"code": "model_output_too_large"}}
            return json.loads(output_path.read_text())
        except subprocess.TimeoutExpired:
            return {"error": {"code": "stage_timeout"}}
        finally:
            stop_process_group(process)


def validate_payload(stage: str, payload: dict, job, detection: dict | None) -> tuple[str, dict]:
    if stage == "detection":
        parsed = DetectionPayload.model_validate(payload)
        if (parsed.image_width, parsed.image_height) != (job.image.width, job.image.height):
            raise ValueError("Wrong coordinate space dimensions")
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
    import math

    for obj in parsed.objects:
        box = expected[str(obj.object_id)]["bbox"]
        crop = (math.floor(box[0]), math.floor(box[1]), math.ceil(box[2]), math.ceil(box[3]))
        if obj.crop_bbox != crop:
            raise ValueError("Classification crop does not match detection")
    count = sum(obj.status == "succeeded" for obj in parsed.objects)
    status = "succeeded" if count == len(parsed.objects) else "partial" if count else "failed"
    return status, parsed.model_dump(mode="json")


def finish_job(job) -> None:
    results = {result.stage: result for result in job.results}
    success = all(
        result.status == "succeeded"
        or (
            result.stage == "classification"
            and result.status == "skipped"
            and result.error == {"code": "no_detections"}
        )
        for result in job.results
    )
    useful = any(result.status in {"succeeded", "partial"} for result in job.results)
    job.status = "succeeded" if success else "partial" if useful else "failed"
    job.current_stage = None
    job.finished_at = utcnow()
    job.error = (
        None
        if success
        else {
            "code": "stage_errors",
            "stages": [
                stage for stage, result in results.items() if result.status in {"failed", "partial"}
            ],
        }
    )


def close_unfinished(job, code: str) -> None:
    for result in job.results:
        if result.status not in FINAL_RESULTS:
            result.status = "failed"
            result.error = {"code": code}
            result.finished_at = utcnow()
    finish_job(job)


def process_image(job_id: UUID, sessions, settings: Settings, *, runner=run_stage) -> None:
    deadline = time.monotonic() + settings.job_timeout
    with sessions() as session:
        job = JobRepository(session).get(job_id)
        if job is None or job.status in FINAL_JOBS:
            return
    with inference_lock(settings.lock_path, deadline) as lock_fd:
        try:
            with sessions.begin() as session:
                job = JobRepository(session).get(job_id)
                if job is None or job.status in FINAL_JOBS:
                    return
                if all(result.status in FINAL_RESULTS for result in job.results):
                    finish_job(job)
                    return
                if job.processing_attempts >= 2:
                    close_unfinished(job, "attempts_exhausted")
                    return
                job.processing_attempts += 1
                job.status = "running"
                job.started_at = job.started_at or utcnow()
            for stage in STAGES:
                with sessions.begin() as session:
                    job = JobRepository(session).get(job_id)
                    results = {result.stage: result for result in job.results}
                    result = results[stage]
                    if result.status in FINAL_RESULTS:
                        continue
                    if time.monotonic() >= deadline:
                        close_unfinished(job, "job_timeout")
                        return
                    detection = results["detection"].payload
                    if stage == "classification" and (
                        results["detection"].status != "succeeded" or not detection["objects"]
                    ):
                        result.status = "skipped"
                        result.error = {
                            "code": "no_detections"
                            if results["detection"].status == "succeeded"
                            else "detection_failed"
                        }
                        result.finished_at = utcnow()
                        continue
                    expected = model_specs(settings)[stage]
                    if any(
                        getattr(result, key) != expected[key]
                        for key in ("model_name", "model_revision", "weights_sha256")
                    ):
                        result.status = "failed"
                        result.error = {"code": "model_not_configured"}
                        result.finished_at = utcnow()
                        continue
                    result.status = "running"
                    result.started_at = utcnow()
                    job.current_stage = stage
                    request = dict(
                        stage=stage,
                        image_path=job.image.normalized_path,
                        model_root=str(settings.model_root),
                        parameters=result.parameters,
                        model_name=result.model_name,
                        model_revision=result.model_revision,
                        weights_sha256=result.weights_sha256,
                    )
                    if stage == "classification":
                        request["detection"] = detection
                started = time.monotonic()
                try:
                    response = runner(
                        request, settings, lock_fd, min(settings.stage_timeout, deadline - started)
                    )
                    error = response.get("error")
                    if error:
                        status, payload = "failed", None
                    else:
                        status, payload = validate_payload(
                            stage, response["payload"], job, detection
                        )
                        error = {"code": "crop_errors"} if status in {"partial", "failed"} else None
                except Exception as exc:
                    status, payload = "failed", None
                    error = {"code": "internal_error", "type": type(exc).__name__}
                # DB errors deliberately escape: never ACK an uncommitted result.
                with sessions.begin() as session:
                    result = next(
                        r for r in JobRepository(session).get(job_id).results if r.stage == stage
                    )
                    result.status, result.payload, result.error = status, payload, error
                    result.finished_at = utcnow()
                    result.duration_ms = (time.monotonic() - started) * 1000
            with sessions.begin() as session:
                finish_job(JobRepository(session).get(job_id))
        except SQLAlchemyError:
            raise
        except Exception:
            # Persist unexpected coordinator failures while still holding the lock.
            # A failed commit here also escapes to the task's Reject/requeue path.
            with sessions.begin() as session:
                job = JobRepository(session).get(job_id)
                if job is not None and job.status not in FINAL_JOBS:
                    close_unfinished(job, "internal_error")
