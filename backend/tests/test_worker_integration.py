"""Real PostgreSQL/RabbitMQ and a separate prefork worker; models are synthetic."""

import os
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from kombu import Connection

from backend.processing import ProcessingInterrupted, process_image, run_stage
from backend.repository import JobRepository
from backend.tasks import publish_job
from backend.tests.test_integration import app as app
from backend.tests.test_integration import post_image

pytestmark = pytest.mark.skipif(
    os.environ.get("APP_TEST_INTEGRATION") != "1",
    reason="Requires isolated PostgreSQL and RabbitMQ",
)


def worker(app, *, mode="success", control=None):
    settings = app.state.settings
    env = {
        **os.environ,
        "APP_LOCK_PATH": str(settings.image_root / "inference.lock"),
        "MOCK_MODEL_MODE": mode,
    }
    if control:
        env.update(MOCK_STARTED=str(control / "model_started"), MOCK_SLEEP="2")
    return subprocess.Popen(
        [sys.executable, "-m", "backend.tests.mock_worker"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop_worker(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def wait_job(app, job_id, *, timeout=20):
    end = time.monotonic() + timeout
    with TestClient(app) as client:
        while time.monotonic() < end:
            response = client.get(f"/jobs/{job_id}")
            assert response.status_code == 200, response.text
            job = response.json()
            if job["status"] in {"succeeded", "partial", "failed"}:
                return job
            time.sleep(0.1)
    pytest.fail("Worker did not finish in time")


def clear_queue(app):
    with Connection(app.state.celery.conf.broker_url) as connection:
        queue = connection.SimpleQueue("inference", queue_opts={"durable": True})
        queue.clear()
        queue.close()


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("success", "succeeded"),
        ("empty", "succeeded"),
        ("fail_detection", "partial"),
        ("fail_classification", "partial"),
        ("fail_annotation", "partial"),
    ],
)
def test_real_consumer_pipeline_and_duplicate_delivery(app, mode, expected):
    clear_queue(app)
    accepted = post_image(app)
    consumer = worker(app, mode=mode)
    try:
        publish_job(app.state.celery, accepted["job_id"], 3)
        job = wait_job(app, accepted["job_id"])
        assert job["status"] == expected, job
        assert job["processing_attempts"] == 1
        results = {r["stage"]: r for r in job["results"]}
        if mode == "success":
            obj = results["detection"]["payload"]["objects"][0]
            classified = results["classification"]["payload"]["objects"][0]
            assert classified["object_id"] == obj["object_id"]
            assert classified["crop_bbox"] == [0, 0, 6, 6]
        if mode in {"empty", "fail_detection"}:
            assert results["classification"]["status"] == "skipped"
        if mode != "fail_annotation":
            assert results["annotation"]["payload"]["raw_caption"] == "  Synthetic caption.  "
        publish_job(app.state.celery, accepted["job_id"], 3)
        time.sleep(0.5)
        assert wait_job(app, accepted["job_id"]) == job
    finally:
        stop_worker(consumer)
        clear_queue(app)


def test_recovery_preserves_completed_stage(app):
    accepted = post_image(app)
    settings = app.state.settings.model_copy(
        update={
            "lock_path": app.state.settings.image_root / "test.lock",
            "detection_python": sys.executable,
            "classification_python": sys.executable,
            "annotation_python": sys.executable,
        }
    )
    called = []
    fake = partial(run_stage, module="backend.tests.mock_model")

    def crash(request, *args):
        called.append(request["stage"])
        if request["stage"] == "classification":
            raise ProcessingInterrupted
        return fake(request, *args)

    job_id = UUID(accepted["job_id"])
    with pytest.raises(ProcessingInterrupted):
        process_image(job_id, app.state.sessions, settings, runner=crash)
    with app.state.sessions() as session:
        saved = JobRepository(session).get(job_id)
        detection = next(r.payload for r in saved.results if r.stage == "detection")
    called.clear()

    def recover(request, *args):
        called.append(request["stage"])
        return fake(request, *args)

    process_image(job_id, app.state.sessions, settings, runner=recover)
    job = wait_job(app, job_id)
    assert job["status"] == "succeeded"
    assert job["processing_attempts"] == 2
    assert called == ["classification", "annotation"]
    assert next(r["payload"] for r in job["results"] if r["stage"] == "detection") == detection


def test_exhaustion_and_legacy_checkpoints(app):
    accepted = post_image(app)
    job_id = UUID(accepted["job_id"])
    settings = app.state.settings.model_copy(
        update={"lock_path": app.state.settings.image_root / "test.lock"}
    )
    with app.state.sessions.begin() as session:
        JobRepository(session).get(job_id).processing_attempts = 2
    process_image(
        job_id, app.state.sessions, settings, runner=lambda *args: pytest.fail("Must not run")
    )
    job = wait_job(app, job_id)
    assert job["status"] == "failed"
    assert all(r["error"]["code"] == "attempts_exhausted" for r in job["results"])
    accepted = post_image(app)
    job_id = UUID(accepted["job_id"])
    with app.state.sessions.begin() as session:
        for result in JobRepository(session).get(job_id).results:
            result.model_name = result.model_revision = result.weights_sha256 = None
    process_image(
        job_id, app.state.sessions, settings, runner=lambda *args: pytest.fail("Must not run")
    )
    job = wait_job(app, job_id)
    assert job["status"] == "failed"
    assert {r["error"]["code"] for r in job["results"]} == {
        "model_not_configured",
        "detection_failed",
    }


@pytest.mark.skipif(
    not os.environ.get("APP_TEST_OUTAGE_CONTROL"),
    reason="Requires external DB stop/start controller",
)
def test_database_outage_requeues_uncommitted_result(app):
    control = Path(os.environ["APP_TEST_OUTAGE_CONTROL"])
    clear_queue(app)
    accepted = post_image(app)
    consumer = worker(app, mode="slow", control=control)
    try:
        publish_job(app.state.celery, accepted["job_id"], 3)
        # External controller stops the real DB after mock subprocess starts.
        consumer.wait(timeout=30)
        (control / "worker_stopped").touch()
        end = time.monotonic() + 45
        while not (control / "db_restored").exists():
            assert time.monotonic() < end
            time.sleep(0.1)
        with app.state.sessions() as session:
            job = JobRepository(session).get(UUID(accepted["job_id"]))
            assert job.status == "running"
            assert next(r for r in job.results if r.stage == "detection").payload is None
        consumer = worker(app)
        job = wait_job(app, accepted["job_id"])
        assert job["status"] == "succeeded", job
        assert job["processing_attempts"] == 2
    finally:
        stop_worker(consumer)
        clear_queue(app)


def test_worker_shutdown_requeues_and_cleans_model(app, tmp_path):
    from backend.processing import inference_lock

    clear_queue(app)
    accepted = post_image(app)
    consumer = worker(app, mode="slow", control=tmp_path)
    try:
        publish_job(app.state.celery, accepted["job_id"], 3)
        marker = tmp_path / "model_started"
        deadline = time.monotonic() + 15
        while not marker.exists():
            assert time.monotonic() < deadline
            time.sleep(0.05)
        model_pid = int(marker.read_text())
        consumer.terminate()
        consumer.wait(timeout=10)
        with pytest.raises(ProcessLookupError):
            os.kill(model_pid, 0)
        with inference_lock(app.state.settings.image_root / "inference.lock", time.monotonic() + 1):
            pass
        consumer = worker(app)
        job = wait_job(app, accepted["job_id"])
        assert job["status"] == "succeeded", job
        assert job["processing_attempts"] == 2
    finally:
        stop_worker(consumer)
        clear_queue(app)


def test_killed_coordinator_retains_child_lock_and_recovers(app, tmp_path):
    from backend.processing import LockTimeout, inference_lock

    clear_queue(app)
    accepted = post_image(app)
    consumer = worker(app, mode="slow", control=tmp_path)
    try:
        publish_job(app.state.celery, accepted["job_id"], 3)
        marker = tmp_path / "model_started"
        deadline = time.monotonic() + 15
        while not marker.exists():
            assert time.monotonic() < deadline
            time.sleep(0.05)
        model_pid = int(marker.read_text())
        # The model's parent is the prefork task coordinator, not the main consumer.
        parent_pid = int(Path(f"/proc/{model_pid}/stat").read_text().split()[3])
        os.kill(parent_pid, 9)
        with (
            pytest.raises(LockTimeout),
            inference_lock(
                app.state.settings.image_root / "inference.lock", time.monotonic() + 0.1
            ),
        ):
            pytest.fail("Surviving model lost its lock")
        job = wait_job(app, accepted["job_id"])
        assert job["status"] == "succeeded", job
        assert job["processing_attempts"] == 2
    finally:
        stop_worker(consumer)
        clear_queue(app)


def test_unexpected_coordinator_error_is_persisted(app, monkeypatch):
    accepted = post_image(app)
    settings = app.state.settings.model_copy(
        update={"lock_path": app.state.settings.image_root / "test.lock"}
    )

    def broken_specs(*args):
        raise RuntimeError("Synthetic coordinator error")

    monkeypatch.setattr("backend.processing.model_specs", broken_specs)
    process_image(
        UUID(accepted["job_id"]),
        app.state.sessions,
        settings,
        runner=lambda *args: pytest.fail("Must not run"),
    )
    job = wait_job(app, accepted["job_id"])
    assert job["status"] == "failed"
    assert all(result["error"] == {"code": "internal_error"} for result in job["results"])
