"""Real PostgreSQL/RabbitMQ and a separate Celery consumer, with synthetic models."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from uuid import UUID

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from backend.repository import ImageRepository
from backend.schemas import ImageResult
from backend.tables import ProcessedImage
from backend.tasks import publish_image
from backend.tests.test_integration import app as app
from backend.tests.test_integration import clear_queue, post_image

pytestmark = pytest.mark.skipif(
    os.environ.get("APP_TEST_INTEGRATION") != "1",
    reason="Requires isolated PostgreSQL and RabbitMQ",
)


def worker(app, *, mode="success", control=None, resident=False):
    settings = app.state.settings
    environment = {
        **os.environ,
        "APP_IMAGE_ROOT": str(settings.image_root),
        "APP_LOCK_PATH": str(settings.image_root / "inference.lock"),
        "MOCK_MODEL_MODE": mode,
        "MOCK_RESIDENT": "1" if resident else "0",
    }
    if mode == "parallel":
        environment["MOCK_PARALLEL_DIR"] = str(settings.image_root)
    if control:
        environment.update(MOCK_STARTED=str(control / "model_started"), MOCK_SLEEP="2")
        if mode == "slow_annotation":
            environment["MOCK_SLEEP"] = "30"
        if resident:
            environment["MOCK_RESIDENT_DIR"] = str(control)
    log = settings.image_root / "worker.log"
    with log.open("ab") as stream:
        return subprocess.Popen(
            [sys.executable, "-m", "backend.tests.mock_worker"],
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )


def stop_worker(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def wait_result(app, image_id, *, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with app.state.test_sessions() as session:
            saved = ImageRepository(session).get(UUID(str(image_id)))
        if saved is not None:
            return saved
        time.sleep(0.1)
    pytest.fail("Worker did not save a completed image; inspect the isolated worker.log")


@pytest.mark.parametrize("resident", [False, True])
@pytest.mark.parametrize("mode", ["success", "empty", "parallel"])
def test_separate_consumer_saves_volume_and_database_once(app, resident, mode):
    accepted = post_image(app)
    image_id = UUID(accepted["image_id"])
    consumer = worker(app, mode=mode, resident=resident)
    try:
        saved = wait_result(app, image_id)
        directory = Path(saved.result_path).parent
        manifest = (directory / "results.json").read_bytes()
        parsed = ImageResult.model_validate_json(manifest)
        assert parsed.id == image_id
        assert parsed.annotation.raw_caption == "  Synthetic caption.  "
        assert saved.annotation == {"raw_caption": "  Synthetic caption.  "}
        assert saved.detection == parsed.detection.model_dump(mode="json")
        assert not (app.state.settings.image_root / str(image_id)).exists()
        if mode != "empty":
            assert (
                parsed.detection.objects[0].object_id == parsed.classification.objects[0].object_id
            )
        else:
            assert parsed.detection.objects == parsed.classification.objects == []
        publish_image(app.state.celery, str(image_id), 3)
        time.sleep(0.5)
        assert (directory / "results.json").read_bytes() == manifest
        with app.state.test_sessions() as session:
            assert (
                session.scalar(
                    select(func.count())
                    .select_from(ProcessedImage)
                    .where(ProcessedImage.id == image_id)
                )
                == 1
            )
        # Explicit browser download never removes or changes the primary volume copy.
        with TestClient(app) as client:
            response = client.get(f"/images/{image_id}/archive")
            assert response.status_code == 200
        assert (directory / "results.json").read_bytes() == manifest
    finally:
        stop_worker(consumer)
        clear_queue(app)


@pytest.mark.parametrize("stage", ["detection", "classification", "annotation"])
def test_model_error_is_logged_without_database_error_or_job_record(app, stage):
    accepted = post_image(app)
    image_id = UUID(accepted["image_id"])
    consumer = worker(app, mode="fail_" + stage)
    log = app.state.settings.image_root / "worker.log"
    try:
        deadline = time.monotonic() + 25
        while time.monotonic() < deadline:
            if log.is_file() and "synthetic_failure" in log.read_text():
                break
            time.sleep(0.1)
        else:
            pytest.fail("Synthetic model error was not logged")
        with app.state.test_sessions() as session:
            assert ImageRepository(session).get(image_id) is None
        assert not (app.state.settings.image_root / "results" / str(image_id)).exists()
        assert (app.state.settings.image_root / str(image_id) / "original").is_file()
    finally:
        stop_worker(consumer)
        clear_queue(app)


@pytest.mark.parametrize("resident", [False, True])
def test_shutdown_interrupts_annotation_and_redelivers_image(app, tmp_path, resident):
    image_id = UUID(post_image(app)["image_id"])
    consumer = worker(app, mode="slow_annotation", control=tmp_path, resident=resident)
    marker = tmp_path / "model_started"
    try:
        deadline = time.monotonic() + 20
        while not marker.exists():
            assert consumer.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.05)
        annotation_pid = int(marker.read_text())
        consumer.terminate()
        assert consumer.wait(timeout=10) == 0
        with pytest.raises(ProcessLookupError):
            os.kill(annotation_pid, 0)
        assert not (app.state.settings.image_root / "results" / str(image_id)).exists()
        with app.state.test_sessions() as session:
            assert ImageRepository(session).get(image_id) is None
        consumer = worker(app, resident=resident)
        assert wait_result(app, image_id).id == image_id
    finally:
        stop_worker(consumer)
        clear_queue(app)


def test_lost_resident_annotator_restarts_consumer_and_redelivers_image(app, tmp_path):
    image_id = UUID(post_image(app)["image_id"])
    consumer = worker(app, mode="slow_annotation", control=tmp_path, resident=True)
    marker = tmp_path / "model_started"
    try:
        deadline = time.monotonic() + 20
        while not marker.exists():
            assert consumer.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.05)
        os.kill(int(marker.read_text()), signal.SIGKILL)
        assert consumer.wait(timeout=10) == 0
        assert not (app.state.settings.image_root / "results" / str(image_id)).exists()
        with app.state.test_sessions() as session:
            assert ImageRepository(session).get(image_id) is None
        consumer = worker(app, resident=True)
        assert wait_result(app, image_id).id == image_id
    finally:
        stop_worker(consumer)
        clear_queue(app)


def test_database_outage_requeues_completed_bundle(app):
    if not os.environ.get("APP_TEST_OUTAGE_CONTROL"):
        pytest.skip("Requires the host-side check_db_outage controller")
    control = Path(os.environ["APP_TEST_OUTAGE_CONTROL"])
    accepted = post_image(app)
    image_id = UUID(accepted["image_id"])
    consumer = worker(app, mode="slow", control=control)
    try:
        consumer.wait(timeout=40)
        manifest = app.state.settings.image_root / "results" / str(image_id) / "results.json"
        before = manifest.read_bytes()
        marker_time = (control / "model_started").stat().st_mtime_ns
        (control / "worker_stopped").touch()
        deadline = time.monotonic() + 30
        while not (control / "db_restored").exists():
            assert time.monotonic() < deadline
            time.sleep(0.1)
        consumer = worker(app, mode="slow", control=control)
        wait_result(app, image_id)
        assert manifest.read_bytes() == before
        assert (control / "model_started").stat().st_mtime_ns == marker_time
    finally:
        stop_worker(consumer)
