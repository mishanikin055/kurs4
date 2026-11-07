"""Real PostgreSQL/RabbitMQ and a separate Celery consumer, with synthetic models."""

import os
import signal
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path
from uuid import UUID
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from backend.repository import ImageRepository
from backend.schemas import ImageResult
from backend.tables import ImageArtifact, ImageObject, ImageRecord
from backend.tasks import publish_image
from backend.tests.test_integration import app as app
from backend.tests.test_integration import clear_queue, post_image
from backend.tests.test_storage import image_bytes

pytestmark = pytest.mark.skipif(
    os.environ.get("APP_TEST_INTEGRATION") != "1",
    reason="Requires isolated PostgreSQL and RabbitMQ",
)


def worker(app, *, mode="success", control=None, resident=False):
    settings = app.state.settings
    environment = {
        **os.environ,
        "APP_LOCK_PATH": str(settings.lock_path),
        "MOCK_MODEL_MODE": mode,
        "MOCK_RESIDENT": "1" if resident else "0",
    }
    if mode == "parallel":
        environment["MOCK_PARALLEL_DIR"] = str(app.state.test_scratch)
    if control:
        environment.update(MOCK_STARTED=str(control / "model_started"), MOCK_SLEEP="2")
        if mode == "slow_annotation":
            environment["MOCK_SLEEP"] = "30"
        if resident:
            environment["MOCK_RESIDENT_DIR"] = str(control)
    log = app.state.test_scratch / "worker.log"
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
def test_separate_consumer_publishes_binary_bundle_and_catalogue_once(app, resident, mode):
    accepted = post_image(app)
    image_id = UUID(accepted["image_id"])
    consumer = worker(app, mode=mode, resident=resident)
    try:
        saved = wait_result(app, image_id)
        with app.state.test_sessions() as session:
            bundle = ImageRepository(session).bundle(image_id)
        manifest = bundle["results.json"][1]
        parsed = ImageResult.model_validate_json(manifest)
        assert parsed.id == image_id
        assert parsed.annotation.raw_caption == "  Synthetic caption.  "
        assert saved.manifest["annotation"] == {"raw_caption": "  Synthetic caption.  "}
        assert saved.manifest["detection"] == parsed.detection.model_dump(mode="json")
        assert bundle["original"][1] == image_bytes()
        assert bundle["normalized.png"][0] == bundle["detection.png"][0] == "image/png"
        with app.state.test_sessions() as session:
            catalogue = session.scalars(
                select(ImageObject).where(ImageObject.image_id == image_id)
            ).all()
            assert len(catalogue) == len(parsed.detection.objects)
            for obj in catalogue:
                assert obj.label == "person"
                assert obj.object_caption == "  Synthetic object.  "
                assert obj.attributes == {}
                assert bundle[f"crops/{obj.object_id}.png"][0] == "image/png"
        if mode != "empty":
            assert (
                parsed.detection.objects[0].object_id == parsed.classification.objects[0].object_id
            )
        else:
            assert parsed.detection.objects == parsed.classification.objects == []
        publish_image(app.state.celery, str(image_id), 3)
        time.sleep(0.5)
        with app.state.test_sessions() as session:
            assert ImageRepository(session).bundle(image_id) == bundle
            assert (
                session.scalar(
                    select(func.count()).select_from(ImageRecord).where(ImageRecord.id == image_id)
                )
                == 1
            )
        # Browser ZIP contains exactly the immutable database artifacts.
        with TestClient(app) as client:
            response = client.get(f"/images/{image_id}/archive")
            assert response.status_code == 200
            with ZipFile(BytesIO(response.content)) as archive:
                for name, (_, content) in bundle.items():
                    assert archive.read(f"{image_id}/{name}") == content
        with app.state.test_sessions() as session:
            assert ImageRepository(session).bundle(image_id) == bundle
    finally:
        stop_worker(consumer)
        clear_queue(app)


@pytest.mark.parametrize("stage", ["detection", "classification", "annotation"])
def test_model_error_is_logged_without_database_error_or_job_record(app, stage):
    accepted = post_image(app)
    image_id = UUID(accepted["image_id"])
    consumer = worker(app, mode="fail_" + stage)
    log = app.state.test_scratch / "worker.log"
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
            record, artifacts = ImageRepository(session).input(image_id)
            assert record.manifest is None
            assert artifacts["original"] == image_bytes()
            assert (
                session.scalar(
                    select(func.count())
                    .select_from(ImageArtifact)
                    .where(ImageArtifact.image_id == image_id)
                )
                == 2
            )
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
        with app.state.test_sessions() as session:
            assert ImageRepository(session).get(image_id) is None
        consumer = worker(app, resident=True)
        assert wait_result(app, image_id).id == image_id
    finally:
        stop_worker(consumer)
        clear_queue(app)


def test_database_outage_preserves_input_and_requeues_image(app):
    if not os.environ.get("APP_TEST_OUTAGE_CONTROL"):
        pytest.skip("Requires the host-side check_db_outage controller")
    control = Path(os.environ["APP_TEST_OUTAGE_CONTROL"])
    accepted = post_image(app)
    image_id = UUID(accepted["image_id"])
    with app.state.test_sessions() as session:
        _, original_input = ImageRepository(session).input(image_id)
    consumer = worker(app, mode="slow", control=control)
    try:
        assert consumer.wait(timeout=40) == 0
        marker_time = (control / "model_started").stat().st_mtime_ns
        (control / "worker_stopped").touch()
        deadline = time.monotonic() + 30
        while not (control / "db_restored").exists():
            assert time.monotonic() < deadline
            time.sleep(0.1)
        with app.state.test_sessions() as session:
            repository = ImageRepository(session)
            assert repository.get(image_id) is None
            _, restored_input = repository.input(image_id)
            assert restored_input == original_input
            assert (
                session.scalar(
                    select(func.count())
                    .select_from(ImageArtifact)
                    .where(ImageArtifact.image_id == image_id)
                )
                == 2
            )
        consumer = worker(app, mode="slow", control=control)
        assert wait_result(app, image_id).id == image_id
        assert (control / "model_started").stat().st_mtime_ns > marker_time
        with app.state.test_sessions() as session:
            bundle = ImageRepository(session).bundle(image_id)
            assert {name: bundle[name][1] for name in original_input} == original_input
    finally:
        stop_worker(consumer)
