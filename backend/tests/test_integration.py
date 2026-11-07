"""Use only with isolated PostgreSQL and RabbitMQ; never with production queues."""

import os
from uuid import UUID

import pytest
from celery.exceptions import WorkerShutdown
from fastapi.testclient import TestClient
from kombu import Connection
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError

from backend.db import make_engine, make_sessions
from backend.main import create_app
from backend.repository import ImageRepository
from backend.schemas import ImageResult
from backend.settings import Settings
from backend.tables import ImageArtifact, ImageObject, ImageRecord
from backend.tasks import TASK_NAME, require_processing_task
from backend.tests.test_storage import image_bytes

pytestmark = pytest.mark.skipif(
    os.environ.get("APP_TEST_INTEGRATION") != "1",
    reason="Requires isolated real PostgreSQL and RabbitMQ (APP_TEST_INTEGRATION=1)",
)


def clear_queue(app):
    with Connection(app.state.celery.conf.broker_url) as connection:
        with connection.SimpleQueue("inference", queue_opts={"durable": True}) as queue:
            queue.clear()


@pytest.fixture
def app(tmp_path):
    settings = Settings(lock_path=tmp_path / "inference.lock")
    application = create_app(settings)
    engine = make_engine(settings.database_url.get_secret_value())
    application.state.test_sessions = make_sessions(engine)
    application.state.test_scratch = tmp_path
    application.state.test_image_ids = []
    clear_queue(application)
    try:
        yield application
    finally:
        clear_queue(application)
        with application.state.test_sessions.begin() as session:
            session.execute(
                delete(ImageRecord).where(ImageRecord.id.in_(application.state.test_image_ids))
            )
        engine.dispose()
        application.state.celery.close()


def post_image(app):
    with TestClient(app) as client:
        response = client.post(
            "/images", files={"file": ("photo.png", image_bytes(), "text/plain")}
        )
    assert response.status_code == 202, response.text
    assert set(response.json()) == {"image_id"}
    app.state.test_image_ids.append(UUID(response.json()["image_id"]))
    return response.json()


def test_upload_commits_durable_input_before_confirmed_queue_message(app):
    sessions = app.state.test_sessions
    with sessions() as session:
        before = session.scalar(select(func.count()).select_from(ImageRecord))
    accepted = post_image(app)
    with sessions() as session:
        after = session.scalar(select(func.count()).select_from(ImageRecord))
        image_id = UUID(accepted["image_id"])
        repository = ImageRepository(session)
        assert repository.get(image_id) is None
        record, artifacts = repository.input(image_id)
        assert record.manifest is None
        assert (record.width, record.height) == (16, 12)
        assert artifacts["original"] == image_bytes()
        assert artifacts["normalized.png"].startswith(b"\x89PNG")
        assert (
            session.scalar(
                select(func.count())
                .select_from(ImageArtifact)
                .where(ImageArtifact.image_id == image_id)
            )
            == 2
        )
    assert after == before + 1
    image_id = UUID(accepted["image_id"])
    with Connection(app.state.celery.conf.broker_url) as connection:
        with connection.SimpleQueue("inference", queue_opts={"durable": True}) as queue:
            message = queue.get(block=True, timeout=5)
            assert message.headers["task"] == TASK_NAME
            assert message.headers["id"] == str(image_id)
            assert message.payload[0] == [str(image_id)]
            assert message.properties["delivery_mode"] == 2
            message.ack()


def test_unregistered_worker_refuses_to_consume(app):
    class Worker:
        def __init__(self, celery):
            self.app = celery

    with pytest.raises(WorkerShutdown):
        require_processing_task(sender=Worker(app.state.celery))


def test_failed_binary_publication_rolls_back_manifest_and_keeps_durable_input(app):
    image_id = UUID(post_image(app)["image_id"])
    sessions = app.state.test_sessions
    with sessions() as session:
        _, inputs = ImageRepository(session).input(image_id)
    result = ImageResult(
        id=image_id,
        width=16,
        height=12,
        models={
            stage: {"model_name": "synthetic"}
            for stage in ("detection", "classification", "annotation")
        },
        detection={"objects": []},
        classification={"objects": []},
        annotation={"raw_caption": "Synthetic scene."},
    )
    with pytest.raises(IntegrityError):
        with sessions.begin() as session:
            ImageRepository(session).publish(
                result,
                {**inputs, "detection.png": None},
                result.model_dump(mode="json"),
            )
    with sessions() as session:
        repository = ImageRepository(session)
        assert repository.get(image_id) is None
        record, restored_input = repository.input(image_id)
        assert record.manifest is None
        assert restored_input == inputs
        assert (
            session.scalar(
                select(func.count())
                .select_from(ImageObject)
                .where(ImageObject.image_id == image_id)
            )
            == 0
        )
