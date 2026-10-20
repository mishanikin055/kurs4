"""Use only with isolated PostgreSQL and RabbitMQ; never with production queues."""

import os
from uuid import UUID

import pytest
from celery.exceptions import WorkerShutdown
from fastapi.testclient import TestClient
from kombu import Connection
from sqlalchemy import delete, func, select

from backend.db import make_engine, make_sessions
from backend.main import create_app
from backend.settings import Settings
from backend.tables import ProcessedImage
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
    settings = Settings(image_root=tmp_path)
    application = create_app(settings)
    engine = make_engine(settings.database_url.get_secret_value())
    application.state.test_sessions = make_sessions(engine)
    clear_queue(application)
    try:
        yield application
    finally:
        clear_queue(application)
        with application.state.test_sessions.begin() as session:
            session.execute(
                delete(ProcessedImage).where(
                    ProcessedImage.original_path.startswith(str(tmp_path) + "/")
                )
            )
        engine.dispose()
        application.state.celery.close()


def post_image(app):
    with TestClient(app) as client:
        response = client.post(
            "/images", files={"file": ("photo.png", image_bytes(), "text/plain")}
        )
    assert response.status_code == 202, response.text
    assert set(response.json()) == {"image_id", "output_folder"}
    return response.json()


def test_upload_writes_no_database_records_and_confirms_queue_message(app):
    sessions = app.state.test_sessions
    with sessions() as session:
        before = session.scalar(select(func.count()).select_from(ProcessedImage))
    accepted = post_image(app)
    with sessions() as session:
        after = session.scalar(select(func.count()).select_from(ProcessedImage))
    assert before == after
    image_id = UUID(accepted["image_id"])
    assert (app.state.settings.image_root / str(image_id) / "original").is_file()
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
