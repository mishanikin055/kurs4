"""Run only in an isolated Compose project with disposable DB and broker."""

import os
from datetime import timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from celery.exceptions import WorkerShutdown
from fastapi.testclient import TestClient
from kombu import Connection
from sqlalchemy import delete, select, text
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError

from backend.main import create_app
from backend.settings import Settings
from backend.tables import STAGES, Image, Job, ModelResult, utcnow
from backend.tasks import TASK_NAME, publish_pending, require_processing_task
from backend.tests.test_storage import image_bytes

pytestmark = pytest.mark.skipif(
    os.environ.get("APP_TEST_INTEGRATION") != "1",
    reason="Requires isolated real PostgreSQL and RabbitMQ (APP_TEST_INTEGRATION=1)",
)


@pytest.fixture
def app(tmp_path):
    settings = Settings(image_root=tmp_path, publisher_interval=0.1)
    application = create_app(settings, publisher_enabled=False)
    with TestClient(application):
        yield application
    # Remove only records written to this test's private storage directory.
    sessions = application.state.sessions
    with sessions.begin() as session:
        image_ids = select(Image.id).where(Image.original_path.startswith(str(tmp_path) + "/"))
        job_ids = select(Job.id).where(Job.image_id.in_(image_ids))
        session.execute(delete(ModelResult).where(ModelResult.job_id.in_(job_ids)))
        session.execute(delete(Job).where(Job.image_id.in_(image_ids)))
        session.execute(delete(Image).where(Image.id.in_(image_ids)))
    sessions.kw["bind"].dispose()


def post_image(app):
    with TestClient(app) as client:
        response = client.post(
            "/images", files={"file": ("photo.png", image_bytes(), "text/plain")}
        )
    assert response.status_code == 202, response.text
    return response.json()


def test_upload_and_get_job(app):
    accepted = post_image(app)
    with TestClient(app) as client:
        response = client.get(accepted["status_url"])
    assert response.status_code == 200
    job = response.json()
    assert job["status"] == "queued"
    assert job["publish_status"] == "pending"
    assert job["processing_attempts"] == 0
    assert {result["stage"] for result in job["results"]} == set(STAGES)
    assert all(result["status"] == "pending" for result in job["results"])
    assert all(result["payload"] is None for result in job["results"])
    assert job["image"]["original_mime"] == "image/png"
    assert "original_path" not in job["image"]
    assert "normalized_path" not in job["image"]


def test_repeated_post_creates_new_job(app):
    first = post_image(app)
    second = post_image(app)
    assert first["image_id"] != second["image_id"]
    assert first["job_id"] != second["job_id"]


def test_not_found_and_invalid_uuid(app):
    with TestClient(app) as client:
        assert client.get(f"/jobs/{uuid4()}").status_code == 404
        assert client.get("/jobs/not-a-uuid").status_code == 422


@pytest.mark.parametrize(
    "data,status",
    [(b"bad data", 422), (image_bytes("GIF"), 415), (image_bytes()[:55], 422)],
)
def test_api_rejects_invalid_image(app, data, status):
    with TestClient(app) as client:
        result = client.post("/images", files={"file": ("bad.png", data, "image/png")})
    assert result.status_code == status


def test_database_failure_removes_only_current_upload(app, monkeypatch):
    first = post_image(app)
    original_sessions = app.state.sessions

    def failure(*args, **kwargs):
        raise OperationalError("redacted", {}, Exception("failure"))

    monkeypatch.setattr("backend.repository.JobRepository.create", failure)
    with TestClient(app) as client:
        response = client.post("/images", files={"file": ("new.png", image_bytes())})
    assert response.status_code == 503
    # Existing files remain; only the failed new upload has been removed.
    with original_sessions() as session:
        saved = session.get(Image, UUID(first["image_id"]))
        assert Path(saved.original_path).exists()
        assert len(list(Path(saved.original_path).parent.parent.iterdir())) == 1
    monkeypatch.undo()


def test_database_uniqueness_and_status_constraints(app):
    accepted = post_image(app)
    sessions = app.state.sessions
    with pytest.raises(IntegrityError), sessions.begin() as session:
        session.add(ModelResult(job_id=UUID(accepted["job_id"]), stage="detection"))
    with pytest.raises(IntegrityError), sessions.begin() as session:
        session.execute(
            text("UPDATE jobs SET status='invented' WHERE id=:id"),
            {"id": accepted["job_id"]},
        )


def test_confirmed_publication_and_message_contract(app):
    accepted = post_image(app)
    celery = app.state.celery
    with Connection(celery.conf.broker_url) as connection:
        queue = connection.SimpleQueue("inference", queue_opts={"durable": True})
        queue.clear()
        try:
            assert publish_pending(app.state.sessions, celery, 3) == 1
            message = queue.get(block=True, timeout=5)
            assert message.headers["task"] == TASK_NAME
            assert message.headers["id"] == accepted["job_id"]
            assert message.payload[0] == [accepted["job_id"]]
            assert message.properties["delivery_mode"] == 2
            message.ack()
            assert publish_pending(app.state.sessions, celery, 3) == 0
        finally:
            queue.close()
    with app.state.sessions() as session:
        job = session.get(Job, UUID(accepted["job_id"]))
        assert job.publish_status == "sent"
        assert job.publish_attempts == 1
        assert job.status == "queued"
        assert job.published_at is not None


def test_failed_publication_backoff_and_recovery(app):
    accepted = post_image(app)
    celery = app.state.celery
    url = celery.conf.broker_url
    celery.conf.broker_url = "amqp://unused:unused@127.0.0.1:1/unused"
    try:
        assert publish_pending(app.state.sessions, celery, 0.2) == 0
    finally:
        celery.conf.broker_url = url
    with app.state.sessions.begin() as session:
        job = session.get(Job, UUID(accepted["job_id"]))
        assert job.publish_status == "pending"
        assert job.publish_error["code"] == "broker_unavailable"
        assert job.next_publish_at > utcnow()
    assert publish_pending(app.state.sessions, celery, 3) == 0
    with app.state.sessions.begin() as session:
        session.get(Job, UUID(accepted["job_id"])).next_publish_at = utcnow() - timedelta(seconds=1)
    with Connection(url) as connection:
        queue = connection.SimpleQueue("inference", queue_opts={"durable": True})
        queue.clear()
        try:
            assert publish_pending(app.state.sessions, celery, 3) == 1
            queue.get(block=True, timeout=5).ack()
        finally:
            queue.close()
    with app.state.sessions() as session:
        job = session.get(Job, UUID(accepted["job_id"]))
        assert job.publish_attempts == 2
        assert job.publish_error is None


def test_publication_does_not_overwrite_processing_status(app, monkeypatch):
    accepted = post_image(app)

    def fast_worker(*args, **kwargs):
        with app.state.sessions.begin() as session:
            session.get(Job, UUID(accepted["job_id"])).status = "running"

    monkeypatch.setattr("backend.tasks.publish_job", fast_worker)
    assert publish_pending(app.state.sessions, app.state.celery, 3) == 1
    with app.state.sessions() as session:
        job = session.get(Job, UUID(accepted["job_id"]))
        assert job.status == "running"
        assert job.publish_status == "sent"


def test_lost_publication_commit_allows_duplicate_delivery(app, monkeypatch):
    accepted = post_image(app)
    sessions = app.state.sessions
    original_begin = sessions.begin
    count = 0

    def fail_second_transaction():
        nonlocal count
        count += 1
        if count == 2:
            raise SQLAlchemyError("Simulated DB failure after broker confirmation")
        return original_begin()

    with Connection(app.state.celery.conf.broker_url) as connection:
        queue = connection.SimpleQueue("inference", queue_opts={"durable": True})
        queue.clear()
        try:
            monkeypatch.setattr(sessions, "begin", fail_second_transaction)
            with pytest.raises(SQLAlchemyError):
                publish_pending(sessions, app.state.celery, 3)
            monkeypatch.undo()
            with sessions.begin() as session:
                job = session.get(Job, UUID(accepted["job_id"]))
                assert job.publish_status == "pending"
                job.next_publish_at = utcnow() - timedelta(seconds=1)
            assert publish_pending(sessions, app.state.celery, 3) == 1
            for _ in range(2):
                message = queue.get(block=True, timeout=5)
                assert message.headers["id"] == accepted["job_id"]
                message.ack()
        finally:
            queue.close()


def test_unimplemented_worker_refuses_to_consume(app):
    class Worker:
        def __init__(self, celery):
            self.app = celery

    with pytest.raises(WorkerShutdown):
        require_processing_task(Worker(app.state.celery))


def test_lifespan_publisher_and_health(tmp_path):
    import time

    application = create_app(Settings(image_root=tmp_path, publisher_interval=0.1))
    try:
        with TestClient(application) as client:
            accepted = client.post("/images", files={"file": ("live.png", image_bytes())}).json()
            for _ in range(100):
                response = client.get(accepted["status_url"])
                if response.json()["publish_status"] == "sent":
                    break
                time.sleep(0.05)
            assert response.json()["publish_status"] == "sent"
            health = client.get("/health")
            assert health.status_code == 200, health.text
            assert health.json()["processing"] == "configured"
    finally:
        with application.state.sessions.begin() as session:
            job_id = UUID(accepted["job_id"])
            session.execute(delete(ModelResult).where(ModelResult.job_id == job_id))
            session.execute(delete(Job).where(Job.id == job_id))
            session.execute(delete(Image).where(Image.id == UUID(accepted["image_id"])))
        application.state.sessions.kw["bind"].dispose()
