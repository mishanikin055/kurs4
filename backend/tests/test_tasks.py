"""Worker storage failures retry with UUID-only tasks; no broker or models."""

from contextlib import contextmanager
from unittest.mock import Mock
from uuid import uuid4

import pytest
from celery.exceptions import Retry
from elastic_transport import ConnectionError

from backend.tasks import make_celery, register_processing_task, worker_main


def test_storage_outage_retries_and_closes_clients(repository, monkeypatch):
    settings = repository.settings
    es, s3 = Mock(), Mock()
    monkeypatch.setattr("backend.db.make_elasticsearch", lambda settings: es)
    monkeypatch.setattr("backend.db.make_s3", lambda settings: s3)
    process = Mock(side_effect=ConnectionError("storage offline"))
    monkeypatch.setattr("backend.processing.process_image", process)
    app = make_celery(settings)
    try:
        task = register_processing_task(app, settings)
        retry = Mock(side_effect=Retry())
        monkeypatch.setattr(task, "retry", retry)
        image_id = uuid4()
        with pytest.raises(Retry):
            task.run(str(image_id))
        assert process.call_args.args[0] == image_id
        assert retry.call_args.kwargs["countdown"] == 30
        assert retry.call_args.kwargs["max_retries"] is None
        es.close.assert_called_once()
        s3.close.assert_called_once()
    finally:
        app.close()


@pytest.fixture
def startup(repository, monkeypatch):
    es, s3, app = Mock(), Mock(), Mock()
    monkeypatch.setattr("backend.db.make_elasticsearch", lambda settings: es)
    monkeypatch.setattr("backend.db.make_s3", lambda settings: s3)
    monkeypatch.setattr("backend.repository.ImageRepository", lambda *args: repository)
    monkeypatch.setattr(repository, "health", lambda: {"elasticsearch": True, "s3": True})
    make_app = Mock(return_value=app)
    register = Mock()
    monkeypatch.setattr("backend.tasks.make_celery", make_app)
    monkeypatch.setattr("backend.tasks.register_processing_task", register)
    return repository.settings, app, make_app, register


def test_worker_retries_memory_failure_before_consuming(startup, monkeypatch):
    settings, app, make_app, register = startup
    events = []
    runner = Mock(lock_fd=42)

    @contextmanager
    def runtime(settings):
        events.append("start")
        try:
            if events.count("start") == 1:
                raise MemoryError("host headroom")
            yield runner
        finally:
            events.append("close")

    def sleep(seconds):
        assert seconds == 30
        assert events == ["start", "close"]
        make_app.assert_not_called()
        events.append("wait")

    monkeypatch.setattr("backend.runtime.resident_models", runtime)
    monkeypatch.setattr("time.sleep", sleep)
    worker_main(settings)
    assert events == ["start", "close", "wait", "start", "close"]
    make_app.assert_called_once_with(settings)
    register.assert_called_once_with(app, settings, runner=runner, lock_fd=42)
    app.worker_main.assert_called_once()


def test_worker_does_not_retry_checkpoint_failure(startup, monkeypatch):
    settings, app, make_app, register = startup

    @contextmanager
    def failed_start(settings):
        raise ValueError("checkpoint checksum mismatch")
        yield  # Make the failure happen in __enter__, as with resident_models.

    runtime = Mock(side_effect=failed_start)
    sleep = Mock()
    monkeypatch.setattr("backend.runtime.resident_models", runtime)
    monkeypatch.setattr("time.sleep", sleep)
    with pytest.raises(ValueError, match="checkpoint checksum"):
        worker_main(settings)
    runtime.assert_called_once_with(settings)
    sleep.assert_not_called()
    make_app.assert_not_called()


def test_worker_memory_wait_can_be_interrupted(startup, monkeypatch):
    settings, app, make_app, register = startup
    runtime = Mock(side_effect=MemoryError("host headroom"))
    sleep = Mock(side_effect=SystemExit(0))
    monkeypatch.setattr("backend.runtime.resident_models", runtime)
    monkeypatch.setattr("time.sleep", sleep)
    with pytest.raises(SystemExit):
        worker_main(settings)
    runtime.assert_called_once_with(settings)
    sleep.assert_called_once_with(30)
    make_app.assert_not_called()
