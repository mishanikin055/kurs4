"""Worker storage failures retry with UUID-only tasks; no broker or models."""

from unittest.mock import Mock
from uuid import uuid4

import pytest
from celery.exceptions import Retry
from elastic_transport import ConnectionError

from backend.tasks import make_celery, register_processing_task


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
