"""CPU-only inference pipeline checks; no model weights or database server."""

import json
from contextlib import nullcontext
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy.exc import SQLAlchemyError

from backend.processing import process_image
from backend.schemas import ImageResult
from backend.settings import Settings
from backend.storage import save_image
from backend.tests.mock_model import predict
from backend.tests.test_storage import image_bytes, upload


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    settings = Settings(
        database_url="postgresql+psycopg://unused:unused@localhost/unused",
        broker_url="amqp://unused:unused@localhost/unused",
        image_root=tmp_path / "images",
        lock_path=tmp_path / "inference.lock",
    )
    saved = {}
    calls = []

    class Repository:
        def __init__(self, session):
            pass

        def get(self, image_id):
            return saved.get(image_id)

        def save(self, result, directory):
            # A record is never created before all model outputs and files are complete.
            assert set(path.name for path in directory.iterdir()) == {
                "original",
                "normalized.png",
                "detection.png",
                "results.json",
            }
            assert (
                ImageResult.model_validate_json((directory / "results.json").read_text()) == result
            )
            saved[result.id] = SimpleNamespace(
                result=result, result_path=str(directory / "detection.png")
            )

    class Sessions:
        def __call__(self):
            return nullcontext(None)

        def begin(self):
            return nullcontext(None)

    monkeypatch.setattr("backend.processing.ImageRepository", Repository)
    monkeypatch.setenv("MOCK_MODEL_MODE", "success")
    image_id = uuid4()
    save_image(upload(image_bytes()), image_id, settings)

    def runner(request, *args):
        assert not saved
        calls.append(request["stage"])
        return predict(request)

    return SimpleNamespace(
        settings=settings,
        saved=saved,
        calls=calls,
        sessions=Sessions(),
        id=image_id,
        runner=runner,
        repository=Repository,
        output=settings.image_root / "results" / str(image_id),
    )


def test_three_models_then_volume_then_database(pipeline):
    p = pipeline
    process_image(p.id, p.sessions, p.settings, runner=p.runner)
    assert p.calls == ["detection", "classification", "annotation"]
    result = p.saved[p.id].result
    assert result.annotation.raw_caption == "  Synthetic caption.  "
    assert result.detection.objects[0].object_id == result.classification.objects[0].object_id
    assert not (p.settings.image_root / str(p.id)).exists()
    original = (p.output / "original").read_bytes()
    process_image(p.id, p.sessions, p.settings, runner=p.runner)
    assert len(p.calls) == 3
    assert (p.output / "original").read_bytes() == original


@pytest.mark.parametrize("stage", ["detection", "classification", "annotation"])
def test_failure_never_creates_database_or_success_bundle(pipeline, monkeypatch, stage):
    p = pipeline
    monkeypatch.setenv("MOCK_MODEL_MODE", "fail_" + stage)
    with pytest.raises(RuntimeError, match=stage):
        process_image(p.id, p.sessions, p.settings, runner=p.runner)
    assert not p.saved
    assert not p.output.exists()
    assert (p.settings.image_root / str(p.id) / "original").exists()


def test_no_detections_still_runs_caption_and_saves_empty_classification(pipeline, monkeypatch):
    p = pipeline
    monkeypatch.setenv("MOCK_MODEL_MODE", "empty")
    process_image(p.id, p.sessions, p.settings, runner=p.runner)
    assert p.calls == ["detection", "annotation"]
    assert p.saved[p.id].result.classification.objects == []


def test_database_failure_keeps_volume_and_retry_does_not_run_models(pipeline, monkeypatch):
    p = pipeline
    original = p.repository.save

    def failure(*args):
        raise SQLAlchemyError("database unavailable after inference")

    monkeypatch.setattr(p.repository, "save", failure)
    with pytest.raises(SQLAlchemyError):
        process_image(p.id, p.sessions, p.settings, runner=p.runner)
    manifest = (p.output / "results.json").read_bytes()
    assert not p.saved
    monkeypatch.setattr(p.repository, "save", original)
    process_image(p.id, p.sessions, p.settings, runner=p.runner)
    assert p.calls == ["detection", "classification", "annotation"]
    assert (p.output / "results.json").read_bytes() == manifest
    assert p.id in p.saved


def test_disk_failure_never_commits_result_or_exposes_partial_bundle(pipeline, monkeypatch):
    p = pipeline

    def failure(*args):
        raise OSError("disk full")

    monkeypatch.setattr("backend.storage.save_detection_image", failure)
    with pytest.raises(OSError):
        process_image(p.id, p.sessions, p.settings, runner=p.runner)
    assert not p.saved
    assert not p.output.exists()
    assert not list(p.output.parent.glob(".result-*"))


def test_corrupt_cached_bundle_is_not_saved_to_database(pipeline):
    p = pipeline
    p.output.mkdir(parents=True)
    (p.output / "results.json").write_text(json.dumps({"id": str(p.id)}))
    with pytest.raises(ValueError):
        process_image(p.id, p.sessions, p.settings, runner=p.runner)
    assert not p.saved
    assert not p.calls
