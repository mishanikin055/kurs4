"""CPU-only inference pipeline checks; no model weights or database server."""

import json
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from threading import Event
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy.exc import SQLAlchemyError

from backend.processing import ProcessingInterrupted, process_image
from backend.runtime import ResidentModelFailed
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
                "crops",
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

    def runner(request, *args, **kwargs):
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
    assert sorted(p.calls) == ["annotation", "classification", "detection"]
    assert p.calls.index("detection") < p.calls.index("classification")
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
    assert sorted(p.calls) == ["annotation", "detection"]
    assert p.saved[p.id].result.classification.objects == []
    assert (p.output / "crops").is_dir()
    assert list((p.output / "crops").iterdir()) == []
    assert json.loads((p.output / "results.json").read_text())["crops"] == []


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
    assert sorted(p.calls) == ["annotation", "classification", "detection"]
    assert (p.output / "results.json").read_bytes() == manifest
    assert p.id in p.saved


@pytest.mark.parametrize("function", ["save_detection_image", "crop_object"])
def test_disk_failure_never_commits_result_or_exposes_partial_bundle(
    pipeline, monkeypatch, function
):
    p = pipeline

    def failure(*args):
        raise OSError("disk full")

    monkeypatch.setattr(f"backend.storage.{function}", failure)
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


@pytest.mark.parametrize("missing", ["file", "manifest_entry"])
def test_cached_bundle_with_missing_crop_is_not_committed(pipeline, monkeypatch, missing):
    p = pipeline

    def failure(*args):
        raise SQLAlchemyError("database unavailable")

    monkeypatch.setattr(p.repository, "save", failure)
    with pytest.raises(SQLAlchemyError):
        process_image(p.id, p.sessions, p.settings, runner=p.runner)
    manifest = p.output / "results.json"
    data = json.loads(manifest.read_text())
    if missing == "file":
        (p.output / "crops" / data["crops"][0]).unlink()
    else:
        data["crops"] = []
        manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Incomplete crop bundle"):
        process_image(p.id, p.sessions, p.settings, runner=p.runner)
    assert len(p.calls) == 3
    assert not p.saved


def test_annotation_overlaps_detection_and_classification(pipeline):
    p = pipeline
    annotation_started = Event()
    classification_finished = Event()

    def runner(request, *args, **kwargs):
        stage = request["stage"]
        if stage == "annotation":
            annotation_started.set()
            assert classification_finished.wait(3), "Annotation must overlap the other branch"
        else:
            assert annotation_started.wait(3), "Detection must not block annotation startup"
        response = p.runner(request, *args, **kwargs)
        if stage == "classification":
            classification_finished.set()
        return response

    process_image(p.id, p.sessions, p.settings, runner=runner)
    assert p.id in p.saved


@pytest.mark.parametrize("last_stage", ["annotation", "classification"])
def test_saving_waits_for_both_branches(pipeline, last_stage):
    p = pipeline
    blocked, other_finished, release = Event(), Event(), Event()
    other_stage = "classification" if last_stage == "annotation" else "annotation"

    def runner(request, *args, **kwargs):
        stage = request["stage"]
        if stage == last_stage:
            blocked.set()
            assert release.wait(3)
        response = p.runner(request, *args, **kwargs)
        if stage == other_stage:
            other_finished.set()
        return response

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(process_image, p.id, p.sessions, p.settings, runner=runner)
        try:
            assert blocked.wait(3) and other_finished.wait(3)
            assert not future.done()
            assert not p.saved and not p.output.exists()
        finally:
            release.set()
        future.result(timeout=3)
    assert p.id in p.saved


@pytest.mark.parametrize("failed_stage", ["detection", "annotation"])
def test_model_error_drains_the_other_branch_before_returning(pipeline, failed_stage):
    p = pipeline
    failed, blocked, release, finished = Event(), Event(), Event(), Event()
    other_stage = "annotation" if failed_stage == "detection" else "detection"

    def runner(request, *args, **kwargs):
        if request["stage"] == failed_stage:
            failed.set()
            return {"error": {"code": "synthetic_failure"}}
        if request["stage"] == other_stage:
            blocked.set()
            assert release.wait(3)
            finished.set()
        return p.runner(request, *args, **kwargs)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(process_image, p.id, p.sessions, p.settings, runner=runner)
        try:
            assert failed.wait(3) and blocked.wait(3)
            assert not future.done()
            assert not p.saved and not p.output.exists()
        finally:
            release.set()
        with pytest.raises(RuntimeError, match=failed_stage):
            future.result(timeout=3)
    assert finished.is_set()
    process_image(p.id, p.sessions, p.settings, runner=p.runner)
    assert p.id in p.saved


@pytest.mark.parametrize("failed_stage", ["detection", "annotation"])
def test_fatal_model_failure_cancels_the_other_branch(pipeline, failed_stage):
    p = pipeline
    other_started, other_stopped = Event(), Event()

    def runner(request, *args, cancelled):
        if request["stage"] == failed_stage:
            assert other_started.wait(3)
            raise ResidentModelFailed
        other_started.set()
        assert cancelled.wait(3), "Fatal failure must interrupt the other model request"
        other_stopped.set()
        raise ResidentModelFailed

    with pytest.raises(ResidentModelFailed):
        process_image(p.id, p.sessions, p.settings, runner=runner)
    assert other_stopped.is_set()
    assert not p.saved and not p.output.exists()


def test_coordinator_interrupt_stops_both_branches(pipeline, monkeypatch):
    p = pipeline
    started = {stage: Event() for stage in ("detection", "annotation")}
    stopped = {stage: Event() for stage in started}

    def runner(request, *args, cancelled):
        stage = request["stage"]
        started[stage].set()
        assert cancelled.wait(3)
        stopped[stage].set()
        raise ProcessingInterrupted

    def interrupt(*args, **kwargs):
        assert all(event.wait(3) for event in started.values())
        raise ProcessingInterrupted

    monkeypatch.setattr("backend.processing.wait", interrupt)
    with pytest.raises(ProcessingInterrupted):
        process_image(p.id, p.sessions, p.settings, runner=runner)
    assert all(event.is_set() for event in stopped.values())
    assert not p.saved and not p.output.exists()
