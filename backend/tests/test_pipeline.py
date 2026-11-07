"""CPU inference coordination and Elasticsearch/S3 publication tests; no model weights."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from uuid import uuid4

import pytest
from elastic_transport import ConnectionError

from backend.processing import ProcessingInterrupted, process_image
from backend.repository import ImageRepository
from backend.runtime import ResidentModelFailed
from backend.schemas import ImageResult
from backend.storage import BundleTooLarge, prepare_image
from backend.tests.mock_model import predict
from backend.tests.test_storage import image_bytes, upload


@pytest.fixture
def pipeline(tmp_path, monkeypatch, repository):
    settings = repository.settings
    calls, scratch = [], []
    image_id = uuid4()
    width, height, artifacts = prepare_image(upload(image_bytes()), settings)
    repository.create_input(image_id, width, height, artifacts)
    monkeypatch.setenv("MOCK_MODEL_MODE", "success")

    class Completed:
        def __contains__(self, key):
            return repository.get(key) is not None

        def __bool__(self):
            return image_id in self

        def __getitem__(self, key):
            record = repository.get(key)
            return SimpleNamespace(result=ImageResult.model_validate(record.manifest))

    saved = Completed()

    def runner(request, *args, **kwargs):
        assert not saved
        scratch.append(Path(request["image_path"]).parent)
        calls.append(request["stage"])
        return predict(request)

    return SimpleNamespace(
        settings=settings,
        saved=saved,
        calls=calls,
        repository=repository,
        id=image_id,
        runner=runner,
        repository_type=ImageRepository,
        scratch=scratch,
    )


def test_parallel_branches_then_captions_publish_and_duplicate_skips_models(pipeline):
    p = pipeline
    process_image(p.id, p.repository, p.settings, runner=p.runner)
    assert sorted(p.calls) == ["annotation", "classification", "detection"]
    assert p.calls.index("detection") < p.calls.index("classification")
    assert p.calls.index("annotation") < p.calls.index("classification")
    result = p.saved[p.id].result
    assert result.annotation.raw_caption == "  Synthetic caption.  "
    assert result.classification.objects[0].object_caption == "  Synthetic object.  "
    before = p.repository.bundle(p.id)
    process_image(p.id, p.repository, p.settings, runner=p.runner)
    assert len(p.calls) == 3
    assert p.repository.bundle(p.id) == before
    assert all(not path.exists() for path in p.scratch)
    assert not p.repository._input_path(p.id).exists()


@pytest.mark.parametrize("stage", ["detection", "classification", "annotation"])
def test_model_failure_leaves_only_durable_input(pipeline, monkeypatch, stage):
    p = pipeline
    monkeypatch.setenv("MOCK_MODEL_MODE", "fail_" + stage)
    with pytest.raises(RuntimeError, match=stage):
        process_image(p.id, p.repository, p.settings, runner=p.runner)
    assert not p.saved
    assert set(p.repository.input(p.id)[1]) == {"normalized.png"}
    with pytest.raises(FileNotFoundError):
        p.repository.bundle(p.id)
    assert all(not path.exists() for path in p.scratch)


def test_no_detections_saves_empty_crops_and_classification(pipeline, monkeypatch):
    p = pipeline
    monkeypatch.setenv("MOCK_MODEL_MODE", "empty")
    process_image(p.id, p.repository, p.settings, runner=p.runner)
    assert sorted(p.calls) == ["annotation", "detection"]
    assert p.saved[p.id].result.classification.objects == []
    assert p.repository.get(p.id).manifest["crops"] == []
    assert len(p.repository.bundle(p.id)) == 1


def test_failed_publication_is_not_ready_and_retry_recomputes(pipeline, monkeypatch):
    p = pipeline
    original = p.repository.es.index

    def failure(*args, **kwargs):
        raise ConnectionError("failed before publication")

    monkeypatch.setattr(p.repository.es, "index", failure)
    with pytest.raises(ConnectionError):
        process_image(p.id, p.repository, p.settings, runner=p.runner)
    assert not p.saved
    assert all(not path.exists() for path in p.scratch)
    assert p.repository.input(p.id)[0].manifest is None
    monkeypatch.setattr(p.repository.es, "index", original)
    process_image(p.id, p.repository, p.settings, runner=p.runner)
    assert len(p.calls) == 6
    assert p.id in p.saved


def test_oversized_result_is_never_published(pipeline):
    p = pipeline
    p.settings.max_bundle_bytes = 1
    with pytest.raises(BundleTooLarge):
        process_image(p.id, p.repository, p.settings, runner=p.runner)
    assert not p.saved
    assert all(not path.exists() for path in p.scratch)


def test_annotation_overlaps_detection_but_crops_wait_for_florence(pipeline):
    p = pipeline
    annotation_started, detection_finished, annotation_finished = Event(), Event(), Event()

    def runner(request, *args, **kwargs):
        stage = request["stage"]
        assert request["image_path"].endswith("normalized.png")
        if stage == "annotation":
            assert request["parameters"]["prompt"] == "<DETAILED_CAPTION>"
            annotation_started.set()
            assert detection_finished.wait(3)
            annotation_finished.set()
        elif stage == "detection":
            assert annotation_started.wait(3)
            detection_finished.set()
        else:
            assert annotation_finished.is_set() and detection_finished.is_set()
            assert request["parameters"]["prompt"] == "<CAPTION>"
        return p.runner(request, *args, **kwargs)

    process_image(p.id, p.repository, p.settings, runner=runner)
    assert p.id in p.saved


@pytest.mark.parametrize("last_stage", ["annotation", "classification"])
def test_saving_waits_for_scene_and_all_crops(pipeline, last_stage):
    p = pipeline
    blocked, release = Event(), Event()

    def runner(request, *args, **kwargs):
        if request["stage"] == last_stage:
            blocked.set()
            assert release.wait(3)
        return p.runner(request, *args, **kwargs)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(process_image, p.id, p.repository, p.settings, runner=runner)
        try:
            assert blocked.wait(3)
            assert not future.done()
            assert not p.saved
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
        future = executor.submit(process_image, p.id, p.repository, p.settings, runner=runner)
        try:
            assert failed.wait(3) and blocked.wait(3)
            assert not future.done()
            assert not p.saved
        finally:
            release.set()
        with pytest.raises(RuntimeError, match=failed_stage):
            future.result(timeout=3)
    assert finished.is_set()
    process_image(p.id, p.repository, p.settings, runner=p.runner)
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
        process_image(p.id, p.repository, p.settings, runner=runner)
    assert other_stopped.is_set()
    assert not p.saved


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
        process_image(p.id, p.repository, p.settings, runner=runner)
    assert all(event.is_set() for event in stopped.values())
    assert not p.saved
