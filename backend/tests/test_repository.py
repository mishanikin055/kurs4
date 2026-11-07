"""Publication failures keep inputs; completed results contain only metadata and crops."""

import json
from uuid import uuid4

import pytest
from elastic_transport import ConnectionError

from backend.schemas import ImageResult
from backend.storage import build_artifacts, prepare_image
from backend.tests.test_storage import image_bytes, upload


def prepared(repository, tmp_path):
    image_id, object_id = uuid4(), uuid4()
    width, height, inputs = prepare_image(upload(image_bytes()), repository.settings)
    repository.create_input(image_id, width, height, inputs)
    (tmp_path / "normalized.png").write_bytes(inputs["normalized.png"])
    result = ImageResult(
        id=image_id,
        width=width,
        height=height,
        models={stage: {} for stage in ("detection", "classification", "annotation")},
        annotation={"raw_caption": "  A scene.  "},
        detection={"objects": [{"object_id": object_id, "bbox": [0, 0, 8, 8], "label": "cat"}]},
        classification={"objects": [{"object_id": object_id, "object_caption": "  A cat.  "}]},
    )
    manifest, artifacts = build_artifacts(result, tmp_path, repository.settings)
    return result, manifest, artifacts


def test_only_crops_in_s3_metadata_in_es_and_no_overwrite(repository, tmp_path):
    result, manifest, artifacts = prepared(repository, tmp_path)
    repository.publish(result, artifacts, manifest)
    assert len(repository.s3.objects) == 1
    assert all("/crops/" in key and key.endswith(".png") for key in repository.s3.objects)
    source = repository.es.documents[str(result.id)]["_source"]
    assert json.loads(json.dumps(source))["manifest"] == manifest
    assert source["manifest"]["annotation"]["raw_caption"] == "  A scene.  "
    assert source["manifest"]["classification"]["objects"][0]["object_caption"] == "  A cat.  "
    before = dict(repository.s3.objects)
    repository.publish(result, artifacts, manifest)
    assert repository.s3.objects == before
    assert set(repository.bundle(result.id)) == set(artifacts)


@pytest.mark.parametrize("failure", ["s3", "elasticsearch"])
def test_failure_never_publishes_and_retains_input(repository, tmp_path, monkeypatch, failure):
    result, manifest, artifacts = prepared(repository, tmp_path)

    def fail(**kwargs):
        raise ConnectionError("synthetic outage")

    target, method = (repository.s3, "put_object") if failure == "s3" else (repository.es, "index")
    with monkeypatch.context() as patch:
        patch.setattr(target, method, fail)
        with pytest.raises(ConnectionError):
            repository.publish(result, artifacts, manifest)
        assert repository.get(result.id) is None
        assert set(repository.input(result.id)[1]) == {"normalized.png"}
    repository.publish(result, artifacts, manifest)
    assert repository.get(result.id).manifest == manifest


def test_missing_or_corrupt_crop_does_not_become_a_successful_download(repository, tmp_path):
    result, manifest, artifacts = prepared(repository, tmp_path)
    repository.publish(result, artifacts, manifest)
    name = next(name for name in artifacts if name.startswith("crops/"))
    key = next(iter(repository.s3.objects))
    repository.s3.objects[key] = b"corrupt"
    with pytest.raises(OSError, match="integrity"):
        repository.artifact(result.id, name)
    repository.s3.objects.clear()
    with pytest.raises(OSError, match="Missing stored crop"):
        repository.artifact(result.id, name)


def test_lost_publish_confirmation_can_be_recovered_without_overwriting(
    repository, tmp_path, monkeypatch
):
    result, manifest, artifacts = prepared(repository, tmp_path)
    original = repository.es.index

    def publish_then_timeout(**kwargs):
        original(**kwargs)
        raise ConnectionError("confirmation lost after durable write")

    with monkeypatch.context() as patch:
        patch.setattr(repository.es, "index", publish_then_timeout)
        with pytest.raises(ConnectionError):
            repository.publish(result, artifacts, manifest)
    assert repository.get(result.id).manifest == manifest
    before = dict(repository.s3.objects)
    repository.publish(result, artifacts, manifest)
    assert repository.s3.objects == before
    repository.cleanup_input(result.id)
    assert not repository._input_path(result.id).exists()


def test_conflicting_publisher_cannot_replace_winner(repository, tmp_path, monkeypatch):
    result, manifest, artifacts = prepared(repository, tmp_path)
    original = repository.es.index

    def competing_publish(**kwargs):
        original(**kwargs)
        return original(**kwargs)  # Simulate a second write using the stale sequence number.

    monkeypatch.setattr(repository.es, "index", competing_publish)
    repository.publish(result, artifacts, manifest)
    assert repository.get(result.id).manifest == manifest
