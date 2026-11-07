"""Import synthetic old bundles; never access the real image volume."""

import json
from uuid import uuid4

import pytest
from sqlalchemy import select

from backend.import_legacy import import_bundle
from backend.repository import ImageRepository
from backend.schemas import ImageResult
from backend.settings import Settings
from backend.tables import ImageObject
from backend.tests.test_storage import image_bytes


def legacy_bundle(tmp_path, *, crops=True):
    image_id, object_id = uuid4(), uuid4()
    directory = tmp_path / str(image_id)
    directory.mkdir()
    result = ImageResult(
        id=image_id,
        width=16,
        height=12,
        models={stage: {} for stage in ("detection", "classification", "annotation")},
        detection={"objects": [{"object_id": object_id, "label": "cat", "bbox": [1, 1, 8, 8]}]},
        classification={"objects": [{"object_id": object_id, "label": "tabby"}]},
        annotation={"raw_caption": "  Original untouched caption.  "},
    )
    manifest = result.model_dump(mode="json", exclude_none=True)
    if crops:
        manifest["crops"] = [f"{object_id}.png"]
        (directory / "crops").mkdir()
        (directory / "crops" / f"{object_id}.png").write_bytes(image_bytes(size=(7, 7)))
    for name in ("original", "normalized.png", "detection.png"):
        (directory / name).write_bytes(image_bytes())
    (directory / "results.json").write_bytes(json.dumps(manifest, indent=3).encode())
    return directory


@pytest.fixture
def settings():
    return Settings(database_url="sqlite://", broker_url="amqp://unused")


@pytest.mark.parametrize("crops", [True, False])
def test_import_preserves_bytes_is_idempotent_and_catalogues_actual_labels(
    tmp_path, database, settings, crops
):
    directory = legacy_bundle(tmp_path, crops=crops)
    before = {
        p.relative_to(directory).as_posix(): p.read_bytes()
        for p in directory.rglob("*")
        if p.is_file()
    }
    _, sessions = database
    assert import_bundle(directory, sessions, settings)
    assert not import_bundle(directory, sessions, settings)
    image_id = ImageResult.model_validate_json(before["results.json"]).id
    with sessions() as session:
        repo = ImageRepository(session)
        assert {name: content for name, (_, content) in repo.bundle(image_id).items()} == before
        obj = session.scalars(select(ImageObject)).one()
        assert obj.label == "cat"
        assert obj.object_caption is None
        assert obj.attributes == {}
        assert repo.get(image_id).manifest["classification"]["objects"][0]["label"] == "tabby"
    assert {
        p.relative_to(directory).as_posix(): p.read_bytes()
        for p in directory.rglob("*")
        if p.is_file()
    } == before


@pytest.mark.parametrize("problem", ["missing", "symlink", "uuid", "crops", "too_large"])
def test_invalid_bundle_never_creates_ready_record(tmp_path, database, settings, problem):
    directory = legacy_bundle(tmp_path)
    if problem == "missing":
        (directory / "detection.png").unlink()
    elif problem == "symlink":
        (directory / "detection.png").unlink()
        (directory / "detection.png").symlink_to(directory / "normalized.png")
    elif problem == "uuid":
        manifest = json.loads((directory / "results.json").read_bytes())
        manifest["id"] = str(uuid4())
        (directory / "results.json").write_text(json.dumps(manifest))
    elif problem == "crops":
        next((directory / "crops").iterdir()).unlink()
    else:
        settings.max_bundle_bytes = 1
    _, sessions = database
    with pytest.raises(ValueError):
        import_bundle(directory, sessions, settings)
    with sessions() as session:
        from backend.tables import ImageRecord

        assert session.scalars(select(ImageRecord)).all() == []


def test_conflict_refuses_overwrite(tmp_path, database, settings):
    directory = legacy_bundle(tmp_path)
    _, sessions = database
    assert import_bundle(directory, sessions, settings)
    image_id = ImageResult.model_validate_json((directory / "results.json").read_bytes()).id
    (directory / "original").write_bytes(image_bytes(mode="RGBA"))
    with pytest.raises(ValueError, match="Conflicting"):
        import_bundle(directory, sessions, settings)
    with sessions() as session:
        assert ImageRepository(session).artifact(image_id, "original").content == image_bytes()
