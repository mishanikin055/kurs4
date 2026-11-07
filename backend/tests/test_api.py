"""HTTP contracts backed by SQL transactions, without running models or a broker."""

import json
from io import BytesIO
from uuid import UUID, uuid4
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError

from backend.main import create_app
from backend.repository import ImageRepository
from backend.schemas import ImageResult
from backend.settings import Settings
from backend.tables import ImageRecord
from backend.tests.test_storage import image_bytes


@pytest.fixture
def app(database, monkeypatch):
    engine, sessions = database
    settings = Settings(
        database_url="sqlite://", broker_url="amqp://unused:unused@127.0.0.1:1/unused"
    )
    monkeypatch.setattr("backend.main.make_engine", lambda url: engine)
    monkeypatch.setattr("backend.service.publish_image", lambda *args: None)
    application = create_app(settings)
    application.state.test_sessions = sessions
    return application


def ready_bundle(app, *, legacy=False, empty=False):
    image_id, object_id = uuid4(), uuid4()
    objects = [] if empty else [{"object_id": str(object_id), "label": "cat", "bbox": [1, 1, 8, 8]}]
    classified = (
        []
        if empty
        else [
            {
                "object_id": str(object_id),
                **({"label": "tabby"} if legacy else {"object_caption": "  <b>A cat</b>.  "}),
            }
        ]
    )
    result = ImageResult(
        id=image_id,
        width=16,
        height=12,
        models={stage: {} for stage in ("detection", "classification", "annotation")},
        detection={"objects": objects},
        classification={"objects": classified},
        annotation={"raw_caption": "  A detailed scene.  "},
    )
    manifest = result.model_dump(mode="json", exclude_none=True)
    manifest["crops"] = [] if empty else [f"{object_id}.png"]
    artifacts = {name: image_bytes() for name in ("original", "normalized.png", "detection.png")}
    if not empty:
        artifacts[f"crops/{object_id}.png"] = image_bytes(size=(7, 7))
    artifacts["results.json"] = json.dumps(manifest, indent=3).encode()
    with app.state.test_sessions.begin() as session:
        repo = ImageRepository(session)
        repo.create_input(
            image_id, 16, 12, {name: artifacts[name] for name in ("original", "normalized.png")}
        )
        repo.publish(result, artifacts, manifest)
    return image_id, manifest, artifacts


def test_upload_commits_inputs_before_publishing_and_pending_is_not_visible(app, monkeypatch):
    published = []

    def publish(celery, image_id, timeout):
        with app.state.test_sessions() as session:
            record, inputs = ImageRepository(session).input(UUID(image_id))
            assert record.manifest is None
            assert inputs["original"] == image_bytes()
        published.append(image_id)

    monkeypatch.setattr("backend.service.publish_image", publish)
    with TestClient(app) as client:
        response = client.post("/images", files={"file": ("photo.png", image_bytes())})
        assert response.status_code == 202
        assert set(response.json()) == {"image_id"}
        image_id = response.json()["image_id"]
        assert published == [image_id]
        for path in ("", "/files/original", "/archive"):
            assert client.get(f"/images/{image_id}{path}").status_code == 404
        assert client.get(f"/images/{image_id}").headers["cache-control"] == "no-store"


def test_lost_broker_confirm_retains_durable_input(app, monkeypatch):
    def fail(*args):
        raise ConnectionError("confirm lost")

    monkeypatch.setattr("backend.service.publish_image", fail)
    with TestClient(app) as client:
        response = client.post("/images", files={"file": ("photo.png", image_bytes())})
        assert response.status_code == 503
    with app.state.test_sessions() as session:
        record = session.scalars(select(ImageRecord)).one()
        assert record.manifest is None
        assert ImageRepository(session).input(record.id)[1]["original"] == image_bytes()


@pytest.mark.parametrize("data,status", [(b"bad", 422), (image_bytes("GIF"), 415)])
def test_invalid_upload_never_persists_or_publishes(app, monkeypatch, data, status):
    monkeypatch.setattr(
        "backend.service.publish_image", lambda *args: pytest.fail("Invalid upload published")
    )
    with TestClient(app) as client:
        assert client.post("/images", files={"file": ("bad.png", data)}).status_code == status
    with app.state.test_sessions() as session:
        assert session.scalar(select(func.count()).select_from(ImageRecord)) == 0


def test_files_zip_and_captions_preserve_original_bytes(app, monkeypatch):
    monkeypatch.setattr(
        "backend.service.compatible_labels", lambda: pytest.fail("Caption used mapping")
    )
    image_id, manifest, artifacts = ready_bundle(app)
    with TestClient(app) as client:
        assert client.get(f"/images/{image_id}").json() == manifest
        for name, content in artifacts.items():
            response = client.get(f"/images/{image_id}/files/{name}")
            assert response.content == content
            assert response.headers["cache-control"] == "no-store"
        response = client.get(f"/images/{image_id}/archive")
        assert response.status_code == 200
        with ZipFile(BytesIO(response.content)) as archive:
            assert set(archive.namelist()) == {f"{image_id}/{name}" for name in artifacts}
            for name, content in artifacts.items():
                assert archive.read(f"{image_id}/{name}") == content
        for name in ("secret", "crops/not-a-crop.png", "crops/../../secret"):
            assert client.get(f"/images/{image_id}/files/{name}").status_code == 404


def test_legacy_mapping_changes_only_preview(app):
    image_id, manifest, artifacts = ready_bundle(app, legacy=True)
    with TestClient(app) as client:
        preview = client.get(f"/images/{image_id}").json()
        assert preview["classification"]["objects"][0]["refinement"] == "tabby"
        assert preview["annotation"] == manifest["annotation"]
        assert (
            client.get(f"/images/{image_id}/files/results.json").content
            == artifacts["results.json"]
        )


def test_empty_crops_folder_is_present_in_export(app):
    image_id, _, _ = ready_bundle(app, empty=True)
    with TestClient(app) as client:
        with ZipFile(BytesIO(client.get(f"/images/{image_id}/archive").content)) as archive:
            assert f"{image_id}/crops/" in archive.namelist()


def test_database_outage_returns_503_without_internal_details(app, monkeypatch):
    def unavailable(*args):
        raise OperationalError("secret query", {}, Exception("password"))

    monkeypatch.setattr(ImageRepository, "get", unavailable)
    with TestClient(app) as client:
        response = client.get(f"/images/{uuid4()}")
        assert response.status_code == 503
        assert response.json() == {"detail": "database_unavailable"}


def test_streaming_body_limit_without_content_length(app):
    app.state.settings.max_upload_bytes = 50
    application = create_app(app.state.settings)

    def body():
        yield b'--test\r\nContent-Disposition: form-data; name="file"; filename="big.png"\r\n\r\n'
        for _ in range(80):
            yield b"x" * 1024
        yield b"\r\n--test--\r\n"

    with TestClient(application) as client:
        response = client.post(
            "/images",
            content=body(),
            headers={"Content-Type": "multipart/form-data; boundary=test"},
        )
    assert response.status_code == 413


def test_health_checks_both_dependencies(app, monkeypatch):
    monkeypatch.setattr("backend.api.check_broker", lambda *args: True)
    with TestClient(app) as client:
        assert client.get("/health").json() == {"rabbitmq": True, "database": True}
        monkeypatch.setattr("backend.api.check_broker", lambda *args: False)
        assert client.get("/health").status_code == 503


def test_database_failure_during_upload_does_not_publish(app, monkeypatch):
    def unavailable(*args):
        raise OperationalError("private SQL", {}, Exception("password"))

    monkeypatch.setattr(ImageRepository, "create_input", unavailable)
    monkeypatch.setattr(
        "backend.service.publish_image", lambda *args: pytest.fail("Undurable input published")
    )
    with TestClient(app) as client:
        response = client.post("/images", files={"file": ("photo.png", image_bytes())})
        assert response.status_code == 503
        assert response.json() == {"detail": "database_unavailable"}
