"""Uploads, previews and downloads work without accessing PostgreSQL."""

import json
from io import BytesIO
from uuid import UUID, uuid4
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

from backend.main import create_app
from backend.settings import Settings
from backend.tests.test_storage import image_bytes


@pytest.fixture
def app(tmp_path, monkeypatch):
    settings = Settings(
        database_url="postgresql+psycopg://unused:unused@127.0.0.1:1/unused",
        broker_url="amqp://unused:unused@127.0.0.1:1/unused",
        image_root=tmp_path,
    )
    monkeypatch.setattr("backend.service.publish_image", lambda *args: None)
    return create_app(settings)


def test_upload_publishes_directly_without_database(app, monkeypatch):
    published = []
    monkeypatch.setattr("backend.service.publish_image", lambda *args: published.append(args))
    with TestClient(app) as client:
        response = client.post("/images", files={"file": ("photo.png", image_bytes())})
        assert response.status_code == 202
        accepted = response.json()
        image_id = UUID(accepted["image_id"])
        assert accepted == {"image_id": str(image_id), "output_folder": f"results/{image_id}"}
        assert published == [(app.state.celery, str(image_id), 3)]
        assert (app.state.settings.image_root / str(image_id) / "original").is_file()
        assert not (app.state.settings.image_root / "results").exists()
        assert client.get(f"/jobs/{image_id}").status_code == 404
        assert client.get(f"/jobs/{image_id}/status").status_code == 404
        pending = client.get(f"/images/{image_id}")
        assert pending.status_code == 404
        assert pending.headers["cache-control"] == "no-store"


def test_broker_failure_retains_inputs_in_case_confirmation_was_lost(app, monkeypatch):
    def fail(*args):
        raise ConnectionError("confirm lost")

    monkeypatch.setattr("backend.service.publish_image", fail)
    with TestClient(app) as client:
        response = client.post("/images", files={"file": ("photo.png", image_bytes())})
    assert response.status_code == 503
    assert len(list(app.state.settings.image_root.glob("*/normalized.png"))) == 1


@pytest.mark.parametrize("data,status", [(b"bad", 422), (image_bytes("GIF"), 415)])
def test_invalid_upload_is_not_published(app, monkeypatch, data, status):
    def unexpected_publish(*args):
        pytest.fail("Invalid upload was published")

    monkeypatch.setattr("backend.service.publish_image", unexpected_publish)
    with TestClient(app) as client:
        assert client.post("/images", files={"file": ("bad.png", data)}).status_code == status
    assert not list(app.state.settings.image_root.iterdir())


def test_downloads_only_expose_completed_bundle_inside_volume(app, tmp_path):
    image_id = uuid4()
    directory = tmp_path / "results" / str(image_id)
    directory.mkdir(parents=True)
    files = {
        "original": image_bytes(),
        "normalized.png": image_bytes(),
        "detection.png": image_bytes(),
        "results.json": b'{"annotation":{"raw_caption":"  Caption.  "}}',
    }
    with TestClient(app) as client:
        url = f"/images/{image_id}"
        for name, content in files.items():
            if name != "results.json":
                (directory / name).write_bytes(content)
        assert client.get(f"{url}/files/detection.png").status_code == 404
        assert client.get(f"{url}/archive").status_code == 404
        (directory / "results.json").write_bytes(files["results.json"])
        for name, content in files.items():
            response = client.get(f"{url}/files/{name}")
            assert response.content == content
            assert response.headers["cache-control"] == "no-store"
        archive = client.get(f"{url}/archive")
        assert archive.headers["content-type"] == "application/zip"
        with ZipFile(BytesIO(archive.content)) as bundle:
            assert set(bundle.namelist()) == {f"{image_id}/{name}" for name in files}
            assert bundle.read(f"{image_id}/results.json") == files["results.json"]
        assert client.get(f"{url}/files/secret").status_code == 404
        assert client.get("/images/invalid/archive").status_code == 422
        outside = tmp_path / "secret.png"
        outside.write_bytes(b"private")
        (directory / "detection.png").unlink()
        (directory / "detection.png").symlink_to(outside)
        assert client.get(f"{url}/files/detection.png").status_code == 404
        assert client.get(f"{url}/archive").status_code == 404


def test_file_size_is_checked_before_publication(app):
    app.state.settings.max_upload_bytes = 50
    with TestClient(app) as client:
        response = client.post("/images", files={"file": ("large.png", image_bytes())})
    assert response.status_code == 413
    assert not list(app.state.settings.image_root.iterdir())


def test_crops_are_downloadable_and_included_in_zip(app, tmp_path):
    image_id, object_id = uuid4(), uuid4()
    directory = tmp_path / "results" / str(image_id)
    (directory / "crops").mkdir(parents=True)
    name = f"{object_id}.png"
    crop = image_bytes(size=(5, 6))
    (directory / "crops" / name).write_bytes(crop)
    for filename in ("original", "normalized.png", "detection.png"):
        (directory / filename).write_bytes(image_bytes())
    manifest = directory / "results.json"
    manifest.write_text(json.dumps({"crops": [name]}))
    with TestClient(app) as client:
        url = f"/images/{image_id}"
        assert client.get(f"{url}/files/crops/{name}").content == crop
        with ZipFile(BytesIO(client.get(f"{url}/archive").content)) as archive:
            assert archive.read(f"{image_id}/crops/{name}") == crop
        assert client.get(f"{url}/files/crops/not-a-crop.png").status_code == 404
        outside = tmp_path / "private.png"
        outside.write_bytes(b"private")
        (directory / "crops" / name).unlink()
        (directory / "crops" / name).symlink_to(outside)
        assert client.get(f"{url}/files/crops/{name}").status_code == 404
        assert client.get(f"{url}/archive").status_code == 404
        (directory / "crops" / name).unlink()
        manifest.write_text(json.dumps({"crops": []}))
        with ZipFile(BytesIO(client.get(f"{url}/archive").content)) as archive:
            assert f"{image_id}/crops/" in archive.namelist()


def test_preview_only_shows_compatible_refinements_and_preserves_raw_bundle(app, tmp_path):
    image_id = uuid4()
    directory = tmp_path / "results" / str(image_id)
    directory.mkdir(parents=True)
    pairs = [
        ("person", "unicycle", None),
        ("skateboard", "Loafer", None),
        ("person", "whiskey jug", None),
        ("cat", "tabby", "tabby"),
        ("dog", "Labrador retriever", "Labrador retriever"),
        ("dog", "tabby", None),
        ("zebra", "zebra", None),
        ("dog", "unknown", None),
        ("dog", None, None),
    ]
    result = {
        "detection": {
            "objects": [
                {"object_id": str(i), "label": coco} for i, (coco, _, _) in enumerate(pairs)
            ]
        },
        "classification": {
            "objects": [
                {"object_id": str(i), "label": label} for i, (_, label, _) in enumerate(pairs)
            ][::-1]
        },
        "annotation": {"raw_caption": "  A person on a skateboard.  "},
    }
    raw = json.dumps(result).encode()
    (directory / "results.json").write_bytes(raw)
    with TestClient(app) as client:
        response = client.get(f"/images/{image_id}")
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        preview = response.json()
        assert preview["annotation"] == result["annotation"]
        assert preview["detection"] == result["detection"]
        for obj in preview["classification"]["objects"]:
            i = int(obj["object_id"])
            assert obj["label"] == pairs[i][1]
            assert obj["refinement"] == pairs[i][2]
        assert client.get(f"/images/{image_id}/files/results.json").content == raw
    assert (directory / "results.json").read_bytes() == raw


def test_streaming_body_limit_without_content_length(app):
    application = create_app(app.state.settings.model_copy(update={"max_upload_bytes": 50}))

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
    assert not list(app.state.settings.image_root.iterdir())
