"""Opt-in checks against isolated Elasticsearch and S3; no model weights or production data."""

import json
import os
from io import BytesIO
from uuid import UUID, uuid4
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

from backend.db import initialize, make_elasticsearch, make_s3
from backend.main import create_app
from backend.processing import process_image
from backend.repository import ImageRepository
from backend.settings import Settings
from backend.tests.mock_model import predict
from backend.tests.test_storage import image_bytes

pytestmark = pytest.mark.skipif(
    os.environ.get("APP_TEST_INTEGRATION") != "1",
    reason="Requires isolated Elasticsearch/S3 (APP_TEST_INTEGRATION=1)",
)


@pytest.fixture
def storage(tmp_path):
    # Create unique disposable resources; never use a configured application index/bucket.
    suffix = uuid4().hex
    settings = Settings(
        elasticsearch_index=f"test-images-{suffix}",
        s3_bucket=f"test-crops-{suffix}",
        pending_input_dir=tmp_path / "pending",
        lock_path=tmp_path / "inference.lock",
    )
    es, s3 = make_elasticsearch(settings), make_s3(settings)
    try:
        initialize(settings, es, s3)
        initialize(settings, es, s3)
        yield ImageRepository(es, s3, settings)
    finally:
        response = s3.list_objects_v2(Bucket=settings.s3_bucket)
        for obj in response.get("Contents", []):
            s3.delete_object(Bucket=settings.s3_bucket, Key=obj["Key"])
        s3.delete_bucket(Bucket=settings.s3_bucket)
        es.indices.delete(index=settings.elasticsearch_index)
        es.close()
        s3.close()


@pytest.mark.parametrize("empty", [False, True])
def test_real_storage_api_and_mock_pipeline(storage, monkeypatch, empty):
    monkeypatch.setenv("MOCK_MODEL_MODE", "empty" if empty else "success")
    delivered, calls = [], []
    monkeypatch.setattr(
        "backend.service.publish_image", lambda app, id, timeout: delivered.append(id)
    )

    def runner(request, *args, **kwargs):
        calls.append(request["stage"])
        return predict(request)

    with TestClient(create_app(storage.settings)) as client:
        response = client.post("/images", files={"file": ("photo.png", image_bytes())})
        assert response.status_code == 202
        image_id = UUID(response.json()["image_id"])
        assert delivered == [str(image_id)]
        assert client.get(f"/images/{image_id}").status_code == 404
        process_image(image_id, storage, storage.settings, runner=runner)
        assert not storage._input_path(image_id).exists()
        manifest = client.get(f"/images/{image_id}").json()
        assert manifest["annotation"]["raw_caption"] == "  Synthetic caption.  "
        expected = {"results.json", *(f"crops/{name}" for name in manifest["crops"])}
        with ZipFile(BytesIO(client.get(f"/images/{image_id}/archive").content)) as archive:
            names = {f"{image_id}/{name}" for name in expected}
            if empty:
                names.add(f"{image_id}/crops/")
            assert set(archive.namelist()) == names
            assert json.loads(archive.read(f"{image_id}/results.json")) == manifest
        for name in ("original", "normalized.png", "detection.png"):
            assert client.get(f"/images/{image_id}/files/{name}").status_code == 404
        before = len(calls)
        process_image(image_id, storage, storage.settings, runner=runner)
        assert len(calls) == before
        storage.es.indices.refresh(index=storage.index)
        hits = storage.es.search(
            index=storage.index, query={"match": {"manifest.annotation.raw_caption": "Synthetic"}}
        )
        assert hits["hits"]["total"]["value"] == 1
        objects = storage.s3.list_objects_v2(Bucket=storage.bucket).get("Contents", [])
        assert len(objects) == len(manifest["crops"])
        assert all("/crops/" in obj["Key"] for obj in objects)
