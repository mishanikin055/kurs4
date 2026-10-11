import os
import subprocess
import sys
import time
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.api import get_service
from backend.main import create_app
from backend.processing import LockTimeout, inference_lock, run_stage, validate_payload
from backend.schemas import AcceptedImage, DetectionPayload
from backend.settings import Settings
from backend.tests.test_storage import image_bytes


@pytest.fixture
def settings(tmp_path):
    return Settings(
        database_url="postgresql+psycopg://unused:unused@localhost/unused",
        broker_url="amqp://unused:unused@localhost/unused",
        image_root=tmp_path,
        lock_path=tmp_path / "inference.lock",
        detection_python=sys.executable,
    )


def test_api_service_can_be_overridden(settings):
    application = create_app(settings, publisher_enabled=False)
    accepted = AcceptedImage(
        image_id=uuid4(),
        job_id=uuid4(),
        status="queued",
        publish_status="pending",
        status_url="/jobs/test",
    )
    application.dependency_overrides[get_service] = lambda: SimpleNamespace(
        upload=lambda file: accepted
    )
    with TestClient(application) as client:
        response = client.post("/images", files={"file": ("test.png", image_bytes())})
    assert response.status_code == 202
    assert response.json()["job_id"] == str(accepted.job_id)


@pytest.mark.parametrize(
    "box,score",
    [
        ([-1, 0, 2, 2], 0.5),
        ([0, 0, 20, 2], 0.5),
        ([2, 0, 1, 2], 0.5),
        ([0, 0, 2, 2], float("nan")),
        ([0, 0, 2, 2], 1.1),
    ],
)
def test_invalid_detection_is_rejected(box, score):
    with pytest.raises(ValidationError):
        DetectionPayload.model_validate(
            dict(
                coordinate_space="normalized_image",
                bbox_format="xyxy",
                image_width=16,
                image_height=12,
                objects=[
                    dict(
                        object_id=str(uuid4()),
                        bbox=box,
                        label="person",
                        class_id=1,
                        label_space="coco",
                        score=score,
                    )
                ],
            )
        )


def test_classification_cannot_lose_objects():
    with pytest.raises(ValueError):
        validate_payload(
            "classification", {"objects": []}, None, {"objects": [{"object_id": str(uuid4())}]}
        )


def test_raw_caption_is_unchanged():
    status, payload = validate_payload("annotation", {"raw_caption": "  a cat  "}, None, None)
    assert status == "succeeded"
    assert payload["raw_caption"] == "  a cat  "


def test_lock_contention_is_bounded(settings):
    with inference_lock(settings.lock_path, time.monotonic() + 1):
        with pytest.raises(LockTimeout), inference_lock(settings.lock_path, time.monotonic() + 0.1):
            pytest.fail("Lock acquired twice")
    with inference_lock(settings.lock_path, time.monotonic() + 1):
        pass


def test_child_retains_lock_after_parent_exit(settings, tmp_path):
    marker = tmp_path / "child.pid"
    code = """import fcntl,subprocess,sys
from pathlib import Path
f=open(sys.argv[1],"a+")
fcntl.flock(f,fcntl.LOCK_EX)
p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"],pass_fds=(f.fileno(),))
Path(sys.argv[2]).write_text(str(p.pid))
"""
    subprocess.run([sys.executable, "-c", code, str(settings.lock_path), str(marker)], check=True)
    pid = int(marker.read_text())
    try:
        with pytest.raises(LockTimeout), inference_lock(settings.lock_path, time.monotonic() + 0.1):
            pytest.fail("Parent exit released child's lock")
    finally:
        os.kill(pid, 9)
    with inference_lock(settings.lock_path, time.monotonic() + 2):
        pass


def test_timeout_stops_child_before_releasing_lock(settings, tmp_path, monkeypatch):
    from PIL import Image

    image = tmp_path / "test.png"
    Image.new("RGB", (16, 12)).save(image)
    marker = tmp_path / "started"
    monkeypatch.setenv("MOCK_MODEL_MODE", "slow")
    monkeypatch.setenv("MOCK_STARTED", str(marker))
    with inference_lock(settings.lock_path, time.monotonic() + 1) as fd:
        response = run_stage(
            {"stage": "detection", "image_path": str(image)},
            settings,
            fd,
            0.5,
            module="backend.tests.mock_model",
        )
        assert response == {"error": {"code": "stage_timeout"}}
        with pytest.raises(ProcessLookupError):
            os.kill(int(marker.read_text()), 0)
    with inference_lock(settings.lock_path, time.monotonic() + 1):
        pass
