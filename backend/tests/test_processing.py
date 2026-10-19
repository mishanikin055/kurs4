import os
import subprocess
import sys
import time
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from backend.models import classify, detect
from backend.processing import LockTimeout, inference_lock, run_stage, validate_payload
from backend.settings import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(
        database_url="postgresql+psycopg://unused:unused@localhost/unused",
        broker_url="amqp://unused:unused@localhost/unused",
        image_root=tmp_path,
        lock_path=tmp_path / "inference.lock",
        detection_python=sys.executable,
    )


@pytest.mark.parametrize(
    "box",
    [
        [-1, 0, 2, 2],
        [0, 0, 20, 2],
        [2, 0, 1, 2],
        [0, 0, float("nan"), 2],
        [0, 0, float("inf"), 2],
    ],
)
def test_invalid_detection_is_rejected(box):
    with pytest.raises(ValueError):
        validate_payload(
            "detection",
            dict(
                objects=[
                    dict(
                        object_id=str(uuid4()),
                        bbox=box,
                        label="person",
                    )
                ],
            ),
            (16, 12),
            None,
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


@pytest.mark.parametrize("invalid", [False, True])
@pytest.mark.parametrize(
    "bbox,expected_size",
    [([0.1, 0.1, 5.2, 5.2], (6, 6)), ([4, 3, 12, 9], (10, 8)), ([0, 0, 16, 12], (16, 12))],
)
def test_classifier_returns_only_one_label_or_error(
    tmp_path, monkeypatch, invalid, bbox, expected_size
):
    from PIL import Image

    logits = MagicMock()
    logits.shape = (1, 2) if invalid else (1, 1000)
    logits.argmax.return_value.item.return_value = 0
    model = MagicMock(return_value=logits)
    transform = MagicMock(return_value=MagicMock())
    weights = SimpleNamespace(IMAGENET1K_V1=SimpleNamespace(transforms=lambda: transform))
    torch = MagicMock()
    torch.inference_mode = nullcontext
    torch.isfinite.return_value.all.return_value = True
    torch.cuda.OutOfMemoryError = MemoryError
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torchvision", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "torchvision.models",
        SimpleNamespace(ViT_B_16_Weights=weights, vit_b_16=lambda **kwargs: model),
    )
    object_id = str(uuid4())
    result = classify(
        Image.new("RGB", (16, 12)),
        tmp_path,
        {"device": "cpu"},
        {"objects": [{"object_id": object_id, "bbox": bbox}]},
    )
    assert transform.call_args.args[0].size == expected_size
    obj = result["objects"][0]
    assert set(obj) == {"object_id", "error" if invalid else "label"}
    if not invalid:
        assert obj["label"] == "tench"
        logits.argmax.assert_called_once_with(-1)
    logits.softmax.assert_not_called()


@pytest.mark.parametrize("padding", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_crop_padding_is_rejected(tmp_path, padding):
    from PIL import Image

    with pytest.raises(ValueError, match="Invalid crop padding"):
        classify(
            Image.new("RGB", (16, 12)),
            tmp_path,
            {"crop_padding_fraction": padding},
            {"objects": []},
        )


@pytest.mark.parametrize(
    "record",
    [{}, {"label": ""}, {"label": "cat", "error": {"code": "crop_error"}}, {"error": {}}],
)
def test_classification_rejects_ambiguous_results(record):
    object_id = str(uuid4())
    with pytest.raises(ValueError):
        validate_payload(
            "classification",
            {"objects": [{"object_id": object_id, **record}]},
            None,
            {"objects": [{"object_id": object_id}]},
        )


def test_partial_classification_preserves_object_links():
    first, second = str(uuid4()), str(uuid4())
    payload = {
        "objects": [
            {"object_id": first, "label": "cat"},
            {"object_id": second, "error": {"code": "crop_error"}},
        ]
    }
    status, saved = validate_payload(
        "classification",
        payload,
        None,
        {"objects": [{"object_id": first}, {"object_id": second}]},
    )
    assert status == "partial"
    assert saved == payload


@pytest.mark.parametrize(
    "threshold,limit,expected", [(0.3, 100, [0.9, 0.3]), (0.3, 1, [0.9]), (1.0, 100, [])]
)
def test_detector_filters_scores_before_limiting_objects(
    tmp_path, monkeypatch, threshold, limit, expected
):
    from PIL import Image

    class Tensor:
        device = "cpu"

        def __init__(self, value=None):
            self.value = value

        def to(self, device):
            return self

        def unsqueeze(self, dim):
            return self

        def tolist(self):
            return self.value

    class Model:
        def to(self, device):
            return self

        def eval(self):
            return self

        def __call__(self, tensor):
            return {}

    result = {
        "boxes": Tensor([[0, 0, 2, 2]] * 4),
        "scores": Tensor([0.01, 0.9, 0.3, 0.299]),
        "labels": Tensor([1] * 4),
    }
    wrapper = SimpleNamespace(
        model=SimpleNamespace(model=Model(), postprocess=lambda *args, **kwargs: [result]),
        means=[],
        stds=[],
    )
    transforms = SimpleNamespace(
        to_tensor=lambda image: Tensor(),
        resize=lambda tensor, *args, **kwargs: tensor,
        normalize=lambda tensor, *args: tensor,
    )
    for name, module in {
        "torch": SimpleNamespace(
            inference_mode=nullcontext, tensor=lambda *args, **kwargs: Tensor()
        ),
        "rfdetr": SimpleNamespace(RFDETRSmall=lambda **kwargs: wrapper),
        "torchvision": SimpleNamespace(),
        "torchvision.transforms": SimpleNamespace(functional=transforms),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    (tmp_path / "label_mapping.json").write_text('{"categories":[{"id":1,"name":"person"}]}')
    output = detect(
        Image.new("RGB", (16, 12)),
        tmp_path,
        {"device": "cpu", "score_threshold": threshold, "max_detections": limit},
    )
    assert len(output["objects"]) == len(expected)
    assert all(set(obj) == {"object_id", "bbox", "label"} for obj in output["objects"])


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
