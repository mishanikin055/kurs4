"""Portable startup orchestration and device selection, without Docker or models."""

import hashlib
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from backend import models
from backend.settings import Settings, model_specs
from scripts import start_app


@pytest.mark.parametrize("device,dtype", [("cpu", "float32"), ("cuda:0", "float16")])
def test_florence_model_and_floating_inputs_use_device_precision(
    monkeypatch, tmp_path, device, dtype
):
    torch = SimpleNamespace(float16="float16", float32="float32", inference_mode=nullcontext)
    processor, model = MagicMock(), MagicMock()
    loader = MagicMock()
    loader.from_pretrained.return_value.eval.return_value = model
    transformers = SimpleNamespace(AutoProcessor=MagicMock(), AutoModelForCausalLM=loader)
    transformers.AutoProcessor.from_pretrained.return_value = processor
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    pixels, ids = MagicMock(), MagicMock()
    pixels.is_floating_point.return_value = True
    ids.is_floating_point.return_value = False
    processor.return_value = {"pixel_values": pixels, "input_ids": ids}
    loaded = models.load_annotator(tmp_path, device)
    assert loader.from_pretrained.call_args.kwargs["torch_dtype"] == dtype
    models.annotate(
        None, tmp_path, {"device": device, "prompt": "x", "max_new_tokens": 10}, loaded=loaded
    )
    pixels.to.assert_called_once_with(device, dtype=dtype)
    ids.to.assert_called_once_with(device)
    specs = model_specs(Settings.model_construct(device=device))
    assert specs["annotation"]["parameters"]["precision"] == ("fp32" if device == "cpu" else "fp16")


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(start_app, "ROOT", tmp_path)
    (tmp_path / "backend").mkdir()
    (tmp_path / "backend" / "app.env.example").write_text(
        "PASSWORD=REPLACE_WITH_S3_SECRET_HEX\nURL=REPLACE_WITH_S3_SECRET_HEX\n"
        "BROKER=REPLACE_WITH_BROKER_SECRET_HEX\n"
    )
    model = tmp_path / "annotation" / "models" / "test"
    model.mkdir(parents=True)
    (model / "weights").write_bytes(b"weights")
    (tmp_path / "backend" / "checkpoints.json").write_text(
        json.dumps(
            {
                "annotation": {
                    "model_name": "test",
                    "files": [
                        {"path": "weights", "sha256": hashlib.sha256(b"weights").hexdigest()}
                    ],
                }
            }
        )
    )
    return tmp_path


def test_secrets_generate_matching_password_and_never_overwrite(bundle):
    start_app.prepare_secrets()
    path = bundle / ".secrets" / "app.env"
    content = path.read_text()
    lines = content.splitlines()
    assert lines[0].split("=")[1] == lines[1].split("=")[1]
    assert lines[0].split("=")[1] != lines[2].split("=")[1]
    assert "REPLACE" not in content
    start_app.prepare_secrets()
    assert path.read_text() == content


def mock_docker(monkeypatch, *, runtimes=None):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(
            stdout=json.dumps(
                {
                    "OSType": "linux",
                    "Architecture": "x86_64",
                    "Runtimes": {"nvidia": {}} if runtimes is None else runtimes,
                }
            )
        )

    monkeypatch.setattr(start_app.subprocess, "run", run)
    return calls


@pytest.mark.parametrize("wsl", [False, True])
def test_missing_nvidia_runtime_stops_before_config_or_settings(bundle, monkeypatch, capsys, wsl):
    calls = mock_docker(monkeypatch, runtimes={"runc": {}})
    exists = Path.exists
    monkeypatch.setattr(
        Path, "exists", lambda path: wsl if path == Path("/dev/dxg") else exists(path)
    )
    before = sorted(bundle.rglob("*"))
    assert start_app.main(["--device", "cuda"]) == 1
    assert sorted(bundle.rglob("*")) == before
    assert len(calls) == 2
    error = capsys.readouterr().err
    assert "runtime nvidia" in error
    assert ("--device wsl-cuda" if wsl else "NVIDIA Container Toolkit") in error


def test_missing_wsl_device_stops_before_config_or_settings(bundle, monkeypatch, capsys):
    calls = mock_docker(monkeypatch)
    exists = Path.exists
    monkeypatch.setattr(
        Path, "exists", lambda path: False if path == Path("/dev/dxg") else exists(path)
    )
    before = sorted(bundle.rglob("*"))
    assert start_app.main(["--device", "wsl-cuda"]) == 1
    assert sorted(bundle.rglob("*")) == before
    assert len(calls) == 2
    assert "/dev/dxg" in capsys.readouterr().err


@pytest.mark.parametrize("existing_secrets", [False, True])
def test_check_from_other_cwd_is_read_only(bundle, tmp_path, monkeypatch, existing_secrets):
    monkeypatch.chdir(tmp_path.parent)
    if existing_secrets:
        start_app.prepare_secrets()
    secret = bundle / ".secrets" / "app.env"
    original = secret.read_bytes() if existing_secrets else None
    calls = mock_docker(monkeypatch)
    before = sorted(bundle.rglob("*"))
    assert start_app.main(["--device", "cpu", "--check"]) == 0
    assert sorted(bundle.rglob("*")) == before
    assert (secret.read_bytes() if existing_secrets else None) == original
    assert all(call[1]["cwd"] == bundle for call in calls)
    assert all(call[1]["env"]["APP_DEVICE"] == "cpu" for call in calls)
    assert not any("up" in call[0] or "run" in call[0] for call in calls)


def test_bad_weights_stop_before_services_or_settings(bundle, monkeypatch):
    calls = mock_docker(monkeypatch)
    (bundle / "annotation" / "models" / "test" / "weights").write_bytes(b"broken")
    assert start_app.main(["--device", "cuda"]) == 1
    assert not (bundle / ".secrets").exists()
    assert not any("up" in call[0] or "build" in call[0] for call in calls)


def test_start_initializes_storage_before_worker(bundle, monkeypatch):
    calls = mock_docker(monkeypatch)
    assert start_app.main(["--device", "cuda"]) == 0
    commands = [call[0] for call in calls]
    assert commands[-3][-6:] == ["up", "-d", "--wait", "elasticsearch", "s3", "rabbitmq"]
    assert commands[-2][-7:] == [
        "run",
        "--rm",
        "--no-deps",
        "backend",
        "python",
        "-m",
        "backend.db",
    ]
    assert any(
        command[-5:] == ["build", "s3", "backend", "worker", "frontend"] for command in commands
    )
    assert commands[-1][-5:] == ["up", "-d", "backend", "worker", "frontend"]
    assert "compose.app.gpu.yaml" in commands[-1]
    assert (bundle / "storage" / "locks" / "inference.lock").is_file()


@pytest.mark.parametrize("existing", ["symlink", "directory"])
def test_nonregular_lock_rejected(bundle, monkeypatch, existing):
    calls = mock_docker(monkeypatch)
    lock = bundle / "storage" / "locks" / "inference.lock"
    lock.parent.mkdir(parents=True)
    if existing == "symlink":
        lock.symlink_to(bundle / "missing")
    else:
        lock.mkdir()
    assert start_app.main(["--device", "cpu"]) == 1
    assert not any("build" in call[0] for call in calls)


def test_initialization_failure_stops_before_worker(bundle, monkeypatch):
    import subprocess

    calls = mock_docker(monkeypatch)
    mocked_run = start_app.subprocess.run

    def fail_init(command, **kwargs):
        result = mocked_run(command, **kwargs)
        if command[-1] == "backend.db":
            raise subprocess.CalledProcessError(1, command)
        return result

    monkeypatch.setattr(start_app.subprocess, "run", fail_init)
    assert start_app.main(["--device", "cpu"]) == 1
    assert not any(
        call[0][-3:] == ["backend", "worker", "frontend"] and "up" in call[0] for call in calls
    )


def test_existing_postgres_settings_receive_s3_without_changing_broker(bundle):
    template = Path(__file__).resolve().parents[1] / "app.env.example"
    (bundle / "backend" / "app.env.example").write_text(template.read_text())
    secret = bundle / ".secrets" / "app.env"
    secret.parent.mkdir()
    old = "APP_BROKER_URL=amqp://existing\nRABBITMQ_DEFAULT_PASS=keep-me\n"
    secret.write_text(old)
    start_app.prepare_secrets()
    values = dict(
        line.split("=", 1)
        for line in secret.read_text().splitlines()
        if "=" in line and not line.startswith("#")
    )
    assert secret.read_text().startswith(old)
    assert values["APP_BROKER_URL"] == "amqp://existing"
    assert values["RABBITMQ_DEFAULT_PASS"] == "keep-me"
    assert values["APP_S3_SECRET_KEY"] == values["MINIO_ROOT_PASSWORD"]
    assert "REPLACE" not in secret.read_text()
    before = secret.read_bytes()
    start_app.prepare_secrets()
    assert secret.read_bytes() == before
