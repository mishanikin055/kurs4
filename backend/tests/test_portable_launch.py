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
    (tmp_path / "backend" / "translation.json").write_text(
        json.dumps({"model_name": "test", "files": []})
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


def mock_docker(monkeypatch, *, runtimes=None, s3_settings=None):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[-3:] == ["config", "--format", "json"]:
            return SimpleNamespace(
                stdout=json.dumps({"services": {"backend": {"environment": s3_settings or {}}}})
            )
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


def external_settings(bundle, **overrides):
    values = {
        "APP_S3_ENDPOINT_URL": "https://objects.example.invalid",
        "APP_S3_ACCESS_KEY": "external-user",
        "APP_S3_SECRET_KEY": "external-secret",
        "APP_S3_BUCKET": "existing-crops",
        "APP_S3_REGION": "eu-west-1",
    } | overrides
    secret = bundle / ".secrets" / "app.env"
    secret.parent.mkdir(exist_ok=True)
    secret.write_text("".join(f"{name}={value}\n" for name, value in values.items()))
    return values


@pytest.mark.parametrize("endpoint", [None, "", "  ", "http://s3:9000", "http://s3:9000/"])
def test_local_s3_endpoint_keeps_minio_in_startup(bundle, monkeypatch, endpoint):
    values = external_settings(bundle, APP_S3_ENDPOINT_URL=endpoint or "")
    if endpoint is None:
        values.pop("APP_S3_ENDPOINT_URL")
    calls = mock_docker(monkeypatch, s3_settings=values)
    assert start_app.main(["--device", "cpu"]) == 0
    commands = [command for command, _ in calls]
    assert not any("compose.app.external-s3.yaml" in command for command in commands)
    assert any(
        command[-6:] == ["up", "-d", "--wait", "elasticsearch", "s3", "rabbitmq"]
        for command in commands
    )


@pytest.mark.parametrize("device", ["cpu", "cuda", "wsl-cuda"])
def test_external_s3_skips_minio_and_stops_previous_container(bundle, monkeypatch, capsys, device):
    values = external_settings(bundle)
    calls = mock_docker(monkeypatch, s3_settings=values)
    exists = Path.exists
    monkeypatch.setattr(
        Path, "exists", lambda path: True if path == Path("/dev/dxg") else exists(path)
    )
    assert start_app.main(["--device", device]) == 0
    commands = [command for command, _ in calls]
    startup = [command for command in commands if "build" in command or "up" in command]
    assert all(
        "compose.app.external-s3.yaml" in command and "s3" not in command for command in startup
    )
    assert commands[-3][-1] == "backend.db"
    assert commands[-2][-5:] == ["up", "-d", "backend", "worker", "frontend"]
    assert commands[-1][-2:] == ["stop", "s3"]
    text = (bundle / ".secrets" / "app.env").read_text()
    assert f"APP_S3_SECRET_KEY={values['APP_S3_SECRET_KEY']}\n" in text
    assert "MINIO_ROOT" not in text
    output = capsys.readouterr()
    assert values["APP_S3_SECRET_KEY"] not in output.out + output.err
    assert all(
        kwargs["stdout"] == start_app.subprocess.PIPE
        for command, kwargs in calls
        if command[-3:] == ["config", "--format", "json"]
    )


def test_external_check_is_read_only_and_uses_compose_resolved_settings(bundle, monkeypatch):
    values = external_settings(bundle)
    secret = bundle / ".secrets" / "app.env"
    secret.write_text('APP_S3_ENDPOINT_URL="${MY_S3_ENDPOINT}"\n')
    calls = mock_docker(monkeypatch, s3_settings=values)
    before = {path: path.read_bytes() for path in bundle.rglob("*") if path.is_file()}
    assert start_app.main(["--device", "cpu", "--check"]) == 0
    assert {path: path.read_bytes() for path in bundle.rglob("*") if path.is_file()} == before
    assert any("compose.app.external-s3.yaml" in command for command, _ in calls)
    assert not any(
        "build" in command or "up" in command or "stop" in command for command, _ in calls
    )


@pytest.mark.parametrize(
    "name", ["APP_S3_ACCESS_KEY", "APP_S3_SECRET_KEY", "APP_S3_BUCKET", "APP_S3_REGION"]
)
@pytest.mark.parametrize("value", ["", "REPLACE_WITH_S3_SECRET_HEX"])
def test_incomplete_external_settings_fail_without_generating_credentials(
    bundle, monkeypatch, capsys, name, value
):
    values = external_settings(bundle, **{name: value})
    calls = mock_docker(monkeypatch, s3_settings=values)
    secret = bundle / ".secrets" / "app.env"
    original = secret.read_bytes()
    assert start_app.main(["--device", "cpu"]) == 1
    assert secret.read_bytes() == original
    assert name in capsys.readouterr().err
    assert not any(
        "build" in command or "up" in command or "stop" in command for command, _ in calls
    )


@pytest.mark.parametrize(
    "endpoint",
    ["s3.example.invalid", "ftp://s3.example.invalid", "https://user:secret@example.invalid"],
)
def test_invalid_external_endpoint_is_rejected_without_exposing_it(
    bundle, monkeypatch, capsys, endpoint
):
    values = external_settings(bundle, APP_S3_ENDPOINT_URL=endpoint)
    calls = mock_docker(monkeypatch, s3_settings=values)
    assert start_app.main(["--device", "cpu"]) == 1
    output = capsys.readouterr()
    assert "APP_S3_ENDPOINT_URL" in output.err
    assert endpoint not in output.out + output.err
    assert not any("build" in command or "up" in command for command, _ in calls)


def test_external_storage_failure_never_falls_back_to_local_minio(bundle, monkeypatch):
    values = external_settings(bundle)
    calls = mock_docker(monkeypatch, s3_settings=values)
    mocked_run = start_app.subprocess.run

    def fail_init(command, **kwargs):
        result = mocked_run(command, **kwargs)
        if command[-1] == "backend.db":
            raise start_app.subprocess.CalledProcessError(1, command)
        return result

    monkeypatch.setattr(start_app.subprocess, "run", fail_init)
    assert start_app.main(["--device", "cpu"]) == 1
    assert not any("s3" in command for command, _ in calls)
    assert not any(
        command[-5:] == ["up", "-d", "backend", "worker", "frontend"] for command, _ in calls
    )


def test_check_weights_requires_verified_translation_files(bundle):
    directory = bundle / "annotation/models/opus_mt_tc_big_en_zle"
    directory.mkdir()
    target = directory / "weights"
    manifest = {
        "model_name": "opus_mt_tc_big_en_zle",
        "files": [{"path": "weights", "sha256": hashlib.sha256(b"translation").hexdigest()}],
    }
    (bundle / "backend/translation.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="Missing local model file"):
        start_app.check_weights()
    target.write_bytes(b"other")
    with pytest.raises(RuntimeError, match="Model checksum mismatch"):
        start_app.check_weights()
    target.write_bytes(b"translation")
    start_app.check_weights()
