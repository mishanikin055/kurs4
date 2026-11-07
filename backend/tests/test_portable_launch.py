"""Portable startup orchestration and device selection, without Docker or models."""

import hashlib
import json
import sys
from contextlib import nullcontext
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
        "PASSWORD=REPLACE_WITH_RANDOM_HEX\nURL=REPLACE_WITH_RANDOM_HEX\n"
        "BROKER=REPLACE_WITH_ANOTHER_RANDOM_HEX\n"
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


def mock_docker(monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=json.dumps({"OSType": "linux", "Architecture": "x86_64"}))

    monkeypatch.setattr(start_app.subprocess, "run", run)
    return calls


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


def test_start_orders_database_guard_before_worker(bundle, monkeypatch):
    calls = mock_docker(monkeypatch)
    assert start_app.main(["--device", "cuda"]) == 0
    commands = [call[0] for call in calls]
    assert commands[-3][-5:] == ["up", "-d", "--wait", "db", "rabbitmq"]
    assert commands[-2][-1] == start_app.PREPARE_DATABASE
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


@pytest.mark.parametrize(
    "tables,revisions,allowed,upgrade",
    [
        ([], [], True, True),
        ("current", ["current"], True, False),
        ("extra", ["current"], False, False),
        (["alembic_version", "images"], ["old"], False, False),
        (["unknown"], [], False, False),
        (["alembic_version"], [], False, False),
    ],
)
def test_database_guard_never_migrates_existing_schema(
    monkeypatch, tables, revisions, allowed, upgrade
):
    import alembic.command
    import alembic.config
    import alembic.script
    import sqlalchemy

    from backend.tables import Base

    if tables in ("current", "extra"):
        tables = [*Base.metadata.tables, "alembic_version"] + (
            ["jobs"] if tables == "extra" else []
        )

    connection = MagicMock()
    connection.execute.return_value.scalars.return_value.all.return_value = revisions
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value = connection
    monkeypatch.setattr(sqlalchemy, "create_engine", MagicMock(return_value=engine))
    inspector = MagicMock()
    inspector.get_table_names.return_value = tables
    monkeypatch.setattr(sqlalchemy, "inspect", MagicMock(return_value=inspector))
    script = MagicMock()
    script.get_current_head.return_value = "current"
    monkeypatch.setattr(
        alembic.script.ScriptDirectory, "from_config", MagicMock(return_value=script)
    )
    monkeypatch.setattr(alembic.config, "Config", MagicMock())
    migrate = MagicMock()
    monkeypatch.setattr(alembic.command, "upgrade", migrate)
    monkeypatch.setenv("APP_DATABASE_URL", "postgresql://test:test@localhost/test")
    monkeypatch.setenv("APP_BROKER_URL", "amqp://test:test@localhost/test")
    if allowed:
        exec(start_app.PREPARE_DATABASE, {})
    else:
        with pytest.raises(SystemExit, match="backup"):
            exec(start_app.PREPARE_DATABASE, {})
    assert migrate.call_count == int(upgrade)
