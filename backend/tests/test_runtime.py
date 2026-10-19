import socket
import sys
import threading
import time
from pathlib import Path
from unittest.mock import Mock

import pytest

from backend import models
from backend.processing import inference_lock
from backend.runtime import (
    MAX_MESSAGE,
    ResidentModelFailed,
    check_memory,
    receive,
    resident_models,
    send,
)
from backend.settings import Settings, model_specs


@pytest.fixture
def settings(tmp_path):
    return Settings(
        database_url="postgresql+psycopg://unused:unused@localhost/unused",
        broker_url="amqp://unused:unused@localhost/unused",
        lock_path=tmp_path / "inference.lock",
        detection_python=sys.executable,
        classification_python=sys.executable,
        annotation_python=sys.executable,
        stage_timeout=5,
        device="cpu",
    )


def request(settings, stage):
    return dict(model_specs(settings)[stage], stage=stage, model_root=str(settings.model_root))


def test_resident_processes_reused_and_stopped(settings):
    with resident_models(settings, module="backend.tests.mock_model") as runtime:
        pids = {stage: process.pid for stage, process in runtime.processes.items()}
        for _ in range(2):
            detection = runtime(request(settings, "detection"), settings, runtime.lock_fd, 3)
            assert detection["pid"] == pids["detection"]
            classification = runtime(
                dict(request(settings, "classification"), detection=detection["payload"]),
                settings,
                runtime.lock_fd,
                3,
            )
            assert classification["pid"] == pids["classification"]
            assert (
                classification["payload"]["objects"][0]["object_id"]
                == detection["payload"]["objects"][0]["object_id"]
            )
            annotation = runtime(request(settings, "annotation"), settings, runtime.lock_fd, 3)
            assert annotation["pid"] == pids["annotation"]
    assert all(p.poll() is not None for p in runtime.processes.values())
    with inference_lock(settings.lock_path, time.monotonic() + 1):
        pass


def test_startup_failure_cleans_partial_runtime(settings, monkeypatch):
    monkeypatch.setenv("MOCK_MODEL_MODE", "startup_error")
    with pytest.raises(RuntimeError, match="startup failed"):
        with resident_models(settings, module="backend.tests.mock_model"):
            pytest.fail("Must not consume with failed startup")
    with inference_lock(settings.lock_path, time.monotonic() + 1):
        pass


@pytest.mark.parametrize("mode", ["timeout", "process_died", "memory"])
def test_broken_runtime_requires_restart(settings, tmp_path, monkeypatch, mode):
    if mode == "timeout":
        monkeypatch.setenv("MOCK_MODEL_MODE", "slow")
        monkeypatch.setenv("MOCK_STARTED", str(tmp_path / "started"))
    with resident_models(settings, module="backend.tests.mock_model") as runtime:
        if mode == "process_died":
            runtime.processes["detection"].kill()
            runtime.processes["detection"].wait()
        if mode == "memory":
            monkeypatch.setattr("backend.runtime.check_memory", Mock(side_effect=MemoryError))
        with pytest.raises(ResidentModelFailed):
            runtime(request(settings, "detection"), settings, runtime.lock_fd, 0.1)
    assert all(p.poll() is not None for p in runtime.processes.values())


def test_protocol_rejects_large_response():
    left, right = socket.socketpair()
    with left, right:
        left.sendall((MAX_MESSAGE + 1).to_bytes(4, "big"))
        with pytest.raises(ValueError, match="message size"):
            receive(right, deadline=time.monotonic() + 1)


@pytest.mark.parametrize("stage", ["detection", "classification", "annotation"])
def test_server_loads_once_and_rejects_checkpoint_change(settings, tmp_path, monkeypatch, stage):
    loader = Mock(return_value=object())
    verify = Mock(return_value=tmp_path)
    predict = Mock(return_value={"payload": {"test": True}})
    monkeypatch.setattr(models, "verify_files", verify)
    monkeypatch.setattr(models, "load_detector", loader)
    monkeypatch.setattr(models, "load_classifier", loader)
    monkeypatch.setattr(models, "load_annotator", loader)
    monkeypatch.setattr(models, "predict", predict)
    monkeypatch.setitem(sys.modules, "torch", Mock())
    left, right = socket.socketpair()
    errors = []

    def serve():
        try:
            models.serve(right.detach())
        except EOFError:
            pass
        except BaseException as exc:
            errors.append(exc)

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        spec = request(settings, stage)
        send(left, spec)
        assert receive(left, deadline=time.monotonic() + 2)["ready"]
        for _ in range(2):
            send(left, dict(spec, image_path="unused"))
            assert receive(left, deadline=time.monotonic() + 2) == {"payload": {"test": True}}
        send(left, dict(spec, model_revision="changed"))
        assert receive(left, deadline=time.monotonic() + 2) == {
            "error": {"code": "model_not_configured"}
        }
        assert predict.call_count == 3  # One warmup and two requests.
        assert all(call.kwargs["loaded"] is loader.return_value for call in predict.call_args_list)
        loader.assert_called_once()
        verify.assert_called_once()
    finally:
        left.close()
        thread.join(timeout=2)
    assert not thread.is_alive()
    assert not errors


@pytest.mark.parametrize(
    "host,anon,raises", [(2048, 3000, False), (500, 3000, True), (2048, 4800, True)]
)
def test_memory_headroom(host, anon, raises, monkeypatch):
    values = {
        "/proc/meminfo": f"MemAvailable: {host * 1024} kB\n",
        "/sys/fs/cgroup/memory.max": str(5 * 1024**3),
        "/sys/fs/cgroup/memory.stat": f"anon {anon * 1024**2}\nkernel 0\n",
    }
    monkeypatch.setattr(Path, "read_text", lambda path: values[str(path)])
    monkeypatch.setattr(Path, "exists", lambda path: str(path) in values)
    if raises:
        with pytest.raises(MemoryError):
            check_memory()
    else:
        check_memory()
