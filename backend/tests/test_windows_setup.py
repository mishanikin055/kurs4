"""Distribution boundaries and Windows orchestration without OS installation or models."""

import io
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import windows_setup


@pytest.fixture
def windows(monkeypatch, tmp_path):
    monkeypatch.setattr(windows_setup, "ROOT", tmp_path / "app")
    windows_setup.ROOT.mkdir()
    monkeypatch.setattr(windows_setup.sys, "platform", "win32")
    monkeypatch.setattr(windows_setup.platform, "machine", lambda: "AMD64")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.delenv("APP_FRONTEND_PORT", raising=False)
    monkeypatch.setattr(windows_setup, "docker_paths", lambda: [])
    return tmp_path


def test_start_opens_browser_only_after_preparation_and_health(windows, monkeypatch):
    calls = []
    monkeypatch.setattr(windows_setup, "ensure_docker", lambda: calls.append("docker"))
    monkeypatch.setattr(windows_setup.fetch_app_models, "main", lambda args: calls.append("models"))
    monkeypatch.setattr(windows_setup.start_app, "main", lambda args: calls.append(args) or 0)
    monkeypatch.setattr(windows_setup, "wait_for_api", lambda url: calls.append(url))
    monkeypatch.setattr(windows_setup.webbrowser, "open", lambda url: calls.append("browser"))
    assert windows_setup.main(["--device", "cuda"]) == 0
    assert calls == [
        "docker",
        "models",
        ["--device", "cuda", "--build-if-missing"],
        "http://localhost:8080",
        "browser",
    ]


def test_failed_launch_does_not_open_browser(windows, monkeypatch):
    monkeypatch.setattr(windows_setup, "ensure_docker", lambda: None)
    monkeypatch.setattr(windows_setup.fetch_app_models, "main", lambda args: None)
    monkeypatch.setattr(windows_setup.start_app, "main", lambda args: 1)
    browser = Mock()
    monkeypatch.setattr(windows_setup.webbrowser, "open", browser)
    assert windows_setup.main([]) == 1
    browser.assert_not_called()


def test_stop_never_installs_downloads_or_deletes(windows, monkeypatch):
    monkeypatch.setattr(windows_setup, "find_docker", lambda: "docker.exe")
    run = Mock()
    monkeypatch.setattr(windows_setup.subprocess, "run", run)
    assert windows_setup.main(["--action", "stop"]) == 0
    assert run.call_args.args[0] == [
        "docker.exe",
        "compose",
        "-p",
        "kurs4-app",
        "-f",
        "compose.app.yaml",
        "--profile",
        "processing",
        "stop",
    ]


def test_invalid_installer_signature_prevents_installation(windows, monkeypatch):
    monkeypatch.setattr(
        windows_setup.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"untrusted")
    )
    verify = Mock(side_effect=subprocess.CalledProcessError(1, "signature"))
    monkeypatch.setattr(windows_setup, "powershell", verify)
    with pytest.raises(subprocess.CalledProcessError):
        windows_setup.install_docker()
    verify.assert_called_once()
    assert "Get-AuthenticodeSignature" in verify.call_args.args[0]
    assert "Start-Process" not in verify.call_args.args[0]


def test_missing_docker_gets_installed_before_waiting(windows, monkeypatch):
    monkeypatch.setattr(windows_setup, "find_docker", Mock(side_effect=[None, "docker.exe"]))
    install = Mock()
    monkeypatch.setattr(windows_setup, "install_docker", install)
    monkeypatch.setattr(
        windows_setup.subprocess,
        "run",
        Mock(
            return_value=SimpleNamespace(
                stdout=json.dumps({"OSType": "linux", "Architecture": "x86_64"})
            )
        ),
    )
    windows_setup.ensure_docker()
    install.assert_called_once()


def test_wrong_docker_engine_stops_setup(windows, monkeypatch):
    monkeypatch.setattr(windows_setup, "find_docker", lambda: "docker.exe")
    monkeypatch.setattr(
        windows_setup.subprocess,
        "run",
        Mock(
            return_value=SimpleNamespace(
                stdout=json.dumps({"OSType": "windows", "Architecture": "x86_64"})
            )
        ),
    )
    with pytest.raises(RuntimeError, match="Linux containers"):
        windows_setup.ensure_docker()


def test_health_wait_does_not_accept_partial_readiness(monkeypatch):
    monkeypatch.setattr(windows_setup.time, "sleep", lambda _: None)
    responses = [
        io.BytesIO(b'{"rabbitmq":true,"elasticsearch":true,"s3":false}'),
        io.BytesIO(b'{"rabbitmq":true,"elasticsearch":true,"s3":true}'),
    ]
    request = Mock(side_effect=responses)
    monkeypatch.setattr(windows_setup.urllib.request, "urlopen", request)
    windows_setup.wait_for_api("http://localhost:8080")
    assert request.call_count == 2
