"""Offline asset installation never downloads, and preserves existing models."""

import hashlib
import json
import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import offline_setup, start_app, windows_setup


def test_offline_models_install_without_network_and_preserve_existing(tmp_path, monkeypatch):
    root = tmp_path / "app"
    bundle = tmp_path / "offline"
    target = root / "annotation/models/final/weights"
    source = bundle / "models" / target.relative_to(root)
    source.parent.mkdir(parents=True)
    source.write_bytes(b"weights")
    monkeypatch.setattr(offline_setup, "ROOT", root)
    expected = hashlib.sha256(b"weights").hexdigest()
    monkeypatch.setattr(
        offline_setup.fetch_app_models, "sources", lambda: [(target, expected, None, {})]
    )
    monkeypatch.setattr(
        offline_setup.fetch_app_models.urllib.request,
        "urlopen",
        Mock(side_effect=AssertionError("Network is forbidden")),
    )
    offline_setup.prepare_models(bundle)
    assert target.read_bytes() == b"weights"
    assert target.stat().st_mode & 0o777 == 0o644
    source.unlink()
    offline_setup.prepare_models(bundle)
    target.write_bytes(b"user-data")
    with pytest.raises(RuntimeError, match="Установленный файл"):
        offline_setup.prepare_models(bundle)
    assert target.read_bytes() == b"user-data"


def test_bad_offline_model_is_never_published(tmp_path, monkeypatch):
    target = tmp_path / "app/detection/models/final/weights"
    source = tmp_path / "offline/models/detection/models/final/weights"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"wrong")
    monkeypatch.setattr(offline_setup, "ROOT", tmp_path / "app")
    monkeypatch.setattr(
        offline_setup.fetch_app_models, "sources", lambda: [(target, "0" * 64, None, {})]
    )
    with pytest.raises(RuntimeError, match="повреждён"):
        offline_setup.prepare_models(tmp_path / "offline")
    assert not target.exists()


def test_offline_images_are_verified_before_docker_load(tmp_path, monkeypatch):
    archive = tmp_path / "images.tar.gz"
    archive.write_bytes(b"wrong")
    (tmp_path / "manifest.json").write_text(json.dumps({"files": {archive.name: "0" * 64}}))
    monkeypatch.setattr(offline_setup, "images_ready", lambda _: False)
    run = Mock()
    monkeypatch.setattr(offline_setup.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="повреждён"):
        offline_setup.load_images(tmp_path)
    run.assert_not_called()


def test_preinstalled_images_must_match_release_ids(tmp_path, monkeypatch):
    (tmp_path / "manifest.json").write_text(json.dumps({"images": {"app:1": "sha256:release"}}))
    run = Mock(return_value=SimpleNamespace(stdout='[{"Id":"sha256:other"}]'))
    monkeypatch.setattr(offline_setup.subprocess, "run", run)
    assert not offline_setup.images_ready(tmp_path)
    run.return_value.stdout = '[{"Id":"sha256:release"}]'
    assert offline_setup.images_ready(tmp_path)
    offline_setup.load_images(tmp_path)
    assert all(call.args[0][1:3] == ["image", "inspect"] for call in run.call_args_list)


def test_windows_offline_start_never_fetches_models(monkeypatch, tmp_path):
    monkeypatch.setattr(windows_setup.sys, "platform", "win32")
    monkeypatch.setattr(windows_setup.platform, "machine", lambda: "AMD64")
    monkeypatch.setenv("IMAGE_ANALYSIS_OFFLINE_DIR", str(tmp_path))
    monkeypatch.setattr(windows_setup, "ensure_docker", lambda: None)
    monkeypatch.setattr(
        windows_setup.fetch_app_models, "main", Mock(side_effect=AssertionError("No downloads"))
    )
    prepared, loaded, start = Mock(), Mock(), Mock(return_value=0)
    monkeypatch.setattr(offline_setup, "prepare_models", prepared)
    monkeypatch.setattr(offline_setup, "load_images", loaded)
    monkeypatch.setattr(start_app, "main", start)
    monkeypatch.setattr(windows_setup, "wait_for_api", lambda _: None)
    monkeypatch.setattr(windows_setup.webbrowser, "open", lambda _: None)
    assert windows_setup.main([]) == 0
    prepared.assert_called_once_with(tmp_path)
    loaded.assert_called_once_with(tmp_path)
    start.assert_called_once_with(["--device", "cpu", "--offline"])


def test_offline_wsl_uses_local_msi_and_blocks_windows_update(monkeypatch, tmp_path):
    monkeypatch.setenv("IMAGE_ANALYSIS_OFFLINE_DIR", str(tmp_path))
    monkeypatch.setattr(offline_setup, "verified_file", lambda *args: tmp_path / "wsl.x64.msi")
    execute = Mock(side_effect=[None, subprocess.CalledProcessError(3010, "restart")])
    monkeypatch.setattr(windows_setup, "powershell", execute)
    with pytest.raises(RuntimeError, match="Перезагрузите"):
        windows_setup.setup_wsl()
    import base64
    import os

    script = base64.b64decode(os.environ["IMAGE_ANALYSIS_ELEVATED_SCRIPT"]).decode("utf-16le")
    assert "/LimitAccess" in script
    assert "msiexec" in script
    assert "--update" not in script
