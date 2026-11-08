"""Verify a distributable source allowlist without collecting private local files."""

import io
import json
import os
import shutil
import subprocess
import sys
import zipfile

import pytest

from scripts import build_windows, fetch_app_models


def test_release_manifest_excludes_private_and_legacy_data():
    paths = [p.relative_to(build_windows.ROOT).as_posix() for p in build_windows.release_files()]
    assert len(paths) == len(set(paths))
    assert {"scripts/windows_setup.py", "packaging/windows/Launcher.cs", "backend/main.py"} <= set(
        paths
    )
    assert not any(
        part in {".secrets", "models", "migrations", "storage", "data", "archive", "__pycache__"}
        for path in paths
        for part in path.split("/")
    )
    assert "backend/tables.py" not in paths
    with zipfile.ZipFile(io.BytesIO(build_windows.source_archive())) as archive:
        assert set(archive.namelist()) == set(paths)
        assert b"__version__" in archive.read("backend/__init__.py")


def test_release_manifest_never_collects_unlisted_files(tmp_path):
    manifest = tmp_path / "packaging/windows/files.txt"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("safe.py\n")
    (tmp_path / "safe.py").write_text("safe")
    (tmp_path / "credentials.json").write_text("private")
    assert list(build_windows.release_files(tmp_path)) == [tmp_path / "safe.py"]
    manifest.write_text("../outside.py\n")
    with pytest.raises(ValueError, match="Invalid release path"):
        list(build_windows.release_files(tmp_path))


def test_distribution_has_only_final_models_and_preserves_download_contract(tmp_path, monkeypatch):
    def downloads():
        return [
            (path.relative_to(fetch_app_models.ROOT), expected, url, spec)
            for path, expected, url, spec in fetch_app_models.sources()
        ]

    expected = downloads()
    with zipfile.ZipFile(io.BytesIO(build_windows.source_archive())) as archive:
        archive.extractall(tmp_path)
        for stage, model in (("detection", "rf_detr_small"), ("annotation", "florence2_base_ft")):
            config = json.loads(archive.read(f"{stage}/configs/models.json"))
            assert [row["id"] for row in config["models"]] == [model]
        assert not any(
            name.startswith(
                (
                    "scripts/detection/",
                    "scripts/annotation/",
                    "scripts/classification/",
                    "classification/",
                    "archive/",
                )
            )
            for name in archive.namelist()
        )
    monkeypatch.setattr(fetch_app_models, "ROOT", tmp_path)
    assert downloads() == expected


@pytest.mark.skipif(sys.platform != "linux", reason="POSIX launcher")
def test_linux_launcher_installs_and_starts_without_build_or_download(tmp_path):
    root = tmp_path / "bundle with spaces"
    (root / "app").mkdir(parents=True)
    offline = root / "offline"
    offline.mkdir()
    (offline / "images.txt").write_text("sha256:release app:1\n")
    (offline / "backend-image.txt").write_text("app:1\n")
    shutil.copyfile(build_windows.ROOT / "packaging/start-linux.sh", root / "start-linux.sh")
    commands = tmp_path / "commands.jsonl"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        f"#!{sys.executable}\n"
        "import sys,os,json\n"
        "args=sys.argv[1:]\n"
        "with open(os.environ['COMMAND_LOG'],'a') as f: f.write(json.dumps(args)+'\\n')\n"
        "if args[0]=='info': print('linux/x86_64')\n"
        "elif args[0]=='context': print('unix:///var/run/docker.sock')\n"
        "elif args[:2]==['image','inspect']: print('sha256:release')\n"
        "elif '--detect-s3' in args: sys.stdin.read(); print('local')\n"
        "elif 'config' in args: print('{}')\n"
    )
    docker.chmod(0o755)
    browser = bin_dir / "xdg-open"
    browser.write_text("#!/bin/sh\nexit 0\n")
    browser.chmod(0o755)
    env = dict(
        os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"], COMMAND_LOG=str(commands)
    )
    env.pop("DOCKER_HOST", None)
    subprocess.run(
        ["sh", str(root / "start-linux.sh"), "cpu"], check=True, env=env, capture_output=True
    )
    calls = [json.loads(line) for line in commands.read_text().splitlines()]
    assert not any("build" in args or args[0] in {"pull", "load"} for args in calls)
    starts = [args for args in calls if args[0] == "run" or "up" in args or "run" in args]
    assert len(starts) == 5
    assert all(args[args.index("--pull") + 1] == "never" for args in starts)
