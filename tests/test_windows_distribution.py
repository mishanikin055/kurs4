"""Verify a distributable source allowlist without collecting private local files."""

import io
import zipfile

import pytest

from scripts import build_windows


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
