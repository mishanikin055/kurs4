"""Build the Windows EXE and a clean release ZIP on Windows or WSL; stdlib only."""

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION = "3.14.10"
# Published at https://www.python.org/downloads/release/python-31410/
PYTHON_SHA256 = "90905e64a840557d8c1bc4f5b74db3d7ee85e3ffb217611cac1c7813acea4159"


def release_files(root=ROOT):
    """An allowlist: never walk user data, caches, secrets or old SQL modules."""
    manifest = (root / "packaging/windows/files.txt").read_text(encoding="utf-8")
    for name in manifest.splitlines():
        path = root / name
        if not name or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Invalid release path: {name}")
        if not path.is_file():
            raise ValueError(f"Missing release file: {name}")
        yield path


def source_archive():
    checkpoints = json.loads((ROOT / "backend/checkpoints.json").read_text(encoding="utf-8"))
    selected = {
        f"{stage}/configs/models.json": spec["model_name"] for stage, spec in checkpoints.items()
    }
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in release_files():
            name = path.relative_to(ROOT).as_posix()
            if name in selected:
                config = json.loads(path.read_text(encoding="utf-8"))
                models = [row for row in config["models"] if row["id"] == selected[name]]
                if len(models) != 1:
                    raise ValueError(f"Expected one application model in {name}")
                content = {"models": models}
                archive.writestr(name, json.dumps(content, ensure_ascii=False, indent=2) + "\n")
            else:
                archive.write(path, name)
    return stream.getvalue()


def windows_path(path):
    if os.name == "nt":
        return str(path.resolve())
    return subprocess.check_output(["wslpath", "-w", str(path.resolve())], text=True).strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Release ZIP path")
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Require adjacent offline assets; never download",
    )
    parser.add_argument(
        "--python-zip", type=Path, help="Previously downloaded official embedded ZIP"
    )
    args = parser.parse_args(argv)
    compiler = (
        Path(os.environ.get("WINDIR", r"C:\Windows"))
        / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
        if os.name == "nt"
        else Path("/mnt/c/Windows/Microsoft.NET/Framework64/v4.0.30319/csc.exe")
    )
    if not compiler.is_file():
        parser.error("Run on Windows x64 or WSL with .NET Framework 4.8 installed.")
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # WSL /tmp is not a reliable working directory for the Windows compiler.
    with tempfile.TemporaryDirectory(
        prefix="image-analysis-build-", dir=args.output.parent
    ) as directory:
        work = Path(directory)
        python_zip = args.python_zip or work / "python.zip"
        if not args.python_zip:
            urllib.request.urlretrieve(
                f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-embed-amd64.zip",
                python_zip,
            )
        runtime = python_zip.read_bytes()
        if hashlib.sha256(runtime).hexdigest() != PYTHON_SHA256:
            raise ValueError("Embedded Python SHA256 mismatch")
        source = source_archive()
        payload = work / "payload.zip"
        with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as out:
            with zipfile.ZipFile(io.BytesIO(source)) as app:
                for entry in app.infolist():
                    out.writestr(entry.filename, app.read(entry))
            with zipfile.ZipFile(io.BytesIO(runtime)) as python:
                for entry in python.infolist():
                    if entry.filename == "python314._pth":
                        out.writestr("python/" + entry.filename, "python314.zip\n.\n..\n")
                    else:
                        out.writestr("python/" + entry.filename, python.read(entry))
        exe = work / "Image Analysis.exe"
        response = work / "compile.rsp"
        response.write_text(
            "\n".join(
                [
                    "/nologo",
                    "/target:winexe",
                    "/platform:x64",
                    "/optimize+",
                    "/define:OFFLINE" if args.offline_only else "/define:ONLINE",
                    "/reference:System.Windows.Forms.dll",
                    "/reference:System.Drawing.dll",
                    "/reference:System.IO.Compression.dll",
                    f'/win32manifest:"{windows_path(ROOT / "packaging/windows/app.manifest")}"',
                    f'/resource:"{windows_path(payload)}",payload.zip',
                    f'/out:"{windows_path(exe)}"',
                    f'"{windows_path(ROOT / "packaging/windows/Launcher.cs")}"',
                ]
            ),
            encoding="utf-8-sig",
        )
        subprocess.run([str(compiler), "@" + windows_path(response)], cwd=work, check=True)
        contents = {
            "Image Analysis.exe": exe.read_bytes(),
            "README.txt": (ROOT / "packaging/windows/README.txt").read_bytes(),
            "app-source.zip": source,
        }
        sums = "".join(
            f"{hashlib.sha256(data).hexdigest()}  {name}\n" for name, data in contents.items()
        )
        contents["SHA256SUMS.txt"] = sums.encode("utf-8")
        temporary = work / "release.zip"
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, data in contents.items():
                archive.writestr("Image Analysis/" + name, data)
        shutil.copyfile(temporary, args.output)
    checksum = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{checksum}  {args.output.name}\n",
        encoding="utf-8",
    )
    print(f"Built: {args.output} ({args.output.stat().st_size:,} bytes)\nSHA256: {checksum}")


if __name__ == "__main__":
    main()
