"""Bundle six application images, three final models and signed Windows installers."""

import argparse
import gzip
import io
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

from scripts import build_windows, fetch_app_models, start_app

ROOT = Path(__file__).resolve().parents[1]
INSTALLERS = {
    "Docker Desktop Installer.exe": "cd6273d9eaa4f07fa7593fc6813d37e1004381cebc3f26047a2554fa5db24388",
    "wsl.x64.msi": "ae977286974255bd26bf925f00f40f5692c6b2084155cbc0563c09aea091f211",
}


def digest(path):
    return fetch_app_models.digest(path)


def export_images(images, destination):
    state = destination.with_suffix(".json")
    if destination.is_file() and state.is_file() and json.loads(state.read_text()) == images:
        return
    temporary = destination.with_suffix(".part")
    try:
        with subprocess.Popen(["docker", "save", *images], stdout=subprocess.PIPE) as process:
            with temporary.open("wb") as raw:
                with gzip.GzipFile(
                    filename="", mode="wb", fileobj=raw, compresslevel=1, mtime=0
                ) as out:
                    shutil.copyfileobj(process.stdout, out, 4 * 1024 * 1024)
            if process.wait():
                raise RuntimeError("docker save failed")
        temporary.replace(destination)
        state.write_text(json.dumps(images))
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--installers", type=Path, required=True)
    parser.add_argument("--python-zip", type=Path)
    args = parser.parse_args(argv)
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    bundle = work / "Image Analysis"
    offline = bundle / "offline"
    offline.mkdir(parents=True, exist_ok=True)
    for name, expected in INSTALLERS.items():
        source = args.installers / name
        if digest(source) != expected:
            raise ValueError(f"Installer SHA256 mismatch: {name}")
        target = offline / "installers" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or digest(target) != expected:
            shutil.copyfile(source, target)
    compose = [
        "docker",
        "compose",
        "-p",
        "kurs4-app",
        "-f",
        "compose.app.yaml",
        "--profile",
        "processing",
    ]
    config = json.loads(subprocess.check_output(compose + ["config", "--format", "json"], cwd=ROOT))
    version = json.loads((ROOT / "backend/checkpoints.json").read_text())
    assert set(version) == {"detection", "annotation"}
    # The source module has no imports or side effects other than its version constant.
    from backend import __version__

    images = {}
    for service in ("backend", "worker", "frontend", "s3", "elasticsearch", "rabbitmq"):
        image = config["services"][service]["image"]
        if service in {"elasticsearch", "rabbitmq"}:
            alias = f"kurs4-offline-{service}:{__version__}"
            subprocess.run(["docker", "tag", image, alias], check=True)
            image = alias
        row = json.loads(subprocess.check_output(["docker", "image", "inspect", image]))[0]
        if row["Os"] != "linux" or row["Architecture"] != "amd64":
            raise ValueError(f"Expected linux/amd64 image: {image}")
        images[image] = row["Id"]
    print("Exporting six application images…", flush=True)
    export_images(images, offline / "images.tar.gz")
    start_app.check_weights()
    model_files = []
    for path, expected, _, _ in fetch_app_models.sources():
        target = offline / "models" / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or digest(target) != expected:
            shutil.copyfile(path, target)
        if digest(target) != expected:
            raise ValueError(f"Model copy mismatch: {target.name}")
        model_files.append(target)
    small_zip = work / "windows-launcher.zip"
    build_args = ["--output", str(small_zip), "--offline-only"]
    if args.python_zip:
        build_args += ["--python-zip", str(args.python_zip)]
    build_windows.main(build_args)
    with zipfile.ZipFile(small_zip) as release:
        (bundle / "Image Analysis.exe").write_bytes(
            release.read("Image Analysis/Image Analysis.exe")
        )
        source = release.read("Image Analysis/app-source.zip")
    app = bundle / "app"
    with zipfile.ZipFile(io.BytesIO(source)) as archive:
        archive.extractall(app)
        source_files = [app / name for name in archive.namelist()]
    shutil.copyfile(ROOT / "packaging/start-linux.sh", bundle / "start-linux.sh")
    (bundle / "start-linux.sh").chmod(0o755)
    shutil.copyfile(ROOT / "packaging/README-offline.txt", bundle / "README.txt")
    files = {"images.tar.gz": digest(offline / "images.tar.gz")}
    files.update({"installers/" + name: sha for name, sha in INSTALLERS.items()})
    (offline / "manifest.json").write_text(
        json.dumps({"version": __version__, "images": images, "files": files}, indent=2) + "\n"
    )
    (offline / "images.txt").write_text(
        "".join(f"{sha} {image}\n" for image, sha in images.items())
    )
    (offline / "images.sha256").write_text(f"{files['images.tar.gz']}  images.tar.gz\n")
    (offline / "backend-image.txt").write_text(config["services"]["backend"]["image"] + "\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print("Creating offline distribution ZIP…", flush=True)
    distribution = [
        bundle / "Image Analysis.exe",
        bundle / "start-linux.sh",
        bundle / "README.txt",
        *source_files,
        *model_files,
        offline / "images.tar.gz",
        *(offline / "installers" / name for name in INSTALLERS),
        *(
            offline / name
            for name in ("manifest.json", "images.txt", "images.sha256", "backend-image.txt")
        ),
    ]
    with zipfile.ZipFile(args.output, "w", allowZip64=True) as archive:
        for path in distribution:
            # Images and installers are already compressed; avoid another temporary 8+ GB copy.
            info = zipfile.ZipInfo.from_file(
                path, "Image Analysis/" + path.relative_to(bundle).as_posix()
            )
            info.compress_type = zipfile.ZIP_STORED
            # Large chunks also avoid a DrvFS syscall for every 8 KiB on a Windows output disk.
            with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as target:
                shutil.copyfileobj(source, target, 4 * 1024 * 1024)
    checksum = digest(args.output)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{checksum}  {args.output.name}\n"
    )
    print(
        f"Built: {args.output}\nBytes: {args.output.stat().st_size}\nSHA256: {checksum}", flush=True
    )


if __name__ == "__main__":
    main()
