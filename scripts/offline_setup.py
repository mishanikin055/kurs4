"""Install the application's offline assets; no network calls or model execution."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from scripts import fetch_app_models, start_app

ROOT = Path(__file__).resolve().parents[1]


def manifest(bundle):
    return json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))


def verified_file(bundle, name):
    path = bundle / name
    if not path.resolve().is_relative_to(bundle.resolve()):
        raise ValueError("Invalid offline asset path")
    expected = manifest(bundle)["files"][name]
    if not path.is_file() or fetch_app_models.digest(path) != expected:
        raise RuntimeError(f"Файл офлайн-поставки отсутствует или повреждён: {name}")
    return path


def images_ready(bundle):
    images = manifest(bundle)["images"]
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", *images],
            check=True,
            text=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        return False
    actual = json.loads(result.stdout)
    return len(actual) == len(images) and all(
        row["Id"] == expected for row, expected in zip(actual, images.values(), strict=True)
    )


def load_images(bundle):
    if images_ready(bundle):
        return
    archive = verified_file(bundle, "images.tar.gz")
    subprocess.run(["docker", "load", "--input", str(archive)], check=True)
    if not images_ready(bundle):
        raise RuntimeError("Загруженные Docker-образы не совпадают с офлайн-поставкой.")


def prepare_models(bundle):
    for target, expected, _, _ in fetch_app_models.sources():
        if target.exists():
            if not target.is_file() or fetch_app_models.digest(target) != expected:
                raise RuntimeError(
                    "Установленный файл модели отличается; сохраните его и проверьте журнал."
                )
            continue
        source = bundle / "models" / target.relative_to(ROOT)
        if not source.is_file() or fetch_app_models.digest(source) != expected:
            raise RuntimeError("В офлайн-поставке отсутствует или повреждён файл модели.")
        new_directory = not target.parent.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        if new_directory:
            target.parent.chmod(0o755)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=target.parent, prefix=".offline-", delete=False
            ) as out:
                temporary = Path(out.name)
                with source.open("rb") as stream:
                    shutil.copyfileobj(stream, out, 1024 * 1024)
            if fetch_app_models.digest(temporary) != expected:
                raise RuntimeError("Ошибка копирования модели из офлайн-поставки.")
            # The worker runs as UID 1000; the installing Linux user can have another UID.
            temporary.chmod(0o644)
            os.replace(temporary, target)
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--detect-s3", action="store_true")
    args = parser.parse_args(argv)
    if args.detect_s3:
        config = json.load(sys.stdin)
        print(
            "external"
            if start_app.external_s3_settings(config["services"]["backend"]["environment"])
            else "local"
        )
        return
    if args.bundle is None:
        parser.error("--bundle is required")
    prepare_models(args.bundle)
    start_app.prepare_secrets()
    start_app.prepare_lock()
    start_app.check_weights()
    print("Офлайн-модели и настройки подготовлены.")


if __name__ == "__main__":
    main()
