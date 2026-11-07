"""Explicit, read-only-source import: python -m backend.import_legacy /legacy/results."""

import argparse
import json
from io import BytesIO
from pathlib import Path
from uuid import UUID

from PIL import Image
from sqlalchemy.exc import SQLAlchemyError

from backend.db import make_engine, make_sessions
from backend.repository import ImageRepository
from backend.schemas import ImageResult
from backend.settings import Settings
from backend.storage import check_artifacts


def read_bundle(directory: Path, settings: Settings) -> tuple[ImageResult, dict, dict[str, bytes]]:
    """Refuse unsafe paths, missing files and invalid crops; retain all source bytes."""
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("Bundle must be a regular directory")
    image_id = UUID(directory.name)
    if str(image_id) != directory.name:
        raise ValueError("Bundle directory must be a canonical UUID")
    artifacts = {}

    def read(name: str) -> bytes:
        path = directory / name
        if (
            path.is_symlink()
            or not path.is_file()
            or not path.resolve().is_relative_to(directory.resolve())
        ):
            raise ValueError(f"Missing or unsafe artifact: {name}")
        if path.stat().st_size > settings.max_artifact_bytes:
            raise ValueError(f"Artifact too large: {name}")
        if path.stat().st_size + sum(map(len, artifacts.values())) > settings.max_bundle_bytes:
            raise ValueError("Bundle too large")
        content = path.read_bytes()
        artifacts[name] = content
        check_artifacts(artifacts, settings)
        return content

    raw = read("results.json")
    manifest = json.loads(raw)
    result = ImageResult.model_validate(manifest)
    if result.id != image_id:
        raise ValueError("Manifest UUID does not match directory")
    if result.width * result.height > settings.max_image_pixels:
        raise ValueError("Image pixel limit exceeded")
    for name in ("original", "normalized.png", "detection.png"):
        content = read(name)
        with Image.open(BytesIO(content)) as image:
            if image.format not in (("JPEG", "PNG") if name == "original" else ("PNG",)):
                raise ValueError(f"Unsupported image format: {name}")
            if (
                image.width * image.height > settings.max_image_pixels
                or getattr(image, "n_frames", 1) != 1
            ):
                raise ValueError(f"Invalid image dimensions or animation: {name}")
            if name != "original" and image.size != (result.width, result.height):
                raise ValueError(f"Image dimensions differ from manifest: {name}")
            image.verify()
    crops = manifest.get("crops", [])
    expected = [f"{obj.object_id}.png" for obj in result.detection.objects]
    if "crops" in manifest and crops != expected:
        raise ValueError("Crop manifest does not match detected objects")
    crop_directory = directory / "crops"
    if crop_directory.is_symlink():
        raise ValueError("Unsafe crop directory")
    if crop_directory.exists() and (
        not crop_directory.is_dir() or {p.name for p in crop_directory.iterdir()} != set(crops)
    ):
        raise ValueError("Unlisted or missing crop artifacts")
    for name in crops:
        content = read(f"crops/{name}")
        with Image.open(BytesIO(content)) as image:
            if (
                image.format != "PNG"
                or image.width * image.height > settings.max_image_pixels
                or getattr(image, "n_frames", 1) != 1
            ):
                raise ValueError("Invalid crop PNG")
            image.verify()
    return result, manifest, artifacts


def import_bundle(directory: Path, sessions, settings: Settings) -> bool:
    """One bundle per transaction; return False only for an exact existing copy."""
    result, manifest, artifacts = read_bundle(directory, settings)
    with sessions.begin() as session:
        repo = ImageRepository(session)
        existing = repo.get(result.id)
        if existing is not None:
            previous = {name: content for name, (_, content) in repo.bundle(result.id).items()}
            if existing.manifest != manifest or previous != artifacts:
                raise ValueError("Conflicting completed image; refusing overwrite")
            return False
        # create_input refuses pending UUID conflicts through its primary key.
        repo.create_input(
            result.id,
            result.width,
            result.height,
            {name: artifacts[name] for name in ("original", "normalized.png")},
        )
        repo.publish(result, artifacts, manifest)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Импорт старых комплектов в БД без инференса и изменения источника."
    )
    parser.add_argument(
        "source", type=Path, help="Каталог results с подпапками UUID (только чтение)"
    )
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    if not source.is_dir():
        parser.error("source должен быть каталогом results")
    settings = Settings(broker_url="amqp://unused")
    engine = make_engine(settings.database_url.get_secret_value())
    counts = {"imported": 0, "unchanged": 0, "failed": 0}
    try:
        sessions = make_sessions(engine)
        for directory in sorted(source.iterdir()):
            try:
                imported = import_bundle(directory, sessions, settings)
                counts["imported" if imported else "unchanged"] += 1
            except (ValueError, OSError, SQLAlchemyError):
                # No SQL, connection URLs or arbitrary manifest text in CLI errors.
                counts["failed"] += 1
                print(
                    json.dumps(
                        {
                            "bundle": directory.name,
                            "error": "invalid_incomplete_or_conflicting_bundle",
                        }
                    )
                )
        print(json.dumps(counts))
    finally:
        engine.dispose()
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
