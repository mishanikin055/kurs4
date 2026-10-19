import json
import os
import shutil
import tempfile
import warnings
from pathlib import Path
from uuid import UUID

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
from starlette.datastructures import UploadFile

from backend.schemas import ImageResult
from backend.settings import Settings


class UploadError(Exception):
    def __init__(self, status_code: int, code: str):
        self.status_code = status_code
        self.code = code
        super().__init__(code)


def save_detection_image(normalized_path: str, destination: Path, detection: dict) -> Path:
    """Draw COCO boxes without changing source pixels or the saved model responses."""
    directory = destination.parent
    directory.mkdir(parents=True, exist_ok=True)
    with Image.open(normalized_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for obj in detection["objects"]:
        x1, y1, x2, y2 = obj["bbox"]
        box = (int(x1), int(y1), min(image.width - 1, int(x2)), min(image.height - 1, int(y2)))
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]), obj["label"], fill="white", stroke_width=1, stroke_fill="black")
    descriptor, temporary = tempfile.mkstemp(prefix=".detection-", dir=directory)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            image.save(stream, format="PNG")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination


def save_image(upload: UploadFile, image_id: UUID, settings: Settings) -> dict:
    root = settings.image_root
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".upload-", dir=root))
    destination = root / str(image_id)
    try:
        original = temporary / "original"
        size = 0
        with original.open("xb") as stream:
            while block := upload.file.read(1024 * 1024):
                size += len(block)
                if size > settings.max_upload_bytes:
                    raise UploadError(413, "upload_too_large")
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())

        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(original) as image:
                if image.format not in ("JPEG", "PNG"):
                    raise UploadError(415, "unsupported_image_format")
                if image.width * image.height > settings.max_image_pixels:
                    raise UploadError(413, "image_too_large")
                if getattr(image, "n_frames", 1) != 1:
                    raise UploadError(415, "animated_image_not_supported")
                image.verify()
            with Image.open(original) as image:
                image.load()
                normalized = ImageOps.exif_transpose(image).convert("RGB")
                # Drop EXIF/text/ICC metadata from the normalized image.
                normalized.info.clear()
                width, height = normalized.size
                with (temporary / "normalized.png").open("xb") as stream:
                    normalized.save(stream, format="PNG")
                    stream.flush()
                    os.fsync(stream.fileno())
        os.rename(temporary, destination)
        directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return {
            "id": image_id,
            "original_path": str(destination / "original"),
            "normalized_path": str(destination / "normalized.png"),
            "width": width,
            "height": height,
        }
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise UploadError(413, "image_too_large") from exc
    except UnidentifiedImageError as exc:
        raise UploadError(422, "invalid_image") from exc
    except (OSError, SyntaxError, ValueError) as exc:
        # Disk errors must be reported as server errors, not corrupt input.
        if isinstance(exc, OSError) and exc.errno is not None:
            raise
        raise UploadError(422, "invalid_image") from exc
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def result_directory(settings: Settings, image_id: UUID) -> Path:
    root = settings.image_root.resolve()
    destination = (root / "results" / str(image_id)).resolve()
    if not destination.is_relative_to(root):
        raise OSError("Result directory outside image volume")
    return destination


def save_result(result: ImageResult, source: Path, destination: Path) -> None:
    """Publish the complete bundle by one directory rename, before the DB commit."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".result-", dir=destination.parent))
    try:
        for name in ("original", "normalized.png"):
            with (source / name).open("rb") as src, (temporary / name).open("xb") as dst:
                shutil.copyfileobj(src, dst)
                dst.flush()
                os.fsync(dst.fileno())
        save_detection_image(
            str(temporary / "normalized.png"),
            temporary / "detection.png",
            result.detection.model_dump(mode="json"),
        )
        with (temporary / "results.json").open("x", encoding="utf-8") as stream:
            json.dump(
                result.model_dump(mode="json", exclude_none=True),
                stream,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            stream.flush()
            os.fsync(stream.fileno())
        directory_fd = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.rename(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
