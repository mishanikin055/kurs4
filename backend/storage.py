import json
import os
import tempfile
import warnings
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
from starlette.datastructures import UploadFile

from backend.models import crop_object
from backend.schemas import ImageResult
from backend.settings import Settings


class UploadError(Exception):
    def __init__(self, status_code: int, code: str):
        self.status_code = status_code
        self.code = code
        super().__init__(code)


def save_detection_image(normalized_path: str, destination: Path, detection: dict) -> Path:
    """Draw COCO boxes and class names without changing inputs or model responses."""
    directory = destination.parent
    directory.mkdir(parents=True, exist_ok=True)
    with Image.open(normalized_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=max(16, round(max(image.size) / 45)))
    for obj in detection["objects"]:
        x1, y1, x2, y2 = obj["bbox"]
        box = (int(x1), int(y1), min(image.width - 1, int(x2)), min(image.height - 1, int(y2)))
        draw.rectangle(box, outline="red", width=2)
        text = obj["label"]
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        scale = min(
            1,
            max(1, box[2] - box[0] - 4) / max(1, right - left),
            max(1, box[3] - box[1] - 4) / max(1, bottom - top),
        )
        label_font = ImageFont.load_default(size=max(1, int(font.size * scale)))
        left, top, right, bottom = draw.textbbox((0, 0), text, font=label_font)
        x = max(0, min(box[0] + 1, image.width - (right - left)))
        y = max(0, min(box[1] + 1, image.height - (bottom - top)))
        draw.text(
            (x - left, y - top),
            text,
            font=label_font,
            fill="white",
            stroke_width=1 if label_font.size >= 8 else 0,
            stroke_fill="black",
        )
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


class BundleTooLarge(ValueError):
    pass


def check_artifacts(artifacts: dict[str, bytes], settings: Settings) -> None:
    if any(len(data) > settings.max_artifact_bytes for data in artifacts.values()):
        raise BundleTooLarge("Artifact exceeds configured byte limit")
    if sum(map(len, artifacts.values())) > settings.max_bundle_bytes:
        raise BundleTooLarge("Bundle exceeds configured byte limit")


def prepare_image(upload: UploadFile, settings: Settings) -> tuple[int, int, dict[str, bytes]]:
    """Validate bounded input and normalize in memory; do not persist media on disk."""
    original = upload.file.read(settings.max_upload_bytes + 1)
    if len(original) > settings.max_upload_bytes:
        raise UploadError(413, "upload_too_large")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(original)) as image:
                if image.format not in ("JPEG", "PNG"):
                    raise UploadError(415, "unsupported_image_format")
                if image.width * image.height > settings.max_image_pixels:
                    raise UploadError(413, "image_too_large")
                if getattr(image, "n_frames", 1) != 1:
                    raise UploadError(415, "animated_image_not_supported")
                image.verify()
            with Image.open(BytesIO(original)) as image:
                image.load()
                normalized = ImageOps.exif_transpose(image).convert("RGB")
                normalized.info.clear()
                width, height = normalized.size
                stream = BytesIO()
                normalized.save(stream, format="PNG")
        artifacts = {"normalized.png": stream.getvalue()}
        check_artifacts(artifacts, settings)
        return width, height, artifacts
    except BundleTooLarge as exc:
        raise UploadError(413, "normalized_image_too_large") from exc
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise UploadError(413, "image_too_large") from exc
    except UnidentifiedImageError as exc:
        raise UploadError(422, "invalid_image") from exc
    except (OSError, SyntaxError, ValueError) as exc:
        raise UploadError(422, "invalid_image") from exc


def build_artifacts(
    result: ImageResult, source: Path, settings: Settings, *, crop_padding: float = 0.1
) -> tuple[dict, dict[str, bytes]]:
    """Prepare one bounded bundle using task-local scratch, never a permanent directory."""
    artifacts = {}
    manifest = result.model_dump(mode="json", exclude_none=True)
    manifest["crops"] = []
    with Image.open(source / "normalized.png") as source_image:
        image = source_image.convert("RGB")
        for obj in result.detection.objects:
            name = f"{obj.object_id}.png"
            stream = BytesIO()
            crop_object(image, obj.bbox, crop_padding).save(stream, format="PNG")
            artifacts[f"crops/{name}"] = stream.getvalue()
            check_artifacts(artifacts, settings)
            manifest["crops"].append(name)
    artifacts["results.json"] = json.dumps(
        manifest, ensure_ascii=False, indent=2, allow_nan=False
    ).encode("utf-8")
    check_artifacts(artifacts, settings)
    return manifest, artifacts
