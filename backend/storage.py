import hashlib
import os
import shutil
import tempfile
import warnings
from pathlib import Path
from uuid import UUID

from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.datastructures import UploadFile

from backend.settings import Settings


class UploadError(Exception):
    def __init__(self, status_code: int, code: str):
        self.status_code = status_code
        self.code = code
        super().__init__(code)


def save_image(upload: UploadFile, image_id: UUID, settings: Settings) -> dict:
    root = settings.image_root
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".upload-", dir=root))
    destination = root / str(image_id)
    try:
        original = temporary / "original"
        size = 0
        digest = hashlib.sha256()
        with original.open("xb") as stream:
            while block := upload.file.read(1024 * 1024):
                size += len(block)
                if size > settings.max_upload_bytes:
                    raise UploadError(413, "upload_too_large")
                digest.update(block)
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())

        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(original) as image:
                if image.format not in ("JPEG", "PNG"):
                    raise UploadError(415, "unsupported_image_format")
                mime = Image.MIME[image.format]
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
        normalized_digest = hashlib.sha256((temporary / "normalized.png").read_bytes()).hexdigest()
        os.rename(temporary, destination)
        directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        filename = (upload.filename or "image").replace("\\", "/").rsplit("/", 1)[-1][:255]
        filename = (
            "".join(character for character in filename if character.isprintable()) or "image"
        )
        return {
            "id": image_id,
            "original_filename": filename,
            "original_path": str(destination / "original"),
            "original_sha256": digest.hexdigest(),
            "original_mime": mime,
            "original_size_bytes": size,
            "normalized_path": str(destination / "normalized.png"),
            "normalized_sha256": normalized_digest,
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
