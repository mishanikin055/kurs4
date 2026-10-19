from io import BytesIO
from uuid import uuid4

import pytest
from PIL import Image
from starlette.datastructures import UploadFile

from backend.settings import Settings
from backend.storage import UploadError, save_detection_image, save_image


@pytest.fixture
def settings(tmp_path):
    return Settings(
        database_url="postgresql+psycopg://unused:unused@localhost/unused",
        broker_url="amqp://unused:unused@localhost/unused",
        image_root=tmp_path,
    )


def image_bytes(format="PNG", *, size=(16, 12), mode="RGB", exif=None):
    stream = BytesIO()
    image = Image.new(mode, size)
    image.save(stream, format=format, **({"exif": exif} if exif else {}))
    return stream.getvalue()


def upload(data, filename="image.png"):
    return UploadFile(file=BytesIO(data), filename=filename)


def test_original_and_normalized_are_separate(settings):
    data = image_bytes(mode="RGBA")
    image_id = uuid4()
    result = save_image(upload(data, "../../image.png"), image_id, settings)
    directory = settings.image_root / str(image_id)
    assert (directory / "original").read_bytes() == data
    assert set(result) == {"id", "original_path", "normalized_path", "width", "height"}
    with Image.open(directory / "normalized.png") as image:
        assert image.mode == "RGB"
        assert image.size == (16, 12)
        assert not image.getexif()


@pytest.mark.parametrize("empty", [False, True])
def test_detection_image_is_saved_without_changing_source(settings, empty):
    result = save_image(upload(image_bytes(size=(64, 48))), uuid4(), settings)
    source = settings.image_root / str(result["id"]) / "normalized.png"
    before = source.read_bytes()
    destination = source.parent / "detection.png"
    payload = {"objects": [] if empty else [{"bbox": [10.5, 12.5, 64, 48], "label": "cat"}]}
    output = save_detection_image(str(source), destination, payload)
    assert output == destination
    assert source.read_bytes() == before
    with Image.open(output) as image:
        assert image.size == (64, 48)
        assert image.getpixel((63, 40)) == ((0, 0, 0) if empty else (255, 0, 0))
    assert not list(output.parent.glob(".detection-*"))
    assert save_detection_image(str(source), destination, payload) == output


def test_detection_image_disk_failure_cleans_temporary_file(settings, monkeypatch):
    result = save_image(upload(image_bytes()), uuid4(), settings)

    def fail(*args):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("backend.storage.os.replace", fail)
    with pytest.raises(OSError):
        save_detection_image(
            result["normalized_path"], settings.image_root / "detection.png", {"objects": []}
        )
    assert not list(settings.image_root.rglob(".detection-*"))


def test_exif_orientation_changes_normalized_dimensions(settings):
    exif = Image.Exif()
    exif[274] = 6
    data = image_bytes("JPEG", size=(30, 10), exif=exif)
    result = save_image(upload(data, "rotation.jpg"), uuid4(), settings)
    assert (result["width"], result["height"]) == (10, 30)


@pytest.mark.parametrize(
    "data,code",
    [(b"not an image", 422), (image_bytes("GIF"), 415), (image_bytes()[:55], 422)],
)
def test_invalid_images_leave_no_files(settings, data, code):
    with pytest.raises(UploadError) as error:
        save_image(upload(data), uuid4(), settings)
    assert error.value.status_code == code
    assert list(settings.image_root.iterdir()) == []


def test_actual_bytes_limit(settings):
    settings.max_upload_bytes = 50
    with pytest.raises(UploadError, match="upload_too_large"):
        save_image(upload(image_bytes()), uuid4(), settings)
    assert not list(settings.image_root.iterdir())


def test_pixel_limit(settings):
    settings.max_image_pixels = 100
    with pytest.raises(UploadError, match="image_too_large"):
        save_image(upload(image_bytes()), uuid4(), settings)


def test_animated_png_is_rejected(settings):
    stream = BytesIO()
    Image.new("RGB", (5, 5)).save(
        stream,
        format="PNG",
        save_all=True,
        append_images=[Image.new("RGB", (5, 5), "red")],
        duration=100,
        loop=0,
    )
    with pytest.raises(UploadError, match="animated_image_not_supported"):
        save_image(upload(stream.getvalue()), uuid4(), settings)


def test_disk_failure_is_not_reported_as_bad_input(settings, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("backend.storage.os.rename", fail)
    with pytest.raises(OSError):
        save_image(upload(image_bytes()), uuid4(), settings)
    assert not list(settings.image_root.iterdir())
