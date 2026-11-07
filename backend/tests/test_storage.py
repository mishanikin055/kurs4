from io import BytesIO
from uuid import uuid4

import pytest
from PIL import Image
from starlette.datastructures import UploadFile

from backend.schemas import ImageResult
from backend.settings import Settings
from backend.storage import (
    BundleTooLarge,
    UploadError,
    build_artifacts,
    prepare_image,
    save_detection_image,
)


@pytest.mark.parametrize("padding", [0, 0.1, 0.25])
def test_bundle_crops_match_florence_input_pixels(settings, tmp_path, padding):
    import json

    from backend.models import crop_object

    image_id = uuid4()
    source = tmp_path
    _, _, inputs = prepare_image(upload(image_bytes()), settings)
    for name, content in inputs.items():
        (source / name).write_bytes(content)
    image = Image.new("RGB", (16, 12))
    image.putdata([(x * 10, y * 10, 100) for y in range(12) for x in range(16)])
    image.save(source / "normalized.png")
    objects = [
        {"object_id": uuid4(), "bbox": bbox, "label": "car"}
        for bbox in ([0.1, 0.1, 5.2, 5.2], [4, 3, 12, 9], [0, 0, 16, 12])
    ]
    result = ImageResult(
        id=image_id,
        width=16,
        height=12,
        models={stage: {} for stage in ("detection", "classification", "annotation")},
        detection={"objects": objects},
        classification={
            "objects": [
                {"object_id": obj["object_id"], "object_caption": "  A car.  "} for obj in objects
            ]
        },
        annotation={"raw_caption": "A car."},
    )
    manifest, artifacts = build_artifacts(result, source, settings, crop_padding=padding)
    assert json.loads(artifacts["results.json"]) == manifest
    assert manifest["crops"] == [f"{obj['object_id']}.png" for obj in objects]
    for obj, name in zip(objects, manifest["crops"], strict=True):
        expected = crop_object(image, obj["bbox"], padding)
        with Image.open(BytesIO(artifacts[f"crops/{name}"])) as actual:
            assert actual.format == "PNG"
            assert actual.size == expected.size
            assert actual.tobytes() == expected.tobytes()
    assert ImageResult.model_validate(manifest) == result


@pytest.fixture
def settings(tmp_path):
    return Settings(
        s3_secret_key="testing",
        broker_url="amqp://unused:unused@localhost/unused",
    )


def image_bytes(format="PNG", *, size=(16, 12), mode="RGB", exif=None):
    stream = BytesIO()
    image = Image.new(mode, size)
    image.save(stream, format=format, **({"exif": exif} if exif else {}))
    return stream.getvalue()


def upload(data, filename="image.png"):
    return UploadFile(file=BytesIO(data), filename=filename)


def test_only_normalized_input_is_staged(settings):
    data = image_bytes(mode="RGBA")
    width, height, artifacts = prepare_image(upload(data, "../../image.png"), settings)
    assert set(artifacts) == {"normalized.png"}
    assert (width, height) == (16, 12)
    with Image.open(BytesIO(artifacts["normalized.png"])) as image:
        assert image.mode == "RGB"
        assert image.size == (16, 12)
        assert not image.getexif()


def scratch_image(settings, tmp_path, size=(16, 12)):
    _, _, artifacts = prepare_image(upload(image_bytes(size=size)), settings)
    path = tmp_path / "normalized.png"
    path.write_bytes(artifacts["normalized.png"])
    return path


@pytest.mark.parametrize("empty", [False, True])
def test_detection_image_is_saved_without_changing_source(settings, tmp_path, empty):
    source = scratch_image(settings, tmp_path, (64, 48))
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


def test_detection_image_disk_failure_cleans_temporary_file(settings, tmp_path, monkeypatch):
    source = scratch_image(settings, tmp_path)

    def fail(*args):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("backend.storage.os.replace", fail)
    with pytest.raises(OSError):
        save_detection_image(str(source), tmp_path / "detection.png", {"objects": []})
    assert not list(tmp_path.rglob(".detection-*"))


@pytest.mark.parametrize("size", [(960, 720), (720, 960), (24, 16)])
def test_detection_labels_keep_class_names_without_numbers(settings, tmp_path, monkeypatch, size):
    from copy import deepcopy

    from PIL import ImageDraw

    source = scratch_image(settings, tmp_path, size)
    before = source.read_bytes()
    payload = {
        "objects": [
            {"object_id": "second", "bbox": [0, 0, size[0], size[1]], "label": "car"},
            {
                "object_id": "first",
                "bbox": [size[0] - 3, size[1] - 3, size[0], size[1]],
                "label": "car",
            },
        ]
    }
    original_payload = deepcopy(payload)
    labels = []
    draw_text = ImageDraw.ImageDraw.text
    draw_rectangle = ImageDraw.ImageDraw.rectangle

    def record_rectangle(self, xy, **kwargs):
        assert kwargs.get("fill") is None, "Class labels must not have a filled background"
        return draw_rectangle(self, xy, **kwargs)

    def record_text(self, xy, text, **kwargs):
        labels.append(text)
        left, top, right, bottom = self.textbbox(xy, text, font=kwargs["font"])
        assert 0 <= left < size[0] and 0 <= top < size[1]
        if min(size) > 100:
            assert right <= size[0] and bottom <= size[1]
        return draw_text(self, xy, text, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record_text)
    monkeypatch.setattr(ImageDraw.ImageDraw, "rectangle", record_rectangle)
    destination = source.parent / "detection.png"
    save_detection_image(str(source), destination, payload)
    assert labels == ["car", "car"]
    assert payload == original_payload
    assert source.read_bytes() == before
    with Image.open(destination) as image:
        assert image.size == size


def test_exif_orientation_changes_normalized_dimensions(settings):
    exif = Image.Exif()
    exif[274] = 6
    data = image_bytes("JPEG", size=(30, 10), exif=exif)
    width, height, _ = prepare_image(upload(data, "rotation.jpg"), settings)
    assert (width, height) == (10, 30)


@pytest.mark.parametrize(
    "data,code",
    [(b"not an image", 422), (image_bytes("GIF"), 415), (image_bytes()[:55], 422)],
)
def test_invalid_images_leave_no_files(settings, data, code):
    with pytest.raises(UploadError) as error:
        prepare_image(upload(data), settings)
    assert error.value.status_code == code


def test_actual_bytes_limit(settings):
    settings.max_upload_bytes = 50
    with pytest.raises(UploadError, match="upload_too_large"):
        prepare_image(upload(image_bytes()), settings)


def test_pixel_limit(settings):
    settings.max_image_pixels = 100
    with pytest.raises(UploadError, match="image_too_large"):
        prepare_image(upload(image_bytes()), settings)


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
        prepare_image(upload(stream.getvalue()), settings)


def test_normalized_and_bundle_limits_are_enforced(settings):
    from backend.storage import check_artifacts

    settings.max_artifact_bytes = 10
    with pytest.raises(UploadError, match="normalized_image_too_large"):
        prepare_image(upload(image_bytes()), settings)
    settings.max_artifact_bytes = 100
    settings.max_bundle_bytes = 15
    with pytest.raises(BundleTooLarge, match="Bundle"):
        check_artifacts({"one": b"a" * 10, "two": b"b" * 10}, settings)
