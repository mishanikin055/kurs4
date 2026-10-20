"""Save and publish uploads; read completed files and compatible preview labels."""

import json
from functools import cache
from pathlib import Path
from re import fullmatch
from uuid import UUID, uuid4

from starlette.datastructures import UploadFile

from backend.schemas import AcceptedImage
from backend.settings import Settings
from backend.storage import result_directory, save_image
from backend.tasks import publish_image


class BrokerUnavailable(Exception):
    pass


@cache
def compatible_labels() -> dict[str, set[str]]:
    directory = Path(__file__).parent
    classes = json.loads((directory / "imagenet_labels.json").read_text())["classes"]
    names = {item["synset"]: item["name"] for item in classes}
    mapping = json.loads((directory / "imagenet_coco_mapping.json").read_text())["categories"]
    return {coco: {names[synset] for synset in synsets} for coco, synsets in mapping.items()}


class ImageService:
    def __init__(self, settings: Settings, celery):
        self.settings = settings
        self.celery = celery

    def upload(self, file: UploadFile) -> AcceptedImage:
        image_id = uuid4()
        save_image(file, image_id, self.settings)
        try:
            publish_image(self.celery, str(image_id), self.settings.broker_timeout)
        except Exception as exc:
            # A lost confirm can mean successful delivery: retain inputs for the worker.
            raise BrokerUnavailable from exc
        return AcceptedImage(
            image_id=image_id,
            output_folder=f"results/{image_id}",
        )

    def result_file(self, image_id: UUID, filename: str) -> Path:
        names = {"original", "normalized.png", "detection.png", "results.json"}
        if filename not in names and not fullmatch(r"crops/[0-9a-f-]{36}\.png", filename):
            raise FileNotFoundError
        directory = result_directory(self.settings, image_id)
        path = (directory / filename).resolve()
        if (
            not (directory / "results.json").is_file()
            or not path.is_relative_to(directory)
            or not path.is_file()
        ):
            raise FileNotFoundError
        return path

    def result(self, image_id: UUID) -> dict:
        result = json.loads(self.result_file(image_id, "results.json").read_text(encoding="utf-8"))
        detected = {obj["object_id"]: obj["label"] for obj in result["detection"]["objects"]}
        for obj in result["classification"]["objects"]:
            coco = detected.get(obj["object_id"])
            label = obj.get("label")
            obj["refinement"] = (
                label if label != coco and label in compatible_labels().get(coco, set()) else None
            )
        return result
