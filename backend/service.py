"""Commit uploads before broker publication; serve completed database bundles."""

import json
from copy import deepcopy
from functools import cache
from pathlib import Path
from re import fullmatch
from uuid import UUID, uuid4

from starlette.datastructures import UploadFile

from backend.repository import ImageRepository
from backend.schemas import AcceptedImage
from backend.settings import Settings
from backend.storage import prepare_image
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
    def __init__(self, settings: Settings, celery, repository: ImageRepository):
        self.settings = settings
        self.celery = celery
        self.repository = repository

    def upload(self, file: UploadFile) -> AcceptedImage:
        image_id = uuid4()
        width, height, artifacts = prepare_image(file, self.settings)
        self.repository.create_input(image_id, width, height, artifacts)
        try:
            publish_image(self.celery, str(image_id), self.settings.broker_timeout)
        except Exception as exc:
            # A lost confirm can mean successful delivery: retain inputs for the worker.
            raise BrokerUnavailable from exc
        return AcceptedImage(image_id=image_id)

    def result_file(self, image_id: UUID, filename: str) -> tuple[str, bytes]:
        names = {"results.json"}
        if filename not in names and not fullmatch(r"crops/[0-9a-f-]{36}\.png", filename):
            raise FileNotFoundError
        artifact = self.repository.artifact(image_id, filename)
        return artifact.content_type, artifact.content

    def bundle(self, image_id: UUID) -> dict[str, tuple[str, bytes]]:
        return self.repository.bundle(image_id)

    def result(self, image_id: UUID) -> dict:
        record = self.repository.get(image_id)
        if record is None:
            raise FileNotFoundError
        result = deepcopy(record.manifest)
        detected = {obj["object_id"]: obj["label"] for obj in result["detection"]["objects"]}
        for obj in result["classification"]["objects"]:
            if "object_caption" in obj:
                continue
            coco = detected.get(obj["object_id"])
            label = obj.get("label")
            obj["refinement"] = (
                label if label != coco and label in compatible_labels().get(coco, set()) else None
            )
        return result
