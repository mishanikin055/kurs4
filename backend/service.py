"""Validate the upload, save its input files and publish directly to RabbitMQ."""

from pathlib import Path
from uuid import UUID, uuid4

from starlette.datastructures import UploadFile

from backend.schemas import AcceptedImage
from backend.settings import Settings
from backend.storage import result_directory, save_image
from backend.tasks import publish_image


class BrokerUnavailable(Exception):
    pass


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
        if filename not in names:
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
