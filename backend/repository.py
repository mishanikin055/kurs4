"""Durable inputs and atomically published image bundles; callers own transactions."""

from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from backend.schemas import ImageResult
from backend.tables import ImageArtifact, ImageObject, ImageRecord


def content_type(filename: str) -> str:
    if filename.endswith(".png"):
        return "image/png"
    if filename == "results.json":
        return "application/json"
    return "application/octet-stream"


class ImageRepository:
    def __init__(self, session: Session):
        self.session = session

    def get(self, image_id: UUID) -> ImageRecord | None:
        return self.session.scalar(
            select(ImageRecord).where(ImageRecord.id == image_id, ImageRecord.manifest.is_not(None))
        )

    def input(self, image_id: UUID) -> tuple[ImageRecord, dict[str, bytes]]:
        record = self.session.get(ImageRecord, image_id)
        if record is None:
            raise FileNotFoundError
        artifacts = self.session.scalars(
            select(ImageArtifact).where(
                ImageArtifact.image_id == image_id,
                ImageArtifact.filename.in_(["original", "normalized.png"]),
            )
        )
        files = {a.filename: a.content for a in artifacts}
        if set(files) != {"original", "normalized.png"}:
            raise ValueError("Incomplete durable input")
        return record, files

    def create_input(
        self, image_id: UUID, width: int, height: int, artifacts: dict[str, bytes]
    ) -> None:
        self.session.add(ImageRecord(id=image_id, width=width, height=height))
        self.session.flush()
        self.session.add_all(
            ImageArtifact(
                image_id=image_id, filename=name, content_type=content_type(name), content=data
            )
            for name, data in artifacts.items()
        )

    def publish(self, result: ImageResult, artifacts: dict[str, bytes], manifest: dict) -> None:
        record = self.session.scalar(
            select(ImageRecord).where(ImageRecord.id == result.id).with_for_update()
        )
        if record is None:
            raise FileNotFoundError
        if record.manifest is not None:
            raise ValueError("Image already published")
        if (record.width, record.height) != (result.width, result.height):
            raise ValueError("Conflicting image dimensions")
        self.session.execute(delete(ImageArtifact).where(ImageArtifact.image_id == result.id))
        self.session.add_all(
            ImageArtifact(
                image_id=result.id, filename=name, content_type=content_type(name), content=data
            )
            for name, data in artifacts.items()
        )
        captions = {obj.object_id: obj.object_caption for obj in result.classification.objects}
        self.session.add_all(
            ImageObject(
                image_id=result.id,
                object_id=obj.object_id,
                label=obj.label,
                bbox=list(obj.bbox),
                object_caption=captions.get(obj.object_id),
                attributes={},
            )
            for obj in result.detection.objects
        )
        record.manifest = manifest

    def artifact(self, image_id: UUID, name: str) -> ImageArtifact:
        if self.get(image_id) is None:
            raise FileNotFoundError
        artifact = self.session.get(ImageArtifact, (image_id, name))
        if artifact is None:
            raise FileNotFoundError
        return artifact

    def bundle(self, image_id: UUID) -> dict[str, tuple[str, bytes]]:
        if self.get(image_id) is None:
            raise FileNotFoundError
        return {
            a.filename: (a.content_type, a.content)
            for a in self.session.scalars(
                select(ImageArtifact).where(ImageArtifact.image_id == image_id)
            )
        }
