"""Completed inference results; transaction boundaries belong to the worker."""

from pathlib import Path
from uuid import UUID

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from backend.schemas import ImageResult
from backend.tables import ProcessedImage


class ImageRepository:
    def __init__(self, session: Session):
        self.session = session

    def get(self, image_id: UUID) -> ProcessedImage | None:
        return self.session.get(ProcessedImage, image_id)

    def save(self, result: ImageResult, directory: Path) -> None:
        values = result.model_dump(mode="json", exclude_none=True)
        values["id"] = result.id
        values.update(
            original_path=str(directory / "original"),
            normalized_path=str(directory / "normalized.png"),
            result_path=str(directory / "detection.png"),
        )
        self.session.execute(
            insert(ProcessedImage).values(**values).on_conflict_do_nothing(index_elements=["id"])
        )
