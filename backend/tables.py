from uuid import UUID

from sqlalchemy import CheckConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

STAGES = ("detection", "classification", "annotation")


class Base(DeclarativeBase):
    pass


class ProcessedImage(Base):
    """Only completed images, with file paths and the original model responses."""

    __tablename__ = "processed_images"
    __table_args__ = (CheckConstraint("width > 0 AND height > 0", name="processed_dimensions"),)

    id: Mapped[UUID] = mapped_column(primary_key=True)
    original_path: Mapped[str]
    normalized_path: Mapped[str]
    result_path: Mapped[str]
    width: Mapped[int]
    height: Mapped[int]
    models: Mapped[dict] = mapped_column(JSONB)
    detection: Mapped[dict] = mapped_column(JSONB)
    classification: Mapped[dict] = mapped_column(JSONB)
    annotation: Mapped[dict] = mapped_column(JSONB)
