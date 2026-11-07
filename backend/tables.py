from datetime import datetime
from uuid import UUID

from sqlalchemy import JSON, CheckConstraint, DateTime, ForeignKey, Index, LargeBinary, func
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
    models: Mapped[dict] = mapped_column(JSON().with_variant(JSONB, "postgresql"))
    detection: Mapped[dict] = mapped_column(JSON().with_variant(JSONB, "postgresql"))
    classification: Mapped[dict] = mapped_column(JSON().with_variant(JSONB, "postgresql"))
    annotation: Mapped[dict] = mapped_column(JSON().with_variant(JSONB, "postgresql"))


# Binary values live separately so catalogue queries never fetch photograph bytes.

json_value = JSON(none_as_null=True).with_variant(JSONB(none_as_null=True), "postgresql")


class ImageRecord(Base):
    __tablename__ = "image_records"
    __table_args__ = (CheckConstraint("width > 0 AND height > 0", name="image_dimensions"),)

    id: Mapped[UUID] = mapped_column(primary_key=True)
    width: Mapped[int]
    height: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    manifest: Mapped[dict | None] = mapped_column(json_value, nullable=True)


class ImageArtifact(Base):
    __tablename__ = "image_artifacts"

    image_id: Mapped[UUID] = mapped_column(
        ForeignKey("image_records.id", ondelete="CASCADE"), primary_key=True
    )
    filename: Mapped[str] = mapped_column(primary_key=True)
    content_type: Mapped[str]
    content: Mapped[bytes] = mapped_column(LargeBinary)


class ImageObject(Base):
    __tablename__ = "image_objects"
    __table_args__ = (
        Index("image_objects_label_image_idx", "label", "image_id"),
        Index("image_objects_attributes_idx", "attributes", postgresql_using="gin"),
    )

    image_id: Mapped[UUID] = mapped_column(
        ForeignKey("image_records.id", ondelete="CASCADE"), primary_key=True
    )
    object_id: Mapped[UUID] = mapped_column(primary_key=True)
    label: Mapped[str]
    bbox: Mapped[list] = mapped_column(json_value)
    object_caption: Mapped[str | None]
    attributes: Mapped[dict] = mapped_column(json_value, default=dict)
