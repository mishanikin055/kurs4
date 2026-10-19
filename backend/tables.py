from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

STAGES = ("detection", "classification", "annotation")


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Image(Base):
    __tablename__ = "images"
    __table_args__ = (CheckConstraint("width > 0 AND height > 0", name="image_dimensions"),)

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    original_path: Mapped[str]
    normalized_path: Mapped[str]
    width: Mapped[int]
    height: Mapped[int]


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'partial', 'failed')",
            name="job_status",
        ),
        CheckConstraint("publish_status IN ('pending', 'sent')", name="publish_status"),
        CheckConstraint(
            "current_stage IS NULL OR current_stage IN "
            "('detection', 'classification', 'annotation')",
            name="current_stage",
        ),
        CheckConstraint(
            "publish_attempts >= 0 AND processing_attempts BETWEEN 0 AND 2",
            name="job_attempts",
        ),
        Index("ix_jobs_publication", "publish_status", "next_publish_at"),
        Index("ix_jobs_image_id", "image_id"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    image_id: Mapped[UUID] = mapped_column(ForeignKey("images.id"))
    status: Mapped[str] = mapped_column(String(16), default="queued")
    current_stage: Mapped[str | None] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[dict | None] = mapped_column(JSONB)
    publish_status: Mapped[str] = mapped_column(String(16), default="pending")
    publish_attempts: Mapped[int] = mapped_column(default=0)
    next_publish_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    processing_attempts: Mapped[int] = mapped_column(default=0)
    image: Mapped[Image] = relationship()
    results: Mapped[list["ModelResult"]] = relationship(order_by="ModelResult.stage")


class ModelResult(Base):
    __tablename__ = "model_results"
    __table_args__ = (
        CheckConstraint(
            "stage IN ('detection', 'classification', 'annotation')", name="result_stage"
        ),
        CheckConstraint(
            "status IN ('pending', 'running', 'succeeded', 'partial', 'failed', 'skipped')",
            name="result_status",
        ),
    )

    job_id: Mapped[UUID] = mapped_column(ForeignKey("jobs.id"), primary_key=True)
    stage: Mapped[str] = mapped_column(String(16), primary_key=True)
    status: Mapped[str] = mapped_column(String(16), default="pending")
    model_name: Mapped[str | None]
    model_revision: Mapped[str | None]
    payload: Mapped[dict | None] = mapped_column(JSONB)
    error: Mapped[dict | None] = mapped_column(JSONB)


class ProcessedImage(Base):
    """Only completed images; the earlier three tables are retained as legacy data."""

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
