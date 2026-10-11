from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    UniqueConstraint,
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
    __table_args__ = (
        CheckConstraint("width > 0 AND height > 0", name="image_dimensions"),
        CheckConstraint("original_size_bytes > 0", name="image_size"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    original_filename: Mapped[str] = mapped_column(String(255))
    original_path: Mapped[str]
    original_sha256: Mapped[str] = mapped_column(String(64))
    original_mime: Mapped[str] = mapped_column(String(32))
    original_size_bytes: Mapped[int] = mapped_column(BigInteger)
    normalized_path: Mapped[str]
    normalized_sha256: Mapped[str] = mapped_column(String(64))
    width: Mapped[int]
    height: Mapped[int]
    normalization_version: Mapped[str] = mapped_column(String(32), default="exif-rgb-png-v1")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


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
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    publish_attempts: Mapped[int] = mapped_column(default=0)
    next_publish_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    publish_error: Mapped[dict | None] = mapped_column(JSONB)
    processing_attempts: Mapped[int] = mapped_column(default=0)
    image: Mapped[Image] = relationship()
    results: Mapped[list["ModelResult"]] = relationship(order_by="ModelResult.stage")


class ModelResult(Base):
    __tablename__ = "model_results"
    __table_args__ = (
        UniqueConstraint("job_id", "stage", name="uq_model_results_job_stage"),
        CheckConstraint(
            "stage IN ('detection', 'classification', 'annotation')", name="result_stage"
        ),
        CheckConstraint(
            "status IN ('pending', 'running', 'succeeded', 'partial', 'failed', 'skipped')",
            name="result_status",
        ),
        CheckConstraint("duration_ms IS NULL OR duration_ms >= 0", name="result_duration"),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    job_id: Mapped[UUID] = mapped_column(ForeignKey("jobs.id"))
    stage: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), default="pending")
    model_name: Mapped[str | None]
    model_revision: Mapped[str | None]
    weights_sha256: Mapped[str | None] = mapped_column(String(64))
    parameters: Mapped[dict] = mapped_column(JSONB, default=dict)
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    payload: Mapped[dict | None] = mapped_column(JSONB)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    duration_ms: Mapped[float | None]
    error: Mapped[dict | None] = mapped_column(JSONB)
