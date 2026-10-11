from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator


class AcceptedImage(BaseModel):
    image_id: UUID
    job_id: UUID
    status: str
    publish_status: str
    status_url: str


class ImageInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    original_filename: str
    original_sha256: str
    original_mime: str
    original_size_bytes: int
    normalized_sha256: str
    width: int
    height: int
    normalization_version: str
    created_at: datetime


class ResultInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    stage: str
    status: str
    model_name: str | None
    model_revision: str | None
    weights_sha256: str | None
    parameters: dict
    schema_version: int
    payload: dict | None
    started_at: datetime | None
    finished_at: datetime | None
    duration_ms: float | None
    error: dict | None


class JobInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    status: str
    current_stage: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    error: dict | None
    publish_status: str
    published_at: datetime | None
    publish_attempts: int
    next_publish_at: datetime
    publish_error: dict | None
    processing_attempts: int
    image: ImageInfo
    results: list[ResultInfo]


# Payload contracts are also validated in the worker before committing JSONB.


class DetectedObject(BaseModel):
    object_id: UUID
    bbox: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
    label: str
    class_id: int
    label_space: Literal["coco"]
    score: FiniteFloat = Field(ge=0, le=1)


class DetectionPayload(BaseModel):
    coordinate_space: Literal["normalized_image"]
    bbox_format: Literal["xyxy"]
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    objects: list[DetectedObject]

    @model_validator(mode="after")
    def check_boxes(self):
        ids = [obj.object_id for obj in self.objects]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate object_id")
        for obj in self.objects:
            x1, y1, x2, y2 = obj.bbox
            if not (0 <= x1 < x2 <= self.image_width and 0 <= y1 < y2 <= self.image_height):
                raise ValueError("Box outside normalized image")
        return self


class Prediction(BaseModel):
    class_id: int = Field(ge=0, lt=1000)
    synset: str
    label: str
    score: FiniteFloat = Field(ge=0, le=1)


class ClassifiedObject(BaseModel):
    object_id: UUID
    crop_bbox: tuple[int, int, int, int]
    status: Literal["succeeded", "failed"]
    top_k: list[Prediction]
    error: dict | None = None

    @model_validator(mode="after")
    def check_result(self):
        x1, y1, x2, y2 = self.crop_bbox
        if not (0 <= x1 < x2 and 0 <= y1 < y2):
            raise ValueError("Invalid crop")
        if self.status == "succeeded" and (len(self.top_k) != 5 or self.error is not None):
            raise ValueError("Expected top five predictions")
        if self.status == "failed" and (self.top_k or not self.error):
            raise ValueError("Expected crop error")
        return self


class ClassificationPayload(BaseModel):
    objects: list[ClassifiedObject]


class AnnotationPayload(BaseModel):
    raw_caption: str = Field(min_length=1)
