from uuid import UUID

from pydantic import BaseModel, Field, FiniteFloat, model_validator


class AcceptedImage(BaseModel):
    image_id: UUID
    output_folder: str


class DetectedObject(BaseModel):
    object_id: UUID
    bbox: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]
    label: str = Field(min_length=1)


class DetectionPayload(BaseModel):
    objects: list[DetectedObject]

    @model_validator(mode="after")
    def check_boxes(self):
        ids = [obj.object_id for obj in self.objects]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate object_id")
        for obj in self.objects:
            x1, y1, x2, y2 = obj.bbox
            if not (0 <= x1 < x2 and 0 <= y1 < y2):
                raise ValueError("Invalid box")
        return self


class ClassifiedObject(BaseModel):
    object_id: UUID
    label: str | None = Field(default=None, min_length=1)
    error: dict | None = None

    @model_validator(mode="after")
    def check_result(self):
        if (self.label is None) == (self.error is None) or self.error == {}:
            raise ValueError("Expected either one label or a crop error")
        return self


class ClassificationPayload(BaseModel):
    objects: list[ClassifiedObject]


class AnnotationPayload(BaseModel):
    raw_caption: str = Field(min_length=1)


class ImageResult(BaseModel):
    id: UUID
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    models: dict[str, dict[str, str]]
    detection: DetectionPayload
    classification: ClassificationPayload
    annotation: AnnotationPayload

    @model_validator(mode="after")
    def check_results(self):
        detected = {obj.object_id for obj in self.detection.objects}
        classified = [obj.object_id for obj in self.classification.objects]
        if set(classified) != detected or len(classified) != len(detected):
            raise ValueError("Classification object IDs do not match detection")
        if any(obj.error for obj in self.classification.objects):
            raise ValueError("Incomplete classification")
        if any(
            obj.bbox[2] > self.width or obj.bbox[3] > self.height for obj in self.detection.objects
        ):
            raise ValueError("Box outside normalized image")
        if not self.annotation.raw_caption.strip():
            raise ValueError("Empty caption")
        if set(self.models) != {"detection", "classification", "annotation"}:
            raise ValueError("Expected three model checkpoints")
        return self
