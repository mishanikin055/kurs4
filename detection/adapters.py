"""Model adapters with native preprocessing and explicit COCO label mapping."""

from __future__ import annotations

from collections import OrderedDict
from time import perf_counter
from typing import Any

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as F

from detection.common import CATEGORIES, MODELS, model_spec, read_json

CATEGORY_LIST = read_json(CATEGORIES)
COCO_IDS = [c["id"] for c in CATEGORY_LIST]
COCO_NAMES = {c["id"]: c["name"] for c in CATEGORY_LIST}


def to_coco_label(label: int, mapping: str) -> int:
    if mapping == "contiguous_0":
        if not 0 <= label < len(COCO_IDS):
            raise ValueError(f"Invalid continuous class index: {label}")
        return COCO_IDS[label]
    if mapping == "coco_sparse" and label in COCO_NAMES:
        return label
    raise ValueError(f"Unsupported COCO category: {label}, mapping={mapping}")


def normalize_predictions(
    boxes: Any,
    scores: Any,
    labels: Any,
    mapping: str,
    width: int,
    height: int,
    floor: float,
    max_det: int,
    *,
    exclude_coco_gaps: bool = False,
) -> list[dict[str, Any]]:
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    if len(boxes) != len(scores) or len(scores) != len(labels):
        raise ValueError("Prediction fields have different lengths")
    if (
        not np.isfinite(boxes).all()
        or not np.isfinite(scores).all()
        or ((scores < 0) | (scores > 1)).any()
    ):
        raise ValueError("Invalid prediction values")
    rows = []
    for index in np.argsort(-scores, kind="stable"):
        if scores[index] < floor:
            continue
        label = int(labels[index])
        # RF-DETR's pretrained head retains unused COCO slots (including 0).
        # These slots have no dataset category and must never be remapped.
        if exclude_coco_gaps and mapping == "coco_sparse" and 0 <= label <= 90:
            if label not in COCO_NAMES:
                continue
        category = to_coco_label(label, mapping)
        x1, y1, x2, y2 = boxes[index]
        x1, x2 = np.clip([x1, x2], 0, width)
        y1, y2 = np.clip([y1, y2], 0, height)
        if x2 <= x1 or y2 <= y1:
            continue
        rows.append(
            {
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(scores[index]),
                "category_id": category,
                "original_label": label,
            }
        )
        if len(rows) == max_det:
            break
    return rows


class Adapter:
    def __init__(self, model_id: str, config: dict[str, Any]) -> None:
        self.spec = model_spec(model_id)
        self.config = config
        self.device = torch.device(config["device"])
        self.directory = MODELS / model_id
        backend = self.spec["backend"]
        if backend == "ultralytics":
            from ultralytics import YOLO
            from ultralytics.models.yolo.detect import DetectionPredictor

            self.wrapper = YOLO(str(self.directory / "yolo26s.pt"))
            # Match the package predict path, keeping stages separately measurable.
            self.predictor = DetectionPredictor(
                overrides={
                    "conf": config["score_floor"],
                    "iou": 0.7,
                    "imgsz": 640,
                    "rect": True,
                    "max_det": config["max_detections"],
                    "device": str(self.device),
                    "half": False,
                    "verbose": False,
                    "save": False,
                }
            )
            self.predictor.setup_model(model=self.wrapper.model, verbose=False)
            self.predictor.setup_source([np.zeros((640, 640, 3), dtype=np.uint8)])
            self.predictor.batch = (["local.jpg"], None, [""])
            self.module = self.predictor.model
            names = self.wrapper.names
            if [names[i] for i in range(80)] != [c["name"] for c in CATEGORY_LIST]:
                raise ValueError("YOLO class mapping differs from COCO")
        elif backend == "transformers":
            from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

            self.processor = RTDetrImageProcessor.from_pretrained(
                self.directory, local_files_only=True
            )
            self.module = RTDetrV2ForObjectDetection.from_pretrained(
                self.directory, local_files_only=True
            ).to(self.device)
            if [
                {
                    "motorbike": "motorcycle",
                    "aeroplane": "airplane",
                    "sofa": "couch",
                    "pottedplant": "potted plant",
                    "diningtable": "dining table",
                    "tvmonitor": "tv",
                }.get(self.module.config.id2label[i], self.module.config.id2label[i])
                for i in range(80)
            ] != [c["name"] for c in CATEGORY_LIST]:
                raise ValueError("RT-DETR class mapping differs from COCO")
        elif backend == "rfdetr":
            from rfdetr import RFDETRSmall

            self.wrapper = RFDETRSmall(
                pretrain_weights=str(self.directory / "rf-detr-small.pth"), device=str(self.device)
            )
            self.module = self.wrapper.model.model.to(self.device)
            if self.wrapper.model.resolution != 512 or self.wrapper.model.args.num_classes != 90:
                raise ValueError("Unexpected RF-DETR checkpoint configuration")
        elif backend == "torchvision":
            from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

            self.module = fasterrcnn_resnet50_fpn_v2(
                weights=None,
                weights_backbone=None,
                box_score_thresh=config["score_floor"],
                box_detections_per_img=config["max_detections"],
            )
            state = torch.load(
                self.directory / "fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth",
                map_location="cpu",
                weights_only=True,
            )
            self.module.load_state_dict(state)
            self.module.to(self.device)
        else:
            raise ValueError(backend)
        self.module.eval()
        self.parameters = sum(p.numel() for p in self.module.parameters())

    def native_config(self) -> dict[str, Any]:
        backend = self.spec["backend"]
        if backend == "transformers":
            return {"model": self.module.config.to_dict(), "processor": self.processor.to_dict()}
        if backend == "rfdetr":
            return self.wrapper.model_config.model_dump(mode="json")
        if backend == "ultralytics":
            return {
                "architecture": self.wrapper.model.yaml,
                "prediction_args": vars(self.predictor.args),
                "labels": self.wrapper.names,
            }
        return {
            "architecture": "fasterrcnn_resnet50_fpn_v2",
            "weights": "COCO_V1",
            "min_size": list(self.module.transform.min_size),
            "max_size": self.module.transform.max_size,
            "image_mean": self.module.transform.image_mean,
            "image_std": self.module.transform.image_std,
            "box_score_thresh": self.module.roi_heads.score_thresh,
            "box_nms_thresh": self.module.roi_heads.nms_thresh,
            "detections_per_img": self.module.roi_heads.detections_per_img,
        }

    def sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def preprocess(self, image: Image.Image) -> Any:
        backend = self.spec["backend"]
        if backend == "ultralytics":
            self.original_bgr = np.asarray(image)[:, :, ::-1].copy()
            return self.predictor.preprocess([self.original_bgr])
        if backend == "transformers":
            return self.processor(images=image, return_tensors="pt").to(self.device)
        if backend == "rfdetr":
            tensor = F.to_tensor(image).to(self.device)
            tensor = F.resize(tensor, [512, 512], antialias=False)
            return F.normalize(tensor, self.wrapper.means, self.wrapper.stds).unsqueeze(0)
        tensor = F.to_tensor(image).to(self.device)
        return self.module.transform([tensor], None)[0]

    def forward(self, value: Any) -> Any:
        backend = self.spec["backend"]
        if backend == "ultralytics":
            return self.predictor.inference(value)
        if backend == "transformers":
            return self.module(**value)
        if backend == "rfdetr":
            return self.module(value)
        features = self.module.backbone(value.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, _ = self.module.rpn(value, features, None)
        detections, _ = self.module.roi_heads(features, proposals, value.image_sizes, None)
        return detections

    def postprocess(self, raw: Any, prepared: Any, image: Image.Image) -> tuple[Any, Any, Any]:
        backend = self.spec["backend"]
        if backend == "ultralytics":
            self.predictor.batch = (["local.jpg"], [self.original_bgr], [""])
            result = self.predictor.postprocess(raw, prepared, [self.original_bgr])[0]
            return (
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
            )
        if backend == "transformers":
            result = self.processor.post_process_object_detection(
                raw,
                threshold=self.config["score_floor"],
                target_sizes=torch.tensor([[image.height, image.width]], device=self.device),
            )[0]
        elif backend == "rfdetr":
            result = self.wrapper.model.postprocess(
                raw,
                target_sizes=torch.tensor([[image.height, image.width]], device=self.device),
                score_threshold=self.config["score_floor"],
            )[0]
        else:
            result = self.module.transform.postprocess(
                raw, prepared.image_sizes, [(image.height, image.width)]
            )[0]
        return (
            result["boxes"].cpu().numpy(),
            result["scores"].cpu().numpy(),
            result["labels"].cpu().numpy(),
        )

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> tuple[list[dict[str, Any]], dict[str, float]]:
        self.sync()
        start = perf_counter()
        prepared = self.preprocess(image)
        self.sync()
        pre = perf_counter()
        raw = self.forward(prepared)
        self.sync()
        forward = perf_counter()
        boxes, scores, labels = self.postprocess(raw, prepared, image)
        rows = normalize_predictions(
            boxes,
            scores,
            labels,
            self.spec["mapping"],
            image.width,
            image.height,
            self.config["score_floor"],
            self.config["max_detections"],
            exclude_coco_gaps=self.spec["backend"] == "rfdetr",
        )
        self.sync()
        end = perf_counter()
        return rows, {
            "preprocess_ms": (pre - start) * 1000,
            "forward_ms": (forward - pre) * 1000,
            "postprocess_ms": (end - forward) * 1000,
            "total_ms": (end - start) * 1000,
            "excluded_coco_gap_predictions": int(
                np.sum((scores >= self.config["score_floor"]) & ~np.isin(labels, COCO_IDS))
            )
            if self.spec["backend"] == "rfdetr"
            else 0,
        }

    @torch.inference_mode()
    def validate_native(self, image: Image.Image, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Verify staged inference against the package's normal path on a non-square image."""
        backend = self.spec["backend"]
        if backend == "ultralytics":
            self.wrapper.predictor = self.predictor
            result = self.wrapper.predict(
                image,
                conf=self.config["score_floor"],
                iou=0.7,
                imgsz=640,
                rect=True,
                max_det=self.config["max_detections"],
                half=False,
                verbose=False,
                save=False,
            )[0]
            boxes, scores, labels = (
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
                result.boxes.cls.cpu().numpy(),
            )
        elif backend == "rfdetr":
            result = self.wrapper.predict(
                image, threshold=self.config["score_floor"], include_source_image=False
            )
            boxes, scores, labels = result.xyxy, result.confidence, result.class_id
        elif backend == "torchvision":
            result = self.module([F.to_tensor(image).to(self.device)])[0]
            boxes, scores, labels = (
                result["boxes"].cpu().numpy(),
                result["scores"].cpu().numpy(),
                result["labels"].cpu().numpy(),
            )
        else:
            raw = self.module(**self.processor(images=image, return_tensors="pt").to(self.device))
            result = self.processor.post_process_object_detection(
                raw,
                threshold=self.config["score_floor"],
                target_sizes=torch.tensor([[image.height, image.width]], device=self.device),
            )[0]
            boxes, scores, labels = (
                result["boxes"].cpu().numpy(),
                result["scores"].cpu().numpy(),
                result["labels"].cpu().numpy(),
            )
        native = normalize_predictions(
            boxes,
            scores,
            labels,
            self.spec["mapping"],
            image.width,
            image.height,
            self.config["score_floor"],
            self.config["max_detections"],
            exclude_coco_gaps=self.spec["backend"] == "rfdetr",
        )
        if len(native) != len(rows):
            raise ValueError(
                f"Native/staged prediction counts differ: {len(native)} != {len(rows)}"
            )
        if [r["category_id"] for r in rows] != [r["category_id"] for r in native]:
            raise ValueError("Native/staged category ordering differs")
        np.testing.assert_allclose(
            [r["bbox"] for r in rows], [r["bbox"] for r in native], rtol=1e-4, atol=0.03
        )
        np.testing.assert_allclose(
            [r["score"] for r in rows], [r["score"] for r in native], rtol=1e-4, atol=1e-5
        )
        return {
            "native_parity": "passed",
            "n_predictions": len(rows),
            "image_size": list(image.size),
            "bbox_atol_pixels": 0.03,
            "score_atol": 1e-5,
        }
