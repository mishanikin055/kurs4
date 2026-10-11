"""YOLO26s: local Ultralytics weights and native letterbox/predictor stages."""

from __future__ import annotations

from typing import Any

import numpy as np

if __package__:
    from . import common
else:
    import common


class Model:
    mapping = "contiguous_0"
    exclude_coco_gaps = False

    def __init__(self, config: dict[str, Any]) -> None:
        import torch
        from ultralytics import YOLO
        from ultralytics.models.yolo.detect import DetectionPredictor

        self.config = config
        self.device = torch.device(config["device"])
        self.wrapper = YOLO(str(common.MODELS / "yolo26s/yolo26s.pt"))
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
        self.parameters = sum(p.numel() for p in self.module.model.parameters())
        if [self.wrapper.names[i] for i in range(80)] != [c["name"] for c in common.CATEGORY_LIST]:
            raise ValueError("YOLO class mapping differs from COCO")

    def preprocess(self, image: Any) -> Any:
        self.original_bgr = np.asarray(image)[:, :, ::-1].copy()
        return self.predictor.preprocess([self.original_bgr])

    def forward(self, prepared: Any) -> Any:
        return self.predictor.inference(prepared)

    def postprocess(self, raw: Any, prepared: Any, image: Any) -> tuple:
        self.predictor.batch = (["local.jpg"], [self.original_bgr], [""])
        result = self.predictor.postprocess(raw, prepared, [self.original_bgr])[0]
        return (
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
        )

    def native_predict(self, image: Any) -> tuple:
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
        return (
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
        )

    def native_config(self) -> dict:
        return {
            "architecture": self.wrapper.model.yaml,
            "prediction_args": vars(self.predictor.args),
            "labels": self.wrapper.names,
        }


if __name__ == "__main__":
    common.main(Model, "yolo26s")
