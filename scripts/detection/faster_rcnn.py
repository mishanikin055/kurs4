"""Faster R-CNN ResNet50 FPN V2: native torchvision transform, RPN, ROI and NMS."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

if __package__:
    from . import common
else:
    import common


class Model:
    mapping = "coco_sparse"
    exclude_coco_gaps = False

    def __init__(self, config: dict[str, Any]) -> None:
        import torch
        from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

        self.config = config
        self.device = torch.device(config["device"])
        self.module = fasterrcnn_resnet50_fpn_v2(
            weights=None,
            weights_backbone=None,
            box_score_thresh=config["score_floor"],
            box_detections_per_img=config["max_detections"],
        )
        state = torch.load(
            common.MODELS
            / "fasterrcnn_resnet50_fpn_v2/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth",
            map_location="cpu",
            weights_only=True,
        )
        self.module.load_state_dict(state)
        self.module.to(self.device).eval()

    def preprocess(self, image: Any) -> Any:
        from torchvision.transforms import functional as F

        return self.module.transform([F.to_tensor(image).to(self.device)], None)[0]

    def forward(self, prepared: Any) -> Any:
        import torch

        features = self.module.backbone(prepared.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, _ = self.module.rpn(prepared, features, None)
        detections, _ = self.module.roi_heads(features, proposals, prepared.image_sizes, None)
        return detections

    def postprocess(self, raw: Any, prepared: Any, image: Any) -> tuple:
        result = self.module.transform.postprocess(
            raw, prepared.image_sizes, [(image.height, image.width)]
        )[0]
        return (
            result["boxes"].cpu().numpy(),
            result["scores"].cpu().numpy(),
            result["labels"].cpu().numpy(),
        )

    def native_predict(self, image: Any) -> tuple:
        from torchvision.transforms import functional as F

        result = self.module([F.to_tensor(image).to(self.device)])[0]
        return (
            result["boxes"].cpu().numpy(),
            result["scores"].cpu().numpy(),
            result["labels"].cpu().numpy(),
        )

    def native_config(self) -> dict:
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


if __name__ == "__main__":
    common.main(Model, "fasterrcnn_resnet50_fpn_v2")
