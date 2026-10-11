"""RF-DETR Small: local Roboflow checkpoint, 512 resize and sparse COCO slots."""

from __future__ import annotations

from typing import Any

if __package__:
    from . import common
else:
    import common


class Model:
    mapping = "coco_sparse"
    exclude_coco_gaps = True

    def __init__(self, config: dict[str, Any]) -> None:
        import torch
        from rfdetr import RFDETRSmall

        self.config = config
        self.device = torch.device(config["device"])
        self.wrapper = RFDETRSmall(
            pretrain_weights=str(common.MODELS / "rf_detr_small/rf-detr-small.pth"),
            device=str(self.device),
        )
        self.module = self.wrapper.model.model.to(self.device).eval()
        if self.wrapper.model.resolution != 512 or self.wrapper.model.args.num_classes != 90:
            raise ValueError("Unexpected RF-DETR checkpoint configuration")

    def preprocess(self, image: Any) -> Any:
        from torchvision.transforms import functional as F

        tensor = F.to_tensor(image).to(self.device)
        tensor = F.resize(tensor, [512, 512], antialias=False)
        return F.normalize(tensor, self.wrapper.means, self.wrapper.stds).unsqueeze(0)

    def forward(self, prepared: Any) -> Any:
        return self.module(prepared)

    def postprocess(self, raw: Any, prepared: Any, image: Any) -> tuple:
        import torch

        result = self.wrapper.model.postprocess(
            raw,
            target_sizes=torch.tensor([[image.height, image.width]], device=self.device),
            score_threshold=self.config["score_floor"],
        )[0]
        return (
            result["boxes"].cpu().numpy(),
            result["scores"].cpu().numpy(),
            result["labels"].cpu().numpy(),
        )

    def native_predict(self, image: Any) -> tuple:
        result = self.wrapper.predict(
            image, threshold=self.config["score_floor"], include_source_image=False
        )
        return result.xyxy, result.confidence, result.class_id

    def native_config(self) -> dict:
        return self.wrapper.model_config.model_dump(mode="json")


if __name__ == "__main__":
    common.main(Model, "rf_detr_small")
