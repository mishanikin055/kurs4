"""RT-DETRv2 R18: local Transformers checkpoint and its own image processor."""

from __future__ import annotations

from typing import Any

if __package__:
    from . import common
else:
    import common


class Model:
    mapping = "contiguous_0"
    exclude_coco_gaps = False

    def __init__(self, config: dict[str, Any]) -> None:
        import torch
        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection

        self.config = config
        self.device = torch.device(config["device"])
        directory = common.MODELS / "rtdetr_v2_r18vd"
        self.processor = RTDetrImageProcessor.from_pretrained(directory, local_files_only=True)
        self.module = (
            RTDetrV2ForObjectDetection.from_pretrained(directory, local_files_only=True)
            .to(self.device)
            .eval()
        )
        aliases = {
            "motorbike": "motorcycle",
            "aeroplane": "airplane",
            "sofa": "couch",
            "pottedplant": "potted plant",
            "diningtable": "dining table",
            "tvmonitor": "tv",
        }
        labels = [
            aliases.get(self.module.config.id2label[i], self.module.config.id2label[i])
            for i in range(80)
        ]
        if labels != [c["name"] for c in common.CATEGORY_LIST]:
            raise ValueError("RT-DETR class mapping differs from COCO")

    def preprocess(self, image: Any) -> Any:
        return self.processor(images=image, return_tensors="pt").to(self.device)

    def forward(self, prepared: Any) -> Any:
        return self.module(**prepared)

    def postprocess(self, raw: Any, prepared: Any, image: Any) -> tuple:
        import torch

        result = self.processor.post_process_object_detection(
            raw,
            threshold=self.config["score_floor"],
            target_sizes=torch.tensor([[image.height, image.width]], device=self.device),
        )[0]
        return (
            result["boxes"].cpu().numpy(),
            result["scores"].cpu().numpy(),
            result["labels"].cpu().numpy(),
        )

    def native_predict(self, image: Any) -> tuple:
        prepared = self.processor(images=image, return_tensors="pt").to(self.device)
        raw = self.module(**prepared)
        return self.postprocess(raw, prepared, image)

    def native_config(self) -> dict:
        return {"model": self.module.config.to_dict(), "processor": self.processor.to_dict()}


if __name__ == "__main__":
    common.main(Model, "rtdetr_v2_r18vd")
