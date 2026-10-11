"""resnet50: official torchvision IMAGENET1K_V2, local FP32 ImageNet weights."""

from __future__ import annotations

import time
from typing import Any

try:
    from .common import MODELS, labels, main, validate_native
except ImportError:  # Direct command: python scripts/classification/resnet50.py
    from common import MODELS, labels, main, validate_native


class Model:
    def __init__(self, device: str) -> None:
        import torch
        from torchvision.models import ResNet50_Weights, resnet50

        self.torch = torch
        self.device = torch.device(device)
        self.weights = ResNet50_Weights.IMAGENET1K_V2
        self.native = validate_native("resnet50", self.weights)
        self.classes = labels()
        self.transform = self.weights.transforms()
        # weights=None prevents implicit downloads. Only the verified official state dict is read.
        self.model = resnet50(weights=None)
        state = torch.load(
            MODELS / "resnet50" / "resnet50-11ad3fa6.pth",
            map_location="cpu",
            weights_only=True,
        )
        self.model.load_state_dict(state, strict=True)
        del state
        self.model.eval().to(device=self.device, dtype=torch.float32)
        self.parameters = sum(parameter.numel() for parameter in self.model.parameters())

    def sync(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)

    def predict(self, image: Any) -> tuple[list[dict], dict[str, float]]:
        with self.torch.inference_mode():
            self.sync()
            start = time.perf_counter()
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            self.sync()
            prepared = time.perf_counter()
            logits = self.model(tensor)
            self.sync()
            inferred = time.perf_counter()
            if logits.shape != (1, 1000) or not self.torch.isfinite(logits).all():
                raise ValueError("Invalid ImageNet classifier output")
            scores, indices = logits.softmax(-1).topk(5, dim=-1)
            predictions = [
                {**self.classes[index], "score": score}
                for index, score in zip(
                    indices[0].cpu().tolist(), scores[0].cpu().tolist(), strict=True
                )
            ]
            self.sync()
            finished = time.perf_counter()
        return predictions, {
            "preprocess_ms": (prepared - start) * 1000,
            "inference_ms": (inferred - prepared) * 1000,
            "postprocess_ms": (finished - inferred) * 1000,
            "total_ms": (finished - start) * 1000,
        }

    def validate_native(self, image: Any, predictions: list[dict]) -> dict:
        # Independent official one-expression inference path, using the same loaded network.
        with self.torch.inference_mode():
            logits = self.model(self.weights.transforms()(image).unsqueeze(0).to(self.device))
            scores, indices = logits.softmax(-1).topk(5)
        if [p["index"] for p in predictions] != indices[0].cpu().tolist():
            raise ValueError("Top-5 native parity failed")
        delta = max(
            abs(p["score"] - score)
            for p, score in zip(predictions, scores[0].cpu().tolist(), strict=True)
        )
        if delta > 1e-6:
            raise ValueError("Softmax native parity failed")
        return {
            "status": "passed",
            "max_score_difference": delta,
            "image_preprocessing": "official weights.transforms(); stored pixels, RGB; no EXIF transpose",
        }


if __name__ == "__main__":
    main(Model, "resnet50")
