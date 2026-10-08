from __future__ import annotations

import time
from typing import Any

import torch
from PIL import Image
from torchvision.models import get_model, get_model_weights

from classification.common import MODELS, labels, model_spec
from detection.common import object_hash, read_json


class Adapter:
    def __init__(self, model_id: str, config: dict[str, Any]) -> None:
        self.spec = model_spec(model_id)
        self.device = torch.device(config["device"])
        self.weights = get_model_weights(model_id)[self.spec["weights"]]
        self.classes = labels()
        if self.weights.meta["categories"] != [r["name"] for r in self.classes]:
            raise ValueError("Checkpoint class ordering differs from frozen mapping")
        native = read_json(MODELS / model_id / "native_config.json")
        if object_hash(self.native_config()) != object_hash(native):
            raise ValueError("Native preprocessing differs from frozen checkpoint metadata")
        self.model = get_model(model_id, weights=None)
        # weights=None prevents implicit downloads; trusted official state_dict only.
        state = torch.load(
            MODELS / model_id / self.spec["weight_file"], map_location="cpu", weights_only=True
        )
        self.model.load_state_dict(state, strict=True)
        del state
        self.model.eval().to(self.device)
        self.transform = self.weights.transforms()
        self.parameters = sum(p.numel() for p in self.model.parameters())

    def sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def native_config(self) -> dict[str, Any]:
        transform = self.weights.transforms()
        return {
            "schema_version": "1.0",
            "weights": self.spec["weights"],
            "crop_size": transform.crop_size,
            "resize_size": transform.resize_size,
            "mean": transform.mean,
            "std": transform.std,
            "interpolation": transform.interpolation.value,
            "antialias": transform.antialias,
            "class_mapping_hash": object_hash(self.classes),
        }

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> tuple[list[dict[str, Any]], dict[str, float]]:
        self.sync()
        start = time.perf_counter()
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        self.sync()
        prepared = time.perf_counter()
        logits = self.model(tensor)
        self.sync()
        inferred = time.perf_counter()
        if logits.shape != (1, 1000) or not torch.isfinite(logits).all():
            raise ValueError("Invalid classifier output")
        scores, indices = logits.softmax(-1).topk(5, dim=-1)
        scores, indices = scores[0].cpu().tolist(), indices[0].cpu().tolist()
        predictions = [
            {**self.classes[index], "score": score} for index, score in zip(indices, scores)
        ]
        self.sync()
        finished = time.perf_counter()
        return predictions, {
            "preprocess_ms": (prepared - start) * 1000,
            "inference_ms": (inferred - prepared) * 1000,
            "postprocess_ms": (finished - inferred) * 1000,
            "total_ms": (finished - start) * 1000,
        }

    @torch.inference_mode()
    def validate_native(self, image: Image.Image, predictions: list[dict[str, Any]]) -> dict:
        # Same loaded network, independent official one-expression inference path.
        logits = self.model(self.weights.transforms()(image).unsqueeze(0).to(self.device))
        values, indices = logits.softmax(-1).topk(5)
        actual = [p["index"] for p in predictions]
        if actual != indices[0].cpu().tolist():
            raise ValueError("Top-5 native parity failed")
        delta = max(abs(a["score"] - b) for a, b in zip(predictions, values[0].cpu().tolist()))
        if delta > 1e-6:
            raise ValueError("Softmax native parity failed")
        return {"schema_version": "1.0", "status": "passed", "max_score_difference": delta}
