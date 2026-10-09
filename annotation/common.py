"""Pinned annotation artifacts and independent inputs."""

from typing import Any

from detection.common import ROOT, object_hash, read_json, sha256

MODELS = ROOT / "annotation/models"
REGISTRY = ROOT / "annotation/configs/models.json"
CONFIG = ROOT / "annotation/configs/benchmark.json"


def specs() -> dict[str, dict[str, Any]]:
    return {s["id"]: s for s in read_json(REGISTRY)["models"]}


def verify_model(model_id: str) -> dict[str, Any]:
    spec = specs()[model_id]
    record = read_json(MODELS / "manifest.json")["models"][model_id]
    if record["spec_hash"] != object_hash(spec) or record["status"] != "verified":
        raise ValueError("Model is not verified against pinned registry")
    for f in record["files"]:
        path = MODELS / model_id / f["path"]
        if path.stat().st_size != f["size"] or sha256(path) != f["sha256"]:
            raise ValueError(f"Model checksum mismatch: {path}")
    return record


def validate_config(config: dict[str, Any]) -> None:
    if config["repeats"] != 1 or config["batch_size"] != 1:
        raise ValueError("Exactly one pass and batch size 1 are required")
    if config.get("bootstrap_samples", 0) or config.get("confidence_intervals", False):
        raise ValueError("New confidence intervals are prohibited")
    if config["precision"] != "fp16" or config["do_sample"] or config["num_beams"] != 1:
        raise ValueError("Frozen FP16 greedy decoding required")
    if config["models"] != list(specs()) or config["warmup"] != 1:
        raise ValueError("Participants and one warmup are frozen")
    if config["max_new_tokens"] != 96:
        raise ValueError("Frozen token budget required")
