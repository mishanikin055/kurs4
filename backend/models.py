"""Three offline adapters; each resident process owns one initialized model."""

import hashlib
import json
import math
import sys
import time
from pathlib import Path
from uuid import uuid4

from PIL import Image


def verify_files(stage: str, request: dict) -> Path:
    spec = json.loads(Path(__file__).with_name("checkpoints.json").read_text())[stage]
    if any(request[key] != spec[key] for key in ("model_name", "model_revision", "weights_sha256")):
        raise ValueError("Unsupported checkpoint")
    directory = Path(request["model_root"]) / stage / "models" / spec["model_name"]
    for item in spec["files"]:
        digest = hashlib.sha256()
        with (directory / item["path"]).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != item["sha256"]:
            raise ValueError("Checkpoint checksum mismatch")
    return directory


def load_detector(directory: Path, device: str):
    from rfdetr import RFDETRSmall

    wrapper = RFDETRSmall(pretrain_weights=str(directory / "rf-detr-small.pth"), device=device)
    wrapper.model.model.to(device).eval()
    return wrapper


def detect(image: Image.Image, directory: Path, parameters: dict, *, loaded=None) -> dict:
    import torch
    from torchvision.transforms import functional as F

    wrapper = loaded if loaded is not None else load_detector(directory, parameters["device"])
    module = wrapper.model.model
    tensor = F.to_tensor(image).to(parameters["device"])
    tensor = F.resize(tensor, [512, 512], antialias=False)
    tensor = F.normalize(tensor, wrapper.means, wrapper.stds).unsqueeze(0)
    with torch.inference_mode():
        raw = module(tensor)
        result = wrapper.model.postprocess(
            raw,
            target_sizes=torch.tensor([[image.height, image.width]], device=tensor.device),
            score_threshold=parameters["score_threshold"],
        )[0]
    categories = {
        x["id"]: x["name"]
        for x in json.loads((directory / "label_mapping.json").read_text())["categories"]
    }
    objects = []
    values = zip(
        result["boxes"].tolist(), result["scores"].tolist(), result["labels"].tolist(), strict=True
    )
    for box, score, label in sorted(values, key=lambda x: -x[1]):
        if not all(math.isfinite(x) for x in [*box, score]):
            raise ValueError("Nonfinite detector output")
        if score < parameters["score_threshold"]:
            continue
        if label not in categories:
            continue
        x1, y1, x2, y2 = box
        box = [
            max(0, min(image.width, x1)),
            max(0, min(image.height, y1)),
            max(0, min(image.width, x2)),
            max(0, min(image.height, y2)),
        ]
        if box[0] >= box[2] or box[1] >= box[3]:
            continue
        objects.append(
            dict(
                object_id=str(uuid4()),
                bbox=box,
                label=categories[label],
            )
        )
        if len(objects) >= parameters["max_detections"]:
            break
    return {"objects": objects}


def load_classifier(directory: Path, device: str):
    import torch
    from torchvision.models import ViT_B_16_Weights, vit_b_16

    model = vit_b_16(weights=None)
    state = torch.load(directory / "vit_b_16-c867db91.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.eval().to(device)
    transform = ViT_B_16_Weights.IMAGENET1K_V1.transforms()
    labels = json.loads((Path(__file__).with_name("imagenet_labels.json")).read_text())["classes"]
    return model, transform, labels


def classify(
    image: Image.Image, directory: Path, parameters: dict, detection: dict, *, loaded=None
) -> dict:
    padding = parameters.get("crop_padding_fraction", 0.1)
    if not math.isfinite(padding) or not 0 <= padding <= 1:
        raise ValueError("Invalid crop padding")
    import torch

    model, transform, labels = (
        loaded if loaded is not None else load_classifier(directory, parameters["device"])
    )
    objects = []
    for obj in detection["objects"]:
        x1, y1, x2, y2 = obj["bbox"]
        dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
        box = [
            max(0, math.floor(x1 - dx)),
            max(0, math.floor(y1 - dy)),
            min(image.width, math.ceil(x2 + dx)),
            min(image.height, math.ceil(y2 + dy)),
        ]
        record = {"object_id": obj["object_id"]}
        try:
            with torch.inference_mode():
                logits = model(transform(image.crop(box)).unsqueeze(0).to(parameters["device"]))
                if logits.shape != (1, 1000) or not torch.isfinite(logits).all():
                    raise ValueError("Invalid classifier output")
                index = logits.argmax(-1).item()
            record["label"] = labels[index]["name"]
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as exc:
            record["error"] = {"code": "crop_error", "type": type(exc).__name__}
        objects.append(record)
    return {"objects": objects}


def load_annotator(directory: Path, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        str(directory), local_files_only=True, trust_remote_code=True, use_fast=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(directory),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map={"": device},
        attn_implementation="eager",
    ).eval()
    model.language_model._supports_default_dynamic_cache = lambda: False
    return processor, model


def annotate(image: Image.Image, directory: Path, parameters: dict, *, loaded=None) -> dict:
    import torch

    processor, model = (
        loaded if loaded is not None else load_annotator(directory, parameters["device"])
    )
    inputs = processor(text=parameters["prompt"], images=image, return_tensors="pt")
    inputs = {
        key: value.to(parameters["device"], dtype=torch.float16)
        if value.is_floating_point()
        else value.to(parameters["device"])
        for key, value in inputs.items()
    }
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=parameters["max_new_tokens"],
            do_sample=False,
            num_beams=1,
            use_cache=True,
        )
    return {
        "raw_caption": processor.decode(
            output[0][1:], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
    }


def predict(request: dict, directory: Path, *, loaded=None, image=None) -> dict:
    try:
        if image is None:
            with Image.open(request["image_path"]) as source:
                image = source.convert("RGB")
        stage = request["stage"]
        if stage == "detection":
            payload = detect(image, directory, request["parameters"], loaded=loaded)
        elif stage == "classification":
            payload = classify(
                image, directory, request["parameters"], request["detection"], loaded=loaded
            )
        elif stage == "annotation":
            payload = annotate(image, directory, request["parameters"], loaded=loaded)
        else:
            raise ValueError("Unknown stage")
        return {"payload": payload}
    except Exception as exc:
        return {
            "error": {
                "code": "cuda_oom" if type(exc).__name__ == "OutOfMemoryError" else "model_error",
                "type": type(exc).__name__,
            }
        }


def serve(fd: int) -> None:
    import socket

    import torch

    from backend.runtime import receive, send

    torch.set_num_threads(4)
    with socket.socket(fileno=fd) as channel:
        spec = receive(channel)
        directory = verify_files(spec["stage"], spec)
        started = time.monotonic()
        device = spec["parameters"]["device"]
        if spec["stage"] == "detection":
            loaded = load_detector(directory, device)
        elif spec["stage"] == "classification":
            loaded = load_classifier(directory, device)
        else:
            loaded = load_annotator(directory, device)
        # Pay CUDA/kernel initialization once, before consuming user jobs.
        warmup = dict(
            spec, detection={"objects": [{"object_id": str(uuid4()), "bbox": [0, 0, 224, 224]}]}
        )
        response = predict(warmup, directory, loaded=loaded, image=Image.new("RGB", (224, 224)))
        if "error" in response:
            send(channel, response)
            return
        free = None
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
            free = torch.cuda.mem_get_info(device)[0]
        send(channel, {"ready": True, "startup_s": time.monotonic() - started, "vram_free": free})
        while True:
            request = receive(channel)
            if any(request[key] != spec[key] for key in spec):
                send(channel, {"error": {"code": "model_not_configured"}})
                continue
            started = time.monotonic()
            response = predict(request, directory, loaded=loaded)
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            print(f"stage={spec['stage']} inference_s={time.monotonic() - started:.3f}", flush=True)
            send(channel, response)


def main() -> None:
    if sys.argv[1] == "--serve":
        serve(int(sys.argv[2]))
        return
    request_path, output_path = map(Path, sys.argv[1:])
    request = json.loads(request_path.read_text())
    try:
        directory = verify_files(request["stage"], request)
        result = predict(request, directory)
    except Exception as exc:
        result = {"error": {"code": "model_error", "type": type(exc).__name__}}
    output_path.write_text(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
