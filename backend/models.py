"""Three concrete offline model adapters, invoked only in a fresh subprocess."""

import hashlib
import json
import math
import sys
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


def detect(image: Image.Image, directory: Path, parameters: dict) -> dict:
    import torch
    from rfdetr import RFDETRSmall
    from torchvision.transforms import functional as F

    wrapper = RFDETRSmall(
        pretrain_weights=str(directory / "rf-detr-small.pth"), device=parameters["device"]
    )
    module = wrapper.model.model.to(parameters["device"]).eval()
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
                class_id=label,
                label_space="coco",
                score=score,
            )
        )
        if len(objects) >= parameters["max_detections"]:
            break
    return dict(
        coordinate_space="normalized_image",
        bbox_format="xyxy",
        image_width=image.width,
        image_height=image.height,
        objects=objects,
    )


def classify(image: Image.Image, directory: Path, parameters: dict, detection: dict) -> dict:
    import torch
    from torchvision.models import ViT_B_16_Weights, vit_b_16

    model = vit_b_16(weights=None)
    state = torch.load(directory / "vit_b_16-c867db91.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    del state
    model.eval().to(parameters["device"])
    transform = ViT_B_16_Weights.IMAGENET1K_V1.transforms()
    labels = json.loads((Path(__file__).with_name("imagenet_labels.json")).read_text())["classes"]
    objects = []
    for obj in detection["objects"]:
        x1, y1, x2, y2 = obj["bbox"]
        box = [math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)]
        record = dict(
            object_id=obj["object_id"], crop_bbox=box, status="succeeded", top_k=[], error=None
        )
        try:
            with torch.inference_mode():
                logits = model(transform(image.crop(box)).unsqueeze(0).to(parameters["device"]))
                if logits.shape != (1, 1000) or not torch.isfinite(logits).all():
                    raise ValueError("Invalid classifier output")
                scores, indices = logits.softmax(-1).topk(parameters["top_k"])
            record["top_k"] = [
                dict(class_id=i, synset=labels[i]["synset"], label=labels[i]["name"], score=score)
                for i, score in zip(indices[0].tolist(), scores[0].tolist(), strict=True)
            ]
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as exc:
            record.update(status="failed", error={"code": "crop_error", "type": type(exc).__name__})
        objects.append(record)
    return {"objects": objects}


def annotate(image: Image.Image, directory: Path, parameters: dict) -> dict:
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
        device_map={"": parameters["device"]},
        attn_implementation="eager",
    ).eval()
    model.language_model._supports_default_dynamic_cache = lambda: False
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


def main() -> None:
    request_path, output_path = map(Path, sys.argv[1:])
    request = json.loads(request_path.read_text())
    try:
        directory = verify_files(request["stage"], request)
        with Image.open(request["image_path"]) as source:
            image = source.convert("RGB")
        stage = request["stage"]
        if stage == "detection":
            payload = detect(image, directory, request["parameters"])
        elif stage == "classification":
            payload = classify(image, directory, request["parameters"], request["detection"])
        elif stage == "annotation":
            payload = annotate(image, directory, request["parameters"])
        else:
            raise ValueError("Unknown stage")
        result = {"payload": payload}
    except Exception as exc:
        result = {
            "error": {
                "code": "cuda_oom" if type(exc).__name__ == "OutOfMemoryError" else "model_error",
                "type": type(exc).__name__,
            }
        }
    output_path.write_text(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
