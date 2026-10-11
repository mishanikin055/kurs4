"""Synthetic child for infrastructure tests; never loaded by production entrypoints."""

import json
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

from PIL import Image

request_path, output_path = map(Path, sys.argv[1:])
request = json.loads(request_path.read_text())
stage = request["stage"]
mode = os.environ.get("MOCK_MODEL_MODE", "success")
if stage == "detection" and mode == "slow":
    Path(os.environ["MOCK_STARTED"]).write_text(str(os.getpid()))
    time.sleep(float(os.environ.get("MOCK_SLEEP", "30")))
if mode == "fail_" + stage:
    result = {"error": {"code": "synthetic_failure"}}
elif stage == "detection":
    with Image.open(request["image_path"]) as image:
        width, height = image.size
    objects = (
        []
        if mode == "empty"
        else [
            dict(
                object_id=str(uuid4()),
                bbox=[0.1, 0.1, 5.2, 5.2],
                label="person",
                class_id=1,
                label_space="coco",
                score=0.9,
            )
        ]
    )
    result = {
        "payload": dict(
            coordinate_space="normalized_image",
            bbox_format="xyxy",
            image_width=width,
            image_height=height,
            objects=objects,
        )
    }
elif stage == "classification":
    result = {
        "payload": {
            "objects": [
                dict(
                    object_id=obj["object_id"],
                    crop_bbox=[0, 0, 6, 6],
                    status="succeeded",
                    top_k=[
                        dict(class_id=i, synset=f"n{i:08}", label="synthetic", score=0.1)
                        for i in range(5)
                    ],
                )
                for obj in request["detection"]["objects"]
            ]
        }
    }
else:
    result = {"payload": {"raw_caption": "  Synthetic caption.  "}}
output_path.write_text(json.dumps(result))
