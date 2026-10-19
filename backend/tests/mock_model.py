"""Synthetic child for infrastructure tests; never loaded by production entrypoints."""

import json
import os
import socket
import sys
import time
from pathlib import Path
from uuid import uuid4


def predict(request):
    stage = request["stage"]
    mode = os.environ.get("MOCK_MODEL_MODE", "success")
    if stage == "detection" and mode == "slow":
        Path(os.environ["MOCK_STARTED"]).write_text(str(os.getpid()))
        time.sleep(float(os.environ.get("MOCK_SLEEP", "30")))
    if mode == "fail_" + stage:
        return {"error": {"code": "synthetic_failure"}}
    if stage == "detection":
        objects = (
            []
            if mode == "empty"
            else [dict(object_id=str(uuid4()), bbox=[0.1, 0.1, 5.2, 5.2], label="person")]
        )
        return {"payload": {"objects": objects}, "pid": os.getpid()}
    if stage == "classification":
        return {
            "payload": {
                "objects": [
                    dict(object_id=obj["object_id"], label="synthetic")
                    for obj in request["detection"]["objects"]
                ]
            },
            "pid": os.getpid(),
        }
    return {"payload": {"raw_caption": "  Synthetic caption.  "}, "pid": os.getpid()}


def main():
    if sys.argv[1] == "--serve":
        from backend.runtime import receive, send

        with socket.socket(fileno=int(sys.argv[2])) as channel:
            spec = receive(channel)
            if directory := os.environ.get("MOCK_RESIDENT_DIR"):
                Path(directory, spec["stage"] + ".pid").write_text(str(os.getpid()))
            if os.environ.get("MOCK_MODEL_MODE") == "startup_error":
                send(channel, {"error": {"code": "synthetic_startup_failure"}})
                return
            send(channel, {"ready": True})
            try:
                while True:
                    send(channel, predict(receive(channel)))
            except EOFError:
                return
    else:
        request_path, output_path = map(Path, sys.argv[1:])
        output_path.write_text(json.dumps(predict(json.loads(request_path.read_text()))))


if __name__ == "__main__":
    main()
