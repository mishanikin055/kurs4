"""Download only the application's pinned weights; standard library, no inference."""

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def sources():
    checkpoints = json.loads((ROOT / "backend/checkpoints.json").read_text(encoding="utf-8"))
    for stage, checkpoint in checkpoints.items():
        configs = json.loads((ROOT / stage / "configs/models.json").read_text(encoding="utf-8"))
        config = next(row for row in configs["models"] if row["id"] == checkpoint["model_name"])
        files = {row["path"]: row for row in config["files"]}
        for item in checkpoint["files"]:
            spec = files[item["path"]]
            target = ROOT / stage / "models" / checkpoint["model_name"] / item["path"]
            if stage == "annotation":
                url = (
                    f"https://huggingface.co/{config['repo_id']}/resolve/"
                    f"{config['revision']}/{item['path']}"
                )
            else:
                url = spec.get("url")
            yield target, item["sha256"], url, spec

    translation = json.loads((ROOT / "backend/translation.json").read_text(encoding="utf-8"))
    for item in translation["files"]:
        target = ROOT / "annotation/models" / translation["model_name"] / item["path"]
        url = f"https://huggingface.co/{translation['repo_id']}/resolve/{translation['revision']}/{item['path']}"
        yield target, item["sha256"], item.get("url", url), item


def fetch(target, expected, url, spec):
    if target.exists():
        if not target.is_file() or digest(target) != expected:
            raise ValueError(f"Existing file differs; preserve it and resolve manually: {target}")
        print(f"Verified: {target.relative_to(ROOT)}")
        return
    new_directory = not target.parent.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    if new_directory:
        target.parent.chmod(0o755)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".fetch-", delete=False) as out:
            temporary = Path(out.name)
            if spec.get("generated") == "coco_categories":
                categories = json.loads(
                    (ROOT / "detection/configs/coco_categories.json").read_text(encoding="utf-8")
                )
                out.write(
                    (
                        json.dumps(
                            {"mapping": "coco_sparse", "categories": categories},
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n"
                    ).encode("utf-8")
                )
            else:
                print(f"Downloading: {target.relative_to(ROOT)}", flush=True)
                request = urllib.request.Request(
                    url, headers={"User-Agent": "image-analysis/0.2.0"}
                )
                with urllib.request.urlopen(request, timeout=90) as response:
                    total = 0
                    for block in iter(lambda: response.read(1024 * 1024), b""):
                        total += len(block)
                        if total > spec["size"]:
                            raise ValueError(f"Download exceeds pinned size: {target.name}")
                        out.write(block)
        if digest(temporary) != expected:
            raise ValueError(f"SHA256 mismatch: {target.name}")
        # Public weights must also be readable by the container's UID 1000.
        temporary.chmod(0o644)
        os.replace(temporary, target)
        print(f"Verified: {target.relative_to(ROOT)}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    try:
        for arguments in sources():
            fetch(*arguments)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Preparation stopped: {exc}\n")


if __name__ == "__main__":
    main()
