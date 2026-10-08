"""Create a new experiment for the RF-DETR sparse-slot fix, retaining proven runs."""

from __future__ import annotations

import argparse
import ast
import hashlib
import os
import shutil
import tarfile
from pathlib import Path

from detection.common import ROOT, file_lock, object_hash, read_json, verify_model, write_json
from detection.runner import source_fingerprint


class RemoveSparseFix(ast.NodeTransformer):
    """Prove that removing this RF-DETR-only fix restores the original adapter."""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        self.generic_visit(node)
        pairs = [
            (arg, default)
            for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults)
            if arg.arg != "exclude_coco_gaps"
        ]
        node.args.kwonlyargs = [arg for arg, _ in pairs]
        node.args.kw_defaults = [default for _, default in pairs]
        return node

    def visit_If(self, node: ast.If) -> ast.If | None:
        if (
            isinstance(node.test, ast.BoolOp)
            and isinstance(node.test.values[0], ast.Name)
            and node.test.values[0].id == "exclude_coco_gaps"
        ):
            return None
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> ast.Call:
        self.generic_visit(node)
        node.keywords = [kw for kw in node.keywords if kw.arg != "exclude_coco_gaps"]
        return node

    def visit_Dict(self, node: ast.Dict) -> ast.Dict:
        self.generic_visit(node)
        pairs = [
            (key, value)
            for key, value in zip(node.keys, node.values)
            if not (isinstance(key, ast.Constant) and key.value == "excluded_coco_gap_predictions")
        ]
        node.keys = [key for key, _ in pairs]
        node.values = [value for _, value in pairs]
        return node


def validate_adapter_change(old: str, new: str) -> None:
    stripped = RemoveSparseFix().visit(ast.parse(new))
    if ast.dump(ast.parse(old)) != ast.dump(stripped):
        raise ValueError("Adapter changes exceed the isolated RF-DETR sparse-slot fix")


def recover(source: Path, output: Path) -> None:
    if output.exists():
        raise ValueError("Recovery requires a new output directory")
    original = read_json(source / "experiment.json")
    config = read_json(ROOT / "detection/configs/benchmark.json")
    dataset = read_json(ROOT / "data/coco/test.json")
    if original["config"] != config or original["dataset_hash"] != object_hash(dataset):
        raise ValueError("Config/dataset changed; cannot carry results forward")
    if original["model_ids"] != config["models"] or "rf_detr_small" not in config["models"]:
        raise ValueError("Unexpected model selection")
    hashes = {model: object_hash(verify_model(model)) for model in config["models"]}
    if original["model_hashes"] != hashes or original["image_id"] != os.environ.get(
        "CONTAINER_IMAGE_ID"
    ):
        raise ValueError("Weights/container changed; cannot carry results forward")
    baseline = read_json(source / "yolo26s_repeat1/environment.json")["source_files"]
    if object_hash(baseline) != original["source_hash"]:
        raise ValueError("Original source fingerprint differs")
    current = source_fingerprint()
    changed = sorted(
        name for name in set(baseline) | set(current) if baseline.get(name) != current.get(name)
    )
    allowed = {
        "detection/adapters.py",
        "detection/metrics.py",
        "detection/report.py",
        "scripts/detection.sh",
        "scripts/recover_detection.py",
    }
    if set(changed) - allowed:
        raise ValueError(f"Unreviewed source changes: {set(changed) - allowed}")
    with tarfile.open(source / "source_snapshot.tar.gz") as snapshot:

        def old_text(name: str) -> str:
            stream = snapshot.extractfile(name)
            if stream is None:
                raise ValueError(f"Missing archived source: {name}")
            content = stream.read()
            if hashlib.sha256(content).hexdigest() != baseline[name]:
                raise ValueError(f"Archived source hash differs: {name}")
            return content.decode()

        validate_adapter_change(
            old_text("detection/adapters.py"), (ROOT / "detection/adapters.py").read_text()
        )

        # Quality evaluation must remain identical; only bootstrap may change.
        def evaluator_nodes(code: str) -> list[str]:
            return [
                ast.dump(node)
                for node in ast.parse(code).body
                if isinstance(node, ast.FunctionDef)
                and node.name
                in {"empty_detections", "evaluate_coco", "load_predictions", "evaluate_run"}
            ]

        if evaluator_nodes(old_text("detection/metrics.py")) != evaluator_nodes(
            (ROOT / "detection/metrics.py").read_text()
        ):
            raise ValueError(
                "Quality evaluator changed; existing metrics cannot be carried forward"
            )
    keep, failures = [], []
    for model in config["models"]:
        for repeat in range(config["repeats"]):
            run = source / f"{model}_repeat{repeat + 1}"
            meta = read_json(run / "run.json")
            if model == "rf_detr_small":
                failures.append(
                    {
                        "run": str(run.relative_to(ROOT)),
                        "status": meta["status"],
                        "processed": meta.get("processed"),
                        "error": meta.get("error"),
                    }
                )
                continue
            environment = read_json(run / "environment.json")
            metrics = read_json(run / "metrics.json")
            if (
                meta["status"] != "complete"
                or meta["shortened"]
                or meta["config"] != config
                or meta["repeat"] != repeat
                or meta["model_id"] != model
            ):
                raise ValueError(f"Run is not complete and compatible: {run}")
            if (
                object_hash(read_json(run / "dataset_manifest.json")) != original["dataset_hash"]
                or object_hash(read_json(run / "model_manifest.json")) != hashes[model]
            ):
                raise ValueError(f"Run dataset/weights differ: {run}")
            if (
                object_hash(environment["source_files"]) != original["source_hash"]
                or environment["container_image_id"] != original["image_id"]
            ):
                raise ValueError(f"Run source/container differ: {run}")
            if metrics["n_images"] != len(dataset["images"]):
                raise ValueError(f"Incomplete image count: {run}")
            keep.append(run)
    with file_lock(ROOT / "storage/locks/inference.lock"):
        output.mkdir(parents=True)
        write_json(output / "experiment.json", {**original, "source_hash": object_hash(current)})
        with tarfile.open(output / "source_snapshot.tar.gz", "w:gz") as snapshot:
            for name in current:
                snapshot.add(ROOT / name, arcname=name)
        for run in keep:
            destination = output / run.name
            shutil.copytree(run, destination)
            meta = read_json(destination / "run.json")
            meta["carried_forward"] = {
                "from_run": str(run.relative_to(ROOT)),
                "source_hash": original["source_hash"],
                "reason": "RF-DETR-only adapter fix; other inference/evaluation unchanged",
            }
            write_json(destination / "run.json", meta)
        write_json(
            output / "recovery.json",
            {
                "schema_version": "1.0",
                "from_experiment": str(source.relative_to(ROOT)),
                "changed_source_files": changed,
                "carried_forward_runs": [run.name for run in keep],
                "original_failed_runs": failures,
                "rerun_model": "rf_detr_small",
                "original_identity": original,
            },
        )
    print(
        f"Carried forward {len(keep)} completed runs; rerun RF-DETR from image 1: {output}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Восстановление после исправления sparse-ID RF-DETR"
    )
    parser.add_argument("--from-runs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    recover(args.from_runs.resolve(), args.output_dir.resolve())
