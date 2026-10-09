"""Verify real single-pass caption artifacts without loading neural model weights."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import math
import platform
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from annotation.common import CONFIG, MODELS, specs, validate_config, verify_model
from annotation.report import read_samples
from detection.common import ROOT, object_hash, read_json, sha256, write_json


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def source_files(snapshot: Path) -> dict[str, str]:
    result = {}
    with tarfile.open(snapshot) as archive:
        for member in archive.getmembers():
            if not member.isfile():
                continue
            handle = archive.extractfile(member)
            require(handle is not None and member.name not in result, "Invalid source snapshot")
            result[member.name] = hashlib.sha256(handle.read()).hexdigest()
    return result


def verify_numeric_row(row: dict[str, Any], run: Path) -> None:
    """Recompute quality and aggregates from captions rather than trusting the CSV."""
    import numpy as np

    from annotation.metrics import quality

    meta, dataset, samples = read_samples(run)
    scores, _, _ = quality(samples, dataset["images"])
    expected = {"images": len(samples), **scores}
    for field in [
        "preprocess_ms",
        "generate_ms",
        "postprocess_ms",
        "total_ms",
        "total_with_read_ms",
        "tokens_per_second",
    ]:
        values = [s["timing"][field] for s in samples]
        expected[field + "_mean"] = float(np.mean(values))
        expected[field + "_p50"] = float(np.percentile(values, 50))
        expected[field + "_p95"] = float(np.percentile(values, 95))
    expected.update(
        output_tokens_mean=float(np.mean([s["output_tokens"] for s in samples])),
        words_mean=float(np.mean([len(s["raw_caption"].split()) for s in samples])),
        token_budget_reached=sum(s["output_tokens"] >= 96 for s in samples),
        cold_load_ms=meta["cold_load_ms"],
        **meta["resources"],
    )
    for key, value in expected.items():
        require(
            row.get(key) is None
            if value is None
            else isinstance(row.get(key), (int, float))
            and math.isclose(row[key], value, rel_tol=1e-12, abs_tol=1e-12),
            f"Report aggregate differs from raw captions: {key}",
        )


def verify(experiment: Path, report: Path | None) -> dict[str, Any]:
    from transformers import AutoTokenizer, BartTokenizerFast

    identity = read_json(experiment / "experiment.json")
    validate_config(identity["config"])
    require(identity["mode"] == "test", "Verification requires the single main test")
    require(identity["config"] == read_json(CONFIG), "Frozen protocol changed")
    sources = source_files(experiment / "source_snapshot.tar.gz")
    require(object_hash(sources) == identity["source_hash"], "Snapshot source differs")
    expected_ids = None
    inference_commit = None
    participants = {}
    for model_id in specs():
        run = experiment / model_id
        meta, dataset, samples = read_samples(run)
        require(meta["identity"] == identity, "Participant identity differs")
        require(len(samples) == 500 and meta["expected_images"] == 500, "Requires 500 captions")
        ids = [r["image_id"] for r in samples]
        require(expected_ids is None or ids == expected_ids, "Participants differ")
        expected_ids = ids
        dev_ids = {
            r["image_id"] for r in read_json(ROOT / "annotation/configs/dev100.json")["images"]
        }
        require(not dev_ids.intersection(ids), "Dev and test overlap")
        environment = read_json(run / "environment.json")
        require(
            inference_commit is None or inference_commit == environment["git_commit"],
            "Inference commits differ",
        )
        inference_commit = environment["git_commit"]
        for package, version in environment["packages"].items():
            require(importlib.metadata.version(package) == version, "Runtime packages differ")
        require(environment["source_files"] == sources, "Worker source differs from snapshot")
        require(
            environment["container_image_id"] == identity["container_image_id"], "Container differs"
        )
        require(
            object_hash(verify_model(model_id)) == identity["model_hashes"][model_id],
            "Weights differ",
        )
        require((run / "errors.jsonl").read_text() == "", "Participant has errors")
        for kind in ["instances", "captions"]:
            path = ROOT / f"data/coco/annotations/{kind}_val2017.json"
            require(sha256(path) == dataset[f"{kind}_sha256"], "GT annotations differ")
        tokenizer_class = BartTokenizerFast if model_id == "florence2_base_ft" else AutoTokenizer
        tokenizer = tokenizer_class.from_pretrained(
            MODELS / model_id,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=model_id == "florence2_base_ft",
        )
        for image, sample in zip(dataset["images"], samples):
            require(sha256(ROOT / image["path"]) == image["sha256"], "Image content changed")
            decoded = tokenizer.decode(
                sample["output_token_ids"],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            require(decoded == sample["raw_caption"], "raw_caption is not unmodified decoded text")
            require(0 < sample["output_tokens"] <= 96, "Token budget violated")
            t = sample["timing"]
            require(all(math.isfinite(v) and v >= 0 for v in t.values()), "Invalid timing")
            require(
                math.isclose(
                    t["total_ms"],
                    sum(t[f"{k}_ms"] for k in ["preprocess", "generate", "postprocess"]),
                    abs_tol=1e-6,
                ),
                "Timing decomposition mismatch",
            )
            require(
                math.isclose(
                    t["total_with_read_ms"], t["total_ms"] + sample["read_ms"], abs_tol=1e-6
                ),
                "Read timing mismatch",
            )
            require(
                math.isclose(
                    t["tokens_per_second"],
                    sample["output_tokens"] / (t["generate_ms"] / 1000),
                    rel_tol=1e-10,
                ),
                "Token throughput mismatch",
            )
            expected_prompt = (
                None
                if model_id == "blip_base"
                else identity["config"]["florence_task"]
                if model_id == "florence2_base_ft"
                else identity["config"]["prompt"]
            )
            require(sample["prompt"] == expected_prompt, "Prompt changed")
        participants[model_id] = {
            "status": "verified",
            "images": len(samples),
            "samples_sha256": sha256(run / "samples.jsonl"),
            "dataset_hash": object_hash(dataset),
            "environment_sha256": sha256(run / "environment.json"),
            "resources": meta["resources"],
            "errors": 0,
        }
    if report:
        saved = read_json(report / "report_manifest.json")
        require(saved["human_scores"] == "excluded_by_user", "Human evaluation scope differs")
        for key, path in {
            "script_sha256": ROOT / "annotation/report.py",
            "metrics_sha256": ROOT / "annotation/metrics.py",
            "evaluator_sources_sha256": ROOT / "annotation/evaluation/sources.json",
            "experiment_sha256": experiment / "experiment.json",
            "source_snapshot_sha256": experiment / "source_snapshot.tar.gz",
        }.items():
            require(saved[key] == sha256(path), "Report provenance changed")
        require(set(saved["inputs"]) == set(specs()), "Report participants differ")
        for model_id, hashes in saved["inputs"].items():
            require(
                set(hashes) == {p.name for p in (experiment / model_id).iterdir() if p.is_file()},
                "Report inputs missing",
            )
            for name, digest in hashes.items():
                require(sha256(experiment / model_id / name) == digest, "Report input changed")
        require(
            [r["model"] for r in saved["rows"]] == list(specs())
            and all(r["status"] == "complete" for r in saved["rows"]),
            "Report incomplete",
        )
        for row in saved["rows"]:
            verify_numeric_row(row, experiment / row["model"])
        with (report / "annotation.csv").open(newline="", encoding="utf-8") as handle:
            csv_rows = list(csv.DictReader(handle))
        expected_csv = [
            {k: "" if v is None else str(v) for k, v in row.items()} for row in saved["rows"]
        ]
        require(csv_rows == expected_csv, "CSV and numeric manifest differ")
    from pycocoevalcap.tokenizer import ptbtokenizer

    jar = Path(ptbtokenizer.__file__).with_name("stanford-corenlp-3.4.1.jar")
    java = subprocess.run(["java", "-version"], capture_output=True, text=True, check=True)
    return {
        "verifier_sha256": sha256(Path(__file__)),
        "numeric_report_recomputed": report is not None,
        "human_evaluation": "excluded_by_user",
        "report_manifest_sha256": sha256(report / "report_manifest.json") if report else None,
        "report_outputs": {
            p.name: sha256(p)
            for p in sorted(report.iterdir())
            if p.is_file()
            and (
                p.name in {"annotation.csv", "annotation.md", "report_manifest.json"}
                or p.name.endswith(("_per_image.csv", "_chair.json"))
            )
        }
        if report
        else {},
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        "evaluator": {
            "pycocoevalcap": importlib.metadata.version("pycocoevalcap"),
            "nltk": importlib.metadata.version("nltk"),
            "ptb_jar_sha256": sha256(jar),
            "java": java.stderr.strip(),
            "sources_manifest_sha256": sha256(ROOT / "annotation/evaluation/sources.json"),
        },
        "status": "verified",
        "experiment": str(experiment),
        "images_per_model": 500,
        "repeats": 1,
        "confidence_intervals": "not calculated",
        "participants": participants,
        "source_hash": identity["source_hash"],
        "source_snapshot_sha256": sha256(experiment / "source_snapshot.tar.gz"),
        "inference_git_commit": inference_commit,
        "container_image_id": identity["container_image_id"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.experiment, args.report)
    write_json(args.output, result)
    print(result["status"])


if __name__ == "__main__":
    main()
