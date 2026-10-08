from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from classification.common import registry
from classification.crops.metrics import load
from classification.report import write_csv
from detection.common import object_hash, read_json, sha256, write_json


def report(runs: Path, output: Path, include_ci: bool = False) -> None:
    output.mkdir(parents=True, exist_ok=True)
    experiment = read_json(runs / "experiment.json")
    boot = (
        read_json(runs / "bootstrap.json")
        if include_ci and (runs / "bootstrap.json").exists()
        else None
    )
    if boot and boot["identity"]["experiment_hash"] != object_hash(experiment):
        raise ValueError("Bootstrap experiment differs")
    rows = []
    per_class = []
    provenance = {}
    for spec in registry()["models"]:
        model = spec["id"]
        paths = sorted(runs.glob(f"{model}_repeat*"))
        metas = [(p, read_json(p / "run.json")) for p in paths if (p / "run.json").exists()]
        complete = [(p, m) for p, m in metas if m["status"] == "complete"]
        final = len(complete) == experiment["config"]["repeats"] and not experiment["shortened"]
        row = {
            "model": spec["name"],
            "status": "complete"
            if final
            else "shortened"
            if len(complete) == experiment["config"]["repeats"]
            else "incomplete",
            "mode": None,
            "completed_repeats": len(complete),
            "n_scenes": None,
            "n_crops": None,
            "n_eligible_crops": None,
            "mapping_object_coverage": None,
            "top1_coarse": None,
            "top5_coarse": None,
            "macro_f1_observed": None,
            "top1_ci95_low": None,
            "top1_ci95_high": None,
            "acceptance_rate": None,
            "accepted_top1_accuracy": None,
            "detector_only_matched_accuracy": None,
            "hypothetical_gated_matched_accuracy": None,
            "helpful_changes": None,
            "harmful_changes": None,
            "missed_supported_gt": None,
            "unmatched_proposals": None,
            "matched_unsupported_proposals": None,
            "crowd_ignored_proposals": None,
            "detector_only_supported_recall": None,
            "hypothetical_gated_supported_recall": None,
            "p50_ms": None,
            "p95_ms": None,
            "preprocess_mean_ms": None,
            "inference_mean_ms": None,
            "postprocess_mean_ms": None,
            "read_mean_ms": None,
            "crop_mean_ms": None,
            "write_mean_ms": None,
            "cold_load_mean_ms": None,
            "RAM_peak_MiB": None,
            "VRAM_allocated_peak_MiB": None,
            "VRAM_reserved_peak_MiB": None,
            "parameters": None,
            "revision": spec["revision"],
            "failures": ";".join(
                f"{p.name}:{m['status']}:{m.get('error', '')}"
                for p, m in metas
                if m["status"] != "complete"
            ),
        }
        records = []
        writes = []
        quality = None
        for path, meta in complete:
            digest = sha256(path / "samples.jsonl")
            if digest != meta["samples_sha256"]:
                raise ValueError("Completed prediction hash changed")
            samples, dataset = load(path)
            if object_hash(dataset) != experiment["dataset_hash"]:
                raise ValueError("Dataset hash differs")
            records.extend(samples)
            writes.extend(
                json.loads(s)["write_ms"]
                for s in (path / "write_timings.jsonl").read_text().splitlines()
            )
            provenance[path.name] = {
                n: sha256(path / n)
                for n in [
                    "run.json",
                    "samples.jsonl",
                    "metrics.json",
                    "environment.json",
                    "dataset_manifest.json",
                    "model_manifest.json",
                    "config.json",
                ]
            }
            if meta["repeat"] == 0:
                quality = read_json(path / "metrics.json")
        if final and quality:
            for key in row:
                if key in quality:
                    row[key] = quality[key]
            for item in quality["per_class"]:
                per_class.append({"model": model, **item})
            times = np.array([r["timing"]["total_ms"] for r in records])
            row.update(
                p50_ms=float(np.percentile(times, 50)),
                p95_ms=float(np.percentile(times, 95)),
                write_mean_ms=float(np.mean(writes)),
                cold_load_mean_ms=float(np.mean([m["cold_load_ms"] for _, m in complete])),
                parameters=complete[0][1]["parameters"],
            )
            for stage in ["preprocess", "inference", "postprocess", "read", "crop"]:
                row[f"{stage}_mean_ms"] = float(
                    np.mean([r["timing"][f"{stage}_ms"] for r in records])
                )
            for label, resource in [
                ("RAM", "rss"),
                ("VRAM_allocated", "cuda_allocated"),
                ("VRAM_reserved", "cuda_reserved"),
            ]:
                row[f"{label}_peak_MiB"] = (
                    max(m["resources"].get(f"{resource}_peak_bytes", 0) for _, m in complete)
                    / 1024**2
                )
            if boot:
                if (
                    boot["identity"]["predictions_sha256"][model]
                    != complete[0][1]["samples_sha256"]
                ):
                    raise ValueError("Bootstrap predictions differ")
                row["top1_ci95_low"], row["top1_ci95_high"] = boot["intervals"][model][
                    "top1_coarse"
                ]
        if not include_ci:
            row = {key: value for key, value in row.items() if "_ci95_" not in key}
        rows.append(row)
    write_csv(output / "classification_crops.csv", rows)
    if per_class:
        write_csv(output / "per_class.csv", per_class)
    if boot:
        write_csv(output / "paired_differences.csv", boot["paired_differences"])
    write_json(
        output / "report_manifest.json",
        {
            "schema_version": "1.0",
            "experiment": experiment,
            "sources": provenance,
            "bootstrap_sha256": sha256(runs / "bootstrap.json") if boot else None,
        },
    )
    columns = [
        "model",
        "status",
        "n_crops",
        "n_eligible_crops",
        "top1_coarse",
        "top5_coarse",
        "p50_ms",
        "p95_ms",
        "RAM_peak_MiB",
        "VRAM_reserved_peak_MiB",
    ]
    lines = [
        "# Сравнение классификаторов на вырезках COCO",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                "—"
                if row[c] is None
                else f"{row[c]:.5f}"
                if isinstance(row[c], float)
                else str(row[c])
                for c in columns
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Качество — первый повтор; время и память — все три. Доли от 0 до 1.",
            "Top-1/Top-5 оценивают общие COCO-категории после mapping; это не точность пород и подтипов.",
            "Рамки и исходные категории детектора сохраняются. Gated-замена в CSV — гипотетический эксперимент, а не автоматическое изменение результата.",
            "Порог score/margin заранее зафиксирован; он не калиброван для надёжного отказа на неизвестных объектах.",
            "GT-crops и detector-crops имеют отдельные manifests, таблицы и интервалы. Bootstrap переносит целые сцены вместе со всеми их объектами.",
            "Задержка p50/p95 включает препроцессинг, модель и постпроцессинг. Чтение оригинала, вырезание и запись выделены отдельно.",
            "CSV содержит пропущенные объекты, ложные рамки, неподдерживаемые категории и исправленные/внесённые ошибки; подробные JSON сохраняются в каждом проходе.",
        ]
    )
    (output / "classification_crops.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
