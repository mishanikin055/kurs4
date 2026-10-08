from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from classification.common import model_spec, registry
from classification.metrics import load_samples
from detection.common import object_hash, read_json, sha256, write_json


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_report(runs: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    experiment = (
        read_json(runs / "experiment.json") if (runs / "experiment.json").exists() else None
    )
    bootstrap = read_json(runs / "bootstrap.json") if (runs / "bootstrap.json").exists() else None
    if bootstrap and bootstrap["identity"]["experiment_hash"] != object_hash(experiment):
        raise ValueError("Bootstrap belongs to another experiment")
    sources = {}
    rows = []
    per_class = []
    confusion = []
    for spec in registry()["models"]:
        model_id = spec["id"]
        all_runs = sorted(runs.glob(f"{model_id}_repeat*"))
        metadata = [
            (run, read_json(run / "run.json")) for run in all_runs if (run / "run.json").exists()
        ]
        complete = [(p, m) for p, m in metadata if m["status"] == "complete"]
        required = experiment["config"]["repeats"] if experiment else 3
        is_complete = len(complete) == required and all(not m["shortened"] for _, m in complete)
        failures = [
            f"{p.name}:{m['status']}:{m.get('error', '')}"
            for p, m in metadata
            if m["status"] not in {"complete", "smoke_passed"}
        ]
        row = {
            "model": spec["name"],
            "status": "complete"
            if is_complete
            else "shortened"
            if len(complete) == required
            else "incomplete"
            if metadata
            else "not_run",
            "completed_repeats": len(complete),
            "n_images": None,
            "top1": None,
            "top5": None,
            "macro_f1": None,
            "top1_ci95_low": None,
            "top1_ci95_high": None,
            "top5_ci95_low": None,
            "top5_ci95_high": None,
            "p50_ms": None,
            "p95_ms": None,
            "images_per_second": None,
            "preprocess_mean_ms": None,
            "inference_mean_ms": None,
            "postprocess_mean_ms": None,
            "read_mean_ms": None,
            "write_mean_ms": None,
            "cold_load_mean_ms": None,
            "RAM_peak_MiB": None,
            "VRAM_allocated_peak_MiB": None,
            "VRAM_reserved_peak_MiB": None,
            "parameters": None,
            "weight_MiB": None,
            "dataset_split": None,
            "device": None,
            "precision_mode": None,
            "revision": spec["revision"],
            "weight_sha256": None,
            "git_commit": None,
            "git_dirty": None,
            "container_image_id": None,
            "source_hash": experiment["source_hash"] if experiment else None,
            "failures": ";".join(failures),
        }
        records = []
        write_times = []
        first_quality = None
        for path, meta in complete:
            if sha256(path / "samples.jsonl") != meta["samples_sha256"]:
                raise ValueError("Completed predictions changed")
            samples = load_samples(path)
            records.extend(samples)
            for name in [
                "run.json",
                "metrics.json",
                "environment.json",
                "samples.jsonl",
                "model_manifest.json",
                "dataset_manifest.json",
                "native_config.json",
            ]:
                sources[str(path / name)] = sha256(path / name)
            if meta["repeat"] == 0:
                first_quality = read_json(path / "metrics.json")
                dataset = read_json(path / "dataset_manifest.json")
                env = read_json(path / "environment.json")
                manifest = read_json(path / "model_manifest.json")
                weight = next(f for f in manifest["files"] if f["path"] == spec["weight_file"])
                row.update(
                    n_images=first_quality["n_images"],
                    top1=first_quality["top1"],
                    top5=first_quality["top5"],
                    macro_f1=first_quality["macro_f1"],
                    parameters=meta["parameters"],
                    weight_MiB=weight["size"] / 1024**2,
                    dataset_split=dataset["split"],
                    device=meta["config"]["device"],
                    precision_mode=meta["config"]["precision"],
                    weight_sha256=weight["sha256"],
                    git_commit=env["git_commit"],
                    git_dirty=env["dirty"],
                    container_image_id=env["container_image_id"],
                )
                per_class.extend({"model": spec["name"], **r} for r in first_quality["per_class"])
                confusion.extend(
                    {"model": spec["name"], **r} for r in first_quality["confusion_sparse"]
                )
            with (path / "write_timings.jsonl").open() as handle:
                times = {
                    json.loads(line)["image_id"]: json.loads(line)["write_ms"] for line in handle
                }
            write_times.extend(times.values())
        if records:
            timing = [r["timing"] for r in records]
            total = np.array([r["total_ms"] for r in timing])
            row.update(
                p50_ms=float(np.percentile(total, 50)),
                p95_ms=float(np.percentile(total, 95)),
                images_per_second=1000 / float(total.mean()),
                cold_load_mean_ms=float(np.mean([m["cold_load_ms"] for _, m in complete])),
            )
            for key in ["preprocess", "inference", "postprocess", "read"]:
                row[key + "_mean_ms"] = float(np.mean([r[key + "_ms"] for r in timing]))
            row["write_mean_ms"] = float(np.mean(write_times)) if write_times else None
            for target, source in [
                ("RAM_peak_MiB", "rss_peak_bytes"),
                ("VRAM_allocated_peak_MiB", "cuda_allocated_peak_bytes"),
                ("VRAM_reserved_peak_MiB", "cuda_reserved_peak_bytes"),
            ]:
                values = [m["resources"][source] for _, m in complete if source in m["resources"]]
                row[target] = max(values) / 1024**2 if values else None
        if bootstrap and is_complete:
            if (
                sha256(runs / f"{model_id}_repeat1/samples.jsonl")
                != bootstrap["identity"]["predictions_sha256"][model_id]
            ):
                raise ValueError("Bootstrap predictions changed")
            for metric in ["top1", "top5"]:
                row[metric + "_ci95_low"], row[metric + "_ci95_high"] = bootstrap["ci95"][model_id][
                    metric
                ]
        rows.append(row)
    write_csv(output / "classification.csv", rows)
    write_csv(
        output / "classification_per_class.csv",
        per_class,
        ["model", "index", "support", "correct", "accuracy", "f1"],
    )
    write_csv(
        output / "classification_confusion.csv",
        confusion,
        ["model", "target", "predicted", "count"],
    )
    paired = []
    if bootstrap:
        sources[str(runs / "bootstrap.json")] = sha256(runs / "bootstrap.json")
        for pair, metrics in bootstrap["paired_difference_ci95"].items():
            a, b = pair.split(" - ")
            for metric, interval in metrics.items():
                by_name = {model_spec(m)["name"]: m for m in [a, b]}
                quality_by_id = {by_name[r["model"]]: r for r in rows if r["model"] in by_name}
                paired.append(
                    {
                        "pair": pair,
                        "metric": metric,
                        "difference": quality_by_id[a][metric] - quality_by_id[b][metric],
                        "ci95_low": interval[0],
                        "ci95_high": interval[1],
                        "includes_zero": interval[0] <= 0 <= interval[1],
                    }
                )
    write_csv(
        output / "classification_paired.csv",
        paired,
        ["pair", "metric", "difference", "ci95_low", "ci95_high", "includes_zero"],
    )

    def fmt(value: object) -> str:
        return "—" if value is None else f"{value:.5f}" if isinstance(value, float) else str(value)

    text = [
        "# Сравнение классификаторов",
        "",
        "Сравниваются готовые ImageNet-1K checkpoint без обучения. Качество — первый повтор; задержки — все завершённые повторы. Статус каждой строки показывает полноту эксперимента.",
        "",
        "| Модель | Статус | N | Top-1 | Top-5 | Macro-F1 | p50, мс | p95, мс | RAM, MiB | VRAM reserved, MiB |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        text.append(
            "| "
            + " | ".join(
                fmt(row[k])
                for k in [
                    "model",
                    "status",
                    "n_images",
                    "top1",
                    "top5",
                    "macro_f1",
                    "p50_ms",
                    "p95_ms",
                    "RAM_peak_MiB",
                    "VRAM_reserved_peak_MiB",
                ]
            )
            + " |"
        )
    text.extend(
        [
            "",
            "Top-1/Top-5 и Macro-F1 — доли 0–1. Throughput = 1000 / mean(total_ms) при batch 1; total включает preprocessing/transfer, forward и postprocessing с CUDA sync, исключает IO, загрузку и запись. RAM — peak RSS процесса, включая загрузку; VRAM — PyTorch allocated/reserved, не полная память GPU.",
            "",
            "Штатные входы: ResNet-50 224 (resize 232), EfficientNetV2-S 384 (resize 384), ConvNeXt-Tiny 224 (resize 236), ViT-B/16 224 (resize 256). Отличаются обучающие рецепты и preprocessing; выводы относятся к готовым checkpoint и данному оборудованию. Набор не является доказанным мировым топ-4 по популярности.",
        ]
    )
    if paired:
        text.extend(
            [
                "",
                f"Парный стратифицированный bootstrap: {bootstrap['samples']} выборок, percentile CI95. Класс сохраняет свою долю; для всех моделей используются одинаковые image_id. Шесть парных интервалов без поправки на множественные сравнения.",
                "",
                "| Разность | Метрика | Оценка | CI95 low | CI95 high | Включает ноль |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        text.extend(
            "| "
            + " | ".join(
                fmt(r[k])
                for k in ["pair", "metric", "difference", "ci95_low", "ci95_high", "includes_zero"]
            )
            + " |"
            for r in paired
        )
    else:
        text.extend(
            [
                "",
                "Доверительные интервалы пока не рассчитаны. До полного запуска все отсутствующие числовые результаты оставлены пустыми.",
            ]
        )
    if all(r["status"] == "complete" for r in rows):
        best = max(rows, key=lambda r: r["top1"])
        fast = min(rows, key=lambda r: r["p50_ms"])
        text.extend(
            [
                "",
                f"На этой выборке наибольшая Top-1 у {best['model']} ({best['top1']:.5f}); минимальная медианная задержка у {fast['model']} ({fast['p50_ms']:.5f} мс). Устойчивость различий оценивается по парным CI; разность с интервалом, включающим ноль, не подтверждает превосходство.",
            ]
        )
    else:
        text.extend(
            [
                "",
                "Оценка неполная либо не запущена: итоговый победитель не назначается. Smoke на COCO не является ImageNet benchmark. GT-crops, detector-crops и сравнение detector-only с уточнением требуют отдельных протоколов и mapping; их показатели здесь отсутствуют.",
            ]
        )
    (output / "classification.md").write_text("\n".join(text) + "\n")
    write_json(
        output / "classification_report_manifest.json",
        {
            "schema_version": "1.0",
            "script_sha256": sha256(Path(__file__)),
            "experiment": experiment,
            "sources": sources,
            "outputs": [
                "classification.csv",
                "classification.md",
                "classification_per_class.csv",
                "classification_confusion.csv",
                "classification_paired.csv",
            ],
        },
    )
