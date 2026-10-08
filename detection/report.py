"""Build four-row comparison tables from saved artifacts, without inference."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from detection.common import read_json, registry


def build_report(runs_dir: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[Path]] = {}
    for p in sorted(runs_dir.glob("*/run.json")):
        groups.setdefault(read_json(p)["model_id"], []).append(p.parent)
    intervals = (
        read_json(runs_dir / "bootstrap.json") if (runs_dir / "bootstrap.json").exists() else {}
    )
    rows = []
    for spec in registry()["models"]:
        runs = groups.get(spec["id"], [])
        good = [r for r in runs if read_json(r / "run.json")["status"] == "complete"]
        expected_repeats = (
            read_json(runs_dir / "experiment.json")["config"]["repeats"]
            if (runs_dir / "experiment.json").exists()
            else 3
        )
        row: dict[str, Any] = {
            "model": spec["name"],
            "status": "not_run",
            "completed_repeats": len(good),
            "n_images": None,
            "mAP": None,
            "AP50": None,
            "AP75": None,
            "AP_small": None,
            "AP_medium": None,
            "AP_large": None,
            "AR100": None,
            "precision": None,
            "recall": None,
            "F1": None,
            "mAP_ci95_low": None,
            "mAP_ci95_high": None,
            "p50_ms": None,
            "p95_ms": None,
            "images_per_second": None,
            "cold_load_ms_mean": None,
            "VRAM_allocated_MiB": None,
            "VRAM_reserved_MiB": None,
            "RAM_peak_MiB": None,
            "parameters": None,
            "weight_MiB": None,
            "device": None,
            "precision_mode": None,
            "dataset_split": None,
            "model_revision": spec["revision"],
            "run_ids": ";".join(r.name for r in runs),
            "weight_sha256": None,
            "git_commit": None,
            "git_dirty": None,
            "container_image_id": None,
            "source_hash": None,
            "failures": None,
            "reused_from": None,
            "excluded_coco_gap_predictions": None,
        }
        if runs:
            failed = [
                read_json(r / "run.json")
                for r in runs
                if read_json(r / "run.json")["status"] not in {"complete", "smoke_passed"}
            ]
            row["failures"] = ";".join(f"{m['status']}: {m.get('error', '')}" for m in failed)
            row["status"] = "incomplete" if failed or len(good) < expected_repeats else "complete"
        if good:
            metrics = [read_json(r / "metrics.json") for r in good]
            meta = read_json(good[0] / "run.json")
            from detection.metrics import load_predictions

            latency = [t["total_ms"] for r in good for t in load_predictions(r)[2]]
            # Quality comes from repeat 1, not an average of AP from dataset chunks.
            canonical = next((r for r in good if read_json(r / "run.json")["repeat"] == 0), None)
            if canonical:
                quality = read_json(canonical / "metrics.json")["quality"]
                for key in (
                    "mAP",
                    "AP50",
                    "AP75",
                    "AP_small",
                    "AP_medium",
                    "AP_large",
                    "AR100",
                    "precision",
                    "recall",
                    "F1",
                ):
                    row[key] = quality[key]
            ci = intervals.get("mAP_ci95", {}).get(spec["id"])
            if ci:
                row["mAP_ci95_low"], row["mAP_ci95_high"] = ci
            sizes = {m["n_images"] for m in metrics}
            if len(sizes) != 1:
                raise ValueError("Repeated runs use different image counts")
            env = read_json(good[0] / "environment.json")
            model_manifest = read_json(good[0] / "model_manifest.json")
            dataset = read_json(good[0] / "dataset_manifest.json")
            row.update(
                n_images=metrics[0]["n_images"],
                p50_ms=float(np.percentile(latency, 50)),
                p95_ms=float(np.percentile(latency, 95)),
                images_per_second=float(1000 / np.mean(latency)),
                cold_load_ms_mean=float(np.mean([m["cold_load_ms"] for m in metrics])),
                RAM_peak_MiB=max(m["resources"].get("rss_peak_bytes", 0) for m in metrics)
                / 1024**2,
                parameters=metrics[0]["parameters"],
                weight_MiB=sum(
                    f["size"]
                    for f in model_manifest["files"]
                    if f["path"].endswith((".pt", ".pth", ".safetensors"))
                )
                / 1024**2,
                device=meta["config"]["device"],
                precision_mode=meta["config"]["precision"],
                dataset_split=dataset["split"],
                weight_sha256=";".join(
                    f["sha256"]
                    for f in model_manifest["files"]
                    if f["path"].endswith((".pt", ".pth", ".safetensors"))
                ),
                git_commit=env["git_commit"],
                git_dirty=env["dirty"],
                container_image_id=env["container_image_id"],
            )
            from detection.common import object_hash

            row["source_hash"] = object_hash(env["source_files"])
            row["reused_from"] = (
                ";".join(
                    read_json(r / "run.json").get("carried_forward", {}).get("from_run", "")
                    for r in good
                ).strip(";")
                or None
            )
            row["excluded_coco_gap_predictions"] = sum(
                int(t.get("excluded_coco_gap_predictions", 0))
                for r in good
                for t in load_predictions(r)[2]
            )
            if row["device"].startswith("cuda"):
                row["VRAM_allocated_MiB"] = (
                    max(m["resources"].get("cuda_allocated_peak_bytes", 0) for m in metrics)
                    / 1024**2
                )
                row["VRAM_reserved_MiB"] = (
                    max(m["resources"].get("cuda_reserved_peak_bytes", 0) for m in metrics)
                    / 1024**2
                )
            if any(m["shortened"] for m in metrics):
                row["status"] = (
                    "shortened" if row["status"] == "complete" else "incomplete_shortened"
                )
        rows.append(row)
    with (output / "detection.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    columns = [
        "model",
        "status",
        "n_images",
        "mAP",
        "AP50",
        "AP75",
        "p50_ms",
        "p95_ms",
        "VRAM_allocated_MiB",
        "RAM_peak_MiB",
        "completed_repeats",
    ]

    def display(value: Any) -> str:
        return (
            "—"
            if value is None
            else f"{value:.4f}"
            if isinstance(value, float)
            else str(value).replace("|", "\\|")
        )

    markdown = "# Сравнение детекторов\n\nМетрики AP представлены в диапазоне 0–1. Задержка: препроцессинг + forward + постпроцессинг; чтение и запись измерены отдельно. Throughput — обратное среднее этой задержки, batch size 1. Полные сведения о воспроизводимости — в CSV и каталогах запусков.\n\n"
    markdown += "| " + " | ".join(columns) + " |\n| " + " | ".join("---" for _ in columns) + " |\n"
    markdown += "\n".join("| " + " | ".join(display(r[c]) for c in columns) + " |" for r in rows)
    markdown += "\n\n`shortened` — сокращённая проверка, не основание для итоговых выводов. `incomplete` — есть пропуски повторов или ошибки. Незавершённые модели сохраняются в таблице. Качество взято из первого повтора; задержки объединены по завершённым повторам.\n"
    if not intervals:
        markdown += "\nBootstrap-интервалы пока не рассчитаны. Запустите команду `bootstrap` для полного протокола.\n"
    else:
        markdown += f"\n95% интервалы mAP и парных различий рассчитаны по image_id: {intervals['samples']} bootstrap-повторов; сохранены в `bootstrap.json`. Сокращённая выборка: {intervals.get('shortened', False)}.\n"
        markdown += "\n| model | mAP CI95 low | mAP CI95 high |\n| --- | --- | --- |\n"
        markdown += (
            "\n".join(
                f"| {r['model']} | {display(r['mAP_ci95_low'])} | {display(r['mAP_ci95_high'])} |"
                for r in rows
            )
            + "\n"
        )
        markdown += "\n| paired mAP difference | CI95 low | CI95 high |\n| --- | --- | --- |\n"
        markdown += (
            "\n".join(
                f"| {name} | {display(ci[0])} | {display(ci[1])} |"
                for name, ci in intervals.get("paired_mAP_difference_ci95", {}).items()
            )
            + "\n"
        )
    if (runs_dir / "recovery.json").exists():
        recovery = read_json(runs_dir / "recovery.json")
        markdown += f"\nВосстановленная версия: {len(recovery['carried_forward_runs'])} неизменённых проходов перенесены из `{recovery['from_experiment']}` с исходными environment/source manifests. RF-DETR выполнен заново после исправления неиспользуемых COCO slots. Три исходных сбоя сохранены в исходном каталоге и перечислены в `recovery.json`; столбец CSV `reused_from` показывает происхождение.\n"
        markdown += "\nИсключённые RF-DETR предсказания с неиспользуемыми COCO ID считаются отдельно в сырых timings и CSV; их ID не преобразуются в реальные категории.\n"
    (output / "detection.md").write_text(markdown, encoding="utf-8")
    complete = [r for r in rows if r["status"] == "complete" and r["mAP"] is not None]
    if len(complete) == 4:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for axis, label, name in [
            ("p50_ms", "p50 задержки, мс", "quality_latency"),
            ("RAM_peak_MiB", "Пиковая RAM, MiB", "quality_ram"),
            ("VRAM_allocated_MiB", "Пиковая VRAM allocated, MiB", "quality_vram"),
        ]:
            usable = [r for r in complete if r[axis] is not None]
            if not usable:
                continue
            fig, ax = plt.subplots(figsize=(8, 5))
            for r in usable:
                ax.scatter(r[axis], r["mAP"])
                ax.annotate(r["model"], (r[axis], r["mAP"]), fontsize=8)
            ax.set(xlabel=label, ylabel="COCO mAP@[.50:.95]")
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(output / f"{name}.png", dpi=160)
            fig.savefig(output / f"{name}.svg")
            plt.close(fig)
    print("Tables:", output / "detection.csv", output / "detection.md", flush=True)
