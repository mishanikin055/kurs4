"""Export all saved detection metrics without repeating inference or bootstrap."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect_yolo_metadata(runs: Path, output: Path) -> None:
    # The frozen AutoBackend wrapper exposed zero parameters. Count the actual
    # underlying network in the same pinned runtime, offline, on CPU only.
    from detection.adapters import Adapter
    from detection.common import ROOT as runtime_root
    from detection.common import file_lock, verify_model

    run = runs / "yolo26s_repeat1"
    meta = read(run / "run.json")
    saved = read(run / "model_manifest.json")
    with file_lock(runtime_root / "storage/locks/inference.lock"):
        current = verify_model("yolo26s")
        if current["files"] != saved["files"]:
            raise ValueError("YOLO weights differ from the measured checkpoint")
        config = {**meta["config"], "device": "cpu"}
        adapter = Adapter("yolo26s", config)
        count = sum(parameter.numel() for parameter in adapter.module.model.parameters())
        if count <= 0:
            raise ValueError("Underlying YOLO network has no parameters")
        output.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "1.0",
            "model_id": "yolo26s",
            "parameters": count,
            "original_recorded_parameters": meta["parameters"],
            "method": "sum(p.numel() for p in adapter.module.model.parameters())",
            "mode": "post-hoc metadata, CPU, no inference, same pinned image and weights",
            "container_image_id": __import__("os").environ.get("CONTAINER_IMAGE_ID"),
            "model_manifest_sha256": digest(run / "model_manifest.json"),
            "script_sha256": digest(Path(__file__)),
        }
        (output / "detection_yolo_metadata.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"YOLO runtime parameters: {count}")


def build(runs: Path, output: Path) -> None:
    comparison = runs / "comparison/detection.csv"
    with comparison.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    registry = read(ROOT / "detection/configs/models.json")
    by_name = {spec["name"]: spec["id"] for spec in registry["models"]}
    if len(rows) != 4 or any(row["status"] != "complete" for row in rows):
        raise ValueError("Full report requires four complete detectors")
    output.mkdir(parents=True, exist_ok=True)
    sources = [comparison, runs / "experiment.json", ROOT / "detection/configs/models.json"]
    config = read(runs / "experiment.json")["config"]
    per_class = []
    metadata_path = output / "detection_yolo_metadata.json"
    metadata = read(metadata_path) if metadata_path.exists() else None
    if metadata:
        sources.append(metadata_path)
    for row in rows:
        model_id = by_name[row["model"]]
        folders = sorted(
            path.parent
            for path in runs.glob("*/run.json")
            if read(path)["model_id"] == model_id and read(path)["status"] == "complete"
        )
        canonical = next(path for path in folders if read(path / "run.json")["repeat"] == 0)
        metrics = read(canonical / "metrics.json")
        row.update(metrics["quality"])
        for key in (
            "protocol_version",
            "batch_size",
            "seed",
            "warmup",
            "score_floor",
            "max_detections",
            "cpu_threads",
            "bootstrap_samples",
        ):
            row[key] = config[key]
        all_metrics = [read(path / "metrics.json") for path in folders]
        for key in metrics["latency"]:
            if key.endswith("_mean_ms"):
                row[key] = sum(
                    item["latency"][key] * item["n_images"] for item in all_metrics
                ) / sum(item["n_images"] for item in all_metrics)
        row["offline_inference"] = "yes: network_mode=none"
        row["parameters_note"] = "recorded runtime network"
        if int(row["parameters"]) <= 0:
            row["parameters"] = None
            row["parameters_note"] = "missing: frozen wrapper recorded zero"
            if model_id == "yolo26s" and metadata:
                if (
                    metadata["model_manifest_sha256"] != digest(canonical / "model_manifest.json")
                    or metadata["container_image_id"] != row["container_image_id"]
                ):
                    raise ValueError("YOLO metadata provenance differs from the experiment")
                row["parameters"] = metadata["parameters"]
                row["parameters_note"] = metadata["mode"]
        for category_id, values in metrics["per_class"].items():
            tp, fp, fn = (values[key] for key in ("TP", "FP", "FN"))
            per_class.append(
                {
                    "model": row["model"],
                    "category_id": category_id,
                    "category_name": values["name"],
                    "AP": values["AP"],
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "precision": tp / (tp + fp) if tp + fp else 0.0,
                    "recall": tp / (tp + fn) if tp + fn else 0.0,
                    "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                }
            )
        for folder in folders:
            sources.extend([folder / "run.json", folder / "metrics.json"])
        sources.extend([canonical / "model_manifest.json", canonical / "native_config.json"])
    if len(per_class) != 320:
        raise ValueError("Expected 80 categories for each of four detectors")
    write_csv(output / "detection_full.csv", rows)
    write_csv(output / "detection_per_class.csv", per_class)

    def display(value: object) -> str:
        if value is None or value == "":
            return "—"
        if isinstance(value, float):
            return f"{value:.6g}"
        try:
            if isinstance(value, str) and "." in value:
                return f"{float(value):.6g}"
        except ValueError:
            pass
        return str(value).replace("|", "\\|").replace("\n", " ")

    text = "# Полная таблица сравнения детекторов\n\n"
    text += "Все доступные показатели из сохранённых test-v2: COCO test, 4500 изображений на повтор, три повтора, FP32, batch 1. По столбцам — модели, по строкам — показатели. Исходные экспериментальные файлы не изменены. Полная точность чисел — в detection_full.csv; CSV записан в UTF-8 с BOM для Excel.\n\n"
    text += "| Показатель | " + " | ".join(row["model"] for row in rows) + " |\n"
    text += "| --- | " + " | ".join("---" for _ in rows) + " |\n"
    for key in rows[0]:
        if key != "model":
            text += f"| {key} | " + " | ".join(display(row[key]) for row in rows) + " |\n"
    text += "\n## Как читать показатели\n\n"
    text += "- mAP, AP50/AP75, AP_small/medium/large, AR1/10/100 и AR_small/medium/large, precision/recall/F1 и TP/FP/FN относятся к первому повтору. AP/AR представлены в диапазоне 0–1. Precision/recall/F1 и ошибки рассчитаны при score ≥ 0,5, IoU ≥ 0,5, max detections 100; настройки также перечислены в таблице.\n"
    text += "- p50_ms/p95_ms — объединённые измерения трёх повторов: preprocess + forward + postprocess. images_per_second = 1000 / среднее этого времени; чтение и запись исключены. Средние времена отдельных стадий, чтения и записи — взвешенные по числу изображений средние сохранённых повторов. cold_load_ms_mean — средняя холодная загрузка.\n"
    text += "- VRAM_allocated_MiB — пик памяти тензоров PyTorch, VRAM_reserved_MiB — пик его резерва; значения не складываются и не включают полную память CUDA-контекста/драйвера. RAM_peak_MiB — пик RSS процесса, включая загрузку. Пики берутся как максимум по повторам. weight_MiB — размер файлов весов на диске.\n"
    text += "- parameters — число параметров исполняемой сети. У YOLO исходная оболочка сохранила некорректный ноль; уточнение внутренней сети выполняется отдельным офлайн CPU-подсчётом, с проверкой SHA весов и ID исходного контейнера. Если уточнения нет, число отсутствует, а не принимается равным нулю.\n"
    text += "- Ошибки по всем 80 классам (AP, TP/FP/FN, precision/recall/F1) — detection_per_class.csv. Парные доверительные интервалы — detection_paired.csv; подробности bootstrap — detection_analysis.md и исходный bootstrap.json.\n"
    text += "\n## Покрытие плана и ограничения\n\n"
    text += "Основные метрики сравнения детекторов из раздела 5 плана включены. Сбои текущей версии отражает failures; три исходных сбоя RF-DETR версии test-v1 сохранены в recovery.json и исходных логах. Девять проходов перенесены без изменения; происхождение показывает reused_from. Офлайн-инференс выполнен в контейнерах без сети. Полные штатные настройки препроцессинга, разрешения и нормализации находятся в native_config.json каждого запуска.\n\n"
    text += "Полное потребление GPU по nvidia-smi в этих прогонах не измерялось. Сравнение на искажённых изображениях и отдельном внешнем наборе, классификация, captions и полный конвейер ещё не выполнены. Эти результаты нельзя получить только перестроением отчёта.\n"
    (output / "detection_full.md").write_text(text, encoding="utf-8")
    sources.extend([runs / "bootstrap.json"])
    manifest = {
        "schema_version": "1.0",
        "runs_dir": str(runs),
        "script_sha256": digest(Path(__file__)),
        "sources": {str(path): digest(path) for path in sources},
        "outputs": ["detection_full.csv", "detection_full.md", "detection_per_class.csv"],
    }
    (output / "detection_full_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(output / "detection_full.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Полный отчёт по сохранённым результатам детекции")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "reports/detection/test-v2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/comparisons")
    parser.add_argument("--collect-yolo-metadata", action="store_true")
    args = parser.parse_args()
    if args.collect_yolo_metadata:
        collect_yolo_metadata(args.runs_dir.resolve(), args.output_dir.resolve())
    else:
        build(args.runs_dir.resolve(), args.output_dir.resolve())
