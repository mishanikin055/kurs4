"""Build precise paired tables and an analysis of completed detection artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(runs: Path, output: Path) -> None:
    comparison = runs / "comparison/detection.csv"
    bootstrap_path = runs / "bootstrap.json"
    with comparison.open() as stream:
        rows = list(csv.DictReader(stream))
    bootstrap = json.loads(bootstrap_path.read_text())
    registry = json.loads((ROOT / "detection/configs/models.json").read_text())
    names = {model["id"]: model["name"] for model in registry["models"]}
    by_name = {row["model"]: row for row in rows}
    if len(rows) != 4 or set(by_name) != set(names.values()):
        raise ValueError("Analysis requires all four detectors")
    if any(
        row["status"] != "complete"
        or int(row["n_images"]) != 4500
        or int(row["completed_repeats"]) != 3
        for row in rows
    ):
        raise ValueError("Analysis requires complete 4500-image, three-repeat results")
    if (
        bootstrap["samples"] != 1000
        or bootstrap["shortened"]
        or set(bootstrap["raw_mAP"]) != set(names)
    ):
        raise ValueError("Analysis requires the complete paired 1000-sample bootstrap")
    if any(len(values) != 1000 for values in bootstrap["raw_mAP"].values()):
        raise ValueError("Bootstrap values are incomplete")
    output.mkdir(parents=True, exist_ok=True)
    paired = []
    for key, interval in bootstrap["paired_mAP_difference_ci95"].items():
        first, second = key.split(" - ")
        paired.append(
            {
                "model_a": names[first],
                "model_b": names[second],
                "mAP_a_minus_b": float(by_name[names[first]]["mAP"])
                - float(by_name[names[second]]["mAP"]),
                "ci95_low": interval[0],
                "ci95_high": interval[1],
                "includes_zero": interval[0] <= 0 <= interval[1],
            }
        )
    with (output / "detection_paired.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    environment_path = next(runs.glob("*/environment.json"))
    gpu = json.loads(environment_path.read_text())["gpu"]
    text = f"# Анализ полного сравнения детекторов\n\nCOCO test: 4500 изображений; FP32, batch 1, три повтора, {gpu}. Качество — первый повтор; задержка — все три. Парный image bootstrap: 1000 выборок, percentile CI95. Все числа взяты из сохранённых локальных артефактов.\n\n"
    text += "| Модель | mAP | CI95 mAP | p50, мс | p95, мс |\n| --- | --- | --- | --- | --- |\n"
    for model, name in names.items():
        row = by_name[name]
        low, high = bootstrap["mAP_ci95"][model]
        text += f"| {name} | {float(row['mAP']):.4f} | [{low:.6g}; {high:.6g}] | {float(row['p50_ms']):.4f} | {float(row['p95_ms']):.4f} |\n"
    text += "\n| Разность mAP | Оценка | CI95 low | CI95 high | Включает ноль |\n| --- | --- | --- | --- | --- |\n"
    for row in paired:
        text += f"| {row['model_a']} − {row['model_b']} | {row['mAP_a_minus_b']:.6g} | {row['ci95_low']:.6g} | {row['ci95_high']:.6g} | {'да' if row['includes_zero'] else 'нет'} |\n"
    text += "\n## Выводы для этих checkpoint и условий\n\n"
    best = max(rows, key=lambda row: float(row["mAP"]))
    fastest = min(rows, key=lambda row: float(row["p50_ms"]))

    def interval_for(first: str, second: str) -> tuple[float, float]:
        for pair in paired:
            if (pair["model_a"], pair["model_b"]) == (first, second):
                return pair["ci95_low"], pair["ci95_high"]
            if (pair["model_b"], pair["model_a"]) == (first, second):
                return -pair["ci95_high"], -pair["ci95_low"]
        raise ValueError("Missing paired interval")

    text += f"- Максимальный mAP у {best['model']}; p50 {float(best['p50_ms']):.4f} мс.\n"
    if all(
        interval_for(best["model"], other["model"])[0] > 0 for other in rows if other is not best
    ):
        text += "  Парные интервалы его разности с каждым из остальных детекторов полностью положительны; преимущество устойчиво в данном bootstrap.\n"
    text += f"- Минимальная задержка у {fastest['model']}: p50 {float(fastest['p50_ms']):.4f} мс.\n"
    low, high = interval_for(names["rtdetr_v2_r18vd"], names["yolo26s"])
    text += f"- RT-DETR − YOLO: CI [{low:.6g}; {high:.6g}]. "
    if min(abs(low), abs(high)) < 1e-5:
        text += "Граница почти совпадает с нулём: при 1000 выборках это слабое основание для категоричного ранжирования. Округление до четырёх знаков скрывает малую границу. "
    text += "Точные границы сохранены здесь, в detection_paired.csv и bootstrap.json.\n"
    low, high = interval_for(names["yolo26s"], names["fasterrcnn_resnet50_fpn_v2"])
    text += f"- YOLO − Faster R-CNN: CI [{low:.6g}; {high:.6g}]. "
    if low <= 0 <= high:
        text += (
            "Интервал включает ноль: надёжное превосходство по mAP этим сравнением не установлено. "
        )
    faster = by_name[names["fasterrcnn_resnet50_fpn_v2"]]
    text += f"p50 Faster R-CNN: {float(faster['p50_ms']):.4f} мс.\n"
    text += "\nОграничения: внутренний COCO test, одно оборудование, разные штатные разрешения/предобучение. Интервалы относятся к качеству на подобных изображениях, а не ко всем возможным сценам; показаны шесть отдельных парных CI без поправки на множественные сравнения. Набор не является доказанным рейтингом популярности.\n"
    if (runs / "recovery.json").exists():
        recovery = json.loads((runs / "recovery.json").read_text())
        text += f"\nВосстановленная версия: {len(recovery['carried_forward_runs'])} проходов перенесены из {recovery['from_experiment']} с исходными environment manifests. RF-DETR измерялся позже после исправления sparse slots; исходные сбои сохранены.\n"
    (output / "detection_analysis.md").write_text(text)
    manifest = {
        "schema_version": "1.0",
        "comparison": str(comparison.relative_to(ROOT)),
        "comparison_sha256": digest(comparison),
        "bootstrap": str(bootstrap_path.relative_to(ROOT)),
        "bootstrap_sha256": digest(bootstrap_path),
        "script_sha256": digest(Path(__file__)),
    }
    (output / "detection_analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(output / "detection_analysis.md", output / "detection_paired.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Анализ полных локальных детекторных результатов")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "reports/detection/test-v2")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/comparisons")
    args = parser.parse_args()
    build(args.runs_dir.resolve(), args.output_dir.resolve())
