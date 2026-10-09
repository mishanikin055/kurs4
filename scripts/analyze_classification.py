"""Build descriptive analyses from verified whole-image and crop results, without CI."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classification.common import registry
from detection.common import ROOT, read_json, sha256, write_json

MODES = {
    "validation50000": ("validation50000-v1", "classification.csv", "ImageNet validation50k"),
    "gt": ("gt-crops-v1", "classification_crops.csv", "COCO GT-crops"),
    "detector": ("detector-crops-v1", "classification_crops.csv", "COCO detector-crops"),
}


def table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(row) + " |" for row in rows]
    )


def load(mode: str) -> tuple[list[dict], dict, Path]:
    experiment, filename, _ = MODES[mode]
    report_dir = ROOT / "reports/comparisons" / f"classification-{experiment}"
    with (report_dir / filename).open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    verification = read_json(ROOT / "reports/verification" / f"classification-{experiment}.json")
    models = registry()["models"]
    if set(r["model"] for r in rows) != {m["name"] for m in models} or len(rows) != 4:
        raise ValueError("Analysis requires the four frozen models")
    if any(r["status"] != "complete" or r["completed_repeats"] != "3" for r in rows):
        raise ValueError("Analysis requires complete full-size, three-repeat results")
    for spec in models:
        verified = verification["models"][spec["id"]]
        row = next(r for r in rows if r["model"] == spec["name"])
        if not verified["predictions_identical_between_repeats"]:
            raise ValueError("Repeat predictions differ")
        first = verified["runs"][0]
        for key in (
            ["top1", "top5", "macro_f1"]
            if mode == "validation50000"
            else ["top1_coarse", "top5_coarse", "macro_f1_observed"]
        ):
            if float(row[key]) != first[key]:
                raise ValueError("CSV differs from verified quality")
        for verified_run in verified["runs"]:
            run_dir = ROOT / "reports/classification" / experiment / verified_run["run"]
            if any(
                sha256(run_dir / name) != digest
                for name, digest in verified_run["artifacts"].items()
            ):
                raise ValueError("Artifacts changed after verification")
    return rows, verification, report_dir


def quality_table(rows: list[dict], crop: bool = False) -> str:
    keys = (
        ["top1_coarse", "top5_coarse", "macro_f1_observed"]
        if crop
        else ["top1", "top5", "macro_f1"]
    )
    return table(
        [
            "Модель",
            "Top-1, %",
            "Top-5, %",
            "Macro-F1",
            "p50, мс",
            "p95, мс",
            "RAM, MiB",
            "VRAM reserved, MiB",
        ],
        [
            [
                r["model"],
                f"{100 * float(r[keys[0]]):.3f}",
                f"{100 * float(r[keys[1]]):.3f}",
                f"{float(r[keys[2]]):.5f}",
            ]
            + [
                f"{float(r[k]):.2f}"
                for k in ["p50_ms", "p95_ms", "RAM_peak_MiB", "VRAM_reserved_peak_MiB"]
            ]
            for r in rows
        ],
    )


def analyze(mode: str, rows: list[dict], verification: dict, output: Path) -> str:
    _, _, title = MODES[mode]
    key = "top1" if mode == "validation50000" else "top1_coarse"
    best = max(rows, key=lambda r: float(r[key]))
    fast = min(rows, key=lambda r: float(r["p50_ms"]))
    text = f"# Сравнительный анализ: {title}\n\n"
    text += "Все четыре готовых checkpoint завершили три повтора. Обучение, fine-tuning и LoRA не выполнялись. Качество — первый повтор; время и пиковые ресурсы — три повтора. Предсказания, включая top5 и softmax, совпали между повторами. Новые доверительные интервалы не рассчитывались.\n\n"
    if mode == "validation50000":
        text += "Полная официальная ImageNet validation: 50 000 одинаковых изображений, 1000 классов по 50 изображений. Источник ILSVRC/imagenet-1k, revision 49e2ee26f3810fb5a7536bbf732a7b07389a47b5. Настройки на этом наборе не подбирались. Выполнено 600 000 обработок, число разных изображений — 50 000. Предварительная evaluation5000 входит в эти 50 000 и не является независимым набором.\n\n"
    else:
        first = next(iter(verification["models"].values()))["runs"][0]
        text += f"500 заранее выбранных сцен COCO test, seed 42; {first['n_crops']} вырезок, в условную точность включены {first['n_eligible_crops']}. Словарь: 287 ImageNet synsets → 56 категорий COCO. Поддерживаемые GT составляют {100 * first['mapping_object_coverage']:.2f}% non-crowd объектов сцен. Неотображаемый ответ на поддерживаемом объекте считается ошибкой. Macro-F1 усредняется по {first['observed_categories']} категориям с ненулевым GT-support.\n\n"
        text += "Top-1/Top-5 оценивают общую категорию COCO, а не породу или подтип. Пять ImageNet-классов могут отображаться в одну COCO-категорию. Точность тонких подтипов по этим данным не измеряется.\n\n"
    text += quality_table(rows, mode != "validation50000") + "\n\n"
    text += f"Наибольшая Top-1 в этом режиме у {best['model']}: {100 * float(best[key]):.3f}%. Минимальная p50 у {fast['model']}: {float(fast['p50_ms']):.2f} мс. Это описательное ранжирование измеренных значений.\n\n"
    if mode == "validation50000":
        second = sorted(rows, key=lambda r: float(r[key]), reverse=True)[1]
        text += f"Разница между первым и вторым результатом ({second['model']}) составляет {100 * (float(best[key]) - float(second[key])):.3f} процентного пункта, или {round(50000 * (float(best[key]) - float(second[key])))} правильных первых ответов. Их отношение медианных задержек — {float(best['p50_ms']) / float(second['p50_ms']):.2f}.\n\n"
        text += "Штатные входы: EfficientNetV2-S 384×384; ResNet-50, ConvNeXt-Tiny и ViT-B/16 224×224 с разными resize. Отличаются обучающие рецепты готовых весов. Разницу нельзя приписывать только архитектуре. Сохранённые повторы ResNet-50 выполнены ранее, остальные — при продолжении; порядок временных измерений не полностью перемешан. Влияние состояния ноутбука на задержки не исключено.\n\n"
    else:
        text += (
            table(
                [
                    "Модель",
                    "Принятые ответы, %",
                    "Точность принятых на eligible, %",
                    "not_mappable",
                    "uncertain",
                ],
                [
                    [
                        r["model"],
                        f"{100 * float(r['acceptance_rate']):.2f}",
                        f"{100 * float(r['accepted_top1_accuracy']):.2f}",
                        str(
                            verification["models"][m["id"]]["runs"][0]["status_counts"].get(
                                "not_mappable", 0
                            )
                        ),
                        str(
                            verification["models"][m["id"]]["runs"][0]["status_counts"].get(
                                "uncertain", 0
                            )
                        ),
                    ]
                    for m in registry()["models"]
                    for r in rows
                    if r["model"] == m["name"]
                ],
            )
            + "\n\n"
        )
        text += "Доля принятых ответов считается среди всех non-crowd proposals, точность принятых — среди eligible с принятым ответом: у detector-crops знаменатели различаются. Пороги score ≥ 0.5 и top1−top2 ≥ 0.1 диагностические, зафиксированы до оценки и не менялись после просмотра результатов. Они не калиброваны на неизвестные объекты; высокий softmax не подтверждает принадлежность словарю.\n\n"
        if mode == "detector":
            text += f"Рамки RF-DETR Small repeat1: score ≥ 0.5, не более 100 на сцену. Сопоставление с GT без проверки категории, один к одному, IoU ≥ 0.5. Из {first['n_supported_gt']} поддерживаемых GT сопоставлены {first['matched_supported_gt']}, пропущены {first['missed_supported_gt']}; unmatched proposals — {first['unmatched_proposals']}, matched unsupported — {first['matched_unsupported_proposals']}, crowd ignored — {first['crowd_ignored_proposals']}. Классификатор не создаёт новых рамок.\n\n"
            text += (
                table(
                    [
                        "Модель",
                        "Detector-only accuracy, %",
                        "Гипотетическая замена, %",
                        "Исправлено",
                        "Внесено ошибок",
                        "Detector-only recall, %",
                        "Замена recall, %",
                    ],
                    [
                        [
                            r["model"],
                            f"{100 * float(r['detector_only_matched_accuracy']):.3f}",
                            f"{100 * float(r['hypothetical_gated_matched_accuracy']):.3f}",
                            r["helpful_changes"],
                            r["harmful_changes"],
                            f"{100 * float(r['detector_only_supported_recall']):.3f}",
                            f"{100 * float(r['hypothetical_gated_supported_recall']):.3f}",
                        ]
                        for r in rows
                    ],
                )
                + "\n\n"
            )
            text += "Гипотетическая замена использует категорию классификатора только при прохождении порогов. Эти значения оценивают возможное правило; в реальных сохранённых результатах effective_category_id и исходная метка детектора остаются неизменными. Условная matched accuracy исключает пропущенные, unsupported и crowd; supported recall учитывает пропуски. Ни одна из этих величин не заменяет COCO mAP полного детектора.\n\n"
    text += "FP32, batch 1, 20 прогревов, 4 CPU threads, RTX 4050 Laptop 6 GiB / i5-12450H, WSL2 Linux Docker, PyTorch 2.7.1 CUDA 12.8 / torchvision 0.22.1. Строго один модельный дочерний процесс с общим inference.lock, runtime без сети и токена. CUDA синхронизируется на границах стадий. p50/p95 включают preprocessing/transfer, forward и postprocessing; чтение, crop, запись/fsync и холодная загрузка вынесены отдельно. RAM — peak RSS; VRAM — пики PyTorch, не полная память GPU.\n\n"
    experiment = MODES[mode][0]
    text += f"Таблица и детализация — в этой папке; сырые predictions/timings/errors и source_snapshot — reports/classification/{experiment}/. Проверка — reports/verification/classification-{experiment}.json. Команды воспроизведения — classification/README.md и classification/crops/README.md; scripts/verify_classification.py проверяет хеши, эталоны, метрики и повторы, scripts/analyze_classification.py строит этот текст без нового инференса.\n"
    (output / "analysis.md").write_text(text, encoding="utf-8")
    return text


def build(plots: bool) -> None:
    loaded = {mode: load(mode) for mode in MODES}
    sources = {}
    for mode, (rows, verification, directory) in loaded.items():
        analyze(mode, rows, verification, directory)
        for name in [MODES[mode][1], "analysis.md"]:
            sources[str((directory / name).relative_to(ROOT))] = sha256(directory / name)
        path = ROOT / "reports/verification" / f"classification-{MODES[mode][0]}.json"
        sources[str(path.relative_to(ROOT))] = sha256(path)
    whole = loaded["validation50000"][0]
    gt = loaded["gt"][0]
    det = loaded["detector"][0]
    best_whole = max(whole, key=lambda r: float(r["top1"]))
    best_gt = max(gt, key=lambda r: float(r["top1_coarse"]))
    best_det = max(det, key=lambda r: float(r["top1_coarse"]))
    text = "# Описание результатов сравнения классификаторов\n\n"
    text += "Завершены три отдельных режима сравнения четырёх готовых моделей: ImageNet validation50k, GT-вырезки COCO и вырезки RF-DETR Small. Три повтора каждого режима успешны; предсказания совпадают между повторами. Новые доверительные интервалы не рассчитывались, обучение не выполнялось.\n\n"
    text += (
        table(
            [
                "Модель",
                "ImageNet Top-1, %",
                "GT-crops Top-1, %",
                "Detector-crops Top-1, %",
                "Detector-crops p50, мс",
            ],
            [
                [
                    r["model"],
                    f"{100 * float(r['top1']):.3f}",
                    f"{100 * float(g['top1_coarse']):.3f}",
                    f"{100 * float(d['top1_coarse']):.3f}",
                    f"{float(d['p50_ms']):.2f}",
                ]
                for r in whole
                for g in gt
                if g["model"] == r["model"]
                for d in det
                if d["model"] == r["model"]
            ],
        )
        + "\n\n"
    )
    text += "ImageNet проверяет 1000 классов целого изображения; COCO — общую категорию поддерживаемых объектов после mapping. Между столбцами разные задачи и знаменатели, их нельзя складывать в общий балл или трактовать как падение точности на одном наборе.\n\n"
    text += f"Для классификации целого изображения наибольшая Top-1 у {best_whole['model']}. На GT-вырезках — {best_gt['model']}, на вырезках выбранного детектора — {best_det['model']}. Для последующего этапа конвейера предлагается {best_det['model']} как классификатор вырезок с максимальной измеренной условной Top-1; сравнение времени и памяти приведено в отдельных таблицах. Выбор не подтверждает точность пород и подтипов: в COCO нет соответствующего эталона.\n\n"
    harmed = sum(
        float(r["hypothetical_gated_matched_accuracy"]) < float(r["detector_only_matched_accuracy"])
        for r in det
    )
    best_gated = max(det, key=lambda r: float(r["hypothetical_gated_matched_accuracy"]))
    text += f"При гипотетической замене категории фиксированное правило снижает matched accuracy относительно detector-only у {harmed} из четырёх моделей. Наибольшая matched accuracy этого правила у {best_gated['model']}: {100 * float(best_gated['hypothetical_gated_matched_accuracy']):.3f}%, detector-only — {100 * float(best_gated['detector_only_matched_accuracy']):.3f}%.\n\n"
    text += "В сохранённых результатах исходная метка детектора не заменяется. Классификатор выдаёт отдельные top-k, synset, mapping и статус согласия/конфликта/отказа. Рабочая калибровка отказа и правила замены остаётся отдельной задачей на независимых данных; диагностические test-результаты не используются для подбора порогов.\n\n"
    text += "Ограничения: mapping покрывает 56 категорий и 55.81% non-crowd GT выбранных сцен; словарь ImageNet закрытый и не покрывает все объекты/подтипы. Для detector-crops использован только RF-DETR Small, максимум 100 рамок и score ≥ 0.5; качество после других детекторов не измерялось. Модели имеют разные native transforms и обучающие рецепты. Задержки относятся к указанному ноутбуку, FP32/batch1 и модельным стадиям; API, очередь и интерфейс ещё не измерены. Приложение и аннотаторы не входят в выполненную здесь реализацию.\n\n"
    text += "Подробные результаты: [ImageNet](reports/comparisons/classification-validation50000-v1/analysis.md), [GT-crops](reports/comparisons/classification-gt-crops-v1/analysis.md), [detector-crops](reports/comparisons/classification-detector-crops-v1/analysis.md). Исходные программы и команды — [classification/README.md](classification/README.md) и [classification/crops/README.md](classification/crops/README.md).\n"
    target = ROOT / "Описание результатов классификации.md"
    target.write_text(text, encoding="utf-8")
    sources[str(target.relative_to(ROOT))] = sha256(target)
    if plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
        for axis, (mode, (rows, _, _)) in zip(axes, loaded.items(), strict=True):
            for row in rows:
                x = float(row["p50_ms"])
                y = 100 * float(row["top1" if mode == "validation50000" else "top1_coarse"])
                axis.scatter(x, y, s=50)
                axis.annotate(
                    row["model"], (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8
                )
            axis.set(title=MODES[mode][2], xlabel="p50, ms", ylabel="Top-1, %")
            axis.margins(x=0.4, y=0.3)
            axis.grid(alpha=0.25)
        path = ROOT / "reports/comparisons/classification_quality_time.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        sources[str(path.relative_to(ROOT))] = sha256(path)
    write_json(
        ROOT / "reports/comparisons/classification_analysis_manifest.json",
        {
            "schema_version": "1.0",
            "script_sha256": sha256(Path(__file__)),
            "confidence_intervals": "not calculated",
            "outputs_and_sources": sources,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Описательный анализ завершённой классификации без CI"
    )
    parser.add_argument(
        "--plots", action="store_true", help="Также создать график с matplotlib в CPU-среде"
    )
    build(parser.parse_args().plots)
