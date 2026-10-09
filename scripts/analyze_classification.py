"""Build descriptive analyses from verified whole-image and crop results, without CI."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classification.common import registry
from detection.common import ROOT, read_json, sha256, write_json

MODES = {
    "validation50000": ("validation50000-v1", "classification.csv", "ImageNet validation50k"),
    "gt": ("gt-crops-v1", "classification_crops.csv", "COCO GT-crops"),
    "detector": ("detector-crops-v1", "classification_crops.csv", "COCO detector-crops"),
}
EXPECTED = {
    "validation50000": (2, 50000, 50000),
    "gt": (3, 1883, 1883),
    "detector": (3, 2296, 1054),
}


def verify_resources(row: dict, run_dirs: list[Path], crop: bool) -> None:
    """Recompute pooled timing and peak resources independently of report builders."""
    timings = []
    writes = []
    metadata = []
    for directory in run_dirs:
        metadata.append(read_json(directory / "run.json"))
        with (directory / ("samples.jsonl" if crop else "timings.jsonl")).open() as handle:
            timings.extend(json.loads(line)["timing"] for line in handle)
        with (directory / "write_timings.jsonl").open() as handle:
            writes.extend(json.loads(line)["write_ms"] for line in handle)
    totals = np.array([r["total_ms"] for r in timings])
    expected = {
        "p50_ms": float(np.percentile(totals, 50)),
        "p95_ms": float(np.percentile(totals, 95)),
        "write_mean_ms": float(np.mean(writes)),
        "cold_load_mean_ms": float(np.mean([m["cold_load_ms"] for m in metadata])),
    }
    if not crop:
        expected["images_per_second"] = 1000 / float(totals.mean())
    for stage in ["preprocess", "inference", "postprocess", "read"] + (["crop"] if crop else []):
        expected[f"{stage}_mean_ms"] = float(np.mean([r[f"{stage}_ms"] for r in timings]))
    for label, resource in [
        ("RAM", "rss"),
        ("VRAM_allocated", "cuda_allocated"),
        ("VRAM_reserved", "cuda_reserved"),
    ]:
        expected[f"{label}_peak_MiB"] = (
            max(m["resources"][f"{resource}_peak_bytes"] for m in metadata) / 1024**2
        )
    for key, value in expected.items():
        if not math.isclose(float(row[key]), value, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError(f"CSV resource/timing differs for {row['model']}: {key}")


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
    for name, key in [
        ("experiment.json", "experiment_sha256"),
        ("source_snapshot.tar.gz", "source_snapshot_sha256"),
    ]:
        if sha256(ROOT / "reports/classification" / experiment / name) != verification[key]:
            raise ValueError("Experiment identity or source snapshot changed after verification")
    models = registry()["models"]
    if set(r["model"] for r in rows) != {m["name"] for m in models} or len(rows) != 4:
        raise ValueError("Analysis requires the four frozen models")
    reported_repeats = verification.get("reported_repeats", 3)
    repeats, n_crops, n_eligible = EXPECTED[mode]
    if reported_repeats != repeats:
        raise ValueError("Analysis repeat selection differs from the final protocol")
    manifest_name = (
        "classification_report_manifest.json"
        if mode == "validation50000"
        else "report_manifest.json"
    )
    manifest = read_json(report_dir / manifest_name)
    generator = (
        ROOT / "classification" / ("report.py" if mode == "validation50000" else "crops/report.py")
    )
    if manifest["include_ci"] or manifest["script_sha256"] != sha256(generator):
        raise ValueError("Report provenance differs from the current report generator")
    if mode == "validation50000" and manifest["repeat_selection"] != {
        "reported_repeats": repeats,
        "original_config_repeats": 3,
        "quality_repeat": 1,
        "excluded_runs": ["resnet50_repeat3"],
    }:
        raise ValueError("ImageNet report must retain and exclude the third ResNet-50 run")
    if any(
        r["status"] != "complete" or int(r["completed_repeats"]) != reported_repeats for r in rows
    ):
        raise ValueError("Analysis requires complete full-size results for the selected repeats")
    for spec in models:
        verified = verification["models"][spec["id"]]
        row = next(r for r in rows if r["model"] == spec["name"])
        if not verified["predictions_identical_between_repeats"]:
            raise ValueError("Repeat predictions differ")
        first = verified["runs"][0]
        if len(verified["runs"]) != repeats or [r["run"] for r in verified["runs"]] != [
            f"{spec['id']}_repeat{i}" for i in range(1, repeats + 1)
        ]:
            raise ValueError("Verification does not select the first required repeats")
        for key, expected in (
            [("n_images", n_crops)]
            if mode == "validation50000"
            else [("n_crops", n_crops), ("n_eligible_crops", n_eligible), ("n_scenes", 500)]
        ):
            if int(row[key]) != expected or any(r[key] != expected for r in verified["runs"]):
                raise ValueError("CSV or verification has an unexpected denominator")
        for key in (
            ["top1", "top5", "macro_f1"]
            if mode == "validation50000"
            else [
                "top1_coarse",
                "top5_coarse",
                "macro_f1_observed",
                "mapping_object_coverage",
                "acceptance_rate",
                "accepted_top1_accuracy",
            ]
            + (
                [
                    "detector_only_matched_accuracy",
                    "hypothetical_gated_matched_accuracy",
                    "helpful_changes",
                    "harmful_changes",
                    "missed_supported_gt",
                    "unmatched_proposals",
                    "matched_unsupported_proposals",
                    "crowd_ignored_proposals",
                    "detector_only_supported_recall",
                    "hypothetical_gated_supported_recall",
                ]
                if mode == "detector"
                else []
            )
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
        verify_resources(
            row,
            [ROOT / "reports/classification" / experiment / r["run"] for r in verified["runs"]],
            mode != "validation50000",
        )
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


def resource_table(rows: list[dict]) -> str:
    return table(
        [
            "Модель",
            "Preprocess, мс",
            "Forward, мс",
            "Postprocess, мс",
            "Холодная загрузка, мс",
            "VRAM allocated, MiB",
        ],
        [
            [r["model"]]
            + [
                f"{float(r[key]):.2f}"
                for key in [
                    "preprocess_mean_ms",
                    "inference_mean_ms",
                    "postprocess_mean_ms",
                    "cold_load_mean_ms",
                    "VRAM_allocated_peak_MiB",
                ]
            ]
            for r in rows
        ],
    )


def analyze(mode: str, rows: list[dict], verification: dict, output: Path) -> str:
    _, _, title = MODES[mode]
    key = "top1" if mode == "validation50000" else "top1_coarse"
    best = max(rows, key=lambda r: float(r[key]))
    f1_key = "macro_f1" if mode == "validation50000" else "macro_f1_observed"
    best_f1 = max(rows, key=lambda r: float(r[f1_key]))
    fast = min(rows, key=lambda r: float(r["p50_ms"]))
    text = f"# Сравнительный анализ: {title}\n\n"
    repeats = verification.get("reported_repeats", 3)
    processed = sum(m["processed_total"] for m in verification["models"].values())
    text += f"Все четыре готовых checkpoint завершили выбранные для анализа повторы: {repeats} на модель. Обучение, fine-tuning и LoRA не выполнялись. Качество — первый повтор; время и пиковые ресурсы — первые {repeats} повтора. Предсказания, включая top5 и softmax, совпали между выбранными повторами. Новые доверительные интервалы не рассчитывались.\n\n"
    if mode == "validation50000":
        text += f"Полная официальная ImageNet validation: 50 000 одинаковых изображений, 1000 классов по 50 изображений. Источник ILSVRC/imagenet-1k, revision 49e2ee26f3810fb5a7536bbf732a7b07389a47b5. Настройки на этом наборе не подбирались. В анализ включено {processed} обработок, число разных изображений — 50 000. Предварительная evaluation5000 входит в эти 50 000 и не является независимым набором.\n\n"
        text += "Почему 50 000: использован весь официальный проверочный набор ImageNet, по 50 снимков для каждого из 1000 классов. Размер не подбирался под победителя. Предварительные 5000 (по 5 на класс) служили быстрым первым сравнением, затем пользователь поручил полную проверку. Они входят в полный набор и не образуют второе независимое подтверждение.\n\n"
        if verification.get("excluded_completed_runs"):
            names = ", ".join(r["run"] for r in verification["excluded_completed_runs"])
            text += f"По изменённому указанию пользователя от 09.10.2026 сравнение заканчивается после двух кругов. Исходный repeats=3 сохранён в experiment.json; выбор первых двух повторов записан отдельно в report/verification manifests. Уже готовые дополнительные проходы ({names}) сохранены и исключены из основной таблицы.\n\n"
    else:
        first = next(iter(verification["models"].values()))["runs"][0]
        text += "Почему 500 сцен: прикладная проверка обрабатывает отдельные объекты, которых в одном снимке несколько. Ограниченный случайный набор сцен удерживал время эксперимента в разумных пределах и позволял сопоставить одинаковые сцены по эталонным и найденным рамкам. 1883 и 2296 — получившееся число вырезок, а не отдельно выбранные размеры выборки. Этот режим не заменяет полный ImageNet и не гарантирует качество редких категорий.\n\n"
        text += f"500 заранее выбранных сцен COCO test, seed 42; {first['n_crops']} вырезок, в условную точность включены {first['n_eligible_crops']}. Словарь: 287 ImageNet synsets → 56 категорий COCO. Поддерживаемые GT составляют {100 * first['mapping_object_coverage']:.2f}% non-crowd объектов сцен. Неотображаемый ответ на поддерживаемом объекте считается ошибкой. Macro-F1 усредняется по {first['observed_categories']} категориям с ненулевым GT-support.\n\n"
        text += "Top-1/Top-5 оценивают общую категорию COCO, а не породу или подтип. Пять ImageNet-классов могут отображаться в одну COCO-категорию. Точность тонких подтипов по этим данным не измеряется.\n\n"
        text += f"За все модели и три повтора выполнено {processed} обработок; разных вырезок — {first['n_crops']}, сцен — 500. Повторы не увеличивают число независимых объектов.\n\n"
    text += quality_table(rows, mode != "validation50000") + "\n\n"
    text += f"Наибольшая Top-1 в этом режиме у {best['model']}: {100 * float(best[key]):.3f}%. Минимальная p50 у {fast['model']}: {float(fast['p50_ms']):.2f} мс. Это описательное ранжирование измеренных значений.\n\n"
    n = EXPECTED[mode][2]
    text += (
        "Число правильных первых ответов: "
        + "; ".join(f"{row['model']} — {round(n * float(row[key]))} из {n}" for row in rows)
        + ".\n\n"
    )
    text += f"Максимальный Macro-F1 у {best_f1['model']}: {float(best_f1[f1_key]):.5f}. Этот критерий одинаково взвешивает присутствующие классы, тогда как Top-1 зависит от числа объектов каждого класса.\n\n"
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
            text += f"Исходный детектор правильно классифицирует {round(n * float(rows[0]['detector_only_matched_accuracy']))} из {n} matched-supported объектов. Во всех четырёх строках гипотетическая замена вносит больше ошибок, чем исправляет; оснований для автоматической замены категории по этому правилу нет.\n\n"
    text += resource_table(rows) + "\n\n"
    text += f"Задержки p50/p95 вычислены по объединённым измерениям первых {repeats} повторов. Средние стадий и холодной загрузки приведены в таблице выше; ресурсы — максимальные пики среди выбранных повторов. Холодная загрузка не включает очистку page cache. Развёрнутые CSV содержат также чтение, запись/fsync и, для вырезок, кадрирование.\n\n"
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
        for path in sorted(directory.iterdir()):
            if path.suffix in {".csv", ".md", ".json"}:
                sources[str(path.relative_to(ROOT))] = sha256(path)
        path = ROOT / "reports/verification" / f"classification-{MODES[mode][0]}.json"
        sources[str(path.relative_to(ROOT))] = sha256(path)
        for name in ["experiment.json", "source_snapshot.tar.gz"]:
            path = ROOT / "reports/classification" / MODES[mode][0] / name
            sources[str(path.relative_to(ROOT))] = sha256(path)
    whole = loaded["validation50000"][0]
    gt = loaded["gt"][0]
    det = loaded["detector"][0]
    best_whole = max(whole, key=lambda r: float(r["top1"]))
    best_gt = max(gt, key=lambda r: float(r["top1_coarse"]))
    best_det = max(det, key=lambda r: float(r["top1_coarse"]))
    text = "# Описание результатов сравнения классификаторов\n\n"
    whole_repeats = loaded["validation50000"][1].get("reported_repeats", 3)
    text += f"Завершены три отдельных режима сравнения четырёх готовых моделей: ImageNet validation50k ({whole_repeats} повтора на модель), GT-вырезки COCO и вырезки RF-DETR Small (по три повтора, уже выполненных до изменения расписания). Выбранные для анализа повторы успешны; предсказания совпадают между ними. Прежний третий полный ResNet-50 сохранён отдельно от основной таблицы. Новые доверительные интервалы не рассчитывались, обучение не выполнялось.\n\n"
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
    text += "Знаменатели качества: ImageNet — 50 000 изображений (1000 классов по 50); GT-crops — 1883 поддерживаемых объекта из 500 сцен; detector-crops — 1054 matched-supported объекта из 2296 рамок тех же сцен. При выбранных повторах выполнено 400 000, 22 596 и 27 552 обработки соответственно. Это 50 000 разных ImageNet-изображений и 500 разных COCO-сцен; повторная обработка не расширяет набор.\n\n"
    text += f"Для классификации целого изображения наибольшая Top-1 у {best_whole['model']}. На GT-вырезках — {best_gt['model']}, на вырезках выбранного детектора — {best_det['model']}. Для последующего этапа конвейера предлагается {best_det['model']} как классификатор вырезок с максимальной измеренной условной Top-1; сравнение времени и памяти приведено в отдельных таблицах. Выбор не подтверждает точность пород и подтипов: в COCO нет соответствующего эталона.\n\n"
    best_macro = max(det, key=lambda r: float(r["macro_f1_observed"]))
    fastest = min(det, key=lambda r: float(r["p50_ms"]))
    text += f"Выбор зависит от критерия: в detector-crops наибольший Macro-F1 у {best_macro['model']} ({float(best_macro['macro_f1_observed']):.5f}), минимальная p50 — у {fastest['model']} ({float(fastest['p50_ms']):.2f} мс). {best_det['model']} правильно определяет {round(1054 * float(best_det['top1_coarse']))} из 1054 объектов; его p50 {float(best_det['p50_ms']):.2f} мс, RAM {float(best_det['RAM_peak_MiB']):.2f} MiB и VRAM reserved {float(best_det['VRAM_reserved_peak_MiB']):.0f} MiB. Эти ресурсы укладываются в измеренный режим ноутбука, но скорость полного приложения ещё не проверена.\n\n"
    harmed = sum(
        float(r["hypothetical_gated_matched_accuracy"]) < float(r["detector_only_matched_accuracy"])
        for r in det
    )
    best_gated = max(det, key=lambda r: float(r["hypothetical_gated_matched_accuracy"]))
    text += f"При гипотетической замене категории фиксированное правило снижает matched accuracy относительно detector-only у {harmed} из четырёх моделей. Наибольшая matched accuracy этого правила у {best_gated['model']}: {100 * float(best_gated['hypothetical_gated_matched_accuracy']):.3f}%, detector-only — {100 * float(best_gated['detector_only_matched_accuracy']):.3f}%.\n\n"
    text += "Из 1883 supported GT рамки сопоставлены с 1054, пропущены 829. Detector-only верно определяет категорию 1014 из 1054 сопоставленных объектов; его supported recall — 53.850%. Классификатор не восстанавливает пропущенные рамки. У всех четырёх моделей гипотетическая замена исправляет меньше категорий, чем портит, поэтому её не предлагается включать в конвейер.\n\n"
    text += "В сохранённых результатах исходная метка детектора не заменяется. Классификатор выдаёт отдельные top-k, synset, mapping и статус согласия/конфликта/отказа. Рабочая калибровка отказа и правила замены остаётся отдельной задачей на независимых данных; диагностические test-результаты не используются для подбора порогов.\n\n"
    text += "Ограничения: mapping покрывает 56 категорий и 55.81% non-crowd GT выбранных сцен; словарь ImageNet закрытый и не покрывает все объекты/подтипы. Для detector-crops использован только RF-DETR Small, максимум 100 рамок и score ≥ 0.5; качество после других детекторов не измерялось. Модели имеют разные native transforms и обучающие рецепты. Задержки относятся к указанному ноутбуку, FP32/batch1 и модельным стадиям; API, очередь и интерфейс ещё не измерены. Приложение и аннотаторы не входят в выполненную здесь реализацию.\n\n"
    text += "Подробные результаты: [ImageNet](reports/comparisons/classification-validation50000-v1/analysis.md), [GT-crops](reports/comparisons/classification-gt-crops-v1/analysis.md), [detector-crops](reports/comparisons/classification-detector-crops-v1/analysis.md). Исходные программы и команды — [classification/README.md](classification/README.md) и [classification/crops/README.md](classification/crops/README.md).\n"
    text += "\nПочему выбраны такие объёмы: предварительные 5000 ImageNet-снимков (по 5 на класс) дали быстрый первый результат; итоговые 50 000 — весь официальный проверочный набор (по 50 на класс), чтобы не ограничивать основной вывод маленькой подвыборкой. Для прикладных вырезок выбраны 500 COCO-сцен как отдельный бюджет проверки на объектах; из них автоматически получились 1883 эталонные и 2296 найденные вырезки. Число изображений не подбиралось по полученному качеству. Два полных круга вместо трёх — последующее указание пользователя о времени работы, а не уменьшение 50 000 разных снимков.\n"
    target = ROOT / "Описание результатов классификации.md"
    target.write_text(text, encoding="utf-8")
    sources[str(target.relative_to(ROOT))] = sha256(target)
    if plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
        offsets = {
            "ResNet-50": ((-8, -12), "right", "top"),
            "EfficientNetV2-S": ((-8, 8), "right", "bottom"),
            "ConvNeXt-Tiny": ((8, -12), "left", "top"),
            "ViT-B/16": ((8, 8), "left", "bottom"),
        }
        for axis, (mode, (rows, _, _)) in zip(axes, loaded.items(), strict=True):
            for row in rows:
                x = float(row["p50_ms"])
                y = 100 * float(row["top1" if mode == "validation50000" else "top1_coarse"])
                axis.scatter(x, y, s=50)
                offset, horizontal, vertical = offsets[row["model"]]
                axis.annotate(
                    row["model"],
                    (x, y),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=8,
                    ha=horizontal,
                    va=vertical,
                )
            axis.set(title=MODES[mode][2], xlabel="p50, мс", ylabel="Top-1, %")
            axis.margins(x=0.4, y=0.3)
            axis.grid(alpha=0.25)
        path = ROOT / "reports/comparisons/classification_quality_time.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        sources[str(path.relative_to(ROOT))] = sha256(path)
    write_json(
        ROOT / "reports/comparisons/classification_analysis_manifest.json",
        {
            "schema_version": "1.1",
            "script_sha256": sha256(Path(__file__)),
            "confidence_intervals": "not calculated",
            "repeat_selection": {
                "validation50000": [1, 2],
                "gt": [1, 2, 3],
                "detector": [1, 2, 3],
            },
            "quality_repeat": 1,
            "excluded_completed_runs": ["validation50000-v1/resnet50_repeat3"],
            "resource_verification": "pooled timings and peak run resources recomputed from artifacts",
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
