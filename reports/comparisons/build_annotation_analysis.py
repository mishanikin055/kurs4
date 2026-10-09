"""Build Russian interpretation and dataset coverage from completed caption artifacts."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from detection.common import ROOT, read_json, sha256, write_json

NAMES = {
    "blip_base": "BLIP base",
    "florence2_base_ft": "Florence-2 base-ft",
    "qwen3_vl_2b": "Qwen3-VL 2B Instruct",
    "smolvlm2_500m": "SmolVLM2 500M",
}


def build(experiment: Path, output: Path, document: Path, verification: Path) -> None:
    verified = read_json(verification)
    if verified["status"] != "verified" or verified["images_per_model"] != 500:
        raise ValueError("Analysis requires successful artifact verification")
    manifest = read_json(output / "report_manifest.json")
    if verified["report_manifest_sha256"] != sha256(output / "report_manifest.json"):
        raise ValueError("Report changed after numerical verification")
    for name, digest in verified["report_outputs"].items():
        if sha256(output / name) != digest:
            raise ValueError("Report output changed after verification")
    rows = manifest["rows"]
    if len(rows) != 4 or any(r["status"] != "complete" for r in rows):
        raise ValueError("Analysis requires all four completed participants")
    dataset_path = experiment / "blip_base/dataset.json"
    dataset = read_json(dataset_path)
    ids = {r["image_id"] for r in dataset["images"]}
    gt_path = ROOT / "data/coco/annotations/instances_val2017.json"
    if sha256(gt_path) != dataset["instances_sha256"] or len(ids) != 500:
        raise ValueError("Dataset or ground truth changed")
    gt = read_json(gt_path)
    counts: Counter[int] = Counter()
    crowd: Counter[int] = Counter()
    scenes: dict[int, set[int]] = defaultdict(set)
    for item in gt["annotations"]:
        if item["image_id"] in ids:
            counts[item["category_id"]] += 1
            crowd[item["category_id"]] += item["iscrowd"]
            scenes[item["category_id"]].add(item["image_id"])
    coverage = [
        {
            "category_id": c["id"],
            "category": c["name"],
            "instances": counts[c["id"]],
            "crowd_instances": crowd[c["id"]],
            "scenes": len(scenes[c["id"]]),
        }
        for c in gt["categories"]
    ]
    with (output / "dataset_category_coverage.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(coverage[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(coverage)
    crop_path = ROOT / "reports/classification/crop-data-v1/gt_crops.json"
    crops = read_json(crop_path)
    intersection = len(ids & {r["image_id"] for r in crops["scenes"]})
    missing = [c["category"] for c in coverage if not c["scenes"]]
    rare = [c for c in coverage if 0 < c["scenes"] <= 4]
    lexical_best = max(rows, key=lambda r: r["CIDEr"])
    fastest = min(rows, key=lambda r: r["total_ms_p50"])
    hallucination_best = min(rows, key=lambda r: r["CHAIRi"])
    registry_path = ROOT / "annotation/configs/models.json"
    checkpoints = [
        "## Закреплённые модели",
        "",
        "Четыре заранее выбранных представителя captioning и компактных VLM; доказанного рейтинга популярности не строилось. Revisions и лицензии проверены по официальным источникам перед test.",
        "",
        "| Checkpoint | Revision | Лицензия checkpoint |",
        "|---|---|---|",
    ]
    for spec in read_json(registry_path)["models"]:
        url = f"{spec['source']}/blob/{spec['revision']}/README.md"
        checkpoints.append(
            f"| [{spec['repo_id']}]({url}) | {spec['revision']} | {spec['license']} |"
        )
    checkpoints += [
        "",
        "У Florence карта checkpoint указывает MIT; авторские файлы Python также содержат уведомления Apache 2.0, они сохранены. Лицензионные сведения и SHA выбранных файлов — docs/annotation_model_selection.md и annotation/models/manifest.json.",
    ]
    text = [
        "# Описание результатов аннотирования",
        "",
        "Автоматическое сравнение четырёх готовых checkpoint завершено: по одному полному проходу на одинаковых 500 COCO-изображениях, всего 2000 captions. Обучение, fine-tuning и LoRA не проводились. Новые доверительные интервалы не вычислялись. Экспертные оценки и внешний набор пока отсутствуют.",
        "",
        "## Данные и границы выводов",
        "",
        f"Набор — случайная выборка без возвращения из существующего detector test4500 (COCO val2017): shuffle с seed 42, первые 500 image_id. Dev100 взят отдельно из detector dev500; пересечений нет. В наборе {sum(len(r['references']) for r in dataset['images'])} эталонных captions, по пять на изображение, и {sum(counts.values())} GT instances, включая crowd. Представлены {sum(bool(c['scenes']) for c in coverage)}/80 категорий; отсутствует {', '.join(missing)}.",
        "",
        "Редкие категории: "
        + "; ".join(f"{c['category']} — сцен: {c['scenes']}" for c in rare)
        + ". Полная таблица поддержки — dataset_category_coverage.csv. Выбор не стратифицирован; присутствие категории не означает надёжную отдельную оценку её качества.",
        "",
        f"Объём 500 выбран как практический бюджет первого основного протокола внутри разрешённого диапазона 500–1000. Формальная достаточность этого объёма и устойчивость порядка моделей не доказаны. С 500 исходными сценами эксперимента crops совпадают {intersection}; выбор аннотаторов не воспроизводит набор crops. Выводы относятся к этим 500 сценам и этому протоколу.",
        "",
        "BLIP готовили с COCO; карта SmolVLM2 перечисляет sharegpt4v_coco. Точное пересечение выбранных image_id с обучением не установлено; для Qwen/Florence полная независимость тоже не подтверждена. Это сравнение готовых моделей на COCO, без проверенной внешней оценки.",
        "",
        *checkpoints,
        "",
        "## Протокол",
        "",
        "Исходное RGB-изображение передавалось без overlay, меток детектора и вырезок классификатора. Batch 1, FP16, без offload/quantization, greedy decoding, максимум 96 новых токенов, один прогрев на отдельном dev-изображении. Общий inference.lock наследуется потомком; процесс предыдущей модели завершался до следующего.",
        "",
        "BLIP — unconditional captioning, Florence — task token <CAPTION>, Qwen/SmolVLM — одинаковая инструкция кратко описать видимые объекты, действия и обстановку по-английски без догадок. Нативные processors/decoding и EOS checkpoint сохранены; overrides — do_sample=False, num_beams=1, max_new_tokens=96, use_cache=True. BLIP/Florence — eager attention, Qwen/Smol — SDPA. Qwen ограничен 65536–262144 pixels. Разрешения и токенизаторы различаются.",
        "",
        "Florence использует проверенные файлы Microsoft с закреплёнными revision/SHA и local-only trust_remote_code; новая DynamicCache отключена для авторского tuple-cache. Веса и forward не менялись. Остальные модели используют штатные классы Transformers с trust_remote_code=False.",
        "",
        "## Автоматические метрики",
        "",
        "| Модель | CIDEr | BLEU-4 | ROUGE-L | CHAIRs | CHAIRi | Объектная полнота | Средние tokens | Средние слова | Лимит 96 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        text.append(
            f"| {NAMES[r['model']]} | {r['CIDEr']:.4f} | {r['BLEU_4']:.4f} | {r['ROUGE_L']:.4f} | {100 * r['CHAIRs']:.2f}% | {100 * r['CHAIRi']:.2f}% | {100 * r['object_recall_micro']:.2f}% | {r['output_tokens_mean']:.2f} | {r['words_mean']:.2f} | {r['token_budget_reached']}/500 |"
        )
    text += [
        "",
        "CIDEr — штатный scorer pycocoevalcap 1.2: IDF рассчитывается на выбранных 500 сценах, используется Gaussian length penalty (sigma=6). Длинные captions дополнительно штрафуются за расхождение с длиной эталонов; метрика не изолирует семантическое качество. Это подтверждено [исходным кодом evaluator](https://github.com/tylin/coco-caption/blob/3a9afb2682141a03e1cdc02b0df6770d2c884f6f/pycocoevalcap/cider/cider_scorer.py). CIDEr сохранён в исходной шкале evaluator; для шкалы ×100 умножить на 100. BLEU-4/ROUGE-L/CIDEr оценивают совпадение с эталонным текстом, а не всю фактическую корректность. CHAIRs — доля описаний с неподтверждённым объектом из словаря COCO; CHAIRi — доля таких упоминаний. GT объединяет instances и объекты эталонных captions. Словарь, singularization и правила пар слов закреплены по реализации авторов.",
        "",
        "Объектная полнота — дополнительная micro-доля названных уникальных категорий GT в каждом изображении, с объединением instances/captions. Это не экспертная полнота. CHAIR не проверяет действия, свойства и объекты вне словаря; например, исходный словарь не распознаёт каждое описание профессии/действия как person. Низкий CHAIR может сопровождаться краткими и неполными captions.",
        "",
        "Колонка «Лимит 96» считает ответы ровно с 96 новыми токенами, включая специальные terminal/forced токены; она не доказывает, что каждый такой ответ незавершён. Слова посчитаны простым split по пробелам. Длинные ответы сохранены целиком, без обрезки предложений, ручной правки и повторной генерации.",
        "",
        f"На этой выборке максимальный CIDEr у {NAMES[lexical_best['model']]} ({lexical_best['CIDEr']:.4f}). Минимальная доля неподтверждённых распознанных упоминаний CHAIRi у {NAMES[hallucination_best['model']]} ({100 * hallucination_best['CHAIRi']:.2f}%). Это отдельные критерии; общий произвольный балл не строится, экспертный победитель не определяется.",
        "",
        "Florence-2 base-ft лидирует в измеренном протоколе по CIDEr, BLEU-4 и ROUGE-L и имеет минимальные CHAIRs/CHAIRi. Это обосновывает его как основной кандидат для коротких английских описаний на проверенном ноутбуке. Фактическую корректность действий и свойств ещё должны оценить люди; без этих оценок окончательный выбор для приложения не подтверждён.",
        "",
        f"Qwen имеет максимальную измеренную объектную полноту ({100 * next(r['object_recall_micro'] for r in rows if r['model'] == 'qwen3_vl_2b'):.2f}%), но создаёт значительно более длинные ответы. Низкий lexical CIDEr Qwen/Smol нельзя трактовать как нулевую способность описывать изображение: длина ответов и длина эталонов сильно различаются. Их превосходство или отставание по общему смысловому качеству этим сравнением не доказано.",
        "",
        "## Задержка и память",
        "",
        "| Модель | p50, с | p95, с | Preprocess mean, мс | Generate mean, мс | Postprocess mean, мс | tokens/s mean | Load, с | RSS peak, MiB | CUDA allocated peak, MiB | CUDA reserved peak, MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        text.append(
            f"| {NAMES[r['model']]} | {r['total_ms_p50'] / 1000:.3f} | {r['total_ms_p95'] / 1000:.3f} | {r['preprocess_ms_mean']:.2f} | {r['generate_ms_mean']:.2f} | {r['postprocess_ms_mean']:.2f} | {r['tokens_per_second_mean']:.2f} | {r['cold_load_ms'] / 1000:.2f} | {r['rss_peak_bytes'] / 2**20:.0f} | {r['vram_allocated_peak_bytes'] / 2**20:.0f} | {r['vram_reserved_peak_bytes'] / 2**20:.0f} |"
        )
    text += [
        "",
        f"Самый низкий p50 у {NAMES[fastest['model']]} ({fastest['total_ms_p50'] / 1000:.3f} с). Задержка включает синхронизированные preprocess/generate/postprocess; чтение/декодирование исходного файла сохранено отдельно, а CSV дополнительно содержит total_with_read_ms. Генерация авторегрессионная: длина ответа заметно влияет на время. Одинаковый предел 96 токенов не обеспечивает одинаковую длину или равное число слов.",
        "",
        "Tokens/s — среднее per-image число новых токенов / время generate. Учитываются terminal/forced специальные токены, начальный decoder token и входной prompt исключены. Токенизаторы различаются, поэтому это не одинаковые текстовые единицы. RSS/CUDA peaks включают загрузку и прогрев; allocated/reserved не равны полной занятости GPU с графикой и CUDA-контекстом.",
        "",
        "Cold load — загрузка Adapter в новом процессе после SHA-верификации и записи среды; CUDA-контекст уже инициализирован, файловый кэш ОС не сбрасывался. Измерения выполнены на RTX 4050 Laptop 6 GB, при лимите контейнера 7 GiB и WSL около 7,6 GiB. Полный test завершён без failed/OOM у всех четырёх моделей. Предварительные сбои совместимости smoke сохранены отдельно и не включены в test-качество.",
        "",
        "Один проход не доказывает межпрогонную устойчивость задержек или порядка моделей. Различаются native preprocessing, длина ответа и attention kernels. Ресурсная пригодность подтверждена для фактически измеренного режима на этих COCO-изображениях.",
        "",
        "## Незавершённые оценки и использование результата",
        "",
        "Подготовлена слепая перемешанная форма: 100 общих изображений × четыре модели, 400 строк. Эксперты должны оценить корректность, полноту, связность и полезность для поиска по 1–5 и отдельно отметить выдуманные объекты/действия/свойства. Баллы не заполнены; желательно два оценщика, при одном ограничение указывается. LLM-судья не использовался. Формы с исходными captions и закрытый ключ локальны и исключены из Git.",
        "",
        "Внешние собственные/разрешённые изображения с независимыми эталонами не предоставлены. SPICE evaluator не валидирован и метрика не вычислялась. Русская аннотация/перевод и реальная результативность поиска не измерялись. Выбор модели для общего качества требует человеческой фактической оценки и внешней проверки; текущие измерения показывают отдельные соотношения lexical quality, объектов, задержки и памяти.",
        "",
        "## Воспроизведение и проверки",
        "",
        "Числа собраны из сохранённых captions/timings/errors единственного test500-v1. inference: scripts/annotation.sh benchmark --output reports/annotation/test500-v1; отчёт: scripts/annotation.sh report --experiment reports/annotation/test500-v1 --output reports/comparisons/annotation-test500-v1. Новый полный повтор не запускать без новой задачи. Отчёт перестраивается без модели и без CI.",
        "",
        "54 CPU-теста (включая 9 тестов аннотаторов), Ruff check/format и отдельные четыре реальные офлайн GPU-smoke прошли. Проверяющий scripts/verify_annotation.py проверяет image/model/source hashes, одинаковые 500 image_id, отсутствие dev-overlap, errors, decomposed timings и точную decode parity raw_caption/token IDs. Source snapshot, config, container ID и environment каждого участника сохранены.",
        "",
        "Официальные checkpoint: [BLIP](https://huggingface.co/Salesforce/blip-image-captioning-base), [Florence](https://huggingface.co/microsoft/Florence-2-base-ft), [Qwen](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct), [SmolVLM2](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct). Методика [COCO caption](https://github.com/tylin/coco-caption) и [CHAIR](https://github.com/LisaAnne/Hallucination). Конкретные revisions/SHA и лицензии — annotation/configs/models.json, annotation/models/manifest.json и annotation/evaluation/sources.json.",
    ]
    content = "\n".join(text) + "\n"
    document.write_text(content, encoding="utf-8")
    (output / "analysis.md").write_text(content, encoding="utf-8")
    write_json(
        output / "analysis_manifest.json",
        {
            "script_sha256": sha256(Path(__file__)),
            "report_manifest_sha256": sha256(output / "report_manifest.json"),
            "verification_sha256": sha256(verification),
            "dataset_sha256": sha256(dataset_path),
            "instances_sha256": sha256(gt_path),
            "model_registry_sha256": sha256(registry_path),
            "crop_scenes_manifest_sha256": sha256(crop_path),
            "crop_overlap_scenes": intersection,
            "gt_instances": sum(counts.values()),
            "categories_present": sum(bool(c["scenes"]) for c in coverage),
            "document_sha256": sha256(document),
            "analysis_sha256": sha256(output / "analysis.md"),
            "category_coverage_sha256": sha256(output / "dataset_category_coverage.csv"),
            "report_outputs": verified["report_outputs"],
            "confidence_intervals": "not calculated",
            "human_evaluation": "pending",
            "external_dataset": "not provided",
        },
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--document", type=Path, required=True)
    p.add_argument("--verification", type=Path, required=True)
    args = p.parse_args()
    build(args.experiment, args.output, args.document, args.verification)


if __name__ == "__main__":
    main()
