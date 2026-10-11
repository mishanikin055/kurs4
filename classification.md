# Сравнительный анализ классификаторов

## Участники и ограничения

**ResNet-50 IMAGENET1K_V2, EfficientNetV2-S IMAGENET1K_V1, ConvNeXt-Tiny IMAGENET1K_V1, ViT-B/16 IMAGENET1K_V1**. Готовые головы ImageNet-1K, одинаковые 1000 классов и изображения, без дообучения. Zero-shot исключён: не добавлять CLIP/SigLIP2 в таблицы, отдельное сравнение или конвейер без новой задачи. Набор выбран пользователем как представители архитектур, не доказанный рейтинг популярности.

## Сохранённые результаты на 09.10.2026

- Четыре checkpoint загружены в `classification/models/`; реальные офлайн GPU-smoke и native parity пройдены.
- ImageNet validation 50 000 загружен и проверен: по 50 изображений на класс, 14 HF-шардов. Источник ILSVRC/imagenet-1k, revision `49e2ee26f3810fb5a7536bbf732a7b07389a47b5`.
- Предварительный evaluation5000 завершён: 4 × 3 × 5000, по 5 на класс, seed 42. Отчёт — `reports/comparisons/classification-evaluation5000-v1/`.
- Полный validation50k завершён для всех четырёх моделей: первые два полных повтора по 50 000, 400 000 обработок. Качество берётся из repeat1; время и ресурсы — из repeat1/2. Ранее выполненные три ResNet-50 использованы повторно; третий сохранён и исключён из основной таблицы. [Таблица](reports/comparisons/classification-validation50000-v1/classification.md), [анализ](reports/comparisons/classification-validation50000-v1/analysis.md), [проверка](reports/verification/classification-validation50000-v1.json).
- ImageNet Top-1: ResNet-50 80.854%, EfficientNetV2-S 84.238%, ConvNeXt-Tiny 82.514%, ViT-B/16 81.068%. EfficientNetV2-S лучший по качеству этого набора, ResNet-50 самый быстрый; ConvNeXt-Tiny даёт промежуточное соотношение качества и времени. Это сравнение готовых checkpoint с различными рецептами обучения и native transforms, а не изолированный эффект архитектуры.
- Папки сырых результатов: `reports/classification/{validation50000-v1,gt-crops-v1,detector-crops-v1}/`; последовательные команды и логи — `reports/classification/completion-v1/`. Контроллер закончил второй ViT-B/16 и остановил планировщик до нового третьего круга; активного benchmark нет. Исходные experiment config/snapshots с repeats=3 сохранены, выбор двух повторов фиксируется отдельно в report/verification manifests.
- **Новые доверительные интервалы не считать и в основной отчёт не включать во всех трёх сравнениях проекта.** Ранее выполненный bootstrap сохраняется в исходных артефактах. Поле bootstrap_samples в старом конфиге не отменяет это указание.
- Исторические GT/detector-crops завершены: 4 × 3 × 1883 GT-crops и 4 × 3 × 2296 detector-crops. [GT-анализ](reports/comparisons/classification-gt-crops-v1/analysis.md), [detector-анализ](reports/comparisons/classification-detector-crops-v1/analysis.md); проверки — `reports/verification/classification-{gt-crops,detector-crops}-v1.json`. Предсказания совпадают между повторами. Итог всех трёх режимов — [Описание результатов классификации](Описание%20результатов%20классификации.md). Приложение не реализовано.

CPU-проверки после завершения: 32 pytest-теста классификации/crops и Ruff check/format в Docker. Реальный GPU-smoke четырёх моделей и все выбранные полные проходы проверены отдельно. Лучший по Top-1/Top-5 на crops — ViT-B/16; на detector-crops лучший Macro-F1 у EfficientNetV2-S. Гипотетическая замена метки детектора ухудшает matched accuracy у всех четырёх, поэтому метка детектора сохраняется. Dev-калибровка отказа и проверка пород/подтипов остаются вне завершённого сравнения.

## Новые запуски после упрощения 11.10.2026

Приоритет имеет [REFACTORING_TASK.md](REFACTORING_TASK.md). Активный код —
ровно `scripts/classification/{resnet50,efficientnet_v2_s,convnext_tiny,vit_b_16,common}.py`.
В модельных файлах явно указаны torchvision constructor, точный weight enum,
локальный state dict, native transforms и Top-5. Общий файл отвечает за проверки,
замеры, один дочерний процесс, метрики и CSV/Markdown. Зависимостей от
`detection.*` нет. Тесты — `classification/tests/`; подготовка — `scripts/prepare.py`.
Старые CLI, runner, shell-обёртки и код новых crop-запусков удалены.

Новые сравнения используют **только весь ImageNet validation**,
`data/imagenet/validation.json`, 50 000 изображений (50 на класс).
FP32, batch 1, 20 прогревов, четыре CPU threads, CUDA, **один проход** каждой модели.
Допустим положительный `--limit` для короткой проверки; все четыре участника
под одним именем должны иметь одинаковые параметры, набор, код и образ.
Исходные configs с repeats=3 и bootstrap не изменяются задним числом:
актуальный протокол записан отдельно в новом `comparison.json`.

Команды — в [classification/README.md](classification/README.md):
`docker compose -f compose.yaml -f compose.gpu.yaml run --rm classification scripts/classification/resnet50.py --run NAME --limit 2`.
Остальные три файла запускаются последовательно с тем же `--run NAME`.
Runtime офлайн, без токена, данные и веса read-only. Новые полные прогоны
ради рефакторинга не запускать. Повторная запись результата модели запрещена;
после прерывания требуется новое имя. Resume и новые CI не реализуются.

Результаты — `reports/runs/classification/NAME/`: по модели только
`samples.jsonl` и `run.json`, рядом `comparison.json`, `comparison.csv/.md`,
`per_class.csv`. В таблице всегда четыре строки с явными `not_run`, `failed`,
`oom`, `incomplete`; незавершённое качество пустое. Короткий набор помечается
`shortened=True`. `--report-only` перестраивает таблицы без модели и инференса.
Предсказания проверяются по target index/synset, class mapping, Top-5,
вероятностям, числу строк и hashes. Native parity первого изображения
автоматически сохраняется в `run.json`.

Top-1, Top-5, Macro-F1 и показатели всех 1000 классов рассчитываются из JSONL.
Macro-F1 включает классы без support; метрики короткой проверки не характеризуют
полный validation. p50/p95/скорость учитывают preprocessing/transfer, forward,
postprocessing; чтение и запись хранятся отдельно. Холодная загрузка — новый
процесс без очистки page cache. RAM — peak RSS; VRAM — PyTorch
allocated/reserved с загрузкой модели, не вся память GPU; в таблице GiB.
Разные native inputs и рецепты обучения остаются ограничениями сравнения
готовых checkpoint, а не изолированного эффекта архитектуры.

Общий `storage/locks/inference.lock` удерживается родителем и наследуется
реальным ребёнком до выхода. Приватный worker проверяет fd/inode и наличие
lock; SIGTERM/SIGINT родителя завершают process group ребёнка с ожиданием.
При внезапной гибели родителя живой ребёнок сохраняет lock. CUDA OOM и
cgroup OOM kill фиксируются отдельно. CPU-тесты и offline GPU-smoke — разные
проверки, ни одна не заменяет другую.

## Исторические отдельные режимы и ограничения

Все прежние `reports/`, configs, manifests и source snapshots сохраняются.
ImageNet evaluation5000 пересекается с validation50k; это не независимые выборки.
Уже завершённый полный ImageNet использует одинаковые первые два повтора
всех четырёх участников (400 000 обработок); третий ResNet-50 сохранён отдельно.
evaluation5000 и оба crop-сравнения сохраняют три завершённых повтора.
Старый resume и прежние команды относятся к сохранённому коду, не к новой структуре.

GT/detector-crops больше не запускаются новой реализацией;
[сведения и ссылки на результаты](classification/crops/README.md).
Mapping 287 synsets → 56 COCO-категорий покрывает 55.81% non-crowd GT.
Top-5 после mapping не означает пять разных COCO-категорий. COCO не содержит
эталонов пород/подтипов. Пороги score 0.5 / margin 0.1 были диагностическими,
отказ не калиброван на dev; гипотетическая замена категории ухудшает matched
accuracy всех моделей. Исходные метки детектора сохранены.

Будущий конвейер остаётся отдельной задачей: классифицировать все вырезки
одним процессом этапа, сохранять detector label и отдельные top-k/synset,
не заменять категорию автоматически. Высокий softmax не доказывает, что
неизвестный объект принадлежит словарю. См. [pipeline.md](pipeline.md).

## Общие итоги сравнений

[Финальные таблицы и причины размера выборок](Итоги%20сравнительного%20анализа%20моделей.md),
[понятное описание выполненных шагов](Как%20выполнялся%20сравнительный%20анализ.md).
Все три сравнения завершены в согласованном объёме; человеческая оценка
аннотаторов исключена пользователем. Приложение остаётся отдельной задачей.
