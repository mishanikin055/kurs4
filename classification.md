# Сравнительный анализ классификаторов

## Участники и ограничения

**ResNet-50 IMAGENET1K_V2, EfficientNetV2-S IMAGENET1K_V1, ConvNeXt-Tiny IMAGENET1K_V1, ViT-B/16 IMAGENET1K_V1**. Готовые головы ImageNet-1K, одинаковые 1000 классов и изображения, без дообучения. Zero-shot исключён: не добавлять CLIP/SigLIP2 в таблицы, отдельное сравнение или конвейер без новой задачи. Набор выбран пользователем как представители архитектур, не доказанный рейтинг популярности.

## Состояние на 09.10.2026

- Реализован `classification/`, четыре checkpoint загружены в `classification/models/`; реальные офлайн GPU-smoke и native parity пройдены.
- ImageNet validation 50 000 загружен и проверен: по 50 изображений на класс, 14 HF-шардов. Источник ILSVRC/imagenet-1k, revision `49e2ee26f3810fb5a7536bbf732a7b07389a47b5`.
- Предварительный evaluation5000 завершён: 4 × 3 × 5000, по 5 на класс, seed 42. Отчёт — `reports/comparisons/classification-evaluation5000-v1/`.
- Полный validation50k завершён **только для ResNet-50**: три повтора по 50 000, Top-1 80.854%, Top-5 95.438%. Отчёт — `reports/comparisons/classification-validation50000-v1/`; проверка — `reports/verification/classification_validation50000.json`.
- По новой задаче пользователя от 09.10.2026 начато завершение остальных моделей на полном validation и полных GT/detector-crops. Папки: `reports/classification/{validation50000-v1,gt-crops-v1,detector-crops-v1}/`; последовательные команды и логи — `reports/classification/completion-v1/`. Полный четырёхмодельный победитель ещё не определён; статус проверять по run.json.
- **Новые доверительные интервалы не считать и в основной отчёт не включать во всех трёх сравнениях проекта.** Ранее выполненный bootstrap сохраняется в исходных артефактах. Поле bootstrap_samples в старом конфиге не отменяет это указание.
- GT/detector-crops реализованы в `classification/crops/`, полные сравнения обоих режимов завершены: 4 × 3 × 1883 GT-crops и 4 × 3 × 2296 detector-crops. [GT-анализ](reports/comparisons/classification-gt-crops-v1/analysis.md), [detector-анализ](reports/comparisons/classification-detector-crops-v1/analysis.md); проверки — `reports/verification/classification-{gt-crops,detector-crops}-v1.json`. Предсказания совпадают между повторами. Законченного анализа всех трёх режимов пока нет: оставшийся ImageNet50k выполняется. Приложение не реализовано.

## Что читать и где работать

- [classification/README.md](classification/README.md) — существующие команды whole-image benchmark.
- [classification/crops/README.md](classification/crops/README.md) — отдельные прикладные режимы.
- `classification/{adapters,runner,datasets,metrics,report,download,cli,common}.py`, `classification/tests/` и `classification/crops/tests/` — код/проверки.
- `classification/configs/`, `classification/models/manifest.json` — точные веса, native transforms, labels/synsets, конфиги и проверка загрузки.
- `data/imagenet/{evaluation5000,validation}.json`, `reports/classification/crop-data-v1/` — manifests.
- `scripts/classification.sh`, `scripts/classification_crops.sh`, `compose.classification*.yaml` — контейнерный запуск.
- [docs/classification_handoff.md](docs/classification_handoff.md) — подробное происхождение результатов и сведения о продолжении, если они нужны задаче.
- [experiments.md](experiments.md) — общие требования к новым экспериментам.

## Раздельные режимы оценки

1. **Целое изображение ImageNet**: Top-1, Top-5, Macro-F1, accuracy по классам, sparse confusion matrix. Отменённое разделение 5000 dev / 45 000 test не восстанавливать. evaluation5000 входит в validation50k; это не независимые выборки.
2. **GT-crops**: условная точность по категориям с однозначным mapping ImageNet synset → COCO, покрытие словаря и отказы.
3. **Detector-crops**: matching по IoU, ошибки локализации отдельно от классификации, исправленные и внесённые ошибки относительно detector-only. Источник рамок закреплён: RF-DETR Small repeat1, 500 COCO test-сцен.

Не смешивать метрики трёх режимов. Текущий mapping: 287 synsets → 56 COCO-категорий, остальные `not_mappable`; покрытие non-crowd GT 55.81%. COCO не подтверждает породы и тонкие подтипы ImageNet. Пороги score 0.5 / margin 0.1 диагностические, отказ на dev не калиброван.

Сохранять метку детектора и отдельные top-k, synset, mapping, согласие/конфликт/отказ. Высокий softmax не доказывает принадлежность неизвестного объекта словарю. Не заменять метку автоматически. В pipeline классифицировать все вырезки в одном процессе этапа — [pipeline.md](pipeline.md).

## Протокол и проверки

FP32, batch 1, 20 прогревов, три повтора, 4 CPU threads; штатный preprocess каждого checkpoint. Строго одна модель, общий `storage/locks/inference.lock`, runtime без сети и токена. Качество по одному повтору; три повтора времени не превращают 50 000 изображений в 150 000 независимых примеров.

Для нового эксперимента новая папка. Старый resume требует сохранённого source snapshot и совпадающей identity; завершённый отчёт не требует нового инференса. Проверять порядок всех 1000 меток, SHA-256, top-k, crops/mapping и report через существующие CPU-тесты; реальные smoke отмечать отдельно. Команды — в README выбранного режима.
