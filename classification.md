# Сравнительный анализ классификаторов

## Участники и ограничения

**ResNet-50 IMAGENET1K_V2, EfficientNetV2-S IMAGENET1K_V1, ConvNeXt-Tiny IMAGENET1K_V1, ViT-B/16 IMAGENET1K_V1**. Готовые головы ImageNet-1K, одинаковые 1000 классов и изображения, без дообучения. Zero-shot исключён: не добавлять CLIP/SigLIP2 в таблицы, отдельное сравнение или конвейер без новой задачи. Набор выбран пользователем как представители архитектур, не доказанный рейтинг популярности.

## Состояние на 09.10.2026

- Реализован `classification/`, четыре checkpoint загружены в `classification/models/`; реальные офлайн GPU-smoke и native parity пройдены.
- ImageNet validation 50 000 загружен и проверен: по 50 изображений на класс, 14 HF-шардов. Источник ILSVRC/imagenet-1k, revision `49e2ee26f3810fb5a7536bbf732a7b07389a47b5`.
- Предварительный evaluation5000 завершён: 4 × 3 × 5000, по 5 на класс, seed 42. Отчёт — `reports/comparisons/classification-evaluation5000-v1/`.
- Полный validation50k завершён для всех четырёх моделей: первые два полных повтора по 50 000, 400 000 обработок. Качество берётся из repeat1; время и ресурсы — из repeat1/2. Ранее выполненные три ResNet-50 использованы повторно; третий сохранён и исключён из основной таблицы. [Таблица](reports/comparisons/classification-validation50000-v1/classification.md), [анализ](reports/comparisons/classification-validation50000-v1/analysis.md), [проверка](reports/verification/classification-validation50000-v1.json).
- ImageNet Top-1: ResNet-50 80.854%, EfficientNetV2-S 84.238%, ConvNeXt-Tiny 82.514%, ViT-B/16 81.068%. EfficientNetV2-S лучший по качеству этого набора, ResNet-50 самый быстрый; ConvNeXt-Tiny даёт промежуточное соотношение качества и времени. Это сравнение готовых checkpoint с различными рецептами обучения и native transforms, а не изолированный эффект архитектуры.
- Папки сырых результатов: `reports/classification/{validation50000-v1,gt-crops-v1,detector-crops-v1}/`; последовательные команды и логи — `reports/classification/completion-v1/`. Контроллер закончил второй ViT-B/16 и остановил планировщик до нового третьего круга; активного benchmark нет. Исходные experiment config/snapshots с repeats=3 сохранены, выбор двух повторов фиксируется отдельно в report/verification manifests.
- **Новые доверительные интервалы не считать и в основной отчёт не включать во всех трёх сравнениях проекта.** Ранее выполненный bootstrap сохраняется в исходных артефактах. Поле bootstrap_samples в старом конфиге не отменяет это указание.
- GT/detector-crops реализованы в `classification/crops/`, полные сравнения обоих режимов завершены: 4 × 3 × 1883 GT-crops и 4 × 3 × 2296 detector-crops. [GT-анализ](reports/comparisons/classification-gt-crops-v1/analysis.md), [detector-анализ](reports/comparisons/classification-detector-crops-v1/analysis.md); проверки — `reports/verification/classification-{gt-crops,detector-crops}-v1.json`. Предсказания совпадают между повторами. Итог всех трёх режимов — [Описание результатов классификации](Описание%20результатов%20классификации.md). Приложение не реализовано.

CPU-проверки после завершения: 32 pytest-теста классификации/crops и Ruff check/format в Docker. Реальный GPU-smoke четырёх моделей и все выбранные полные проходы проверены отдельно. Лучший по Top-1/Top-5 на crops — ViT-B/16; на detector-crops лучший Macro-F1 у EfficientNetV2-S. Гипотетическая замена метки детектора ухудшает matched accuracy у всех четырёх, поэтому метка детектора сохраняется. Dev-калибровка отказа и проверка пород/подтипов остаются вне завершённого сравнения.

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

FP32, batch 1, 20 прогревов, 4 CPU threads; штатный preprocess каждого checkpoint. По новому указанию пользователя от 09.10.2026 полный ImageNet закончить после **второго круга**: качество по первому повтору, время и ресурсы — одинаковые первые два повтора каждой модели. Третий круг остальных моделей не запускать; ранее законченный третий ResNet-50 сохранить, но не включать в основную таблицу. Уже завершённые evaluation5000 и GT/detector-crops сохраняют три повтора. Два круга полного ImageNet — 400 000 обработок, 50 000 разных изображений. Исходный конфиг с repeats=3 и experiment identity не менять задним числом; новый выбор повторов явно фиксировать в report/verification manifests. Строго одна модель, общий `storage/locks/inference.lock`, runtime без сети и токена.

Для нового эксперимента новая папка. Старый resume требует сохранённого source snapshot и совпадающей identity; завершённый отчёт не требует нового инференса. Проверять порядок всех 1000 меток, SHA-256, top-k, crops/mapping и report через существующие CPU-тесты; реальные smoke отмечать отдельно. Команды — в README выбранного режима.

## Общие итоги сравнений

[Финальные таблицы и причины размера выборок](Итоги%20сравнительного%20анализа%20моделей.md),
[понятное описание выполненных шагов](Как%20выполнялся%20сравнительный%20анализ.md).
Все три сравнения завершены в согласованном объёме; человеческая оценка
аннотаторов исключена пользователем. Приложение остаётся отдельной задачей.
