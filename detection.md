# Сравнительный анализ детекторов

## Назначение и состояние

Сравнить четыре готовых COCO-детектора: **YOLO26s, RT-DETRv2 R18, RF-DETR Small, Faster R-CNN ResNet50 FPN V2 COCO_V1**. Это зафиксированные архитектурные представители, не доказанный мировой топ популярности. Обучения нет.

На 09.10.2026 модуль `detection/` реализован, веса и COCO загружены, реальные офлайн smoke пройдены. Полный восстановленный **test-v2 завершён: 4 × 3 × 4500**; парный bootstrap — 1000 выборок. Девять неизменённых проходов перенесены из test-v1, три прохода RF-DETR выполнены заново после исправления sparse COCO slots. Исходные сбои и происхождение переноса сохранены. Не представлять восстановление как 12 новых прогонов.

## Что читать и где работать

- [detection/README.md](detection/README.md) — реализованные команды и подробный протокол.
- `detection/{adapters,runner,datasets,metrics,report,download,cli,common}.py`, `detection/tests/` — код и проверки.
- `detection/configs/` — закреплённые веса, mapping, benchmark и split; веса/кэши — `detection/models/`, фактический manifest — `detection/models/manifest.json`.
- `scripts/detection.sh`, `compose.detection.yaml`, `compose.detection.wsl.yaml` — запуск.
- `reports/detection/test-v1/`, `test-v2/` — сырые результаты; `reports/comparisons/detection*.csv`, `detection*.md` — итоговые таблицы и анализ.
- [experiments.md](experiments.md) — общие правила при изменении эксперимента; [infrastructure.md](infrastructure.md) — при изменении среды.

## Зафиксированный протокол

COCO val2017: 5000 изображений, seed 42, 500 dev и 4500 test; manifests — `data/coco/{dev,test,validation}.json`. Основной режим: FP32, batch 1, 20 прогревов, три повтора, строго последовательные модельные процессы. Короткий dev-прогон не использовать в итоговых test-выводах.

Метрики: COCO mAP@[0.50:0.95], AP50/AP75, AP по размерам, AR; precision/recall/F1 при score 0.5 и IoU 0.5; ошибки по классам, p50/p95, throughput, RAM/VRAM, сбои. Для AP score floor 0.001, до 300 исходных детекций, evaluator maxDets [1, 10, 100]. UI-порог не обрезает данные для AP.

Различать sparse COCO category_id и непрерывные индексы модели. RF-DETR исключает только неиспользуемые slots своей головы; неизвестные ID вне допустимого диапазона остаются ошибкой. Не перенумеровывать категории произвольно. bbox_xyxy — пиксели исходного изображения, COCO bbox — xywh; проверять letterbox, resize и EXIF.

## Продолжение и проверки

Не запускать повтор завершённого benchmark ради правки текста. Отчёт строить из сохранённых артефактов. Для новых настроек/кода создавать новую версию эксперимента; resume требует совпадения identity и исходников. Детали восстановления — `scripts/recover_detection.py` и [docs/decisions.md](docs/decisions.md).

Релевантные CPU-проверки — `detection/tests/`, Ruff и Compose из README. После изменения адаптера — реальный офлайн smoke с native parity и проверкой координат/mapping. Полную память GPU не приравнивать к PyTorch allocated/reserved. Интеграция выбранного детектора в приложение — отдельная задача по [pipeline.md](pipeline.md).
