# Выбор детекторов, 08.10.2026

Зафиксирован набор из раздела 3.2 `PROJECT_PLAN.md`: YOLO26s, RT-DETRv2 R18, RF-DETR Small, Faster R-CNN ResNet50 FPN V2 COCO_V1. Цель — сопоставление готовых моделей разных архитектур на COCO при batch size 1 и ограничении RTX 4050 6 ГБ / RAM 16 ГБ. Состав не меняется после test-результатов без нового протокола.

| Участник | Архитектура и назначение в сравнении | Препроцессинг | Ограничения |
|---|---|---|---|
| YOLO26s | Современный быстрый одностадийный представитель Ultralytics | Штатный letterbox/rect, imgsz 640 | Индексы 0–79 преобразуются в sparse COCO ID; интерфейс зависит от фиксированной версии Ultralytics |
| RT-DETRv2 R18 | CNN backbone и transformer decoder; лёгкий гибрид | Processor checkpoint, resize 640×640 | 80 continuous индексов; часть названий — синонимы COCO |
| RF-DETR Small | Transformer backbone DINOv2 и DETR decoder | Resize 512×512; штатная ImageNet normalization | Не сравнивать с другими вариантами RF-DETR; checkpoint содержит sparse COCO голову |
| Faster R-CNN ResNet50 FPN V2 | Двухстадийный CNN-контрольный вариант | Short side 800, max 1333; штатный transform | Более крупное разрешение; RPN/ROI NMS внутри forward; контрольная точка COCO_V1 |

Различия готовых моделей включают штатный resize, объём/состав авторского предобучения и реализацию постобработки. Сравнение не доказывает преимущество одной архитектуры при одинаковом обучении. До локального benchmark победитель не назначается, авторские показатели не переносятся в таблицы локальных измерений.

Для обзора собраны десять кандидатур: YOLO26, YOLO11, YOLOv8, RT-DETR, RT-DETRv2, RF-DETR, Faster R-CNN, RetinaNet, FCOS, SSD. Первичные ссылки и доступные API-счётчики находятся в `reports/model_selection/detection_candidates.json`; повторный сбор — `python3 scripts/collect_model_candidates.py`. Инженерная пригодность выбранной четвёрки проверяется собственными офлайн-smoke, а не предположением о размере VRAM.

**Критерий текущего отбора — представители разных архитектур, предусмотренные планом.** Для буквального рейтинга можно использовать месячные загрузки конкретных официальных HF checkpoints среди моделей с COCO-весами и доступным локальным инференсом; охват такого рейтинга ограничен Hugging Face. В текущем наборе каналы распространения различаются (GitHub assets, HF, GCS, PyTorch), поэтому этот счётчик не сопоставим для всех участников. Пропущенные значения остаются `null`, звёзды Ultralytics относятся к общему репозиторию нескольких семейств, звёзды torchvision не приписываются отдельным моделям. Набор не объявляется четырьмя мировыми лидерами популярности.

Первичные источники: [YOLO26](https://docs.ultralytics.com/models/yolo26/), [RT-DETRv2 R18](https://huggingface.co/PekingU/rtdetr_v2_r18vd), [RF-DETR](https://github.com/roboflow/rf-detr), [torchvision Faster R-CNN V2](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html), [правила счётчиков HF](https://huggingface.co/docs/hub/models-download-stats).

Точные revisions, URL, полные SHA-256, размер файлов и лицензии закреплены в `detection/configs/models.json`. Состояние фактических локальных файлов — в `detection/models/manifest.json`. У YOLO — AGPL-3.0, RT-DETRv2/RF-DETR — Apache-2.0, torchvision — BSD-3-Clause; эти записи относятся к источникам пакетов/артефактов, а не утверждают универсальную лицензию всех исходных обучающих данных.
