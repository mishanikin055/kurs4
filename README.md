# Курсовой проект: детекция, классификация и аннотирование

Подготовлен первый экспериментальный модуль — сравнение четырёх детекторов из раздела 3.2 плана. Также подготовлен whole-image модуль классификации ImageNet-1K: [classification/README.md](classification/README.md). Все четыре классификатора прошли офлайн GPU-smoke и предварительное сравнение на 5000 ImageNet-изображениях, три повтора. [Таблица](reports/comparisons/classification-evaluation5000-v1/classification.md), [анализ](reports/comparisons/classification-evaluation5000-v1/analysis.md). [Прикладные crops](classification/crops/README.md) реализованы и проверены только небольшим smoke. Полный ImageNet50k и сравнительные crop-прогоны отложены по указанию пользователя; доверительные интервалы в основной таблице не используются. Приложение и аннотаторы пока не реализованы.

Инструкция с командами: [detection/README.md](detection/README.md). Обоснование участников: [docs/model_selection.md](docs/model_selection.md). Решения по окружению и протоколу: [docs/decisions.md](docs/decisions.md).

Веса детекторов находятся в `detection/models/`, классификаторов — в `classification/models/`; данные — в `data/coco/` и `data/imagenet/`, результаты — в `reports/`. Веса и датасет исключены из Git. Необходимы Linux/WSL2, Docker и Docker Compose. Для GPU требуется NVIDIA Container Toolkit либо подготовленный профиль WSL2.

Полное сравнение детекторов завершено: COCO test 4500 изображений, три повтора, 1000 парных bootstrap-выборок. [Таблица](reports/comparisons/detection.md), [анализ](reports/comparisons/detection_analysis.md), [описание результатов](Описание%20результатов%20детекции.md). Исходные сбои test-v1 сохранены; восстановление и происхождение отражены в test-v2/recovery.json.

Краткое состояние для продолжения в новом чате: [docs/classification_handoff.md](docs/classification_handoff.md).

Полный ImageNet50k для ResNet-50 завершён: три повтора, Top-1 80.854%, Top-5 95.438%.
[Таблица полного прогона](reports/comparisons/classification-validation50000-v1/classification.md),
[анализ](reports/comparisons/classification-validation50000-v1/analysis.md).
Остальные полные прогоны остаются отложенными. Статус — docs/classification_handoff.md.
