# Курсовой проект: детекция, классификация и аннотирование

Реализованы экспериментальные модули детекции и классификации: [detection/README.md](detection/README.md), [classification/README.md](classification/README.md). Все четыре классификатора прошли офлайн GPU-smoke и предварительное сравнение на 5000 ImageNet-изображениях, три повтора. [Таблица](reports/comparisons/classification-evaluation5000-v1/classification.md), [анализ](reports/comparisons/classification-evaluation5000-v1/analysis.md). [Прикладные crops](classification/crops/README.md) оцениваются отдельно. Приложение и аннотаторы пока не реализованы. Новые доверительные интервалы не рассчитываются ни в одном сравнении проекта.

Инструкция с командами: [detection/README.md](detection/README.md). Обоснование участников: [docs/model_selection.md](docs/model_selection.md). Решения по окружению и протоколу: [docs/decisions.md](docs/decisions.md).

Веса детекторов находятся в `detection/models/`, классификаторов — в `classification/models/`; данные — в `data/coco/` и `data/imagenet/`, результаты — в `reports/`. Веса и датасет исключены из Git. Необходимы Linux/WSL2, Docker и Docker Compose. Для GPU требуется NVIDIA Container Toolkit либо подготовленный профиль WSL2.

Полное сравнение детекторов завершено: COCO test 4500 изображений, три повтора, 1000 парных bootstrap-выборок. [Таблица](reports/comparisons/detection.md), [анализ](reports/comparisons/detection_analysis.md), [описание результатов](Описание%20результатов%20детекции.md). Исходные сбои test-v1 сохранены; восстановление и происхождение отражены в test-v2/recovery.json.

Краткое состояние для продолжения в новом чате: [docs/classification_handoff.md](docs/classification_handoff.md).

Полный ImageNet50k для ResNet-50 завершён: три повтора, Top-1 80.854%, Top-5 95.438%.
[Таблица полного прогона](reports/comparisons/classification-validation50000-v1/classification.md),
[анализ](reports/comparisons/classification-validation50000-v1/analysis.md).
09.10.2026 по новой задаче начато завершение классификации: сохранённый полный
ResNet-50 используется повторно, остальные три checkpoint продолжаются с `--resume`.
Полное сравнение [GT-crops](reports/comparisons/classification-gt-crops-v1/analysis.md)
уже завершено: четыре модели × три прохода по 1883 вырезкам на 500 сценах.
Detector-crops и оставшийся validation50k выполняются последовательно; актуальный
статус — [docs/classification_handoff.md](docs/classification_handoff.md) и run.json.
Полный анализ всех трёх режимов ещё не готов. После его проверки пользователь
поручил передать сравнение аннотаторов субагенту с отдельным контекстом.
