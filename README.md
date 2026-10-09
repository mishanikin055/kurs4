# Курсовой проект: детекция, классификация и аннотирование

Реализованы экспериментальные модули детекции и классификации: [detection/README.md](detection/README.md), [classification/README.md](classification/README.md). Все четыре классификатора прошли офлайн GPU-smoke и предварительное сравнение на 5000 ImageNet-изображениях, три повтора. [Таблица](reports/comparisons/classification-evaluation5000-v1/classification.md), [анализ](reports/comparisons/classification-evaluation5000-v1/analysis.md). [Прикладные crops](classification/crops/README.md) оцениваются отдельно. Добавлен модуль сравнения аннотаторов: [annotation/README.md](annotation/README.md);
четыре офлайн GPU-smoke пройдены, основной test500 ещё не выполнен. Приложение пока не реализовано. Новые доверительные интервалы не рассчитываются ни в одном сравнении проекта.

Инструкция с командами: [detection/README.md](detection/README.md). Обоснование участников: [docs/model_selection.md](docs/model_selection.md). Решения по окружению и протоколу: [docs/decisions.md](docs/decisions.md).

Веса детекторов находятся в `detection/models/`, классификаторов — в `classification/models/`; данные — в `data/coco/` и `data/imagenet/`, результаты — в `reports/`. Веса и датасет исключены из Git. Необходимы Linux/WSL2, Docker и Docker Compose. Для GPU требуется NVIDIA Container Toolkit либо подготовленный профиль WSL2.

Полное сравнение детекторов завершено: COCO test 4500 изображений, три повтора, 1000 парных bootstrap-выборок. [Таблица](reports/comparisons/detection.md), [анализ](reports/comparisons/detection_analysis.md), [описание результатов](Описание%20результатов%20детекции.md). Исходные сбои test-v1 сохранены; восстановление и происхождение отражены в test-v2/recovery.json.

Краткое состояние для продолжения в новом чате: [docs/classification_handoff.md](docs/classification_handoff.md).

Сравнительный анализ классификаторов завершён 09.10.2026:
[описание результатов](Описание%20результатов%20классификации.md),
[ImageNet50k](reports/comparisons/classification-validation50000-v1/classification.md),
[GT-crops](reports/comparisons/classification-gt-crops-v1/analysis.md),
[detector-crops](reports/comparisons/classification-detector-crops-v1/analysis.md).
Полный ImageNet — четыре модели × два прохода по 50 000, оба crop-режима сохранили
уже завершённые три повтора: 1883 GT и 2296 detector-crops на 500 COCO-сценах.
Предыдущие полные ResNet-50 использованы повторно; сохранённый третий не включён
в основную ImageNet-таблицу. Новых третьих кругов не запускалось.

EfficientNetV2-S лучший по ImageNet Top-1 (84.238%), ResNet-50 самый быстрый,
ViT-B/16 лучший по условным Top-1/Top-5 на crops. Все диагностические варианты
замены категории ухудшили точность исходного детектора; его метка сохраняется.
Словарь покрывает 55.81% non-crowd GT; dev-калибровка отказа и точность пород
не проверены. Оценки режимов не объединяются в общий балл.
Все выбранные проходы проверены по сырым артефактам; 32 CPU-теста и Ruff прошли.
Аннотаторы сравнивать по **одному полному прогону каждой модели**, без новых
доверительных интервалов. Пользователь поручил этот следующий этап субагенту
с отдельным контекстом после проверки классификации.
