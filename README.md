# Курсовой проект: детекция, классификация и аннотирование

Подготовлен первый экспериментальный модуль — сравнение четырёх детекторов из раздела 3.2 плана. Также подготовлен whole-image модуль классификации ImageNet-1K: [classification/README.md](classification/README.md). Все четыре классификатора прошли реальный офлайн GPU-smoke; основные ImageNet-прогоны ожидают датасет. Прикладные crops, приложение и аннотаторы пока не реализованы.

Инструкция с командами: [detection/README.md](detection/README.md). Обоснование участников: [docs/model_selection.md](docs/model_selection.md). Решения по окружению и протоколу: [docs/decisions.md](docs/decisions.md).

Все веса находятся в `detection/models/`, данные — в `data/coco/`, результаты — в `reports/`. Веса и датасет исключены из Git. Необходимы Linux/WSL2, Docker и Docker Compose. Для GPU требуется NVIDIA Container Toolkit либо подготовленный профиль WSL2.

Полное сравнение детекторов завершено: COCO test 4500 изображений, три повтора, 1000 парных bootstrap-выборок. [Таблица](reports/comparisons/detection.md), [анализ](reports/comparisons/detection_analysis.md), [описание результатов](Описание%20результатов%20детекции.md). Исходные сбои test-v1 сохранены; восстановление и происхождение отражены в test-v2/recovery.json.
