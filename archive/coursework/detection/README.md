# Детекция: четыре модели и один общий файл

Команды выполняются из `/home/mikhail/kurs4` в Linux Docker Compose.
`compose.gpu.yaml` предоставляет GPU текущей WSL через `/dev/dxg`.

## Подготовка и запуск

Данные COCO и веса уже подготовлены. Проверка или явная загрузка выполняется
`scripts/prepare.py`, как описано в корневом README. Модельные скрипты работают
офлайн; данные и веса подключаются read-only. Сохранены закреплённые версии
в `detection/requirements.lock.txt` и исходные SHA-256 в `detection/configs/models.json`.

Ниже команды для будущего использования. По последнему указанию пользователя
от 11.10.2026 в рамках текущей переделки новые прогоны, пересчёт метрик и
заполнение таблиц остановлены; существующие артефакты сохранены.

Короткая реальная проверка четырёх моделей, строго последовательно:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml run --rm detection scripts/detection/yolo26s.py --run my-check --limit 2
docker compose -f compose.yaml -f compose.gpu.yaml run --rm detection scripts/detection/rtdetr_v2_r18.py --run my-check --limit 2
docker compose -f compose.yaml -f compose.gpu.yaml run --rm detection scripts/detection/rf_detr_small.py --run my-check --limit 2
docker compose -f compose.yaml -f compose.gpu.yaml run --rm detection scripts/detection/faster_rcnn.py --run my-check --limit 2
```

Без `--limit` обрабатывается весь закреплённый COCO test: 4500 изображений,
один проход каждой модели. Полные проверки ради переделки кода не запускать.
CPU разрешён через `-e KURS4_DEVICE=cpu`; его результаты сохранять отдельным именем.
Все участники одного сравнения используют одинаковые `--run`, лимит и устройство.
Уже существующий результат модели не перезаписывается. После прерывания требуется
новое имя; продолжения посреди набора нет.

Восстановление таблиц из новых сохранённых результатов без загрузки моделей:

```bash
docker compose run --rm detection scripts/detection/yolo26s.py --run my-check --report-only
```

Для `--report-only` годится любой из четырёх файлов; лимит берётся из сохранённого
сравнения. Веса и исходные изображения не читаются; для пересчёта метрик нужны
сохранённые JSONL и закреплённая COCO-разметка. Старые форматы test-v1/test-v2
этой командой не переписываются.

## Назначение файлов

- `scripts/detection/yolo26s.py`: Ultralytics YOLO26s, штатный letterbox/rect 640,
  явные стадии DetectionPredictor и независимый native predict.
- `scripts/detection/rtdetr_v2_r18.py`: Transformers RT-DETRv2 R18, локальные
  RTDetrImageProcessor и checkpoint, resize 640×640 и обратное масштабирование рамок.
- `scripts/detection/rf_detr_small.py`: Roboflow RF-DETR Small, resize 512×512,
  ImageNet normalization, исключение пустых sparse COCO slots.
- `scripts/detection/faster_rcnn.py`: torchvision Faster R-CNN ResNet50 FPN V2
  COCO_V1, short side 800/max 1333, transform/backbone/RPN/ROI/native NMS.
- `scripts/detection/common.py`: чтение и проверка набора, label mapping и xywh,
  замеры стадий, официальный COCO evaluator, сохранение и таблицы; родитель
  запускает выбранный файл в новом процессе под общим lock.
- `detection/configs/`: категории, закреплённые источники и исторический протокол;
  `detection/models/`: локальные веса, лицензии и проверенный manifest.
- `detection/tests/test_protocol.py`: быстрые CPU/mock-проверки без весов.
- `detection/Dockerfile`, `requirements.in`, `requirements.lock.txt`: отдельная
  закреплённая среда детекции. `detection/__init__.py` сохраняет пакет тестов.

## Результаты и измерения

`reports/runs/detection/NAME/comparison.json` фиксирует общие условия.
`MODEL_ID/run.json` содержит статус, `processed`, `expected_images`, параметры,
происхождение весов/набора/исходников, окружение, cold load, peak ресурсы,
результат native parity, метрики и времена записи. `MODEL_ID/samples.jsonl`
содержит рамки и тайминги каждого изображения. `process.log` сохраняет вывод
модельного процесса; `original.png` и `overlay.png` — первое изображение.
Native parity дополнительно проверяется на первом неквадратном изображении набора
(`run.json/native_parity_non_square`).

`comparison.csv` и `comparison.md` всегда содержат четыре строки, включая
`not_run`, `failed`, `oom` и `incomplete`; `per_class.csv` — показатели по категориям.
После каждой команды компактная таблица четырёх моделей печатается в консоль.
Сохранённую CSV можно вывести отдельно: `python3 scripts/export_tables.py --table reports/runs/detection/NAME/comparison.csv`.
Неполные данные не получают итоговое качество. Сокращённый набор явно помечается.
Исходные результаты `reports/detection/`, CI и итоговые научные таблицы сохранены.

AP/AR — стандартный pycocotools COCO evaluator, maxDets `[1, 10, 100]`, исходный
score floor 0.001 и до 300 рамок. Precision/recall — score 0.5, IoU 0.5, 100 рамок;
crowd/ignore обрабатывает evaluator. По категориям сохраняются AP и TP/FP/FN.
Все вычисления без новых доверительных интервалов.

Batch 1, FP32, seed 42, 20 прогревов. CUDA синхронизируется между native стадиями.
Сумма preprocessing/forward/postprocessing определяет p50/p95 и
`images_per_second = 1000 / mean(total_ms)`; чтение/декодирование, запись с fsync
и холодная загрузка измеряются отдельно. Faster R-CNN включает RPN/ROI NMS в
forward. Времена записи по image_id находятся в `run.json/write_timings`.
Холодная загрузка означает новый процесс, а не очищенный дисковый кэш.
RAM — peak RSS процесса; VRAM — peak PyTorch allocated/reserved, в таблицах GiB.
Эти значения не измеряют всю память GPU/драйвера. Разные штатные transforms,
предобучение и отсутствие внешнего набора ограничивают выводы о готовых моделях.

Общий `storage/locks/inference.lock` наследуется реальным модельным ребёнком;
блокировка сохраняется даже при аварийном исчезновении родителя. По SIGTERM
родитель завершает группу ребёнка и ждёт выхода, при необходимости применяет
SIGKILL. CPU-оценка начинается после выхода модельного процесса.

## Быстрые проверки

```bash
docker compose run --rm detection -m pytest detection/tests
docker compose run --rm detection -m ruff check scripts/detection detection/tests
docker compose run --rm detection -m ruff format --check scripts/detection detection/tests
```

CPU/mock-проверки не запускают реальные нейросети. Реальные короткие GPU-прогоны
и native parity отмечаются отдельно от тестов.
