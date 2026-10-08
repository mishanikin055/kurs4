# Сравнение детекторов

Участники: **YOLO26s**, **RT-DETRv2 R18**, **RF-DETR Small**, **Faster R-CNN ResNet50 FPN V2 COCO_V1**. Используются готовые COCO-веса; обучения нет. Это представители четырёх подходов из плана, а не доказанный мировой рейтинг популярности.

Команды выполняются в Bash из корня проекта `/home/mikhail/kurs4`. Скрипт `scripts/detection.sh` запускает Linux-контейнер через Compose и записывает SHA-256 ID фактически используемого образа. В текущей WSL2 без NVIDIA Container Toolkit он автоматически подключает профиль `compose.detection.wsl.yaml`. В обычном Linux применяется GPU reservation. Профиль WSL пробрасывает `/dev/dxg` и read-only библиотеки Windows-драйвера; Docker socket в контейнер не передаётся.

## Подготовка

```bash
cd /home/mikhail/kurs4
scripts/detection.sh build
scripts/detection.sh download --dry-run
scripts/detection.sh download
scripts/detection.sh verify
scripts/detection.sh prepare --download
```

Веса и COCO уже загружены в рамках подготовки этого модуля. Повторная загрузка проверяет SHA-256 и пропускает корректные локальные файлы. Перед скачиванием выводятся размеры весов и свободное место; `.part` позволяет продолжить сетевой обрыв. Неизвестные/неверные хеши не принимаются. Файлы моделей, источники, лицензии и состояние проверки перечислены в `detection/models/manifest.json`; окончательные источники и полные хеши закреплены в `detection/configs/models.json`.

COCO val2017 содержит 5000 изображений. Seed 42 задаёт shuffle отсортированных image_id: первые 500 — dev, следующие 4500 — test. Полные manifests `data/coco/dev.json`, `test.json`, `validation.json` содержат хеши изображений и разметки; версия split фиксируется также в `detection/configs/dataset_split.json`. Команда `prepare` проверяет 80 категорий, число изображений и распределения классов/размеров/объектов. Изменившийся существующий manifest не перезаписывается. Официальный S3 bucket COCO доступен через HTTPS-адрес `s3.amazonaws.com/images.cocodataset.org`; проверка TLS не отключается. ZIP распаковывается только по ожидаемым именам; исходные архивы сохранены для проверки CRC и повторного восстановления.

## Проверка моделей и короткий прогон

```bash
scripts/detection.sh smoke --output-dir reports/detection/my-smoke
scripts/detection.sh benchmark \
  --manifest data/coco/dev.json --limit 20 \
  --output-dir reports/detection/my-dev20
```

Smoke запускает модели строго по одной, на реальном изображении COCO 397133, в FP32 и без сети. Сохраняются исходное изображение, рамки, сырые детекции, RAM/VRAM и сравнение staged-адаптера со штатным predict-путём библиотеки (`native_parity.json`). Smoke сам по себе не оценивает качество датасета. Короткий benchmark выполняет все три повтора, строит таблицу по 20 dev-изображениям и маркирует строки `shortened`. Его нельзя использовать для итоговых выводов или выбора победителя. Новый прогон требует нового output-dir, если изменён код/конфиг/веса.

## Основное сравнение

```bash
scripts/detection.sh benchmark --output-dir reports/detection/test-v1
scripts/detection.sh bootstrap --runs-dir reports/detection/test-v1 --samples 1000
scripts/detection.sh report \
  --runs-dir reports/detection/test-v1 \
  --output-dir reports/comparisons
```

Первая команда по умолчанию использует `data/coco/test.json`: 4500 изображений, batch size 1, FP32, 20 прогревов, три повтора. Во втором повторе порядок участников сдвигается на одну позицию, в третьем — на две. Каждая модель обрабатывает весь набор, её дочерний процесс завершается, затем запускается следующая. Межпроцессный lock `storage/locks/inference.lock` удерживается до завершения дочернего процесса и сохраняется у ребёнка даже при аварийном завершении родителя. Будущий worker обязан использовать этот же lock; перед benchmark его следует остановить.

Вторая команда вычисляет **парный bootstrap по image_id**: одна и та же выборка для всех моделей, COCO dataset-level AP пересчитывается на каждой выборке с корректным дублированием GT/predictions. В `bootstrap.json` сохраняются 95% интервалы mAP, парных различий и сырые значения. 1000 повторов COCO evaluator требуют значительного времени CPU; GPU и веса для них не нужны. Bootstrap запускается отдельно, чтобы таблицу можно было получить сразу после инференса; до расчёта CI отчёт прямо отмечает их отсутствие.

Третья команда повторно строит `reports/comparisons/detection.csv` и `detection.md`, графики качество–задержка/RAM/VRAM из сохранённых файлов, без запуска моделей. После benchmark таблица уже доступна в `reports/detection/test-v1/comparison/`. В таблице ровно четыре строки: отсутствующие, failed/OOM и неполные участники не исчезают. Числа AP представлены в диапазоне 0–1. Для полной стандартной COCO validation-оценки передайте `--manifest data/coco/validation.json`; после настройки на dev это не независимый test.

Полный читаемый отчёт по завершённому test-v2 — `reports/comparisons/detection_full.md`, таблица для Excel — `detection_full.csv`, ошибки по всем 80 классам — `detection_per_class.csv`. Они дополнительно включают сохранённые AR1/AR10/AR по размерам, TP/FP/FN, средние времена отдельных стадий и условия протокола. Генератор использует стандартную библиотеку Python и не повторяет инференс или bootstrap:

```bash
python3 reports/comparisons/build_detection_full_report.py
```

У YOLO замороженный AutoBackend записал число параметров оболочки как ноль. В полном отчёте это исправлено отдельным подсчётом внутренней исполняемой сети: **9 496 140** параметров. Весовые SHA и ID исходного образа проверены; исходные результаты не переписаны. Происхождение — `detection_yolo_metadata.json`. Повторить только уточнение метаданных при необходимости можно в закреплённом CPU-контейнере без сети и без инференса:

```bash
export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"
export DETECTION_IMAGE_ID="$(docker image inspect kurs4-detection:v1 --format '{{.Id}}')"
docker compose -f compose.detection.yaml run --rm --entrypoint python cpu \
  reports/comparisons/build_detection_full_report.py --collect-yolo-metadata
python3 reports/comparisons/build_detection_full_report.py
```

Полное потребление GPU с учётом CUDA-контекста/драйвера не измерено: имеющиеся VRAM allocated/reserved отражают PyTorch. Сложные условия, внешний набор, классификация, captions и конвейер требуют отдельных экспериментов; перестроение таблицы не создаёт эти результаты.

Порог precision/recall/F1 заранее фиксирован на 0.5, IoU 0.5; подбора по test нет. Если понадобится подбор на dev, создайте новую версию конфига до test-прогона. Для AP используется score floor 0.001 и до 300 исходных детекций; COCO evaluator применяет стандартные maxDets `[1, 10, 100]`. UI-порог не влияет на AP. Crowd/ignore и matching обрабатываются pycocotools. Пер-классовые AP, TP/FP/FN находятся в `metrics.json`.

## Возобновление и CPU

```bash
scripts/detection.sh benchmark --output-dir reports/detection/test-v1 --resume
```

Resume допускается только при совпадении кода, image ID контейнера, конфигурации, весов и dataset manifest. Завершённые повторы пропускаются, незавершённые продолжаются по durable per-image записям; недописанная последняя строка удаляется. Качество пересчитывается по всему выбранному набору после завершения. История ошибок и повторных загрузок сохраняется. Не объединяйте каталоги разных экспериментов.

Поддерживается CPU/FP32: скопируйте `detection/configs/benchmark.json`, замените `device` на `cpu` и запустите `docker compose -f compose.detection.yaml run --rm cpu benchmark --config ... --output-dir ...`. Укажите переменную `DETECTION_IMAGE_ID` из `docker image inspect kurs4-detection:v1 --format '{{.Id}}'` для привязки результатов к образу. CPU и GPU выводы храните в отдельных каталогах. Другие dtype, batch size, квантование, offload, экспорт и ускорители не входят в текущий протокол.

## Артефакты и смысл измерений

Для каждого участника/повтора сохраняются `config.json`, `environment.json`, `model_manifest.json`, `dataset_manifest.json`, `native_config.json`, `run.json`, `samples.jsonl`, `predictions.jsonl`, `predictions_coco.json`, `timings.jsonl`, `write_timings.jsonl`, `metrics.json`, `errors.jsonl`, `process.log`, `summary.md`. `samples.jsonl` — источник для resume; отдельные файлы predictions/timings можно пересобрать командой внутренней оценки `_evaluate` после полного инференса. В корне эксперимента лежат `experiment.json` и `source_snapshot.tar.gz`. При отсутствии Git-коммита записываются dirty status и хеши/снимок кода, выдуманный commit не используется.

Тайминги: холодная загрузка отдельно; чтение/декодирование отдельно; preprocessing (включая transfer/resize), forward и postprocessing синхронизируются CUDA на границах; запись с fsync — отдельно. Для Faster R-CNN его штатные RPN/ROI NMS входят в forward, обратное преобразование bbox — в postprocess; для YOLO разделение следует библиотечному predictor. Поэтому стадии объясняются вместе с архитектурой. p50/p95 и throughput относятся к сумме трёх модельных стадий, исключают IO и загрузку. Throughput — `1000 / mean(total_ms)`, batch 1. VRAM показывает peaks PyTorch allocated/reserved; это не вся память Windows/GPU. RAM — peak RSS процесса, включая холодную загрузку. Лимит контейнера — 6 GiB; в текущей WSL доступно 7.6 GiB всего. Кэш ОС влияет на понятие «холодная загрузка»; здесь это новое создание модели в новом процессе, а не очистка дискового кэша.

Preprocessing штатный: YOLO letterbox/rect 640, RT-DETRv2 resize 640×640 с processor checkpoint, RF-DETR Small resize 512×512 с ImageNet normalization, Faster R-CNN short side 800/max side 1333. Разные размеры, предобучение и архитектуры — ограничения сравнения готовых моделей. EXIF с поворотом отклоняется: нельзя менять изображение, сохраняя GT в прежних координатах. Координаты в raw `bbox_xyxy` — пиксели исходного изображения; COCO `bbox` — xywh. Sparse COCO ID и continuous YOLO/RT-DETR индексы явно различаются.

## Быстрые проверки кода

```bash
docker compose -f compose.detection.yaml run --rm checks -m pytest
docker compose -f compose.detection.yaml run --rm checks -m ruff check .
docker compose -f compose.detection.yaml run --rm checks -m ruff format --check .
```

CPU-тесты не скачивают и не загружают модели. Настоящие GPU smoke и короткий COCO benchmark отмечаются отдельно. Скрипты из плана также доступны как `scripts/download_models.py`, `prepare_datasets.py`, `smoke_models.py`, `benchmark_detection.py`, `build_report.py` внутри контейнера; они используют тот же CLI.


## Исправленный test-v2 после ошибки RF-DETR

В полном test-v1 RF-DETR остановился на неиспользуемом COCO ID 66. Исправление отфильтровывает фон и gaps в sparse-голове только этой модели; количество исключений сохранено в timings/CSV. ID вне допустимой головы остаются ошибкой. Пороги, веса и dataset split не менялись. Подробности — docs/decisions.md.

Исходный test-v1 сохранён с тремя ошибками. test-v2 содержит девять перенесённых завершённых проходов и три заново выполненных полных прохода RF-DETR. Ссылки происхождения — recovery.json, run.json/carried_forward и CSV/reused_from. Перенесённые environment manifests сохраняют исходную версию кода; это явно восстановленное сравнение, а не двенадцать новых запусков. scripts/recover_detection.py проверяет совместимость и ограниченность исправления; это средство для данной ошибки, не способ обойти произвольное несовпадение resume.

Продолжение текущего исправленного эксперимента:

```bash
scripts/detection.sh benchmark --output-dir reports/detection/test-v2 --resume
scripts/detection.sh bootstrap --runs-dir reports/detection/test-v2 --samples 1000
scripts/detection.sh report --runs-dir reports/detection/test-v2 --output-dir reports/comparisons
```

Bootstrap повторно использует независимые совпадения COCO по изображениям, затем для каждой выборки заново выполняет официальный dataset-level accumulate. Это эквивалентно дублированию изображений/GT/predictions; проверено регрессионным сравнением с прямым COCO-пересчётом. Файл bootstrap.progress.json сохраняет выполненные выборки. Повтор той же команды продолжит с checkpoint; другой seed, число повторов, данные, predictions или код отклоняются. Bootstrap не загружает нейросетевые веса. Markdown включает интервалы mAP и парных различий; различие качества нельзя объявлять подтверждённым, если его интервал включает ноль.


После полного bootstrap точный CSV парных различий и текстовый анализ можно пересобрать без моделей:

```bash
python3 reports/comparisons/build_detection_analysis.py --runs-dir reports/detection/test-v2 --output-dir reports/comparisons
```

Генератор работает со стандартной библиотекой Python, проверяет полноту 4500 × 3 и 1000 bootstrap-выборок, сохраняет точные малые границы CI. Его исходник и входные SHA-256 фиксируются в detection_analysis_manifest.json. Он расположен среди версионируемых результатов постобработки, чтобы не менять замороженный source fingerprint инференса. Все команды текущего test-v2 завершены успешно.
