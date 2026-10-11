# Детекция, классификация и аннотирование изображений

Курсовой проект: три сравнения четырёх готовых моделей без обучения.
Завершённые научные сравнения сохранены; приложение FastAPI/PostgreSQL/Streamlit,
общий конвейер и поиск ещё не реализованы.

## Папки и служебные файлы

- `scripts/detection/`, `scripts/classification/`, `scripts/annotation/` — по
  четыре скрипта с явной загрузкой конкретной модели и один `common.py`:
  набор, замеры, метрики, сохранение и таблица. Общий модуль запускает этот же
  скрипт в дочернем процессе с наследуемой блокировкой и ждёт его завершения.
- `scripts/prepare.py` — явная загрузка из закреплённых источников, SHA-256,
  подготовка и проверка manifests. Сравнения его автоматически не вызывают.
- `scripts/export_tables.py` — вывод сохранённых сравнительных CSV в консоль;
  только стандартная библиотека Python.
- `data/` — COCO, ImageNet и ресурсы метрик; `*/models/` — локальные веса.
  Данные, веса, кэши, окружения и `.secrets/` игнорируются Git.
- `detection/`, `classification/`, `annotation/` — закреплённые configs,
  Dockerfile, requirements и тесты. `annotation/evaluation/` — проверенный
  сторонний код CHAIR/Pattern, словари, источники и лицензии.
- `compose.yaml` — три отдельных ML-окружения; `compose.gpu.yaml` — GPU текущей
  WSL через `/dev/dxg` и библиотеки драйвера read-only. Docker socket не нужен.
- `reports/runs/<направление>/<run>/` — новые однопроходные результаты;
  остальные `reports/` — сохранённые исторические эксперименты, таблицы и DOCX.
- `tests/` — проверки подготовки; `*/tests/` — метрики и инварианты направлений;
  `pyproject.toml` — pytest/Ruff; `docs/decisions.md` — принятые решения;
  `docs/refactoring-baseline.json` — исходные SHA-256 2046 исторических файлов.
- `AGENTS.md` и тематические `*.md` — инструкции работы;
  `REFACTORING_TASK.md` — согласованное задание переделки. Исходные материалы
  в `Материалы для курсовой/` не перемещаются и автоматически не коммитятся.

## Подготовка

Bash, `/home/mikhail/kurs4`. Используются три проверенных образа;
`docker compose build` собирает их по закреплённым Dockerfile.

```bash
cd /home/mikhail/kurs4
export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"
export DETECTION_IMAGE_ID="$(docker image inspect kurs4-detection:v1 --format '{{.Id}}')"
export CLASSIFICATION_IMAGE_ID="$(docker image inspect kurs4-classification:v1 --format '{{.Id}}')"
export ANNOTATION_IMAGE_ID="$(docker image inspect kurs4-annotation:v1 --format '{{.Id}}')"
docker compose run --rm detection scripts/prepare.py detection --verify-only
docker compose run --rm classification scripts/prepare.py classification --verify-only
docker compose run --rm annotation scripts/prepare.py annotation --verify-only
```

Данные и веса уже подготовлены. Для новой установки загрузка разрешается
явно: `--download`, сеть `COMPARISON_NETWORK=bridge` и запись
`PREPARE_ACCESS=rw`. Без этих переменных runtime офлайн, данные и веса read-only.

```bash
COMPARISON_NETWORK=bridge PREPARE_ACCESS=rw docker compose run --rm detection scripts/prepare.py detection --download
COMPARISON_NETWORK=bridge PREPARE_ACCESS=rw docker compose run --rm classification -v "$PWD/.secrets/hf_token:/workspace/.secrets/hf_token:ro" scripts/prepare.py classification --download
COMPARISON_NETWORK=bridge PREPARE_ACCESS=rw docker compose run --rm annotation scripts/prepare.py annotation --download
```

Для ImageNet пользователь заранее принимает условия HF; токен только в
`.secrets/hf_token`, в вывод не попадает. Для подготовки manifests из локальных
файлов: `PREPARE_ACCESS=rw docker compose run --rm <сервис> scripts/prepare.py <направление>`.
`--dry-run` показывает источники без записи. Существующие отличающиеся manifests
не заменяются. Для аннотаторов сначала должны быть подготовлены COCO и разметка.

## Двенадцать команд запуска

Ниже команды для будущих явно запрошенных запусков: **два изображения**, один проход.
По последнему указанию пользователя в ходе переделки больше не запускать
модели и не пересобирать сравнительные таблицы.
Для полного набора удалите `--limit 2` и выберите новое имя `--run`;
полные сравнения ради рефакторинга повторять не нужно.
Четыре модели направления запускайте последовательно с одинаковыми параметрами.

```bash
docker compose -f compose.yaml -f compose.gpu.yaml run --rm detection scripts/detection/yolo26s.py --run my-check --limit 2
docker compose -f compose.yaml -f compose.gpu.yaml run --rm detection scripts/detection/rtdetr_v2_r18.py --run my-check --limit 2
docker compose -f compose.yaml -f compose.gpu.yaml run --rm detection scripts/detection/rf_detr_small.py --run my-check --limit 2
docker compose -f compose.yaml -f compose.gpu.yaml run --rm detection scripts/detection/faster_rcnn.py --run my-check --limit 2

docker compose -f compose.yaml -f compose.gpu.yaml run --rm classification scripts/classification/resnet50.py --run my-check --limit 2
docker compose -f compose.yaml -f compose.gpu.yaml run --rm classification scripts/classification/efficientnet_v2_s.py --run my-check --limit 2
docker compose -f compose.yaml -f compose.gpu.yaml run --rm classification scripts/classification/convnext_tiny.py --run my-check --limit 2
docker compose -f compose.yaml -f compose.gpu.yaml run --rm classification scripts/classification/vit_b_16.py --run my-check --limit 2

docker compose -f compose.yaml -f compose.gpu.yaml run --rm annotation scripts/annotation/blip_base.py --run my-check --limit 2
docker compose -f compose.yaml -f compose.gpu.yaml run --rm annotation scripts/annotation/florence2_base_ft.py --run my-check --limit 2
docker compose -f compose.yaml -f compose.gpu.yaml run --rm annotation scripts/annotation/qwen3_vl_2b.py --run my-check --limit 2
docker compose -f compose.yaml -f compose.gpu.yaml run --rm annotation scripts/annotation/smolvlm2_500m.py --run my-check --limit 2
```

По умолчанию: детекция — 4500 COCO test/FP32/20 прогревов;
классификация — 50 000 ImageNet validation/FP32/20 прогревов;
аннотирование — COCO test500/FP16/один dev-прогрев/greedy/96 новых токенов.
Batch 1. Неизменный `raw_caption` сохраняется. Новых CI и crop-запусков нет.

Один `--run` фиксирует набор, параметры, лимит и исходники. Чужие параметры
отклоняются; существующий результат модели не перезаписывается. После обрыва
нужно новое имя. Lock `storage/locks/inference.lock` защищает модельный процесс
до его завершения. После SIGKILL модельного потомка старые артефакты сохраняются.

В каталоге сравнения находятся `comparison.csv`, `comparison.md`,
`comparison.json`, результаты моделей `run.json` и `samples.jsonl` с
предсказаниями/измерениями по изображениям, дополнительные таблицы по классам.
Четыре строки включают `not_run`, ошибки/OOM и неполные запуски. `shortened`
отмечает сокращённую проверку; её нельзя выдавать за полное сравнение.
Детекторы сохраняют первое изображение и overlay; детекция и классификация
проверяют native parity на первом изображении.

## Восстановление отчёта и вывод в консоль

Без GPU и загрузки весов; можно выбрать любой файл модели направления:

```bash
docker compose run --rm detection scripts/detection/yolo26s.py --run my-check --report-only
docker compose run --rm classification scripts/classification/resnet50.py --run my-check --report-only
docker compose run --rm annotation scripts/annotation/blip_base.py --run my-check --report-only
```

После запуска модели таблица четырёх участников выводится в консоль и сохраняется
в CSV/Markdown. Для отдельного просмотра сохранённых таблиц без Docker:

```bash
python3 scripts/export_tables.py --table reports/runs/detection/my-check/comparison.csv
python3 scripts/export_tables.py --table reports/runs/classification/my-check/comparison.csv --full
python3 scripts/export_tables.py --table reports/runs/annotation/my-check/comparison.csv
```

`--table` можно повторить, `--full` показывает также все столбцы ресурсов.
Word-экспорт исключён пользователем 11.10.2026. Прежние DOCX и manifests сохранены;
новые документные зависимости и окружение экспорта не нужны.

## Проверки и ограничения

```bash
docker compose run --rm detection -m pytest tests detection/tests -q
docker compose run --rm classification -m pytest classification/tests -q
docker compose run --rm annotation -m pytest annotation/tests -q
docker compose run --rm detection -m ruff check .
docker compose run --rm detection -m ruff format --check .
docker compose -f compose.yaml -f compose.gpu.yaml config --quiet
```

CPU/mock-тесты и реальные GPU-проверки — отдельные результаты. p50/p95
исключают загрузку/прогрев; точный состав скорости указан в таблицах направления.
RAM — RSS модельного процесса, allocated/reserved — память PyTorch,
без всей памяти драйвера/GPU. У разных checkpoint различаются native transforms
и обучение авторов. Внешний набор и человеческая оценка не использованы.
Сохранённые описания и source snapshots относятся к прежней ревизии;
исторические команды внутри них не являются текущими командами запуска.
Исторические выводы — [итоги](Итоги%20сравнительного%20анализа%20моделей.md)
и [описание шагов](Как%20выполнялся%20сравнительный%20анализ.md).
