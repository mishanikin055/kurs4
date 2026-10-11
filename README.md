# Детекция, классификация и аннотирование изображений

Курсовой проект: три сравнения четырёх готовых моделей без обучения.
Завершённые научные сравнения сохранены. Приложение принимает изображения через
FastAPI, сохраняет задания в PostgreSQL и публикует их в RabbitMQ. Реализована
Celery-задача `process_image`: RF-DETR Small → ViT-B/16 всех вырезок → Florence-2
base-ft, отдельный процесс на этап. Модели выбраны пользователем 11.10.2026.
CPU/mock и интеграция инфраструктуры проверяются отдельно от реального GPU-инференса;
smoke выбранной тройки пока не выполнен. Сравнения повторно не запускались.

## План приложения

Пользователь загружает изображение через `POST /images` и получает ID задания.
FastAPI сохраняет файл и запись в БД, publisher отправляет `job_id` в RabbitMQ.
Одна Celery-задача последовательно запускает детектор,
классификатор всех вырезок и аннотатор с сохранением результатов в PostgreSQL. Результаты читаются через
`GET /jobs/{job_id}`, состояние инфраструктуры — через `GET /health`.

Три таблицы: `images`, `jobs`, `model_results`. Файлы изображений — в общем volume
backend/worker. Поля публикации в jobs предотвращают потерю задания при обрыве
между БД и брокером. Повторная публикация возможна; защита обработки от дублей
реализована в `process_image`. Рамки привязаны
к сохранённому изображению для будущего построения overlay без нового инференса.
Выдача изображений, UI, поиск и согласование отложены.

Активные документы: [AGENTS.md](AGENTS.md), [PROJECT_PLAN.md](PROJECT_PLAN.md),
[backend.md](backend.md) для делегирования, [infrastructure.md](infrastructure.md)
и этот README. Дублирующие инструкции и старые задания удалены. Справочные README
работающего кода, научные результаты и история проверок сохраняются.

`compose.app.yaml` содержит `backend`, `db`, `rabbitmq`, `worker` и постоянные
volumes. Worker включается явно профилем `processing`. `compose.app.gpu.yaml`
подключает GPU только worker в текущей WSL. `compose.yaml` и `compose.gpu.yaml`
сохраняют отдельный стек научных сравнений.

## Запуск backend: загрузка и доставка

Секреты — только в игнорируемом `.secrets/app.env`. Для новой установки создать
файл один раз, не выводя пароли в терминал:

```bash
mkdir -p .secrets
python3 - <<'PYSETUP'
import os
import secrets
from pathlib import Path
content = Path("backend/app.env.example").read_text()
content = content.replace("REPLACE_WITH_RANDOM_HEX", secrets.token_hex(24))
content = content.replace("REPLACE_WITH_ANOTHER_RANDOM_HEX", secrets.token_hex(24))
fd = os.open(".secrets/app.env", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
with os.fdopen(fd, "w") as stream:
    stream.write(content)
PYSETUP
```

Существующий env сохранять: изменение пароля в файле не меняет пароль уже
инициализированной БД/брокера. Команды из корня репозитория:

```bash
docker compose -p kurs4-app -f compose.app.yaml config --quiet
docker compose -p kurs4-app -f compose.app.yaml build backend
docker compose -p kurs4-app -f compose.app.yaml up -d --wait db rabbitmq
docker compose -p kurs4-app -f compose.app.yaml run --rm backend alembic -c backend/alembic.ini upgrade head
docker compose -p kurs4-app -f compose.app.yaml up -d backend
```

[Swagger UI](http://127.0.0.1:8000/docs): `POST /images`, `GET /jobs/{job_id}`,
`GET /health`. Порт можно изменить через `APP_HTTP_PORT`. Загрузка JPEG/PNG —
до 10 MiB и 25 млн пикселей; проверяются реальные байты, EXIF и отсутствие
анимации. Оригинал и нормализованный RGB PNG сохраняются до транзакции БД.
На ошибке удаляются только файлы текущей загрузки. Ответы не раскрывают пути.
Повторный POST создаёт новое задание, одинаковые SHA-256 не объединяются.

Ответ `202` означает сохранение задания; `publish_status` меняется `pending → sent`
после publisher confirm. При недоступном RabbitMQ загрузка сохраняется, publisher
повторяет отправку с задержкой 5–60 секунд. При недоступной БД API возвращает `503`.
Один API-процесс и один publisher обязательны. `/health` проверяет БД, брокер
и цикл публикации; `processing: configured` означает наличие обработчика,
а не проверку доступности GPU, весов или работающего consumer.

При создании задания фиксируются модель, revision, SHA-256 весов и параметры из
`backend/checkpoints.json`. Старые задания с пустыми checkpoint не получают новые
модели задним числом: worker завершает их с `model_not_configured`, классификацию
пропускает с `detection_failed`. Для обработки такого изображения нужна новая загрузка.

Backend разделён на слои: `api.py` переводит HTTP-запросы и ошибки, `service.py`
управляет приёмом изображения и транзакцией, `repository.py` содержит SQL-запросы,
`db.py` создаёт подключения и сессию через `Depends(get_session)`, `tables.py`
описывает ORM. `main.py` собирает приложение и lifespan publisher. Зависимость
`get_service` можно подменить через `app.dependency_overrides`.

## Worker и модели

Новые команды запускают прикладной инференс, поэтому выполнять их после явного
решения о smoke/обработке изображений. В этой задаче реальных моделей не запускали:

```bash
docker compose -p kurs4-app -f compose.app.yaml --profile processing build worker
docker compose -p kurs4-app -f compose.app.yaml -f compose.app.gpu.yaml --profile processing up -d worker
```

Веса уже подготовлены сравнениями и подключаются read-only, автоматических
скачиваний нет. Внутри worker два окружения: `/opt/detection` (Transformers 5.19.0,
RF-DETR и ViT) и `/opt/annotation` (Transformers 4.57.6, Florence).
Точные ML-зависимости ограничены историческими lock-файлами; единственное исключение
для annotation — typing-extensions 4.16.0 для совместимости с текущим DB runtime.
`pip check` обязателен при сборке каждого окружения. Датасеты worker не нужны.
`imagenet_labels.json` — закреплённый словарь из classification/configs, сохранён
отдельно для автономного backend. `checkpoints.json` фиксирует хеши весов и файлов
модели; перед загрузкой в subprocess они проверяются, включая код Florence.

Общий `storage/locks/inference.lock` наследуется модельным процессом; блокировка
удерживается до его завершения. Batch 1, одна модель в памяти. Дефолтные таймауты:
600 секунд на этап, 2400 секунд на доставку задания с ожиданием lock; процессы
останавливаются по группам. Бюджет брокера — 7200 секунд. Настройки `APP_STAGE_TIMEOUT`
и `APP_JOB_TIMEOUT` менять согласованно с `rabbitmq.conf` и временем остановки.
При занятом lock после истечения бюджета сообщение перепубликуется через Celery retry.
Успешные/ошибочные конечные этапы неизменяемы, после аварии выполняются только
незавершённые; максимум две попытки обработки. Ноль объектов пропускает классификацию,
но сохраняет успешный итог при успешном caption. Ошибка детектора не мешает caption.
При сбое commit БД сообщение requeue, consumer останавливается; после перезапуска
он ждёт БД перед чтением очереди. Exactly-once не обещается. Remote control,
mingle и gossip отключены: для одного worker служебный обмен не нужен.

Обычная остановка сохраняет данные:

```bash
docker compose -p kurs4-app -f compose.app.yaml down
```

`down -v` удаляет volumes и не является штатным перезапуском. Для backup нужны
PostgreSQL и volume изображений; RabbitMQ хранит очередь в собственном volume.

## Проверки backend

Тестовый образ не включает модели. Для интеграции используются отдельные volumes
и сеть проекта `kurs4-backend-test`. **Не запускать интеграционные тесты против
рабочего брокера: тесты очищают тестовую очередь `inference`.** API-сервис тестового
проекта во время pytest должен быть остановлен; publisher запускается внутри тестов.

```bash
docker build --target test -f backend/Dockerfile -t kurs4-backend-test:v2 .
docker compose -p kurs4-backend-test -f compose.app.yaml up -d --wait db rabbitmq
docker compose -p kurs4-backend-test -f compose.app.yaml run --rm backend alembic -c backend/alembic.ini upgrade head
docker run --rm --init --network kurs4-backend-test_default --env-file .secrets/app.env -e APP_TEST_INTEGRATION=1 kurs4-backend-test:v2 python -m pytest backend/tests -q
docker compose -p kurs4-backend-test -f compose.app.yaml run --rm backend alembic -c backend/alembic.ini check
# Дополнительная проверка реального отключения БД (только изолированный проект):
.venv-backend/bin/python backend/tests/check_db_outage.py
docker compose -p kurs4-backend-test -f compose.app.yaml down
```

Без `APP_TEST_INTEGRATION=1` выполняются CPU-тесты, интеграция пропускается.
Для локальной разработки: `uv venv --python 3.12 .venv-backend`, затем
`uv pip sync --python .venv-backend/bin/python backend/requirements-dev.txt`.
Полные списки зависимостей закреплены в `backend/requirements*.txt`; `.in` —
исходные ограничения. Локальные команды: `.venv-backend/bin/python -m pytest backend/tests -q`,
`RUFF_CACHE_DIR=/tmp/kurs4-ruff .venv-backend/bin/python -m ruff check .` и
`RUFF_CACHE_DIR=/tmp/kurs4-ruff .venv-backend/bin/python -m ruff format --check .`.

Интеграционные тесты запускают отдельный настоящий prefork consumer и синтетические
модельные subprocess. `check_db_outage.py` — host-контроллер теста остановки БД,
он работает только с проектом `kurs4-backend-test` и образом тестов `v2`.
Эти проверки не подтверждают качество и совместимость реального GPU-инференса.

Проверки от 11.10.2026: 46 backend-тестов из финального Docker-образа,
отдельно 1 тест настоящего отключения БД; 73 CPU-теста сравнений в своих
окружениях. Ruff check/format, Alembic check, Compose config и сборки api/test/worker
прошли; импорты обоих ML-окружений и хеши файлов checkpoint проверены без загрузки
моделей. TestClient выдаёт одно предупреждение об устаревающей зависимости httpx.
Отчёт: [backend-layered-inference-2026-10-11.json](reports/verification/backend-layered-inference-2026-10-11.json).

## Папки и служебные файлы

- `backend/` — сборка приложения (`main.py`), HTTP (`api.py`), бизнес-логика (`service.py`),
  SQL (`repository.py`), сессии/DI (`db.py`), ORM (`tables.py`), ответы и payload (`schemas.py`),
  файлы (`storage.py`), Celery/publisher (`tasks.py`), конвейер (`processing.py`),
  модельные адаптеры (`models.py`), env (`settings.py`),
  миграции, Docker targets api/worker/test и отдельные CPU/интеграционные тесты.
- `compose.app.yaml`, `rabbitmq.conf` — инфраструктура приложения; `backend/app.env.example`
  — шаблон локальных секретов.
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
- `AGENTS.md` — навигация и правила; `PROJECT_PLAN.md` — архитектура;
  `backend.md` — краткое задание; `infrastructure.md` — контейнеры и ресурсы.
  Исходные материалы в `Материалы для курсовой/` не перемещаются и автоматически
  не коммитятся. README в каталогах сравнений документируют существующий код.

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
