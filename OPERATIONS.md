# Техническая документация Image Analysis

Версия 0.3.1. Запуск — в [README](README.md). Старые учебные материалы находятся в archive/coursework.

## Компоненты и хранение

Docker поднимает nginx frontend, FastAPI backend, Elasticsearch, локальное S3 MinIO, RabbitMQ и Celery worker. API модели не загружает. Очередь содержит только UUID; worker concurrency/prefetch/batch = 1. Elasticsearch и S3 не публикуют порты на хост; API/frontend доступны через localhost.

API проверяет JPEG/PNG, применяет EXIF orientation и создаёт RGB PNG без EXIF. Исходные байты не записываются в постоянное хранилище. Для передачи между API и worker нормализованный PNG временно лежит в volume `pending_inputs`; метаданные входа — в Elasticsearch. После этого API отправляет UUID с подтверждением RabbitMQ. Потерянное подтверждение может означать доставленное сообщение, поэтому временный вход сохраняется для повтора.

RF-DETR Small выполняет детекцию параллельно с описанием сцены Florence-2-base-ft. Затем тот же Florence последовательно описывает вырезки с padding 10%; bbox и COCO label остаются исходными. Классификация здесь — описание объектов, без ViT, определения цвета или выдуманных структурированных атрибутов.

В S3 постоянно сохраняются **только вырезки PNG**. Elasticsearch хранит UUID, размеры, дату, исходные ответы аннотации/классификации/детекции, список вырезок и их S3 keys/размеры/SHA256. В индексе нет изображений, base64 или оригинальных байтов. `results.json` формируется из manifest по запросу, в S3 не записывается. Mapping сохраняет описания как text, COCO label/object_id как keyword, коллекции объектов как nested. Интерфейс поиска пока отсутствует.

После всех вычислений worker загружает вырезки, затем одной условной записью публикует полный ES-документ. До этой записи результат недоступен. Используются `_seq_no`/`_primary_term`, чтобы повторная доставка не перезаписала готовый результат ([документация Elasticsearch](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/optimistic-concurrency-control)). Готовый UUID не пересчитывается. После публикации временный вход удаляется; повторная доставка также завершает удаление, если процесс прервался после записи ES.

Модельные subprocess не импортируют клиенты хранилища. Для них ограничено число malloc arenas и включён ранний возврат освобождённых CPU-буферов загрузки моделей; пороги RAM/VRAM остаются действующими ([параметры glibc](https://sourceware.org/glibc/manual/latest/html_node/Malloc-Tunable-Parameters.html)).

Общей транзакции ES/S3/RabbitMQ нет. Сбой S3 или ES может оставить вырезки без ссылки; неоднозначное подтверждение ES не даёт безопасно удалить их сразу. Автоматической очистки таких файлов и reconciliation необработанных входов нет. При ошибке нормализованный вход сохраняется, поскольку нужен для повтора; при повторной попытке возможен новый инференс. Exactly-once не обещается. Общий inference-lock и временный scratch моделей сохранены.

## HTTP API

| Метод и путь | Результат |
|---|---|
| `POST /images` | Multipart `file`; `202 {"image_id":"UUID"}` после сохранения временного входа/метаданных и подтверждения брокера. |
| `GET /images/{id}` | Готовый manifest, `404` до готовности. |
| `GET /images/{id}/files/results.json` | JSON из Elasticsearch. |
| `GET /images/{id}/files/crops/{object_id}.png` | Вырезка из S3 с проверкой размера/SHA256. |
| `GET /images/{id}/archive` | ZIP: `results.json` и `crops/`; полный исходник в него не входит. |
| `GET /health` | Готовность RabbitMQ, Elasticsearch index и S3 bucket; не качество моделей. |

Исходник, `normalized.png` и `detection.png` после обработки по API недоступны. Браузер держит локальный preview выбранного файла в текущей вкладке; применяет EXIF orientation и проверяет размеры перед наложением bbox. При невозможности preview JSON и вырезки остаются доступны. Новый выбор файла во время ожидания не подменяет preview уже отправленного изображения. Двухколоночная компоновка и выделение всех bbox сохранены.

Ошибки загрузки: 413 — лимит, 415 — формат, 422 — некорректное изображение. Недоступность хранилищ/брокера — 503 без внутренних параметров. Браузер автоматически ждёт результат; отдельного endpoint статуса нет, ошибки модели видны в worker logs.

## Настройки и запуск

Launcher создаёт `.secrets/app.env` со случайными паролями; этот файл не коммитится и не выводится. API/worker используют существующий S3-клиент с path addressing ([настройки Boto3](https://docs.aws.amazon.com/boto3/latest/guide/configuration.html)). Сборка настроена только для локального S3. Подключение удалённого хранилища, его авторизация/TLS и перенос данных не входят в эту версию.

| Переменная | Назначение |
|---|---|
| `APP_ELASTICSEARCH_URL` | `http://elasticsearch:9200` внутри Docker. |
| `APP_ELASTICSEARCH_INDEX` | `image-results`. |
| `APP_S3_ENDPOINT_URL` | `http://s3:9000` внутри Docker. |
| `APP_S3_ACCESS_KEY`, `APP_S3_SECRET_KEY` | Совпадают с локальными `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`. |
| `APP_S3_BUCKET`, `APP_S3_REGION` | `image-analysis`, `us-east-1`. |
| `APP_PENDING_INPUT_DIR` | `/var/lib/kurs4/pending`, общий временный вход API/worker. |
| `APP_BROKER_URL` | URL RabbitMQ из секретов. |
| `APP_DEVICE` | Явный `cpu`/`cuda`; launcher выбирает режим. |
| `APP_MAX_UPLOAD_BYTES`, `APP_MAX_IMAGE_PIXELS` | 10 МиБ, 25 000 000. |
| `APP_MAX_ARTIFACT_BYTES`, `APP_MAX_BUNDLE_BYTES` | 128 МиБ, 256 МиБ. |
| `APP_STAGE_TIMEOUT`, `APP_JOB_TIMEOUT` | 600 и 2400 секунд. |

Порты `APP_FRONTEND_PORT`/`APP_HTTP_PORT` (8080/8000) и `APP_WORKER_MEMORY` задаются в окружении Compose. Остальные `APP_*` — в `.secrets/app.env`. ES использует heap 512 МиБ и лимит контейнера 1 ГБ; это локальная single-node конфигурация. Linux x86_64, CPU и NVIDIA CUDA остаются явными режимами. Реальный CPU-инференс не проверялся; проверку CUDA см. в конце документа.

MinIO собирается из закреплённого upstream release в `docker/minio.Dockerfile`, лицензия включена в образ. Первой сборке нужен интернет для Go-зависимостей. Отдельный MinIO контейнер имеет постоянный volume `s3data`; Elasticsearch — `esdata`. Модели и секреты не входят в образы/архив Git.

Инициализация существующего локального окружения без запуска моделей:

```bash
docker compose -p kurs4-app -f compose.app.yaml build s3 backend frontend
docker compose -p kurs4-app -f compose.app.yaml up -d --wait elasticsearch s3 rabbitmq
docker compose -p kurs4-app -f compose.app.yaml run --rm --no-deps backend python -m backend.db
docker compose -p kurs4-app -f compose.app.yaml up -d backend frontend
```

Обычный запуск с обработкой — `python3 scripts/start_app.py --device cpu` (либо `cuda`). Режим `cuda` требует зарегистрированного NVIDIA runtime Docker и проверяется до сборки. Если локальный Docker Engine запущен прямо в WSL без этого runtime, используйте `python3 scripts/start_app.py --device wsl-cuda`: отдельный override передаёт `/dev/dxg` и библиотеки WSL, сохраняя `APP_DEVICE=cuda`. CPU автоматически не подставляется. Настройка стандартного NVIDIA runtime описана в [документации NVIDIA](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

`MemoryError: Less than 768 MiB host RAM available` означает нехватку доступной RAM WSL/хоста во время загрузки моделей. Защиту не отключайте: освободите память или увеличьте лимит WSL. После такой ошибки worker освобождает частично загруженные модели и ждёт 30 секунд до новой попытки, не забирая задания из очереди. Heap Elasticsearch ограничен 512 МиБ, чтобы оставить больше памяти двум модельным процессам; это не отменяет общего требования к ресурсам.

Остановка обработки/всего приложения:

```bash
docker compose -p kurs4-app -f compose.app.yaml --profile processing stop
docker compose -p kurs4-app -f compose.app.yaml --profile processing down
```

При обычной остановке не добавляйте `-v`: это удалит новые результаты и вырезки. Для сохранения текущего состояния нужны согласованные данные ES, S3, незавершённой очереди/временных входов и секреты. Публичное многопользовательское развёртывание не входит в поставку.

## Переход с PostgreSQL

Пользователь разрешил начать с пустой базы и удалить старую PostgreSQL без дампа/переноса. Перед переходом остановите прежние backend/worker; старые UUID очереди относятся к прежнему хранилищу и должны быть удалены из очереди `inference`, если они есть. Создайте настройки Elasticsearch/S3 по `backend/app.env.example`, сохранив действующие настройки RabbitMQ. Launcher дополняет старый файл отсутствующими настройками нового хранилища без замены существующих паролей.

Сервис `db` и volume `pgdata` больше не используются Compose. Старые SQL-модули/миграции и их тесты оставлены на месте, чтобы сохранить уже имевшиеся локальные пользовательские изменения; текущий runtime их не импортирует, SQL-зависимости из сборки удалены. Тесты старых миграций/импорта/старого worker исключены из текущего pytest. Команды Alembic/старого импорта к новой системе не применимы.

## Разработка и проверки

```bash
.venv-backend/bin/python -m pytest -q backend/tests
.venv-backend/bin/python -m ruff check --no-cache .
.venv-backend/bin/python -m ruff format --check --no-cache .
```

`APP_TEST_INTEGRATION=1` включает тесты настоящих Elasticsearch/S3. Используйте отдельные тестовые контейнеры и тестовые credentials: тест создаёт и удаляет только свои случайные index/bucket. Инференс в нём синтетический, веса не загружаются. Новые отчёты, benchmark и проверки производительности без отдельного поручения не создаются.

Проверено 07.11.2026 для 0.3.1: 122 CPU/mock теста backend, 2 интеграционных теста пропущены в этом прогоне; Ruff check/format, сборки и конфигурации Docker прошли. После увеличения локального лимита WSL с 8 до 12 ГБ RAM и swap с 2 до 4 ГБ восстановлена обработка уже принятого изображения 810×1080 на NVIDIA GPU: сохранены аннотация, детекция, описания 5 объектов и 5 вырезок в S3. Получены JSON и ZIP без полных изображений, проверено удаление временного входа. Это функциональная проверка одного задания; настоящий CPU-инференс не проверен. Бенчмарки и нагрузочные проверки не выполнялись.
