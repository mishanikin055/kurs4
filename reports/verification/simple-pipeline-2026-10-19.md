# Проверка простой обработки изображений — 19.10.2026

## Реализованный поток

FastAPI сохраняет входы и непосредственно публикует image_id с RabbitMQ confirms.
Записей БД во время загрузки нет. Worker выполняет три этапа, атомарно публикует
original/normalized.png/detection.png/results.json в image_data/results/UUID,
затем одной транзакцией сохраняет processed_images. Основная копия в volume
обязательна. Папка на компьютере и ZIP — только дополнительное ручное сохранение.
Jobs API, polling и outbox удалены. Ошибки моделей — логи, без записей в новой БД.

Новая миграция c81f0e392ab4 добавляет processed_images; старые таблицы/данные не
удаляет. После публикации файлов сбой commit восстанавливается из results.json
без повторения моделей. Готовый UUID обрабатывается идемпотентно.

## Проверки

| Проверка | Результат |
|---|---|
| `.venv-backend/bin/python -m pytest -q backend/tests` | 67 passed, 11 skipped; пропуски — настоящая инфраструктура и отдельный outage-controller. |
| Изолированный PostgreSQL + RabbitMQ + отдельный Celery consumer, `APP_TEST_INTEGRATION=1`, тот же набор backend/tests | 77 passed, 1 skipped; модели синтетические, проверены обычные и постоянные subprocess. |
| `backend/tests/check_db_outage.py` | 1 passed: настоящая остановка тестового PostgreSQL во время инференса, публикация файлов, requeue, восстановление БД, повтор только записи БД. |
| `ruff check --no-cache .` | All checks passed. |
| `ruff format --check --no-cache .` | 77 files already formatted. |
| `git diff --check` | Без ошибок. |
| Compose config: приложение, приложение + GPU/processing | Оба прошли. |
| `node --check` frontend/app.js и frontend/tests/browser.cjs | Оба прошли. |
| Playwright Chromium, frontend/tests/browser.cjs | Прошли загрузка, отсутствие polling, ручная запись четырёх файлов в папку, ZIP, попытка скачивания до готовности, ошибка upload, fallback без API выбора папки и мобильная ширина. |

Интеграция выполнялась в отдельном Compose-проекте kurs4-simple-test,
на существующем test-образе с примонтированным текущим backend/ и pyproject.toml.
До тестов применены миграции. Тест миграций проверил старые данные в отдельной
PostgreSQL-схеме, соответствие ORM и `alembic check`. После проверки тестовые
контейнеры/сеть остановлены и удалены; рабочий kurs4-app не изменялся.
Outage-controller останавливал только db проекта kurs4-simple-test.

Playwright запускался через внешний установленный пакет и уже установленный
Chromium. Недостающие libnspr4/libnss3/libasound2t64 извлечены из deb-пакетов
в /tmp/kurs4-browser-deps; системные пакеты и зависимости проекта не менялись.
File System Access API в браузерном тесте подменён тестовым directory handle
и WritableStream; проверены имена и байты всех четырёх файлов. Настоящий системный
диалог выбора папки вручную не проверялся. Десктопный screenshot просмотрен.

## Ограничения

Полный `pytest -q` всего репозитория в backend-окружении остановился на collection:
нет pycocotools для detection/tests и nltk для annotation/tests. Эти зависимости
научных сравнений ради переделки приложения не устанавливались. Релевантные тесты
backend выполнены отдельно; научные сравнения и таблицы не запускались.

Есть прежние предупреждения Starlette/TestClient (httpx) и Alembic path_separator;
ошибок в релевантных проверках нет. Настоящие RF-DETR/ViT/Florence не запускались,
нового GPU smoke нет. Рабочие контейнеры, очередь, БД, её миграции и volumes не
обновлялись. Новая версия требует завершения старой очереди прежним worker,
миграции и пересборки/пересоздания по README. Исторический GPU smoke не является
проверкой изменённого потока.

Дополнительное сохранение требует открытой вкладки и ручного нажатия после
обработки; автоматической доставки на компьютер нет. Выбор папки зависит от
File System Access API и localhost/HTTPS; иначе доступен ZIP. Без авторизации
доступ к готовым файлам остаётся по UUID, как и в предыдущем локальном приложении.


## Обновление рабочей установки по следующему запросу пользователя

После исходной проверки пользователь явно разрешил пересборку контейнеров и
удаление старой БД. Обновлён проект kurs4-app на коде cc2f4b5:

- Перед остановкой очередь inference была пуста: ready=0, unacknowledged=0.
  Старые API/frontend/worker остановлены; сообщения не удалялись.
- Пересобраны образы backend, worker и frontend. Зависимости использовали Docker cache.
- Удалены только старый контейнер db и volume kurs4-app_pgdata. В старой БД было
  29 images и 29 jobs. Пользователь разрешил удаление без сохранения старой БД.
  Image volume и RabbitMQ volume не удалялись.
- PostgreSQL создан заново; применены миграции до c81f0e392ab4. Новая таблица
  processed_images пуста. Исторические таблицы из цепочки миграций тоже пустые;
  новый поток их не использует.
- Backend/worker/frontend пересозданы, worker использует compose.app.gpu.yaml.
  Все пять сервисов запущены, PostgreSQL и RabbitMQ healthy.
- GET /health на localhost:8000 и GET /api/health на localhost:8080 вернули
  {"rabbitmq": true}. OpenAPI содержит только новые upload/download/health routes.
  Выдаваемые frontend HTML/JS/CSS побайтно соответствуют файлам репозитория.
- `alembic check` в рабочем backend: No new upgrade operations detected.
- Worker загрузил и прогрел настоящие RF-DETR Small, ViT-B/16 и Florence-2 base-ft:
  startup_s 9.144 / 2.861 / 4.889 соответственно. Celery сообщил ready;
  RabbitMQ показывает один consumer, ноль ожидающих/неподтверждённых сообщений.
  Память worker после готовности — около 3.84 GiB при лимите 5 GiB.

Таким образом, ограничения предыдущего раздела об отсутствии обновления рабочей
установки относятся к первоначальной проверке, до дополнительного разрешения.
Сейчас новая версия запущена: http://localhost:8080 (UI), http://localhost:8000/docs.
Настоящая загрузка/прогрев моделей подтверждены; новый пользовательский файл через
весь изменённый pipeline после пересборки не запускался. Научных сравнений нет.
