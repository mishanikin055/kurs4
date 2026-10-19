# Запуск проекта через Docker

Команды — в WSL/Bash. Приложение использует `compose.app.yaml`, пять контейнеров
и GPU-дополнение `compose.app.gpu.yaml`. `compose.yaml` — только сравнения.

## Первый запуск или обновление кода

Подготовить `.secrets/app.env` по [README](README.md#1-подготовить-настройки).
Для существующего приложения сначала прекратить загрузки, завершить прежнюю
очередь старым worker, остановить API/worker и сохранить backup БД и image volume.
Старая версия сообщений содержит job_id, новая — image_id; смешивать их нельзя.
Миграция c81f0e392ab4 только добавляет processed_images; более ранняя компактная
миграция b7d91f4e20ac остаётся необратимой, если ещё не была применена.

```bash
cd /home/mikhail/kurs4
dc() { docker compose -p kurs4-app -f compose.app.yaml -f compose.app.gpu.yaml --profile processing "$@"; }
dc build backend worker frontend
dc up -d --wait db rabbitmq
dc run --rm backend alembic -c backend/alembic.ini upgrade head
dc up -d backend worker frontend
```

Продолжать только после успеха команды. Нужны уже загруженные выбранные веса;
worker при старте загружает и прогревает три постоянные модели. Это реальный
модельный запуск; выполнять только в рамках явно разрешённого запуска приложения.

## Обычное возобновление после остановки

Если код/схема уже обновлены, использовать существующие контейнеры:

```bash
dc start --wait db rabbitmq
dc start backend worker frontend
```

Для Paused сначала `dc unpause db rabbitmq`, затем `dc unpause backend worker frontend`.
Открыть [localhost:8080](http://localhost:8080), выбрать изображение и отправить.
Основная копия всегда в image volume: `results/<image_id>/`. Никаких заданий или
статусов в БД при загрузке не создаётся. В интерфейсе можно дополнительно выбрать
папку компьютера и после обработки нажать «Сохранить копию»; без папки скачивается ZIP.
Автоматического опроса нет. Полные пути и API — [README](README.md).

```bash
dc exec backend ls /var/lib/kurs4/images/results
dc logs --tail 50 -f backend worker
```

`Ctrl+C` завершает просмотр логов. Ошибки инференса остаются в логах worker.

## PostgreSQL

```bash
dc exec db sh -c 'exec psql -U "$POSTGRES_USER" -d "$POSTGRES_DB"'
```

Внутри psql:

```sql
\dt
\d processed_images
SELECT id, result_path, annotation FROM processed_images LIMIT 10;
\q
```

Изображения хранятся файлами, БД содержит пути и ответы моделей. Старые таблицы
остаются историей и новым потоком не используются. Порт БД на хост не опубликован.

## Остановка

```bash
dc stop
```

Данные и очередь сохраняются. `down -v` удаляет постоянные данные и для перезапуска
не нужен. Тесты нового потока используют mock-модели; нового GPU smoke этой переделки нет.
