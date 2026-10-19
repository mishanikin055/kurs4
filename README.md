# Запуск приложения

Команды для текущей WSL2: нужны запущенный Docker с Compose, Python 3 и доступ к NVIDIA GPU.
Веса RF-DETR Small, ViT-B/16 и Florence-2 base-ft должны уже находиться в
`detection/models/rf_detr_small`, `classification/models/vit_b_16` и
`annotation/models/florence2_base_ft`. При запуске они не скачиваются.

19.10.2026 рабочий стек пересобран и запущен по разрешению пользователя: старая БД
удалена, новая схема применена, image volume сохранён. Три модели загружены и прогреты;
нового end-to-end прогона пользовательского изображения не было. Подробности —
[проверка и обновление](reports/verification/simple-pipeline-2026-10-19.md).

## 1. Подготовить настройки

Выполнить в терминале WSL. Если `.secrets/app.env` уже существует, он сохранится.

```bash
cd /home/mikhail/kurs4
mkdir -p .secrets
python3 - <<'PY'
import os
import secrets
from pathlib import Path

path = Path(".secrets/app.env")
if not path.exists():
    content = Path("backend/app.env.example").read_text()
    content = content.replace("REPLACE_WITH_RANDOM_HEX", secrets.token_hex(24))
    content = content.replace("REPLACE_WITH_ANOTHER_RANDOM_HEX", secrets.token_hex(24))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(content)
PY
```

## 2. Собрать и запустить

Выполнять по порядку; при ошибке остановиться и проверить её. Первая сборка worker может занять длительное время.

```bash
docker compose -p kurs4-app -f compose.app.yaml --profile processing build backend worker frontend
docker compose -p kurs4-app -f compose.app.yaml up -d --wait db rabbitmq
docker compose -p kurs4-app -f compose.app.yaml run --rm backend alembic -c backend/alembic.ini upgrade head
docker compose -p kurs4-app -f compose.app.yaml -f compose.app.gpu.yaml --profile processing up -d backend worker frontend
```

Worker сначала проверяет веса, загружает и прогревает все три модели, затем
начинает обрабатывать загруженные картинки, включая уже накопленные задания.
Модели остаются в памяти до остановки worker; на новом изображении веса не
перечитываются. Вычисления остаются последовательными, одно задание одновременно.
При нехватке RAM/VRAM consumer не запускается. Подробности текущего короткого
GPU smoke и его ограничений — [проверка постоянных моделей](reports/verification/resident-models-2026-10-13.md).

`backend/runtime.py` отвечает за постоянные модельные subprocess и локальный IPC;
`backend/tasks.py` — за Celery, `backend/processing.py` — за этапы и сохранение результатов.
Проверка весов выполняется при старте. Перед сменой весов или параметров завершить
очередь и пересоздать worker. Он удерживает общий inference-lock весь срок жизни
моделей, поэтому перед запуском научного сравнения worker нужно остановить:

```bash
docker compose -p kurs4-app -f compose.app.yaml --profile processing stop worker
```

## 3. Загрузить картинку и сохранить копию

Открыть [localhost:8080](http://localhost:8080), выбрать JPEG/PNG и нажать
«Отправить». API отвечает `202` только после подтверждения RabbitMQ.
Во время загрузки в БД ничего не создаётся. В сообщении очереди — только `image_id`.
API не загружает модели и не зависит от доступности PostgreSQL.

Worker выполняет RF-DETR Small → ViT-B/16 на всех вырезках → Florence-2 base-ft.
Вырезки расширяются на 10% с каждой стороны в пределах изображения; bbox не меняется.
Если объектов нет, классификация сохраняется как пустой список, аннотатор работает.

После полного инференса **всегда** сохраняется основная копия в `image_data`:

```text
/var/lib/kurs4/images/results/<image_id>/
  original          # исходные байты JPEG/PNG
  normalized.png    # RGB PNG после EXIF orientation
  detection.png     # PNG с исходными COCO-рамками и подписями
  results.json      # ответы трёх моделей и имена/ревизии checkpoint
```

Сначала комплект записывается во временный каталог, затем публикуется одним
переименованием. Только после этого worker одной транзакцией сохраняет запись БД.
`raw_caption` не меняется, `object_id` связывает bbox и класс вырезки. Scores,
top-k, замеры и параметры в БД не записываются. Семантическое объединение меток
в этой простой версии не выполняется; исходные ответы остаются отдельно.

**Дополнительная папка на компьютере** выбирается кнопкой «Выбрать папку».
После обработки нажать «Сохранить копию»: браузер создаст подпапку `<image_id>`
и запишет четыре файла. Это дополнительная копия; файлы в volume остаются всегда.
Папку можно не выбирать — тогда скачивается ZIP. Выбор папки требует поддержки
File System Access API и защищённого контекста (например, localhost в Chromium).
В остальных браузерах доступен ZIP через обычное окно загрузки браузера.

Автоматического опроса, callback, статусов и сообщений об ошибках инференса нет.
Если файлы ещё не появились, ручное сохранение возвращает `404`; попробовать позже.
Вкладка должна оставаться открытой для сохранения в выбранную папку. После обновления
страницы скачивание по известному UUID доступно через API; повторная загрузка
изображения создаёт новый UUID. Состояние браузерной папки на сервер не передаётся.

Посмотреть основную копию прямо в volume без загрузки на компьютер:

```bash
docker compose -p kurs4-app -f compose.app.yaml exec backend ls /var/lib/kurs4/images/results
# Для конкретного UUID:
# docker compose -p kurs4-app -f compose.app.yaml exec backend cat /var/lib/kurs4/images/results/UUID/results.json
```

HTTP API: [Swagger](http://localhost:8000/docs).

| Endpoint | Назначение |
|---|---|
| `POST /images` | Multipart `file`, JPEG/PNG до 10 MiB и 25 Мп; `202`: `image_id`, `output_folder` относительно image volume. |
| `GET /images/{image_id}/files/{filename}` | Явное скачивание `original`, `normalized.png`, `detection.png` или `results.json` после появления полного комплекта. |
| `GET /images/{image_id}/archive` | ZIP с четырьмя файлами готового результата. |
| `GET /health` | Доступность RabbitMQ для приёма загрузок, без проверки моделей/БД. |

`/jobs/...` удалены. HTTP-валидация загрузок и коды HTTP при недоступности брокера
сохранены: невалидный файл не публикуется, неуспешная отправка не выдаётся за успешную.
При потере подтверждения публикации входные файлы остаются: брокер уже мог принять
сообщение. Такая неопределённая загрузка не повторяется автоматически; новый POST
создаёт новый UUID. Зависшие входные каталоги автоматически не очищаются.

## База данных и обновление

Активная таблица — `processed_images`: UUID (PK), три пути (`original_path`,
`normalized_path`, `result_path`), ширина/высота и четыре JSONB-поля: `models`,
`detection`, `classification`, `annotation`. Одна строка — полностью обработанное
изображение. БД хранит ссылки на файлы в volume и ответы моделей, а не бинарные PNG.
В новой таблице нет заданий, статусов, ошибок, истории доставки или попыток.

Новая миграция `c81f0e392ab4` только добавляет эту таблицу. Старые `images`, `jobs`
и `model_results` остаются для сохранности прежних данных; новый поток их не
читает и не заполняет. ORM этих таблиц оставлен для проверки исторических миграций.
Старые результаты не перенесены и не пересчитаны. Предыдущая миграция
`b7d91f4e20ac` остаётся необратимой: при обновлении с более ранней версии сначала
сохранить backup БД и файлов, как и раньше. Автоматического downgrade нет.

Перед обновлением существующего приложения прекратить загрузки и **завершить
старую очередь прежним worker**: старые сообщения содержат `job_id`, новые —
`image_id`. Без завершения очереди версии несовместимы. Затем остановить API/worker,
сохранить backup, собрать образы и применить миграции командами раздела 2:

```bash
docker compose -p kurs4-app -f compose.app.yaml --profile processing stop backend worker
```

Новый worker проверяет доступность `processed_images` до загрузки моделей.
Повторная доставка готового изображения не запускает инференс заново.
Если файлы уже записаны, а commit БД не удался, после восстановления БД worker
читает сохранённый `results.json` и повторяет только запись БД. ACK — после commit.
Ошибка модели записывается только в логи Celery; готовой записи/комплекта не будет,
входные файлы останутся. Автоматического продолжения отдельных этапов больше нет;
при потере процесса до публикации комплекта вычисления могут повториться целиком.

## Frontend и проверки

`frontend/` — статические HTML/CSS/JavaScript и nginx. `/api/` проксируется к API;
у nginx нет доступа к БД, image volume, моделям и секретам. Слева находится
локальный предпросмотр выбранного или перетащенного изображения, справа — выбор
файла, дополнительной папки и отправка на обработку. Предпросмотр через object URL
не отправляет файл на сервер. Под выбором папки нет постоянных пояснений;
сообщения появляются после действий пользователя. Шрифт Golos Text и его лицензия
OFL находятся в `frontend/fonts/` и выдаются локально, без запросов к внешним CDN.
Сборка JavaScript и npm для приложения не нужны.
Тест `frontend/tests/browser.cjs` использует Playwright
с перехваченным API: отправку в очередь, отсутствие polling, запись выбранной
папки (mock File System Access API), ZIP, локальный предпросмотр и мобильную ширину.
`FRONTEND_URL` задаёт адрес (по умолчанию теста — localhost:8098), `NODE_PATH` — путь
к внешнему Playwright, `PLAYWRIGHT_CHROMIUM_EXECUTABLE` — уже установленный Chromium.
Для localhost:8098 можно отдельно запустить `python3 -m http.server 8098 --bind 127.0.0.1 --directory frontend`.

```bash
.venv-backend/bin/python -m pytest -q backend/tests
.venv-backend/bin/python -m ruff check --no-cache .
.venv-backend/bin/python -m ruff format --check --no-cache .
```

Полный `pytest` всего репозитория дополнительно требует зависимости научных
сравнений (в частности, pycocotools и nltk); backend-окружение их не содержит.

Интеграционные тесты (`APP_TEST_INTEGRATION=1`) запускаются **только с отдельной
тестовой БД и брокером**: они очищают тестовую очередь. Они проверяют настоящий
PostgreSQL/RabbitMQ и отдельный Celery worker с синтетическими модельными процессами.
Пример повторяемого запуска без GPU, с уже собранным тестовым образом:

```bash
docker compose -p kurs4-simple-test -f compose.app.yaml up -d --wait db rabbitmq
docker build --target test -t kurs4-backend-test:v2 -f backend/Dockerfile .
docker run --rm --network kurs4-simple-test_default --env-file .secrets/app.env -e APP_TEST_INTEGRATION=1 -v "$PWD/backend:/app/backend:ro" -v "$PWD/pyproject.toml:/app/pyproject.toml:ro" kurs4-backend-test:v2 sh -c 'alembic -c backend/alembic.ini upgrade head && python -m pytest -q backend/tests'
docker compose -p kurs4-simple-test -f compose.app.yaml down
```

Mock-инференс не подтверждает работоспособность настоящих моделей. В этой переделке
реальный GPU smoke и научные сравнения не запускались. Короткий прежний GPU smoke
от 13.10.2026 описан в [отчёте](reports/verification/resident-models-2026-10-13.md).

## Логи и остановка

```bash
docker compose -p kurs4-app -f compose.app.yaml --profile processing logs --tail 100 -f backend worker
docker compose -p kurs4-app -f compose.app.yaml --profile processing down
```

Не добавлять `-v`: эта опция удаляет постоянные volumes.
Научные сравнения и сохранённые отчёты не меняются:
[детекция](detection/README.md), [классификация](classification/README.md),
[аннотирование](annotation/README.md). Архитектура: [PROJECT_PLAN.md](PROJECT_PLAN.md).
