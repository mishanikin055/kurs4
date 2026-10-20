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
перечитываются. На одном изображении параллельно выполняются две ветки:
RF-DETR Small → ViT-B/16 и Florence-2 base-ft. Одно задание одновременно,
concurrency/prefetch/batch 1. Потоки координатора обмениваются IPC с отдельными
модельными процессами; CUDA в Celery prefork не загружается.
При нехватке RAM/VRAM consumer не запускается. Подробности текущего короткого
GPU smoke и его ограничений — [проверка постоянных моделей](reports/verification/resident-models-2026-10-13.md).
Он проверял последовательный инференс. Ускорение и расход памяти при параллельной
нагрузке пока не измерены; изменение кода проверяется CPU/mock и отдельной
интеграцией PostgreSQL/RabbitMQ/Celery без настоящих моделей.
Проверки параллельной версии — [отчёт от 19.10.2026](reports/verification/parallel-annotation-2026-10-19.md).
После обновления кода worker нужно пересобрать и пересоздать командами раздела 2;
в этой задаче рабочий GPU-контейнер не обновлялся и модели не запускались.

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

Worker запускает RF-DETR Small → ViT-B/16 на всех вырезках и одновременно
Florence-2 base-ft на целом нормализованном изображении. Сохранение файлов и
запись в PostgreSQL начинаются после успешного завершения обеих веток.
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
top-k, замеры и параметры в БД не записываются. Исходные ответы остаются отдельно;
для интерфейса GET вычисляет совместимое уточнение без записи производных данных в БД.

После отправки результат отображается **автоматически**: браузер сразу запрашивает
`GET /images/{image_id}` и повторяет запрос через 2 секунды после `404`, без
перекрывающихся запросов. После успеха загружается PNG, опрос прекращается.
Управление, описание и таблица слева, изображение справа; на узком экране — один столбец.
Новая загрузка отменяет ожидание прежнего результата. Ошибка сети/сервера или 40 минут
ожидания останавливают опрос и показывают «Повторить ожидание»; один запрос ограничен
15 секундами. Это ожидание готового комплекта, без таблиц заданий и endpoint статуса.
Путь в volume и технические сведения о хранении в UI не выводятся.

Колонка «Уточнение · ImageNet» заполняется только совместимыми с COCO метками
по существующему `backend/imagenet_coco_mapping.json` и `imagenet_labels.json`,
с сопоставлением по `object_id`. Конфликт, неизвестное соответствие, отсутствующая
метка и точное повторение COCO оставляют ячейку пустой. Например, `dog` →
`Labrador retriever` показывается, `person` → `unicycle` и `skateboard` → `Loafer` — нет.
Словарь проверяет совместимость классов, но не подтверждает породу/подтип по пикселям:
ошибка среди совместимых подтипов остаётся возможной. Исходный top-1, COCO-рамки,
`raw_caption` и скачиваемый `results.json` сохраняются без изменений.

**Дополнительная папка на компьютере** выбирается кнопкой «Выбрать папку».
После показа результата браузер **автоматически** создаёт подпапку `<image_id>`
и записывает четыре файла. Если выбрать папку после получения результата, запись
начнётся сразу после выбора. Кнопок «Сохранить копию» и «Показать результат» нет.
Это дополнительная копия; файлы в volume остаются всегда.
Без выбранной папки доступна кнопка **«Скачать ZIP»**. При ошибке записи/потере
разрешения результат остаётся видимым; можно выбрать папку снова или скачать ZIP.
Выбор папки требует поддержки File System Access API и защищённого контекста
(например, localhost в Chromium). Разрешение на запись получается при выборе папки;
автоматическое сохранение проверяет его без запроса разрешения вне действия пользователя.
В остальных браузерах доступен ZIP через обычное окно загрузки браузера.

Callback, статусов и сообщений об ошибках инференса нет: при длительном отсутствии
комплекта интерфейс предлагает повторить ожидание. Вкладка должна оставаться открытой
для автоматического сохранения в выбранную папку. После обновления страницы скачивание
по известному UUID доступно через API; повторная загрузка изображения создаёт новый UUID.
Состояние браузерной папки на сервер не передаётся.

Посмотреть основную копию прямо в volume без загрузки на компьютер:

```bash
docker compose -p kurs4-app -f compose.app.yaml exec backend ls /var/lib/kurs4/images/results
# Для конкретного UUID:
# docker compose -p kurs4-app -f compose.app.yaml exec backend cat /var/lib/kurs4/images/results/UUID/results.json
```

HTTP API: [Swagger через frontend](http://localhost:8080/api/docs)
или [напрямую через backend](http://localhost:8000/docs). `/docs` на frontend
перенаправляет на `/api/docs`. Страница `frontend/swagger.html` и инициализатор
`frontend/swagger.js` задают правильный адрес схемы `/api/openapi.json` и сервер
`/api` для Try it out. Только для страницы Swagger CSP разрешает CDN jsDelivr,
откуда загружаются стандартные Swagger JS/CSS; без доступа к CDN документация
не загрузится. CSP основного интерфейса не изменён.

| Endpoint | Назначение |
|---|---|
| `POST /images` | Multipart `file`, JPEG/PNG до 10 MiB и 25 Мп; `202`: `image_id`, `output_folder` относительно image volume. |
| `GET /images/{image_id}` | Готовый результат с исходными ответами и вычисленным `classification.objects[].refinement` для UI; до готовности `404`, без кэширования. |
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

Миграция `c81f0e392ab4` добавляет эту таблицу, следующая `d20f4a6b819c` по запросу
пользователя от 20.10.2026 удаляет старые `images`, `jobs` и `model_results`.
В актуальной БД остаются `processed_images` и служебная `alembic_version`.
Старые миграции сохранены; исторические преобразования проверяются отдельно.
Миграция удаления старых таблиц необратима: перед обновлением остановить загрузки,
завершить очередь и сохранить дамп БД. Автоматического downgrade нет.
Чистая установка через `alembic upgrade head` получает ту же схему.

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
Обычная ошибка модели дожидается завершения соседней ветки без публикации
комплекта. При остановке координатора, потере процесса, ошибке IPC или OOM
активные запросы прерываются, а worker перезапускается перед повторной доставкой.
Это не позволяет следующему изображению получить ответ прерванного запроса.

## Frontend и проверки

`frontend/` — статические HTML/CSS/JavaScript и nginx. `/api/` проксируется к API;
у nginx нет доступа к БД, image volume, моделям и секретам. Интерфейс оформлен
в тёплой палитре без боковых полос, со строкой трёх
этапов обработки. Под выбором папки нет постоянных пояснений; сообщения появляются
после действий пользователя.
Сборка JavaScript и npm для приложения не нужны.
Тест `frontend/tests/browser.cjs` использует Playwright
с перехваченным API: автоматический показ PNG/описания/объектов, 404/ошибку/повтор/пустой результат,
совместимые и пустые уточнения, безопасный вывод текста модели, остановку опроса после
успеха/ошибки/новой загрузки, таймауты, автоматическую запись выбранной папки
до/после получения результата (mock File System Access API), ошибки разрешения/записи
с предложением ZIP, мобильную ширину и Swagger/CSP с mock CDN.
`FRONTEND_URL` задаёт адрес (по умолчанию теста — localhost:8098), `NODE_PATH` — путь
к внешнему Playwright, `PLAYWRIGHT_CHROMIUM_EXECUTABLE` — уже установленный Chromium.
Для localhost:8098 нужен собранный nginx frontend (включая маршрут `/api/docs`):
`docker run --rm -p 127.0.0.1:8098:80 kurs4-frontend:v1`.

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
