# Инструкция по установке и использованию

Программное средство детекции объектов и аннотирования изображений
с использованием искусственных нейронных сетей.

Дата подготовки: 28 октября 2026 года. Репозиторий:
https://gitlab.com/tatarov-k/img_ii, ветка `dev`.

## 1. Назначение и поддерживаемые данные

Приложение принимает фотографию, обнаруживает объекты, показывает их классы
и рамки, формирует описание каждого объекта и всей сцены. Результаты доступны
в браузере, через HTTP API и в скачиваемом комплекте файлов.
Используются готовые веса; обучение и дообучение для запуска не требуются.

| Вход | Поддержка | Обработка |
|---|---|---|
| JPEG / JPG | Да | Проверка, учёт EXIF orientation, преобразование в RGB PNG |
| PNG | Да | Проверка и преобразование в RGB PNG |
| PDF, DOCX, XLSX, видео | Нет | Перед загрузкой требуется отдельно получить JPEG или PNG |

Ограничения по умолчанию: один файл до 10 MiB, не более 25 миллионов пикселей.
Расширение файла само по себе не подтверждает корректность изображения.
Датасеты COCO и ImageNet для обычной обработки собственных фотографий не нужны.

## 2. Модели и схема обработки

| Этап | Модель | Вход | Результат |
|---|---|---|---|
| Детекция | RF-DETR Small | Нормализованное изображение | COCO-класс, bbox и object_id каждого объекта |
| Аннотирование сцены | Florence-2-base-ft, `<DETAILED_CAPTION>` | Полное изображение без рамок | Исходное описание annotation.raw_caption |
| Описание объектов | Тот же экземпляр Florence, `<CAPTION>` | Каждая вырезка с отступом 10% | classification.objects: object_id и object_caption |

Детектор и аннотатор сцены работают параллельно. Когда завершены обе ветки,
Florence последовательно описывает вырезки. Вся вырезка передаётся штатному
процессору Florence; дополнительной центральной обрезки нет. Описание объекта —
свободный текст, а не подтверждённый подтип COCO-класса. ViT в текущем приложении
не загружается. Имена, ревизии и контрольные суммы закреплены в
`backend/checkpoints.json`.

Приложение состоит из пяти контейнеров:

| Сервис | Назначение |
|---|---|
| frontend | Статический интерфейс nginx; прокси /api/ к backend |
| backend | FastAPI: проверка и сохранение входа, публикация image_id в RabbitMQ, выдача результатов |
| rabbitmq | Доставка image_id worker; подтверждение публикации |
| worker | Celery, два постоянных модельных процесса, обработка одного изображения одновременно |
| db | PostgreSQL: одна запись processed_images после готовности файлов |

FastAPI не запускает модели. Сначала worker атомарно публикует комплект файлов,
затем сохраняет результат в БД. При готовых файлах и сбое commit повторная доставка
восстанавливает запись без повторного инференса. Ошибки модели видны в логах worker;
отдельных таблиц заданий и endpoint статуса нет.

## 3. Требования к среде

Команды ниже выполняются в Linux/Bash. Для основной рабочей машины используются
Windows с WSL2 Ubuntu, Docker Engine с Compose v2 и доступом к NVIDIA GPU из WSL.
GPU-дополнение `compose.app.gpu.yaml` рассчитано именно на WSL и устройство `/dev/dxg`;
оно не является универсальной настройкой GPU для обычного Linux.

Оборудование рабочей машины: RTX 4050 Laptop, 6 ГБ VRAM; Intel Core i5-12450H;
16 ГБ RAM хоста, около 7,6 GiB доступно WSL. Это описание среды разработки,
а не гарантия достаточности памяти для любой нагрузки. Worker проверяет запас
памяти до приёма изображений. Для первой сборки и загрузки весов нужен интернет;
дальше модельный runtime использует локальные веса.

Проверить инструменты и свободные ресурсы:

```bash
git --version
docker --version
docker compose version
docker info
free -h
df -h .
ls -l /dev/dxg
/usr/lib/wsl/lib/nvidia-smi
```

Если Docker недоступен или GPU не виден из WSL, сначала исправить окружение.
Не продолжать запуск при ошибке предыдущей команды. Первая сборка worker
скачивает зависимости PyTorch/CUDA и может занять значительное время и место.

## 4. Получение исходного кода

Для новой установки:

```bash
git clone --branch dev https://gitlab.com/tatarov-k/img_ii.git
cd img_ii
git status --short --branch
```

На основной машине уже существует `/home/mikhail/kurs4`; повторное клонирование
для работы в этой копии не требуется. Далее команды выполняются из корня проекта.

| Каталог или файл | Содержание |
|---|---|
| backend/ | API, Celery, хранение, модели БД, миграции и тесты |
| frontend/ | HTML, CSS, JavaScript, nginx и браузерные тесты |
| compose.app.yaml | Пять сервисов приложения |
| compose.app.gpu.yaml | Доступ worker к GPU в WSL |
| scripts/ | Подготовка ресурсов и отдельные научные сравнения |
| detection/, classification/, annotation/ | Конфигурации, зависимости и материалы сравнений |
| reports/ | Сохранённые отчёты и результаты исследований |
| README.md | Подробное описание текущего приложения |

В Git не включаются веса, датасеты, `.secrets`, окружения, Docker volumes
и пользовательские изображения. После клонирования их требуется подготовить отдельно.

## 5. Локальные настройки и сборка

### Шаг 1. Создать настройки без перезаписи существующих

```bash
mkdir -p .secrets
python3 - <<'PY'
import os
import secrets
from pathlib import Path

path = Path('.secrets/app.env')
if not path.exists():
    content = Path('backend/app.env.example').read_text()
    content = content.replace('REPLACE_WITH_RANDOM_HEX', secrets.token_hex(24))
    content = content.replace('REPLACE_WITH_ANOTHER_RANDOM_HEX', secrets.token_hex(24))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        stream.write(content)
PY
```

Пароли PostgreSQL и RabbitMQ и соответствующие URL должны совпадать.
Файл `.secrets/app.env` хранится только локально. Для существующих volumes
нельзя просто сгенерировать новые пароли вместо действующих.

### Шаг 2. Проверить конфигурацию и собрать образы

```bash
docker compose -p kurs4-app -f compose.app.yaml --profile processing config --quiet
docker compose -p kurs4-app -f compose.app.yaml --profile processing build backend worker frontend
```

Эта сборка не запускает инференс. `compose.yaml` и `compose.gpu.yaml` относятся
к научным сравнениям; для запуска приложения используются файлы с `.app` в имени.

## 6. Подготовка двух выбранных моделей

Если проект переносится с рабочей машины, можно скопировать уже проверенные
каталоги `detection/models/rf_detr_small/` и
`annotation/models/florence2_base_ft/` целиком, включая конфигурации, процессор,
токенизатор и локальный Python-код Florence. Одного файла весов недостаточно.
Не заменять закреплённый checkpoint произвольной новой версией.

Для новой установки после сборки worker команда ниже вызывает существующие
функции `scripts/prepare.py`: скачивает только две выбранные модели и проверяет
их файлы, без загрузки датасетов, запуска моделей и научных сравнений.
Работающий worker перед подготовкой остановить: он удерживает общий inference-lock.

```bash
mkdir -p detection/models annotation/models storage/locks
docker run --rm -i --user "$(id -u):$(id -g)" \
  --entrypoint /opt/annotation/bin/python \
  -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 -e HF_HOME=/tmp/hf \
  -v "$PWD:/workspace" -w /workspace kurs4-worker:v1 - <<'PY'
from scripts.prepare import (
    download_annotation_models,
    download_detection_models,
    file_lock,
    read_json,
    verify_weights,
    write_json,
    ROOT,
)

with file_lock(ROOT / 'storage/locks/inference.lock'):
    # Git contains manifests, but not the model files themselves.
    for direction, model_id in [
        ('detection', 'rf_detr_small'),
        ('annotation', 'florence2_base_ft'),
    ]:
        path = ROOT / direction / 'models/manifest.json'
        if path.exists():
            manifest = read_json(path)
            entry = manifest.get('models', {}).get(model_id)
            if entry and any(
                not (path.parent / model_id / item['path']).is_file()
                for item in entry.get('files', [])
            ):
                del manifest['models'][model_id]
                write_json(path, manifest)
    download_detection_models('rf_detr_small')
    download_annotation_models(['florence2_base_ft'])
    verify_weights('detection', 'rf_detr_small')
    verify_weights('annotation', 'florence2_base_ft')
PY
```

Источники и ревизии берутся из `detection/configs/models.json` и
`annotation/configs/models.json`; runtime дополнительно сверяет файлы с
`backend/checkpoints.json`. При несовпадении SHA-256 запуск прекращается.
Команда требует сетевого доступа к закреплённым источникам.
Для отсутствующих файлов выбранных моделей она сбрасывает только их локальные
записи manifests перед скачиванием: в Git сохранены метаданные прежней подготовки,
но сами веса отсутствуют. Готовые файлы проверяются без такого сброса.
Общий CLI `scripts/prepare.py ... --download` подготавливает также ресурсы
сравнений, поэтому для установки только приложения выше выбраны отдельные функции.

## 7. Первый запуск приложения

### Шаг 1. Запустить БД и брокер

```bash
docker compose -p kurs4-app -f compose.app.yaml up -d --wait db rabbitmq
```

### Шаг 2. Применить миграции

Для новой пустой БД:

```bash
docker compose -p kurs4-app -f compose.app.yaml run --rm backend \
  alembic -c backend/alembic.ini upgrade head
```

Для существующей установки сначала выполнить раздел 11 о резервном копировании.
Миграция `d20f4a6b819c` удаляет прежние `images`, `jobs`, `model_results`;
она необратима. В актуальной схеме остаются `processed_images` и `alembic_version`.
Замена описателя вырезок на Florence сама по себе новой миграции не требует.

### Шаг 3. Запустить API, интерфейс и worker

```bash
docker compose -p kurs4-app -f compose.app.yaml -f compose.app.gpu.yaml \
  --profile processing up -d backend worker frontend
docker compose -p kurs4-app -f compose.app.yaml --profile processing ps
docker compose -p kurs4-app -f compose.app.yaml --profile processing logs --tail 100 worker
```

Этот шаг загружает и прогревает настоящие модели, затем начинает обрабатывать
сообщения очереди. Статус контейнера сам по себе не означает готовности моделей:
проверить логи worker. При нехватке RAM/VRAM приём изображений не начинается.

Открыть http://localhost:8080. Swagger через frontend:
http://localhost:8080/api/docs; напрямую через backend: http://localhost:8000/docs.
Swagger использует CDN jsDelivr и требует доступа к нему. Сервисы публикуются
на localhost; БД и RabbitMQ не открывают порты на хост.

## 8. Работа пользователя

1. Открыть интерфейс и выбрать JPEG/PNG.
2. При необходимости нажать «Выбрать папку» и разрешить браузеру запись.
3. Нажать «Отправить». Не закрывать вкладку во время ожидания и автосохранения.
4. Дождаться автоматического показа изображения, таблицы и описания сцены.
5. Нажать строку/номер в таблице либо объект на изображении для выделения.
6. Получить копию в выбранной папке либо нажать «Скачать ZIP».

На большом экране загрузка и таблица расположены слева, изображение и
«Результат аннотирования» — справа. На узком экране блоки идут вертикально.
Таблица содержит «Номер», «Класс», «Описание объекта». Все рамки видны постоянно:
обычные красные, выбранная синяя. На пересечении выбирается наименьшая рамка.
Повторное нажатие, пустое место изображения или Escape снимает выделение.

Браузер повторяет GET готового результата через 2 секунды после 404.
Один запрос ограничен 15 секундами, всё ожидание — 40 минутами. При ошибке или
таймауте доступно «Повторить ожидание». Новый файл прекращает прежнее ожидание;
повторная отправка создаёт новый image_id.

Описание сцены и объектов может быть на английском. Интерфейс скрывает ответ
`unanswerable` и убирает из начала описания сцены «In this image I can see» /
«In this image we can see», начиная оставшийся текст с заглавной буквы.
Исходные ответы в БД, API, JSON и скачиваемых файлах сохраняются без изменений.
Старые комплекты с ImageNet-классами доступны, но старый label не выдаётся
за описание объекта: соответствующая колонка пуста.

## 9. Сохранение и формат результатов

Основная копия всегда сохраняется сервером в Docker volume `image_data`:

```text
results/<image_id>/
  original
  normalized.png
  detection.png
  results.json
  crops/
    <object_id>.png
```

| Элемент | Содержание |
|---|---|
| original | Исходные байты загруженного JPEG/PNG |
| normalized.png | RGB-изображение с учётом EXIF orientation |
| detection.png | Изображение с COCO-рамками и классами, без номеров объектов |
| results.json | Ответы моделей, object_id, bbox, checkpoint и список файлов crops |
| crops/ | PNG-вырезки с отступом 10%; при отсутствии объектов папка пустая |

В браузере рамки накладываются на normalized.png по сохранённым bbox.
Копия на компьютере пользователя создаётся автоматически после показа результата:
браузер записывает подпапку `<image_id>` в выбранной папке. Выбор папки уже после
обработки сразу начинает запись. Для этого нужны File System Access API и
защищённый контекст, например localhost в поддерживающем Chromium-браузере.
При отсутствии поддержки, разрешения или ошибке записи доступен ZIP;
результат остаётся видимым. Кнопок «Показать результат» и «Сохранить копию» нет.

Автокопия и ZIP содержат основные файлы и crops. Старые комплекты без вырезок
доступны без повторной обработки. Папка, выбранная в браузере, не передаётся серверу.
Постоянные volumes не входят в Git и требуют отдельного резервного копирования.

## 10. Работа через API

| Метод и путь backend | Назначение |
|---|---|
| POST /images | Multipart-поле file; 202 после подтверждения RabbitMQ, image_id в ответе |
| GET /images/{image_id} | Готовые результаты; 404 до готовности |
| GET /images/{image_id}/files/{filename} | Основной файл или crops/<object_id>.png |
| GET /images/{image_id}/archive | Полный ZIP-комплект |
| GET /health | Доступность RabbitMQ для приёма загрузок |

Через frontend к путям добавляется `/api`. Health не проверяет модели и БД.
Пример для уже запущенного приложения; путь к изображению заменить своим:

```bash
curl --fail-with-body -F 'file=@/absolute/path/to/image.jpg' http://localhost:8000/images
# IMAGE_ID заменить UUID из ответа POST.
curl --fail-with-body http://localhost:8000/images/IMAGE_ID
curl --fail-with-body -o result.zip http://localhost:8000/images/IMAGE_ID/archive
```

404 означает отсутствие готового комплекта, а не процент выполнения.
В API нет отдельного endpoint статуса или ошибок инференса; при длительном
отсутствии результата проверить логи. При потере подтверждения публикации нельзя
считать, что брокер точно не принял сообщение: автоматический повтор POST может
создать повторную обработку с другим UUID.

## 11. Обновление, резервное копирование и остановка

Перед обновлением прекратить загрузки и дождаться завершения очереди старым worker.
Особенно важно для старой версии, где сообщение содержит job_id вместо image_id.
Затем остановить API и worker, сохранив БД и основной комплект изображений:

```bash
docker compose -p kurs4-app -f compose.app.yaml --profile processing stop backend worker
mkdir -p storage/backups
backup_stamp=$(date +%Y%m%d-%H%M%S)
docker compose -p kurs4-app -f compose.app.yaml exec -T db \
  sh -c 'exec pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' \
  > "storage/backups/db-$backup_stamp.dump"
docker compose -p kurs4-app -f compose.app.yaml run --rm --no-deps -T backend \
  tar -C /var/lib/kurs4/images -czf - . \
  > "storage/backups/images-$backup_stamp.tar.gz"
```

Проверить успешное завершение команд и размер архивов; копию хранить также
вне диска проекта. Не применять удаляющие миграции без проверенного дампа.
Затем обновить код, собрать образы и выполнить шаги раздела 7.
На рабочей копии перед обновлением проверить `git status`, сохранив локальные правки.
Не удалять volumes и не менять имя Compose-проекта при обычном перезапуске.

Обычное возобновление уже настроенной версии:

```bash
docker compose -p kurs4-app -f compose.app.yaml up -d --wait db rabbitmq
docker compose -p kurs4-app -f compose.app.yaml -f compose.app.gpu.yaml \
  --profile processing up -d backend worker frontend
```

Остановка с сохранением данных:

```bash
docker compose -p kurs4-app -f compose.app.yaml --profile processing stop
```

`down -v` удаляет постоянные volumes и для остановки не нужен.
Перед научными сравнениями worker требуется остановить: приложение и сравнения
используют один `storage/locks/inference.lock`.

## 12. Типичные проблемы

| Признак | Что проверить |
|---|---|
| Страница не открывается | docker compose ps, порт 8080, логи frontend/backend |
| POST возвращает 503 | Доступность и healthcheck RabbitMQ, локальные настройки брокера |
| GET долго возвращает 404 | Логи worker, успешный прогрев, память, доступность БД и наличие файлов весов |
| Worker не начинает обработку | SHA-256, пути двух моделей, /dev/dxg, RAM/VRAM, занятый inference-lock |
| Ошибка проверки весов | Восстановить закреплённые файлы; не отключать проверку SHA-256 |
| Ошибка/таймаут ожидания | Проверить логи; затем «Повторить ожидание» для того же image_id |
| Не выбирается папка | Поддержка API браузером, localhost, разрешение на запись; использовать ZIP |
| Ошибка автокопии | Выбрать папку снова или скачать ZIP; основная копия остаётся на сервере |
| Пустое описание | Нет объектов, unanswerable либо старый комплект без object_caption |
| Swagger пустой | Доступ к CDN jsDelivr; правильный адрес /api/docs |
| Новое поведение UI не видно | Обновить вкладку после пересборки frontend |

Команды просмотра состояния и логов:

```bash
docker compose -p kurs4-app -f compose.app.yaml --profile processing ps
docker compose -p kurs4-app -f compose.app.yaml --profile processing logs --tail 100 backend worker frontend
```

Для длительного просмотра добавить `-f`; Ctrl+C завершает просмотр логов.

## 13. Научные сравнения и ограничения

Сохранённая научная часть включает отдельные сравнения детекторов, классификаторов
и аннотаторов по четыре готовые модели. Эти скрипты не являются обязательной
частью запуска веб-приложения. Участники и результаты сохранены в репозитории;
замена ViT на Florence в приложении не изменяет прежние измерения классификации.

| Направление | Данные и актуальный объём | Где читать |
|---|---|---|
| Детекция | COCO; ранее завершённые три повтора | detection/README.md и reports/comparisons/ |
| Классификация | ImageNet; первые два полных повтора всех четырёх моделей | classification/README.md |
| Аннотирование | 500 COCO-изображений, один проход каждой модели | annotation/README.md |

Новые доверительные интервалы и человеческая оценка в текущий объём не входят.
Для аннотаторов внешний набор не использован; это ограничение сравнения на COCO.
Старые результаты и отчёты сохраняются. Новые исследовательские запуски
выполняются только по отдельному поручению, с результатами в `reports/runs/`.

Описания моделей могут содержать ошибки и выдуманные детали. Словесное описание
вырезки не подтверждает правильность подтипа по пикселям. Система обрабатывает
одно изображение одновременно и не предоставляет поиск по архиву, OCR,
обработку офисных документов или видео.

При подготовке этой инструкции модели и GPU benchmark не запускались.
Исторические проверки перечислены в README и reports/verification/;
CPU/mock-тесты не подтверждают качество или скорость реального инференса.

## 14. Основа инструкции

Фактическое поведение сверено с README.md, PROJECT_PLAN.md, compose.app.yaml,
backend/checkpoints.json, кодом API, worker и frontend. Образцы «ИНСТРУКЦИЯ.docx»
и «Отчёт.docx» использованы для структуры пошагового руководства и сводных таблиц.
Их примеры OCR, обучения, PDF/DOCX и сторонних конвейеров не являются требованиями
к этому приложению и в его возможности не включены.
