# Сравнение классификаторов

Участники закреплены по плану: ResNet-50 `IMAGENET1K_V2`, EfficientNetV2-S,
ConvNeXt-Tiny и ViT-B/16 `IMAGENET1K_V1`. Готовые головы ImageNet-1K,
без обучения. Это представители архитектур по указанию пользователя;
строгий мировой рейтинг популярности не заявляется. Zero-shot не используется.

Команды выполняются в Bash из `/home/mikhail/kurs4`. Запуск через Linux Docker
Compose; GPU-профиль WSL выбирается скриптом так же, как для детекции.
Лимит контейнера 6 GiB, batch 1, FP32, одна загруженная модель во всей системе.
CPU — отдельный профиль с отдельными результатами.

## Подготовка весов

```bash
scripts/classification.sh build
scripts/classification.sh download --dry-run
scripts/classification.sh download
scripts/classification.sh verify
scripts/classification.sh smoke --output-dir reports/classification/my-smoke
```

Веса — `classification/models/`. Четыре файла весов вместе занимают
650 009 420 байт; дополнительные JSON/лицензии малы. Источники — официальный
сервер PyTorch; точные URL, версии, размеры и SHA-256 закреплены в
`classification/configs/models.json`. Загрузка потоковая, `.part` сохраняются
для возобновления. `manifest.json` отражает фактическую проверку файлов.
Архитектура создаётся с `weights=None`, затем читается локальный state_dict
через `weights_only=True`: скрытого скачивания в инференсе нет.

Torch 2.7.1 CUDA 12.8 и torchvision 0.22.1 поставляются закреплённым базовым
образом. Собственный `requirements.lock.txt` фиксирует проверенное окружение;
системный Python WSL не используется для ML. Препроцессинг берётся из точного
enum весов и сохраняется рядом с ними. Все модели проверяют один порядок
1000 названий и synset: названия из torchvision, synset из закреплённого
справочника timm v1.0.15, с SHA-256 источника. Все четыре головы должны совпасть.

Smoke проходит без сети на одном существующем изображении COCO. Для каждого
участника сохраняются исходное изображение, top-5, timings, RAM/VRAM и
native-parity с официальным transform. GT для ImageNet у этого изображения
нет: smoke не считает и не подставляет Top-1/Top-5 accuracy.

## Доступ к ImageNet

Источник — [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k),
полная validation-часть, 50 000 изображений, 1000 классов по 50 изображений.
Нужно принять условия на сайте под своим аккаунтом, затем создать токен Read
либо ограниченный токен чтения этого dataset. Токен не передавать в чате.

В WSL сохранить токен интерактивно (при запросе вставить токен с сайта и нажать Enter):

```bash
cd /home/mikhail/kurs4
mkdir -p .secrets
chmod 700 .secrets
read -rsp 'Вставьте HF токен: ' task_hf_token
printf '%s' "$task_hf_token" > .secrets/hf_token
unset task_hf_token
chmod 600 .secrets/hf_token
```

Ввод скрыт; после вставки нажать Enter. Файл доступен только владельцу;
`.secrets/` исключена из Git и Docker build context. Только сервис tools
получает read-only mount токена; benchmark не получает его и работает без сети.
После загрузки токен можно отозвать на сайте. Пользовательские изображения
никуда не отправляются: сетевой этап получает только файлы датасета и весов.

```bash
scripts/classification.sh fetch-data --dry-run
scripts/classification.sh fetch-data
scripts/classification.sh import-data
```

До загрузки закреплены revision датасета и 14 validation-shards, около 6,7 GB
parquet. Train/test не скачиваются. При авторизованном доступе сначала
получаются официальные LFS SHA-256, план сохраняется до загрузки; затем файлы
проверяются по SHA-256 и размеру. Полные gated-хеши без доступа не выдумываются.
Исходные shards остаются в `data/imagenet/source/`, исходные байты JPEG без
перекодирования импортируются в `data/imagenet/val/<synset>/`.
Данные читаются небольшими Arrow batches; все картинки не помещаются в RAM.
Для parquet и распакованных изображений проверяется свободное место с запасом.

Импорт проверяет порядок эталонных меток по всем 1000 названиям, 50 000
оригинальных filename, формат JPEG и размер изображения. Итоговая проверка:
ровно 50 изображений на каждый класс. Manifest включает image_id, target_index,
synset, размер, SHA-256 и происхождение. EXIF не меняет сохранённые пиксели;
используется штатный torchvision transform без автоматического поворота.
Повтор не перезаписывает изменившийся manifest.

Для уже полученной локальной ImageNet validation-папки с synset-каталогами:

```bash
scripts/classification.sh prepare --images-dir data/imagenet/val
```

## Протокол и сравнение

`evaluation5000.json` заранее фиксирует по 5 изображений на каждый класс,
seed 42; отдельный dev не выделяется, настройки не подбираются. Это внутренняя
evaluation-подвыборка официальной validation, не закрытый test ImageNet.

```bash
scripts/classification.sh benchmark --output-dir reports/classification/evaluation5000-v1
scripts/classification.sh report --runs-dir reports/classification/evaluation5000-v1 --output-dir reports/comparisons/classification-evaluation5000-v1
```

Полный ImageNet был начат отдельным экспериментом с исходным repeats=3.
По актуальному указанию от 09.10.2026 он завершён после второго круга.
Ниже команда пересборки основного отчёта из первых двух повторов; старый benchmark или `completion-v1/run.sh` повторно не запускать:

```bash
scripts/classification.sh report --runs-dir reports/classification/validation50000-v1 --output-dir reports/comparisons/classification-validation50000-v1 --repeats 2
```

20 прогревов, 4 CPU threads; два повтора полного ImageNet по актуальному указанию.
Уже выполненные evaluation5000 и GT/detector-crops сохраняют три повтора.
Исходный repeats=3 в старых configs/manifests не переписывается; report отдельно
фиксирует выбор первых двух повторов, не включая сохранённый третий ResNet-50.
TF32 и cuDNN autotuning выключены;
deterministic algorithms и CUBLAS_WORKSPACE_CONFIG фиксируются.
Порядок моделей сдвигается между повторами. Модельный дочерний процесс
обрабатывает весь manifest и завершается до старта следующего. Родитель не
загружает веса. Используется общий `storage/locks/inference.lock` детекции;
lock наследуется ребёнком, поэтому SIGKILL родителя не разрешает вторую модель.
Перед benchmark будущий application worker должен быть остановлен.

Top-1, Top-5, Macro-F1 и accuracy по 1000 классам считаются по первому повтору.
Sparse confusion CSV хранит все ненулевые клетки матрицы, включая диагональ.
Все top-5 включают полный классификационный словарь, индекс, synset, имя и
softmax. Softmax не является оценкой неизвестного объекта или текста caption.

По последнему указанию пользователя новые доверительные интервалы не рассчитываются.
Основной отчёт по умолчанию не читает bootstrap.json и не включает CI в CSV/Markdown.
Уже рассчитанный до этого указания bootstrap сохранён в исходных артефактах;
инструмент остаётся доступным для отдельной явно запрошенной проверки.
Полная validation и evaluation5000 пересекаются: это не две независимые оценки.

Холодная загрузка — создание модели в новом процессе (без очистки page cache),
после проверки файлов и environment. Чтение/декодирование, preprocessing и
transfer, forward, postprocessing, запись/fsync измеряются отдельно.
CUDA синхронизируется на границах модельных стадий. p50/p95 и throughput
относятся к их сумме, исключают чтение/запись и холодную загрузку.
VRAM allocated/reserved — пики PyTorch, включая модель, не полная память GPU;
RAM — peak RSS процесса с загрузкой. Environment сохраняет реальные Linux RAM,
cgroup limit, CPU/GPU, версии, commit/dirty status и actual image ID.

Разные штатные входы учитываются: EfficientNetV2-S 384×384, остальные
224×224 с разными resize. Обучающие рецепты авторов различаются; эксперимент
сравнивает готовые checkpoint, не чистый эффект архитектуры. Авторские
показатели не подставляются в локальные таблицы.

## Артефакты, resume и проверки

Каждый повтор сохраняет config, model/dataset/environment manifests,
`native_config.json`, `run.json`, `samples.jsonl`, `predictions.jsonl`,
`timings.jsonl`, `write_timings.jsonl`, `metrics.json`, `errors.jsonl`, `process.log`.
В корне эксперимента — `experiment.json` и `source_snapshot.tar.gz`.
Durable `samples.jsonl` — источник resume; недописанная последняя строка
отбрасывается, дубликаты/чужие image_id/изменённые эталоны отклоняются.
Возобновление требует совпадения конфига, кода, контейнера, весов и данных:

```bash
scripts/classification.sh benchmark --output-dir reports/classification/evaluation5000-v1 --resume
docker compose -f compose.classification.yaml run --rm checks -m pytest classification/tests
docker compose -f compose.classification.yaml run --rm checks -m ruff check .
docker compose -f compose.classification.yaml run --rm checks -m ruff format --check .
```

GPU smoke — отдельная проверка реальных моделей; CPU-тесты не скачивают веса.
Ошибки/OOM и partial-артефакты сохраняются; CSV всегда содержит четыре строки.
До реального прогона соответствующие числовые результаты отсутствуют.
08.10.2026 завершён предварительный evaluation5000: четыре модели × три повтора ×
5000 изображений. По указанию пользователя validation50k отложен.
После изменения исходников возобновление старого эксперимента требует его
сохранённого source_snapshot; для новых прогонов использовать новую папку.

Модуль реализует whole-image ImageNet и отдельные [GT/detector-crops](crops/README.md).
Mapping, matching и диагностические метрики реализованы; оба crop-режима завершили
по три полных прохода каждой модели. Результаты и команды — в crops/README.md.
Порог отказа не калиброван на dev. Оценки crops не смешиваются с ImageNet
Top-1/Top-5; приложение и общий pipeline остаются отдельными этапами.

Исторически сначала был выполнен только полный ResNet-50 с `--model resnet50`.
Фильтр `--model` выбирает исполняемую модель из зафиксированных четырёх,
не меняя протокол качества и dataset identity. Старые команды запуска не являются
инструкцией повторять уже завершённый benchmark.

До продолжения 09.10.2026 полный ResNet-50 завершил все три повтора на 50 000 изображений.
[Итоги](../reports/comparisons/classification-validation50000-v1/classification.md):
Top-1 80.854%, Top-5 95.438%, Macro-F1 0.80632. Тогда остальные full-модели были not_run.
Число обработок за три повтора — 150 000, разных изображений — 50 000.

По задаче 09.10.2026 остальные checkpoint завершены через resume с прежним
source snapshot и Docker image ID. После изменения расписания закончены первые
два прохода всех четырёх моделей; третий круг остальных моделей не запускался.
Сохранённые три полных ResNet-50 не повторялись; третий исключён из основной
сравнительной таблицы. Контроллер `reports/classification/completion-v1/finish_round2.py`
завершил vit_b_16_repeat2, выполнил CPU-evaluate и прекратил приостановленный
планировщик до запуска третьего круга. `stop_after_repeat2.json` имеет статус
`complete_after_repeat2`; исходный experiment/config с repeats=3 сохранён.
Все выбранные проходы и идентичность предсказаний между повторами проверены.
Для пересборки отчёта без инференса:

```bash
scripts/classification.sh report --runs-dir reports/classification/validation50000-v1 --output-dir reports/comparisons/classification-validation50000-v1 --repeats 2
```

Независимая верификация уже выполнена; её команда:

```bash
docker compose -f compose.classification.yaml run --rm checks scripts/verify_classification.py --runs-dir reports/classification/validation50000-v1 --output reports/verification/classification-validation50000-v1.json --repeats 2
```

Аналогичные проверки уже выполнены для `gt-crops-v1` и `detector-crops-v1` по всем
трём готовым повторам, без `--repeats 2`. После трёх
проверенных режимов итоговый описательный анализ строится без нового инференса:

```bash
docker compose -f compose.classification.yaml run --rm checks scripts/analyze_classification.py
```

Результат — `Описание результатов классификации.md` и analysis.md в трёх папках
сравнения. Для графика опционально использовать `--plots` в CPU checks образа
compose.detection.yaml, где уже установлен matplotlib; детектор не загружается.
Новые доверительные интервалы не считать во всех сравнениях проекта.
Фактический статус и последовательность продолжения — [handoff](../docs/classification_handoff.md).
