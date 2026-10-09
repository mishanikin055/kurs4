# Состояние работы для продолжения

Обновлено 09.10.2026. Репозиторий `/home/mikhail/kurs4` в WSL2,
ветка `codex/classification-completion`, origin `mishanikin055/kurs4`.
Прочитать [AGENTS.md](../AGENTS.md) и [classification.md](../classification.md);
команды — в classification/README.md. PROJECT_PLAN.md использовать только для нужного
раздела, если тематических инструкций недостаточно; весь план читать не требуется.

## Новая задача 09.10.2026: завершение классификации

Эта секция имеет приоритет над историей ниже. Пользователь поручил закончить
сравнение классификаторов, затем создать субагента с отдельным контекстом и
поручить ему сравнение моделей аннотирования по AGENTS.md / annotation.md.
Субагента запускать только после проверки и оформления полного анализа
классификаторов. Новые доверительные интервалы запрещены во **всех** сравнениях.

Запущен последовательный `reports/classification/completion-v1/run.sh`:
GT-crops → detector-crops → ImageNet validation50k с `--resume`. Три прежних
полных ResNet-50 пропускаются; все fingerprint-файлы совпали с source_snapshot,
контейнер и веса прежние. GT-crops завершены (4 × 3 × 1883); остальные этапы
проверять по run.json и Docker, не запускать второй benchmark параллельно.
Для повторного продолжения завершённых/частичных новых crops использовать
`--resume`; исходники benchmark не менять, пока идёт эксперимент.

Добавлены `scripts/verify_classification.py` и `scripts/analyze_classification.py`.
Первый проверяет полный четырёхмодельный эксперимент без загрузки моделей и CI;
второй строит описательный анализ после проверки всех трёх режимов. Новые файлы
не входят в fingerprint старого benchmark. CPU-тесты: 30 passed, Ruff check/format
пройдены; свежий офлайн GPU-smoke четырёх моделей — smoke-completion-v1.

Итоговые папки reports/comparisons/classification-{validation50000,gt-crops,detector-crops}-v1/.
Проверки сохранять как reports/verification/classification-{validation50000,gt-crops,detector-crops}-v1.json.
Команды: `docker compose -f compose.classification.yaml run --rm checks
scripts/verify_classification.py --runs-dir <raw-runs> --output <verification>`;
затем `docker compose -f compose.classification.yaml run --rm checks
scripts/analyze_classification.py`. Для графика имеется `--plots`, matplotlib
есть в CPU checks образа compose.detection.yaml; это не запуск детекторных моделей.

После завершения обновить статусы/README, поправить устаревшую фразу об интервалах
в classification/crops/report.py и перестроить crop-таблицы (не менять этот
fingerprint-файл во время crop-прогонов). Проверить отчёты, ограничения и выводы,
коммитить и push в существующий origin. Затем запустить ровно одного субагента
через collaboration.spawn_agent с fork_turns="none", рабочая папка та же.
Промпт по шаблону пользователя: «Начни делать сравнительный анализ моделей
аннотирования. Как делать — читай AGENTS.md и annotation.md по ссылке из него.
Доверительные интервалы не считать». Указать, что модели выполняются строго по одной,
классификация закончена, приложение в эту задачу не входит. Повторных GPU-прогонов
классификации ради правки отчёта не требуется.

## История прежних указаний и результатов

- Новое указание: запустить полный ImageNet50k только для первой модели, ResNet-50,
  три повтора по прежнему протоколу. Остальные три модели и полные crops отложены.
  Папка: reports/classification/validation50000-v1/. Статус читать из run.json
  соответствующих repeat-папок и process.log; фактическое завершение проверить.
- Новые доверительные интервалы не рассчитывать. Основной отчёт без CI;
  ранее выполненный bootstrap сохранён в исходных артефактах.
- Коммитить проверенные изменения и пушить в существующий origin.
  Авторизация GitHub сохранена Windows GCM; push обеих веток проверен. Секреты в чат не выводить.

## Завершено

- `3492a99`: детекция, четыре модели × три повтора × 4500 COCO test,
  таблицы/анализ reports/comparisons/detection.*. Ранние 500 были dev.
- `1c2b98c`: классификация, четыре готовых ImageNet1K головы, контейнеры и GPU smoke.
- `30e705a`: предварительные результаты, crops-модуль/smoke и --model;
  отправлен в origin/codex/classification-comparison. Детекция отправлена
  в origin/codex/detection-comparison; remote commit refs сверены с локальными.
- ImageNet validation полностью загружен: 14 HF-шардов, 50 000 исходных JPEG,
  50 на класс; SHA-256 и labels проверены. Revision
  `49e2ee26f3810fb5a7536bbf732a7b07389a47b5`, ILSVRC/imagenet-1k.
  Данные и токен игнорируются Git, benchmark работает офлайн без токена.
- Предварительный evaluation5000 завершён: 5 на класс, seed42,
  четыре модели × три повтора, FP32/batch1. Все 12 проходов успешны,
  предсказания каждого checkpoint идентичны между повторами.
  Результаты — reports/comparisons/classification-evaluation5000-v1/.
  Top1: ResNet 80.70%, EfficientNetV2S 83.96%, ConvNeXtTiny 82.60%, ViT 80.92%.
  Это оценка выбранных 5000, не полный ImageNet50k и не качество на crops.
- Прикладной модуль classification/crops/ реализован. Mapping 287 synsets →
  56 COCO категорий, остальные not_mappable. Подготовлены 500 одинаковых
  COCO test сцен, 1883 GT-crops, 2296 RF-DETR-Small crops, покрытие 55.81%.
  Пока выполнен только GPU-smoke по две вырезки каждого режима для каждой
  модели. Качество smoke отсутствует; полный сравнительный анализ не выполнен.
- Верификация: reports/verification/classification_runs.json и classification.md.

## Полный ResNet-50 завершён

Проверено 09.10.2026: все три resnet50_repeat1/2/3 имеют status=complete,
по 50 000 изображений, по одной попытке, errors.jsonl пусты. Контейнер и
модельные процессы завершились. Predicted top5/softmax совпадают между повторами.
Top1 0.80854 (40 427 / 50 000), Top5 0.95438 (47 719 / 50 000),
Macro-F1 0.8063235756082204. Проверены sample SHA-256, dataset/source hashes
и правильные ответы по всем сохранённым строкам.

Итоговые CSV/Markdown и анализ:
reports/comparisons/classification-validation50000-v1/.
Проверочный manifest: reports/verification/classification_validation50000.json.
Сырые данные/snapshots: reports/classification/validation50000-v1/.
Source commit запуска — 30e705a. Новых доверительных интервалов нет.
Остальные модели на полном validation — not_run; полные crops отложены.
Старую shell-сессию 37549 и контейнер не использовать как активные.
Автоматическая проверка resnet-50 удалена 09.10.2026 по указанию пользователя;
не создавать её повторно. Итоговые отчёты закоммичены и отправлены в origin.

## Авторизация GitHub

В .git/config настроен существующий Windows GCM (mingw64/bin), wincred,
username mishanikin055; аккаунт сохранён после одного browser login.
Для проверок/пуша без новых окон задавать GCM_INTERACTIVE=never и
GIT_TERMINAL_PROMPT=0 и передавать эти имена через WSLENV: Linux-переменные
не поступают в Windows-процессы автоматически. Значения секретов не выводить.
Новых browser login без необходимости не открывать. Повторные git push и
ls-remote уже успешно выполнены без окна входа.

## Воспроизводимость и ограничения

Точные модели/веса/native transforms — classification/configs/ и models/manifest.json.
Данные — data/imagenet/{evaluation5000,validation}.json; crops manifests —
reports/classification/crop-data-v1/. Raw predictions/timings/errors/environment,
experiment identity и source snapshots — reports/classification/; они вне Git.
Для старого resume нужен его source snapshot: последняя правка отчёта меняет
source fingerprint, завершённые результаты не требуют повторного инференса.
Новые эксперименты писать в новые папки.

Строго одна модель в RAM/GPU; следующий child только после окончания предыдущего.
Общий lock storage/locks/inference.lock с детекцией; runtime без сети,
read-only веса/данные. Docker image и характеристики машины записаны в environment.
GT и detector crops оцениваются отдельно. Сохраняется метка детектора;
softmax не доказывает правильность неизвестного объекта. Пороги score/margin
заранее заданы как диагностические; dev-калибровка отказа не выполнена.
COCO не проверяет породы/подтипы. Приложение, UI и аннотаторы ещё не реализованы.

## Просмотр одного изображения

reports/classification/smoke-v1/efficientnet_v2_s_repeat1/samples.jsonl — результат,
prediction.json рядом — удобная JSON-копия первой записи. original.png — вход.
Классификатор целого изображения выдаёт top5, без рамок и общей текстовой аннотации.

## Продолжение после нового указания

Проверить git status и docker ps; не запускать повтор работающего эксперимента.
Для полного ImageNet — scripts/classification.sh benchmark с validation.json;
для crops — scripts/classification_crops.sh, команды в classification/crops/README.md.
Для выбора только одной модели имеется --model resnet50 (или ID другой модели).
Фильтр изменяет расписание исполнения, не состав участников/метрики протокола;
позже --resume с другим --model может заполнить ту же зафиксированную папку.
Не вызывать bootstrap. После прогона — report, анализ, проверки, commit и push.

Новый чат: «Прочитай docs/classification_handoff.md и продолжи с учётом последних
ограничений пользователя». Полный старый чат и длинные логи переносить не нужно.
