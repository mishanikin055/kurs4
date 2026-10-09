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

**Последнее изменение пользователя от 09.10.2026:** полный ImageNet закончить
после второго круга (4 × 2 × 50 000), для аннотаторов — ровно один полный прогон
каждой модели. Это имеет приоритет над прежним требованием трёх кругов и текстом
старого heartbeat. Сохранённые третьи повторы не удалять. В основной ImageNet-таблице
выбирать одинаковые первые два повтора, включая только первые два ResNet-50.

## Итог классификации на 09.10.2026

Все выбранные проходы завершены и проверены:

- ImageNet: 4 × 2 × 50 000, предсказания совпадают между первыми двумя повторами;
- GT-crops: 4 × 3 × 1883, 500 сцен;
- detector-crops: 4 × 3 × 2296, 500 сцен, 1054 matched-supported для условной точности.

Сохранённые полные resnet50_repeat1/2/3 использованы без повторного инференса.
Третий ResNet-50 исключён из основной таблицы времени/ресурсов, но не удалён.
Исходный config/experiment identity с repeats=3 и source_snapshot сохранён.
Во время vit_b_16_repeat2 приостановлен только родительский планировщик;
контроллер `reports/classification/completion-v1/finish_round2.py` дождался всех
50 000 изображений, завершения worker и CPU-evaluate, затем прекратил планировщик
до старта новых третьих проходов. `stop_after_repeat2.json` имеет статус
`complete_after_repeat2`, контейнер удалён; ожидаемый exit137 оболочки относится
к прекращению планировщика, все выбранные run.json имеют status=complete без ошибок.
**Не запускать прежний run.sh или обычный benchmark --resume:** это вернёт третий круг.

Результаты: [Описание результатов классификации](../Описание%20результатов%20классификации.md),
`reports/comparisons/classification-{validation50000,gt-crops,detector-crops}-v1/`,
график `reports/comparisons/classification_quality_time.png`.
Проверки: `reports/verification/classification-{validation50000,gt-crops,detector-crops}-v1.json`.
Независимый пересчёт без моделей/CI — `scripts/verify_classification.py`;
для полного ImageNet `--repeats 2`, для готовых crops три повтора по умолчанию.
Пересборка whole-image отчёта — `scripts/classification.sh report ... --repeats 2`.
Анализ: `docker compose -f compose.detection.yaml run --rm checks
scripts/analyze_classification.py --plots`; CPU-образ содержит matplotlib,
детекторная модель не загружается. Report/analysis manifests сохраняют hashes
источников и явный выбор повторов отдельно от замороженного эксперимента.

EfficientNetV2-S: ImageNet Top-1 84.238%, лучший по качеству готового checkpoint.
ResNet-50: p50 10.224 ms, самый быстрый; ConvNeXt-Tiny: Top-1 82.514%, p50 10.821 ms.
ViT-B/16 лучший по Top-1/Top-5 на crops; на detector-crops лучший Macro-F1 у
EfficientNetV2-S. Замена категории при score≥0.5/margin≥0.1 ухудшает matched
accuracy у всех четырёх; метку детектора сохранять. Покрытие словаря 55.81%,
пропущено 829 supported GT; условную точность не выдавать за качество конвейера.
Отказ на dev не калиброван, породы/подтипы COCO не проверяет.

CPU-проверки: 32 pytest-теста классификации/crops, Ruff check/format пройдены.
Свежий офлайн GPU-smoke четырёх моделей — smoke-completion-v1, отдельная проверка
native parity. Таблицы и их выводы сформированы и независимо проверены субагентом
`classification_reports` с отдельным контекстом по последнему указанию пользователя.
Подтверждены все 12 строк, знаменатели, ресурсы и 27 хешей итогового manifest.
Устаревшая фраза об интервалах в classification/crops/report.py исправлена только
после окончания crops, их отчёты перестроены без нового инференса.

## Следующий этап

Сравнение аннотаторов делегировано субагенту annotation_comparison с отдельным
контекстом; отметка — reports/classification/completion-v1/annotation_delegation.json.
Четыре единственных test500-прохода завершены. После остановки субагента из-за
лимитов основной агент по новому поручению пользователя оформил автоматические
таблицы и отчёт. [Итог аннотаторов](../Описание%20результатов%20аннотирования.md),
[актуальная точка продолжения](annotation_handoff.md). Повторно субагента и
классификационные прогоны не запускать ради готовых отчётов.
Автоматическая проверка раз в10 минут удалена пользователем; не создавать её.

## История прежних указаний и результатов

- Новое указание: запустить полный ImageNet50k только для первой модели, ResNet-50,
  три повтора по прежнему протоколу. Остальные три модели и полные crops отложены.
  Папка: reports/classification/validation50000-v1/. Статус читать из run.json
  соответствующих repeat-папок и process.log; фактическое завершение проверить.
- Новые доверительные интервалы не рассчитывать. Основной отчёт без CI;
  ранее выполненный bootstrap сохранён в исходных артефактах.
- Коммитить проверенные изменения и пушить в существующий origin.
  Авторизация GitHub сохранена Windows GCM; push обеих веток проверен. Секреты в чат не выводить.

## Исторически завершено до продолжения

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
COCO не проверяет породы/подтипы. Приложение и UI ещё не реализованы; автоматическое сравнение аннотаторов завершено, человеческие оценки остаются pending (annotation_handoff.md).

## Просмотр одного изображения

reports/classification/smoke-v1/efficientnet_v2_s_repeat1/samples.jsonl — результат,
prediction.json рядом — удобная JSON-копия первой записи. original.png — вход.
Классификатор целого изображения выдаёт top5, без рамок и общей текстовой аннотации.

## Исторические команды продолжения (не запускать для завершённого сравнения)

Проверить git status и docker ps; не запускать повтор работающего эксперимента.
Для полного ImageNet — scripts/classification.sh benchmark с validation.json;
для crops — scripts/classification_crops.sh, команды в classification/crops/README.md.
Для выбора только одной модели имеется --model resnet50 (или ID другой модели).
Фильтр изменяет расписание исполнения, не состав участников/метрики протокола;
позже --resume с другим --model может заполнить ту же зафиксированную папку.
Не вызывать bootstrap. После прогона — report, анализ, проверки, commit и push.

Новый чат: «Прочитай docs/classification_handoff.md и продолжи с учётом последних
ограничений пользователя». Полный старый чат и длинные логи переносить не нужно.
