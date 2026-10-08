# Состояние работы для продолжения

Обновлено 08.10.2026. Репозиторий `/home/mikhail/kurs4` в WSL2,
ветка `codex/classification-comparison`, origin `mishanikin055/kurs4`.
Прочитать AGENTS.md, PROJECT_PLAN.md и classification/README.md.

## Последние указания пользователя

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

## Активный запуск

Full ResNet-50 запущен командой из classification/README.md.
Docker container: kurs4-benchmark-run-0c68c4ce1cd4; shell session текущего чата: 37549.
Первый повтор: running, более 3600 / 50 000 обработано на момент записи.
Все три повтора выполняются одним родительским расписанием; следующие модели
пропускаются фильтром --model resnet50. Source commit запуска — 30e705a.
Смотреть reports/classification/validation50000-v1/resnet50_repeat*/run.json
и process.log. **Не запускать второй GPU-процесс и не менять исходники из
source fingerprint, конфиг/manifest/контейнер до окончания.** Родитель после
трёх повторов сам генерирует comparison без доверительных интервалов.
Фактические результаты full брать только после status=complete во всех трёх.

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
