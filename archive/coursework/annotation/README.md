# Сравнение аннотаторов

Новая реализация — пять файлов в `scripts/annotation/`: `blip_base.py`,
`florence2_base_ft.py`, `qwen3_vl_2b.py`, `smolvlm2_500m.py` явно загружают
свои локальные Transformers checkpoint, готовят исходное RGB-изображение и
вызывают `generate`. `common.py` читает набор, проверяет SHA, запускает один
модельный дочерний процесс, измеряет ресурсы, вычисляет метрики и строит таблицы.
В `annotation/` остаются configs, игнорируемые models, Dockerfile и закреплённые
зависимости, тесты и необходимый сторонний код метрик `evaluation/` с источниками
и лицензионными уведомлениями. Подготовка всех направлений — `scripts/prepare.py`.

Сначала подготовить данные/веса по корневому [README](../README.md).
Runtime офлайн; автоматического скачивания нет. Четыре модели запускаются
последовательно с одним новым именем сравнения и одинаковым `--limit`:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml run --rm annotation scripts/annotation/blip_base.py --run annotation-smoke-v2 --limit 2
docker compose -f compose.yaml -f compose.gpu.yaml run --rm annotation scripts/annotation/florence2_base_ft.py --run annotation-smoke-v2 --limit 2
docker compose -f compose.yaml -f compose.gpu.yaml run --rm annotation scripts/annotation/qwen3_vl_2b.py --run annotation-smoke-v2 --limit 2
docker compose -f compose.yaml -f compose.gpu.yaml run --rm annotation scripts/annotation/smolvlm2_500m.py --run annotation-smoke-v2 --limit 2
```

Без `--limit` каждый скрипт выполняет один проход по 500 изображениям из
`annotation/configs/test500.json`. Полные сравнения уже завершены; ради
рефакторинга повторять их не требуется. Даже короткая проверка использует один
отдельный прогрев на dev-изображении. Batch 1, FP16 без offload/квантования,
greedy decoding, 96 новых токенов. BLIP/Florence — eager, Qwen/SmolVLM — SDPA.
Qwen: 65536–262144 pixels. Florence: закреплённый проверенный Microsoft remote
code из локального каталога, fast tokenizer и отключение DynamicCache для
совместимости tuple cache. Остальные модели `trust_remote_code=False`.

Результаты: `reports/runs/annotation/NAME/comparison.json`, `comparison.csv`,
`comparison.md`; у каждой модели — `run.json` и `samples.jsonl`. После каждого запуска и report-only компактная сравнительная таблица печатается
в консоль. В сохранённой таблице всегда
ровно четыре строки, включая `not_run`, `failed`, `oom`, `incomplete`, `evaluation_failed`.
`complete` означает обработку всех выбранных изображений; короткий запуск
явно помечается. Разные limit/набор/конфиг/исходники не смешиваются под одним
именем. Ошибка метрик получает `evaluation_failed`; успешный report-only может восстановить статус.
Существующий результат модели не перезаписывается: после прерывания
нужно новое имя, resume нет. Отчёт восстанавливается без весов/инференса:

```bash
docker compose -f compose.yaml run --rm annotation scripts/annotation/blip_base.py --run annotation-smoke-v2 --report-only
```

Восстановление использует сохранённый протокол и проверенные captions/instances;
при изменении набора оно завершается ошибкой. Общий
`storage/locks/inference.lock` наследуется реальным модельным процессом;
следующая модель запускается только после его выхода. SIGTERM/INT родителя
останавливают группу дочернего процесса, затем родитель обновляет статус.

`raw_caption` сохраняется неизменным. Completion token IDs включают конечные
специальные токены, исключают начальный decoder token/входной prompt.
Сохраняются точные prompts, синхронизированные CUDA
preprocess/generate/postprocess, чтение, cold load, peak RSS и PyTorch
allocated/reserved VRAM (bytes). Cold load не включает проверку SHA и создание
CUDA-контекста, кэш файлов ОС не сбрасывается. Пики памяти включают загрузку и
прогрев, не учитывают память драйвера/других процессов. Словари tokenizer
различаются: tokens/s не равнозначен скорости слов.

[Метрики](evaluation/README.md): CIDEr в исходной шкале, BLEU-4, ROUGE-L,
CHAIRs/CHAIRi (доли) и дополнительная объектная полнота. CHAIR использует
GT captions+instances и проверенный словарь авторов; действия, свойства и
вне-COCO объекты он не оценивает. Новые CI и человеческие баллы не создаются.
Внешний набор и SPICE не использованы; известное использование COCO при
подготовке моделей ограничивает независимость оценки.

CPU-проверки отдельно от реальных GPU-smoke:

```bash
docker compose -f compose.yaml run --rm annotation -m pytest annotation/tests -q
docker compose -f compose.yaml run --rm annotation -m ruff check scripts/annotation annotation/tests
docker compose -f compose.yaml run --rm annotation -m ruff format --check scripts/annotation annotation/tests
```

Исторический test500-v1 завершён: четыре модели × 500 изображений, один проход,
2000 raw captions, без ошибок. [CSV](../reports/comparisons/annotation-test500-v1/annotation.csv),
[таблица](../reports/comparisons/annotation-test500-v1/annotation.md),
[анализ](../Описание%20результатов%20аннотирования.md),
[верификация](../reports/verification/annotation-test500-v1.json).
Старые результаты, configs/manifests, source snapshots и незаполненные
исторические формы сохранены; новый report-only читает только новый формат.


## Проверки упрощённой реализации 11.10.2026

20 CPU-тестов проходят, scoped Ruff check/format без замечаний. Все четыре
модели завершили реальный офлайн GPU-smoke по двум изображениям, FP16,
без замены checkpoint и без OOM. Все восемь raw_caption и последовательностей
completion IDs точно совпали с историческим нативным инференсом на этих сценах.
Report-only проверен отдельно без GPU. Полный инференс не повторялся.

CPU-пересчёт шести автоматических метрик на всех 2000 сохранённых captions
совпал с историческим CSV с допуском 1e-12; исходные captions неизменны.
[Проверка паритета](../reports/runs/verification/annotation-parity.json),
[короткая таблица](../reports/runs/annotation/refactor-smoke-20261011-v1/comparison.md).
Короткие результаты не заменяют завершённое основное сравнение test500-v1.
