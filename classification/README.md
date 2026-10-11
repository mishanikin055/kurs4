# Классификация ImageNet

Новые сравнения запускаются четырьмя Python-скриптами в
`scripts/classification/`. После запуска в консоли выводится компактная таблица всех четырёх участников.
Каждый файл явно создаёт свою torchvision-модель,
загружает локальный state dict, выполняет штатные transforms и выдаёт Top-5.
`common.py` проверяет данные и веса, удерживает общий lock, запускает один
модельный процесс, измеряет стадии и собирает таблицы. Дополнительного CLI нет.

| Скрипт | Готовые веса | Штатный вход |
| --- | --- | --- |
| `resnet50.py` | ResNet50_Weights.IMAGENET1K_V2 | 224×224, resize 232 |
| `efficientnet_v2_s.py` | EfficientNet_V2_S_Weights.IMAGENET1K_V1 | 384×384, resize 384 |
| `convnext_tiny.py` | ConvNeXt_Tiny_Weights.IMAGENET1K_V1 | 224×224, resize 236 |
| `vit_b_16.py` | ViT_B_16_Weights.IMAGENET1K_V1 | 224×224, resize 256 |

## Подготовка и запуск

Команды из корня проекта. Данные и веса уже подготовлены; явная повторная
подготовка — через `scripts/prepare.py` по основной инструкции в [README](../README.md).
Закреплённые источники, SHA-256, порядок 1000 synsets и native preprocessing
сохранены в `classification/configs/` и `classification/models/manifest.json`.
Сравнение ничего не скачивает. Docker dependencies закреплены в
`requirements.lock.txt`; `Dockerfile` сохраняет отдельное ML-окружение.

Перед модельными командами зафиксировать реальный ID существующего образа:

```bash
export CLASSIFICATION_IMAGE_ID="$(docker image inspect kurs4-classification:v1 --format '{{.Id}}')"
```

В новых запусках используется только `data/imagenet/validation.json`: 50 000
изображений, по 50 на каждый из 1000 классов. Один проход, batch 1, FP32,
CUDA, 20 прогревов, четыре CPU threads. Ниже короткая проверка двух изображений;
все четыре команды выполняются последовательно с одинаковым новым именем.

```bash
docker compose -f compose.yaml -f compose.gpu.yaml run --rm classification scripts/classification/resnet50.py --run imagenet-smoke-v2 --limit 2
docker compose -f compose.yaml -f compose.gpu.yaml run --rm classification scripts/classification/efficientnet_v2_s.py --run imagenet-smoke-v2 --limit 2
docker compose -f compose.yaml -f compose.gpu.yaml run --rm classification scripts/classification/convnext_tiny.py --run imagenet-smoke-v2 --limit 2
docker compose -f compose.yaml -f compose.gpu.yaml run --rm classification scripts/classification/vit_b_16.py --run imagenet-smoke-v2 --limit 2
```

Для отдельного явно запрошенного полного сравнения убрать `--limit` и выбрать
новое имя. Полные прогоны ради рефакторинга не требуются. Набор, параметры,
версия кода и образ должны совпадать под одним именем. Повторная запись
результата модели запрещена; после прерывания выбрать новое имя. Resume и
новые CI отсутствуют. Короткая проверка отмечена `shortened=True`; её метрики
относятся только к указанной части набора.

Родитель удерживает `storage/locks/inference.lock`, ребёнок наследует его fd
до выхода. SIGTERM/SIGINT родителя завершают process group ребёнка, при
необходимости — SIGKILL с ожиданием. Даже при SIGKILL родителя у живого ребёнка
остаётся lock. Следующая модель создаётся после выхода предыдущего процесса.

## Результаты и проверки

Новые артефакты: `reports/runs/classification/NAME/`:

- `comparison.json`: набор, настройки одного прохода, hashes и ID образа;
- `<model_id>/samples.jsonl`: предсказания с индексами, synsets и вероятностями,
  чтение и модельные тайминги каждого изображения;
- `<model_id>/run.json`: статус, processed/expected_images, холодная загрузка,
  ресурсы, provenance, native preprocessing и parity первого изображения;
- `comparison.csv`, `comparison.md`: четыре строки, включая `not_run`,
  `failed`, `oom`, `incomplete`; у неполных запусков качество пустое;
- `per_class.csv`: показатели всех 1000 классов каждой завершённой модели.

Top-1/Top-5 и Macro-F1 рассчитываются по сохранённым предсказаниям.
Macro-F1 включает все 1000 классов, в том числе отсутствующие в коротком наборе.
p50/p95 и скорость включают preprocessing/transfer, forward и postprocessing,
исключают чтение, запись, загрузку и прогрев. Холодная загрузка измеряется в
новом процессе без очистки page cache. RAM — peak RSS процесса, VRAM —
PyTorch allocated/reserved с загрузкой модели, не полная занятость GPU;
таблицы используют GiB. В JSONL `previous_write_ms` относится к
`previous_write_image_id`; запись последней строки — `last_write_ms` в run.json.
Это позволяет сохранить фактические flush/fsync без переписывания предсказаний.

Восстановление таблиц без загрузки весов и инференса:

```bash
docker compose -f compose.yaml run --rm classification scripts/classification/resnet50.py --run imagenet-smoke-v2 --report-only
docker compose -f compose.yaml run --rm classification -m pytest classification/tests
docker compose -f compose.yaml run --rm classification -m ruff check scripts/classification classification/tests
docker compose -f compose.yaml run --rm classification -m ruff format --check scripts/classification classification/tests
```

CPU-тесты проверяют метрики, Top-5/labels/synsets, целостность JSONL, неполные
результаты, таблицу четырёх моделей и наследование lock. Они не заменяют
реальный offline GPU-smoke каждой модели с native parity.

Исторические результаты остаются в `reports/classification/` и
`reports/comparisons/classification-*`. Полный ImageNet завершён ранее по двум
проходам всех четырёх моделей; третий ResNet-50 сохранён отдельно. evaluation5000
и оба crop-сравнения завершены по трём проходам. Старые configs/manifests и
source snapshots не переписываются; прежний код доступен через Git/snapshots.
Детали вырезок — [историческая справка](crops/README.md).

Проверка новой реализации 11.10.2026: 25 CPU-тестов, Ruff check/format и
`--report-only` прошли. Offline GPU-smoke четырёх checkpoint по двум изображениям
прошёл с native parity (нулевое расхождение Top-5); это проверка реализации,
а не новый полный сравнительный анализ.
