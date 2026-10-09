# Сравнение аннотаторов

Четыре независимых готовых checkpoint: BLIP base, Florence-2 base-ft,
Qwen3-VL 2B Instruct и SmolVLM2 500M Video Instruct. Исходное RGB-изображение
передаётся без overlay, детекций и результатов классификатора.
Веса находятся в **annotation/models/** (Git ignore), конкретные revisions,
официальные источники, лицензии и размеры — `configs/models.json`;
проверенные SHA-256 локальных файлов — `models/manifest.json`.

Linux Docker Compose, RTX 4050 Laptop 6 GB VRAM; контейнер ограничен 7 GiB,
WSL видит 7,6 GiB. Только одна модель в RAM/GPU: координатор наследует общий
`storage/locks/inference.lock` потомку и ждёт полного завершения.
FP16 без offload/квантования, batch 1, greedy decoding, максимум 96 новых
токенов, один прогрев на dev. BLIP/Florence используют eager attention,
Qwen/SmolVLM — SDPA; это явное различие реализации готовых архитектур.
Processors закреплены с use_fast=False (Florence tokenizer — fast,
поскольку официальный репозиторий содержит tokenizer.json без merges.txt); у Qwen 65536–262144 pixels.

Основной протокол: **ровно один проход на модель** по одним и тем же
500 COCO val2017 из project-local detector test4500. Dev100 — отдельные
изображения из detector dev500, seed 42. Нет новых доверительных интервалов.
Не запускать benchmark в другой папке как второй полный повтор.
Существующий output не перезаписывается; неполный проход сохраняется со статусом.

```bash
scripts/annotation.sh build
scripts/annotation.sh download
docker compose -f compose.annotation.yaml run --rm --entrypoint python tools -c 'import nltk; nltk.download("punkt_tab", download_dir="/workspace/data/annotation_evaluator/nltk_data")'
scripts/annotation.sh prepare
scripts/annotation.sh smoke --model qwen3_vl_2b --output reports/annotation/smoke-qwen-v1
scripts/annotation.sh dev --limit 10 --output reports/annotation/dev10-v1
scripts/annotation.sh benchmark --output reports/annotation/test500-v1
scripts/annotation.sh report --experiment reports/annotation/test500-v1 --output reports/comparisons/annotation-test500-v1
```

COCO captions предварительно извлекаются из официального
[annotations_trainval2017.zip](https://images.cocodataset.org/annotations/annotations_trainval2017.zip)
в data/coco/annotations/captions_val2017.json. Во время inference сеть отключена,
веса/data read-only. Скачать checkpoint во время benchmark нельзя.

Каждая строка содержит неизменный raw_caption, completion token IDs (включая
конечные специальные токены, исключая начальный decoder token/входной prompt),
prompt и rendered_prompt, синхронизированные CUDA preprocess/generate/postprocess,
read time. Сохраняются cold load, peak RSS, CUDA allocated/reserved peaks и
source snapshot/container ID. Downloads/hash verification не входят в cold load.
Словари tokenizer различаются, tokens/s не равнозначен скорости слов.

[Методика метрик](evaluation/README.md): CIDEr, BLEU-4, ROUGE-L, CHAIRs/CHAIRi
и дополнительная объектная полнота. Эксперты заполняют 400 перемешанных строк
для 100 общих изображений; ключ модели хранится отдельно. Автоматические
числа не заменяют человеческую оценку фактической корректности.
Внешние разрешённые изображения/эталоны не предоставлены; SPICE не вычисляется.
Известное использование COCO при подготовке готовых моделей ограничивает выводы
о независимом обобщении; pretrained overlap нельзя устранить новым split val2017.

Mock/CPU-проверки:
```bash
docker compose -f compose.annotation.yaml run --rm checks -m pytest annotation/tests -q
docker compose -f compose.annotation.yaml run --rm checks -m ruff check .
docker compose -f compose.annotation.yaml run --rm checks -m ruff format --check .
```
Реальные GPU-smoke считаются отдельными проверками; приложение здесь не реализуется.
