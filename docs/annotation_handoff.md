# Точка продолжения сравнения аннотаторов

Обновлено 09.10.2026. Рабочая копия `/home/mikhail/kurs4`,
ветка `codex/classification-completion`, существующий origin.
Прочитать [AGENTS.md](../AGENTS.md), [annotation.md](../annotation.md)
и [annotation/README.md](../annotation/README.md).

## Фактическая точка остановки

Основной test500-v1 **завершён**: BLIP, Florence-2, Qwen3-VL и SmolVLM2
имеют status=complete, по 500 captions, пустые errors.jsonl. Всего 2000 ответов,
ровно один полный проход каждой модели. Benchmark-контейнер завершился.
Inference commit — `8619dc8c2ffad0148734392f4663c888b1c43d30`.

Пользователь сначала поручил остановиться после прогона ради лимитов, затем
явно поручил основному агенту сформировать финальные таблицы и отчёт.
Субагент annotation_comparison остановился из-за лимита аккаунта и повторно
не запускался. Основной агент выполнил CPU-report, пересчёт/верификацию
метрик, таймингов, ресурсов и точной decode parity всех captions, анализ
и 54 CPU-теста/Ruff. Повторный model inference и новые CI не выполнялись.

## Готовые результаты

- [Описание результатов](../Описание%20результатов%20аннотирования.md).
- [Таблица](../reports/comparisons/annotation-test500-v1/annotation.md),
  [CSV](../reports/comparisons/annotation-test500-v1/annotation.csv).
- [Независимая верификация](../reports/verification/annotation-test500-v1.json).
- `reports/comparisons/annotation-test500-v1/`: analysis/report manifests,
  per-image CIDEr/timings, CHAIR details, category coverage.
- `reports/annotation/test500-v1/`: исходные run.json, неизменные raw_caption,
  token IDs, timings, environment, dataset/config, errors, experiment identity
  и source_snapshot.tar.gz. Эти сырые файлы локальны и игнорируются Git.
- Веса/manifest — `annotation/models/`; данные — `data/coco/`.
- Генераторы и команды — annotation/report.py,
  scripts/verify_annotation.py, reports/comparisons/build_annotation_analysis.py,
  [README](../annotation/README.md). Перестройка отчёта не загружает модель.

Florence-2 base-ft лучший по CIDEr/BLEU-4/ROUGE-L, CHAIRs/CHAIRi и p50
в измеренном протоколе коротких английских captions. Qwen наиболее полно
называет объекты, но ответы значительно длиннее эталонов. Не объявлять
автоматические lexical metrics экспертной фактической корректностью.

## Что осталось для последующего поручения

1. Реальные человеческие оценки: 100 общих изображений × четыре модели,
   400 строк. Локальная пустая форма
   `reports/comparisons/annotation-test500-v1/expert_blind_form.csv`,
   закрытый ключ `expert_private_key.csv`; оба игнорируются Git.
   [Методика](annotation_expert_protocol.md). Баллы не выдумывать.
2. Внешние собственные/разрешённые изображения с независимыми эталонами
   не предоставлены. Известный COCO training overlap остаётся ограничением.
3. SPICE не вычислялся, evaluator не валидирован; русский вывод, поиск,
   приложение и полный конвейер в этот benchmark не входят.

500 — практический бюджет, случайный seed42 без стратификации,79/80 категорий.
Формальная достаточность и устойчивость ранжирования не доказаны; один проход
не измеряет межпрогонную устойчивость. Все модели FP16/batch1/greedy, максимум96
новых токенов, общий inference.lock, строго по одной. Новые доверительные
интервалы запрещены во всех сравнениях. Готовые модели повторно не запускать
без нового указания и не перезаписывать experiment identity/source snapshot.

Сырые результаты, веса и незавершённые экспертные формы не удалять. Исходный
каталог `Материалы для курсовой/` и DOCX не трогать. Автоматическая проверка
раз в10 минут удалена по указанию пользователя; не создавать её повторно.

Пример следующего запроса:

> Продолжи с docs/annotation_handoff.md: автоматические таблицы уже готовы.
> Нужно завершить человеческую оценку или обработать предоставленный внешний
> набор. Сохранённые прогоны не повторять, доверительные интервалы не считать.
