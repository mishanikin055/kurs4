# Проверка восстановления полного сравнения детекторов

Дата: 08.10.2026. Полное сравнение test-v2 завершено: 12 complete-проходов, по 4500 изображений, без shortened. Исходные три failed-прохода RF-DETR в test-v1 сохранены (по 253 обработанных изображения).

## Исправление и проверки

- Причина: неиспользуемый sparse COCO ID 66 в pretrained голове RF-DETR, проявившийся на изображении 179285.
- Исправление: исключаются только gaps/фон в slots 0–90 RF-DETR, вне диапазона остаётся ошибка. ID не перенумеровываются. По трём полным повторам исключены суммарно 60 предсказаний; счётчик есть в timings/CSV.
- Скрипт запуска проверяет наличие NVIDIA runtime средствами Bash, без зависимости от rg.
- Smoke на проблемном изображении 428×640 прошёл офлайн на GPU; native-parity passed, 296 валидных предсказаний и 3 исключённые позиции.
- Pytest: 13 passed. Ruff check/format, Bash syntax и Compose WSL config прошли.
- Проверен парный bootstrap: dataset-level COCO AP на дубликатах совпадает с прямым пересчётом, включая crowd и равные scores; проверены checkpoint resume и отклонение другого seed.

## Восстановление и происхождение

scripts/recover_detection.py проверил config, dataset, веса, контейнер, старый source fingerprint и snapshot. AST-сравнение подтвердило, что RF-DETR-only исправление восстанавливает исходный адаптер при удалении добавленного поведения; evaluator обычных метрик не изменился. Девять complete-проходов перенесены с исходными environment manifests, три RF-DETR выполнены заново. История не скрывается: recovery.json, run.json/carried_forward и CSV/reused_from содержат ссылки на original test-v1.

Фактически использованные команды восстановления:

```bash
export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"
export DETECTION_IMAGE_ID="$(docker image inspect kurs4-detection:v1 --format '{{.Id}}')"
docker compose -f compose.detection.yaml run --rm --entrypoint python cpu scripts/recover_detection.py --from-runs reports/detection/test-v1 --output-dir reports/detection/test-v2
scripts/detection.sh benchmark --output-dir reports/detection/test-v2 --resume
```

После всех complete-проходов запущена последовательность постобработки (CPU, сеть отключена):

```bash
docker compose -f compose.detection.yaml run --rm cpu bootstrap --runs-dir reports/detection/test-v2 --samples 1000
docker compose -f compose.detection.yaml run --rm cpu report --runs-dir reports/detection/test-v2 --output-dir reports/comparisons
```

Bootstrap завершён: четыре модели по 1000 выборок, seed 42, shortened=false. Все команды завершились с exit=0; итоговый CSV содержит четыре complete-строки, по 4500 изображений и три повтора, с заполненными CI95. Анализ и точные шесть парных интервалов — reports/comparisons/detection_analysis.md и detection_paired.csv. Прогресс: reports/detection/test-v2/bootstrap.progress.json; лог: bootstrap.log; финальная сборка: report.log. Версионируемый manifest проверки: reports/verification/detection_recovery.json.

Ограничения: один набор COCO и одно оборудование; девять проходов измерены ранее, RF-DETR позднее; разные штатные разрешения и данные предобучения. Это сравнение конкретных готовых checkpoint, а не доказанный рейтинг популярности и не оценка любых объектов вне COCO.
