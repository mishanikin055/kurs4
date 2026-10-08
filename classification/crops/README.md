# Прикладное сравнение классификаторов на объектах COCO

Участники и готовые веса совпадают с основным ImageNet-сравнением. Два режима
не смешиваются: эталонные GT-рамки и сохранённые рамки RF-DETR-Small.
Инференс идёт по всем 1000 ImageNet-классам, затем применяется зафиксированный
семантический mapping; метка детектора не ограничивает softmax классификатора.

## Данные и словарь

До crop-прогонов выбраны 500 сцен из COCO test, seed 42. Часть уже оценённого
детекторного test используется как вход для нового crop-эксперимента. Это не
независимое повторное сравнение детекторов. RF-DETR выбран как единственный
источник рамок; выводы о classifier после других детекторов потребуют отдельной
проверки. Набор сцен и mapping одинаковы для всех четырёх классификаторов.

Таблица `configs/mapping.json` содержит все 1000 synsets: 287 семантических
подтипов соответствуют 56 общим категориям COCO; остальные — not_mappable.
Отображение вручную проверено по названиям зафиксированного torchvision-словаря
и реальным sparse category_id COCO. CSV-копия:
`reports/model_selection/classification_coco_mapping.csv`.
Например golden retriever → dog (COCO ID 18), zebra → zebra (ID 24),
tench → not_mappable. Волк не становится собакой, bookcase не становится book.
Даже поддерживаемая COCO-категория может быть покрыта ImageNet лишь частично.

GT-crops включают поддерживаемые non-crowd объекты. Detector-crops включают
все рамки после score ≥ 0.5, максимум 100 на сцену, в порядке score. Сопоставление
с GT — жадное, один к одному, без проверки предсказанного класса, IoU ≥ 0.5.
Это позволяет отдельно обнаружить ошибочную категорию правильно найденного
объекта. Crowd-overlap игнорируется по intersection / proposal area, когда
обычного совпадения нет. Не найденные, неподдерживаемые и crowd-объекты не
попадают в условную точность matched-supported, но сохраняются в учёте покрытия.

Подготовленные manifests содержат 1883 GT-crops и 2296 detector-crops.
Поддерживаемые GT составляют около 55.81% non-crowd объектов выбранных сцен.
Это покрытие категорий, а не гарантия наличия всех их подтипов в ImageNet.
Рамки округляются наружу и ограничиваются размерами оригинала, JPEG не меняется.
Оригинал читается с диска; все декодированные сцены/crops в RAM не удерживаются.

## Метрики и ограничения

Top-1 и Top-5 оценивают общую категорию после mapping, не породу/подтип.
Top-5 содержит пять ImageNet-классов; несколько могут перейти в одну категорию,
поэтому это не пять различных COCO-классов. Macro-F1 усредняется по категориям
с ненулевым GT-support в одинаковом общем наборе. Неотображаемый ответ считается
ошибкой на поддерживаемом объекте, а не удаляется из знаменателя.

В detector-crops дополнительно считаются detector-only matched accuracy,
пропущенные GT, unmatched и unsupported proposals, исправленные и внесённые
ошибки при **гипотетической** замене категории. Для этой диагностической замены
заранее зафиксированы score ≥ 0.5 и top1–top2 ≥ 0.1; иначе сохраняется detector.
Эти пороги не настроены на dev и не служат доказательством надёжного отказа
на неизвестных объектах. В сохраняемом реальном результате detector_category_id
и effective_category_id остаются прежними, top-k и agreement/conflict/uncertain/
not_mappable хранятся рядом. Для рабочего правила отказа нужна отдельная
dev-калибровка; после просмотра этих test-результатов пороги здесь не меняются.

Supported recall учитывает пропуски относительно всех поддерживаемых GT.
Условная matched accuracy сама по себе не является качеством полного конвейера.
Новых рамок классификатор не создаёт. COCO не содержит эталонных пород,
поэтому точность тонкого уточнения отсутствует, даже если ответ выглядит разумным.

Качество берётся из первого повтора, скорость и ресурсы — из трёх. Новые
доверительные интервалы по указанию пользователя не рассчитываются.
Ранее подготовленный bootstrap по целым сценам остаётся отдельным инструментом;
он не запускается benchmark автоматически. p50/p95 учитывают препроцессинг,
нейросеть и постпроцессинг; чтение, кадрирование и запись измеряются отдельно.
VRAM — peak PyTorch allocated/reserved, RAM — peak RSS дочернего процесса.

## Статус

08.10.2026 успешно выполнен только smoke: по две вырезки для каждой из четырёх
моделей в каждом режиме. Качество smoke не рассчитывается. Полные comparative
прогоны и dev-калибровка отказа отложены по указанию пользователя.

## Запуск после отдельного продолжения

Используется проверенный образ `kurs4-classification:v1`, установленный по основной
инструкции. Команды выполнять из `/home/mikhail/kurs4`, после окончания других
GPU-экспериментов. Данные и веса в runtime read-only, сеть отключена. Модельные
процессы строго последовательны; общий lock совпадает с детекцией.

```bash
scripts/classification_crops.sh prepare
scripts/classification_crops.sh benchmark --manifest reports/classification/crop-data-v1/gt_crops.json --output-dir reports/classification/gt-crops-v1
scripts/classification_crops.sh report --runs-dir reports/classification/gt-crops-v1 --output-dir reports/classification/gt-crops-v1/comparison
scripts/classification_crops.sh benchmark --manifest reports/classification/crop-data-v1/detector_crops.json --output-dir reports/classification/detector-crops-v1
scripts/classification_crops.sh report --runs-dir reports/classification/detector-crops-v1 --output-dir reports/classification/detector-crops-v1/comparison
```

Для smoke добавить `--limit 2` и использовать другую папку. Сокращённый запуск
не участвует в итоговом отчёте и bootstrap. Для resume добавить `--resume`:
сверяются конфиг, manifest, веса, исходники и Docker image ID, а также записи
каждого sample_id. Ошибки и OOM остаются в run.json/errors.jsonl/process.log.

Результаты: `<runs>/comparison/classification_crops.csv` и `.md`, per_class.csv,
paired_differences.csv и report_manifest.json. В каждом модельном проходе
samples.jsonl содержит исходную рамку, crop, matching, top-k, решение и timings.
Связанные оригиналы указаны в dataset_manifest.json и защищены SHA-256.
