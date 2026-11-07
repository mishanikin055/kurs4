# Исторические сравнения на вырезках COCO

Новые классификационные скрипты работают только с ImageNet. Код запуска
GT-crops и detector-crops удалён при согласованном упрощении 11.10.2026.
Исторические данные, результаты, configs, manifests и исходные source snapshots
сохранены. Классификация вырезок в будущем приложении остаётся отдельной задачей.

Оба сравнения завершены 09.10.2026: четыре готовые ImageNet-модели, по три
прохода, 500 COCO test-сцен. GT-crops: 1883 объекта; detector-crops: 2296 рамок
RF-DETR Small, 1054 matched-supported объекта, 829 пропущенных supported GT.
Предсказания одинаковы между повторами; новых CI нет.

Справочники в `configs/` сохранены: 287 ImageNet synsets соответствуют 56
COCO-категориям, остальные `not_mappable`. Покрытие non-crowd GT — 55.81%.
Инференс выполнялся по всем 1000 классам; detector label не ограничивал softmax.
Метрики этих режимов нельзя смешивать с целым изображением ImageNet.

GT использовал поддерживаемые non-crowd объекты. Detector proposals: score ≥ 0.5,
максимум 100 на сцену, matching один к одному при IoU ≥ 0.5 без учёта класса;
crowd-overlap учитывался отдельно. Рамки округлялись наружу и ограничивались
размерами оригинала. Эталонные JPEG не менялись. Top-5 означает пять
ImageNet-классов, которые могут соответствовать одной COCO-категории.

Качество рассчитано по первому проходу, скорость и ресурсы — по трём.
Macro-F1 усреднён по категориям с ненулевым GT-support. Неотображаемые ответы
остаются ошибками. COCO не содержит эталонов пород/подтипов. Пороги
гипотетической замены категории (score ≥ 0.5, top1–top2 ≥ 0.1) не калиброваны на dev;
замена ухудшила matched accuracy всех четырёх моделей. Исходная метка детектора
сохранена; это ограничение важно для будущего конвейера.

Сохранённые результаты:

- [GT-анализ](../../reports/comparisons/classification-gt-crops-v1/analysis.md),
  [верификация](../../reports/verification/classification-gt-crops-v1.json);
- [detector-анализ](../../reports/comparisons/classification-detector-crops-v1/analysis.md),
  [верификация](../../reports/verification/classification-detector-crops-v1.json);
- сырые результаты: `reports/classification/{gt-crops-v1,detector-crops-v1}/`;
- исходные manifests: `reports/classification/crop-data-v1/`.

Для просмотра готовых CSV/Markdown-таблиц и вывода CSV в консоль новый инференс не требуется.
Старые команды и инструкции продолжения относятся к историческим source snapshots.
