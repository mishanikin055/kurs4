# Проверки классификации

Дата: 08.10.2026. Полный ImageNet50k и полные crops отложены по указанию пользователя.

- Linux Compose: закреплены PyTorch 2.7.1 CUDA12.8 / torchvision0.22.1;
  pip check успешен, фактический image ID сохранён в каждом environment.
- Все четыре официальных state_dict скачаны и проверены по полным SHA-256,
  размерам, label order и native transforms. Веса/кэши/токены вне Git.
- Реальный офлайн GPU-smoke всех четырёх на одном COCO изображении без ImageNet GT:
  native top5/softmax parity успешна, FP32/batch1, последовательные child.
- ImageNet: авторизованная загрузка 14 validation-shards (6 693 093 726 байт),
  полные LFS SHA-256 проверены. Потоковый импорт сохранил 50 000 исходных JPEG,
  50 на каждый класс. Проверены synsets из имён файлов и три заранее разрешённых
  различия написания HF/torchvision, не изменяющие class order.
- Реальный evaluation5000: 12 успешных проходов, 5000 изображений на каждый,
  все модели × 3 повтора. Хеши samples соответствуют manifests; предсказания
  каждой модели идентичны между тремя повторами. Сбои/OOM не обнаружены.
  Метрики — первый повтор, задержки и ресурсы — три, без повторного обучения.
- Crop GPU-smoke: 4 модели × 2 режима × 2 вырезки. Все восемь модельных проходов
  успешны. Проверены чтение, округление/обрезка рамки, top5, mapping, decisions
  и сохранение исходной категории. В metrics.json smoke quality=null;
  короткие проверки не выданы за итоговую crop-оценку.
- CPU-тесты классификации и crops: 23 passed. Детекция отдельно в своём
  закреплённом образе: 13 passed. Общая попытка в классификационном образе
  отклонена из-за отсутствия pycocotools; зависимости не менялись ради неё.
  Покрыты balanced split, Top1/Top5/F1, sparse confusion, source identity и resume,
  class-agnostic matching, one-to-one/crowd, sparse category mapping,
  исправленные/внесённые ошибки, сцены с несколькими объектами, сохранение
  detector label и shared lock после SIGKILL родителя. Default report не читает
  bootstrap и не публикует интервалы. Это быстрые проверки, а не model smoke.
- Ruff check/format, Bash syntax и Compose WSL/crops config: успешны.

Исходные конфиги и snapshots защищают результаты от изменений текущего кода.
Уже рассчитанный bootstrap сохранён до отмены пользователем дальнейших CI;
основные новые CSV/Markdown не включают интервалы. Полный validation50k,
полное сравнение crops, dev-калибровка отказа, UI и приложение не проверялись.
Подробные хеши запусков — classification_runs.json; загрузки — imagenet_download.json.
