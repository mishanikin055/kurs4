# Проверки подготовки классификации

Дата: 08.10.2026. Реализован whole-image ImageNet benchmark; итоговая оценка ещё не выполнялась.

- Собран Linux Compose-образ с закреплёнными PyTorch 2.7.1 CUDA 12.8 / torchvision 0.22.1; pip check прошёл.
- Четыре официальных state_dict загружены в classification/models/, проверены полные SHA-256, размеры, class order и native transforms. Веса и кэши исключены из Git.
- Реальный офлайн GPU-smoke: ResNet-50, EfficientNetV2-S, ConvNeXt-Tiny, ViT-B/16 — smoke_passed. Каждый в отдельном последовательном child, FP32/batch1, одно изображение COCO без ImageNet GT. Native top-5/softmax parity прошла. Артефакты — reports/classification/smoke-v1; исходники/snapshot/image ID сохранены. Версионируемое происхождение — classification_smoke.json.
- CPU-тесты классификации: 13 passed. Проверены точные Top-1/Top-5/Macro-F1 и sparse confusion, порядок эталонов, фиксированный balanced split, потерянная последняя строка resume, дубликаты/несовпадение labels, идентичность парного stratified bootstrap, изменение seed, выход пути/симлинка за data/, пустые отчёты без фиктивного качества, удержание shared lock реальным дочерним процессом после SIGKILL родителя.
- Ruff check/format, Bash syntax, Compose WSL config проходят. Повторная проверка детекции — 13 passed; GPU-бенчмарк детекции не повторялся.
- Dry-run ImageNet: закреплены revision и 14 validation-shards общим размером 6 693 093 726 байт; train/test не входят в загрузку.

Авторизованная загрузка и импорт реального ImageNet пока не проверены: ожидается локальный read token после принятия пользователем условий. Основные evaluation5000/validation50000 и их bootstrap не запущены. Эти проверки не создают итоговый сравнительный анализ и не доказывают качество на GT/detector-crops. Прикладной mapping/crops остаются невыполненными.
