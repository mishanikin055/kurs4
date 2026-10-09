# Выбор аннотаторов, 09.10.2026

Сравниваются четыре заранее предложенных представителя разных подходов:
специализированный captioner BLIP, task-token Florence, компактные instruct VLM
Qwen и SmolVLM. Это не рейтинг четырёх наиболее популярных моделей.
HF downloads за месяц — динамический вспомогательный снимок в registry,
а не доказательство качества или сопоставимых captioning-лидерств.

- [BLIP base](https://huggingface.co/Salesforce/blip-image-captioning-base): BSD-3-Clause,
  captioning checkpoint с ViT base; COCO training указан авторами.
- [Florence-2 base-ft](https://huggingface.co/microsoft/Florence-2-base-ft): MIT,
  готовое дообучение авторов, task `<CAPTION>`. Собственного fine-tuning нет.
- [Qwen3-VL 2B Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct): Apache-2.0,
  около 2,13B параметров, image+instruction. Полный список image IDs обучения
  не раскрыт, независимость test500 не подтверждена.
- [SmolVLM2 500M Video Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct):
  Apache-2.0, image inference поддерживается; карта перечисляет
  `llava-onevision/sharegpt4v_coco`, точное пересечение val2017 не установлено.
  Video в названии не означает, что в проект добавлено видео.

Revisions конкретных официальных репозиториев закреплены до test в
annotation/configs/models.json. Общий объём выбранных checkpoint/processor
примерно 7,8 GB; исключены TF, ONNX и дубликаты PyTorch, кроме BLIP,
для которого выбранный официальный revision имеет только pytorch_model.bin.
Torch 2.7.1 проходит текущую проверку безопасной загрузки этого формата.
Лицензии подтверждены model cards; Florence содержит отдельный LICENSE,
для остальных уведомление/model card сохраняется вместе с checkpoint.

Для Florence официальный старый processor несовместим с native
Florence2Processor в Transformers 4.57.6 (нет tokenizer.image_token);
native загрузка не используется, чтобы не получить несопоставленную архитектуру.
Три официальных Python-файла проверены и закреплены SHA-256/revision.
Проверка: импорты torch/transformers/timm/einops/numpy и относительный config;
выполняемые сети, shell, exec/eval и загрузки посторонних файлов отсутствуют.
Requests в modeling встречается только в docstring-примере.
`trust_remote_code=True` разрешён явно только этому checkpoint и только
из локального read-only каталога; остальным False. Совместимость smoke
фиксируется отдельно, никакая test-метрика не используется для выбора режима.

Florence на Transformers 4.57.6 использует штатный eager attention и авторский
tuple-cache: `_supports_default_dynamic_cache` отключён на language_model.
Это адаптация интерфейса GenerationMixin без изменения checkpoint/forward.
Tokenizer использует полный официальный tokenizer.json (fast), image processor
остаётся CLIPImageProcessor из закреплённого preprocessor_config.
