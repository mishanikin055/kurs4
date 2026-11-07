# Датасеты сравнительного анализа

01.11.2026 по запросу пользователя удалены локальные изображения COCO val2017
и ImageNet-1k validation, исходные архивы, Parquet-файлы и кэш загрузки ImageNet.
Освобождено 15 431 622 656 байт (14,37 ГиБ) внутри WSL.
Удалены также распакованные аннотации COCO и одно изображение `data/coco/smoke/`.
Приложение использует собственные загруженные изображения и эти данные не требует.

Результаты и отчёты в `reports/`, исходные конфиги сравнений, веса моделей,
пользовательские изображения, БД и Docker volumes сохранены. Небольшие локальные
JSON-манифесты в `data/coco/` и `data/imagenet/`, включая `source_provenance.json`,
`source/download_plan.json` и `source/download_manifest.json`, оставлены для
проверки прежнего состава и контрольных сумм. `data/` по-прежнему исключён из Git.
Ресурсы метрик в `data/annotation_evaluator/` сохранены.

## Источники

| Набор | Что использовали | Где скачать |
|---|---|---|
| COCO 2017 validation | 5 000 изображений; общая основа детекции, исторических crops и аннотирования | [Страница COCO](https://cocodataset.org/#download), [изображения val2017.zip](https://s3.amazonaws.com/images.cocodataset.org/zips/val2017.zip), [аннотации annotations_trainval2017.zip](https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2017.zip) |
| ImageNet-1k validation | 50 000 изображений, 1 000 классов; также подвыборка evaluation5000 | [ILSVRC/imagenet-1k на Hugging Face](https://huggingface.co/datasets/ILSVRC/imagenet-1k), [закреплённая ревизия](https://huggingface.co/datasets/ILSVRC/imagenet-1k/tree/49e2ee26f3810fb5a7536bbf732a7b07389a47b5/data) |

ImageNet требует принять условия доступа на Hugging Face. Read-токен хранится
только в `.secrets/hf_token`. Нужны 14 файлов `validation-*-of-00014.parquet`;
обучающая часть не использовалась и скачивать её не требуется.

Состав COCO закреплён в `detection/configs/dataset_split.json`, аннотирования —
в `annotation/configs/dev100.json` и `test500.json`. Источник, ревизия и размеры
ImageNet закреплены в `classification/configs/dataset_source.json`, классы —
в `classification/configs/imagenet_labels.json`. Исторические вырезки получались
из COCO; отдельного скачиваемого датасета crops нет.

## Восстановление без запуска моделей

Команды выполняются из корня проекта в WSL, когда данные снова понадобятся.
Используются существующие Docker-образы сравнений и функции `scripts/prepare.py`.
Нужен запущенный Docker; GPU не требуется. Ничего из этого при очистке не запускалось.

COCO: восстановить изображения, instances и captions, затем проверить состав:

```bash
COMPARISON_NETWORK=bridge PREPARE_ACCESS=rw docker compose run --rm detection -c 'from scripts.prepare import prepare_coco, verify_dataset; prepare_coco(True); verify_dataset("detection")'
COMPARISON_NETWORK=bridge PREPARE_ACCESS=rw docker compose run --rm annotation -c 'from scripts.prepare import prepare_annotation_resources, prepare_captions, verify_dataset; prepare_annotation_resources(True); prepare_captions(); verify_dataset("annotation")'
```

ImageNet: скачать закреплённые validation shards, извлечь исходные изображения,
восстановить и проверить манифесты. Токен монтируется отдельным файлом только для чтения:

```bash
COMPARISON_NETWORK=bridge PREPARE_ACCESS=rw docker compose run --rm \
  -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \
  -v "$PWD/.secrets/hf_token:/run/secrets/hf_token:ro" \
  classification -c 'from pathlib import Path; from scripts.prepare import fetch_validation, import_huggingface, verify_dataset; fetch_validation(Path("/run/secrets/hf_token"), False); import_huggingface(); verify_dataset("classification")'
```

Загрузчик ImageNet требует около 18,7 ГиБ свободного места с запасом на импорт.
COCO с архивами и распаковкой занимает ещё около 1,8 ГиБ. Общая подготовка через
`scripts/prepare.py --download` также подготавливает веса; команды выше восстанавливают
только данные. Они намеренно вызывают импорт ImageNet даже при сохранённом
`validation.json`: один манифест не означает, что изображения уже на диске.

`data/coco/smoke/` не нужен для сравнений; при необходимости отдельного smoke
можно взять изображение из восстановленного `val2017/`. Реальные model smoke
и benchmark запускаются только по отдельному поручению.

## Место Windows и WSL

Удаление освобождает блоки внутри Linux. Файл виртуального диска WSL
`ext4.vhdx` может сохранять прежний размер на Windows. Это отдельная операция
сжатия диска, для которой может понадобиться остановить WSL и Docker.
Не удалять сам VHDX и не использовать `wsl --unregister`.
Справка: [управление диском WSL от Microsoft](https://learn.microsoft.com/en-us/windows/wsl/disk-space).

После удаления выполнен `fstrim -v /`. Внутри WSL занято около 84 ГиБ;
логический размер файла `ext4.vhdx` остался 155 057 127 424 байта.
Сжатие не выполнялось: остановка текущей WSL прервала бы рабочую сессию.

Чтобы вернуть место Windows, сохранить открытые документы, закрыть Docker Desktop
и рабочие сессии WSL, затем открыть **PowerShell от имени администратора** под
тем же пользователем Windows и выполнить:

```powershell
$distro = Get-ChildItem HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss |
    ForEach-Object { Get-ItemProperty $_.PSPath } |
    Where-Object DistributionName -eq 'Ubuntu-26.04'
if (@($distro).Count -ne 1) { throw 'Не найден единственный дистрибутив Ubuntu-26.04' }
$vhd = Join-Path $distro.BasePath 'ext4.vhdx'
if (-not (Test-Path -LiteralPath $vhd)) { throw 'Файл ext4.vhdx не найден' }
wsl --shutdown
if ($LASTEXITCODE -ne 0) { throw 'Не удалось остановить WSL' }
@("select vdisk file=`"$vhd`"", 'compact vdisk', 'exit') | diskpart
```

Дождаться успешного завершения DiskPart, после этого можно снова запустить WSL
и Docker. Фактическая экономия зависит от расположения свободных блоков;
сообщение TRIM о числе обработанных байтов не означает такой же прирост места
в Windows. Команда сжатия описана в [документации Microsoft](https://learn.microsoft.com/en-us/windows-server/administration/windows-commands/compact-vdisk).
