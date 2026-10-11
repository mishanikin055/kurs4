# Инфраструктура: Linux, Docker и ресурсы

Рабочая копия `/home/mikhail/kurs4`, WSL2 Ubuntu 26.04, Bash.
Один основной `compose.yaml` содержит три ML-сервиса: detection, classification,
annotation. Дополнение `compose.gpu.yaml` подключает GPU текущей WSL через
`/dev/dxg`, `/usr/lib/wsl/lib` и `/usr/lib/wsl/drivers` read-only, без Docker socket.
Основной файл также используется для CPU-метрик/тестов/report-only/подготовки.
Команды — [README](README.md). Прежние Compose/shell-обёртки удалены.

## Окружения

Три `*/Dockerfile` сохраняют базовый digest PyTorch/CUDA и точные requirements;
образы `kurs4-detection:v1`, `kurs4-classification:v1`, `kurs4-annotation:v1`.
Transformers детекции и аннотирования имеют разные версии, объединять их нельзя
без отдельной проверки. Entry point — Python, запуск прямым путём скрипта.
`init: true` собирает потомков; лимиты RAM — 6 GiB детекция/классификация,
7 GiB аннотирование. Данные/веса read-only, результаты/locks — writable.

Runtime `network_mode: none`, HF offline, токен не монтируется.
Явная подготовка использует `COMPARISON_NETWORK=bridge PREPARE_ACCESS=rw`
и `scripts/prepare.py <направление> --download`. HF-токен для новой загрузки
ImageNet подключается отдельным read-only mount `.secrets/hf_token`.
Не выводить токены в чат/логи и не включать их в образ/Git.
Локальная подготовка без сети также поддерживается.

После сборки записать фактические ID образов переменными DETECTION_IMAGE_ID,
CLASSIFICATION_IMAGE_ID, ANNOTATION_IMAGE_ID из `docker image inspect`.
JSON запуска фиксирует image ID, версии, исходники и Git revision/dirty status.
Данные не встраиваются в Docker context; `.dockerignore` исключает datasets,
веса всех трёх направлений, secrets, reports и исходные материалы.

## Память и процессы

RTX 4050 Laptop, 6141 MiB VRAM; Intel Core i5-12450H; хост 16 GiB RAM,
WSL видит около 7.6 GiB. Перед GPU-проверками измерять реальные available RAM,
свободную VRAM, cgroup limit и отсутствие конкурирующего инференса.
PyTorch allocated/reserved не равны полной памяти GPU/Windows-драйвера.
Подключение `/dev/dxg` не предполагает NVIDIA Container Toolkit; это настройка
именно текущей WSL, для другого Linux требуется собственный проверенный профиль.

Строго одна модель в RAM/GPU, batch 1. `storage/locks/inference.lock` удерживается
модельным дочерним процессом до выхода; следующая модель стартует после завершения.
Остановка родителя должна завершить process group, а SIGKILL родителя не должен
освободить lock живого потомка. `empty_cache()` это требование не заменяет.
Cgroup OOM-kill и CUDA OOM должны иметь явный статус, неполный набор не становится
полным результатом. После прерывания новое имя сравнения, без сложного resume.

## Проверки и границы задачи

По последнему указанию пользователя новые модельные запуски, пересчёт
метрик из исторических предсказаний и пересборка таблиц в текущей переделке
прекращены. Дополнительная реальная проверка — только по новой задаче.

Локальные pytest в трёх Docker-окружениях, Ruff check/format, Compose config,
метрики на сохранённых предсказаниях и последовательные реальные офлайн smoke
всех 12 моделей. Полные сравнения ради переделки не повторяются.
Контроль сохранности — `docs/refactoring-baseline.json`; исторические отчёты,
DOCX и source snapshots не изменяются. Mock/CPU и GPU результаты указывать отдельно.
GitHub Actions удалены по указанию пользователя, проверки выполняются локально.

Приложение FastAPI/PostgreSQL/Streamlit, worker, UI и поиск не входят в рефакторинг.
При их будущей реализации порядок детектор → классификатор всех вырезок →
аннотатор → согласование и общий lock сохраняются; PostgreSQL integration
обязателен при приёмке приложения. Не создавать фиктивные сервисы сейчас.
