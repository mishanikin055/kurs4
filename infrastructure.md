# Инфраструктура: Docker, RabbitMQ, Celery и ресурсы

Схема для презентации и описание взаимодействий: [инфраструктура приложения](docs/diagrams/project-infrastructure.md).

## План приложения от 11.10.2026

Архитектура — [PROJECT_PLAN.md](PROJECT_PLAN.md), задание реализации —
[backend.md](backend.md). `compose.app.yaml` уже содержит четыре сервиса.
Backend/db/rabbitmq и код worker реализованы. Worker включается явно профилем
`processing`; реальный инференс пока не прошёл smoke. `compose.app.gpu.yaml`
подключает GPU текущей WSL только к worker. Интеграционные тесты используют
настоящий отдельный Celery consumer и синтетические модельные процессы:


| Сервис | Запуск и доступ |
|---|---|
| `backend` | FastAPI/Uvicorn, один worker; `127.0.0.1:8000:8000`. Приём файлов, чтение БД и publisher pending-заданий. Без GPU/весов. |
| `db` | PostgreSQL, healthcheck, volume `pgdata`. Внутренний адрес `db:5432`, без публикации порта на хост. |
| `rabbitmq` | Один узел RabbitMQ, healthcheck, фиксированный hostname и volume `rabbitmq_data`. AMQP `rabbitmq:5672` внутри сети Compose. |
| `worker` | Celery, `--pool=prefork --concurrency=1 --prefetch-multiplier=1 -Q inference`, `init: true`; один экземпляр, GPU и веса только здесь. |

Общий volume `image_data` монтируется в backend и worker по одинаковому пути;
API пишет исходные/нормализованные файлы, worker читает их. Временные crops —
временный каталог worker. Отдельный bind mount `./storage/locks` у worker
и сравнений обеспечивает один и тот же `inference.lock`. БД, брокер и файлы
не хранятся только в слое контейнера; для backup нужны БД и файлы изображений.
RabbitMQ volume сохраняет очередь при обычном пересоздании контейнера.
`down -v` удаляет данные и не является штатной командой перезапуска.

У backend/worker общий код и Dockerfile: api, test и ML worker target.
Worker использует два Python-окружения для несовместимых Transformers 5.19.0/4.57.6,
общий PyTorch/CUDA закреплён digest. ML pins сохраняются из lock-файлов сравнений,
annotation typing-extensions повышен до 4.16.0 из-за текущего DB runtime.
`pip check` обоих окружений прошёл. Checkpoint подключаются read-only по отдельным
директориям, datasets не нужны. Лимиты: db/rabbitmq/backend по 512 MiB, worker 5 GiB;
суммарную фактическую память проверить перед GPU smoke.
Секреты PostgreSQL и отдельного пользователя/vhost RabbitMQ берутся из игнорируемых
локальных файлов/env; не использовать межконтейнерный доступ через guest/guest,
не включать секреты в Git, образ и логи. Management UI брокера по умолчанию не нужен.
Контейнеры работают в обычной сети Compose: `network_mode: none` сравнений здесь
не подходит. Весам достаточно локального read-only mount, HF работает offline.

Последовательность запуска: здоровые db/rabbitmq → миграции из API-образа → API
и Celery consumer. API и worker обязаны переподключаться после потери зависимостей;
`depends_on` не заменяет восстановление во время работы. Начальный publisher
один, запускается lifespan FastAPI. Он только отправляет сохранённые задания,
не выполняет инференс. Состояния публикации и повторов — в плане.

Очередь `inference`/exchange — durable, сообщения persistent, publisher confirms.
Результаты хранятся в прикладных таблицах PostgreSQL; отдельный Celery result backend,
Redis, Beat и Flower не нужны. `rabbitmq.conf` согласует `consumer_timeout` с верхней
границей задачи (ожидание lock + три этапа + очистка). У worker должны быть
проверенные timeout остановки subprocess и `stop_grace_period`; одного ACK/изменения
статуса недостаточно для освобождения GPU.

GPU worker подключается по проверенному образцу текущей WSL (`/dev/dxg`, библиотеки
драйвера read-only). API, PostgreSQL и RabbitMQ GPU не получают. После выбора тройки
проверить совместимость ML-зависимостей; при необходимости отдельные Python-окружения
внутри worker. Не объединять исторические requirements вслепую, не передавать
Docker socket. Перед реальным запуском проверить суммарный бюджет RAM четырёх
сервисов: лимит старого одиночного ML-контейнера нельзя переносить автоматически.

Проверки приложения: Compose config, миграции, настоящие PostgreSQL/RabbitMQ/Celery,
сбои публикации/ACK и сервисов, сохранность volumes, последовательность процессов.
CPU/mock и eager mode отмечать отдельно. Smoke трёх моделей — только по отдельной
явной задаче. Проверенные команды первого этапа находятся в README. Проверки загрузки,
публикации, отказов БД/брокера и сохранности volumes выполнены; настоящего
GPU-приёмки пока нет. Текущие интеграционные проверки Celery/ACK/requeue и
остановки процессов перечислены в README; модели в этих проверках синтетические.

## Существующие окружения сравнений

Рабочая копия `/home/mikhail/kurs4`, WSL2 Ubuntu 26.04, Bash.
Один основной `compose.yaml` содержит три ML-сервиса: detection, classification,
annotation. Дополнение `compose.gpu.yaml` подключает GPU текущей WSL через
`/dev/dxg`, `/usr/lib/wsl/lib` и `/usr/lib/wsl/drivers` read-only, без Docker socket.
Основной файл также используется для CPU-метрик/тестов/report-only/подготовки.
Команды — [README](README.md). Прежние Compose/shell-обёртки удалены.

### ML-образы

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

### Память и процессы

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

### Проверки и границы задачи

По последнему указанию пользователя новые модельные запуски, пересчёт
метрик из исторических предсказаний и пересборка таблиц в текущей переделке
прекращены. Дополнительная реальная проверка — только по новой задаче.

Для текущих правок кода — локальные CPU/mock pytest в трёх Docker-окружениях,
Ruff check/format и Compose config. Ранее выполненные проверки метрик и smoke
12 моделей остаются историческими результатами, не командой на новый запуск.
Полные сравнения ради переделки не повторяются.
Контроль сохранности — `docs/refactoring-baseline.json`; исторические отчёты,
DOCX и source snapshots не изменяются. Mock/CPU и GPU результаты указывать отдельно.
GitHub Actions удалены по указанию пользователя, проверки выполняются локально.
