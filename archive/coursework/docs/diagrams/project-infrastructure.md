# Инфраструктура приложения

Текущая схема от 19.10.2026. Пять сервисов уже определены в compose.app.yaml.
[Сохранённые PNG](project-infrastructure.png) и [SVG](project-infrastructure.svg)
относятся к прежней архитектуре с заданиями/статусами и сохранены как история.
Актуальная схема — ниже; подробности в [плане](../../PROJECT_PLAN.md).

```mermaid
flowchart LR
    F[Браузер через nginx] -->|Загрузка| A[FastAPI]
    A -->|image_id| Q[RabbitMQ: inference]
    Q --> W[Celery worker · GPU]
    W --> M[RF-DETR Small → ViT-B/16 → Florence-2 base-ft]
    A -->|Входы| V[(image_data volume)]
    V -->|Чтение| W
    M -->|Полный комплект| V
    W -->|После файлов: одна запись| D[(PostgreSQL: processed_images)]
    W -->|ACK после commit| Q
    V -->|Ручное скачивание через API| F
    F -->|Необязательная копия| L[Папка компьютера или ZIP]
```

API не обращается к БД и не создаёт заданий. Worker хранит три модели в постоянных
процессах, вычисляет последовательно и сохраняет итоговый комплект в volume всегда.
После сохранения файлов — одна строка processed_images. Ошибки — только в логах.
В интерфейсе нет опроса статусов; дополнительная копия запрашивается вручную.
У nginx нет image volume/секретов/весов. Веса worker read-only, image volume writable.
Общий inference.lock занят весь срок жизни моделей. GPU smoke и mock-интеграция
отличаются: эта переделка проверяется без запуска настоящих моделей.
