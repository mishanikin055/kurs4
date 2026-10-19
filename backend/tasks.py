import logging

from celery import Celery, signals
from celery.exceptions import WorkerShutdown
from celery.worker.request import Request
from kombu import Exchange, Queue

from backend.settings import Settings

logger = logging.getLogger(__name__)
TASK_NAME = "backend.process_image"


def make_celery(settings: Settings) -> Celery:
    app = Celery("kurs4", broker=settings.broker_url.get_secret_value())
    exchange = Exchange("inference", type="direct", durable=True)
    app.conf.update(
        task_queues=(Queue("inference", exchange, routing_key="inference", durable=True),),
        task_default_queue="inference",
        task_default_exchange="inference",
        task_default_routing_key="inference",
        task_routes={TASK_NAME: {"queue": "inference", "routing_key": "inference"}},
        task_create_missing_queues=False,
        task_serializer="json",
        accept_content=["json"],
        task_ignore_result=True,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        task_default_delivery_mode="persistent",
        worker_concurrency=1,
        worker_prefetch_multiplier=1,
        worker_enable_remote_control=False,
        broker_connection_timeout=settings.broker_timeout,
        broker_connection_retry_on_startup=True,
        broker_transport_options={
            "confirm_publish": True,
            "read_timeout": settings.broker_timeout,
            "write_timeout": settings.broker_timeout,
        },
        task_publish_retry=False,
        enable_utc=True,
        timezone="UTC",
    )
    return app


def publish_image(app: Celery, image_id: str, timeout: float) -> None:
    # A fresh bounded connection avoids stale pooled sockets after broker restart.
    with app.connection_for_write(
        connect_timeout=timeout,
        transport_options={
            "confirm_publish": True,
            "read_timeout": timeout,
            "write_timeout": timeout,
        },
    ) as connection:
        connection.ensure_connection(max_retries=0)
        app.send_task(
            TASK_NAME,
            args=[image_id],
            task_id=image_id,
            connection=connection,
            retry=False,
            delivery_mode=2,
            queue="inference",
            routing_key="inference",
            timeout=timeout,
            confirm_timeout=timeout,
        )


def check_broker(app: Celery, timeout: float) -> bool:
    try:
        with app.connection_for_write(connect_timeout=timeout) as connection:
            connection.ensure_connection(max_retries=0)
        return True
    except Exception:
        return False


@signals.worker_init.connect
def require_processing_task(sender=None, **kwargs):
    # Do not let an unfinished consumer ACK/discard unregistered production tasks.
    if sender is not None and TASK_NAME not in sender.app.tasks:
        logger.error("Processing task is missing; refusing to consume.")
        raise WorkerShutdown(1)


@signals.worker_process_init.connect
def ignore_idle_interrupt(**kwargs):
    import signal

    signal.signal(signal.SIGUSR2, signal.SIG_IGN)


@signals.worker_shutting_down.connect
def interrupt_active_processing(**kwargs):
    import os
    import signal

    from billiard import active_children

    # Warm shutdown normally waits for tasks; interrupt our coordinator first so
    # it can kill the model process group and requeue before releasing the lock.
    for child in active_children():
        try:
            os.kill(child.pid, signal.SIGUSR2)
        except ProcessLookupError:
            pass


class ResidentRequest(Request):
    def on_failure(self, exc_info, *args, **kwargs):
        import os
        import signal

        from billiard.exceptions import WorkerLostError

        # A killed coordinator may leave a running model or an unread IPC response.
        exc = getattr(exc_info.exception, "exc", exc_info.exception)
        if isinstance(exc, WorkerLostError):
            os.kill(os.getpid(), signal.SIGTERM)
        return super().on_failure(exc_info, *args, **kwargs)


def register_processing_task(app: Celery, settings: Settings, *, runner=None, lock_fd=None):
    from uuid import UUID

    from celery.exceptions import Reject
    from sqlalchemy.exc import SQLAlchemyError

    from backend.db import make_engine, make_sessions
    from backend.processing import LockTimeout, ProcessingInterrupted, interrupt, process_image

    @app.task(
        name=TASK_NAME,
        acks_late=True,
        reject_on_worker_lost=True,
        max_retries=None,
        Request=ResidentRequest if lock_fd is not None else Request,
    )
    def task(image_id: str):
        import os
        import signal

        engine = make_engine(settings.database_url.get_secret_value())
        previous = signal.signal(signal.SIGTERM, interrupt)
        previous_interrupt = signal.signal(signal.SIGUSR2, interrupt)
        try:
            options = {"runner": runner} if runner is not None else {}
            process_image(
                UUID(image_id), make_sessions(engine), settings, lock_fd=lock_fd, **options
            )
        except (SQLAlchemyError, ProcessingInterrupted) as exc:
            # Interrupted parallel requests leave unread IPC: restart before the next image.
            signal.signal(signal.SIGUSR2, signal.SIG_IGN)
            os.kill(os.getppid(), signal.SIGTERM)
            raise Reject("processing_interrupted", requeue=True) from exc
        except LockTimeout as exc:
            raise task.retry(exc=exc, countdown=30, max_retries=None)
        finally:
            signal.signal(signal.SIGTERM, previous)
            signal.signal(signal.SIGUSR2, previous_interrupt)
            engine.dispose()

    return task


def worker_main(settings: Settings, *, runner=None) -> None:
    import time

    from sqlalchemy.exc import SQLAlchemyError

    from backend.db import make_engine, make_sessions

    # Readiness gate also prevents immediate requeue loops during database outages.
    engine = make_engine(settings.database_url.get_secret_value())
    try:
        while True:
            try:
                with make_sessions(engine)() as session:
                    # Fail readiness until the new completed-results table exists.
                    from sqlalchemy import select

                    from backend.tables import ProcessedImage

                    session.execute(select(ProcessedImage.id).limit(1))
                break
            except SQLAlchemyError:
                logger.warning("Worker waiting for database")
                time.sleep(5)
    finally:
        engine.dispose()
    from contextlib import nullcontext

    from backend.runtime import resident_models

    runtime = resident_models(settings) if runner is None else nullcontext(runner)
    with runtime as runner:
        app = make_celery(settings)
        register_processing_task(
            app, settings, runner=runner, lock_fd=getattr(runner, "lock_fd", None)
        )
        app.worker_main(
            [
                "worker",
                "--pool=prefork",
                "--concurrency=1",
                "--prefetch-multiplier=1",
                "-Q",
                "inference",
                "--loglevel=INFO",
                "--without-mingle",
                "--without-gossip",
            ]
        )


if __name__ == "__main__":
    worker_main(Settings())
