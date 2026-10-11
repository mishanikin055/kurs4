import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from starlette.concurrency import run_in_threadpool

from backend.api import router
from backend.db import make_engine, make_sessions
from backend.settings import Settings
from backend.tasks import make_celery, publish_pending

logger = logging.getLogger(__name__)


class UploadLimitMiddleware:
    """Bound the multipart body before the parser writes arbitrary data to disk."""

    def __init__(self, app, limit: int):
        self.app = app
        self.limit = limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        size = 0

        async def limited_receive():
            nonlocal size
            message = await receive()
            size += len(message.get("body", b""))
            if size > self.limit:
                raise HTTPException(413, "request_too_large")
            return message

        await self.app(scope, limited_receive, send)


def create_app(settings: Settings | None = None, *, publisher_enabled: bool = True) -> FastAPI:
    settings = settings or Settings()
    engine = make_engine(settings.database_url.get_secret_value())
    sessions = make_sessions(engine)
    celery = make_celery(settings)
    publisher_state = {"last_tick": 0.0, "database_ok": False}

    def tick():
        publisher_state["last_tick"] = time.monotonic()

    async def publisher(stop: asyncio.Event):
        while not stop.is_set():
            tick()
            try:
                await run_in_threadpool(
                    publish_pending, sessions, celery, settings.broker_timeout, tick=tick
                )
                publisher_state["database_ok"] = True
            except Exception:
                publisher_state["database_ok"] = False
                logger.warning("Publisher cycle failed; retrying after interval")
            tick()
            try:
                await asyncio.wait_for(stop.wait(), settings.publisher_interval)
            except TimeoutError:
                pass

    @asynccontextmanager
    async def lifespan(app):
        stop = asyncio.Event()
        task = asyncio.create_task(publisher(stop)) if publisher_enabled else None
        try:
            yield
        finally:
            stop.set()
            if task is not None:
                await task
            engine.dispose()
            celery.close()

    app = FastAPI(title="Kurs4 Image API", lifespan=lifespan)
    app.add_middleware(UploadLimitMiddleware, limit=settings.max_upload_bytes + 64 * 1024)
    app.state.sessions = sessions
    app.state.celery = celery
    app.state.publisher = publisher_state

    app.state.settings = settings
    app.state.publisher_enabled = publisher_enabled
    app.include_router(router)

    return app
