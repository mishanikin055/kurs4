from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from backend.api import router
from backend.settings import Settings
from backend.tasks import make_celery


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


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    celery = make_celery(settings)

    @asynccontextmanager
    async def lifespan(app):
        try:
            yield
        finally:
            celery.close()

    app = FastAPI(title="Kurs4 Image API", lifespan=lifespan)
    app.add_middleware(UploadLimitMiddleware, limit=settings.max_upload_bytes + 64 * 1024)
    app.state.celery = celery
    app.state.settings = settings
    app.include_router(router)
    return app
