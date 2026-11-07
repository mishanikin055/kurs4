"""Uploads, previews and bounded exports backed by PostgreSQL."""

import json
from io import BytesIO
from typing import Annotated
from uuid import UUID
from zipfile import ZipFile

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

from backend.schemas import AcceptedImage
from backend.service import BrokerUnavailable, ImageService
from backend.storage import UploadError
from backend.tasks import check_broker

router = APIRouter()


def get_service(request: Request) -> ImageService:
    return ImageService(
        request.app.state.settings, request.app.state.celery, request.app.state.sessions
    )


Service = Annotated[ImageService, Depends(get_service)]


@router.post("/images", response_model=AcceptedImage, status_code=202)
def upload_image(service: Service, file: UploadFile = File(...)):
    try:
        return service.upload(file)
    except UploadError as exc:
        raise HTTPException(exc.status_code, exc.code) from exc
    except BrokerUnavailable as exc:
        raise HTTPException(503, "broker_unavailable") from exc
    except OSError as exc:
        raise HTTPException(503, "storage_unavailable") from exc


@router.get("/health")
def health(request: Request):
    state = request.app.state
    broker_ready = check_broker(state.celery, state.settings.broker_timeout)
    try:
        with state.sessions() as session:
            session.execute(text("SELECT 1"))
        database_ready = True
    except SQLAlchemyError:
        database_ready = False
    return JSONResponse(
        status_code=200 if broker_ready and database_ready else 503,
        content={"rabbitmq": broker_ready, "database": database_ready},
    )


@router.get("/images/{image_id}")
def preview_result(image_id: UUID, service: Service):
    try:
        result = service.result(image_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, "result_not_found", headers={"Cache-Control": "no-store"}) from exc
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


@router.get("/images/{image_id}/files/{filename:path}")
def download_file(image_id: UUID, filename: str, service: Service):
    try:
        media_type, content = service.result_file(image_id, filename)
    except FileNotFoundError as exc:
        raise HTTPException(404, "result_not_found") from exc
    return Response(
        content,
        media_type=media_type,
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Content-Disposition": f'attachment; filename="{filename.rsplit("/", 1)[-1]}"',
        },
    )


@router.get("/images/{image_id}/archive")
def download_archive(image_id: UUID, service: Service):
    try:
        artifacts = service.bundle(image_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, "result_not_found") from exc
    # Only one bounded image bundle is materialized, never the archive catalogue.
    if sum(len(content) for _, content in artifacts.values()) > service.settings.max_bundle_bytes:
        raise HTTPException(413, "bundle_too_large")
    stream = BytesIO()
    with ZipFile(stream, "w") as bundle:
        for name, (_, content) in artifacts.items():
            bundle.writestr(f"{image_id}/{name}", content)
        manifest = json.loads(artifacts["results.json"][1])
        if manifest.get("crops") == []:
            bundle.mkdir(f"{image_id}/crops/")
    return Response(
        stream.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{image_id}.zip"',
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )
