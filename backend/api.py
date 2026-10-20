"""Uploads, result previews and downloads; database writes belong to the worker."""

import shutil
import tempfile
from pathlib import Path
from typing import Annotated
from uuid import UUID
from zipfile import ZipFile

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.background import BackgroundTask

from backend.schemas import AcceptedImage
from backend.service import BrokerUnavailable, ImageService
from backend.storage import UploadError
from backend.tasks import check_broker

router = APIRouter()


def get_service(request: Request) -> ImageService:
    return ImageService(request.app.state.settings, request.app.state.celery)


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
    ready = check_broker(state.celery, state.settings.broker_timeout)
    return JSONResponse(status_code=200 if ready else 503, content={"rabbitmq": ready})


@router.get("/images/{image_id}")
def preview_result(image_id: UUID, service: Service):
    try:
        result = service.result(image_id)
    except FileNotFoundError as exc:
        raise HTTPException(404, "result_not_found", headers={"Cache-Control": "no-store"}) from exc
    return JSONResponse(result, headers={"Cache-Control": "no-store"})


@router.get("/images/{image_id}/files/{filename}")
def download_file(image_id: UUID, filename: str, service: Service):
    try:
        path = service.result_file(image_id, filename)
    except FileNotFoundError as exc:
        raise HTTPException(404, "result_not_found") from exc
    media_type = {".png": "image/png", ".json": "application/json"}.get(path.suffix)
    return FileResponse(
        path,
        media_type=media_type or "application/octet-stream",
        filename=filename,
        headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
    )


@router.get("/images/{image_id}/archive")
def download_archive(image_id: UUID, service: Service):
    try:
        paths = [
            service.result_file(image_id, name)
            for name in ("original", "normalized.png", "detection.png", "results.json")
        ]
    except FileNotFoundError as exc:
        raise HTTPException(404, "result_not_found") from exc
    temporary = Path(tempfile.mkdtemp(prefix="kurs4-download-"))
    try:
        archive = temporary / "result.zip"
        with ZipFile(archive, "w") as bundle:
            for path in paths:
                bundle.write(path, arcname=f"{image_id}/{path.name}")
        return FileResponse(
            archive,
            media_type="application/zip",
            filename=f"{image_id}.zip",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
            background=BackgroundTask(shutil.rmtree, temporary),
        )
    except BaseException:
        shutil.rmtree(temporary)
        raise
