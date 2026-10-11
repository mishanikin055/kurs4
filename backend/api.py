"""HTTP contracts and dependency injection."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from backend.db import get_session
from backend.repository import JobRepository
from backend.schemas import AcceptedImage, JobInfo
from backend.service import ImageService, JobNotFound
from backend.storage import UploadError

router = APIRouter()


def get_service(
    request: Request, session: Annotated[Session, Depends(get_session)]
) -> ImageService:
    return ImageService(JobRepository(session), request.app.state.settings)


Service = Annotated[ImageService, Depends(get_service)]


@router.post("/images", response_model=AcceptedImage, status_code=202)
def upload_image(service: Service, file: UploadFile = File(...)):
    try:
        return service.upload(file)
    except UploadError as exc:
        raise HTTPException(exc.status_code, exc.code) from exc
    except SQLAlchemyError as exc:
        raise HTTPException(503, "database_unavailable") from exc
    except OSError as exc:
        raise HTTPException(503, "storage_unavailable") from exc


@router.get("/jobs/{job_id}", response_model=JobInfo)
def get_job(job_id: UUID, service: Service):
    try:
        return service.get_job(job_id)
    except JobNotFound as exc:
        raise HTTPException(404, "job_not_found") from exc
    except SQLAlchemyError as exc:
        raise HTTPException(503, "database_unavailable") from exc


@router.get("/health")
def health(request: Request, service: Service):
    state = request.app.state
    content = service.health(state.celery, state.publisher, state.publisher_enabled)
    return JSONResponse(status_code=200 if content["status"] == "ok" else 503, content=content)
