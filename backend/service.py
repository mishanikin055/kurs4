"""Image acceptance and job reads, independent of HTTP."""

import shutil
import time
from uuid import UUID, uuid4

from sqlalchemy.exc import SQLAlchemyError
from starlette.datastructures import UploadFile

from backend.repository import JobRepository
from backend.schemas import AcceptedImage, JobInfo
from backend.settings import Settings, model_specs
from backend.storage import save_image
from backend.tasks import check_broker


class JobNotFound(Exception):
    pass


class ImageService:
    def __init__(self, repository: JobRepository, settings: Settings):
        self.repository = repository
        self.settings = settings

    def upload(self, file: UploadFile) -> AcceptedImage:
        image_id = uuid4()
        committed = False
        try:
            metadata = save_image(file, image_id, self.settings)
            with self.repository.session.begin():
                job = self.repository.create(metadata, model_specs(self.settings))
            committed = True
            return AcceptedImage(
                image_id=image_id,
                job_id=job.id,
                status=job.status,
                publish_status=job.publish_status,
                status_url=f"/jobs/{job.id}",
            )
        finally:
            if not committed:
                shutil.rmtree(self.settings.image_root / str(image_id), ignore_errors=True)

    def get_job(self, job_id: UUID) -> JobInfo:
        job = self.repository.get(job_id)
        if job is None:
            raise JobNotFound
        return JobInfo.model_validate(job)

    def health(self, celery, publisher: dict, publisher_enabled: bool) -> dict:
        try:
            self.repository.ping()
            database_ok = True
        except SQLAlchemyError:
            database_ok = False
        broker_ok = check_broker(celery, self.settings.broker_timeout)
        publisher_ok = (
            publisher_enabled
            and publisher["database_ok"]
            and time.monotonic() - publisher["last_tick"]
            < self.settings.publisher_interval + self.settings.broker_timeout + 10
        )
        ready = database_ok and broker_ok and publisher_ok
        return {
            "status": "ok" if ready else "degraded",
            "database": database_ok,
            "rabbitmq": broker_ok,
            "publisher": publisher_ok,
            "processing": "configured",
        }
