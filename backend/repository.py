"""SQL operations; transaction boundaries belong to the caller."""

from uuid import UUID, uuid4

from sqlalchemy import select, text
from sqlalchemy.orm import Session, selectinload

from backend.tables import STAGES, Image, Job, ModelResult


class JobRepository:
    def __init__(self, session: Session):
        self.session = session

    def create(self, metadata: dict, model_specs: dict) -> Job:
        job = Job(id=uuid4(), image=Image(**metadata))
        job.results = [ModelResult(stage=stage, **model_specs[stage]) for stage in STAGES]
        self.session.add(job)
        self.session.flush()
        return job

    def get(self, job_id: UUID) -> Job | None:
        return self.session.scalar(
            select(Job)
            .where(Job.id == job_id)
            .options(selectinload(Job.image), selectinload(Job.results))
        )

    def ping(self) -> None:
        self.session.execute(text("SELECT 1 FROM jobs LIMIT 1"))

    def due_publications(self) -> list[UUID]:
        from backend.tables import utcnow

        return list(
            self.session.scalars(
                select(Job.id)
                .where(
                    Job.status == "queued",
                    Job.publish_status == "pending",
                    Job.next_publish_at <= utcnow(),
                )
                .order_by(Job.next_publish_at, Job.id)
                .limit(10)
            )
        )

    def reserve_publication(self, job_id: UUID) -> bool:
        from datetime import timedelta

        from backend.tables import utcnow

        job = self.session.get(Job, job_id)
        if job is None or job.publish_status != "pending" or job.status != "queued":
            return False
        job.publish_attempts += 1
        job.next_publish_at = utcnow() + timedelta(
            seconds=min(60, 5 * 2 ** min(job.publish_attempts - 1, 4))
        )
        return True

    def publication_result(self, job_id: UUID, error: dict | None) -> None:
        from backend.tables import utcnow

        job = self.session.get(Job, job_id)
        job.publish_error = error
        if error is None:
            job.publish_status = "sent"
            job.published_at = utcnow()
