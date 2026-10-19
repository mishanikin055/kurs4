"""Database connection and request-scoped SQLAlchemy sessions."""

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


def make_engine(url: str) -> Engine:
    # Create worker engines inside the task, after Celery forks.
    return create_engine(
        url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 3, "options": "-c statement_timeout=5000"},
        hide_parameters=True,
    )


def make_sessions(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)
