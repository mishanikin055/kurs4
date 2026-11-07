"""Local SQL transactions for CPU tests; production remains PostgreSQL."""

import pytest
from sqlalchemy import create_engine, event

from backend.db import make_sessions
from backend.tables import Base


@pytest.fixture
def database(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False}
    )

    @event.listens_for(engine, "connect")
    def foreign_keys(connection, record):
        connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    try:
        yield engine, make_sessions(engine)
    finally:
        engine.dispose()
