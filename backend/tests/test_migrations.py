"""Migrate legacy data in a private PostgreSQL schema, without running models."""

import json
import os
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url

from backend.settings import Settings
from backend.tables import Base

pytestmark = pytest.mark.skipif(
    os.environ.get("APP_TEST_INTEGRATION") != "1", reason="Requires isolated PostgreSQL"
)


def test_compact_migration_preserves_results_and_matches_orm(monkeypatch):
    url = Settings().database_url.get_secret_value()
    engine = create_engine(url)
    schema = "migration_" + uuid4().hex
    with engine.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    private_url = make_url(url).update_query_dict({"options": f"-csearch_path={schema}"})
    monkeypatch.setenv("APP_DATABASE_URL", private_url.render_as_string(hide_password=False))
    # Production connect_args override URL options; retain this test's private search_path.
    monkeypatch.setattr("backend.db.make_engine", create_engine)
    private = create_engine(private_url)
    config = Config("backend/alembic.ini")
    image_id, job_id, object_id = (str(uuid4()) for _ in range(3))
    caption = {"raw_caption": "  Original caption.  "}
    objects = [
        {
            "object_id": object_id,
            "bbox": [0.1, 1, 5.2, 6],
            "label": "person",
            "score": 0.9,
            "class_id": 1,
            "label_space": "coco",
        },
        {"object_id": str(uuid4()), "bbox": [6, 1, 8, 6], "label": "cup", "score": 0.8},
    ]
    classified = [
        {
            "object_id": object_id,
            "crop_bbox": [0, 1, 6, 6],
            "status": "succeeded",
            "top_k": [{"label": "second", "score": 0.1}, {"label": "best", "score": 0.8}],
        },
        {
            "object_id": objects[1]["object_id"],
            "status": "failed",
            "top_k": [],
            "error": {"code": "crop_error"},
        },
    ]
    try:
        command.upgrade(config, "a2f004752f43")
        with private.begin() as connection:
            connection.execute(
                text("""
                INSERT INTO images VALUES (
                    :id, 'original.png', '/original', :sha, 'image/png', 100,
                    '/normalized.png', :sha, 16, 12, 'exif-rgb-png-v1', now()
                )
            """),
                {"id": image_id, "sha": "a" * 64},
            )
            connection.execute(
                text("""
                INSERT INTO jobs (
                    id, image_id, status, created_at, publish_status,
                    publish_attempts, next_publish_at, processing_attempts
                ) VALUES (:id, :image_id, 'partial', now(), 'sent', 1, now(), 1)
            """),
                {"id": job_id, "image_id": image_id},
            )
            for stage, payload in (
                ("detection", {"image_width": 16, "image_height": 12, "objects": objects}),
                ("classification", {"objects": classified}),
                ("annotation", caption),
            ):
                connection.execute(
                    text("""
                    INSERT INTO model_results (
                        id, job_id, stage, status, parameters, schema_version, payload
                    ) VALUES (:id, :job_id, :stage, 'succeeded', '{}', 1, CAST(:payload AS jsonb))
                """),
                    {
                        "id": str(uuid4()),
                        "job_id": job_id,
                        "stage": stage,
                        "payload": json.dumps(payload),
                    },
                )
        command.upgrade(config, "head")
        command.check(config)
        with private.connect() as connection:
            image = connection.execute(
                text("SELECT id, original_path, normalized_path, width, height FROM images")
            ).one()
            assert tuple(map(str, image)) == (image_id, "/original", "/normalized.png", "16", "12")
            payloads = dict(
                connection.execute(text("SELECT stage, payload FROM model_results")).all()
            )
            assert payloads["annotation"] == caption
            assert payloads["detection"] == {
                "objects": [
                    {key: obj[key] for key in ("object_id", "bbox", "label")} for obj in objects
                ]
            }
            assert payloads["classification"] == {
                "objects": [
                    {"object_id": object_id, "label": "best"},
                    {"object_id": objects[1]["object_id"], "error": {"code": "crop_error"}},
                ]
            }
            inspector = inspect(connection)
            for table in Base.metadata.sorted_tables:
                assert {col["name"] for col in inspector.get_columns(table.name)} == set(
                    table.c.keys()
                )
            assert inspector.get_pk_constraint("model_results")["constrained_columns"] == [
                "job_id",
                "stage",
            ]
        with pytest.raises(RuntimeError, match="irreversible"):
            command.downgrade(config, "a2f004752f43")
    finally:
        private.dispose()
        with engine.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        engine.dispose()
