"""In-memory ES/S3 doubles for storage and CPU coordination contracts."""

from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError
from elastic_transport import ApiResponseMeta, NodeConfig
from elasticsearch import ConflictError, NotFoundError

from backend.repository import ImageRepository
from backend.settings import Settings

# PostgreSQL history is retained while pre-existing local migration edits are present.
# These tests target the retired storage, not the Elasticsearch application.
collect_ignore = [
    "test_migrations.py",
    "test_import_legacy.py",
    "test_worker_integration.py",
]


def es_error(error_type, status):
    return error_type(
        "synthetic failure",
        meta=ApiResponseMeta(status, "1.1", {}, 0, NodeConfig("http", "localhost", 9200)),
        body={},
    )


class MemoryES:
    def __init__(self):
        self.documents = {}
        self.indices = SimpleNamespace(exists=lambda **kwargs: True)

    def get(self, *, index, id):
        if id not in self.documents:
            raise es_error(NotFoundError, 404)
        return deepcopy(self.documents[id])

    def create(self, *, index, id, document):
        if id in self.documents:
            raise es_error(ConflictError, 409)
        self.documents[id] = {"_source": deepcopy(document), "_seq_no": 0, "_primary_term": 1}

    def index(self, *, index, id, document, if_seq_no, if_primary_term):
        old = self.documents[id]
        if (old["_seq_no"], old["_primary_term"]) != (if_seq_no, if_primary_term):
            raise es_error(ConflictError, 409)
        self.documents[id] = {
            "_source": deepcopy(document),
            "_seq_no": if_seq_no + 1,
            "_primary_term": 1,
        }

    def close(self):
        pass


class MemoryS3:
    def __init__(self):
        self.objects = {}

    def head_bucket(self, **kwargs):
        pass

    def put_object(self, *, Bucket, Key, Body, ContentType):
        self.objects[Key] = Body

    def get_object(self, *, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": BytesIO(self.objects[Key])}

    def close(self):
        pass


@pytest.fixture
def repository(tmp_path):
    settings = Settings(
        broker_url="amqp://unused",
        s3_secret_key="testing",
        pending_input_dir=tmp_path / "pending",
        lock_path=tmp_path / "inference.lock",
    )
    return ImageRepository(MemoryES(), MemoryS3(), settings)
