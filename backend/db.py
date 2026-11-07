"""Elasticsearch/S3 connections and idempotent local storage initialization."""

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from elasticsearch import BadRequestError, Elasticsearch

from backend.settings import Settings

INDEX_MAPPING = {
    "dynamic": "strict",
    "properties": {
        "width": {"type": "integer"},
        "height": {"type": "integer"},
        "created_at": {"type": "date"},
        "manifest": {
            "type": "object",
            "dynamic": False,
            "properties": {
                "annotation": {"properties": {"raw_caption": {"type": "text"}}},
                "detection": {
                    "properties": {
                        "objects": {
                            "type": "nested",
                            "properties": {
                                "object_id": {"type": "keyword"},
                                "label": {"type": "keyword"},
                                "bbox": {"type": "float", "index": False},
                            },
                        }
                    }
                },
                "classification": {
                    "properties": {
                        "objects": {
                            "type": "nested",
                            "properties": {
                                "object_id": {"type": "keyword"},
                                "object_caption": {"type": "text"},
                            },
                        }
                    }
                },
            },
        },
        "crop_files": {"type": "object", "enabled": False},
    },
}


def make_elasticsearch(settings: Settings) -> Elasticsearch:
    return Elasticsearch(settings.elasticsearch_url, request_timeout=5, max_retries=0)


def make_s3(settings: Settings):
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_endpoint_url,
        aws_access_key_id=settings.s3_access_key.get_secret_value(),
        aws_secret_access_key=settings.s3_secret_key.get_secret_value(),
        region_name=settings.s3_region,
        config=Config(
            connect_timeout=3,
            read_timeout=5,
            retries={"total_max_attempts": 1},
            s3={"addressing_style": "path"},
        ),
    )


def initialize(settings: Settings, es, s3) -> None:
    if not es.indices.exists(index=settings.elasticsearch_index):
        try:
            es.indices.create(
                index=settings.elasticsearch_index,
                settings={"number_of_shards": 1, "number_of_replicas": 0},
                mappings=INDEX_MAPPING,
            )
        except BadRequestError as exc:
            if exc.error != "resource_already_exists_exception":
                raise
    try:
        s3.head_bucket(Bucket=settings.s3_bucket)
    except ClientError as exc:
        if str(exc.response["Error"]["Code"]) not in {"404", "NoSuchBucket"}:
            raise
        options = (
            {}
            if settings.s3_region == "us-east-1"
            else {"CreateBucketConfiguration": {"LocationConstraint": settings.s3_region}}
        )
        try:
            s3.create_bucket(Bucket=settings.s3_bucket, **options)
        except ClientError as create_error:
            if create_error.response["Error"]["Code"] != "BucketAlreadyOwnedByYou":
                raise
    settings.pending_input_dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    settings = Settings()
    es, s3 = make_elasticsearch(settings), make_s3(settings)
    try:
        initialize(settings, es, s3)
    finally:
        es.close()
        s3.close()
