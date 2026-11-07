"""S3 configuration and bucket initialization contracts without network access."""

from unittest.mock import Mock

import pytest
from botocore.exceptions import ClientError

from backend.db import initialize, make_s3
from backend.settings import Settings


@pytest.fixture
def settings(tmp_path):
    return Settings(
        s3_endpoint_url="https://objects.example.test",
        s3_access_key="test-access",
        s3_secret_key="test-secret",
        s3_bucket="test-crops",
        s3_region="eu-west-1",
        broker_url="amqp://unused",
        pending_input_dir=tmp_path / "pending",
    )


@pytest.mark.parametrize(
    "endpoint,expected",
    [
        ("", "http://s3:9000"),
        (" \t ", "http://s3:9000"),
        (" http://s3:9000/ ", "http://s3:9000"),
        (" https://objects.example.test/ ", "https://objects.example.test"),
    ],
)
def test_settings_normalize_s3_endpoint(settings, endpoint, expected):
    values = settings.model_dump()
    values["s3_endpoint_url"] = endpoint
    assert Settings(**values).s3_endpoint_url == expected


def test_client_uses_configured_s3_and_keeps_tls_verification(settings, monkeypatch):
    client = Mock()
    monkeypatch.setattr("backend.db.boto3.client", client)

    assert make_s3(settings) is client.return_value

    args, kwargs = client.call_args
    assert args == ("s3",)
    assert kwargs["endpoint_url"] == "https://objects.example.test"
    assert kwargs["aws_access_key_id"] == "test-access"
    assert kwargs["aws_secret_access_key"] == "test-secret"
    assert kwargs["region_name"] == "eu-west-1"
    assert kwargs["config"].s3["addressing_style"] == "path"
    assert kwargs.get("verify", True) is True
    assert kwargs.get("use_ssl", True) is True


def test_existing_bucket_does_not_require_create_permission(settings):
    es, s3 = Mock(), Mock()
    es.indices.exists.return_value = True

    initialize(settings, es, s3)

    s3.head_bucket.assert_called_once_with(Bucket="test-crops")
    s3.create_bucket.assert_not_called()
    assert settings.pending_input_dir.is_dir()


@pytest.mark.parametrize("code", ["404", "NoSuchBucket"])
@pytest.mark.parametrize("region", ["us-east-1", "eu-west-1"])
def test_missing_bucket_uses_region_correct_creation(settings, code, region):
    settings.s3_region = region
    es, s3 = Mock(), Mock()
    es.indices.exists.return_value = True
    s3.head_bucket.side_effect = ClientError({"Error": {"Code": code}}, "HeadBucket")

    initialize(settings, es, s3)

    expected = {"Bucket": "test-crops"}
    if region != "us-east-1":
        expected["CreateBucketConfiguration"] = {"LocationConstraint": region}
    s3.create_bucket.assert_called_once_with(**expected)


def test_forbidden_bucket_is_not_recreated(settings):
    es, s3 = Mock(), Mock()
    es.indices.exists.return_value = True
    error = ClientError({"Error": {"Code": "403"}}, "HeadBucket")
    s3.head_bucket.side_effect = error

    with pytest.raises(ClientError) as caught:
        initialize(settings, es, s3)

    assert caught.value is error
    s3.create_bucket.assert_not_called()
    assert not settings.pending_input_dir.exists()
