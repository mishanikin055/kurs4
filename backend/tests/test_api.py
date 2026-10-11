from fastapi.testclient import TestClient

from backend.main import create_app
from backend.settings import Settings
from backend.tests.test_storage import image_bytes


def test_file_size_is_checked_before_database_access(tmp_path):
    app = create_app(
        Settings(
            database_url="postgresql+psycopg://unused:unused@127.0.0.1:1/unused",
            broker_url="amqp://unused:unused@127.0.0.1:1/unused",
            image_root=tmp_path,
            max_upload_bytes=50,
        ),
        publisher_enabled=False,
    )
    with TestClient(app) as client:
        response = client.post("/images", files={"file": ("large.png", image_bytes())})
    assert response.status_code == 413
    assert response.json()["detail"] == "upload_too_large"
    assert not list(tmp_path.iterdir())


def test_streaming_body_limit_without_content_length(tmp_path):
    app = create_app(
        Settings(
            database_url="postgresql+psycopg://unused:unused@127.0.0.1:1/unused",
            broker_url="amqp://unused:unused@127.0.0.1:1/unused",
            image_root=tmp_path,
            max_upload_bytes=50,
        ),
        publisher_enabled=False,
    )

    def body():
        yield b'--test\r\nContent-Disposition: form-data; name="file"; filename="big.png"\r\n\r\n'
        for _ in range(80):
            yield b"x" * 1024
        yield b"\r\n--test--\r\n"

    with TestClient(app) as client:
        response = client.post(
            "/images",
            content=body(),
            headers={"Content-Type": "multipart/form-data; boundary=test"},
        )
    assert response.status_code == 413
    assert not list(tmp_path.iterdir())
