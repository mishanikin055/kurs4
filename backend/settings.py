from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", extra="ignore")

    database_url: SecretStr
    broker_url: SecretStr
    image_root: Path = Path("/var/lib/kurs4/images")
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    max_image_pixels: int = Field(default=25_000_000, gt=0)
    broker_timeout: float = Field(default=3, gt=0, le=30)
    lock_path: Path = Path("storage/locks/inference.lock")
    model_root: Path = Path(".")
    detection_python: str = "/opt/detection/bin/python"
    classification_python: str = "/opt/detection/bin/python"
    annotation_python: str = "/opt/annotation/bin/python"
    device: str = "cuda"
    stage_timeout: float = Field(default=600, gt=0)
    job_timeout: float = Field(default=2400, gt=0, lt=6000)


def model_specs(settings: Settings) -> dict:
    import json

    specs = json.loads(Path(__file__).with_name("checkpoints.json").read_text())
    return {
        stage: {
            **{key: spec[key] for key in ("model_name", "model_revision", "weights_sha256")},
            "parameters": {**spec["parameters"], "device": settings.device},
        }
        for stage, spec in specs.items()
    }
