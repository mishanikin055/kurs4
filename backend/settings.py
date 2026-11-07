from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", extra="ignore")

    elasticsearch_url: str = "http://elasticsearch:9200"
    elasticsearch_index: str = "image-results"
    s3_endpoint_url: str = "http://s3:9000"
    s3_access_key: SecretStr = SecretStr("kurs4")
    s3_secret_key: SecretStr
    s3_bucket: str = "image-analysis"
    s3_region: str = "us-east-1"
    pending_input_dir: Path = Path("/var/lib/kurs4/pending")
    broker_url: SecretStr
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)
    max_image_pixels: int = Field(default=25_000_000, gt=0)
    max_artifact_bytes: int = Field(default=128 * 1024 * 1024, gt=0, lt=1024**3)
    max_bundle_bytes: int = Field(default=256 * 1024 * 1024, gt=0)
    broker_timeout: float = Field(default=3, gt=0, le=30)
    lock_path: Path = Path("storage/locks/inference.lock")
    model_root: Path = Path(".")
    detection_python: str = "/opt/detection/bin/python"
    annotation_python: str = "/opt/annotation/bin/python"
    device: str = "cuda"
    stage_timeout: float = Field(default=600, gt=0)
    job_timeout: float = Field(default=2400, gt=0, lt=6000)


def model_specs(settings: Settings) -> dict:
    import json

    specs = json.loads(Path(__file__).with_name("checkpoints.json").read_text())
    specs["classification"] = {
        **specs["annotation"],
        "parameters": {
            **specs["annotation"]["parameters"],
            "prompt": "<CAPTION>",
            "max_new_tokens": 96,
            "crop_padding_fraction": 0.1,
        },
    }
    return {
        stage: {
            **{key: spec[key] for key in ("model_name", "model_revision", "weights_sha256")},
            "parameters": {
                **spec["parameters"],
                "precision": (
                    "fp32" if settings.device == "cpu" else spec["parameters"]["precision"]
                ),
                "device": settings.device,
            },
        }
        for stage, spec in specs.items()
    }
