"""Metadata in Elasticsearch, result PNGs in S3, pending inputs in shared scratch."""

import json
import os
import tempfile
from datetime import datetime, timezone
from hashlib import sha256
from types import SimpleNamespace
from uuid import UUID, uuid4

from botocore.exceptions import ClientError
from elasticsearch import ConflictError, NotFoundError

from backend.schemas import ImageResult
from backend.settings import Settings
from backend.storage import check_artifacts


class ImageRepository:
    def __init__(self, es, s3, settings: Settings):
        self.es, self.s3, self.settings = es, s3, settings
        self.index = settings.elasticsearch_index
        self.bucket = settings.s3_bucket

    def _document(self, image_id: UUID) -> dict:
        try:
            return self.es.get(index=self.index, id=str(image_id))
        except NotFoundError as exc:
            raise FileNotFoundError from exc

    def get(self, image_id: UUID) -> SimpleNamespace | None:
        try:
            source = self._document(image_id)["_source"]
        except FileNotFoundError:
            return None
        return SimpleNamespace(**source) if source.get("manifest") is not None else None

    def _input_path(self, image_id: UUID):
        return self.settings.pending_input_dir / f"{UUID(str(image_id))}.png"

    def input(self, image_id: UUID) -> tuple[SimpleNamespace, dict[str, bytes]]:
        source = self._document(image_id)["_source"]
        return SimpleNamespace(**source), {
            "normalized.png": self._input_path(image_id).read_bytes()
        }

    def create_input(
        self, image_id: UUID, width: int, height: int, artifacts: dict[str, bytes]
    ) -> None:
        check_artifacts(artifacts, self.settings)
        directory = self.settings.pending_input_dir
        directory.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(dir=directory, prefix=".upload-")
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(artifacts["normalized.png"])
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._input_path(image_id))
            directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            from pathlib import Path

            Path(temporary).unlink(missing_ok=True)
        # Retain scratch on an ambiguous ES timeout; a document may have been created.
        self.es.create(
            index=self.index,
            id=str(image_id),
            document={
                "width": width,
                "height": height,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "manifest": None,
                "crop_files": {},
            },
        )

    def cleanup_input(self, image_id: UUID) -> None:
        self._input_path(image_id).unlink(missing_ok=True)

    def publish(self, result: ImageResult, artifacts: dict[str, bytes], manifest: dict) -> None:
        document = self._document(result.id)
        source = document["_source"]
        if source.get("manifest") is not None:
            return
        if (source["width"], source["height"]) != (result.width, result.height):
            raise ValueError("Conflicting image dimensions")
        expected = {f"crops/{obj.object_id}.png" for obj in result.detection.objects}
        expected.add("detection.png")
        if set(artifacts) != expected | {"results.json"}:
            raise ValueError("Expected crops, detection.png and results.json")
        if (
            ImageResult.model_validate(manifest) != result
            or manifest.get("detection_image") != "detection.png"
            or set(manifest["crops"])
            != {name.removeprefix("crops/") for name in expected if name.startswith("crops/")}
        ):
            raise ValueError("Conflicting result manifest")
        check_artifacts(artifacts, self.settings)
        attempt = uuid4()
        files = {}
        for name in sorted(expected):
            content = artifacts[name]
            key = f"{result.id}/{attempt}/{name}"
            self.s3.put_object(Bucket=self.bucket, Key=key, Body=content, ContentType="image/png")
            files[name] = {"key": key, "size": len(content), "sha256": sha256(content).hexdigest()}
        # Keep the existing crop_files field for all PNG references; no index migration needed.
        # Publish only after every PNG is stored; ambiguous writes must not trigger deletion.
        try:
            self.es.index(
                index=self.index,
                id=str(result.id),
                document={**source, "manifest": manifest, "crop_files": files},
                if_seq_no=document["_seq_no"],
                if_primary_term=document["_primary_term"],
            )
        except ConflictError:
            if self.get(result.id) is None:
                raise

    def artifact(self, image_id: UUID, name: str) -> SimpleNamespace:
        document = self._document(image_id)["_source"]
        if document.get("manifest") is None:
            raise FileNotFoundError
        if name == "results.json":
            return SimpleNamespace(
                content_type="application/json",
                content=json.dumps(
                    document["manifest"], ensure_ascii=False, indent=2, allow_nan=False
                ).encode("utf-8"),
            )
        reference = document["crop_files"].get(name)
        if reference is None:
            raise FileNotFoundError
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=reference["key"])
        except ClientError as exc:
            if exc.response["Error"]["Code"] in {"NoSuchKey", "404"}:
                raise OSError("Missing stored image") from exc
            raise
        body = response["Body"]
        try:
            content = body.read(self.settings.max_artifact_bytes + 1)
        finally:
            body.close()
        if len(content) != reference["size"] or sha256(content).hexdigest() != reference["sha256"]:
            raise OSError("Stored image integrity check failed")
        return SimpleNamespace(content_type="image/png", content=content)

    def bundle(self, image_id: UUID) -> dict[str, tuple[str, bytes]]:
        document = self._document(image_id)["_source"]
        if document.get("manifest") is None:
            raise FileNotFoundError
        files, size = {}, 0
        for name in ["results.json", *document["crop_files"]]:
            artifact = self.artifact(image_id, name)
            size += len(artifact.content)
            if size > self.settings.max_bundle_bytes:
                from backend.storage import BundleTooLarge

                raise BundleTooLarge("Bundle exceeds configured byte limit")
            files[name] = (artifact.content_type, artifact.content)
        return files

    def health(self) -> dict[str, bool]:
        es_ready, s3_ready = False, False
        try:
            es_ready = bool(self.es.indices.exists(index=self.index))
        except Exception:
            pass
        try:
            self.s3.head_bucket(Bucket=self.bucket)
            s3_ready = True
        except Exception:
            pass
        return {"elasticsearch": es_ready, "s3": s3_ready}
