"""Start the application from a checkout or unpacked archive; Python stdlib only."""

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def run(command, *, env, capture=False):
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )


def prepare_secrets(*, external_s3=False):
    path = ROOT / ".secrets" / "app.env"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    values = dict(
        line.split("=", 1)
        for line in existing.splitlines()
        if "=" in line and not line.lstrip().startswith("#")
    )
    s3_secret = (
        values.get("MINIO_ROOT_PASSWORD")
        or values.get("APP_S3_SECRET_KEY")
        or secrets.token_hex(24)
    )
    s3_user = values.get("MINIO_ROOT_USER") or values.get("APP_S3_ACCESS_KEY") or "kurs4"
    content = (ROOT / "backend" / "app.env.example").read_text(encoding="utf-8")
    content = content.replace("REPLACE_WITH_S3_SECRET_HEX", s3_secret)
    content = content.replace("REPLACE_WITH_BROKER_SECRET_HEX", secrets.token_hex(24))
    content = content.replace("MINIO_ROOT_USER=kurs4", "MINIO_ROOT_USER=" + s3_user)
    content = content.replace("APP_S3_ACCESS_KEY=kurs4", "APP_S3_ACCESS_KEY=" + s3_user)
    missing = [
        line
        for line in content.splitlines()
        if "=" in line
        and line.split("=", 1)[0] not in values
        and not (external_s3 and line.startswith("MINIO_"))
    ]
    if existing:
        if missing:
            with path.open("a", encoding="utf-8") as stream:
                stream.write(("" if existing.endswith("\n") else "\n") + "\n".join(missing) + "\n")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(
            path, "x", encoding="utf-8", opener=lambda p, flags: os.open(p, flags, 0o600)
        ) as f:
            f.write(content)
    except FileExistsError:
        pass


def use_external_s3(compose, env):
    if not (ROOT / ".secrets" / "app.env").exists():
        return False
    # Let Compose resolve dotenv quoting/interpolation exactly as for the containers.
    config = json.loads(run(compose + ["config", "--format", "json"], env=env, capture=True).stdout)
    values = config["services"]["backend"]["environment"]
    endpoint = (values.get("APP_S3_ENDPOINT_URL") or "").strip().rstrip("/")
    if not endpoint or endpoint == "http://s3:9000":
        return False
    url = urlsplit(endpoint)
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password:
        raise RuntimeError("APP_S3_ENDPOINT_URL должен быть HTTP(S)-адресом S3 без ключей доступа.")
    for name in ("APP_S3_ACCESS_KEY", "APP_S3_SECRET_KEY", "APP_S3_BUCKET", "APP_S3_REGION"):
        value = values.get(name) or ""
        if not value.strip() or value.startswith("REPLACE_WITH_"):
            raise RuntimeError(f"Для внешнего S3 задайте {name} в .secrets/app.env.")
    return True


def check_weights():
    specs = json.loads((ROOT / "backend" / "checkpoints.json").read_text(encoding="utf-8"))
    translation = json.loads((ROOT / "backend/translation.json").read_text(encoding="utf-8"))
    specs["translation"] = translation
    for stage, spec in specs.items():
        stage = "annotation" if stage == "translation" else stage
        directory = ROOT / stage / "models" / spec["model_name"]
        for item in spec["files"]:
            path = directory / item["path"]
            if not path.is_file():
                raise RuntimeError(f"Missing local model file: {path}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest() != item["sha256"]:
                raise RuntimeError(f"Model checksum mismatch: {path}")


def build_images_available(compose, services, env):
    config = json.loads(run(compose + ["config", "--format", "json"], env=env, capture=True).stdout)
    images = [config["services"][name]["image"] for name in services]
    try:
        run(["docker", "image", "inspect", *images], env=env, capture=True)
    except subprocess.CalledProcessError:
        return False
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "wsl-cuda"), required=True)
    parser.add_argument(
        "--check", action="store_true", help="Read-only preflight; do not start services"
    )
    parser.add_argument(
        "--build-if-missing",
        action="store_true",
        help="Reuse local release images; build only when an application image is missing",
    )
    args = parser.parse_args(argv)
    env = dict(os.environ, APP_DEVICE="cpu" if args.device == "cpu" else "cuda")
    env.setdefault("APP_WORKER_MEMORY", "8g" if args.device == "cpu" else "5g")
    compose = ["docker", "compose", "-p", "kurs4-app", "-f", "compose.app.yaml"]
    if args.device != "cpu":
        override = (
            "compose.app.wsl-gpu.yaml" if args.device == "wsl-cuda" else "compose.app.gpu.yaml"
        )
        compose += ["-f", override]
    compose += ["--profile", "processing"]
    try:
        run(["docker", "compose", "version"], env=env, capture=True)
        info = json.loads(
            run(["docker", "info", "--format", "{{json .}}"], env=env, capture=True).stdout
        )
        if info.get("OSType") != "linux" or info.get("Architecture") not in ("x86_64", "amd64"):
            raise RuntimeError("This bundle requires a Linux x86_64 Docker engine.")
        if args.device == "cuda" and "nvidia" not in (info.get("Runtimes") or {}):
            advice = (
                "Для прямого запуска в WSL используйте --device wsl-cuda."
                if Path("/dev/dxg").exists()
                else "Установите и настройте NVIDIA Container Toolkit для Docker."
            )
            raise RuntimeError(f"Docker не имеет runtime nvidia для режима cuda. {advice}")
        if args.device == "wsl-cuda" and not Path("/dev/dxg").exists():
            raise RuntimeError("Режим wsl-cuda требует /dev/dxg в локальной WSL.")
        external_s3 = use_external_s3(compose, env)
        if external_s3:
            compose += ["-f", "compose.app.external-s3.yaml"]
        run(compose + ["config", "--quiet", "--no-env-resolution"], env=env)
        check_weights()
        print("S3: внешний, локальный MinIO отключён." if external_s3 else "S3: локальный MinIO.")
        if args.check:
            print("Preflight passed. No services or settings were changed.")
            if not (ROOT / ".secrets" / "app.env").exists():
                print(".secrets/app.env will be generated on launch.")
            return 0
        prepare_secrets(external_s3=external_s3)
        lock = ROOT / "storage" / "locks" / "inference.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        if lock.is_symlink() or (lock.exists() and not lock.is_file()):
            raise RuntimeError(f"Inference lock must be a regular file: {lock}")
        try:
            with lock.open("x"):
                pass
            # The container uses UID 1000; host users may have another UID.
            lock.chmod(0o666)
        except FileExistsError:
            pass
        local_s3 = [] if external_s3 else ["s3"]
        build_services = [*local_s3, "backend", "worker", "frontend"]
        if not args.build_if_missing or not build_images_available(compose, build_services, env):
            run(compose + ["build", *build_services], env=env)
        run(compose + ["up", "-d", "--wait", "elasticsearch", *local_s3, "rabbitmq"], env=env)
        run(
            compose + ["run", "--rm", "--no-deps", "backend", "python", "-m", "backend.db"],
            env=env,
        )
        run(compose + ["up", "-d", "backend", "worker", "frontend"], env=env)
        if external_s3:
            # Stop a previous local deployment only after the new storage is initialized.
            run(compose + ["stop", "s3"], env=env)
        print("Services started. Open http://localhost:" + env.get("APP_FRONTEND_PORT", "8080"))
        print("Models initialize in the worker; inspect readiness with:")
        print(" ".join(compose + ["logs", "--tail", "100", "worker"]))
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Launch stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
