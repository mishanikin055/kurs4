"""Start the application from a checkout or unpacked archive; Python stdlib only."""

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Existing databases are never migrated automatically: older revisions delete tables.
PREPARE_DATABASE = """
from alembic import command
from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, inspect, text
from backend.settings import Settings
from backend.tables import Base

config = Config('backend/alembic.ini')
head = ScriptDirectory.from_config(config).get_current_head()
with create_engine(Settings().database_url.get_secret_value()).connect() as connection:
    tables = inspect(connection).get_table_names()
    revision = None
    if 'alembic_version' in tables:
        revisions = connection.execute(text('SELECT version_num FROM alembic_version')).scalars().all()
        revision = revisions[0] if len(revisions) == 1 else None
if not tables:
    command.upgrade(config, 'head')
elif revision != head or set(tables) != set(Base.metadata.tables) | {'alembic_version'}:
    raise SystemExit('Existing database has an older or unknown schema. Stop: make a database '
                     'backup and follow the manual migration instructions in README.md.')
"""


def run(command, *, env, capture=False):
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )


def prepare_secrets():
    path = ROOT / ".secrets" / "app.env"
    if path.exists():
        return
    content = (ROOT / "backend" / "app.env.example").read_text(encoding="utf-8")
    for placeholder in ("REPLACE_WITH_RANDOM_HEX", "REPLACE_WITH_ANOTHER_RANDOM_HEX"):
        content = content.replace(placeholder, secrets.token_hex(24))
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(
            path, "x", encoding="utf-8", opener=lambda p, flags: os.open(p, flags, 0o600)
        ) as f:
            f.write(content)
    except FileExistsError:
        pass


def check_weights():
    specs = json.loads((ROOT / "backend" / "checkpoints.json").read_text(encoding="utf-8"))
    for stage, spec in specs.items():
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "wsl-cuda"), required=True)
    parser.add_argument(
        "--check", action="store_true", help="Read-only preflight; do not start services"
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
        run(compose + ["config", "--quiet", "--no-env-resolution"], env=env)
        check_weights()
        if args.device == "wsl-cuda" and not Path("/dev/dxg").exists():
            raise RuntimeError("wsl-cuda requires /dev/dxg in the local WSL installation.")
        if args.check:
            print("Preflight passed. No services or settings were changed.")
            if not (ROOT / ".secrets" / "app.env").exists():
                print(".secrets/app.env will be generated on launch.")
            return 0
        prepare_secrets()
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
        run(compose + ["build", "backend", "worker", "frontend"], env=env)
        run(compose + ["up", "-d", "--wait", "db", "rabbitmq"], env=env)
        run(
            compose + ["run", "--rm", "--no-deps", "backend", "python", "-c", PREPARE_DATABASE],
            env=env,
        )
        run(compose + ["up", "-d", "backend", "worker", "frontend"], env=env)
        print("Services started. Open http://localhost:" + env.get("APP_FRONTEND_PORT", "8080"))
        print("Models initialize in the worker; inspect readiness with:")
        print(" ".join(compose + ["logs", "--tail", "100", "worker"]))
        return 0
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"Launch stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
