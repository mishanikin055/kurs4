"""Host-side controller: stop only the isolated test database during model work."""

import os
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPOSE = ["docker", "compose", "-p", "kurs4-simple-test", "-f", "compose.app.yaml"]


def wait_marker(path: Path, process: subprocess.Popen, timeout: float = 35) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if process.poll() is not None:
            raise RuntimeError("Test exited before " + path.name)
        if time.monotonic() >= deadline:
            raise TimeoutError(path.name)
        time.sleep(0.1)


def main():
    with tempfile.TemporaryDirectory(prefix="kurs4-db-outage-") as directory:
        control = Path(directory)
        os.chmod(control, 0o777)
        command = [
            "docker",
            "run",
            "--rm",
            "--name",
            "kurs4-db-outage-check",
            "--init",
            "--network",
            "kurs4-simple-test_default",
            "--env-file",
            ".secrets/app.env",
            "-e",
            "APP_TEST_INTEGRATION=1",
            "-e",
            "APP_TEST_OUTAGE_CONTROL=/control",
            "-v",
            f"{control}:/control",
            "-v",
            f"{ROOT / 'backend'}:/app/backend:ro",
            "kurs4-backend-test:v2",
            "python",
            "-m",
            "pytest",
            "-q",
            "backend/tests/test_worker_integration.py::test_database_outage_requeues_completed_bundle",
        ]
        process = subprocess.Popen(command, cwd=ROOT)
        try:
            wait_marker(control / "model_started", process)
            subprocess.run([*COMPOSE, "stop", "db"], cwd=ROOT, check=True)
            wait_marker(control / "worker_stopped", process)
            subprocess.run([*COMPOSE, "up", "-d", "--wait", "db"], cwd=ROOT, check=True)
            (control / "db_restored").touch()
            if process.wait(timeout=60):
                raise RuntimeError("Database outage test failed")
        finally:
            subprocess.run([*COMPOSE, "up", "-d", "--wait", "db"], cwd=ROOT, check=True)
            if process.poll() is None:
                subprocess.run(["docker", "rm", "-f", "kurs4-db-outage-check"], check=False)
                process.wait(timeout=10)


if __name__ == "__main__":
    main()
