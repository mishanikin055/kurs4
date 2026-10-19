"""Resident model subprocesses and bounded local IPC, owned by the Celery supervisor."""

import json
import logging
import os
import select
import signal
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

from backend.processing import ProcessingInterrupted, inference_lock, stop_process_group
from backend.settings import Settings, model_specs

logger = logging.getLogger(__name__)
MAX_MESSAGE = 5 * 1024 * 1024


class ResidentModelFailed(ProcessingInterrupted):
    """The worker must restart before using any of its IPC channels again."""


def send(channel: socket.socket, message: dict) -> None:
    data = json.dumps(message, allow_nan=False).encode()
    if len(data) > MAX_MESSAGE:
        raise ValueError("Model message too large")
    channel.sendall(len(data).to_bytes(4, "big") + data)


def receive(channel: socket.socket, *, deadline: float | None = None, check=lambda: None) -> dict:
    def read(size):
        data = bytearray()
        while len(data) < size:
            check()
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Model response timed out")
                if not select.select([channel], [], [], min(0.1, remaining))[0]:
                    continue
            block = channel.recv(min(size - len(data), 65536))
            if not block:
                raise EOFError("Model channel closed")
            data.extend(block)
        return data

    size = int.from_bytes(read(4), "big")
    if not 0 < size <= MAX_MESSAGE:
        raise ValueError("Invalid model message size")
    message = json.loads(read(size))
    if not isinstance(message, dict):
        raise ValueError("Invalid model response")
    return message


def check_memory() -> None:
    """Keep host and container headroom; file cache is reclaimable."""
    info = Path("/proc/meminfo")
    if info.exists():
        available = next(
            line.split()[1]
            for line in info.read_text().splitlines()
            if line.startswith("MemAvailable:")
        )
        if int(available) * 1024 < 768 * 1024**2:
            raise MemoryError("Less than 768 MiB host RAM available")
    group = Path("/sys/fs/cgroup")
    if (group / "memory.max").exists():
        limit = (group / "memory.max").read_text().strip()
        if limit != "max":
            stats = dict(line.split() for line in (group / "memory.stat").read_text().splitlines())
            used = int(stats["anon"]) + int(stats["kernel"])
            if used > int(limit) - 512 * 1024**2:
                raise MemoryError("Less than 512 MiB container RAM headroom")


class ResidentModels:
    def __init__(self, settings: Settings, lock_fd: int, *, module="backend.models"):
        self.settings = settings
        self.lock_fd = lock_fd
        self.module = module
        self.channels: dict[str, socket.socket] = {}
        self.processes: dict[str, subprocess.Popen] = {}

    def start(self) -> None:
        for stage, spec in model_specs(self.settings).items():
            check_memory()
            channel, child = socket.socketpair()
            self.channels[stage] = channel
            try:
                process = subprocess.Popen(
                    [
                        getattr(self.settings, stage + "_python"),
                        "-m",
                        self.module,
                        "--serve",
                        str(child.fileno()),
                    ],
                    pass_fds=(self.lock_fd, child.fileno()),
                    start_new_session=True,
                    env={
                        **os.environ,
                        "HF_HUB_OFFLINE": "1",
                        "TRANSFORMERS_OFFLINE": "1",
                        "OMP_NUM_THREADS": "4",
                    },
                    # Local startup logs contain no request payloads; never expose them through API.
                    stdout=None,
                    stderr=None,
                )
                self.processes[stage] = process
            finally:
                child.close()
            spec = dict(spec, stage=stage, model_root=str(self.settings.model_root))
            send(channel, spec)
            response = receive(
                channel, deadline=time.monotonic() + self.settings.stage_timeout, check=check_memory
            )
            if not response.get("ready"):
                raise RuntimeError(f"Model startup failed: {stage}")
            if response.get("vram_free") is not None and response["vram_free"] < 1024**3:
                raise MemoryError("Less than 1 GiB GPU headroom after warmup")
            logger.warning(
                "Resident model ready stage=%s startup_s=%.3f", stage, response.get("startup_s", 0)
            )

    def __call__(self, request: dict, settings: Settings, lock_fd: int, timeout: float) -> dict:
        channel = self.channels[request["stage"]]
        try:
            check_memory()
            channel.settimeout(max(0.01, timeout))
            deadline = time.monotonic() + timeout
            send(channel, request)
            response = receive(channel, deadline=deadline, check=check_memory)
            if response.get("error", {}).get("code") == "cuda_oom":
                raise MemoryError("CUDA out of memory")
            return response
        except (OSError, EOFError, ValueError, MemoryError) as exc:
            raise ResidentModelFailed from exc
        finally:
            channel.settimeout(None)

    def close(self) -> None:
        for process in self.processes.values():
            stop_process_group(process)
        for channel in self.channels.values():
            channel.close()


@contextmanager
def resident_models(settings: Settings, *, module="backend.models"):
    with inference_lock(settings.lock_path, time.monotonic() + settings.job_timeout) as lock_fd:
        runtime = ResidentModels(settings, lock_fd, module=module)

        def interrupted(signum, frame):
            raise ProcessingInterrupted

        previous = signal.signal(signal.SIGTERM, interrupted)
        try:
            try:
                runtime.start()
            finally:
                signal.signal(signal.SIGTERM, previous)
            yield runtime
        finally:
            runtime.close()
