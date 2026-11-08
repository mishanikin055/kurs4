"""Windows desktop setup; stdlib only, with the existing application launcher."""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

from scripts import fetch_app_models, start_app

ROOT = Path(__file__).resolve().parents[1]
DOCKER_DOWNLOAD = "https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe"


def status(message):
    print("@@STATUS " + message, flush=True)


def powershell(script, *, timeout=None):
    executable = Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    return subprocess.run(
        [str(executable), "-NoProfile", "-NonInteractive", "-Command", script],
        check=True,
        timeout=timeout,
    )


def docker_paths():
    return [
        Path(os.environ["LOCALAPPDATA"]) / "Programs/DockerDesktop",
        Path(os.environ.get("ProgramW6432", r"C:\Program Files")) / "Docker/Docker",
    ]


def find_docker():
    for directory in docker_paths():
        cli = directory / "resources/bin/docker.exe"
        if cli.is_file():
            os.environ["PATH"] = str(cli.parent) + os.pathsep + os.environ.get("PATH", "")
            return str(cli)
    return shutil.which("docker")


def install_docker():
    status("1/4. Скачивание официального установщика Docker Desktop…")
    target = ROOT.parent / "downloads" / "Docker Desktop Installer.exe"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".part")
    try:
        with urllib.request.urlopen(DOCKER_DOWNLOAD, timeout=90) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output, 1024 * 1024)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    os.environ["IMAGE_ANALYSIS_DOCKER_INSTALLER"] = str(target)
    # Trust Windows certificate verification, not an unpinned download alone.
    powershell(
        "$ErrorActionPreference='Stop'; "
        "$s=Get-AuthenticodeSignature -LiteralPath $env:IMAGE_ANALYSIS_DOCKER_INSTALLER; "
        "if ($s.Status -ne 'Valid' -or "
        "$s.SignerCertificate.Subject -notmatch '(?:^|,\\s*)O=Docker Inc\\.?(?:,|$)') "
        "{ throw 'Docker installer signature is invalid' }",
        timeout=120,
    )
    status("1/4. Завершите установку Docker в открывшемся окне. Возможна перезагрузка.")
    # The native installer presents its own terms; do not silently accept a license.
    powershell(
        "$ErrorActionPreference='Stop'; "
        "$p=Start-Process -FilePath $env:IMAGE_ANALYSIS_DOCKER_INSTALLER "
        "-ArgumentList 'install','--backend=wsl-2' -Verb RunAs -Wait -PassThru; "
        "exit $p.ExitCode"
    )
    if not find_docker():
        raise RuntimeError("Завершите установку Docker, перезагрузите Windows и повторите запуск.")


def ensure_docker():
    status("1/4. Проверка Docker Desktop…")
    cli = find_docker()
    if not cli:
        install_docker()
        cli = find_docker()
    desktop = next(
        (p / "Docker Desktop.exe" for p in docker_paths() if (p / "Docker Desktop.exe").is_file()),
        None,
    )
    if desktop:
        subprocess.Popen([str(desktop)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    status(
        "1/4. Ожидание Docker. Если он запросит условия использования, подтвердите их в его окне."
    )
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        try:
            result = subprocess.run(
                [cli, "info", "--format", "{{json .}}"],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
            info = json.loads(result.stdout)
        except (OSError, ValueError, subprocess.SubprocessError):
            time.sleep(3)
            continue
        if info.get("OSType") != "linux" or info.get("Architecture") not in {"x86_64", "amd64"}:
            raise RuntimeError("В Docker Desktop включите Linux containers для x86_64.")
        return
    raise RuntimeError(
        "Docker не готов. Завершите его настройку; при ошибке WSL нажмите «Настроить WSL». "
        "Проверьте виртуализацию в BIOS/UEFI. После перезагрузки снова нажмите «Запустить»."
    )


def setup_wsl():
    status("Установка или обновление WSL. Подтвердите запрос Windows.")
    powershell(
        "$ErrorActionPreference='Stop'; "
        "$p=Start-Process -FilePath wsl.exe -ArgumentList '--install','--no-distribution' "
        "-Verb RunAs -Wait -PassThru; "
        "if ($p.ExitCode -ne 0 -and $p.ExitCode -ne 3010) { exit $p.ExitCode }; "
        "$p=Start-Process -FilePath wsl.exe -ArgumentList '--update' "
        "-Verb RunAs -Wait -PassThru; exit $p.ExitCode"
    )
    status(
        "WSL настроена. Если Windows просит перезагрузку, выполните её и снова запустите приложение."
    )


def wait_for_api(url, *, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url + "/api/health", timeout=5) as response:
                health = json.load(response)
            if all(health.get(key) is True for key in ("rabbitmq", "elasticsearch", "s3")):
                return
        except (OSError, ValueError):
            pass
        time.sleep(2)
    raise RuntimeError("Сервисы запущены, но приложение пока не отвечает. Откройте журнал.")


def self_test():
    """Exercise the actual embedded stdlib and payload, without Docker or inference."""
    import ssl
    import zipfile

    ssl.create_default_context()
    assert zipfile.is_zipfile(Path(sys.executable).parent / "python314.zip")
    files = list(fetch_app_models.sources())
    assert files and all(len(item[1]) == 64 for item in files)
    for relative in ("compose.app.yaml", "backend/Dockerfile", "frontend/index.html"):
        assert (ROOT / relative).is_file(), relative
    print("Windows payload and embedded Python: OK", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--action", choices=("start", "stop", "wsl", "self-test"), default="start")
    args = parser.parse_args(argv)
    try:
        if args.action == "self-test":
            self_test()
            return 0
        if sys.platform != "win32" or platform.machine().lower() not in {"amd64", "x86_64"}:
            raise RuntimeError("Требуется Windows x64. Для Linux используйте scripts/start_app.py.")
        if args.action == "wsl":
            setup_wsl()
            return 0
        if args.action == "stop":
            cli = find_docker()
            if not cli:
                raise RuntimeError("Docker не установлен; останавливать нечего.")
            status("Остановка приложения с сохранением данных…")
            subprocess.run(
                [
                    cli,
                    "compose",
                    "-p",
                    "kurs4-app",
                    "-f",
                    "compose.app.yaml",
                    "--profile",
                    "processing",
                    "stop",
                ],
                cwd=ROOT,
                check=True,
            )
            status("Приложение остановлено. Данные сохранены.")
            return 0
        ensure_docker()
        status("2/4. Подготовка моделей. Первая загрузка может занять много времени…")
        fetch_app_models.main([])
        status("3/4. Сборка и запуск приложения. Не закрывайте окно; первая сборка длительная…")
        if start_app.main(["--device", args.device, "--build-if-missing"]):
            raise RuntimeError("Не удалось запустить приложение. Подробности — в журнале.")
        port = int(os.environ.get("APP_FRONTEND_PORT", "8080"))
        if not 1 <= port <= 65535:
            raise ValueError("APP_FRONTEND_PORT должен быть в пределах 1–65535.")
        url = f"http://localhost:{port}"
        status("4/4. Проверка приложения…")
        wait_for_api(url)
        webbrowser.open(url)
        status(
            "Приложение открыто в браузере. Модели прогреваются; результат фотографии появится автоматически."
        )
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, SystemExit) as exc:
        print(f"Setup stopped: {exc}", flush=True)
        message = (
            str(exc)
            if isinstance(exc, RuntimeError)
            else (
                "Не удалось завершить настройку. Откройте журнал и повторите запуск после устранения ошибки."
            )
        )
        print(f"@@ERROR {message}", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
