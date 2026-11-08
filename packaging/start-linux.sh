#!/bin/sh
# Linux x86_64 with an existing local Docker Engine and Compose v2.
set -eu
bundle_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
app_root="$bundle_root/app"
offline_root="$bundle_root/offline"
device=${1:-cpu}
action=${2:-start}
case "$device" in cpu|cuda|wsl-cuda) ;; *) echo 'Устройство: cpu, cuda или wsl-cuda.' >&2; exit 1 ;; esac
case "$action" in start|check|stop) ;; *) echo 'Действие: start, check или stop.' >&2; exit 1 ;; esac
case "$(docker info --format '{{.OSType}}/{{.Architecture}}')" in linux/x86_64|linux/amd64) ;; *) echo 'Нужен локальный Linux x86_64 Docker Engine.' >&2; exit 1 ;; esac
docker compose version
# Only local engines can mount these paths; do not send the bundle to a remote daemon.
endpoint=${DOCKER_HOST:-$(docker context inspect --format '{{.Endpoints.docker.Host}}')}
case "$endpoint" in unix://*) ;; *) echo 'Используйте локальный Docker Engine с unix socket.' >&2; exit 1 ;; esac
cd "$app_root"
set -- docker compose -p "${APP_COMPOSE_PROJECT:-kurs4-app}" -f compose.app.yaml -f compose.app.offline.yaml --profile processing
if [ "$action" = stop ]; then "$@" stop; exit; fi
if [ "$device" = cuda ]; then
  case "$(docker info --format '{{json .Runtimes}}')" in *'"nvidia"'*) ;; *) echo 'Сначала настройте NVIDIA Container Toolkit или выберите cpu.' >&2; exit 1 ;; esac
  set -- "$@" -f compose.app.gpu.yaml
fi
if [ "$device" = wsl-cuda ]; then
  [ -e /dev/dxg ] || { echo 'Режиму wsl-cuda нужен /dev/dxg.' >&2; exit 1; }
  set -- "$@" -f compose.app.wsl-gpu.yaml
fi
APP_DEVICE=cpu
APP_WORKER_MEMORY=${APP_WORKER_MEMORY:-8g}
if [ "$device" != cpu ]; then APP_DEVICE=cuda; APP_WORKER_MEMORY=${APP_WORKER_MEMORY_GPU:-5g}; fi
export APP_DEVICE APP_WORKER_MEMORY
images_ready() {
  while read -r expected image; do
    actual=$(docker image inspect --format '{{.Id}}' "$image" 2>/dev/null) || return 1
    [ "$actual" = "$expected" ] || return 1
  done < "$offline_root/images.txt"
}
if ! images_ready; then
  echo 'Проверка и загрузка готовых Docker-образов из архива…'
  (cd "$offline_root" && sha256sum -c images.sha256)
  docker load --input "$offline_root/images.tar.gz"
  images_ready || { echo 'Docker-образы не совпадают с поставкой.' >&2; exit 1; }
fi
backend_image=$(cat "$offline_root/backend-image.txt")
echo 'Подготовка моделей и локальных настроек без сети…'
docker run --rm --pull never --network none --user "$(id -u):$(id -g)" \
  --volume "$app_root:/bundle" --volume "$offline_root:/offline:ro" \
  --workdir /bundle --entrypoint python "$backend_image" \
  -m scripts.offline_setup --bundle /offline
s3_mode=$("$@" config --format json | docker run --rm -i --pull never --network none \
  --volume "$app_root:/bundle:ro" --workdir /bundle --entrypoint python "$backend_image" \
  -m scripts.offline_setup --detect-s3)
local_s3=s3
if [ "$s3_mode" = external ]; then
  local_s3=
  set -- "$@" -f compose.app.external-s3.yaml
fi
"$@" config --quiet
if [ "$action" = check ]; then echo 'Офлайн-поставка подготовлена. Сервисы не запускались.'; exit; fi
"$@" up -d --wait --pull never --no-build elasticsearch $local_s3 rabbitmq
"$@" run --rm --no-deps --pull never backend python -m backend.db
"$@" up -d --pull never --no-build backend worker frontend
if [ "$s3_mode" = external ]; then "$@" stop s3; fi
url="http://localhost:${APP_FRONTEND_PORT:-8080}"
echo "Приложение запущено: $url. Модели прогреваются в worker."
if command -v xdg-open >/dev/null 2>&1; then xdg-open "$url" >/dev/null 2>&1 || true; fi
