#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
compose_args=(-f compose.detection.yaml)
service=tools
case "${1:-}" in
  benchmark|smoke)
    service=benchmark
    if [[ "$(uname -r)" == *microsoft* ]]; then
      docker_runtimes="$(docker info --format '{{json .Runtimes}}')"
      if [[ "$docker_runtimes" != *'"nvidia"'* ]]; then
        compose_args+=(-f compose.detection.wsl.yaml)
      fi
    fi
    ;;
esac
if [[ "${1:-}" == "build" ]]; then
  exec docker compose "${compose_args[@]}" build tools
fi
export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"
export DETECTION_IMAGE_ID="$(docker image inspect kurs4-detection:v1 --format '{{.Id}}')"
exec docker compose "${compose_args[@]}" run --rm "$service" "$@"
