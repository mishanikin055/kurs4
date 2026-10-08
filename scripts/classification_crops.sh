#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
compose_args=(-f compose.classification.yaml -f compose.classification.crops.yaml)
service=cpu
if [[ "${1:-}" == "benchmark" ]]; then
  service=benchmark
  if [[ "$(uname -r)" == *microsoft* ]]; then
    docker_runtimes="$(docker info --format '{{json .Runtimes}}')"
    if [[ "$docker_runtimes" != *'"nvidia"'* ]]; then
      compose_args+=(-f compose.classification.wsl.yaml)
    fi
  fi
fi
export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"
export CLASSIFICATION_IMAGE_ID="$(docker image inspect kurs4-classification:v1 --format '{{.Id}}')"
export PROJECT_GIT_COMMIT="$(git rev-parse HEAD)"
export PROJECT_GIT_STATUS="$(git status --porcelain)"
exec docker compose "${compose_args[@]}" run --rm --entrypoint python "$service" -m classification.crops.cli "$@"
