#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
compose_args=(-f compose.annotation.yaml)
service=tools
case "${1:-}" in
  benchmark|smoke|dev)
    service=benchmark
    if [[ "$(uname -r)" == *microsoft* ]]; then
      docker_runtimes="$(docker info --format '{{json .Runtimes}}')"
      if [[ "$docker_runtimes" != *'"nvidia"'* ]]; then
        compose_args+=(-f compose.annotation.wsl.yaml)
      fi
    fi
    ;;
  report|expert) service=cpu ;;
esac
export LOCAL_UID="$(id -u)" LOCAL_GID="$(id -g)"
if [[ "${1:-}" == "build" ]]; then
  exec docker compose "${compose_args[@]}" build tools
fi
export ANNOTATION_IMAGE_ID="$(docker image inspect kurs4-annotation:v1 --format '{{.Id}}')"
export PROJECT_GIT_COMMIT="$(git rev-parse HEAD)"
export PROJECT_GIT_STATUS="$(git status --porcelain)"
exec docker compose "${compose_args[@]}" run --rm "$service" "$@"
