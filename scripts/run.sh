#!/usr/bin/env bash
set -Eeuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"

export RUNTIME_UID="${RUNTIME_UID:-$(id -u)}"
export RUNTIME_GID="${RUNTIME_GID:-$(id -g)}"

controller="${1:-}"
mode="${2:-}"
case "${controller}" in
  baseline|rl|nmpc) ;;
  *) echo "usage: run.sh {baseline|rl|nmpc} [--cpu]" >&2; exit 64 ;;
esac

compose_args=(-f compose.yaml)
if [[ "${mode}" == "--cpu" ]]; then
  compose_args+=(-f compose.cpu.yaml)
  export YOLO_DEVICE=cpu NMPC_DEVICE=cpu
elif [[ -n "${mode}" ]]; then
  echo "usage: run.sh {baseline|rl|nmpc} [--cpu]" >&2
  exit 64
else
  compose_args+=(-f compose.gpu.yaml)
fi

mkdir -p runs/home runs/ros/log
./scripts/doctor.sh "${controller}"
exec docker compose "${compose_args[@]}" --profile "${controller}" up --build "${controller}"
