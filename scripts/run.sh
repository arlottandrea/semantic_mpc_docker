#!/usr/bin/env bash
set -Eeuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"

export RUNTIME_UID="${RUNTIME_UID:-$(id -u)}"
export RUNTIME_GID="${RUNTIME_GID:-$(id -g)}"

usage() {
  echo "usage: run.sh {baseline|rl|nmpc} [--cpu|--gpu] [--gp-rrt|--no-gp-rrt]" >&2
}

controller="${1:-}"
if [[ -n "${controller}" ]]; then
  shift
fi
case "${controller}" in
  baseline|rl|nmpc) ;;
  *) usage; exit 64 ;;
esac

mode="gpu"
nmpc_use_gp_rrt="${NMPC_USE_GP_RRT:-true}"
gp_rrt_option_seen=false

while (($#)); do
  case "$1" in
    --cpu)
      mode="cpu"
      ;;
    --gpu)
      mode="gpu"
      ;;
    --gp-rrt)
      nmpc_use_gp_rrt="true"
      gp_rrt_option_seen=true
      ;;
    --no-gp-rrt)
      nmpc_use_gp_rrt="false"
      gp_rrt_option_seen=true
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 64
      ;;
  esac
  shift
done

if [[ "${controller}" != "nmpc" && "${gp_rrt_option_seen}" == true ]]; then
  echo "ERROR: --gp-rrt/--no-gp-rrt only applies to nmpc." >&2
  exit 64
fi

compose_args=(-f compose.yaml)
if [[ "${mode}" == "cpu" ]]; then
  compose_args+=(-f compose.cpu.yaml)
  export YOLO_DEVICE=cpu NMPC_DEVICE=cpu
else
  compose_args+=(-f compose.gpu.yaml)
fi
export NMPC_USE_GP_RRT="${nmpc_use_gp_rrt}"

mkdir -p runs/home runs/ros/log
./scripts/doctor.sh "${controller}"
exec docker compose "${compose_args[@]}" --profile "${controller}" up --build "${controller}"
