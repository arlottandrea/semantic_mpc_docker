#!/usr/bin/env bash
set -Eeuo pipefail

controller="${1:-}"
case "${controller}" in
  baseline|rl|nmpc) ;;
  *) echo "usage: run-controller.sh {baseline|rl|nmpc}" >&2; exit 64 ;;
esac

require_file() {
  if [[ ! -s "$1" ]]; then
    echo "ERROR: required runtime model is missing or empty: $1" >&2
    exit 66
  fi
}

require_file /models/yolo/apples.pt

if [[ "${controller}" == "rl" ]]; then
  require_file "${RL_POLICY_PATH:-/models/rl/final_model.zip}"
fi

if [[ "${controller}" == "nmpc" ]]; then
  shopt -s nullglob
  perception=(/models/nmpc/best_model_epoch_*.pth)
  if (( ${#perception[@]} == 0 )); then
    echo "ERROR: NMPC requires /models/nmpc/best_model_epoch_*.pth" >&2
    exit 66
  fi

fi

exec roslaunch /workspace/docker/launch/runtime.launch \
  controller:="${controller}" \
  tcp_port:=10000 \
  yolo_device:="${YOLO_DEVICE:-cuda}" \
  nmpc_device:="${NMPC_DEVICE:-cuda}" \
  rl_policy_path:="${RL_POLICY_PATH:-/models/rl/final_model.zip}"
