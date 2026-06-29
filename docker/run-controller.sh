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
  ripe=(/models/nmpc/ripe/best_model_epoch_*.pth)
  raw=(/models/nmpc/raw/best_model_epoch_*.pth)
  if (( ${#ripe[@]} == 0 || ${#raw[@]} == 0 )); then
    echo "ERROR: NMPC requires both /models/nmpc/ripe/best_model_epoch_*.pth and /models/nmpc/raw/best_model_epoch_*.pth" >&2
    exit 66
  fi

  package_models=/workspace/src/semantic_mpc/semantic_mpc/src/semantic_mpc_package/models
  mkdir -p "${package_models}"
  ln -sfn /models/nmpc/ripe "${package_models}/ripe"
  ln -sfn /models/nmpc/raw "${package_models}/raw"
fi

exec roslaunch /workspace/docker/launch/runtime.launch \
  controller:="${controller}" \
  tcp_port:=10000 \
  yolo_device:="${YOLO_DEVICE:-cuda}" \
  nmpc_device:="${NMPC_DEVICE:-cuda}" \
  rl_policy_path:="${RL_POLICY_PATH:-/models/rl/final_model.zip}"
