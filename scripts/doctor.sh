#!/usr/bin/env bash
set -uo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
controller="${1:-baseline}"
errors=0

check_file() {
  if [[ -s "$1" ]]; then
    printf 'OK    %s\n' "$1"
  else
    printf 'ERROR %s is missing or empty\n' "$1" >&2
    errors=$((errors + 1))
  fi
}

case "${controller}" in
  baseline|rl|nmpc) ;;
  *) echo "usage: doctor.sh {baseline|rl|nmpc}" >&2; exit 64 ;;
esac

command -v docker >/dev/null 2>&1 || { echo "ERROR docker is not installed" >&2; errors=$((errors + 1)); }
docker compose version >/dev/null 2>&1 || { echo "ERROR Docker Compose v2 is unavailable" >&2; errors=$((errors + 1)); }
docker info >/dev/null 2>&1 || { echo "ERROR Docker daemon is unavailable" >&2; errors=$((errors + 1)); }

check_file models/yolo/apples.pt
[[ "${controller}" != rl ]] || check_file models/rl/final_model.zip

if [[ "${controller}" == nmpc ]]; then
  compgen -G 'models/nmpc/ripe/best_model_epoch_*.pth' >/dev/null || { echo "ERROR ripe NMPC checkpoint is missing" >&2; errors=$((errors + 1)); }
  compgen -G 'models/nmpc/raw/best_model_epoch_*.pth' >/dev/null || { echo "ERROR raw NMPC checkpoint is missing" >&2; errors=$((errors + 1)); }
fi

if command -v ss >/dev/null 2>&1 && ss -ltn "sport = :${UNITY_TCP_PORT:-10000}" | tail -n +2 | grep -q .; then
  echo "ERROR TCP port ${UNITY_TCP_PORT:-10000} is already in use" >&2
  errors=$((errors + 1))
fi

if [[ "${YOLO_DEVICE:-cuda}" == cuda ]] && ! docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
  echo "ERROR NVIDIA Container Toolkit is unavailable; use --cpu or install it." >&2
  errors=$((errors + 1))
fi

if (( errors > 0 )); then
  echo "Doctor found ${errors} problem(s)." >&2
  exit 1
fi

echo "Doctor checks passed for ${controller}."
