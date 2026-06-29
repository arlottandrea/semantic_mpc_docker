#!/usr/bin/env bash
set -Eeuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"

fail=0
for command in git docker; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "ERROR: ${command} is not installed." >&2
    fail=1
  fi
done

if ! git lfs version >/dev/null 2>&1; then
  echo "ERROR: Git LFS is required for the runtime models." >&2
  fail=1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "ERROR: Docker Compose v2 is required." >&2
  fail=1
fi

(( fail == 0 )) || exit 1

git lfs install --local
git lfs pull
mkdir -p runs/ros/log models/yolo models/rl models/nmpc/ripe models/nmpc/raw

if [[ -f src/yolov7-ros/weights/apples.pt && ! -s models/yolo/apples.pt ]]; then
  cp src/yolov7-ros/weights/apples.pt models/yolo/apples.pt
fi

if [[ -f src/active_rl_classification/artifacts/gym/models/final_model.zip && ! -s models/rl/final_model.zip ]]; then
  cp src/active_rl_classification/artifacts/gym/models/final_model.zip models/rl/final_model.zip
fi

(cd models && sha256sum --check checksums.sha256)
docker compose config --quiet

echo "Setup checks passed. Run: ./scripts/run.sh baseline"
