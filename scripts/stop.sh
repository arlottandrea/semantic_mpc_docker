#!/usr/bin/env bash
set -Eeuo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${root}"
exec docker compose --profile baseline --profile rl --profile nmpc down --remove-orphans
