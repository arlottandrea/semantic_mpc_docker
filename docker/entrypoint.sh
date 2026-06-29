#!/usr/bin/env bash
set -Eeo pipefail

source /opt/ros/noetic/setup.bash
source /workspace/devel/setup.bash

exec /workspace/docker/run-controller.sh "$@"
