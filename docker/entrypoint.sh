#!/usr/bin/env bash
# Catkin's generated environment hooks inspect variables that may legitimately
# be unset, so nounset cannot be enabled while sourcing the ROS environments.
set -Eeo pipefail

source /opt/ros/noetic/setup.bash
source /workspace/devel/setup.bash

if ! mkdir -p "${ROS_LOG_DIR:-/runs/ros/log}"; then
  echo "ERROR: cannot create ROS log directory ${ROS_LOG_DIR:-/runs/ros/log}; check ownership of the host runs/ directory." >&2
  exit 73
fi

exec /workspace/docker/run-controller.sh "$@"
