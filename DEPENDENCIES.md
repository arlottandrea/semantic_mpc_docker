# Vendored source revisions

The top-level repository vendors snapshots of the following source trees so users do not need nested clones.

| Path | Upstream | Revision |
|---|---|---|
| `src/semantic_mpc` | `https://github.com/arlottandrea/semantic_mpc.git` | `9d337d1e021e55c89c67b6f2c07cf2252d68e1cb` |
| `src/ROS-TCP-Endpoint` | `https://github.com/arlottandrea/ROS-TCP-Endpoint.git` | `078820b0254c60fb3b80712179cb2ec460c838e5` |
| `src/yolov7-ros` | `https://github.com/arlottandrea/yolov7-ros.git` | `2c790635e62f972bb31c5c0d6700249fd46766de` |
| `src/vision_msgs` | `https://github.com/ros-perception/vision_msgs.git` | `0c916da414830456bd7d011af08ba150d1dc5627` |
| `src/active_rl_classification` | local vendored repository; no remote configured | `4a73406c6151e237e2329de8f3fb5c3c1b8b451d` |
| Docker build only: L4CasADi | `https://github.com/Tim-Salzmann/l4casadi.git` | `9fe2894533e05009bcbfa7706966745c5236fa4c` |

Nested `.git` directories in the development checkout are local metadata. They are ignored by the superproject and are not present in a fresh clone.
