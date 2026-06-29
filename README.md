# Semantic MPC Linux runtime

This repository is the Linux-first runtime for the baseline, RL, and neural MPC controllers. ROS Noetic, YOLOv7, data association, the selected controller, and the Unity ROS TCP endpoint run in one isolated container. Unity runs natively on the host and connects through TCP.

## Runtime layout

```text
Unity on the Linux host
        |
        | TCP 127.0.0.1:10000
        v
Docker container
  ROS master + ROS TCP endpoint
  YOLOv7 + data association
  exactly one of: baseline | RL | NMPC
```

Only port 10000 is exposed, and it is bound to host loopback. ROS port 11311 remains inside the container.

## Host prerequisites

- Linux x86-64
- Git and Git LFS
- Docker Engine with Compose v2
- For GPU mode: NVIDIA driver 520.61.05 or newer and NVIDIA Container Toolkit
- A Unity build containing Unity's ROS TCP Connector and the matching project messages

Docker is the only ROS/Python installation required on the host.

## Clone and setup

Replace the URL below with the remote of this top-level repository after it is published.

```bash
git clone <superproject-url> semantic-mpc-runtime
cd semantic-mpc-runtime
./scripts/setup.sh
```

`setup.sh` pulls Git LFS models, verifies checksums, creates the writable `runs/` directory, and validates the Compose configuration.

## Start a controller

GPU mode:

```bash
./scripts/run.sh baseline
./scripts/run.sh rl
./scripts/run.sh nmpc
```

CPU mode is useful for baseline/RL validation, but YOLO will be substantially slower:

```bash
./scripts/run.sh baseline --cpu
./scripts/run.sh rl --cpu
```

Only run one controller profile at a time. Stop the stack with `Ctrl-C`, followed by:

```bash
./scripts/stop.sh
```

Run preflight checks without starting anything:

```bash
./scripts/doctor.sh baseline
./scripts/doctor.sh rl
./scripts/doctor.sh nmpc
```

## Connect Unity

1. Start one Docker controller profile.
2. Wait until Compose reports the service as healthy, or until `/unity_endpoint` appears in the logs.
3. Configure the Unity ROS TCP Connector for IP `127.0.0.1` and port `10000`.
4. Start the Unity scene.

To use another host port, create `.env` from `.env.example` and change `UNITY_TCP_PORT`. The container always listens on port 10000.

```bash
cp .env.example .env
```

Do not expose the TCP endpoint on `0.0.0.0` at the host level unless remote Unity access is explicitly required and protected by a firewall.

## Models

The Docker image contains code but no runtime models. Compose mounts `models/` read-only. Git LFS versions the available YOLO and RL files.

NMPC will intentionally refuse to start until both compatible checkpoint sets exist:

```text
models/nmpc/ripe/best_model_epoch_<N>.pth
models/nmpc/raw/best_model_epoch_<N>.pth
```

The checkpoint found in the old workspace root does not match the current model layout and was not packaged as a valid NMPC model. See `models/README.md`.

## Build details

The image uses Ubuntu 20.04, ROS Noetic, CUDA 11.8, Python 3.8, PyTorch 2.4.1, and Stable-Baselines3 2.4.1. Python dependencies are pinned in `docker/requirements-linux.txt`; L4CasADi is pinned by Git revision in the Dockerfile.

The RL policy was produced by the newer Windows environment. Its complete SB3 archive metadata cannot be loaded with NumPy 1.x, but the runtime node deliberately loads only `policy.pth`. That path was smoke-tested successfully with SB3 2.4.1. A full Linux container build should still be included in release CI once Docker is available.

Container logs and experiment outputs are written under `runs/`. The model directory is read-only inside the container. The run script maps the invoking host user's UID/GID into the container; the container also drops Linux capabilities and uses `no-new-privileges`.

## Publishing this repository

This directory is configured as one monorepo so a recipient needs only one clone. Before pushing:

```bash
git lfs install
git status
git add .
git commit -m "Add Linux Docker runtime"
git remote add origin <superproject-url>
git push -u origin main
```

Confirm redistribution rights for the YOLO weight and trained checkpoints before pushing them to a public remote. Several component `package.xml` files still contain `TODO` licenses; those should be corrected before public distribution.

## Troubleshooting

- `port 10000 is already in use`: stop the other stack or change `UNITY_TCP_PORT` in `.env`.
- `NVIDIA Container Toolkit is unavailable`: install it, restart Docker, or use `--cpu`.
- Unity cannot connect: ensure the container is healthy and Unity uses `127.0.0.1`, not the container IP.
- No images/detections: verify Unity publishes `/agent_0/camera/color/image/compressed` and depth/camera-info topics.
- Data association waits forever: Unity must provide `/obj_pose_srv` and the required TF frames.
- NMPC exits immediately: install both `ripe` and `raw` checkpoint sets described above.
