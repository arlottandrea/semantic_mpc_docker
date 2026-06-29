# Agri Neural MPC

## Project Layout

```text
.
├── docs/media/                 # README and documentation media
├── semantic_mpc/                   # ROS package
├── notebooks/                  # Interactive experiments and analysis
├── ros/                        # ROS environment/submodule support files
├── tools/dataset/              # Dataset generation utilities
├── tools/launch/               # Local run helpers
├── requirements.txt            # Python dependencies
└── README.md
```

## Clone the Repository with Submodules

```bash
git clone --recurse-submodules https://github.com/newline-lab/semantic_mpc.git
cd semantic_mpc
git submodule update --init --recursive
```

## Set Up Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Install PyTorch

For CUDA 11.8:

```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

For CPU-only:

```bash
pip install torch torchvision torchaudio
```

## Set Up ROS Dependencies

```bash
source /opt/ros/noetic/setup.bash
rosdep update
rosdep install --from-paths semantic_mpc ros/src --ignore-src -r -y
```

## Build the ROS Workspace

Build from the outer catkin workspace that contains `semantic_mpc` in its `src` folder.
For the local Pixi workspace used by this checkout:

```bash
cd /c/ros1
pixi run build
source devel/setup.bash
```

With a normal ROS workspace:

```bash
cd <workspace-root>
catkin_make
source devel/setup.bash
```

Or with `catkin_tools`:

```bash
cd <workspace-root>
catkin build
source devel/setup.bash
```

## Common Tools

Generate datasets:

```bash
python tools/dataset/generate_dataset.py
```

Run multiple training/simulation sessions:

```bash
tools/launch/run_multi.sh [n_train] [trees_number]
```
