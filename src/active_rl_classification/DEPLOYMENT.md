# Active RL Classification ROS Deployment Guide

## Overview

This repository contains a Python-based ROS1 wrapper for the `active_rl_classification` gym environment. The wrapper node exposes the RL environment to Unity via ROS TCP Connector using standard ROS topics.

The package is deployed as a ROS catkin package inside a ROS1 workspace, which itself lives within a pixi environment for reproducible Python dependency management.

The package implements:
- `TreeClassificationEnv` — the RL environment with tree classification dynamics
- `scripts/ros_rl_node.py` — the ROS1 node that publishes observations and receives actions
- `setup.py` and `CMakeLists.txt` updates to install the ROS node correctly

## Installation

### 1. Activate the pixi environment

Assuming the ROS1 workspace is located at `~/ros1/` and a `pixi.lock` or `pyproject.toml` exists in the workspace root:

```bash
cd ~/ros1
pixi shell
```

This activates the pixi environment with all necessary dependencies pre-configured for ROS1.

### 2. Navigate to the ROS workspace

```bash
cd ~/ros1
```

### 3. Ensure the package is in the workspace

The `active_rl_classification` package should already exist under:

```
~/ros1/src/active_rl_classification/
```

Verify the structure:

```bash
ls -la src/active_rl_classification/
# Should show: CMakeLists.txt  package.xml  setup.py  scripts/  src/
```

### 4. Build the ROS workspace

From the ROS workspace root (inside the pixi environment):

```bash
catkin_make
source devel/setup.bash
```

The `setup.py` entry point will automatically install the Python package in the pixi environment's site-packages.

### 5. Verify installation

Check that the ROS node is available:

```bash
rosrun active_rl_classification ros_rl_node.py --help
```

Or directly run it (see Usage section below).

## Architecture

### Components

- `active_rl_classification/env.py`
  - Defines `TreeClassificationEnv`, a Gymnasium `gym.Env` environment.
  - Models a single drone in a 2D arena that classifies trees as ripe or not-ripe.
  - Handles action stepping, reward computation, and state management.

- `scripts/ros_rl_node.py`
  - A modular ROS1 bridge (`RosRLBridge` class).
  - Subscribes to action commands from an external RL agent.
  - Steps the internal environment with each action.
  - Publishes reward, done flag, and episode info back to the RL agent.
  - Automatically resets the environment when an episode ends.

### Data Flow

```
RL Agent/Policy (external)
         ↓ publishes action
    ROS Topic: rl/action
         ↓
   RosRLBridge (this node)
         ↓ steps environment
   TreeClassificationEnv
         ↓ computes reward/done
    ROS Topics: rl/reward, rl/done, rl/info
         ↓
RL Agent/Policy (receives feedback)
```

### Design Rationale

- **Modular**: The bridge and environment are decoupled. The environment logic is in `env.py`, ROS communication is isolated in the bridge.
- **Simple**: The node focuses on one job: forward actions to the environment and publish results.
- **Extensible**: RL algorithms can run externally (in Python, C++, ROS services, etc.) and simply publish actions.
- **Testable**: The environment can be tested independently of ROS.

### Runtime Flow

1. The node starts and initializes `TreeClassificationEnv`.
2. The environment is reset (episode starts).
3. An external RL agent publishes an action on `rl/action`.
4. The node receives the action, steps the environment, and gets: reward, done, info.
5. The node publishes these results to `rl/reward`, `rl/done`, `rl/info`.
6. The RL agent receives the results and computes the next action.
7. Loop repeats until the episode ends (done=True).
8. When done, the environment automatically resets for the next episode.

## I/O Topics

### Input

- `rl/action` (`std_msgs/Float32MultiArray`)
  - **Purpose**: Receive action commands from the RL agent.
  - **Format**: First 3 values: `[forward, lateral, yaw_rate]`
  - **Range**: Each value should be in `[-1, 1]`
  - **Example publisher** (test command):
    ```bash
    rostopic pub /rl/action std_msgs/Float32MultiArray "data: [0.5, 0.0, -0.1]"
    ```

### Outputs

- `rl/reward` (`std_msgs/Float32`)
  - **Purpose**: Scalar reward signal from the environment step.
  - **Meaning**: Reward for the action just taken. Higher is better.

- `rl/done` (`std_msgs/Bool`)
  - **Purpose**: Episode termination flag.
  - **Meaning**: `true` when the episode has ended. The environment will auto-reset for the next episode.

- `rl/info` (`std_msgs/Float32MultiArray`)
  - **Purpose**: Episode metadata.
  - **Format**: `[num_tracked, episode_length, success]`
    - `num_tracked`: Number of trees successfully tracked/classified
    - `episode_length`: Number of steps taken in current episode
    - `success`: `1.0` if episode was successful, `0.0` otherwise

### Optional: Access Environment Observations Directly

If your RL agent needs the full observation from the environment (tree positions, belief, etc.), you can:
1. Import the environment directly: `from active_rl_classification import TreeClassificationEnv`
2. Create an instance and call `.reset()` and `.step(action)` directly.
3. Or extend the bridge to publish additional observation topics (see Customization below).

## Usage Examples

### Running the ROS node

After building the workspace (in the pixi environment):

```bash
# Ensure you are in the pixi environment
cd ~/ros1
pixi shell

# Source the workspace
source devel/setup.bash

# Run the node
rosrun active_rl_classification ros_rl_node.py
```

### In separate terminals (also within pixi environment)

If you need additional terminals for testing or visualization:

```bash
cd ~/ros1
pixi shell
source devel/setup.bash

# Then run additional commands, e.g., rostopic echo or roslaunch
rostopic echo /rl/reward
```

## Usage

### 1. Start ROS Master

```bash
cd ~/ros1
pixi shell
source devel/setup.bash
roscore
```

### 2. Run the ROS-RL Bridge

In a new terminal (also in pixi environment):

```bash
cd ~/ros1
pixi shell
source devel/setup.bash
rosrun active_rl_classification ros_rl_node.py
```

Output should show:
```
[INFO] ROS-RL Bridge initialized. Listening on rl/action
[INFO] Publishing rewards to: rl/reward, rl/done, rl/info
```

### 3. Publish Test Actions

In a third terminal, send a test action:

```bash
cd ~/ros1
pixi shell
source devel/setup.bash

# Publish a single action: [forward=0.5, lateral=0.0, yaw_rate=-0.1]
rostopic pub -1 /rl/action std_msgs/Float32MultiArray "data: [0.5, 0.0, -0.1]"
```

### 4. Monitor Environment Feedback

In a fourth terminal, subscribe to the results:

```bash
cd ~/ros1
pixi shell
source devel/setup.bash

# Watch the reward signal
rostopic echo /rl/reward

# Or watch the done/info signals
rostopic echo /rl/done
rostopic echo /rl/info
```

### 5. Integrate RL Algorithm

Your external RL agent should:

1. **Publish actions** to `/rl/action` with format `[forward, lateral, yaw_rate]` ∈ [-1, 1]
2. **Subscribe to results** from `/rl/reward`, `/rl/done`, `/rl/info`
3. **Process feedback** in your algorithm loop:
   ```python
   # Pseudocode for RL agent
   observation, info = env.reset()
   for step in range(max_steps):
       action = policy.predict(observation)
       pub_action.publish(action)  # Publish to ROS
       
       reward = sub_reward.get()   # Subscribe from ROS
       done = sub_done.get()
       info = sub_info.get()
       
       if done:
           break
   ```

### Environment Configuration via ROS Parameters

```bash
rosrun active_rl_classification ros_rl_node.py \
  _raw_csv:='/path/to/RawData.csv' \
  _ripe_csv:='/path/to/RipeData.csv' \
  _layout:='random' \
  _k_obs:=5 \
  _side:=25.0 \
  _horizon:=100 \
  _use_oracle:=true
```

**Available parameters:**
- `_raw_csv`: Path to raw (not-ripe) tree data CSV
- `_ripe_csv`: Path to ripe tree data CSV
- `_layout`: Arena layout mode (`random`, `grid`, `mixed`)
- `_k_obs`: Number of tree observations per step
- `_obs_range`: Maximum observation range in meters
- `_side`: Half-side length of square arena
- `_horizon`: Maximum steps per episode
- `_use_oracle`: Use oracle classification (true) or noisy sensors (false)

### Topic Remapping (if needed)

If you want to use different topic names:

```bash
rosrun active_rl_classification ros_rl_node.py \
  _action_topic:='/my_agent/action' \
  _reward_topic:='/my_agent/reward' \
  _done_topic:='/my_agent/done' \
  _info_topic:='/my_agent/info'
```

## How to Use

In Unity, connect to ROS and publish an action such as:

```text
[0.5, 0.0, 0.1]
```

Then read the observation and status topics to drive RL behavior.

## Example: Multi-Terminal Workflow with Trained Policy

This is the **production deployment** setup: environment bridge + trained RL agent.

**Terminal 1 — ROS Master:**
```bash
cd ~/ros1
pixi shell
source devel/setup.bash
roscore
```

**Terminal 2 — ROS-RL Environment Bridge:**
```bash
cd ~/ros1
pixi shell
source devel/setup.bash
rosrun active_rl_classification ros_rl_node.py
```

Output:
```
[INFO] ROS-RL Bridge initialized. Listening on rl/action
[INFO] Publishing observations to: rl/obs/*, rl/reward, rl/done, rl/info
```

**Terminal 3 — Deploy Trained RL Agent (this publishes actions):**
```bash
cd ~/ros1
pixi shell
source devel/setup.bash

# Run the agent with your trained policy
rosrun active_rl_classification rl_agent_node.py \
  _policy_path:="/path/to/your/trained_policy.zip" \
  _policy_type:='PPO' \
  _deterministic:=true \
  _step_frequency:=10.0
```

Output:
```
[INFO] Loading policy from: /path/to/your/trained_policy.zip
[INFO] Policy loaded successfully: trained_policy.zip (PPO)
[INFO] RL Agent Node initialized
[INFO] Policy type: PPO
[INFO] Step frequency: 10.0 Hz
[INFO] Deterministic: True
[INFO] Episode 1 done | Total Reward: 42.500 | Steps: 100 | Success: True
[INFO] Episode 2 done | Total Reward: 38.200 | Steps: 100 | Success: False
```

**Terminal 4 — Monitor Performance (optional):**
```bash
cd ~/ros1
pixi shell
source devel/setup.bash

# Watch rewards in real-time
rostopic echo /rl/reward

# Or watch episode done signals
rostopic echo /rl/done

# Or check episode info
rostopic echo /rl/info
```

### Agent Node Parameters

- `_policy_path` (required): Path to trained `.zip` policy file
  - Example: `/home/user/artifacts/gym/artifacts/gym/models/ppo_tree_classifier.zip`
  
- `_policy_type` (default: `PPO`): Type of trained model
  - Options: `PPO`, `A2C`, `DQN`
  - Must match the training algorithm used
  
- `_deterministic` (default: `true`): Use deterministic (greedy) or stochastic actions
  - `true`: Always pick action with highest probability
  - `false`: Sample from policy distribution (exploration)
  
- `_step_frequency` (default: `10.0`): How often to publish actions (Hz)
  - Adjust based on environment dynamics
  - Bridge publishes observations at same rate

### Full Workflow

1. **Training phase (done once):**
   ```bash
   cd ~/ros1
   pixi shell
   python src/active_rl_classification/train.py
   # Output: artifacts/gym/artifacts/gym/models/ppo_tree_classifier.zip
   ```

2. **Deployment phase (runs continuously):**
   - Terminal 1: Start ROS master
   - Terminal 2: Start environment bridge
   - Terminal 3: Start RL agent with trained policy
   - Optional Terminal 4: Monitor in real-time

The agent automatically steps the policy at the configured frequency, publishes actions, and receives observations and rewards from the environment bridge.

### Key Differences from Training

| Phase | Node | Role |
|-------|------|------|
| Training | `train.py` | Trains policy, saves to disk |
| Deployment | `ros_rl_node.py` | Environment simulator |
| Deployment | `rl_agent_node.py` | Loads policy, publishes actions |

During training, the policy interacts with the environment in-process. During deployment, they communicate via ROS topics, allowing:
- Different machines for policy (ML server) and environment (physics sim)
- Real-world hardware integration (e.g., actual drone)
- Unity simulator as environment

## Notes

- The environment will automatically reset when an episode ends.
- The default CSV paths are located at the package root: `data/RawData.csv` and `data/RipeData.csv`.
- Make sure the CSV data files exist before starting the node.
- This package runs within a pixi-managed Python environment. Always activate the pixi shell before running ROS or catkin commands.
- Dependencies are managed by pixi; do not use system pip outside the pixi environment when working with this workspace.
- The agent node handles both PPO (and other on-policy methods) and off-policy methods (DQN). Ensure `_policy_type` matches your trained model.
- Policy inference runs asynchronously at the configured frequency, allowing smooth policy execution without blocking the environment.
- For very high-frequency control (>100 Hz), consider using a dedicated C++ node or direct library calls instead of ROS topics.
