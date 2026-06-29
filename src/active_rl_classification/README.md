# Active RL Classification

ROS package and Gymnasium environment for active tree classification.

## Layout

```text
.
├── data/                       # Perception CSV inputs
├── scripts/                    # ROS executable scripts
├── src/active_rl_classification/
│   ├── env.py                  # Gymnasium environment
│   ├── train.py                # PPO training entrypoint
│   ├── model.py                # Policy feature extractor
│   └── perception.py           # Perception CSV model
├── tests/                      # Standalone validation scripts
├── tools/ros/                  # ROS dry-run/debug helpers
└── artifacts/
    ├── gym/                    # Gym training outputs
    │   ├── checkpoints/
    │   ├── models/
    │   ├── tensorboard/
    │   └── wandb/
    └── ros/                    # ROS runtime outputs
        └── wandb/
```

## Training Outputs

`python -m active_rl_classification.train` writes Gym artifacts to `artifacts/gym` by default:

- checkpoints: `artifacts/gym/checkpoints`
- TensorBoard logs: `artifacts/gym/tensorboard`
- WandB offline runs: `artifacts/gym/wandb`

## ROS Runtime Outputs

`scripts/rl_agent_ros_node.py` loads the default policy from:

```text
artifacts/gym/models/final_model.zip
```

ROS WandB logs default to:

```text
artifacts/ros/wandb
```
