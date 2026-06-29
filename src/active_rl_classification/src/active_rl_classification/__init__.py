from gymnasium.envs.registration import register
from .env import TreeClassificationEnv

__all__ = ["TreeClassificationEnv"]

# Register the environment with gymnasium
register(
    id="TreeClassificationEnv-v0",
    entry_point="active_rl_classification:TreeClassificationEnv",
    max_episode_steps=100,
)

