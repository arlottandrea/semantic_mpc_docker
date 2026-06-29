"""
Training script for tree classification policy using SB3 PPO + wandb.

Usage:
    uv run python -m active_rl_classification.train --total-timesteps 1000000 --wandb-project my_project

PPO hyperparameters preserved from the original active classification codebase
(configs_v2.py + trainer_PPO_transf_torch_wandb.py).
"""

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from active_rl_classification.env import TreeClassificationEnv
from active_rl_classification.model import TreeClassFeatureExtractor


def package_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def default_artifact_dir(*parts):
    return os.path.join(package_root(), "artifacts", "gym", *parts)


def parse_args():
    parser = argparse.ArgumentParser(description="Train tree classification policy")

    # Environment
    parser.add_argument("--ntargets", type=int, nargs="+", default=[80,100],
                        help="Number of trees (single int or [min, max])")
    parser.add_argument("--horizon", type=int, default=100000,
                        help="Max steps per episode")
    parser.add_argument("--side", type=float, default=50.0,
                        help="Half-side of arena")
    parser.add_argument("--layout", type=str, default="grid",
                        choices=["random", "grid", "mixed"],
                        help="Tree layout mode")
    parser.add_argument("--mixed-grid-prob", type=float, default=0.5)
    parser.add_argument("--grid-n-rows", type=int, default=10)
    parser.add_argument("--grid-n-cols", type=int, default=10)
    parser.add_argument("--grid-row-spacing", type=float, default=5.0)
    parser.add_argument("--grid-col-spacing", type=float, default=5.0)
    parser.add_argument("--grid-jitter-std", type=float, default=0.0)
    parser.add_argument("--perception-csvs", type=str, nargs=2, required=True,
                        help="Paths to pre-computed perception CSVs: "
                             "--perception-csvs not_ripe.csv ripe.csv")
    parser.add_argument("--no-oracle", dest="use_oracle", action="store_false",
                        help="Disable the oracle correction on perception "
                             "outputs. By default the oracle is on (matches "
                             "the original codebase).")
    parser.set_defaults(use_oracle=True)

    # Training
    parser.add_argument("--total-timesteps", type=int, default=2_000_000)
    parser.add_argument("--n-envs", type=int, default=8,
                        help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto",
                        help="Device for training (auto/cpu/cuda)")

    # Logging & checkpoints
    parser.add_argument("--wandb-project", type=str, default="active-rl-classification")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--save-freq", type=int, default=50_000,
                        help="Checkpoint save frequency (timesteps)")
    parser.add_argument("--save-dir", type=str, default=default_artifact_dir("checkpoints"))
    parser.add_argument("--wandb-dir", type=str, default=default_artifact_dir("wandb"))
    parser.add_argument("--tensorboard-dir", type=str, default=default_artifact_dir("tensorboard"))

    # Restore
    # Observation
    parser.add_argument("--k-obs", type=int, default=5,
                        help="Number of closest untracked trees visible to the policy")
    parser.add_argument("--obs-range", type=float, default=5.0,
                        help="Max distance (m) at which trees are observable")

    # Restore
    parser.add_argument("--restore", type=str, default=None,
                        help="Path to checkpoint (.zip) to continue training from")

    return parser.parse_args()


def main():
    args = parse_args()
    import wandb
    from wandb.integration.sb3 import WandbCallback

    # Build environment config
    env_kwargs = {
        "config": {
            "ntargets": args.ntargets if len(args.ntargets) > 1 else args.ntargets[0],
            "horizon": args.horizon,
            "side": args.side,
            "layout": args.layout,
            "mixed_grid_prob": args.mixed_grid_prob,
            "grid_n_rows": args.grid_n_rows,
            "grid_n_cols": args.grid_n_cols,
            "grid_row_spacing": args.grid_row_spacing,
            "grid_col_spacing": args.grid_col_spacing,
            "grid_jitter_std": args.grid_jitter_std,
            "perception_csvs": args.perception_csvs,
            "use_oracle": args.use_oracle,
            "k_obs": args.k_obs,
            "obs_range": args.obs_range,
        }
    }

    # Build full config dict (authoritative for training + wandb logging)
    ppo_hparams = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 256,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.001,
        "max_grad_norm": 0.5,
    }
    full_config = {
        **env_kwargs["config"],
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "seed": args.seed,
        **ppo_hparams,
    }

    # Initialize wandb
    os.makedirs(args.wandb_dir, exist_ok=True)
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=full_config,
        sync_tensorboard=True,
        dir=args.wandb_dir,
    )
    tensorboard_log = os.path.join(args.tensorboard_dir, run.id)

    # Create vectorized environments
    vec_env = make_vec_env(
        TreeClassificationEnv,
        n_envs=args.n_envs,
        seed=args.seed,
        env_kwargs=env_kwargs,
    )

    # PPO with set-transformer feature extractor
    policy_kwargs = dict(
        features_extractor_class=TreeClassFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        share_features_extractor=False,
        net_arch=dict(pi=[], vf=[]),
    )

    if args.restore:
        model = PPO.load(args.restore, env=vec_env, device=args.device)
        model.tensorboard_log = tensorboard_log
        model.verbose = 1
        print(f"Restored from {args.restore}")
    else:
        model = PPO(
            "MultiInputPolicy",
            vec_env,
            policy_kwargs=policy_kwargs,
            **ppo_hparams,
            seed=args.seed,
            verbose=1,
            tensorboard_log=tensorboard_log,
            device=args.device,
        )

    # Callbacks
    os.makedirs(args.save_dir, exist_ok=True)
    callbacks = CallbackList([
        WandbCallback(
            model_save_path=f"{args.save_dir}/{run.id}",
            verbose=2,
        ),
        CheckpointCallback(
            save_freq=max(args.save_freq // args.n_envs, 1),
            save_path=args.save_dir,
            name_prefix="tree_class_policy",
        ),
    ])

    # Train
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
    )

    # Save final model
    final_path = os.path.join(args.save_dir, "final_model")
    model.save(final_path)
    print(f"Final model saved to {final_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
