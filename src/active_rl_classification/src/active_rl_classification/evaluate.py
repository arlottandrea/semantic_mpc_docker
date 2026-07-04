"""
Evaluate a trained PPO policy in the tree classification env with a visual
interface (live matplotlib window) or record to video (mp4 / gif).
"""

import argparse
import io
import zipfile

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch as th
from matplotlib.patches import Rectangle, Circle, FancyArrow
from matplotlib.lines import Line2D
from matplotlib import animation

from stable_baselines3 import PPO

from active_rl_classification.env import TreeClassificationEnv, DIM_TARGET
from active_rl_classification.model import TreeClassFeatureExtractor
from tqdm import tqdm
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

CLASS_COLORS = {0: "green", 1: "red"}
CLASS_LABELS = {0: "not-ripe", 1: "ripe"}


def _load_model(model_path, env, device="auto"):
    """Load an SB3 model, tolerating NumPy 2 metadata under NumPy 1."""
    try:
        return PPO.load(model_path, device=device)
    except ModuleNotFoundError as exc:
        if not (exc.name or "").startswith("numpy._core"):
            raise
        print(
            "Model metadata requires NumPy 2; loading policy weights "
            "directly with the current NumPy version."
        )

    class DummyEnv(gym.Env):
        metadata = {}

        def __init__(self):
            super().__init__()
            self.observation_space = env.observation_space
            self.action_space = env.action_space

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            return self.observation_space.sample(), {}

        def step(self, action):
            return self.observation_space.sample(), 0.0, False, False, {}

    policy_kwargs = dict(
        features_extractor_class=TreeClassFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
        share_features_extractor=False,
        net_arch=dict(pi=[], vf=[]),
    )
    model = PPO(
        "MultiInputPolicy",
        DummyEnv(),
        policy_kwargs=policy_kwargs,
        verbose=0,
        device=device,
    )
    map_location = "cpu" if device == "auto" else device
    with zipfile.ZipFile(model_path, "r") as archive:
        policy_bytes = archive.read("policy.pth")
    state_dict = th.load(io.BytesIO(policy_bytes), map_location=map_location)
    model.policy.load_state_dict(state_dict)
    model.policy.eval()
    return model


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate a tree classification policy with live viz or video recording.",
    )
    p.add_argument("--model", type=str, required=True,
                   help="Path to saved SB3 PPO model (.zip)")
    p.add_argument("--perception-csvs", type=str, nargs=2, required=True,
                   help="--perception-csvs not_ripe.csv ripe.csv")

    p.add_argument("--ntargets", type=int, nargs="+", default=[100])
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--side", type=float, default=50.0)
    p.add_argument("--layout", type=str, default="grid",
                   choices=["random", "grid", "mixed"])
    p.add_argument("--mixed-grid-prob", type=float, default=0.5)
    p.add_argument("--grid-n-rows", type=int, default=10)
    p.add_argument("--grid-n-cols", type=int, default=10)
    p.add_argument("--grid-row-spacing", type=float, default=5.0)
    p.add_argument("--grid-col-spacing", type=float, default=5.0)
    p.add_argument("--grid-jitter-std", type=float, default=0.0)
    p.add_argument("--no-oracle", dest="use_oracle", action="store_false")
    p.set_defaults(use_oracle=True)

    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true",
                   help="Use deterministic policy actions")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--save-video", type=str, default=None,
                   help="Output path; .mp4 uses ffmpeg, .gif uses pillow. "
                        "If omitted, shows interactively.")
    return p.parse_args()


def _snapshot(env):
    """Grab the minimal state needed to render a frame."""
    return {
        "drone": env.drone.copy(),
        "trees": [t.copy() for t in env.trees],
        "tree_classes": env.tree_classes.copy(),
        "beliefs": env.beliefs.copy(),
        "tracked": env.tracked.copy(),
        "step": env.steps,
    }


def _draw_frame(ax, snap, side, cum_reward=None, episode_idx=None):
    ax.clear()
    ax.add_patch(Rectangle(
        (-side, -side), 2 * side, 2 * side,
        fill=False, edgecolor="black", linewidth=2,
    ))

    # Trees (color = true class; gold edge = tracked; label shows P(ripe) belief)
    for t, tree_xy in enumerate(snap["trees"]):
        cls = int(snap["tree_classes"][t])
        belief_ripe = float(snap["beliefs"][t, 1])
        edge = "gold" if snap["tracked"][t] else "black"
        ax.add_patch(Circle(
            (tree_xy[0], tree_xy[1]), DIM_TARGET,
            color=CLASS_COLORS[cls], ec=edge, lw=2, zorder=3,
        ))
        ax.text(
            tree_xy[0], tree_xy[1] + DIM_TARGET + 0.3,
            f"T{t}\nb(ripe)={belief_ripe:.2f}",
            ha="center", fontsize=8, zorder=4,
        )

    # Drone
    dx, dy, dh_deg = snap["drone"]
    heading = np.radians(dh_deg)
    ax.add_patch(Circle((dx, dy), 0.4, color="blue", zorder=5))
    ax.add_patch(FancyArrow(
        dx, dy, 1.5 * np.cos(heading), 1.5 * np.sin(heading),
        width=0.15, color="blue", zorder=5,
    ))

    legend_items = [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor=CLASS_COLORS[c], markersize=12,
               label=f"Tree: {CLASS_LABELS[c]}")
        for c in sorted(CLASS_COLORS)
    ]
    legend_items += [
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="blue", markersize=12, label="Drone"),
        Line2D([0], [0], marker="o", color="w",
               markerfacecolor="none", markeredgecolor="gold",
               markersize=12, label="Tracked"),
    ]
    ax.legend(handles=legend_items, loc="upper right")

    ax.set_xlim(-side - 1, side + 1)
    ax.set_ylim(-side - 1, side + 1)
    ax.set_aspect("equal")

    n_tracked = int(snap["tracked"].sum())
    n_total = len(snap["trees"])
    title = f"step {snap['step']}  tracked {n_tracked}/{n_total}"
    if episode_idx is not None:
        title = f"ep {episode_idx}  " + title
    if cum_reward is not None:
        title += f"  cumR {cum_reward:.1f}"
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")


def _run_episode(model, env, deterministic):
    obs, _ = env.reset()
    frames = [(_snapshot(env), 0.0)]
    cum_r = 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, r, term, trunc, info = env.step(action)
        cum_r += float(r)
        frames.append((_snapshot(env), cum_r))
        done = term or trunc
    return frames, info


def main():
    args = parse_args()

    env = TreeClassificationEnv(config={
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
        "collision_check": False,
    })
    env.reset(seed=args.seed)

    model = _load_model(args.model, env, device="auto")

    all_frames = []  # list of (episode_idx, snapshot, cum_reward)
    print(f"Collecting rollouts for {args.episodes} episode(s)...")
    for ep in tqdm(range(args.episodes), desc="Episodes"):
        frames, info = _run_episode(model, env, args.deterministic)
        print(
            f"episode {ep}: len={info['episode_length']} "
            f"tracked={info['num_tracked']} success={info['success']}"
        )
        if ep == args.episodes-1:
            all_frames.extend([(ep, snap, cum_r) for snap, cum_r in frames])

    fig, ax = plt.subplots(figsize=(15, 15))

    def animate(i):
        ep, snap, cum_r = all_frames[i]
        _draw_frame(ax, snap, env.side, cum_reward=cum_r, episode_idx=ep)
        return []

    interval_ms = 1000.0 / args.fps

    if args.save_video:
        writer = "pillow" if args.save_video.lower().endswith(".gif") else "ffmpeg"
        print(f"Rendering {len(all_frames)} frames to {args.save_video} ({writer})...")
        anim = animation.FuncAnimation(
            fig, animate, frames=len(all_frames), interval=interval_ms, blit=False,
        )
        anim.save(args.save_video, writer=writer, fps=args.fps)
        print(f"Saved video to {args.save_video}")
    else:
        print("Showing interactively — close the window to exit.")
        anim = animation.FuncAnimation(
            fig, animate, frames=len(all_frames), interval=interval_ms,
            blit=False, repeat=False,
        )
        plt.show()


if __name__ == "__main__":
    main()
