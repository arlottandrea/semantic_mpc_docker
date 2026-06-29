"""Visualize a greedy baseline run step-by-step."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Rectangle
from matplotlib.lines import Line2D
from pathlib import Path

from active_rl_classification.env import TreeClassificationEnv, DIM_TARGET, BELIEF_THRESHOLD


def plot_env_state(env, step, reward, save_path):
    """Plot arena, trees, drone, drone trajectory, and belief confidence."""
    fig, ax = plt.subplots(figsize=(12, 12))

    # Arena boundary
    ax.add_patch(Rectangle(
        (-env.side, -env.side), 2 * env.side, 2 * env.side,
        fill=False, edgecolor="black", linewidth=2,
    ))

    class_colors = {0: "green", 1: "red"}

    # Trees with belief-based coloring
    for t in range(env.ntargets):
        tree_xy = env.trees[t]
        true_cls = int(env.tree_classes[t])
        belief_on_class = env.beliefs[t][true_cls]

        # Color: green if confident not-ripe, red if confident ripe, yellow if uncertain
        if belief_on_class > BELIEF_THRESHOLD:
            color = class_colors[true_cls]
            edge = "black"
            linewidth = 3
        else:
            color = "yellow"
            edge = "gray"
            linewidth = 1

        ax.add_patch(Circle(
            (tree_xy[0], tree_xy[1]), DIM_TARGET,
            color=color, ec=edge, linewidth=linewidth, zorder=3,
        ))
        ax.text(tree_xy[0], tree_xy[1], f"{belief_on_class:.2f}",
                ha="center", va="center", fontsize=8, fontweight="bold", zorder=4)

    # Drone
    drone_xy = env.drone[:2]
    heading_rad = np.radians(env.drone[2])
    ax.add_patch(Circle(
        (drone_xy[0], drone_xy[1]), 0.4,
        color="blue", ec="darkblue", linewidth=2, zorder=5,
    ))
    arrow_len = 1.5
    ax.add_patch(FancyArrow(
        drone_xy[0], drone_xy[1],
        arrow_len * np.cos(heading_rad), arrow_len * np.sin(heading_rad),
        width=0.15, color="blue", zorder=5, head_width=0.3, head_length=0.2,
    ))

    # Legend
    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="green",
               markersize=12, label="Tracked: not-ripe", markeredgecolor="black", markeredgewidth=2),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red",
               markersize=12, label="Tracked: ripe", markeredgecolor="black", markeredgewidth=2),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="yellow",
               markersize=12, label="Untracked", markeredgecolor="gray"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="blue",
               markersize=12, label="Drone"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=10)

    ax.set_xlim(-env.side - 2, env.side + 2)
    ax.set_ylim(-env.side - 2, env.side + 2)
    ax.set_aspect("equal")

    num_tracked = np.sum(env.tracked)
    ax.set_title(
        f"Greedy Baseline: Step {step} | Reward {reward:.2f}\n"
        f"Tracked {num_tracked}/{env.ntargets} trees",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def visualize_greedy_run(seed=42):
    """Run greedy baseline and save frames."""
    data_dir = Path(__file__).parent.parent / "data"
    csvs = [
        str(data_dir / "RawData.csv"),
        str(data_dir / "RipeData.csv"),
    ]

    env = TreeClassificationEnv(
        config={
            "horizon": 100,
            "side": 30.0,
            "layout": "grid",
            "grid_n_rows": 2,
            "grid_n_cols": 5,
            "grid_row_spacing": 10.0,
            "grid_col_spacing": 8.0,
            "perception_csvs": csvs,
        }
    )
    obs, _ = env.reset(seed=seed)

    out_dir = Path("visualizations")
    out_dir.mkdir(exist_ok=True)

    step = 0
    prev_tracked = 0
    plot_env_state(env, step, 0.0, out_dir / f"frame_{step:04d}.png")

    episode_done = False
    cumulative_reward = 0.0

    while not episode_done:
        # Greedy: find closest untracked tree and move towards it
        untracked_indices = [t for t in range(env.ntargets) if not env.tracked[t]]

        if not untracked_indices:
            break

        drone_pos = env.drone[:2]
        distances = [
            np.linalg.norm(env.trees[t] - drone_pos) for t in untracked_indices
        ]
        closest_idx = untracked_indices[np.argmin(distances)]
        closest_dist = min(distances)
        tree_pos = env.trees[closest_idx]
        rel_vec = tree_pos - drone_pos

        # Desired heading: face the tree
        desired_heading_rad = np.arctan2(rel_vec[1], rel_vec[0])
        desired_heading_deg = np.degrees(desired_heading_rad)
        current_heading_deg = env.drone[2]

        # Compute heading error (shortest angle)
        heading_error = desired_heading_deg - current_heading_deg
        heading_error = np.arctan2(np.sin(np.radians(heading_error)),
                                   np.cos(np.radians(heading_error)))
        heading_error_deg = np.degrees(heading_error)

        ORBIT_RADIUS = 3.0
        # P-controller to face the tree
        yaw_cmd = np.clip(heading_error_deg / 45.0, -1, 1)

        if closest_dist > ORBIT_RADIUS + 0.5:
            # Move towards tree while rotating to face it
            direction = rel_vec / (np.linalg.norm(rel_vec) + 1e-6)
            c = np.cos(np.radians(env.drone[2]))
            s = np.sin(np.radians(env.drone[2]))
            rot_t = np.array([[c, s], [-s, c]])
            body_direction = rot_t @ direction
            action = np.array(
                [np.clip(body_direction[0], -1, 1),
                 np.clip(body_direction[1], -1, 1),
                 yaw_cmd],
                dtype=np.float32,
            )
        else:
            # Orbit CCW: strafe left (lateral=+1) while facing tree.
            # Feed-forward the yaw rate needed to track the tree during the strafe:
            # tangential speed = lateral_cmd * action_scale[1] * delta_t = 1*2*0.25 = 0.5 m/step
            # required angular rate = v/r radians → convert to action units via 15°/step per unit
            tangential_speed = 1.0 * env.action_scale[1] * env.delta_t  # m/step
            orbit_rate_deg = np.degrees(tangential_speed / (closest_dist + 1e-6))
            yaw_ff = orbit_rate_deg / 15.0  # action units (1.0 = 15°/step)
            radial_error = closest_dist - ORBIT_RADIUS
            action = np.array(
                [np.clip(radial_error / 1.0, -1, 1),       # hold orbit radius
                 -1.0,                                       # CCW tangential strafe
                 np.clip(yaw_cmd + yaw_ff, -1, 1)],         # track tree heading
                dtype=np.float32,
            )

        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += reward
        step += 1
        episode_done = terminated or truncated

        num_tracked = np.sum(env.tracked)
        plot_env_state(env, step, cumulative_reward, out_dir / f"frame_{step:04d}.png")
        if num_tracked > prev_tracked or episode_done:
            prev_tracked = num_tracked
            print(f"Step {step}: Tracked {num_tracked}/{env.ntargets}, reward={cumulative_reward:.1f}")

    print(f"\nEpisode ended at step {step}.")
    print(f"Success: {info['success']}, Tracked: {info['num_tracked']}/{env.ntargets}")
    print(f"Frames saved to {out_dir}/")


if __name__ == "__main__":
    visualize_greedy_run(seed=42)
