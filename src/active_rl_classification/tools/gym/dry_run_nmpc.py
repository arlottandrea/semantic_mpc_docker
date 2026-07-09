#!/usr/bin/env python3
"""Run the CasADi NMPC controller against the Gym environment plant."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import casadi as ca
import numpy as np
from scipy.stats import entropy

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src" / "active_rl_classification" / "src"))
sys.path.insert(0, str(ROOT / "src" / "semantic_mpc" / "semantic_mpc" / "src"))

from active_rl_classification.env import TreeClassificationEnv
from semantic_mpc_package.nmpc_optimizer import NmpcOptimizer


def total_entropy(env):
    return float(sum(entropy(belief, base=2) for belief in env.beliefs))


def gym_action_from_acceleration(env, velocity, acceleration, dt):
    desired_world_velocity = velocity[:2] + dt * acceleration[:2]
    heading = math.radians(env.drone[2])
    world_to_body = np.asarray(
        [[math.cos(heading), math.sin(heading)], [-math.sin(heading), math.cos(heading)]]
    )
    body_velocity = world_to_body @ desired_world_velocity
    desired_yaw_rate = velocity[2] + dt * acceleration[2]
    action = np.asarray(
        [body_velocity[0] / 2.0, body_velocity[1] / 2.0, desired_yaw_rate / math.radians(60.0)],
        dtype=np.float32,
    )
    return np.clip(action, -1.0, 1.0)


def measured_velocity(previous_pose, current_pose, dt):
    xy = (current_pose[:2] - previous_pose[:2]) / dt
    yaw_delta = (current_pose[2] - previous_pose[2] + 180.0) % 360.0 - 180.0
    return np.asarray([xy[0], xy[1], math.radians(yaw_delta) / dt])


def nearest_indices_with_padding(points, robot_position, count):
    points = np.asarray(points, dtype=float)
    distances = np.linalg.norm(points - np.asarray(robot_position, dtype=float), axis=1)
    selected = np.argsort(distances)[:count].astype(int).tolist()
    padding_index = selected[-1] if selected else 0
    while len(selected) < count:
        selected.append(padding_index)
    return np.asarray(selected, dtype=int)


def load_surrogate(args):
    import l4casadi as l4c
    import torch

    from semantic_mpc_package.perception_model import MultiLayerPerceptron

    model = MultiLayerPerceptron(
        input_dim=3,
        hidden_size=args.hidden_size,
        hidden_layers=args.hidden_layers,
        output_dim=6,
        threshold=5.0,
        gate_slope=10.0,
        yaw_harmonics=args.yaw_harmonics,
        include_alignment_features=args.include_alignment_features,
        output_temperature=args.output_temperature,
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    name = f"gym_nmpc_perception_h{args.mpc_horizon}_n{args.active_target_count}"
    return l4c.L4CasADi(
        model,
        batched=True,
        device="cpu",
        name=name,
        generate_jac=True,
        generate_adj1=args.l4_generate_adjoints,
        generate_jac_adj1=args.l4_generate_adjoints,
        generate_jac_jac=False,
    )


def synthetic_surrogate(batch_size):
    """Constant reliable surrogate for dependency-free NMPC visualization."""
    poses = ca.MX.sym("synthetic_perception_input", batch_size, 3)
    output = ca.repmat(
        ca.DM([[0.0, 0.85, 0.15, 0.0, 0.15, 0.85]]), batch_size, 1
    )
    return ca.Function("synthetic_gym_perception", [poses], [output])


def analytical_surrogate(batch_size, observation_range, gate_slope=4.0):
    """Smooth pose-dependent conditional sensor model for NMPC diagnostics."""
    poses = ca.MX.sym("analytical_perception_input", batch_size, 3)
    rows = []
    for i in range(batch_size):
        x = poses[i, 0]
        y = poses[i, 1]
        yaw = poses[i, 2]
        distance = ca.sqrt(x * x + y * y + 1e-8)
        range_gate = 1.0 / (1.0 + ca.exp(-gate_slope * (observation_range - distance)))
        yaw_to_tree = ca.atan2(-y, -x)
        facing = 0.5 * (1.0 + ca.cos(yaw_to_tree - yaw))
        accuracy = 0.5 + 0.49 * facing
        raw_row = ca.horzcat(
            1.0 - range_gate,
            range_gate * accuracy,
            range_gate * (1.0 - accuracy),
        )
        ripe_row = ca.horzcat(
            1.0 - range_gate,
            range_gate * (1.0 - accuracy),
            range_gate * accuracy,
        )
        rows.append(ca.horzcat(raw_row, ripe_row))
    return ca.Function("analytical_gym_perception", [poses], [ca.vcat(rows)])


def run_episode(args, seed, surrogate):
    env = TreeClassificationEnv(
        {
            "ntargets": args.grid_rows * args.grid_cols,
            "horizon": args.steps,
            "layout": "grid",
            "side": args.field_side,
            "grid_n_rows": args.grid_rows,
            "grid_n_cols": args.grid_cols,
            "grid_row_spacing": args.grid_spacing,
            "grid_col_spacing": args.grid_spacing,
            "grid_jitter_std": 0.0,
            "perception_csvs": [args.raw_csv, args.ripe_csv],
            "use_oracle": True,
        }
    )
    env.reset(seed=seed)
    ipopt_options = {
        "print_level": 0,
        "sb": "yes",
        "acceptable_tol": 3e-2,
        "acceptable_iter": 5,
        "max_iter": 100,
    }
    hessian_mode = args.ipopt_hessian_approximation
    if hessian_mode == "auto":
        hessian_mode = (
            "limited-memory"
            if args.checkpoint and not args.l4_generate_adjoints
            else "exact"
        )
    if hessian_mode == "limited-memory":
        ipopt_options["hessian_approximation"] = "limited-memory"
    params = {
            "state_dim": 3,
            "control_dim": 3,
            "optimizer_state_dim": 6,
            "dt": 0.25,
            "model_device": "cpu",
            "mpc_horizon": args.mpc_horizon,
            "num_target_trees": args.active_target_count,
            "num_obstacle_trees": args.active_obstacle_count,
            "safe_distance": args.safe_distance,
            "observation_range": args.observation_range,
            "movement_weight": args.movement_weight,
            "yaw_movement_weight": args.yaw_movement_weight,
            "acceleration_regularization_weight": 0.0001,
            "information_gain_weight": args.information_gain_weight,
            "information_discount": args.information_discount,
            "exploration_weight": args.exploration_weight,
            "exploration_sigma": args.exploration_sigma,
            "attraction_weight": args.attraction_weight,
            "camera_facing_weight": args.camera_facing_weight,
            "camera_yaw_offset": 0.0,
            "camera_activation_sigma": args.camera_activation_sigma,
            "observation_standoff": args.observation_standoff,
            "observation_standoff_weight": args.observation_standoff_weight,
            "running_camera_facing_weight": args.running_camera_facing_weight,
            "running_observation_standoff_weight": args.running_observation_standoff_weight,
            "orbit_velocity_weight": args.orbit_velocity_weight,
            "field_margin": 3.0,
            "max_heading_abs": 3.0 * np.pi,
            "max_velocity": args.max_velocity,
            "max_yaw_velocity": math.radians(args.max_yaw_velocity_deg),
            "max_accel_xy": args.max_accel_xy,
            "max_accel_yaw": math.radians(args.max_accel_yaw_deg),
            "ipopt": ipopt_options,
        }
    optimizer = NmpcOptimizer(params, surrogate)
    initial = total_entropy(env)
    initial_pose = env.drone.copy()
    trajectory = [env.drone.copy()]
    entropy_history = [initial]
    total_distance = 0.0
    closest_tree_distance = float("inf")
    velocity = np.zeros(3)
    mpc_step = decision = multipliers = None
    solve_times = []
    previous_entropy = initial
    stagnation_steps = 0
    target_cooldown_until = {}
    terminated = truncated = False
    while not (terminated or truncated):
        pose = np.asarray([env.drone[0], env.drone[1], math.radians(env.drone[2])])
        state = np.concatenate([pose, velocity])
        selection_beliefs = env.beliefs.copy()
        for target_index, cooldown_until in list(target_cooldown_until.items()):
            if env.steps < cooldown_until:
                selection_beliefs[target_index] = [1.0, 0.0]
            else:
                del target_cooldown_until[target_index]
        target_indices, target_mask = optimizer.select_nearest_untracked(
            env.trees,
            selection_beliefs,
            pose[:2],
            args.active_target_count,
            0.95,
        )
        if args.primary_target_only:
            target_mask[1:] = 0.0
        obstacle_indices = nearest_indices_with_padding(
            env.trees,
            pose[:2],
            args.active_obstacle_count,
        )
        target_trees = np.asarray(env.trees)[target_indices]
        target_beliefs = env.beliefs[target_indices]
        obstacle_trees = np.asarray(env.trees)[obstacle_indices]
        parameter = ca.vertcat(
            ca.DM(state),
            ca.reshape(ca.DM(target_trees), 2 * args.active_target_count, 1),
            ca.reshape(ca.DM(target_beliefs), 2 * args.active_target_count, 1),
            ca.reshape(ca.DM(target_mask), args.active_target_count, 1),
            ca.reshape(ca.DM(obstacle_trees), 2 * args.active_obstacle_count, 1),
        )
        started = time.perf_counter()
        if mpc_step is None:
            mpc_step, command, _, decision, multipliers = optimizer.mpc_opt(
                target_trees,
                target_beliefs,
                target_mask,
                obstacle_trees,
                np.asarray([-env.side, -env.side]),
                np.asarray([env.side, env.side]),
                state,
                steps=args.mpc_horizon,
            )
        else:
            command, _, decision, multipliers = mpc_step(parameter, decision, multipliers)
        solve_times.append(time.perf_counter() - started)
        action = gym_action_from_acceleration(
            env, velocity, np.asarray(command).reshape(-1), params["dt"]
        )
        previous_pose = env.drone.copy()
        _, _, terminated, truncated, _ = env.step(action)
        total_distance += float(np.linalg.norm(env.drone[:2] - previous_pose[:2]))
        closest_tree_distance = min(
            closest_tree_distance,
            float(np.min(np.linalg.norm(np.asarray(env.trees) - env.drone[:2], axis=1))),
        )
        velocity = measured_velocity(previous_pose, env.drone, params["dt"])
        trajectory.append(env.drone.copy())
        current_entropy = total_entropy(env)
        entropy_history.append(current_entropy)
        if previous_entropy - current_entropy > args.entropy_progress_epsilon:
            stagnation_steps = 0
        else:
            stagnation_steps += 1
        previous_entropy = current_entropy
        if stagnation_steps >= args.stagnation_steps and len(target_indices):
            target_cooldown_until[int(target_indices[0])] = (
                env.steps + args.target_cooldown_steps
            )
            stagnation_steps = 0

    final = total_entropy(env)
    result = {
        "seed": seed,
        "initial_entropy": initial,
        "final_entropy": final,
        "entropy_reduction": initial - final,
        "tracked": int(np.sum(env.tracked)),
        "tracked_indices": np.where(env.tracked)[0].astype(int).tolist(),
        "max_beliefs": np.max(env.beliefs, axis=1).round(4).tolist(),
        "steps": env.steps,
        "success": bool(terminated),
        "initial_pose": initial_pose.tolist(),
        "final_pose": env.drone.tolist(),
        "distance_travelled": total_distance,
        "closest_tree_distance": closest_tree_distance,
        "mean_solve_time_s": float(np.mean(solve_times)),
    }
    if args.plot and seed == 0:
        plot_episode(
            args.plot,
            np.asarray(trajectory),
            np.asarray(entropy_history),
            np.asarray(env.trees),
            np.asarray(env.tree_classes),
            env.obs_range,
        )
    env.close()
    return result


def plot_episode(path, trajectory, entropy_history, trees, tree_classes, observation_range):
    import matplotlib.pyplot as plt

    figure, (trajectory_axis, entropy_axis) = plt.subplots(
        1, 2, figsize=(13, 5.5), constrained_layout=True
    )
    steps = np.arange(len(trajectory))
    trajectory_axis.plot(trajectory[:, 0], trajectory[:, 1], color="0.7", linewidth=1.2)
    points = trajectory_axis.scatter(
        trajectory[:, 0], trajectory[:, 1], c=steps, cmap="viridis", s=18, zorder=3
    )
    colors = np.where(tree_classes == 1, "tab:red", "tab:green")
    trajectory_axis.scatter(
        trees[:, 0], trees[:, 1], c=colors, marker="o", s=110,
        edgecolors="black", linewidths=0.8, zorder=4,
    )
    for index, tree in enumerate(trees):
        circle = plt.Circle(
            tree, observation_range, fill=False, color=colors[index], alpha=0.15
        )
        trajectory_axis.add_patch(circle)
        trajectory_axis.annotate(str(index), tree, ha="center", va="center", color="white")
    trajectory_axis.scatter(
        trajectory[0, 0], trajectory[0, 1], marker="s", s=90, color="tab:blue", label="start"
    )
    trajectory_axis.scatter(
        trajectory[-1, 0], trajectory[-1, 1], marker="X", s=100, color="black", label="end"
    )
    trajectory_axis.set_title("NMPC trajectory in Gym environment")
    trajectory_axis.set_xlabel("x [m]")
    trajectory_axis.set_ylabel("y [m]")
    trajectory_axis.set_aspect("equal", adjustable="box")
    trajectory_axis.grid(alpha=0.25)
    trajectory_axis.legend()
    figure.colorbar(points, ax=trajectory_axis, label="control step")

    entropy_axis.plot(steps, entropy_history, color="tab:purple", linewidth=2)
    entropy_axis.fill_between(steps, entropy_history, alpha=0.15, color="tab:purple")
    entropy_axis.set_title("Ripe/raw belief entropy evolution")
    entropy_axis.set_xlabel("control step")
    entropy_axis.set_ylabel("total entropy [bits]")
    entropy_axis.grid(alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--ripe-csv", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--synthetic-surrogate", action="store_true")
    parser.add_argument("--analytical-surrogate", action="store_true")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--mpc-horizon", type=int, default=3)
    parser.add_argument("--grid-rows", type=int, default=1)
    parser.add_argument("--grid-cols", type=int, default=5)
    parser.add_argument("--grid-spacing", type=float, default=5.0)
    parser.add_argument("--field-side", type=float, default=25.0)
    parser.add_argument("--active-target-count", type=int, default=1)
    parser.add_argument("--active-obstacle-count", type=int, default=5)
    parser.add_argument("--safe-distance", type=float, default=1.5)
    parser.add_argument("--observation-range", type=float, default=5.0)
    parser.add_argument("--observation-standoff", type=float, default=4.0)
    parser.add_argument("--movement-weight", type=float, default=0.01)
    parser.add_argument("--yaw-movement-weight", type=float, default=0.01)
    parser.add_argument("--information-gain-weight", type=float, default=1.0)
    parser.add_argument("--information-discount", type=float, default=0.99)
    parser.add_argument("--exploration-weight", type=float, default=0.25)
    parser.add_argument("--exploration-sigma", type=float, default=5.0)
    parser.add_argument("--attraction-weight", type=float, default=0.1)
    parser.add_argument("--camera-facing-weight", type=float, default=0.5)
    parser.add_argument("--camera-activation-sigma", type=float, default=5.0)
    parser.add_argument("--observation-standoff-weight", type=float, default=0.5)
    parser.add_argument("--running-camera-facing-weight", type=float, default=0.0)
    parser.add_argument("--running-observation-standoff-weight", type=float, default=0.0)
    parser.add_argument("--orbit-velocity-weight", type=float, default=0.25)
    parser.add_argument("--max-velocity", type=float, default=2.0)
    parser.add_argument("--max-yaw-velocity-deg", type=float, default=60.0)
    parser.add_argument("--max-accel-xy", type=float, default=8.0)
    parser.add_argument("--max-accel-yaw-deg", type=float, default=240.0)
    parser.add_argument(
        "--ipopt-hessian-approximation",
        choices=["auto", "exact", "limited-memory"],
        default="auto",
    )
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=4)
    parser.add_argument("--yaw-harmonics", type=int, default=4)
    parser.add_argument("--include-alignment-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-temperature", type=float, default=0.4)
    parser.add_argument(
        "--l4-generate-adjoints",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--stagnation-steps", type=int, default=50)
    parser.add_argument("--target-cooldown-steps", type=int, default=150)
    parser.add_argument("--entropy-progress-epsilon", type=float, default=1e-4)
    parser.add_argument(
        "--primary-target-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Commit terminal attraction/orientation costs to one tree at a time.",
    )
    parser.add_argument("--plot", help="Write the first episode trajectory plot to this path.")
    args = parser.parse_args()
    if args.synthetic_surrogate and args.analytical_surrogate:
        parser.error("choose only one surrogate mode")
    if args.active_target_count < 1 or args.active_obstacle_count < 1:
        parser.error("active target and obstacle counts must be positive")
    if args.synthetic_surrogate:
        surrogate = synthetic_surrogate(args.mpc_horizon * args.active_target_count)
    elif args.analytical_surrogate:
        surrogate = analytical_surrogate(
            args.mpc_horizon * args.active_target_count,
            args.observation_range,
        )
    elif args.checkpoint:
        surrogate = load_surrogate(args)
    else:
        parser.error(
            "--checkpoint is required unless --synthetic-surrogate or "
            "--analytical-surrogate is used"
        )
    records = [run_episode(args, seed, surrogate) for seed in range(args.episodes)]
    reductions = np.asarray([record["entropy_reduction"] for record in records])
    print(
        json.dumps(
            {
                "controller": "NmpcOptimizer",
                "episodes": records,
                "mean_entropy_reduction": float(reductions.mean()),
                "all_episodes_reduced_entropy": bool(np.all(reductions > 0.0)),
                "success_rate": float(np.mean([record["success"] for record in records])),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
