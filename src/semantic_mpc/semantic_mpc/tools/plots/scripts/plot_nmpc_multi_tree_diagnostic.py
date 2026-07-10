#!/usr/bin/env python3
"""Offline multi-tree NMPC/MLP-CasADi diagnostic.

This script exercises the same optimizer used by the ROS NMPC node, but with a
deterministic synthetic orchard and a closed-loop predicted-state rollout.  It
is meant to validate multi-target selection, safety constraints, relative-yaw
MLP features, and belief updates without starting ROS.
"""

import argparse
import csv
from pathlib import Path
import sys
import time

import casadi as ca
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_nmpc_one_tree_diagnostic import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    ROOT,
    bayes_update,
    entropy_bits,
    relative_mlp_yaw_from_pose,
    wrap_angle,
)
from semantic_mpc_package.casadi_mlp_sensor import trained_mlp_sensor  # noqa: E402
from semantic_mpc_package.nmpc_config import default_nmpc_params  # noqa: E402
from semantic_mpc_package.nmpc_optimizer import NmpcOptimizer  # noqa: E402


def multi_tree_params(args):
    params = default_nmpc_params()
    params.update(
        {
            "model_device": "gpu",
            "num_target_trees": int(args.active_targets),
            "num_obstacle_trees": int(args.obstacles),
            "active_target_count": int(args.active_targets),
            "active_obstacle_count": int(args.obstacles),
            "dt": float(args.dt),
            "mpc_horizon": int(args.horizon),
            "safe_distance": float(args.safe_distance),
            "movement_weight": 1e-3,
            "yaw_movement_weight": 1e-3,
            "acceleration_regularization_weight": 1e-5,
            "information_gain_weight": float(args.information_gain_weight),
            "entropy_time_pressure_weight": float(args.entropy_time_pressure_weight),
            "exploration_weight": 0.0,
            "attraction_weight": 0.0,
            "camera_facing_weight": 0.5,
            "observation_standoff_weight": 0.2,
            "running_camera_facing_weight": 0.1,
            "running_observation_standoff_weight": 0.1,
            "orbit_velocity_weight": 0.2,
            "max_velocity": float(args.max_velocity),
            "max_yaw_velocity": float(np.deg2rad(args.max_yaw_velocity_deg)),
            "max_accel_xy": float(args.max_accel_xy),
            "max_accel_yaw": float(np.deg2rad(args.max_accel_yaw_deg)),
            "ipopt": {
                "print_level": 0,
                "sb": "yes",
                "max_iter": int(args.ipopt_max_iter),
                "tol": 1e-2,
                "acceptable_tol": 1e-1,
                "acceptable_iter": 2,
                "max_cpu_time": float(args.ipopt_max_cpu_time),
                "warm_start_init_point": "yes",
                "warm_start_bound_push": 1e-6,
                "warm_start_mult_bound_push": 1e-6,
                "mu_init": 1e-4,
                "hessian_approximation": "limited-memory",
            },
        }
    )
    return params


def orchard_positions(count, spacing=4.5):
    count = int(count)
    cols = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / cols))
    points = []
    for row in range(rows):
        for col in range(cols):
            if len(points) >= count:
                break
            x = col * float(spacing)
            y = row * float(spacing)
            if row % 2:
                x += 0.5 * float(spacing)
            points.append([x, y])
    points = np.asarray(points, dtype=float)
    points -= np.mean(points, axis=0, keepdims=True)
    points += np.asarray([8.0, 0.0])
    return points


def default_true_classes(count):
    labels = []
    for index in range(int(count)):
        labels.append("ripe" if index % 2 == 0 else "raw")
    return np.asarray(labels, dtype=object)


def initial_state_for_orchard(trees, params, flip_yaw=False):
    lower = np.min(trees, axis=0)
    upper = np.max(trees, axis=0)
    start_xy = np.asarray(
        [
            lower[0] - float(params["observation_range"]) - 1.0,
            lower[1] - 0.5 * float(params["observation_range"]),
        ]
    )
    center = 0.5 * (lower + upper)
    yaw = np.arctan2(center[1] - start_xy[1], center[0] - start_xy[0])
    if flip_yaw:
        yaw = wrap_angle(yaw + np.pi)
    return np.asarray([start_xy[0], start_xy[1], yaw, 0.0, 0.0, 0.0], dtype=float)


def nearest_indices(points, query, count):
    points = np.asarray(points, dtype=float)
    query = np.asarray(query, dtype=float).reshape(2)
    order = np.argsort(np.linalg.norm(points - query[None, :], axis=1)).astype(int).tolist()
    selected = order[: int(count)]
    padding = selected[-1] if selected else 0
    while len(selected) < int(count):
        selected.append(padding)
    return np.asarray(selected, dtype=int)


def optimizer_parameter_vector(state, target_trees, target_beliefs, target_mask, obstacle_trees):
    target_count = len(target_trees)
    obstacle_count = len(obstacle_trees)
    return ca.vertcat(
        ca.DM(state),
        ca.reshape(ca.DM(target_trees), target_count * 2, 1),
        ca.reshape(ca.DM(target_beliefs), target_count * 2, 1),
        ca.reshape(ca.DM(target_mask), target_count, 1),
        ca.reshape(ca.DM(obstacle_trees), obstacle_count * 2, 1),
    )


def measurement_features(pose, trees, params):
    pose = np.asarray(pose, dtype=float).reshape(-1)[:3]
    relative = pose[None, :2] - trees[:, :2]
    relative_yaws = [
        relative_mlp_yaw_from_pose(pose, tree, params.get("camera_yaw_offset", 0.0))
        for tree in trees
    ]
    return np.column_stack((relative, np.asarray(relative_yaws, dtype=float)))


def realized_measurement_likelihoods(pose, trees, true_classes, sensor, params):
    features = measurement_features(pose, trees, params)
    outputs = np.asarray(sensor(ca.DM(features)), dtype=float).reshape(len(trees), 2, 3)
    categories = np.zeros(len(trees), dtype=int)
    likelihoods = np.ones((len(trees), 2), dtype=float)
    for index, true_class in enumerate(true_classes):
        class_index = 0 if true_class == "raw" else 1
        observation = int(np.argmax(outputs[index, class_index]))
        categories[index] = observation
        if observation != 0:
            likelihoods[index] = [outputs[index, 1, observation], outputs[index, 0, observation]]
    return likelihoods, categories, outputs


def run_multi_tree_diagnostic(args):
    args.active_targets = min(int(args.active_targets), int(args.tree_count))
    args.obstacles = min(int(args.obstacles), int(args.tree_count))
    params = multi_tree_params(args)
    trees = orchard_positions(args.tree_count, args.tree_spacing)
    true_classes = default_true_classes(args.tree_count)
    state = initial_state_for_orchard(trees, params, flip_yaw=args.flip_start_yaw)
    initial_state = state.copy()
    beliefs = np.full((len(trees), 2), 0.5, dtype=float)
    checkpoint = Path(args.checkpoint)
    sensor = trained_mlp_sensor(
        int(params["mpc_horizon"]) * int(params["num_target_trees"]),
        checkpoint,
        weights_cache_path=args.weights_cache,
        output_temperature=params.get("nn_output_temperature", 0.4),
        yaw_harmonics=params.get("nn_yaw_harmonics", 4),
        include_alignment_features=params.get("nn_include_alignment_features", True),
    )
    update_sensor = trained_mlp_sensor(
        len(trees),
        checkpoint,
        weights_cache_path=args.weights_cache,
        output_temperature=params.get("nn_output_temperature", 0.4),
        yaw_harmonics=params.get("nn_yaw_harmonics", 4),
        include_alignment_features=params.get("nn_include_alignment_features", True),
    )
    optimizer = NmpcOptimizer(params, sensor)

    lb = np.minimum(np.min(trees, axis=0), initial_state[:2]) - (
        float(params["observation_range"]) + 3.0
    )
    ub = np.maximum(np.max(trees, axis=0), initial_state[:2]) + (
        float(params["observation_range"]) + 3.0
    )

    states = [state.copy()]
    belief_history = [beliefs.copy()]
    entropy_history = [np.asarray([entropy_bits(belief) for belief in beliefs])]
    total_entropy = [float(np.sum(entropy_history[-1]))]
    min_tree_distances = [float(np.min(np.linalg.norm(trees - state[:2], axis=1)))]
    categories_history = [np.full(len(trees), -1, dtype=int)]
    target_history = []
    commands = []
    solve_times = []
    mpc_step = None
    x_dec_prev = None
    lam_g_prev = None
    termination_reason = "max_iterations"

    for _iteration in range(int(args.iterations)):
        tracked = np.max(beliefs, axis=1) >= float(args.confidence_threshold)
        if bool(np.all(tracked)):
            termination_reason = "all_trees_tracked"
            break

        target_indices, target_mask = optimizer.select_nearest_untracked(
            trees,
            beliefs,
            state[:2],
            int(params["num_target_trees"]),
            float(args.confidence_threshold),
        )
        obstacle_indices = nearest_indices(trees, state[:2], int(params["num_obstacle_trees"]))
        target_trees = trees[target_indices]
        target_beliefs = beliefs[target_indices]
        obstacle_trees = trees[obstacle_indices]
        target_history.append(target_indices.copy())

        start = time.perf_counter()
        if mpc_step is None:
            mpc_step, command, predicted, x_dec_prev, lam_g_prev = optimizer.mpc_opt(
                target_trees,
                target_beliefs,
                target_mask,
                obstacle_trees,
                lb,
                ub,
                state,
                steps=params["mpc_horizon"],
            )
        else:
            p0_val = optimizer_parameter_vector(
                state,
                target_trees,
                target_beliefs,
                target_mask,
                obstacle_trees,
            )
            command, predicted, x_dec_prev, lam_g_prev = mpc_step(
                p0_val,
                x_dec_prev,
                lam_g_prev,
            )
        solve_times.append(time.perf_counter() - start)
        command = np.asarray(command, dtype=float).reshape(-1)
        predicted = np.asarray(predicted, dtype=float)
        commands.append(command)
        state = predicted[:, 1].copy()

        likelihoods, categories, _outputs = realized_measurement_likelihoods(
            state[:3],
            trees,
            true_classes,
            update_sensor,
            params,
        )
        distances = np.linalg.norm(trees - state[:2], axis=1)
        update_mask = (
            (distances <= float(params["observation_range"]))
            & (np.max(beliefs, axis=1) < float(args.confidence_threshold))
        )
        updated = beliefs.copy()
        for index in range(len(trees)):
            if update_mask[index]:
                updated[index] = bayes_update(beliefs[index], likelihoods[index])
        beliefs = updated

        states.append(state.copy())
        belief_history.append(beliefs.copy())
        entropy_history.append(np.asarray([entropy_bits(belief) for belief in beliefs]))
        total_entropy.append(float(np.sum(entropy_history[-1])))
        min_tree_distances.append(float(np.min(distances)))
        categories_history.append(categories.copy())

    return {
        "params": params,
        "trees": trees,
        "true_classes": true_classes,
        "initial_state": initial_state,
        "trajectory": np.asarray(states, dtype=float).T,
        "belief_history": np.asarray(belief_history, dtype=float),
        "entropy_history": np.asarray(entropy_history, dtype=float),
        "total_entropy": np.asarray(total_entropy, dtype=float),
        "min_tree_distances": np.asarray(min_tree_distances, dtype=float),
        "categories_history": np.asarray(categories_history, dtype=int),
        "target_history": target_history,
        "commands": np.asarray(commands, dtype=float),
        "solve_times": np.asarray(solve_times, dtype=float),
        "termination_reason": termination_reason,
        "checkpoint": str(checkpoint),
        "weights_cache": str(args.weights_cache),
        "confidence_threshold": float(args.confidence_threshold),
        "flip_start_yaw": bool(args.flip_start_yaw),
    }


def write_csv(path, data):
    trajectory = data["trajectory"]
    belief_history = data["belief_history"]
    entropy_history = data["entropy_history"]
    solve_times = np.concatenate(([0.0], data["solve_times"]))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        header = [
            "iteration",
            "time_s",
            "x",
            "y",
            "yaw_deg",
            "solve_time_s",
            "total_entropy_bits",
            "min_tree_distance",
        ]
        for index in range(len(data["trees"])):
            header.extend(
                [
                    "tree{}_belief_ripe".format(index),
                    "tree{}_belief_raw".format(index),
                    "tree{}_entropy_bits".format(index),
                    "tree{}_observation".format(index),
                ]
            )
        writer.writerow(header)
        for iteration in range(trajectory.shape[1]):
            categories = data["categories_history"][iteration]
            row = [
                iteration,
                iteration * float(data["params"]["dt"]),
                trajectory[0, iteration],
                trajectory[1, iteration],
                np.rad2deg(trajectory[2, iteration]),
                solve_times[iteration],
                data["total_entropy"][iteration],
                data["min_tree_distances"][iteration],
            ]
            for index in range(len(data["trees"])):
                row.extend(
                    [
                        belief_history[iteration, index, 0],
                        belief_history[iteration, index, 1],
                        entropy_history[iteration, index],
                        int(categories[index]),
                    ]
                )
            writer.writerow(row)


def plot_multi_tree(path, data):
    params = data["params"]
    trees = data["trees"]
    trajectory = data["trajectory"]
    belief_history = data["belief_history"]
    entropy_history = data["entropy_history"]
    time_axis = np.arange(trajectory.shape[1]) * float(params["dt"])
    yaw_deg = np.rad2deg(trajectory[2])
    solve_times = data["solve_times"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    ax_traj, ax_pose, ax_exec, ax_entropy, ax_conf, ax_final = axes.ravel()

    ax_traj.plot(trajectory[0], trajectory[1], color="#f97316", linewidth=1.6, label="trajectory")
    ax_traj.scatter(trajectory[0, 0], trajectory[1, 0], marker="x", s=90, color="#b91c1c", label="start")
    ax_traj.scatter(trajectory[0, -1], trajectory[1, -1], marker="*", s=130, color="#1d4ed8", label="final")
    for index, tree in enumerate(trees):
        color = "#22c55e" if data["true_classes"][index] == "ripe" else "#f97316"
        ax_traj.scatter(tree[0], tree[1], marker="^", s=100, color=color)
        ax_traj.text(tree[0] + 0.08, tree[1] + 0.08, str(index), fontsize=9)
        ax_traj.add_patch(
            plt.Circle(
                tree,
                float(params["safe_distance"]),
                fill=False,
                color="#dc2626",
                linewidth=0.8,
                alpha=0.45,
            )
        )
    arrow_step = max(1, trajectory.shape[1] // 8)
    for idx in range(0, trajectory.shape[1], arrow_step):
        yaw = trajectory[2, idx]
        ax_traj.arrow(
            trajectory[0, idx],
            trajectory[1, idx],
            0.35 * np.cos(yaw),
            0.35 * np.sin(yaw),
            head_width=0.10,
            color="#78350f",
            length_includes_head=True,
        )
    ax_traj.set_title("Multi-tree Trajectory")
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("y [m]")
    ax_traj.axis("equal")
    ax_traj.grid(True, alpha=0.25)
    ax_traj.legend(fontsize=8)

    ax_pose.plot(time_axis, trajectory[0], label="x", color="#2563eb")
    ax_pose.plot(time_axis, trajectory[1], label="y", color="#16a34a")
    ax_pose_yaw = ax_pose.twinx()
    ax_pose_yaw.plot(time_axis, yaw_deg, label="yaw", color="#dc2626", linestyle="--")
    ax_pose.set_title("Pose")
    ax_pose.set_xlabel("time [s]")
    ax_pose.set_ylabel("position [m]")
    ax_pose_yaw.set_ylabel("yaw [deg]")
    lines, labels = ax_pose.get_legend_handles_labels()
    yaw_lines, yaw_labels = ax_pose_yaw.get_legend_handles_labels()
    ax_pose.legend(lines + yaw_lines, labels + yaw_labels, fontsize=8)
    ax_pose.grid(True, alpha=0.25)

    ax_exec.plot(np.arange(len(solve_times)), solve_times, marker=".", color="#4f46e5")
    active = solve_times[solve_times > 0.0]
    warm = active[1:] if len(active) > 1 else np.asarray([])
    ax_exec.text(
        0.98,
        0.92,
        "active {:.3f}s\nwarm {:.3f}s".format(
            float(np.mean(active)) if len(active) else 0.0,
            float(np.mean(warm)) if len(warm) else 0.0,
        ),
        transform=ax_exec.transAxes,
        ha="right",
        va="top",
    )
    ax_exec.set_title("Solve Time")
    ax_exec.set_xlabel("iteration")
    ax_exec.set_ylabel("seconds")
    ax_exec.grid(True, alpha=0.25)

    ax_entropy.plot(time_axis, data["total_entropy"], color="#7c3aed", marker="o", label="total")
    ax_entropy.axhline(0.0, color="#111827", linewidth=0.8)
    ax_entropy.set_title("Total Ripe/Raw Entropy")
    ax_entropy.set_xlabel("time [s]")
    ax_entropy.set_ylabel("bits")
    ax_entropy.grid(True, alpha=0.25)

    confidence = np.max(belief_history, axis=2)
    for index in range(confidence.shape[1]):
        ax_conf.plot(time_axis, confidence[:, index], label="tree {}".format(index), linewidth=1.2)
    ax_conf.axhline(data["confidence_threshold"], color="#111827", linestyle="--", linewidth=0.9)
    ax_conf.set_title("Belief Confidence")
    ax_conf.set_xlabel("time [s]")
    ax_conf.set_ylabel("max P(class)")
    ax_conf.set_ylim(0.45, 1.02)
    ax_conf.grid(True, alpha=0.25)
    ax_conf.legend(fontsize=7, ncol=2)

    final_beliefs = belief_history[-1]
    x = np.arange(len(trees))
    ax_final.bar(x - 0.18, final_beliefs[:, 0], width=0.36, color="#ef4444", label="P(ripe)")
    ax_final.bar(x + 0.18, final_beliefs[:, 1], width=0.36, color="#22c55e", label="P(raw)")
    ax_final.set_xticks(x)
    ax_final.set_xticklabels(["{} {}".format(i, cls) for i, cls in enumerate(data["true_classes"])], rotation=25)
    ax_final.set_ylim(0.0, 1.0)
    ax_final.set_title("Final Beliefs")
    ax_final.set_ylabel("probability")
    ax_final.grid(True, axis="y", alpha=0.25)
    ax_final.legend(fontsize=8)

    fig.suptitle(
        "Multi-tree NMPC diagnostic: trees={}, active_targets={}, termination={}, flip_start_yaw={}".format(
            len(trees),
            int(params["num_target_trees"]),
            data["termination_reason"],
            data["flip_start_yaw"],
        ),
        fontsize=14,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "nmpc_multi_tree_diagnostic"))
    parser.add_argument("--tree-count", type=int, default=5)
    parser.add_argument("--tree-spacing", type=float, default=4.5)
    parser.add_argument("--active-targets", type=int, default=2)
    parser.add_argument("--obstacles", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--dt", type=float, default=0.25)
    parser.add_argument("--safe-distance", type=float, default=1.5)
    parser.add_argument("--confidence-threshold", type=float, default=0.9975245006578829)
    parser.add_argument("--information-gain-weight", type=float, default=20.0)
    parser.add_argument("--entropy-time-pressure-weight", type=float, default=20.0)
    parser.add_argument("--max-velocity", type=float, default=1.75)
    parser.add_argument("--max-yaw-velocity-deg", type=float, default=90.0)
    parser.add_argument("--max-accel-xy", type=float, default=2.0)
    parser.add_argument("--max-accel-yaw-deg", type=float, default=180.0)
    parser.add_argument("--ipopt-max-iter", type=int, default=8)
    parser.add_argument("--ipopt-max-cpu-time", type=float, default=0.12)
    parser.add_argument("--flip-start-yaw", action="store_true")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--weights-cache", default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.weights_cache is None:
        args.weights_cache = str(output_dir / "{}_casadi_weights.npz".format(Path(args.checkpoint).stem))

    data = run_multi_tree_diagnostic(args)
    figure_path = output_dir / "nmpc_multi_tree_diagnostic.png"
    csv_path = output_dir / "nmpc_multi_tree_diagnostic.csv"
    plot_multi_tree(figure_path, data)
    write_csv(csv_path, data)

    active = data["solve_times"][data["solve_times"] > 0.0]
    warm = active[1:] if len(active) > 1 else np.asarray([])
    final_confidence = np.max(data["belief_history"][-1], axis=1)
    print("Saved multi-tree diagnostic plot to {}".format(figure_path))
    print("Saved multi-tree diagnostic data to {}".format(csv_path))
    print("Trees: {}".format(len(data["trees"])))
    print("Active targets: {}".format(data["params"]["num_target_trees"]))
    print("Termination: {}".format(data["termination_reason"]))
    print("Initial state [x, y, yaw_deg]: {}".format(
        [
            float(data["initial_state"][0]),
            float(data["initial_state"][1]),
            float(np.rad2deg(data["initial_state"][2])),
        ]
    ))
    print("Final total entropy: {:.4f} bits".format(float(data["total_entropy"][-1])))
    print("Entropy reduction: {:.4f} bits".format(float(data["total_entropy"][0] - data["total_entropy"][-1])))
    print("Tracked trees: {}/{}".format(
        int(np.sum(final_confidence >= data["confidence_threshold"])),
        len(final_confidence),
    ))
    print("Minimum tree distance: {:.3f}m".format(float(np.min(data["min_tree_distances"]))))
    if len(active):
        print("Mean active solve time: {:.3f}s".format(float(np.mean(active))))
    if len(warm):
        print("Mean warm solve time: {:.3f}s".format(float(np.mean(warm))))
    print("MLP checkpoint: {}".format(data["checkpoint"]))
    print("MLP weights cache: {}".format(data["weights_cache"]))


if __name__ == "__main__":
    main()
