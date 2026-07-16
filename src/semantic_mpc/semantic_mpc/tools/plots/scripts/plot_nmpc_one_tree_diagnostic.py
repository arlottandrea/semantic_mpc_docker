#!/usr/bin/env python3
"""Plot the one-tree NMPC/MLP-CasADi diagnostic scenario.

The script runs a seeded one-tree NMPC solve with the trained structured MLP
converted to native CasADi operations.  It is intentionally independent from
ROS so it can isolate whether the optimizer moves toward an entropy-reducing
pose.
"""

import argparse
import json
import csv
from pathlib import Path
import sys
import time

import casadi as ca
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# The offline diagnostic only needs configuration defaults; keep it runnable
# in the non-ROS Pixi environment used by CI and local plot generation.
try:
    import rospy  # noqa: F401
except ImportError:
    import types

    rospy = types.ModuleType("rospy")
    rospy.get_param = lambda name, default=None: default
    rospy.has_param = lambda name: False
    sys.modules["rospy"] = rospy


def repo_root():
    path = Path(__file__).resolve()
    for parent in [path] + list(path.parents):
        if (parent / "pixi.toml").is_file():
            return parent
    return Path.cwd()


ROOT = repo_root()
SEMANTIC_SRC = ROOT / "src" / "semantic_mpc" / "semantic_mpc" / "src"
if str(SEMANTIC_SRC) not in sys.path:
    sys.path.insert(0, str(SEMANTIC_SRC))

from semantic_mpc_package.nmpc_config import default_nmpc_params
from semantic_mpc_package.nmpc_optimizer import NmpcOptimizer
from semantic_mpc_package.casadi_mlp_sensor import trained_mlp_sensor


DEFAULT_CHECKPOINT = (
    ROOT / "models" / "nmpc" / "fog" / "yaw_enriched" / "5m" / "best_model_epoch_39.pth"
)


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def angular_distance(left, right):
    return np.abs(wrap_angle(np.asarray(left) - right))


def relative_mlp_yaw_from_pose(pose, target, camera_yaw_offset=0.0):
    pose = np.asarray(pose, dtype=float).reshape(-1)
    target = np.asarray(target, dtype=float).reshape(-1)
    direction_to_tree = np.arctan2(target[1] - pose[1], target[0] - pose[0])
    return float(wrap_angle(direction_to_tree - pose[2] - float(camera_yaw_offset)))


def absolute_yaw_from_relative_mlp_yaw(relative_xy, relative_yaw, camera_yaw_offset=0.0):
    relative_xy = np.asarray(relative_xy, dtype=float)
    direction_to_tree = np.arctan2(-relative_xy[..., 1], -relative_xy[..., 0])
    return wrap_angle(direction_to_tree - relative_yaw - float(camera_yaw_offset))


def checkpoint_dataset_paths(checkpoint):
    metadata_path = Path(checkpoint).with_name("training_metadata.json")
    if not metadata_path.is_file():
        return []
    with metadata_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    return [Path(path) for path in metadata.get("sources", [])]


def infer_dataset_label(path, index):
    name = Path(path).as_posix().lower()
    if "raw" in name and "ripe" not in name:
        return "raw"
    if "ripe" in name:
        return "ripe"
    return "raw" if index == 0 else "ripe"


def load_dataset_points(paths, target, max_points=8000):
    points = []
    for index, path in enumerate(paths):
        path = Path(path)
        if not path.is_file():
            continue
        label = infer_dataset_label(path, index)
        with path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                try:
                    rel_x = float(row["x"])
                    rel_y = float(row["y"])
                    yaw = wrap_angle(float(row["yaw"]))
                except (KeyError, TypeError, ValueError):
                    continue
                points.append(
                    {
                        "x": float(target[0] + rel_x),
                        "y": float(target[1] + rel_y),
                        "yaw": float(yaw),
                        "label": label,
                    }
                )
    if not points:
        return []
    max_points = int(max_points)
    if max_points > 0 and len(points) > max_points:
        rng = np.random.default_rng(0)
        selected = []
        for label in ["raw", "ripe"]:
            label_points = [point for point in points if point["label"] == label]
            budget = max(1, max_points // 2)
            if len(label_points) > budget:
                indices = rng.choice(len(label_points), size=budget, replace=False)
                selected.extend(label_points[int(idx)] for idx in indices)
            else:
                selected.extend(label_points)
        points = selected[:max_points]
    return points


def dataset_yaw_values(dataset_points, max_values=37):
    if not dataset_points:
        return []
    yaws = np.asarray([point["yaw"] for point in dataset_points], dtype=float)
    yaws = np.unique(np.round(wrap_angle(yaws), 4))
    yaws.sort()
    if len(yaws) <= max_values:
        return yaws.tolist()
    indices = np.linspace(0, len(yaws) - 1, int(max_values)).round().astype(int)
    return yaws[indices].tolist()


def diagnostic_params(horizon):
    params = default_nmpc_params()
    params.update(
        {
            "model_device": "gpu",
            "num_target_trees": 1,
            "num_obstacle_trees": 1,
            "active_target_count": 1,
            "active_obstacle_count": 1,
            "dt": 0.25,
            "mpc_horizon": int(horizon),
            "safe_distance": 1.0,
            "movement_weight": 1e-3,
            "yaw_movement_weight": 1e-3,
            "acceleration_regularization_weight": 1e-5,
            "information_gain_weight": 20.0,
            "entropy_time_pressure_weight": 20.0,
            "exploration_weight": 0.0,
            "attraction_weight": 0.0,
            "camera_facing_weight": 0.5,
            "observation_standoff_weight": 0.2,
            "running_camera_facing_weight": 0.1,
            "running_observation_standoff_weight": 0.1,
            "orbit_velocity_weight": 0.3,
            "max_velocity": 1.75,
            "max_yaw_velocity": np.pi / 2.0,
            "max_accel_xy": 2.0,
            "max_accel_yaw": np.pi,
            "ipopt": {
                "print_level": 0,
                "sb": "yes",
                "max_iter": 8,
                "tol": 1e-2,
                "acceptable_tol": 1e-1,
                "acceptable_iter": 2,
                "max_cpu_time": 0.12,
                "warm_start_init_point": "yes",
                "warm_start_bound_push": 1e-6,
                "warm_start_mult_bound_push": 1e-6,
                "mu_init": 1e-4,
                "hessian_approximation": "limited-memory",
            },
        }
    )
    return params


def entropy_bits(belief):
    belief = np.clip(np.asarray(belief, dtype=float), 1e-9, 1.0)
    return float(-np.sum(belief * np.log2(belief)))


def bayes_update(belief, likelihood):
    posterior = np.clip(np.asarray(belief, dtype=float), 1e-9, None)
    posterior *= np.clip(np.asarray(likelihood, dtype=float), 1e-9, None)
    posterior /= np.sum(posterior)
    return posterior


def make_optimizer_parameters(state, target_matrix, belief, obstacle, target_mask):
    return ca.vertcat(
        ca.DM(state),
        ca.reshape(ca.DM(target_matrix), 2, 1),
        ca.reshape(ca.DM(belief.reshape(1, 2)), 2, 1),
        ca.reshape(ca.DM(target_mask), 1, 1),
        ca.reshape(ca.DM(obstacle.reshape(1, 2)), 2, 1),
    )


def sample_initial_state(args, target, params, seed=None):
    rng = np.random.default_rng(int(args.seed if seed is None else seed))
    radius_min = float(args.start_radius_min)
    radius_max = float(args.start_radius_max)
    if radius_min <= 0.0 or radius_max < radius_min:
        raise ValueError("start radius bounds must satisfy 0 < min <= max")
    radius = rng.uniform(radius_min, radius_max)
    bearing = rng.uniform(-np.pi, np.pi)
    yaw = rng.uniform(-np.pi, np.pi)
    xy = target + radius * np.asarray([np.cos(bearing), np.sin(bearing)])
    yaw = np.clip(yaw, -float(params["max_heading_abs"]), float(params["max_heading_abs"]))
    return np.asarray([xy[0], xy[1], yaw, 0.0, 0.0, 0.0], dtype=float)


def find_best_mlp_viewpoint(target, params, checkpoint, weights_cache, grid_size=81, yaw_steps=37):
    radius = float(params["observation_range"])
    xs = np.linspace(-radius, radius, int(grid_size))
    ys = np.linspace(-radius, radius, int(grid_size))
    xx, yy = np.meshgrid(xs, ys)
    relative_xy = np.column_stack((xx.reshape(-1), yy.reshape(-1)))
    distances = np.linalg.norm(relative_xy, axis=1)
    valid = (
        (distances >= float(params["safe_distance"]))
        & (distances <= float(params["observation_range"]))
    )
    if not np.any(valid):
        raise ValueError("no valid MLP viewpoint outside safe distance and inside observation range")

    relative_xy = relative_xy[valid]
    sensor_cache = {}
    best = None
    for relative_yaw in np.linspace(-np.pi, np.pi, int(yaw_steps), endpoint=False):
        features = np.column_stack((relative_xy, np.full(len(relative_xy), relative_yaw)))
        outputs = evaluate_mlp_grid(
            features,
            checkpoint,
            weights_cache,
            params,
            chunk_size=512,
            sensor_cache=sensor_cache,
        )
        expected_entropy = expected_entropy_from_mlp_outputs(outputs, np.asarray([0.5, 0.5]))
        best_index = int(np.argmin(expected_entropy))
        candidate = {
            "relative_xy": relative_xy[best_index].copy(),
            "world_xy": target + relative_xy[best_index],
            "relative_yaw": float(relative_yaw),
            "absolute_yaw": float(
                absolute_yaw_from_relative_mlp_yaw(
                    relative_xy[best_index],
                    relative_yaw,
                    params.get("camera_yaw_offset", 0.0),
                )
            ),
            "expected_entropy": float(expected_entropy[best_index]),
            "raw_correct": float(outputs[best_index, 0]),
            "ripe_correct": float(outputs[best_index, 3]),
        }
        if best is None or candidate["expected_entropy"] < best["expected_entropy"]:
            best = candidate
    return best


def opposite_best_initial_state(target, params, best_viewpoint):
    relative = np.asarray(best_viewpoint["relative_xy"], dtype=float)
    opposite_xy = target - relative
    opposite_relative = -relative
    opposite_yaw = absolute_yaw_from_relative_mlp_yaw(
        opposite_relative,
        float(best_viewpoint["relative_yaw"]),
        params.get("camera_yaw_offset", 0.0),
    )
    opposite_yaw = np.clip(
        opposite_yaw,
        -float(params["max_heading_abs"]),
        float(params["max_heading_abs"]),
    )
    return np.asarray([opposite_xy[0], opposite_xy[1], float(opposite_yaw), 0.0, 0.0, 0.0], dtype=float)


def update_belief_from_pose(belief, pose, target, sensor, true_class, camera_yaw_offset=0.0):
    class_index = {"raw": 0, "ripe": 1}[true_class]
    category_labels = ["raw", "ripe"]
    relative_yaw = relative_mlp_yaw_from_pose(pose, target, camera_yaw_offset)
    features = ca.DM([[pose[0] - target[0], pose[1] - target[1], relative_yaw]])
    output = np.asarray(sensor(features), dtype=float).reshape(2, 2)
    observation = int(np.argmax(output[class_index]))
    likelihood = np.asarray([output[1, observation], output[0, observation]], dtype=float)
    return bayes_update(belief, likelihood), category_labels[observation]



def run_diagnostic(args):
    params = diagnostic_params(args.horizon)
    target = np.asarray([args.target_x, args.target_y], dtype=float)
    target_matrix = target.reshape(1, 2)
    checkpoint = Path(args.checkpoint)
    best_viewpoint = find_best_mlp_viewpoint(
        target,
        params,
        checkpoint,
        args.weights_cache,
        grid_size=args.best_view_grid_size,
        yaw_steps=args.best_view_yaw_steps,
    )
    if args.start_mode == "opposite-best":
        initial_state = opposite_best_initial_state(target, params, best_viewpoint)
    else:
        initial_state = sample_initial_state(args, target, params)
    if args.flip_start_yaw:
        initial_state[2] = np.clip(
            wrap_angle(initial_state[2] + np.pi),
            -float(params["max_heading_abs"]),
            float(params["max_heading_abs"]),
        )
    sensor = trained_mlp_sensor(
        params["mpc_horizon"],
        checkpoint,
        weights_cache_path=args.weights_cache,
        output_temperature=params.get("nn_output_temperature", 0.4),
        yaw_harmonics=params.get("nn_yaw_harmonics", 4),
        include_alignment_features=params.get("nn_include_alignment_features", True),
    )
    update_sensor = trained_mlp_sensor(
        1,
        checkpoint,
        weights_cache_path=args.weights_cache,
        output_temperature=params.get("nn_output_temperature", 0.4),
        yaw_harmonics=params.get("nn_yaw_harmonics", 4),
        include_alignment_features=params.get("nn_include_alignment_features", True),
    )
    optimizer = NmpcOptimizer(params, sensor)
    obstacle = np.asarray([[50.0, 50.0]])
    target_mask = np.ones(1)
    lb = np.minimum.reduce(
        [
            target - (float(params["observation_range"]) + 2.0),
            initial_state[:2] - 3.0,
        ]
    )
    ub = np.maximum.reduce(
        [
            target + (float(params["observation_range"]) + 2.0),
            initial_state[:2] + 3.0,
        ]
    )

    state = initial_state.copy()
    belief = np.asarray([0.5, 0.5], dtype=float)  # [ripe, raw]
    states = [state.copy()]
    beliefs = [belief.copy()]
    entropies = [entropy_bits(belief)]
    categories = ["prior"]
    commands = []
    solve_times = []
    mpc_step = None
    x_dec_prev = None
    lam_g_prev = None

    for iteration in range(int(args.iterations)):
        if entropies[-1] <= float(args.stop_entropy):
            commands.append(np.zeros(3))
            solve_times.append(0.0)
            states.append(state.copy())
            beliefs.append(belief.copy())
            entropies.append(entropies[-1])
            categories.append("held")
            continue

        start = time.perf_counter()
        if mpc_step is None:
            mpc_step, command, predicted, x_dec_prev, lam_g_prev = optimizer.mpc_opt(
                target_matrix,
                belief.reshape(1, 2),
                target_mask,
                obstacle,
                lb,
                ub,
                state,
                steps=params["mpc_horizon"],
            )
        else:
            p0_val = make_optimizer_parameters(
                state, target_matrix, belief, obstacle, target_mask
            )
            command, predicted, x_dec_prev, lam_g_prev = mpc_step(
                p0_val, x_dec_prev, lam_g_prev
            )
        solve_times.append(time.perf_counter() - start)
        command = np.asarray(command, dtype=float).reshape(-1)
        predicted = np.asarray(predicted, dtype=float)
        commands.append(command)

        # Closed-loop diagnostic: assume the vehicle tracks the first predicted
        # horizon state exactly for this isolated optimizer test.
        state = predicted[:, 1].copy()
        belief, category = update_belief_from_pose(
            belief,
            state[:3],
            target,
            update_sensor,
            args.true_class,
            params.get("camera_yaw_offset", 0.0),
        )
        states.append(state.copy())
        beliefs.append(belief.copy())
        entropies.append(entropy_bits(belief))
        categories.append(category)

    trajectory = np.asarray(states, dtype=float).T
    beliefs = np.asarray(beliefs, dtype=float)
    entropies = np.asarray(entropies, dtype=float)
    commands = np.asarray(commands, dtype=float)
    solve_times = np.asarray(solve_times, dtype=float)
    time_axis = np.arange(trajectory.shape[1], dtype=float) * params["dt"]
    entropy_reduction = entropies[0] - entropies
    distances = np.linalg.norm(trajectory[:2, :].T - target, axis=1)

    return {
        "params": params,
        "target": target,
        "initial_state": initial_state,
        "best_viewpoint": best_viewpoint,
        "start_mode": args.start_mode,
        "flip_start_yaw": bool(args.flip_start_yaw),
        "seed": int(args.seed),
        "checkpoint": str(checkpoint),
        "weights_cache": str(args.weights_cache),
        "commands": commands,
        "trajectory": trajectory,
        "time_axis": time_axis,
        "solve_times": solve_times,
        "total_solve_time": float(np.sum(solve_times)),
        "beliefs": beliefs,
        "entropies": entropies,
        "entropy_reduction": entropy_reduction,
        "distances": distances,
        "categories": categories,
        "true_class": args.true_class,
    }


def write_csv(path, data):
    yaw_degrees = np.rad2deg(data["trajectory"][2])
    solve_times = np.concatenate(([0.0], data["solve_times"]))
    rows = zip(
        data["time_axis"],
        np.arange(len(data["time_axis"])),
        data["trajectory"][0],
        data["trajectory"][1],
        yaw_degrees,
        solve_times,
        data["distances"],
        data["distances"] - data["params"]["safe_distance"],
        data["entropies"],
        data["entropy_reduction"],
        data["beliefs"][:, 0],
        data["beliefs"][:, 1],
        data["categories"],
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time_s",
                "iteration",
                "x",
                "y",
                "yaw_deg",
                "solve_time_s",
                "distance_to_tree",
                "safe_distance_margin",
                "entropy_bits",
                "entropy_reduction_bits",
                "belief_ripe",
                "belief_raw",
                "observation",
            ]
        )
        writer.writerows(rows)


def plot_diagnostic(path, data):
    params = data["params"]
    target = data["target"]
    trajectory = data["trajectory"]
    yaw_degrees = np.rad2deg(trajectory[2])
    time_axis = data["time_axis"]
    beliefs = data["beliefs"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    ax_traj, ax_pose, ax_exec, ax_entropy, ax_reduction, ax_belief = axes.ravel()

    ax_traj.plot(trajectory[0], trajectory[1], color="#d97706", marker="o", label="NMPC executed")
    ax_traj.scatter(trajectory[0, 0], trajectory[1, 0], color="#b91c1c", marker="x", s=90, label="start")
    ax_traj.scatter(trajectory[0, -1], trajectory[1, -1], color="#1d4ed8", marker="*", s=130, label="final")
    ax_traj.scatter(target[0], target[1], color="#15803d", marker="^", s=120, label="tree")
    if data.get("best_viewpoint") is not None:
        best_xy = np.asarray(data["best_viewpoint"]["world_xy"], dtype=float)
        ax_traj.scatter(best_xy[0], best_xy[1], color="#9333ea", marker="D", s=85, label="MLP best POV")
    observation_circle = plt.Circle(
        target,
        params["observation_range"],
        fill=False,
        linestyle="--",
        color="#15803d",
        alpha=0.45,
    )
    ax_traj.add_patch(observation_circle)
    safety_circle = plt.Circle(
        target,
        params["safe_distance"],
        fill=False,
        linestyle="-",
        color="#dc2626",
        alpha=0.75,
        label="safe distance",
    )
    ax_traj.add_patch(safety_circle)
    arrow_step = max(1, trajectory.shape[1] // 5)
    for idx in range(0, trajectory.shape[1], arrow_step):
        yaw = trajectory[2, idx]
        ax_traj.arrow(
            trajectory[0, idx],
            trajectory[1, idx],
            0.45 * np.cos(yaw),
            0.45 * np.sin(yaw),
            head_width=0.12,
            color="#78350f",
            length_includes_head=True,
        )
    ax_traj.set_title("Trajectory Pose")
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("y [m]")
    ax_traj.axis("equal")
    ax_traj.grid(True, alpha=0.25)
    ax_traj.legend(fontsize=8)

    ax_pose.plot(time_axis, trajectory[0], label="x", color="#2563eb")
    ax_pose.plot(time_axis, trajectory[1], label="y", color="#16a34a")
    ax_pose_yaw = ax_pose.twinx()
    ax_pose_yaw.plot(time_axis, yaw_degrees, label="yaw", color="#dc2626", linestyle="--")
    ax_pose.set_title("Pose Over Iterations")
    ax_pose.set_xlabel("time [s]")
    ax_pose.set_ylabel("position [m]")
    ax_pose_yaw.set_ylabel("yaw [deg]")
    ax_pose.grid(True, alpha=0.25)
    lines, labels = ax_pose.get_legend_handles_labels()
    yaw_lines, yaw_labels = ax_pose_yaw.get_legend_handles_labels()
    ax_pose.legend(lines + yaw_lines, labels + yaw_labels, fontsize=8)

    iteration_axis = np.arange(len(data["solve_times"]))
    active_solve_times = data["solve_times"][data["solve_times"] > 0.0]
    warm_solve_times = active_solve_times[1:] if len(active_solve_times) > 1 else np.asarray([])
    ax_exec.plot(iteration_axis, data["solve_times"], color="#4f46e5", marker=".", linewidth=1.0)
    ax_exec.set_title("Execution Time")
    ax_exec.set_xlabel("iteration")
    ax_exec.set_ylabel("seconds")
    ax_exec.grid(True, alpha=0.25)
    ax_exec.text(
        0.98,
        0.92,
        "total {:.2f}s\nactive {:.3f}s\nwarm {:.3f}s".format(
            data["total_solve_time"],
            float(np.mean(active_solve_times)) if len(active_solve_times) else 0.0,
            float(np.mean(warm_solve_times)) if len(warm_solve_times) else 0.0,
        ),
        transform=ax_exec.transAxes,
        ha="right",
        va="top",
    )

    ax_entropy.plot(time_axis, data["entropies"], color="#7c3aed", marker="o")
    ax_entropy.set_title("Entropy")
    ax_entropy.set_xlabel("time [s]")
    ax_entropy.set_ylabel("bits")
    ax_entropy.set_ylim(bottom=0.0, top=1.05)
    ax_entropy.grid(True, alpha=0.25)

    ax_reduction.plot(time_axis, data["entropy_reduction"], color="#0891b2", marker="o")
    ax_reduction.set_title("Entropy Reduction")
    ax_reduction.set_xlabel("time [s]")
    ax_reduction.set_ylabel("bits reduced")
    ax_reduction.set_ylim(bottom=0.0, top=1.05)
    ax_reduction.grid(True, alpha=0.25)

    ax_belief.plot(time_axis, beliefs[:, 0], color="#ef4444", marker="o", label="P(ripe)")
    ax_belief.plot(time_axis, beliefs[:, 1], color="#22c55e", marker="o", label="P(raw)")
    ax_belief.set_title("Belief Trend")
    ax_belief.set_xlabel("time [s]")
    ax_belief.set_ylabel("probability")
    ax_belief.set_ylim(0.0, 1.0)
    ax_belief.grid(True, alpha=0.25)
    ax_belief.legend(fontsize=8)

    fig.suptitle(
        (
            "One-tree NMPC trained-MLP diagnostic: seed={}, start=({:.1f}, {:.1f}, {:.1f} deg), "
            "target=({:.1f}, {:.1f}), true_class={}, start_mode={}, flip_start_yaw={}"
        ).format(
            data["seed"],
            data["initial_state"][0],
            data["initial_state"][1],
            np.rad2deg(data["initial_state"][2]),
            target[0],
            target[1],
            data["true_class"],
            data["start_mode"],
            data.get("flip_start_yaw", False),
        ),
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(path, dpi=180)
    plt.close(fig)


def expected_entropy_from_mlp_outputs(outputs, prior):
    outputs = np.asarray(outputs, dtype=float).reshape(-1, 2, 2)
    prior = np.asarray(prior, dtype=float).reshape(2)
    prior = np.clip(prior, 1e-9, None)
    prior /= np.sum(prior)
    raw = outputs[:, 0, :]
    ripe = outputs[:, 1, :]
    entropy = np.zeros(outputs.shape[0], dtype=float)
    for observation in range(2):
        ripe_joint = prior[0] * ripe[:, observation]
        raw_joint = prior[1] * raw[:, observation]
        probability = ripe_joint + raw_joint
        safe_probability = np.clip(probability, 1e-12, None)
        posterior_ripe = ripe_joint / safe_probability
        posterior_raw = raw_joint / safe_probability
        posterior = np.column_stack((posterior_ripe, posterior_raw))
        entropy_terms = posterior * np.log2(np.clip(posterior, 1e-12, 1.0))
        entropy -= probability * np.sum(entropy_terms, axis=1)
    return entropy


def evaluate_mlp_grid(
    features,
    checkpoint,
    weights_cache,
    params,
    chunk_size=512,
    sensor_cache=None,
):
    outputs = []
    sensors = {} if sensor_cache is None else sensor_cache
    start = 0
    while start < len(features):
        stop = min(start + int(chunk_size), len(features))
        batch_size = stop - start
        if batch_size not in sensors:
            sensors[batch_size] = trained_mlp_sensor(
                batch_size,
                checkpoint,
                weights_cache_path=weights_cache,
                output_temperature=params.get("nn_output_temperature", 0.4),
                yaw_harmonics=params.get("nn_yaw_harmonics", 4),
                include_alignment_features=params.get("nn_include_alignment_features", True),
            )
        sensor = sensors[batch_size]
        outputs.append(np.asarray(sensor(ca.DM(features[start:stop])), dtype=float))
        start = stop
    return np.vstack(outputs)


def yaw_tolerance_from_frames(yaw_frames, fallback_deg=8.0):
    if len(yaw_frames) < 2:
        return np.deg2rad(float(fallback_deg))
    yaws = np.sort(np.asarray(yaw_frames, dtype=float))
    gaps = np.diff(yaws)
    gaps = gaps[gaps > 1e-6]
    if len(gaps) == 0:
        return np.deg2rad(float(fallback_deg))
    return 0.55 * float(np.median(gaps))


def dataset_points_for_yaw(dataset_points, yaw, tolerance):
    if not dataset_points:
        return []
    return [
        point
        for point in dataset_points
        if angular_distance(point["yaw"], yaw) <= float(tolerance)
    ]


def plot_dataset_overlay(ax, dataset_points, yaw, tolerance):
    points = dataset_points_for_yaw(dataset_points, yaw, tolerance)
    if not points:
        return
    for label, color, marker in [
        ("raw", "#f97316", "."),
        ("ripe", "#22c55e", "."),
    ]:
        selected = [point for point in points if point["label"] == label]
        if not selected:
            continue
        ax.scatter(
            [point["x"] for point in selected],
            [point["y"] for point in selected],
            s=10,
            marker=marker,
            color=color,
            alpha=0.35,
            linewidths=0,
            label="dataset {}".format(label),
        )


def plot_mlp_heatmap(path, data, grid_size=121, radius=6.0, dataset_points=None):
    params = data["params"]
    target = data["target"]
    trajectory = data["trajectory"]
    final_yaw = float(trajectory[2, -1])
    final_relative_yaw = relative_mlp_yaw_from_pose(
        trajectory[:3, -1],
        target,
        params.get("camera_yaw_offset", 0.0),
    )
    checkpoint = Path(data["checkpoint"])
    weights_cache = data["weights_cache"]
    grid_size = max(5, int(grid_size))
    radius = float(radius)

    xs = np.linspace(target[0] - radius, target[0] + radius, grid_size)
    ys = np.linspace(target[1] - radius, target[1] + radius, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    features = np.column_stack(
        (
            xx.reshape(-1) - target[0],
            yy.reshape(-1) - target[1],
            np.full(xx.size, final_relative_yaw),
        )
    )
    outputs = evaluate_mlp_grid(features, checkpoint, weights_cache, params)
    raw_correct = outputs[:, 0].reshape(grid_size, grid_size)
    ripe_correct = outputs[:, 3].reshape(grid_size, grid_size)
    expected_entropy = expected_entropy_from_mlp_outputs(
        outputs,
        data["beliefs"][-1],
    ).reshape(grid_size, grid_size)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    extent = [xs[0], xs[-1], ys[0], ys[-1]]
    panels = [
        (raw_correct, "P(obs=raw | true=raw)", "viridis", 0.0, 1.0),
        (ripe_correct, "P(obs=ripe | true=ripe)", "viridis", 0.0, 1.0),
        (expected_entropy, "Expected posterior entropy [bits]", "magma", 0.0, 1.0),
    ]
    dataset_tolerance = np.deg2rad(8.0)
    for ax, (values, title, cmap, vmin, vmax) in zip(axes, panels):
        image = ax.imshow(
            values,
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        plot_dataset_overlay(ax, dataset_points or [], final_relative_yaw, dataset_tolerance)
        ax.plot(trajectory[0], trajectory[1], color="#f97316", linewidth=1.6, label="trajectory")
        ax.scatter(trajectory[0, -1], trajectory[1, -1], color="#1d4ed8", marker="*", s=110, label="final")
        ax.scatter(target[0], target[1], color="#16a34a", marker="^", s=100, label="tree")
        if data.get("best_viewpoint") is not None:
            best_xy = np.asarray(data["best_viewpoint"]["world_xy"], dtype=float)
            ax.scatter(best_xy[0], best_xy[1], color="#9333ea", marker="D", s=80, label="MLP best POV")
        ax.add_patch(
            plt.Circle(
                target,
                params["safe_distance"],
                fill=False,
                color="#dc2626",
                linewidth=1.2,
            )
        )
        ax.arrow(
            trajectory[0, -1],
            trajectory[1, -1],
            0.55 * np.cos(final_yaw),
            0.55 * np.sin(final_yaw),
            head_width=0.14,
            color="#1d4ed8",
            length_includes_head=True,
        )
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.grid(False)
    axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(
        "Trained MLP heatmap at final MLP relative yaw {:.1f} deg (state yaw {:.1f} deg)".format(
            np.rad2deg(final_relative_yaw),
            np.rad2deg(final_yaw),
        ),
        fontsize=14,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def build_interactive_heatmap_payload(data, dataset_points, grid_size, radius, yaw_steps):
    params = data["params"]
    target = data["target"]
    trajectory = data["trajectory"]
    checkpoint = Path(data["checkpoint"])
    weights_cache = data["weights_cache"]
    grid_size = max(5, int(grid_size))
    radius = float(radius)

    dataset_yaws = dataset_yaw_values(dataset_points, max_values=max(3, int(yaw_steps)))
    if dataset_yaws:
        yaw_frames = dataset_yaws
    else:
        yaw_frames = np.linspace(-np.pi, np.pi, max(3, int(yaw_steps))).tolist()
    final_relative_yaw = relative_mlp_yaw_from_pose(
        trajectory[:3, -1],
        target,
        params.get("camera_yaw_offset", 0.0),
    )
    yaw_frames.append(final_relative_yaw)
    yaw_frames = np.unique(np.round(wrap_angle(np.asarray(yaw_frames)), 4)).tolist()
    yaw_frames.sort()

    xs = np.linspace(target[0] - radius, target[0] + radius, grid_size)
    ys = np.linspace(target[1] - radius, target[1] + radius, grid_size)
    xx, yy = np.meshgrid(xs, ys)
    base_xy = np.column_stack((xx.reshape(-1) - target[0], yy.reshape(-1) - target[1]))
    sensor_cache = {}
    frames = []
    for yaw in yaw_frames:
        features = np.column_stack((base_xy, np.full(len(base_xy), yaw)))
        outputs = evaluate_mlp_grid(
            features,
            checkpoint,
            weights_cache,
            params,
            chunk_size=512,
            sensor_cache=sensor_cache,
        )
        frames.append(
            {
                "yaw_deg": round(float(np.rad2deg(yaw)), 3),
                "raw": np.round(outputs[:, 0], 4).tolist(),
                "ripe": np.round(outputs[:, 3], 4).tolist(),
                "entropy": np.round(
                    # Use a uniform diagnostic prior.  Using the converged
                    # final belief makes entropy almost zero everywhere and
                    # hides the sensor model's spatial information structure.
                    expected_entropy_from_mlp_outputs(outputs, np.asarray([0.5, 0.5])),
                    4,
                ).tolist(),
            }
        )

    tolerance = yaw_tolerance_from_frames(yaw_frames)
    return {
        "gridSize": grid_size,
        "extent": [float(xs[0]), float(xs[-1]), float(ys[0]), float(ys[-1])],
        "target": [float(target[0]), float(target[1])],
        "safeDistance": float(params["safe_distance"]),
        "cameraYawOffsetDeg": round(float(np.rad2deg(params.get("camera_yaw_offset", 0.0))), 3),
        "trajectory": {
            "x": np.round(trajectory[0], 4).tolist(),
            "y": np.round(trajectory[1], 4).tolist(),
            "yawDeg": np.round(np.rad2deg(trajectory[2]), 3).tolist(),
        },
        "dataset": {
            "x": [round(point["x"], 4) for point in dataset_points],
            "y": [round(point["y"], 4) for point in dataset_points],
            "yawDeg": [round(float(np.rad2deg(point["yaw"])), 3) for point in dataset_points],
            "label": [point["label"] for point in dataset_points],
            "yawToleranceDeg": round(float(np.rad2deg(tolerance)), 3),
        },
        "frames": frames,
    }


def write_interactive_heatmap(path, payload):
    html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>NMPC Trained MLP Heatmap</title>
<style>
body { font-family: Arial, sans-serif; margin: 18px; color: #111827; background: #f8fafc; }
.controls { display: flex; gap: 12px; align-items: center; margin: 10px 0 18px; }
.panels { display: grid; grid-template-columns: repeat(3, minmax(260px, 1fr)); gap: 14px; }
.panel { background: white; border: 1px solid #d1d5db; border-radius: 6px; padding: 10px; }
.panel h3 { margin: 0 0 8px; font-size: 15px; font-weight: 600; }
canvas { width: 100%; aspect-ratio: 1 / 1; border: 1px solid #e5e7eb; background: white; }
.legend { font-size: 12px; margin-top: 8px; color: #374151; }
input[type="range"] { width: 420px; max-width: 55vw; }
</style>
</head>
<body>
<h2>Trained MLP Heatmap with Dataset Overlay</h2>
<div class="controls">
  <label for="yawSlider">MLP relative yaw</label>
  <input id="yawSlider" type="range" min="0" max="0" value="0" step="1">
  <strong id="yawValue"></strong>
  <span id="datasetCount"></span>
</div>
<div class="panels">
  <div class="panel"><h3>P(obs=raw | true=raw)</h3><canvas id="rawPanel" width="520" height="520"></canvas></div>
  <div class="panel"><h3>P(obs=ripe | true=ripe)</h3><canvas id="ripePanel" width="520" height="520"></canvas></div>
  <div class="panel"><h3>Expected posterior entropy [bits], prior=[0.5, 0.5]</h3><canvas id="entropyPanel" width="520" height="520"></canvas></div>
</div>
<div class="legend">
Dataset points are filtered to the selected MLP relative-yaw slice. Orange = raw dataset, green = ripe dataset.
Blue star/arrow = final drone pose, violet arrow = selected orientation wrt tree, green triangle = tree, red circle = safe distance.
</div>
<script>
const payload = __PAYLOAD__;
const slider = document.getElementById("yawSlider");
const yawValue = document.getElementById("yawValue");
const datasetCount = document.getElementById("datasetCount");
slider.max = payload.frames.length - 1;

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function lerp(a, b, t) { return a + (b - a) * t; }
function colorMap(value, cmap) {
  const t = clamp(value, 0, 1);
  if (cmap === "magma") {
    const stops = [[0,0,4],[73,16,108],[182,55,121],[251,136,97],[252,253,191]];
    return interpStops(stops, t);
  }
  const stops = [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
  return interpStops(stops, t);
}
function interpStops(stops, t) {
  const scaled = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(scaled));
  const f = scaled - i;
  return [
    Math.round(lerp(stops[i][0], stops[i + 1][0], f)),
    Math.round(lerp(stops[i][1], stops[i + 1][1], f)),
    Math.round(lerp(stops[i][2], stops[i + 1][2], f))
  ];
}
function worldToCanvas(x, y, canvas) {
  const [xmin, xmax, ymin, ymax] = payload.extent;
  return [
    (x - xmin) / (xmax - xmin) * canvas.width,
    canvas.height - (y - ymin) / (ymax - ymin) * canvas.height
  ];
}
function yawDiffDeg(a, b) {
  let d = a - b;
  while (d > 180) d -= 360;
  while (d < -180) d += 360;
  return Math.abs(d);
}
function drawArrow(ctx, x, y, angleRad, length, color, lineWidth, dash) {
  const endX = x + length * Math.cos(angleRad);
  const endY = y - length * Math.sin(angleRad);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.setLineDash(dash || []);
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(endX, endY);
  ctx.stroke();
  ctx.setLineDash([]);
  const head = Math.max(8, length * 0.28);
  const left = angleRad + Math.PI * 0.82;
  const right = angleRad - Math.PI * 0.82;
  ctx.beginPath();
  ctx.moveTo(endX, endY);
  ctx.lineTo(endX + head * Math.cos(left), endY - head * Math.sin(left));
  ctx.lineTo(endX + head * Math.cos(right), endY - head * Math.sin(right));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}
function drawOverlays(ctx, canvas, selectedYawDeg) {
  const traj = payload.trajectory;
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#f97316";
  ctx.beginPath();
  for (let i = 0; i < traj.x.length; i++) {
    const [cx, cy] = worldToCanvas(traj.x[i], traj.y[i], canvas);
    if (i === 0) ctx.moveTo(cx, cy); else ctx.lineTo(cx, cy);
  }
  ctx.stroke();

  let count = 0;
  for (let i = 0; i < payload.dataset.x.length; i++) {
    if (yawDiffDeg(payload.dataset.yawDeg[i], selectedYawDeg) > payload.dataset.yawToleranceDeg) continue;
    const [cx, cy] = worldToCanvas(payload.dataset.x[i], payload.dataset.y[i], canvas);
    ctx.fillStyle = payload.dataset.label[i] === "raw" ? "rgba(249,115,22,0.45)" : "rgba(34,197,94,0.45)";
    ctx.beginPath();
    ctx.arc(cx, cy, 2.0, 0, 2 * Math.PI);
    ctx.fill();
    count += 1;
  }

  const [tx, ty] = worldToCanvas(payload.target[0], payload.target[1], canvas);
  const [sx, sy] = worldToCanvas(payload.target[0] + payload.safeDistance, payload.target[1], canvas);
  ctx.strokeStyle = "#dc2626";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(tx, ty, Math.abs(sx - tx), 0, 2 * Math.PI);
  ctx.stroke();
  ctx.fillStyle = "#16a34a";
  ctx.beginPath();
  ctx.moveTo(tx, ty - 7);
  ctx.lineTo(tx - 7, ty + 7);
  ctx.lineTo(tx + 7, ty + 7);
  ctx.closePath();
  ctx.fill();

  const n = traj.x.length - 1;
  const [fx, fy] = worldToCanvas(traj.x[n], traj.y[n], canvas);
  const bearingToTreeRad = Math.atan2(payload.target[1] - traj.y[n], payload.target[0] - traj.x[n]);
  const selectedYawRad = selectedYawDeg * Math.PI / 180;
  const selectedStateYawRad = bearingToTreeRad - selectedYawRad - payload.cameraYawOffsetDeg * Math.PI / 180;
  const yawRad = traj.yawDeg[n] * Math.PI / 180;
  ctx.strokeStyle = "rgba(124,58,237,0.55)";
  ctx.lineWidth = 2;
  ctx.setLineDash([6, 4]);
  ctx.beginPath();
  ctx.moveTo(fx, fy);
  ctx.lineTo(tx, ty);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = "#1d4ed8";
  ctx.beginPath();
  ctx.arc(fx, fy, 5, 0, 2 * Math.PI);
  ctx.fill();
  drawArrow(ctx, fx, fy, yawRad, 24, "#1d4ed8", 3, []);
  drawArrow(ctx, fx, fy, selectedStateYawRad, 34, "#7c3aed", 3, []);
  return count;
}
function drawPanel(canvasId, values, cmap, selectedYawDeg) {
  const canvas = document.getElementById(canvasId);
  const ctx = canvas.getContext("2d");
  const n = payload.gridSize;
  const image = ctx.createImageData(n, n);
  for (let row = 0; row < n; row++) {
    for (let col = 0; col < n; col++) {
      const src = (n - 1 - row) * n + col;
      const dst = (row * n + col) * 4;
      const [r, g, b] = colorMap(values[src], cmap);
      image.data[dst] = r;
      image.data[dst + 1] = g;
      image.data[dst + 2] = b;
      image.data[dst + 3] = 255;
    }
  }
  const offscreen = document.createElement("canvas");
  offscreen.width = n;
  offscreen.height = n;
  offscreen.getContext("2d").putImageData(image, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(offscreen, 0, 0, canvas.width, canvas.height);
  return drawOverlays(ctx, canvas, selectedYawDeg);
}
function redraw() {
  const frame = payload.frames[Number(slider.value)];
  yawValue.textContent = `${frame.yaw_deg.toFixed(1)} deg`;
  drawPanel("rawPanel", frame.raw, "viridis", frame.yaw_deg);
  drawPanel("ripePanel", frame.ripe, "viridis", frame.yaw_deg);
  const count = drawPanel("entropyPanel", frame.entropy, "magma", frame.yaw_deg);
  datasetCount.textContent = `${count} dataset samples in yaw slice`;
}
slider.addEventListener("input", redraw);
redraw();
</script>
</body>
</html>
"""
    payload_json = json.dumps(payload, separators=(",", ":"))
    path.write_text(html.replace("__PAYLOAD__", payload_json), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "nmpc_one_tree_diagnostic"))
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--target-x", type=float, default=8.0)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--true-class", choices=["ripe", "raw"], default="ripe")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--start-radius-min", type=float, default=7.0)
    parser.add_argument("--start-radius-max", type=float, default=8.0)
    parser.add_argument("--stop-entropy", type=float, default=0.02)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--weights-cache", default=None)
    parser.add_argument("--start-mode", choices=["opposite-best", "random"], default="opposite-best")
    parser.add_argument("--flip-start-yaw", action="store_true")
    parser.add_argument("--best-view-grid-size", type=int, default=81)
    parser.add_argument("--best-view-yaw-steps", type=int, default=37)
    parser.add_argument("--heatmap-grid-size", type=int, default=121)
    parser.add_argument("--heatmap-radius", type=float, default=6.0)
    parser.add_argument("--interactive-heatmap-grid-size", type=int, default=61)
    parser.add_argument("--interactive-yaw-steps", type=int, default=37)
    parser.add_argument("--dataset-csv", nargs="*", default=None)
    parser.add_argument("--dataset-max-points", type=int, default=8000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.weights_cache is None:
        args.weights_cache = str(output_dir / "{}_casadi_weights.npz".format(Path(args.checkpoint).stem))
    dataset_paths = (
        [Path(path) for path in args.dataset_csv]
        if args.dataset_csv is not None
        else checkpoint_dataset_paths(args.checkpoint)
    )
    data = run_diagnostic(args)
    dataset_points = load_dataset_points(
        dataset_paths,
        data["target"],
        max_points=args.dataset_max_points,
    )
    figure_path = output_dir / "nmpc_one_tree_diagnostic.png"
    heatmap_path = output_dir / "nmpc_one_tree_mlp_heatmap.png"
    interactive_heatmap_path = output_dir / "nmpc_one_tree_mlp_heatmap_interactive.html"
    csv_path = output_dir / "nmpc_one_tree_diagnostic.csv"
    plot_diagnostic(figure_path, data)
    plot_mlp_heatmap(
        heatmap_path,
        data,
        grid_size=args.heatmap_grid_size,
        radius=args.heatmap_radius,
        dataset_points=dataset_points,
    )
    write_interactive_heatmap(
        interactive_heatmap_path,
        build_interactive_heatmap_payload(
            data,
            dataset_points,
            grid_size=args.interactive_heatmap_grid_size,
            radius=args.heatmap_radius,
            yaw_steps=args.interactive_yaw_steps,
        ),
    )
    write_csv(csv_path, data)
    print("Saved diagnostic plot to {}".format(figure_path))
    print("Saved MLP heatmap to {}".format(heatmap_path))
    print("Saved interactive MLP heatmap to {}".format(interactive_heatmap_path))
    print("Saved diagnostic data to {}".format(csv_path))
    if dataset_paths:
        print("Dataset paths: {}".format([str(path) for path in dataset_paths]))
    print("Dataset points plotted: {}".format(len(dataset_points)))
    first_command = data["commands"][0] if len(data["commands"]) else np.zeros(3)
    command_degrees = first_command.copy()
    command_degrees[2] = np.rad2deg(command_degrees[2])
    print("Seeded initial state [x, y, yaw_deg]: {}".format(
        [
            data["initial_state"][0],
            data["initial_state"][1],
            np.rad2deg(data["initial_state"][2]),
        ]
    ))
    print("First control command [ax, ay, ayaw_deg_s2]: {}".format(command_degrees))
    print("Iterations: {}".format(args.iterations))
    print("Start mode: {}".format(data["start_mode"]))
    print("Flip start yaw: {}".format(data["flip_start_yaw"]))
    best = data["best_viewpoint"]
    print(
        "MLP best POV [x, y, state_yaw_deg, mlp_relative_yaw_deg, expected_entropy, raw_correct, ripe_correct]: {}".format(
            [
                float(best["world_xy"][0]),
                float(best["world_xy"][1]),
                float(np.rad2deg(best["absolute_yaw"])),
                float(np.rad2deg(best["relative_yaw"])),
                float(best["expected_entropy"]),
                float(best["raw_correct"]),
                float(best["ripe_correct"]),
            ]
        )
    )
    print("MLP checkpoint: {}".format(data["checkpoint"]))
    print("MLP weights cache: {}".format(data["weights_cache"]))
    print("Total solve time: {:.3f}s".format(data["total_solve_time"]))
    print("Minimum distance to tree: {:.3f}m".format(float(np.min(data["distances"]))))
    print("Safe distance: {:.3f}m".format(float(data["params"]["safe_distance"])))
    active_solve_times = data["solve_times"][data["solve_times"] > 0.0]
    warm_solve_times = active_solve_times[1:] if len(active_solve_times) > 1 else np.asarray([])
    if len(active_solve_times):
        print("Cold solve time: {:.3f}s".format(float(active_solve_times[0])))
        print("Mean active solve time: {:.3f}s".format(float(np.mean(active_solve_times))))
    if len(warm_solve_times):
        print("Mean warm solve time: {:.3f}s".format(float(np.mean(warm_solve_times))))
    print("Final entropy reduction: {:.4f} bits".format(data["entropy_reduction"][-1]))


if __name__ == "__main__":
    main()
