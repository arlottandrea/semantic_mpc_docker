#!/usr/bin/env python3
"""Run the CasADi NMPC controller against the Gym environment plant."""

import argparse
import json
import math
import time

import casadi as ca
import l4casadi as l4c
import numpy as np
from scipy.stats import entropy
import torch

from active_rl_classification.env import TreeClassificationEnv
from semantic_mpc_package.nmpc_optimizer import NmpcOptimizer
from semantic_mpc_package.perception_model import MultiLayerPerceptron


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


def load_surrogate(checkpoint):
    model = MultiLayerPerceptron(
        input_dim=3,
        hidden_size=64,
        hidden_layers=3,
        output_dim=3,
        threshold=5.0,
        gate_slope=10.0,
    )
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()
    return l4c.L4CasADi(model, batched=True, device="cpu", name="gym_nmpc_perception")


def run_episode(args, seed, surrogate):
    env = TreeClassificationEnv(
        {
            "ntargets": 5,
            "horizon": args.steps,
            "layout": "grid",
            "grid_n_rows": 1,
            "grid_n_cols": 5,
            "grid_col_spacing": 5.0,
            "grid_jitter_std": 0.0,
            "perception_csvs": [args.raw_csv, args.ripe_csv],
            "use_oracle": True,
        }
    )
    env.reset(seed=seed)
    params = {
            "state_dim": 3,
            "control_dim": 3,
            "optimizer_state_dim": 6,
            "dt": 0.25,
            "model_device": "cpu",
            "mpc_horizon": args.mpc_horizon,
            "num_target_trees": 5,
            "num_obstacle_trees": 5,
            "safe_distance": 1.5,
            "observation_range": 5.0,
            "movement_weight": 0.01,
            "yaw_movement_weight": 0.01,
            "acceleration_regularization_weight": 0.0001,
            "information_gain_weight": 1.0,
            "information_discount": 0.99,
            "exploration_weight": 0.25,
            "exploration_sigma": 5.0,
            "attraction_weight": 0.1,
            "field_margin": 3.0,
            "max_heading_abs": 3.0 * np.pi,
            "max_velocity": 1.75,
            "max_yaw_velocity": np.pi / 4.0,
            "max_accel_xy": 1.0,
            "max_accel_yaw": np.pi / 2.0,
            "ipopt": {
                "print_level": 0,
                "sb": "yes",
                "acceptable_tol": 3e-2,
                "acceptable_iter": 5,
                "max_iter": 100,
            },
        }
    optimizer = NmpcOptimizer(params, surrogate)
    initial = total_entropy(env)
    initial_pose = env.drone.copy()
    total_distance = 0.0
    closest_tree_distance = float("inf")
    velocity = np.zeros(3)
    mpc_step = decision = multipliers = None
    solve_times = []
    terminated = truncated = False
    while not (terminated or truncated):
        pose = np.asarray([env.drone[0], env.drone[1], math.radians(env.drone[2])])
        state = np.concatenate([pose, velocity])
        target_indices, target_mask = optimizer.select_nearest_untracked(
            env.trees, env.beliefs, pose[:2], 5, 0.95
        )
        obstacle_distances = np.linalg.norm(np.asarray(env.trees) - pose[:2], axis=1)
        obstacle_indices = np.argsort(obstacle_distances)[:5]
        target_trees = np.asarray(env.trees)[target_indices]
        target_beliefs = env.beliefs[target_indices]
        obstacle_trees = np.asarray(env.trees)[obstacle_indices]
        parameter = ca.vertcat(
            ca.DM(state),
            ca.reshape(ca.DM(target_trees), 10, 1),
            ca.reshape(ca.DM(target_beliefs), 10, 1),
            ca.reshape(ca.DM(target_mask), 5, 1),
            ca.reshape(ca.DM(obstacle_trees), 10, 1),
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

    final = total_entropy(env)
    result = {
        "seed": seed,
        "initial_entropy": initial,
        "final_entropy": final,
        "entropy_reduction": initial - final,
        "tracked": int(np.sum(env.tracked)),
        "steps": env.steps,
        "success": bool(terminated),
        "initial_pose": initial_pose.tolist(),
        "final_pose": env.drone.tolist(),
        "distance_travelled": total_distance,
        "closest_tree_distance": closest_tree_distance,
        "mean_solve_time_s": float(np.mean(solve_times)),
    }
    env.close()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--ripe-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--mpc-horizon", type=int, default=3)
    args = parser.parse_args()
    surrogate = load_surrogate(args.checkpoint)
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
