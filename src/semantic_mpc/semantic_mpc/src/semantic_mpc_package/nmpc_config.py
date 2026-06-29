import os

import numpy as np
import rospy


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def default_nmpc_params():
    return {
        "hidden_size": 64,
        "hidden_layers": 3,
        "nn_input_dim": 3,
        "nn_output_dim": 2,
        "nn_threshold": 8.0,
        "nn_gate_slope": 10.0,
        "model_device": "cuda",
        "model_labels": ["ripe", "raw"],
        "mpc_horizon": 5,
        "dt": 0.2,
        "state_dim": 3,
        "control_dim": 3,
        "optimizer_state_dim": 6,
        "num_target_trees": 10,
        "num_obstacle_trees": 2,
        "sim_steps": 1200,
        "belief_update_period": 2,
        "entropy_stop_threshold": 0.025,
        "entropy_selection_threshold": 0.025,
        "initial_pose_margin": 1.5,
        "initial_pose_sleep": 1.5,
        "wait_sleep": 0.05,
        "loop_extra_sleep": 0.1,
        "field_margin": 3.0,
        "max_heading_abs": 3.0 * np.pi,
        "max_velocity": 1.75,
        "max_yaw_velocity": np.pi / 4.0,
        "max_accel_xy": 5.0,
        "max_accel_yaw": np.pi,
        "safe_distance": 1.0,
        "q_dist": 1e-3,
        "r_xy": 1e-2,
        "r_theta": 1e-2,
        "entropy_weight": 40.0,
        "entropy_objective_scale": 10.0,
        "attraction_threshold_sq_dist": 9.0,
        "attraction_sigmoid_steepness": 10.0,
        "ipopt": {
            "tol": 1e-6,
            "warm_start_init_point": "yes",
            "print_level": 0,
            "sb": "no",
            "warm_start_bound_push": 1e-8,
            "warm_start_mult_bound_push": 1e-8,
            "mu_init": 1e-5,
            "bound_relax_factor": 1e-9,
            "hessian_approximation": "limited-memory",
            "mu_strategy": "monotone",
            "max_iter": 1000,
        },
        "map_frame": "map",
        "base_frame": "drone_base_link",
        "tree_scores_topic": "tree_scores",
        "cmd_pose_topic": "cmd/pose",
        "cmd_pose_queue_size": 1,
        "tree_scores_queue_size": 1,
        "predicted_path_topic": "predicted_path",
        "tree_markers_topic": "tree_markers",
        "publish_predicted_path": True,
        "publish_tree_markers": True,
        "visualization_publish_period": 5,
        "tree_service": "/obj_pose_srv",
        "run_dir": "/runs/nmpc_runs",
        "wandb_project": "semantic_mpc",
        "wandb_entity": "",
        "wandb_mode": "offline",
        "wandb_log_period": 1.0,
    }


def load_semantic_mpc_params():
    names = [
        "hidden_size",
        "hidden_layers",
        "nn_input_dim",
        "nn_output_dim",
        "nn_threshold",
        "nn_gate_slope",
        "model_device",
        "model_labels",
        "mpc_horizon",
        "dt",
        "state_dim",
        "control_dim",
        "optimizer_state_dim",
        "num_target_trees",
        "num_obstacle_trees",
        "sim_steps",
        "belief_update_period",
        "entropy_stop_threshold",
        "entropy_selection_threshold",
        "initial_pose_margin",
        "initial_pose_sleep",
        "wait_sleep",
        "loop_extra_sleep",
        "field_margin",
        "max_heading_abs",
        "max_velocity",
        "max_yaw_velocity",
        "max_accel_xy",
        "max_accel_yaw",
        "safe_distance",
        "q_dist",
        "r_xy",
        "r_theta",
        "entropy_weight",
        "entropy_objective_scale",
        "attraction_threshold_sq_dist",
        "attraction_sigmoid_steepness",
        "ipopt",
        "map_frame",
        "base_frame",
        "tree_scores_topic",
        "cmd_pose_topic",
        "cmd_pose_queue_size",
        "tree_scores_queue_size",
        "predicted_path_topic",
        "tree_markers_topic",
        "publish_predicted_path",
        "publish_tree_markers",
        "visualization_publish_period",
        "tree_service",
        "run_dir",
        "wandb_project",
        "wandb_entity",
        "wandb_mode",
        "wandb_log_period",
    ]
    loaded = {}
    for name in names:
        if rospy.has_param("~" + name):
            loaded[name] = rospy.get_param("~" + name)

    int_names = {
        "hidden_size",
        "hidden_layers",
        "nn_input_dim",
        "nn_output_dim",
        "mpc_horizon",
        "state_dim",
        "control_dim",
        "optimizer_state_dim",
        "num_target_trees",
        "num_obstacle_trees",
        "sim_steps",
        "belief_update_period",
        "cmd_pose_queue_size",
        "tree_scores_queue_size",
        "visualization_publish_period",
    }
    bool_names = {
        "publish_predicted_path",
        "publish_tree_markers",
    }
    float_names = {
        "nn_threshold",
        "nn_gate_slope",
        "dt",
        "entropy_stop_threshold",
        "entropy_selection_threshold",
        "initial_pose_margin",
        "initial_pose_sleep",
        "wait_sleep",
        "loop_extra_sleep",
        "field_margin",
        "max_heading_abs",
        "max_velocity",
        "max_yaw_velocity",
        "max_accel_xy",
        "max_accel_yaw",
        "safe_distance",
        "q_dist",
        "r_xy",
        "r_theta",
        "entropy_weight",
        "entropy_objective_scale",
        "attraction_threshold_sq_dist",
        "attraction_sigmoid_steepness",
        "wandb_log_period",
    }
    for name in int_names.intersection(loaded):
        loaded[name] = int(loaded[name])
    for name in float_names.intersection(loaded):
        loaded[name] = float(loaded[name])
    for name in bool_names.intersection(loaded):
        loaded[name] = _as_bool(loaded[name])
    return loaded
