#!/usr/bin/env python
import math
import os
import time
import threading

import numpy as np
import rospy

from semantic_mpc_package.baselines import (
    CasadiMpcStepGenerator,
    generate_greedy_order,
    generate_linear_order,
    generate_mower_path,
)
from semantic_mpc_package.experiment_metrics import WandbMetrics
from semantic_mpc_package.ros_experiment import (
    BeliefState,
    RosExperimentContext,
    corner_initial_pose,
    normalize_angle,
)


class BaselineExperiment:
    def __init__(self, mode, run_index, params):
        self.mode = mode
        self.run_index = run_index
        self.params = params
        self.rng = np.random.default_rng(params["seed"] + run_index)

        self.ros = RosExperimentContext(params)
        self.lambda_values = np.array([])
        self.tree_positions = np.empty((0, 2))
        self.trees_gt_id = np.array([])
        self.active_tree_idx = None
        self.observing_tree = threading.Event()
        self.measurement_timer = None

        self.prev_cmd = np.zeros(3)
        self.dt = params["dt"]
        self.rate = rospy.Rate(max(1, int(round(params["hz"]))))
        self.metrics = WandbMetrics(mode, run_index, params)
        self.step_generator = None
        if params["step_generator"] == "casadi_mpc":
            self.step_generator = CasadiMpcStepGenerator(params)
        elif params["step_generator"] == "nav2_ros1_bridge":
            raise RuntimeError(
                "step_generator=nav2_ros1_bridge requires an external ROS2 Nav2 stack and ROS1 bridge node. "
                "Use step_generator=casadi_mpc for the built-in ROS1 baseline planner."
            )

    def setup(self):
        self.tree_positions, self.trees_gt_id = self.ros.get_trees_poses_and_types()
        if self.tree_positions.size == 0:
            raise RuntimeError("No tree positions returned by {}".format(self.params["tree_service"]))

        self.lambda_values = np.full(len(self.tree_positions), 0.5)
        initial_pose = self.resolve_initial_pose()
        if initial_pose is not None:
            self.ros.publish_pose(initial_pose)
            rospy.sleep(2.5)

        self.metrics.start()
        self.initial_entropy = BeliefState.binary_entropy(self.lambda_values)
        self.measurement_timer = rospy.Timer(
            rospy.Duration(self.params["measurement_period"]),
            self.measurement_callback,
        )

    def resolve_initial_pose(self):
        start_pose = self.params["start_pose"]
        if start_pose is not None:
            return np.array(start_pose, dtype=float)
        if not self.params["random_initial_state"]:
            return None

        return corner_initial_pose(self.tree_positions)

    def measurement_callback(self, _event):
        scores = self.ros.get_tree_scores(column=0)
        if scores is None:
            return
        if self.mode != "mower" and not self.observing_tree.is_set():
            return

        if self.mode == "mower":
            for idx in range(min(len(scores), len(self.lambda_values))):
                self.lambda_values[idx] = BeliefState.bayes_binary(self.lambda_values[idx], scores[idx])
        elif self.active_tree_idx is not None and self.active_tree_idx < len(scores):
            idx = self.active_tree_idx
            self.lambda_values[idx] = BeliefState.bayes_binary(self.lambda_values[idx], scores[idx])
            if BeliefState.binary_entropy(np.array([self.lambda_values[idx]])) <= self.params["observation_entropy_threshold"]:
                self.observing_tree.clear()

    def calculate_motion_vector(self, current_position, goal_position):
        attractive = goal_position - current_position
        distance = np.linalg.norm(attractive)
        if distance < 1e-6 or self.mode == "mower":
            return attractive

        attractive_unit = attractive / max(distance, 1e-6)
        deviation = np.zeros(2)
        unsafe = False

        for tree in self.tree_positions:
            if np.linalg.norm(tree - goal_position) < self.params["obstacle_goal_ignore_distance"]:
                continue
            diff = current_position - tree
            tree_distance = np.linalg.norm(diff)
            if tree_distance >= self.params["obstacle_avoidance_distance"] or tree_distance < 1e-6:
                continue
            angle = np.arccos(np.clip(np.dot(attractive_unit, -diff / tree_distance), -1.0, 1.0))
            if angle < self.params["obstacle_cone_angle"]:
                unsafe = True
                radial = diff / tree_distance
                tangent_a = np.array([-radial[1], radial[0]])
                tangent_b = np.array([radial[1], -radial[0]])
                tangent = tangent_a if np.dot(tangent_a, attractive_unit) > np.dot(tangent_b, attractive_unit) else tangent_b
                deviation += self.params["obstacle_tangent_gain"] * tangent - self.params["obstacle_radial_gain"] * radial

        if not unsafe:
            return attractive

        combined = attractive_unit + self.params["obstacle_deviation_gain"] * deviation
        norm = np.linalg.norm(combined)
        return attractive if norm < 1e-6 else combined / norm

    def command_from_goal(self, current_pose, goal_xy, desired_heading=None, desired_velocity=None):
        goal_xy = np.asarray(goal_xy, dtype=float)
        if desired_heading is None:
            desired_heading = math.atan2(goal_xy[1] - current_pose[1], goal_xy[0] - current_pose[0])

        if self.step_generator is not None:
            cmd = self.step_generator.command(
                current_pose,
                goal_xy,
                desired_heading,
                self.tree_positions,
                self.prev_cmd,
                desired_velocity=desired_velocity,
            )
            self.prev_cmd = cmd
            return cmd

        motion = self.calculate_motion_vector(current_pose[:2], goal_xy)
        distance = np.linalg.norm(motion)
        if distance > 1e-6:
            speed = min(
                self.params["max_velocity"],
                max(self.params["min_velocity"], distance / max(self.dt, 1e-6)),
            )
            linear = motion / distance * speed
        else:
            linear = np.zeros(2)

        yaw_error = normalize_angle(desired_heading - current_pose[2])
        yaw_rate = np.clip(yaw_error / max(self.dt, 1e-6), -self.params["max_yaw_velocity"], self.params["max_yaw_velocity"])

        if desired_velocity is not None and np.linalg.norm(goal_xy - current_pose[:2]) <= max(
            self.params["tolerance"],
            self.params["max_velocity"] * self.dt,
        ):
            linear = np.asarray(desired_velocity[:2], dtype=float)

        raw_cmd = np.array([linear[0], linear[1], yaw_rate], dtype=float)
        smoothed = self.params["cmd_smoothing_alpha"] * raw_cmd + (1.0 - self.params["cmd_smoothing_alpha"]) * self.prev_cmd

        max_delta = np.array(
            [
                self.params["max_lin_accel"] * self.dt,
                self.params["max_lin_accel"] * self.dt,
                self.params["max_yaw_accel"] * self.dt,
            ]
        )
        cmd = self.prev_cmd + np.clip(smoothed - self.prev_cmd, -max_delta, max_delta)
        self.prev_cmd = cmd
        return cmd

    def move_to_waypoint(self, target_x, target_y, tolerance=None, desired_heading=None, desired_velocity=None, stop_at_target=True):
        tolerance = self.params["tolerance"] if tolerance is None else tolerance
        target = np.array([target_x, target_y], dtype=float)

        while not rospy.is_shutdown():
            current_pose = self.ros.robot_pose(default=np.zeros(3))
            entropy = BeliefState.binary_entropy(self.lambda_values)
            self.metrics.log_pose(current_pose, entropy)

            if np.linalg.norm(current_pose[:2] - target) < tolerance:
                if stop_at_target:
                    self.prev_cmd = np.zeros(3)
                return

            cmd = self.command_from_goal(current_pose, target, desired_heading, desired_velocity)
            next_pose = np.array(
                [
                    current_pose[0] + cmd[0] * self.dt,
                    current_pose[1] + cmd[1] * self.dt,
                    current_pose[2] + cmd[2] * self.dt,
                ]
            )
            self.ros.publish_pose(next_pose)
            self.metrics.add_distance(np.linalg.norm(cmd[:2] * self.dt))
            self.rate.sleep()

    @staticmethod
    def _is_passthrough_waypoint(previous_waypoint, waypoint, next_waypoint):
        if previous_waypoint is None or next_waypoint is None:
            return False

        previous_xy = np.asarray(previous_waypoint[:2], dtype=float)
        waypoint_xy = np.asarray(waypoint[:2], dtype=float)
        next_xy = np.asarray(next_waypoint[:2], dtype=float)
        incoming = waypoint_xy - previous_xy
        outgoing = next_xy - waypoint_xy
        incoming_norm = np.linalg.norm(incoming)
        outgoing_norm = np.linalg.norm(outgoing)
        if incoming_norm < 1e-6 or outgoing_norm < 1e-6:
            return False

        incoming_unit = incoming / incoming_norm
        outgoing_unit = outgoing / outgoing_norm
        return float(np.dot(incoming_unit, outgoing_unit)) > 0.999

    def observe_tree(self):
        if self.active_tree_idx is None:
            return

        center_x, center_y = self.tree_positions[self.active_tree_idx]
        pose = self.ros.robot_pose(default=np.zeros(3))
        radius = max(np.linalg.norm(pose[:2] - np.array([center_x, center_y])), 0.5)
        phi = math.atan2(pose[1] - center_y, pose[0] - center_x)
        delta_phi = self.dt * 15.0 * math.pi / 180.0
        start = time.time()

        self.observing_tree.set()
        while self.observing_tree.is_set() and not rospy.is_shutdown():
            if time.time() - start > self.params["max_observe_time"]:
                self.observing_tree.clear()
                break

            desired = np.array([center_x + radius * math.cos(phi), center_y + radius * math.sin(phi)])
            desired_heading = normalize_angle(phi + math.pi)
            current_pose = self.ros.robot_pose(default=np.zeros(3))
            entropy = BeliefState.binary_entropy(self.lambda_values)
            self.metrics.log_pose(current_pose, entropy)

            cmd = self.command_from_goal(current_pose, desired, desired_heading)
            next_pose = np.array(
                [
                    current_pose[0] + cmd[0] * self.dt,
                    current_pose[1] + cmd[1] * self.dt,
                    current_pose[2] + cmd[2] * self.dt,
                ]
            )
            self.ros.publish_pose(next_pose)
            self.metrics.add_distance(np.linalg.norm(cmd[:2] * self.dt))
            phi += delta_phi
            self.rate.sleep()

        self.prev_cmd = np.zeros(3)

    def run_mower(self):
        current_pose = self.ros.robot_pose(default=np.zeros(3))
        path = generate_mower_path(
            self.tree_positions,
            current_pose,
            offset=self.params["mower_offset"],
            spacing=self.params["mower_spacing"],
            heading_direction=self.params["mower_heading"],
            axis=self.params["mower_axis"] or None,
            seed=self.params["seed"] + self.run_index,
        )
        rospy.loginfo("Mower baseline generated %d waypoints.", len(path))
        if not path:
            return

        self.ros.publish_pose(np.array(path[0], dtype=float))
        rospy.sleep(1.0)
        cruise_velocity = min(self.params["mower_cruise_velocity"], self.params["max_velocity"])
        for idx, waypoint in enumerate(path[1:], start=1):
            next_waypoint = path[idx + 1] if idx + 1 < len(path) else None
            pass_through = self._is_passthrough_waypoint(path[idx - 1], waypoint, next_waypoint)
            desired_velocity = None
            if pass_through:
                direction = np.asarray(next_waypoint[:2], dtype=float) - np.asarray(waypoint[:2], dtype=float)
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 1e-6:
                    desired_velocity = direction / direction_norm * cruise_velocity

            self.move_to_waypoint(
                waypoint[0],
                waypoint[1],
                desired_heading=waypoint[2],
                desired_velocity=desired_velocity,
                stop_at_target=not pass_through,
            )

    def run_linear(self):
        order = generate_linear_order(
            self.tree_positions,
            self.ros.robot_pose(default=np.zeros(3)),
            same_row_tol=self.params["linear_same_row_tol"],
        )
        rospy.loginfo("Linear baseline generated %d tree visits.", len(order))
        self.visit_trees(order)

    def run_greedy(self):
        order = generate_greedy_order(self.tree_positions, self.ros.robot_pose(default=np.zeros(3)), self.lambda_values)
        rospy.loginfo("Greedy baseline generated %d tree visits.", len(order))
        self.visit_trees(order)

    def run_casadi_mpc(self):
        order = generate_greedy_order(self.tree_positions, self.ros.robot_pose(default=np.zeros(3)), self.lambda_values)
        rospy.loginfo("CasADi MPC baseline generated %d tree visits.", len(order))
        self.visit_trees(order)

    def visit_trees(self, order):
        for idx in order:
            if rospy.is_shutdown():
                return
            self.active_tree_idx = idx
            tree_pos = self.tree_positions[idx]
            self.move_to_waypoint(
                tree_pos[0],
                tree_pos[1],
                tolerance=self.params["tree_observation_tolerance"],
            )
            self.observe_tree()

    def run(self):
        self.setup()
        if self.mode == "mower":
            self.run_mower()
        elif self.mode == "linear":
            self.run_linear()
        elif self.mode == "greedy":
            self.run_greedy()
        elif self.mode == "casadi_mpc":
            if self.params["step_generator"] != "casadi_mpc":
                rospy.logwarn("mode=casadi_mpc overrides step_generator=%s.", self.params["step_generator"])
                self.step_generator = CasadiMpcStepGenerator(self.params)
            self.run_casadi_mpc()
        else:
            raise ValueError("Unsupported baseline mode: {}".format(self.mode))

        final_entropy = BeliefState.binary_entropy(self.lambda_values)
        summary = self.metrics.finish(self.initial_entropy, final_entropy, self.lambda_values)
        rospy.loginfo(
            "%s run %d complete: time=%.2fs distance=%.2fm entropy_reduction=%.4f",
            self.mode,
            self.run_index,
            summary["total_time_execution_s"],
            summary["total_distance_m"],
            summary["entropy_reduction"],
        )

    def shutdown(self):
        if self.measurement_timer is not None:
            self.measurement_timer.shutdown()


def _param(name, default):
    value = rospy.get_param("~" + name, None)
    if value is not None:
        return value
    return rospy.get_param("~baseline_node/" + name, default)


def load_params():
    hz = float(_param("hz", 10.0))
    dt = _param("dt", None)
    dt = (1.0 / hz) if dt is None else float(dt)
    if hz <= 0.0:
        hz = 1.0 / max(dt, 1e-6)

    params = {
        "modes": _param("modes", ["casadi_mpc"]),
        "step_generator": _param("step_generator", "casadi_mpc"),
        "num_runs": int(_param("num_runs", 1)),
        "run_root": _param("run_root", "/runs/baseline_runs"),
        "seed": int(_param("seed", 1)),
        "random_initial_state": bool(_param("random_initial_state", True)),
        "start_pose": _param("start_pose", None),
        "map_frame": _param("map_frame", "map"),
        "base_frame": _param("base_frame", "drone_base_link"),
        "tree_scores_topic": _param("tree_scores_topic", "tree_scores"),
        "cmd_pose_topic": _param("cmd_pose_topic", "cmd/pose"),
        "cmd_pose_queue_size": int(_param("cmd_pose_queue_size", 1)),
        "tree_scores_queue_size": int(_param("tree_scores_queue_size", 1)),
        "tree_service": _param("tree_service", "/obj_pose_srv"),
        "hz": hz,
        "dt": dt,
        "tolerance": float(_param("tolerance", 0.2)),
        "tree_observation_tolerance": float(_param("tree_observation_tolerance", 2.0)),
        "min_velocity": float(_param("min_velocity", 0.0)),
        "max_velocity": float(_param("max_velocity", 1.5)),
        "max_yaw_velocity": float(_param("max_yaw_velocity", np.pi / 4.0)),
        "max_lin_accel": float(_param("max_lin_accel", 0.75)),
        "max_yaw_accel": float(_param("max_yaw_accel", np.pi / 2.0)),
        "cmd_smoothing_alpha": float(_param("cmd_smoothing_alpha", 0.2)),
        "obstacle_avoidance_distance": float(_param("obstacle_avoidance_distance", 3.5)),
        "obstacle_goal_ignore_distance": float(_param("obstacle_goal_ignore_distance", 2.0)),
        "obstacle_cone_angle": float(_param("obstacle_cone_angle", np.pi / 3.0)),
        "obstacle_tangent_gain": float(_param("obstacle_tangent_gain", 3.25)),
        "obstacle_radial_gain": float(_param("obstacle_radial_gain", 1.75)),
        "obstacle_deviation_gain": float(_param("obstacle_deviation_gain", 3.5)),
        "observation_entropy_threshold": float(_param("observation_entropy_threshold", 0.025)),
        "max_observe_time": float(_param("max_observe_time", 30.0)),
        "measurement_period": float(_param("measurement_period", 0.25)),
        "mower_offset": float(_param("mower_offset", 2.0)),
        "mower_spacing": float(_param("mower_spacing", 4.0)),
        "mower_heading": _param("mower_heading", "N"),
        "mower_axis": _param("mower_axis", ""),
        "mower_cruise_velocity": float(_param("mower_cruise_velocity", _param("max_velocity", 1.5))),
        "linear_same_row_tol": float(_param("linear_same_row_tol", 0.01)),
        "mpc_steps": int(_param("mpc_steps", 8)),
        "mpc_max_obstacles": int(_param("mpc_max_obstacles", 8)),
        "mpc_goal_weight": float(_param("mpc_goal_weight", 8.0)),
        "mpc_heading_weight": float(_param("mpc_heading_weight", 0.2)),
        "mpc_control_weight": float(_param("mpc_control_weight", 0.05)),
        "mpc_smooth_weight": float(_param("mpc_smooth_weight", 0.2)),
        "mpc_obstacle_weight": float(_param("mpc_obstacle_weight", 0.25)),
        "mpc_velocity_weight": float(_param("mpc_velocity_weight", 2.0)),
        "mpc_ipopt_max_iter": int(_param("mpc_ipopt_max_iter", 80)),
        "wandb_project": _param("wandb_project", "semantic_mpc_baselines"),
        "wandb_entity": _param("wandb_entity", ""),
        "wandb_mode": _param("wandb_mode", "offline"),
        "wandb_log_period": float(_param("wandb_log_period", 1.0)),
    }

    if isinstance(params["modes"], str):
        params["modes"] = [mode.strip().lower() for mode in params["modes"].split(",") if mode.strip()]
    else:
        params["modes"] = [str(mode).strip().lower() for mode in params["modes"]]
    params["step_generator"] = str(params["step_generator"]).strip().lower()
    if params["start_pose"] in ("", [], None):
        params["start_pose"] = None
    return params


def main():
    rospy.init_node("baseline_node", anonymous=False, log_level=rospy.INFO)
    params = load_params()
    os.makedirs(params["run_root"], exist_ok=True)

    for run_index in range(params["num_runs"]):
        for mode in params["modes"]:
            if rospy.is_shutdown():
                return
            run_params = dict(params)
            run_params["run_dir"] = os.path.join(
                params["run_root"],
                "{}_run_{:03d}".format(mode, run_index),
            )
            os.makedirs(run_params["run_dir"], exist_ok=True)
            experiment = BaselineExperiment(mode, run_index, run_params)
            try:
                experiment.run()
            finally:
                experiment.shutdown()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
