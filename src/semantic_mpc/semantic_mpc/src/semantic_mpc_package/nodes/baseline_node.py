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
    resolve_mower_heading,
    select_greedy_ig_target,
)
from semantic_mpc_package.experiment_metrics import WandbMetrics
from semantic_mpc_package.ros_experiment import (
    BeliefState,
    RosExperimentContext,
    normalize_angle,
    seeded_corner_initial_pose,
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
        self.last_score_sequence = 0

        self.prev_cmd = np.zeros(3)
        self.dt = params["dt"]
        self.rate = rospy.Rate(max(1, int(round(params["hz"]))))
        self.metrics = WandbMetrics(mode, run_index, params)
        self.controller_compute_times_ms = []
        self.observation_episode_count = 0
        self.termination_reason = None
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
            self.apply_initial_pose(initial_pose)

        self.metrics.start()
        self.initial_entropy = BeliefState.binary_entropy(self.lambda_values)
        self.metrics.entropy_metrics.update(0.0, self.lambda_values)
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

        heading_direction = None if self.params["mower_heading_random"] else self.params["mower_heading"]
        initial_heading = resolve_mower_heading(heading_direction, seed=self.params["trial_seed"])
        return seeded_corner_initial_pose(
            self.tree_positions,
            self.params["trial_seed"],
            margin=self.params["initial_corner_margin"],
            heading=initial_heading,
        )

    def apply_initial_pose(self, initial_pose):
        """Publish and verify the requested simulation reset pose."""
        initial_pose = np.asarray(initial_pose, dtype=float).flatten()[:3]
        timeout = max(0.1, self.params["initial_pose_timeout"])
        position_tolerance = self.params["initial_pose_tolerance"]
        heading_tolerance = self.params["initial_heading_tolerance"]
        publish_period = max(0.02, self.params["initial_pose_publish_period"])
        deadline = time.time() + timeout

        rospy.loginfo(
            "Resetting robot to initial pose [%.3f, %.3f, %.3f].",
            initial_pose[0],
            initial_pose[1],
            initial_pose[2],
        )
        while not rospy.is_shutdown() and time.time() < deadline:
            self.ros.publish_pose(initial_pose)
            current_pose = self.ros.robot_pose(default=None, timeout=min(0.2, publish_period))
            if current_pose is not None:
                position_error = np.linalg.norm(current_pose[:2] - initial_pose[:2])
                heading_error = abs(normalize_angle(current_pose[2] - initial_pose[2]))
                if position_error <= position_tolerance and heading_error <= heading_tolerance:
                    self.prev_cmd = np.zeros(3)
                    rospy.loginfo(
                        "Initial pose confirmed (position error %.3fm, heading error %.3frad).",
                        position_error,
                        heading_error,
                    )
                    return
            rospy.sleep(publish_period)

        raise RuntimeError(
            "Robot did not reach the configured start_pose within {:.1f}s; "
            "check the Unity /agent_0/cmd/pose subscriber and map -> drone_base_link TF.".format(timeout)
        )

    def measurement_callback(self, _event):
        if self.mode != "mower" and not self.observing_tree.is_set():
            return
        score_matrix, sequence = self.ros.get_new_tree_scores(self.last_score_sequence)
        if score_matrix is None:
            return
        self.last_score_sequence = sequence
        scores = score_matrix[:, 0]

        if self.mode == "mower":
            for idx in range(min(len(scores), len(self.lambda_values))):
                self.lambda_values[idx] = BeliefState.bayes_binary(self.lambda_values[idx], scores[idx])
        elif self.active_tree_idx is not None and self.active_tree_idx < len(scores):
            idx = self.active_tree_idx
            self.lambda_values[idx] = BeliefState.bayes_binary(self.lambda_values[idx], scores[idx])
            if BeliefState.binary_entropy(np.array([self.lambda_values[idx]])) <= self.params["tree_entropy_threshold"]:
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
            self.metrics.log_pose(
                current_pose,
                entropy,
                belief=self.lambda_values,
                tree_positions=self.tree_positions,
                controller_compute_time_ms=(self.controller_compute_times_ms[-1] if self.controller_compute_times_ms else np.nan),
                selected_target_id=self.active_tree_idx,
                active_target_count_current=self._active_target_count(),
                observation_episode_active=False,
                command_speed_mps=float(np.linalg.norm(self.prev_cmd[:2])),
            )
            if self.experiment_budget_exhausted():
                self.stop_for_experiment_budget(current_pose)
                return False

            distance_to_target = np.linalg.norm(current_pose[:2] - target)
            if distance_to_target <= tolerance:
                if stop_at_target:
                    self.hold_pose(current_pose)
                return True

            compute_start = time.perf_counter()
            cmd = self.command_from_goal(current_pose, target, desired_heading, desired_velocity)
            if stop_at_target:
                cmd = self.apply_waypoint_braking(cmd, distance_to_target, tolerance)
            self.controller_compute_times_ms.append((time.perf_counter() - compute_start) * 1000.0)
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

        return False

    def experiment_budget_exhausted(self):
        return self.metrics.step_count >= self.params["max_experiment_steps"]

    def stop_for_experiment_budget(self, current_pose=None):
        self.termination_reason = "max_experiment_steps"
        if current_pose is None:
            current_pose = self.ros.robot_pose(default=np.zeros(3))
        self.hold_pose(current_pose)

    def apply_waypoint_braking(self, cmd, distance_to_target, tolerance):
        """Limit approach speed so a command cannot carry the robot past a stop waypoint."""
        cmd = np.asarray(cmd, dtype=float).copy()
        # Aim halfway inside the acceptance radius. Stopping exactly on its
        # boundary can produce sub-millimetre setpoint updates that Unity
        # ignores, leaving the loop just outside tolerance.
        remaining = max(0.0, float(distance_to_target) - 0.5 * float(tolerance))
        speed = np.linalg.norm(cmd[:2])
        if speed <= 1e-9:
            return cmd

        acceleration = max(self.params["max_lin_accel"], 1e-6)
        braking_speed = math.sqrt(2.0 * acceleration * remaining)
        one_step_speed = remaining / max(self.dt, 1e-6)
        allowed_speed = min(self.params["max_velocity"], braking_speed, one_step_speed)
        if speed > allowed_speed:
            cmd[:2] *= allowed_speed / speed
            self.prev_cmd = cmd.copy()
        return cmd

    def hold_pose(self, current_pose):
        """Cancel residual motion and keep publishing the measured pose while Unity settles."""
        self.prev_cmd = np.zeros(3)
        hold_target = np.asarray(current_pose, dtype=float).copy()
        settle_time = max(0.0, self.params["waypoint_settle_time"])
        publish_period = max(0.02, self.params["waypoint_hold_publish_period"])
        deadline = time.time() + settle_time
        while not rospy.is_shutdown() and time.time() < deadline:
            self.ros.publish_pose(hold_target)
            rospy.sleep(publish_period)

    def observe_tree(self):
        if self.active_tree_idx is None:
            return

        center_x, center_y = self.tree_positions[self.active_tree_idx]
        pose = self.ros.robot_pose(default=np.zeros(3))
        radius = max(np.linalg.norm(pose[:2] - np.array([center_x, center_y])), 0.5)
        phi = math.atan2(pose[1] - center_y, pose[0] - center_x)
        delta_phi = self.dt * 15.0 * math.pi / 180.0
        start = time.time()

        self.last_score_sequence = self.ros.tree_scores_sequence
        self.observation_episode_count += 1
        self.observing_tree.set()
        while self.observing_tree.is_set() and not rospy.is_shutdown():
            if self.experiment_budget_exhausted():
                self.observing_tree.clear()
                self.stop_for_experiment_budget()
                break
            if time.time() - start > self.params["max_observe_time"]:
                self.observing_tree.clear()
                break

            desired = np.array([center_x + radius * math.cos(phi), center_y + radius * math.sin(phi)])
            desired_heading = normalize_angle(phi + math.pi)
            current_pose = self.ros.robot_pose(default=np.zeros(3))
            entropy = BeliefState.binary_entropy(self.lambda_values)
            self.metrics.log_pose(
                current_pose,
                entropy,
                belief=self.lambda_values,
                tree_positions=self.tree_positions,
                controller_compute_time_ms=(self.controller_compute_times_ms[-1] if self.controller_compute_times_ms else np.nan),
                selected_target_id=self.active_tree_idx,
                active_target_count_current=self._active_target_count(),
                observation_episode_active=True,
                command_speed_mps=float(np.linalg.norm(self.prev_cmd[:2])),
            )

            compute_start = time.perf_counter()
            cmd = self.command_from_goal(current_pose, desired, desired_heading)
            self.controller_compute_times_ms.append((time.perf_counter() - compute_start) * 1000.0)
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

    def _active_target_count(self):
        if self.mode == "mower":
            return 0
        confidence = np.maximum(self.lambda_values, 1.0 - self.lambda_values)
        untracked = int(np.sum(confidence < self.params["belief_tracking_threshold"]))
        return min(untracked, self.params["active_target_count"])

    def run_mower(self):
        current_pose = self.ros.robot_pose(default=np.zeros(3))
        path = generate_mower_path(
            self.tree_positions,
            current_pose,
            offset=self.params["mower_offset"],
            spacing=self.params["mower_spacing"],
            heading_direction=None if self.params["mower_heading_random"] else self.params["mower_heading"],
            axis=self.params["mower_axis"] or None,
            seed=self.params["seed"] + self.run_index,
        )
        rospy.loginfo("Mower baseline generated %d lane-end waypoints.", len(path))
        if not path:
            return

        # Drive to the first lane end instead of publishing it once and
        # assuming Unity reached it. Every remaining point is a real turn.
        for waypoint in path:
            self.move_to_waypoint(
                waypoint[0],
                waypoint[1],
                desired_heading=waypoint[2],
                stop_at_target=True,
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

    def run_greedy_ig(self):
        """Select and observe one tree at a time, then replan from new belief."""
        visits = 0
        while not rospy.is_shutdown():
            if self.experiment_budget_exhausted():
                self.stop_for_experiment_budget()
                return
            current_pose = self.ros.robot_pose(default=np.zeros(3))
            idx = select_greedy_ig_target(
                self.tree_positions,
                current_pose,
                self.lambda_values,
                self.params["active_target_count"],
                self.params["belief_tracking_threshold"],
                self.params["greedy_ig_observation_accuracy"],
            )
            if idx is None:
                rospy.loginfo(
                    "Greedy-IG finished after %d receding tree selections; all trees are tracked.",
                    visits,
                )
                return

            self.active_tree_idx = idx
            tree_pos = self.tree_positions[idx]
            reached = self.move_to_waypoint(
                tree_pos[0],
                tree_pos[1],
                tolerance=self.params["tree_observation_tolerance"],
            )
            if not reached:
                return
            self.observe_tree()
            if self.termination_reason is not None:
                return
            visits += 1

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
        elif self.mode == "greedy_ig":
            self.run_greedy_ig()
        elif self.mode == "casadi_mpc":
            if self.params["step_generator"] != "casadi_mpc":
                rospy.logwarn("mode=casadi_mpc overrides step_generator=%s.", self.params["step_generator"])
                self.step_generator = CasadiMpcStepGenerator(self.params)
            self.run_casadi_mpc()
        else:
            raise ValueError("Unsupported baseline mode: {}".format(self.mode))

        final_entropy = BeliefState.binary_entropy(self.lambda_values)
        compute_times = np.asarray(self.controller_compute_times_ms, dtype=float)
        compute_summary = {
            "mean_controller_compute_time_ms": float(np.mean(compute_times)) if compute_times.size else np.nan,
            "median_controller_compute_time_ms": float(np.median(compute_times)) if compute_times.size else np.nan,
            "p95_controller_compute_time_ms": float(np.percentile(compute_times, 95)) if compute_times.size else np.nan,
            "total_controller_compute_time_ms": float(np.sum(compute_times)) if compute_times.size else 0.0,
            "observation_episode_count": self.observation_episode_count,
            "entropy_reduction_per_observation_episode": (
                float((self.initial_entropy - final_entropy) / self.observation_episode_count)
                if self.observation_episode_count
                else np.nan
            ),
        }
        confidence = np.maximum(self.lambda_values, 1.0 - self.lambda_values)
        num_tracked = int(np.sum(confidence >= self.params["belief_tracking_threshold"]))
        success = bool(num_tracked == len(self.lambda_values)) if self.mode != "mower" else not rospy.is_shutdown()
        compute_summary.update(
            {
                "success": success,
                "termination_reason": self.termination_reason or (
                    "ros_shutdown" if rospy.is_shutdown() else "all_trees_tracked" if success and self.mode != "mower" else "path_complete" if success else "observation_budget_exhausted"
                ),
                "total_steps": self.metrics.step_count,
                "num_tracked_final": num_tracked,
                "total_targets": len(self.lambda_values),
            }
        )
        summary = self.metrics.finish(
            self.initial_entropy,
            final_entropy,
            self.lambda_values,
            extra=compute_summary,
        )
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
    hz_value = _param("hz", None)
    dt = _param("dt", None)
    if dt is not None:
        dt = float(dt)
        hz = float(hz_value) if hz_value is not None else 1.0 / max(dt, 1e-6)
    else:
        hz = float(hz_value) if hz_value is not None else 10.0
        dt = 1.0 / max(hz, 1e-6)
    if hz <= 0.0:
        hz = 1.0 / max(dt, 1e-6)

    active_obstacle_count = int(_param("active_obstacle_count", 5))
    params = {
        "modes": _param("modes", ["casadi_mpc"]),
        "step_generator": _param("step_generator", "casadi_mpc"),
        "num_runs": int(_param("num_runs", 1)),
        "run_index_offset": int(_param("run_index_offset", 0)),
        "run_root": _param("run_root", "/runs/baseline_runs"),
        "seed": int(_param("seed", 1)),
        "random_initial_state": bool(_param("random_initial_state", True)),
        "start_pose": _param("start_pose", None),
        "initial_corner_margin": float(_param("initial_corner_margin", 1.5)),
        "initial_pose_timeout": float(_param("initial_pose_timeout", 15.0)),
        "initial_pose_tolerance": float(_param("initial_pose_tolerance", 0.15)),
        "initial_heading_tolerance": float(_param("initial_heading_tolerance", 0.2)),
        "initial_pose_publish_period": float(_param("initial_pose_publish_period", 0.1)),
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
        "waypoint_settle_time": float(_param("waypoint_settle_time", 0.4)),
        "waypoint_hold_publish_period": float(_param("waypoint_hold_publish_period", 0.05)),
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
        "tree_entropy_threshold": float(_param("tree_entropy_threshold", 0.025)),
        "max_observe_time": float(_param("max_observe_time", 30.0)),
        "measurement_period": float(_param("measurement_period", 0.25)),
        "belief_tracking_threshold": float(_param("belief_tracking_threshold", 0.9975245006578829)),
        "active_target_count": int(_param("active_target_count", 5)),
        "active_obstacle_count": active_obstacle_count,
        "max_experiment_steps": int(_param("max_experiment_steps", 1200)),
        "greedy_ig_observation_accuracy": float(_param("greedy_ig_observation_accuracy", 0.9)),
        "mower_offset": float(_param("mower_offset", 2.0)),
        "mower_spacing": float(_param("mower_spacing", 4.0)),
        "mower_heading": _param("mower_heading", "N"),
        "mower_heading_random": bool(_param("mower_heading_random", False)),
        "mower_axis": _param("mower_axis", ""),
        "linear_same_row_tol": float(_param("linear_same_row_tol", 0.01)),
        "mpc_steps": int(_param("mpc_steps", 8)),
        "mpc_max_obstacles": active_obstacle_count,
        "mpc_goal_weight": float(_param("mpc_goal_weight", 8.0)),
        "mpc_heading_weight": float(_param("mpc_heading_weight", 0.2)),
        "mpc_control_weight": float(_param("mpc_control_weight", 0.05)),
        "mpc_smooth_weight": float(_param("mpc_smooth_weight", 0.2)),
        "mpc_obstacle_weight": float(_param("mpc_obstacle_weight", 0.25)),
        "mpc_velocity_weight": float(_param("mpc_velocity_weight", 2.0)),
        "mpc_ipopt_max_iter": int(_param("mpc_ipopt_max_iter", 80)),
        "wandb_enabled": bool(_param("wandb_enabled", False)),
        "wandb_project": _param("wandb_project", "semantic_mpc_baselines"),
        "wandb_entity": _param("wandb_entity", ""),
        "wandb_mode": _param("wandb_mode", "offline"),
        "wandb_log_every_steps": int(_param("wandb_log_every_steps", 4)),
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

    first_run_index = params["run_index_offset"]
    for run_index in range(first_run_index, first_run_index + params["num_runs"]):
        for mode in params["modes"]:
            if rospy.is_shutdown():
                return
            run_params = dict(params)
            run_params["trial_seed"] = params["seed"] + run_index
            run_params["algorithm"] = mode
            run_params["run_index"] = run_index
            run_params["termination_criterion"] = (
                "path_complete_or_max_experiment_steps"
                if mode == "mower"
                else "all_trees_belief_confidence_threshold_or_max_experiment_steps"
            )
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
