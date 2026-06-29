#!/usr/bin/env python
import math
import os
import time

import casadi as ca
import numpy as np
import rospy
from nav_msgs.msg import Path
from visualization_msgs.msg import MarkerArray

from semantic_mpc_package.experiment_metrics import WandbMetrics
from semantic_mpc_package.baselines import resolve_mower_heading
from semantic_mpc_package.nmpc_config import default_nmpc_params, load_semantic_mpc_params
from semantic_mpc_package.nmpc_model import load_l4casadi_models
from semantic_mpc_package.nmpc_optimizer import NmpcOptimizer
from semantic_mpc_package.ros_com_lib.sensors import (
    create_path_from_mpc_prediction,
    create_tree_markers,
)
from semantic_mpc_package.ros_experiment import (
    BeliefState,
    RosExperimentContext,
    get_domain,
    normalize_angle,
    seeded_corner_initial_pose,
)


class NeuralMPC:
    def __init__(self, run_dir=None, initial_randomic=False, params=None, run_index=0):
        rospy.init_node("nmpc_node", anonymous=True, log_level=rospy.INFO)

        self.params = default_nmpc_params()
        if params:
            self.params.update(params)
        self.params.update(load_semantic_mpc_params())
        if run_dir is not None:
            self.params["run_dir"] = run_dir

        self.run_index = run_index
        self.initial_randomic = bool(
            self.params.get("random_initial_state", rospy.get_param("~initial_randomic", initial_randomic))
        )
        self._load_runtime_params()
        self._validate_comparability()
        self.run_root = self.params["run_dir"]

        self.l4c_nn = load_l4casadi_models(self.params)
        self.optimizer = NmpcOptimizer(self.params, self.l4c_nn)

        self.ros = RosExperimentContext(self.params)
        self.publish_predicted_path = bool(self.params["publish_predicted_path"])
        self.publish_tree_markers = bool(self.params["publish_tree_markers"])
        self.visualization_publish_period = max(1, int(self.params["visualization_publish_period"]))
        self.pred_path_pub = None
        self.tree_markers_pub = None
        if self.publish_predicted_path:
            self.pred_path_pub = rospy.Publisher(self.params["predicted_path_topic"], Path, queue_size=1)
        if self.publish_tree_markers:
            self.tree_markers_pub = rospy.Publisher(self.params["tree_markers_topic"], MarkerArray, queue_size=1)

        self.trees_pos, self.trees_gt_id = self.ros.get_trees_poses_and_types()
        self.num_total_trees = self.trees_pos.shape[0]
        self.entropy_entire_field = self.optimizer.entropy_f(self.num_total_trees)
        self.beliefs_k = ca.DM.ones(self.num_total_trees, 2) * 0.5

    def _load_runtime_params(self):
        self.N = int(self.params["mpc_horizon"])
        self.dt = float(self.params["dt"])
        self.nx = int(self.params["state_dim"])
        self.n_control = int(self.params["control_dim"])
        self.n_state = int(self.params["optimizer_state_dim"])
        self.num_target_trees = int(
            self.params.get("active_target_count", self.params["num_target_trees"])
        )
        self.params["num_target_trees"] = self.num_target_trees
        self.num_obstacle_trees = int(
            self.params.get("active_obstacle_count", self.params["num_obstacle_trees"])
        )
        self.params["num_obstacle_trees"] = self.num_obstacle_trees
        self.sim_steps = int(self.params.get("max_experiment_steps", self.params["sim_steps"]))
        self.initial_pose_margin = float(self.params["initial_pose_margin"])
        self.initial_pose_sleep = float(self.params["initial_pose_sleep"])
        self.wait_sleep = float(self.params["wait_sleep"])
        self.loop_extra_sleep = float(self.params["loop_extra_sleep"])
        self.belief_update_period = int(self.params["belief_update_period"])
        self.belief_tracking_threshold = float(self.params["belief_tracking_threshold"])
        self.observation_range = float(self.params["observation_range"])
        self.num_runs = int(self.params["num_runs"])
        self.run_index_offset = int(self.params["run_index_offset"])
        self.base_seed = int(self.params["seed"])

    def _validate_comparability(self):
        if self.num_target_trees != int(self.params["active_target_count"]):
            raise ValueError("NMPC target count must equal the RL active_target_count")
        if self.num_obstacle_trees != int(self.params["active_obstacle_count"]):
            raise ValueError("NMPC obstacle count must equal the shared active_obstacle_count")
        if self.num_target_trees < 1 or self.num_obstacle_trees < 1:
            raise ValueError("active target and obstacle counts must be positive")
        if not np.isclose(self.params["nn_threshold"], self.observation_range):
            raise ValueError("NMPC surrogate gate nn_threshold must equal observation_range")
        if list(self.params["model_labels"]) != ["ripe", "raw"]:
            raise ValueError("NMPC belief/model class order must be ['ripe', 'raw']")
        if not 0.5 < self.belief_tracking_threshold < 1.0:
            raise ValueError("belief_tracking_threshold must be between 0.5 and 1.0")
        if self.belief_update_period != 1:
            raise ValueError("belief_update_period must be 1; new evidence is already sequence-gated")
        if not np.isclose(self.dt, self.params["measurement_period"]):
            raise ValueError("NMPC control and measurement periods must match the shared protocol")

    def robot_state_update(self):
        pose = self.ros.robot_pose(default=None)
        if pose is None:
            return None
        return pose.tolist()

    def generate_seeded_initial_state(self):
        start_pose = self.params.get("start_pose")
        if start_pose not in (None, "", []):
            return np.asarray(start_pose, dtype=float).flatten()[:3]
        heading_direction = None if self.params["mower_heading_random"] else self.params["mower_heading"]
        heading = resolve_mower_heading(heading_direction, seed=self.params["trial_seed"])
        return seeded_corner_initial_pose(
            self.trees_pos,
            self.params["trial_seed"],
            margin=self.params["initial_corner_margin"],
            heading=heading,
        )

    def apply_initial_pose(self, initial_pose):
        initial_pose = np.asarray(initial_pose, dtype=float).flatten()[:3]
        timeout = max(0.1, float(self.params["initial_pose_timeout"]))
        publish_period = max(0.02, float(self.params["initial_pose_publish_period"]))
        deadline = time.time() + timeout
        while not rospy.is_shutdown() and time.time() < deadline:
            self.ros.publish_pose(initial_pose)
            current = self.robot_state_update()
            if current is not None:
                current = np.asarray(current, dtype=float)
                position_error = np.linalg.norm(current[:2] - initial_pose[:2])
                heading_error = abs(normalize_angle(current[2] - initial_pose[2]))
                if (
                    position_error <= self.params["initial_pose_tolerance"]
                    and heading_error <= self.params["initial_heading_tolerance"]
                ):
                    return current.tolist()
            rospy.sleep(publish_period)
        raise RuntimeError(
            "NMPC trial {} seed {} did not reach its initial pose within {:.1f}s.".format(
                self.run_index, self.params["trial_seed"], timeout
            )
        )

    def get_target_tree_selection(self, robot_position, num_target=None):
        """Select the same nearest-untracked cardinality exposed to the RL policy."""
        num_target = self.num_target_trees if num_target is None else num_target
        beliefs = np.asarray(self.beliefs_k.full(), dtype=float)
        return self.optimizer.select_nearest_untracked(
            self.trees_pos,
            beliefs,
            robot_position,
            num_target,
            self.belief_tracking_threshold,
        )

    def get_nearest_tree_indices(self, robot_position, num_obstacle=None):
        num_obstacle = self.num_obstacle_trees if num_obstacle is None else num_obstacle
        robot_pos = np.array(robot_position).flatten()
        distances = np.linalg.norm(self.trees_pos - robot_pos, axis=1)
        selected = np.argsort(distances)[:num_obstacle].astype(int).tolist()
        padding_index = selected[-1] if selected else 0
        while len(selected) < num_obstacle:
            selected.append(padding_index)
        return np.asarray(selected, dtype=int)

    def run_simulation(self, run_index=None):
        if run_index is not None:
            self.run_index = int(run_index)
        self.params["algorithm"] = "nmpc"
        self.params["termination_criterion"] = "all_trees_belief_confidence_threshold"
        self.params["run_index"] = self.run_index
        self.params["trial_seed"] = self.base_seed + self.run_index
        self.params["run_dir"] = os.path.join(self.run_root, "nmpc_run_{:03d}".format(self.run_index))
        self.beliefs_k = ca.DM.ones(self.num_total_trees, 2) * 0.5
        self.ros.latest_tree_scores = None
        dynamics = self.optimizer.kin_model(self.n_state, self.n_control, self.dt)
        lb, ub = get_domain(self.trees_pos)
        current_state = None

        if self.initial_randomic:
            current_state = self.apply_initial_pose(self.generate_seeded_initial_state())

        rospy.loginfo("Waiting for robot pose...")
        while current_state is None and not rospy.is_shutdown():
            current_state = self.robot_state_update()
            rospy.sleep(self.wait_sleep)
        rospy.loginfo("Robot pose received.")

        vx_k = ca.DM.zeros(self.nx)
        x_k = ca.vertcat(ca.DM(current_state), vx_k)

        all_trajectories = []
        lambda_history = []
        entropy_history = []
        durations = []

        sim_start_time = time.time()
        prev_x = float(x_k[0])
        prev_y = float(x_k[1])
        total_commands = 0
        sum_vx = 0.0
        sum_vy = 0.0
        sum_yaw = 0.0
        sum_trans_speed = 0.0

        initial_entropy = BeliefState.categorical_entropy_sum(self.beliefs_k.full())
        metrics = WandbMetrics("nmpc", self.run_index, self.params)
        metrics.start()
        metrics.entropy_metrics.update(0.0, self.beliefs_k.full())

        rate = rospy.Rate(max(1, int(1 / self.dt)))
        warm_start = True
        x_dec_prev = None
        lam_g_prev = None
        mpc_step = None
        last_score_sequence = 0
        termination_reason = "max_experiment_steps"

        for mpciter in range(self.sim_steps):
            if rospy.is_shutdown():
                termination_reason = "ros_shutdown"
                break

            loop_start = time.time()
            rospy.loginfo_throttle(5.0, "NMPC step: %d", mpciter)
            current_state = self._wait_for_robot_state()
            x_k = ca.vertcat(ca.DM(current_state), vx_k)

            if last_score_sequence == 0:
                scores, score_sequence = self.ros.wait_for_new_tree_scores(last_score_sequence)
            else:
                scores, score_sequence = self.ros.get_new_tree_scores(last_score_sequence)
            if (
                scores is not None
                and self.belief_update_period > 0
                and mpciter % self.belief_update_period == 0
            ):
                distances = np.linalg.norm(self.trees_pos - np.asarray(current_state[:2]), axis=1)
                beliefs = np.asarray(self.beliefs_k.full(), dtype=float)
                update_mask = (
                    (distances <= self.observation_range)
                    & (np.max(beliefs, axis=1) < self.belief_tracking_threshold)
                )
                last_score_sequence = score_sequence
                try:
                    beliefs = self.optimizer.bayes_numpy(
                        beliefs,
                        scores,
                        update_mask=update_mask,
                    )
                except ValueError as exc:
                    rospy.logerr_throttle(5.0, "Skipping invalid tree-score evidence: %s", exc)
                else:
                    self.beliefs_k = ca.DM(beliefs)

            tracked = np.max(np.asarray(self.beliefs_k.full()), axis=1) >= self.belief_tracking_threshold
            if bool(np.all(tracked)):
                rospy.loginfo("All trees reached belief confidence %.3f.", self.belief_tracking_threshold)
                termination_reason = "all_trees_tracked"
                break

            publish_visualization = mpciter % self.visualization_publish_period == 0
            if publish_visualization and self.tree_markers_pub is not None:
                self.tree_markers_pub.publish(create_tree_markers(self.trees_pos, self.beliefs_k.full()))

            robot_xy = np.array(current_state[:2])
            target_indices, target_mask = self.get_target_tree_selection(robot_xy)
            obstacle_indices = self.get_nearest_tree_indices(robot_xy)
            target_trees = self.trees_pos[target_indices]
            obstacle_trees = self.trees_pos[obstacle_indices]
            target_lambdas = self.beliefs_k[target_indices, :]

            step_start = time.perf_counter()
            try:
                if warm_start or mpc_step is None:
                    rospy.loginfo("Running MPC opt (cold start).")
                    mpc_step, u, x_traj, x_dec_prev, lam_g_prev = self.optimizer.mpc_opt(
                        target_trees,
                        target_lambdas,
                        target_mask,
                        obstacle_trees,
                        lb,
                        ub,
                        x_k,
                        steps=self.N,
                    )
                    warm_start = False
                else:
                    p0_val = ca.vertcat(
                        ca.DM(x_k),
                        ca.reshape(ca.DM(target_trees), 2 * self.num_target_trees, 1),
                        ca.reshape(ca.DM(target_lambdas), 2 * self.num_target_trees, 1),
                        ca.reshape(ca.DM(target_mask), self.num_target_trees, 1),
                        ca.reshape(ca.DM(obstacle_trees), 2 * self.num_obstacle_trees, 1),
                    )
                    u, x_traj, x_dec_prev, lam_g_prev = mpc_step(p0_val, x_dec_prev, lam_g_prev)
            except Exception as exc:
                rospy.logerr("Error during MPC optimization at step %d: %s", mpciter, exc)
                return

            step_duration = time.perf_counter() - step_start
            durations.append(step_duration)
            metrics.log({"mpc_step_duration_s": step_duration})

            cmd_pose = dynamics(x_k, u[:, 0])
            if publish_visualization and self.pred_path_pub is not None:
                self.pred_path_pub.publish(create_path_from_mpc_prediction(x_traj[:self.nx, 1:]))
            self.ros.publish_pose(cmd_pose)

            vx_k = cmd_pose[self.nx:]
            vx_val = float(vx_k[0])
            vy_val = float(vx_k[1])
            yaw_val = float(vx_k[2])
            sum_vx += vx_val
            sum_vy += vy_val
            sum_yaw += yaw_val
            sum_trans_speed += math.sqrt(vx_val ** 2 + vy_val ** 2)
            total_commands += 1

            curr_x = float(x_traj[0, 1])
            curr_y = float(x_traj[1, 1])
            distance_step = math.sqrt((curr_x - prev_x) ** 2 + (curr_y - prev_y) ** 2)
            prev_x, prev_y = curr_x, curr_y
            metrics.add_distance(distance_step)

            entropy_k = self.entropy_entire_field(self.beliefs_k)
            entropy_value = ca.sum1(entropy_k).full().flatten()[0]
            lambda_history.append(self.beliefs_k.full().flatten().tolist())
            entropy_history.append(entropy_value)
            all_trajectories.append(x_traj[:self.nx, :].full())
            metrics.log_pose(
                current_state,
                entropy_value,
                belief=self.beliefs_k.full(),
                tree_positions=self.trees_pos,
            )

            rospy.loginfo_throttle(5.0, "Entropy: %s", entropy_value)
            self._sleep_for_rate(rate, loop_start, mpciter)

        self._finish_metrics(
            metrics,
            initial_entropy,
            entropy_history,
            total_commands,
            sum_vx,
            sum_vy,
            sum_yaw,
            sum_trans_speed,
            sim_start_time,
            durations,
            termination_reason,
        )
        return all_trajectories, entropy_history, lambda_history, durations, self.l4c_nn, self.trees_pos, lb, ub

    def _wait_for_robot_state(self):
        current_state = self.robot_state_update()
        while current_state is None and not rospy.is_shutdown():
            rospy.sleep(self.wait_sleep)
            current_state = self.robot_state_update()
        return current_state

    def _sleep_for_rate(self, rate, loop_start, mpciter):
        loop_elapsed = time.time() - loop_start
        if self.dt - loop_elapsed > 0:
            rate.sleep()
        else:
            rospy.logwarn(
                "Loop iteration %d took %.4fs, longer than dt=%.4fs",
                mpciter,
                loop_elapsed,
                self.dt,
            )
        rospy.sleep(self.loop_extra_sleep)

    def _finish_metrics(
        self,
        metrics,
        initial_entropy,
        entropy_history,
        total_commands,
        sum_vx,
        sum_vy,
        sum_yaw,
        sum_trans_speed,
        sim_start_time,
        durations,
        termination_reason,
    ):
        total_execution_time = time.time() - sim_start_time
        final_entropy = entropy_history[-1] if entropy_history else initial_entropy
        summary = metrics.finish(
            initial_entropy,
            final_entropy,
            final_belief=self.beliefs_k.full(),
            extra={
                "avg_command_period_s": total_execution_time / total_commands if total_commands else 0.0,
                "total_commands": total_commands,
                "avg_vx": sum_vx / total_commands if total_commands else 0.0,
                "avg_vy": sum_vy / total_commands if total_commands else 0.0,
                "avg_yaw_rate": sum_yaw / total_commands if total_commands else 0.0,
                "avg_trans_speed": sum_trans_speed / total_commands if total_commands else 0.0,
                "mean_controller_compute_time_ms": float(np.mean(durations) * 1000.0) if durations else np.nan,
                "median_controller_compute_time_ms": float(np.median(durations) * 1000.0) if durations else np.nan,
                "p95_controller_compute_time_ms": float(np.percentile(durations, 95) * 1000.0) if durations else np.nan,
                "termination_reason": termination_reason,
                "success": termination_reason == "all_trees_tracked",
                "num_tracked_final": int(
                    np.sum(
                        np.max(np.asarray(self.beliefs_k.full()), axis=1)
                        >= self.belief_tracking_threshold
                    )
                ),
                "total_targets": self.num_total_trees,
            },
        )
        rospy.loginfo(
            "NMPC complete: time=%.2fs distance=%.2fm entropy_reduction=%.4f",
            summary["total_time_execution_s"],
            summary["total_distance_m"],
            summary["entropy_reduction"],
        )


def main():
    mpc = NeuralMPC()
    for run_index in range(mpc.run_index_offset, mpc.run_index_offset + mpc.num_runs):
        if rospy.is_shutdown():
            break
        mpc.run_simulation(run_index=run_index)


if __name__ == "__main__":
    main()
