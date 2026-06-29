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
        self.num_target_trees = int(self.params["num_target_trees"])
        self.num_obstacle_trees = int(self.params["num_obstacle_trees"])
        self.sim_steps = int(self.params["sim_steps"])
        self.initial_pose_margin = float(self.params["initial_pose_margin"])
        self.initial_pose_sleep = float(self.params["initial_pose_sleep"])
        self.wait_sleep = float(self.params["wait_sleep"])
        self.loop_extra_sleep = float(self.params["loop_extra_sleep"])
        self.belief_update_period = int(self.params["belief_update_period"])
        self.entropy_stop_threshold = float(self.params["entropy_stop_threshold"])
        self.entropy_selection_threshold = float(self.params["entropy_selection_threshold"])
        self.num_runs = int(self.params["num_runs"])
        self.run_index_offset = int(self.params["run_index_offset"])
        self.base_seed = int(self.params["seed"])

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

    def get_target_tree_indices(self, robot_position, num_target=None):
        num_target = self.num_target_trees if num_target is None else num_target
        robot_pos = np.array(robot_position).flatten()
        distances = np.linalg.norm(self.trees_pos[:, :2] - robot_pos, axis=1)
        entropy = self.entropy_entire_field(self.beliefs_k).full()
        candidate_indices = np.where(entropy > self.entropy_selection_threshold)[0]

        if candidate_indices.size == 0:
            rospy.logwarn_throttle(5.0, "No trees above entropy threshold; selecting nearest trees.")
            return np.argsort(distances)[:num_target]

        sorted_candidates = candidate_indices[np.argsort(distances[candidate_indices])]
        if sorted_candidates.size < num_target:
            repeats = int(np.ceil(num_target / sorted_candidates.size))
            sorted_candidates = np.tile(sorted_candidates, repeats)[:num_target]
        return sorted_candidates[:num_target]

    def get_nearest_tree_indices(self, robot_position, num_obstacle=None):
        num_obstacle = self.num_obstacle_trees if num_obstacle is None else num_obstacle
        robot_pos = np.array(robot_position).flatten()
        distances = np.linalg.norm(self.trees_pos - robot_pos, axis=1)
        return np.argsort(distances)[:num_obstacle]

    def run_simulation(self, run_index=None):
        if run_index is not None:
            self.run_index = int(run_index)
        self.params["algorithm"] = "nmpc"
        self.params["termination_criterion"] = "all_tree_entropy_threshold"
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

        for mpciter in range(self.sim_steps):
            if rospy.is_shutdown():
                break

            loop_start = time.time()
            rospy.loginfo_throttle(5.0, "NMPC step: %d", mpciter)
            current_state = self._wait_for_robot_state()
            x_k = ca.vertcat(ca.DM(current_state), vx_k)

            scores = self.ros.wait_for_tree_scores()
            if self.belief_update_period > 0 and mpciter % self.belief_update_period == 0:
                self.beliefs_k = self.optimizer.bayes(self.beliefs_k, ca.DM(scores))

            publish_visualization = mpciter % self.visualization_publish_period == 0
            if publish_visualization and self.tree_markers_pub is not None:
                self.tree_markers_pub.publish(create_tree_markers(self.trees_pos, self.beliefs_k.full()))

            robot_xy = np.array(current_state[:2])
            target_indices = self.get_target_tree_indices(robot_xy)
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
                        obstacle_trees,
                        lb,
                        ub,
                        x_k,
                        steps=self.N,
                    )
                    warm_start = False
                else:
                    p0_val = ca.vertcat(
                        x_k,
                        ca.reshape(target_trees, 2 * self.num_target_trees, 1),
                        ca.reshape(target_lambdas, 2 * self.num_target_trees, 1),
                        ca.reshape(obstacle_trees, 2 * self.num_obstacle_trees, 1),
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
            if all(value <= self.entropy_stop_threshold for value in entropy_k.full().flatten()):
                rospy.loginfo("Entropy target reached.")
                break

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
