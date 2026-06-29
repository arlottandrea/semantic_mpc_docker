import math

import numpy as np

try:
    import casadi as ca
except ImportError:  # pragma: no cover - handled at runtime by the caller
    ca = None


def _normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class CasadiMpcStepGenerator:
    def __init__(self, params):
        if ca is None:
            raise RuntimeError("casadi_mpc baseline selected, but the casadi Python package is not available.")

        self.params = params
        self.horizon = int(params["mpc_steps"])
        self.dt = float(params["dt"])
        self.max_velocity = float(params["max_velocity"])
        self.min_velocity = float(params["min_velocity"])
        self.max_yaw_velocity = float(params["max_yaw_velocity"])
        self.max_lin_accel = float(params["max_lin_accel"])
        self.max_yaw_accel = float(params["max_yaw_accel"])
        self.safe_distance = float(params["obstacle_avoidance_distance"])
        self.goal_ignore_distance = float(params["obstacle_goal_ignore_distance"])
        self.goal_weight = float(params["mpc_goal_weight"])
        self.heading_weight = float(params["mpc_heading_weight"])
        self.control_weight = float(params["mpc_control_weight"])
        self.smooth_weight = float(params["mpc_smooth_weight"])
        self.obstacle_weight = float(params["mpc_obstacle_weight"])
        self.velocity_weight = float(params.get("mpc_velocity_weight", 2.0))
        self.max_obstacles = int(params["mpc_max_obstacles"])
        self.ipopt_max_iter = int(params["mpc_ipopt_max_iter"])

    def command(self, current_pose, goal_xy, desired_heading, tree_positions, previous_cmd, desired_velocity=None):
        current_pose = np.asarray(current_pose, dtype=float).flatten()
        goal_xy = np.asarray(goal_xy, dtype=float).flatten()
        previous_cmd = np.asarray(previous_cmd, dtype=float).flatten()
        desired_heading = float(desired_heading)
        if desired_velocity is not None:
            desired_velocity = np.asarray(desired_velocity, dtype=float).flatten()

        obstacle_positions = self._select_obstacles(current_pose[:2], goal_xy, tree_positions)
        try:
            return self._solve(current_pose, goal_xy, desired_heading, obstacle_positions, previous_cmd, desired_velocity)
        except Exception:
            return self._fallback_command(current_pose, goal_xy, desired_heading, previous_cmd, desired_velocity)

    def _select_obstacles(self, current_xy, goal_xy, tree_positions):
        if tree_positions is None or len(tree_positions) == 0 or self.max_obstacles <= 0:
            return np.empty((0, 2))

        trees = np.asarray(tree_positions, dtype=float)
        goal_distances = np.linalg.norm(trees - goal_xy, axis=1)
        candidates = trees[goal_distances >= self.goal_ignore_distance]
        if len(candidates) == 0:
            return np.empty((0, 2))

        distances = np.linalg.norm(candidates - current_xy, axis=1)
        order = np.argsort(distances)[: self.max_obstacles]
        return candidates[order]

    def _solve(self, current_pose, goal_xy, desired_heading, obstacles, previous_cmd, desired_velocity):
        opti = ca.Opti()
        states = opti.variable(3, self.horizon + 1)
        controls = opti.variable(3, self.horizon)

        opti.subject_to(states[:, 0] == current_pose)
        objective = 0
        pass_through = desired_velocity is not None
        velocity_ref = ca.DM(np.zeros(2) if desired_velocity is None else desired_velocity[:2])

        for step in range(self.horizon):
            vx = controls[0, step]
            vy = controls[1, step]
            wz = controls[2, step]

            opti.subject_to(states[0, step + 1] == states[0, step] + vx * self.dt)
            opti.subject_to(states[1, step + 1] == states[1, step] + vy * self.dt)
            opti.subject_to(states[2, step + 1] == states[2, step] + wz * self.dt)

            opti.subject_to(vx ** 2 + vy ** 2 <= self.max_velocity ** 2)
            opti.subject_to(opti.bounded(-self.max_yaw_velocity, wz, self.max_yaw_velocity))

            if step == 0:
                delta = controls[:, step] - previous_cmd
            else:
                delta = controls[:, step] - controls[:, step - 1]
            opti.subject_to(opti.bounded(-self.max_lin_accel * self.dt, delta[0], self.max_lin_accel * self.dt))
            opti.subject_to(opti.bounded(-self.max_lin_accel * self.dt, delta[1], self.max_lin_accel * self.dt))
            opti.subject_to(opti.bounded(-self.max_yaw_accel * self.dt, delta[2], self.max_yaw_accel * self.dt))

            pos_error = states[:2, step + 1] - goal_xy
            heading_error = ca.atan2(
                ca.sin(states[2, step + 1] - desired_heading),
                ca.cos(states[2, step + 1] - desired_heading),
            )
            if pass_through:
                if step == self.horizon - 1:
                    objective += self.goal_weight * ca.sumsqr(pos_error)
                objective += self.velocity_weight * ca.sumsqr(controls[:2, step] - velocity_ref)
            else:
                objective += self.goal_weight * ca.sumsqr(pos_error)
            objective += self.heading_weight * heading_error ** 2
            objective += self.control_weight * ca.sumsqr(controls[:, step])
            objective += self.smooth_weight * ca.sumsqr(delta)

            for obstacle in obstacles:
                diff = states[:2, step + 1] - obstacle
                dist_sq = ca.sumsqr(diff)
                opti.subject_to(dist_sq >= self.safe_distance ** 2)
                objective += self.obstacle_weight / (dist_sq + 1e-3)

        opti.minimize(objective)
        opti.set_initial(states, np.tile(current_pose.reshape(3, 1), (1, self.horizon + 1)))
        opti.set_initial(controls, np.tile(previous_cmd.reshape(3, 1), (1, self.horizon)))
        opti.solver(
            "ipopt",
            {
                "print_time": False,
<<<<<<< HEAD
=======
                "verbose": False,
>>>>>>> ee87ff70deff38d328f365837d10c4d97d5f2b51
                "ipopt": {
                    "print_level": 0,
                    "max_iter": self.ipopt_max_iter,
                    "sb": "yes",
                },
            },
        )

        solution = opti.solve()
        cmd = np.array(solution.value(controls[:, 0]), dtype=float).flatten()
        speed = np.linalg.norm(cmd[:2])
        distance_to_goal = np.linalg.norm(goal_xy - current_pose[:2])
        if self.min_velocity > 0.0 and speed < self.min_velocity and distance_to_goal > self.params["tolerance"]:
            if speed > 1e-6:
                cmd[:2] *= self.min_velocity / speed
            else:
                cmd[:2] = (goal_xy - current_pose[:2]) / distance_to_goal * self.min_velocity
        return cmd

    def _fallback_command(self, current_pose, goal_xy, desired_heading, previous_cmd, desired_velocity=None):
        diff = goal_xy - current_pose[:2]
        distance = np.linalg.norm(diff)
        if desired_velocity is not None and distance <= max(self.params["tolerance"], self.max_velocity * self.dt):
            linear = np.asarray(desired_velocity[:2], dtype=float)
        elif distance > 1e-6:
            speed = min(self.max_velocity, max(self.min_velocity, distance / max(self.dt, 1e-6)))
            linear = diff / distance * speed
        else:
            linear = np.zeros(2)

        yaw_error = _normalize_angle(desired_heading - current_pose[2])
        yaw_rate = np.clip(yaw_error / max(self.dt, 1e-6), -self.max_yaw_velocity, self.max_yaw_velocity)
        raw_cmd = np.array([linear[0], linear[1], yaw_rate], dtype=float)

        max_delta = np.array(
            [
                self.max_lin_accel * self.dt,
                self.max_lin_accel * self.dt,
                self.max_yaw_accel * self.dt,
            ]
        )
        return previous_cmd + np.clip(raw_cmd - previous_cmd, -max_delta, max_delta)
