import casadi as ca
import numpy as np


class NmpcOptimizer:
    def __init__(self, params, l4c_nn):
        self.params = params
        self.l4c_nn = l4c_nn
        self.nx = int(params["state_dim"])
        self.n_control = int(params["control_dim"])
        self.n_state = int(params["optimizer_state_dim"])
        self.num_target_trees = int(params["num_target_trees"])
        self.num_obstacle_trees = int(params["num_obstacle_trees"])
        self.dt = float(params["dt"])
        self.horizon = int(params["mpc_horizon"])
        self.entropy_target = self.entropy_f(self.num_target_trees)

    @staticmethod
    def kin_model(nx, nu, dt):
        x_sym = ca.SX.sym("x", nx)
        u_sym = ca.SX.sym("u", nu)

        px, py, pw, vx, vy, vw = [x_sym[i] for i in range(nx)]
        ax, ay, aw = [u_sym[i] for i in range(nu)]

        x_dot = ca.vertcat(vx, vy, vw, ax, ay, aw)
        f_continuous = ca.Function("f_cont", [x_sym, u_sym], [x_dot], ["x", "u"], ["x_dot"])
        k1 = f_continuous(x_sym, u_sym)
        k2 = f_continuous(x_sym + dt / 2 * k1, u_sym)
        k3 = f_continuous(x_sym + dt / 2 * k2, u_sym)
        k4 = f_continuous(x_sym + dt * k3, u_sym)
        x_next = x_sym + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return ca.Function("F", [x_sym, u_sym], [x_next], ["x_k", "u_k"], ["x_k1"])

    @staticmethod
    def select_nearest_untracked(
        tree_positions,
        beliefs,
        robot_position,
        count,
        confidence_threshold,
    ):
        """RL-equivalent nearest-untracked selection with explicit padding mask."""
        trees = np.asarray(tree_positions, dtype=float)[:, :2]
        beliefs = np.asarray(beliefs, dtype=float)
        robot = np.asarray(robot_position, dtype=float).reshape(-1)[:2]
        if len(trees) == 0 or len(trees) != len(beliefs):
            raise ValueError("tree_positions and beliefs must be non-empty and aligned")
        distances = np.linalg.norm(trees - robot, axis=1)
        candidates = np.where(np.max(beliefs, axis=1) < confidence_threshold)[0]
        ordered = candidates[np.argsort(distances[candidates])]
        selected = ordered[:count].astype(int).tolist()
        mask = [1.0] * len(selected)
        padding_index = selected[-1] if selected else int(np.argmin(distances))
        while len(selected) < count:
            selected.append(padding_index)
            mask.append(0.0)
        return np.asarray(selected, dtype=int), np.asarray(mask, dtype=float)

    @staticmethod
    def bayes(prior, likelihood):
        """Categorical Bayes update for CasADi matrices with finite normalization."""
        eps = 1e-9
        likelihood = ca.fmax(eps, likelihood)
        likelihood = likelihood / ca.repmat(ca.sum2(likelihood), 1, likelihood.size2())
        unnorm = ca.fmax(eps, prior) * likelihood
        norm = ca.repmat(ca.sum2(unnorm), 1, unnorm.size2())
        return unnorm / ca.fmax(eps, norm)

    @staticmethod
    def bayes_numpy(prior, likelihood, update_mask=None, eps=1e-9):
        """Numerically safe categorical Bayes update used by the runtime and tests."""
        prior = np.asarray(prior, dtype=float)
        likelihood = np.asarray(likelihood, dtype=float)
        if prior.shape != likelihood.shape or prior.ndim != 2:
            raise ValueError("prior and likelihood must have the same N x C shape")
        if not np.all(np.isfinite(prior)) or not np.all(np.isfinite(likelihood)):
            raise ValueError("prior and likelihood must contain only finite values")
        prior = np.clip(prior, eps, None)
        prior /= np.sum(prior, axis=1, keepdims=True)
        likelihood = np.clip(likelihood, eps, None)
        likelihood /= np.sum(likelihood, axis=1, keepdims=True)
        posterior = prior * likelihood
        posterior /= np.sum(posterior, axis=1, keepdims=True)
        if update_mask is None:
            return posterior
        update_mask = np.asarray(update_mask, dtype=bool).reshape(-1)
        if len(update_mask) != len(prior):
            raise ValueError("update_mask length must match the number of beliefs")
        return np.where(update_mask[:, None], posterior, prior)

    @staticmethod
    def entropy_f(num_targets):
        p = ca.MX.sym("input_entropy_f{}_dim".format(num_targets), num_targets, 2)
        eps = 1e-6
        p_clipped = ca.fmax(eps, ca.fmin(1 - eps, p))
        entropy_per_target = -ca.sum2(p_clipped * (ca.log(p_clipped) / ca.log(2)))
        return ca.Function("entropy_f_{}_dim".format(num_targets), [p], [entropy_per_target])

    def expected_posterior_entropy(self, prior, likelihood_if_class0, likelihood_if_class1):
        """Expected entropy after one measurement, marginalizing both outcomes.

        Rows are targets. Prior columns are the two classes. Each likelihood
        matrix contains P(measurement | true class) for the two outcomes.
        """
        eps = 1e-9
        expected_entropy = ca.MX.zeros(prior.size1(), 1)
        for observation in range(2):
            joint0 = prior[:, 0] * likelihood_if_class0[:, observation]
            joint1 = prior[:, 1] * likelihood_if_class1[:, observation]
            observation_probability = ca.fmax(eps, joint0 + joint1)
            posterior = ca.horzcat(
                joint0 / observation_probability,
                joint1 / observation_probability,
            )
            expected_entropy += observation_probability * self.entropy_target(posterior)
        return expected_entropy

    def mpc_opt(
        self,
        target_trees,
        target_lambdas,
        target_mask,
        obstacle_trees,
        lb,
        ub,
        x0,
        steps=None,
    ):
        steps = self.horizon if steps is None else steps
        opti = ca.Opti()
        F_ = self.kin_model(self.n_state, self.n_control, self.dt)

        X = opti.variable(self.n_state, steps + 1)
        U = opti.variable(self.n_control, steps)

        param_size = (
            self.n_state
            + self.num_target_trees * 4
            + self.num_target_trees
            + self.num_obstacle_trees * 2
        )
        P0 = opti.parameter(param_size)

        p_idx = 0
        X0 = P0[p_idx: p_idx + self.n_state]
        p_idx += self.n_state
        target_param = P0[p_idx: p_idx + self.num_target_trees * 2].reshape(
            (self.num_target_trees, 2)
        ).T
        p_idx += self.num_target_trees * 2
        L0 = P0[p_idx: p_idx + self.num_target_trees * 2].reshape((self.num_target_trees, 2))
        p_idx += self.num_target_trees * 2
        target_mask_param = P0[p_idx: p_idx + self.num_target_trees]
        p_idx += self.num_target_trees
        obstacle_param = P0[p_idx: p_idx + self.num_obstacle_trees * 2].reshape(
            (self.num_obstacle_trees, 2)
        ).T
        safe_distance = float(self.params["safe_distance"])
        observation_range = float(self.params["observation_range"])
        movement_weight = float(self.params["movement_weight"])
        yaw_movement_weight = float(self.params["yaw_movement_weight"])
        acceleration_regularization_weight = float(
            self.params["acceleration_regularization_weight"]
        )
        information_gain_weight = float(self.params["information_gain_weight"])
        information_discount = float(self.params["information_discount"])
        attraction_weight = float(self.params["attraction_weight"])
        obj = 0
        information_gain_by_target = ca.MX.zeros(self.num_target_trees, 1)
        terminal_min_dist_sq = None

        opti.subject_to(X[:, 0] == X0)
        ca_batch = []

        for i in range(steps):
            opti.subject_to(opti.bounded(-self.params["max_accel_xy"], U[0:2, i], self.params["max_accel_xy"]))
            opti.subject_to(opti.bounded(-self.params["max_accel_yaw"], U[2, i], self.params["max_accel_yaw"]))
            opti.subject_to(X[:, i + 1] == F_(X[:, i], U[:, i]))
            opti.subject_to(opti.bounded(lb[0] - self.params["field_margin"], X[0, i + 1], ub[0] + self.params["field_margin"]))
            opti.subject_to(opti.bounded(lb[1] - self.params["field_margin"], X[1, i + 1], ub[1] + self.params["field_margin"]))
            opti.subject_to(opti.bounded(-self.params["max_heading_abs"], X[2, i + 1], self.params["max_heading_abs"]))
            opti.subject_to(ca.sumsqr(X[3:5, i + 1]) <= self.params["max_velocity"] ** 2)
            opti.subject_to(opti.bounded(-self.params["max_yaw_velocity"], X[5, i + 1], self.params["max_yaw_velocity"]))

            for j in range(self.num_obstacle_trees):
                dist_sq_obs = ca.sumsqr(X[:2, i + 1] - obstacle_param[:, j])
                opti.subject_to(dist_sq_obs >= safe_distance ** 2)

            distances_sq = []
            nn_batch = []
            for j in range(self.num_target_trees):
                diff = X[:2, i + 1] - target_param[:, j]
                distances_sq.append(
                    ca.sumsqr(diff) + (1.0 - target_mask_param[j]) * 1e6 + 1e-6
                )
                nn_batch.append(ca.horzcat(diff.T, X[2, i + 1]))
            ca_batch.append(ca.vcat([*nn_batch]))

            min_dist_sq = distances_sq[0]
            for j in range(1, self.num_target_trees):
                min_dist_sq = ca.fmin(min_dist_sq, distances_sq[j])
            terminal_min_dist_sq = min_dist_sq
            normalized_speed = ca.sqrt(ca.sumsqr(X[3:5, i + 1]) + 1e-8) / max(
                float(self.params["max_velocity"]), 1e-6
            )
            normalized_yaw_speed = ca.sqrt(X[5, i + 1] ** 2 + 1e-8) / max(
                float(self.params["max_yaw_velocity"]), 1e-6
            )
            normalized_acceleration = (
                ca.sumsqr(U[:2, i]) / max(float(self.params["max_accel_xy"]) ** 2, 1e-6)
                + U[2, i] ** 2 / max(float(self.params["max_accel_yaw"]) ** 2, 1e-6)
            )
            obj += movement_weight * normalized_speed
            obj += yaw_movement_weight * normalized_yaw_speed
            obj += acceleration_regularization_weight * normalized_acceleration

        nn_full_batch_input = ca.vcat(ca_batch)
        surrogate_output_ripe = self.l4c_nn[0](nn_full_batch_input)
        surrogate_output_raw = self.l4c_nn[1](nn_full_batch_input)

        for i in range(steps):
            start = i * self.num_target_trees
            stop = (i + 1) * self.num_target_trees
            expected_entropy = self.expected_posterior_entropy(
                L0,
                surrogate_output_ripe[start:stop, :],
                surrogate_output_raw[start:stop, :],
            )
            prior_entropy = self.entropy_target(L0)
            information_gain_by_target += information_discount ** i * ca.fmax(
                0.0,
                prior_entropy - expected_entropy,
            )

        # Multiple predicted observations may concern the same tree. Cap their
        # accumulated information by the tree's current entropy, since no
        # planner can remove more uncertainty than is present initially.
        bounded_information_gain = ca.fmin(
            self.entropy_target(L0),
            information_gain_by_target,
        )
        information_gain_objective = ca.dot(target_mask_param, bounded_information_gain) / ca.fmax(
            1.0,
            ca.sum1(target_mask_param),
        )

        terminal_distance_excess = ca.fmax(
            0.0,
            ca.sqrt(terminal_min_dist_sq) - observation_range,
        )

        opti.minimize(
            obj
            - information_gain_weight * information_gain_objective
            + attraction_weight * terminal_distance_excess
        )
        options = {"print_time": False, "ipopt": dict(self.params["ipopt"])}
        opti.solver("ipopt", options)
        inputs = [P0, opti.x, opti.lam_g]
        outputs = [U[:, 0], X, opti.x, opti.lam_g]

        p0_val = ca.vertcat(
            ca.DM(x0),
            ca.reshape(ca.DM(target_trees), 2 * self.num_target_trees, 1),
            ca.reshape(ca.DM(target_lambdas), 2 * self.num_target_trees, 1),
            ca.reshape(ca.DM(target_mask), self.num_target_trees, 1),
            ca.reshape(ca.DM(obstacle_trees), 2 * self.num_obstacle_trees, 1),
        )
        opti.set_value(P0, p0_val)

        sol = opti.solve()
        mpc_step_func = opti.to_function(
            "mpc_step",
            inputs,
            outputs,
            ["p", "x_init", "x_lam"],
            ["u_opt", "x_pred", "x_opt", "lam_opt"],
        )

        return (
            mpc_step_func,
            ca.DM(sol.value(U[:, 0])),
            ca.DM(sol.value(X)),
            ca.DM(sol.value(opti.x)),
            ca.DM(sol.value(opti.lam_g)),
        )
