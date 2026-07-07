import casadi as ca
import numpy as np


class NmpcOptimizer:
    _SMOOTH_EPS = 1e-8

    @staticmethod
    def smooth_positive(value, eps=_SMOOTH_EPS):
        """Differentiable approximation of max(0, value)."""
        return 0.5 * (value + ca.sqrt(value * value + eps))

    @staticmethod
    def smooth_min(left, right, eps=_SMOOTH_EPS):
        """Differentiable approximation of min(left, right)."""
        delta = left - right
        return 0.5 * (left + right - ca.sqrt(delta * delta + eps))

    @staticmethod
    def camera_facing_error(robot_pose, tree_position, camera_yaw_offset=0.0):
        """Smooth periodic cost: zero when the camera points at the tree."""
        direction_to_tree = ca.atan2(
            tree_position[1] - robot_pose[1],
            tree_position[0] - robot_pose[0],
        )
        error = direction_to_tree - robot_pose[2] - camera_yaw_offset
        return 1.0 - ca.cos(error)

    @staticmethod
    def perception_features(robot_pose, tree_position):
        """Build the exact ``[x, y, yaw]`` feature used by MLP training.

        Training CSV coordinates are the drone position relative to a tree in
        a world-aligned frame.  The yaw column is the drone heading in that
        same frame, not the bearing to the tree.  Wrapping is differentiable
        almost everywhere and is equivalent under the model's sin/cos yaw
        encoding.
        """
        relative_position = robot_pose[:2] - tree_position[:2]
        wrapped_yaw = ca.atan2(ca.sin(robot_pose[2]), ca.cos(robot_pose[2]))
        return ca.horzcat(relative_position.T, wrapped_yaw)

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
    def entropy_f(num_targets, num_classes=2):
        """Conditional ripe/raw entropy, excluding an optional nothing class.

        Columns must be ordered ``[ripe, raw]`` or ``[ripe, raw, nothing]``.
        The first two columns are re-normalized by their fruit probability
        mass.  A row containing only ``nothing`` has zero fruit entropy.
        """
        if num_classes not in (2, 3):
            raise ValueError("fruit entropy requires 2 or 3 probability columns")
        p = ca.MX.sym(
            "input_entropy_f{}_c{}".format(num_targets, num_classes),
            num_targets,
            num_classes,
        )
        eps = 1e-8
        fruit = ca.fmax(0.0, p[:, :2])
        fruit_mass = ca.sum2(fruit)
        conditional = fruit / ca.repmat(ca.fmax(eps, fruit_mass), 1, 2)
        log_terms = ca.if_else(
            conditional > eps,
            conditional * (ca.log(ca.fmax(eps, conditional)) / ca.log(2)),
            0.0,
        )
        entropy_per_target = ca.if_else(
            fruit_mass > eps,
            -ca.sum2(log_terms),
            0.0,
        )
        return ca.Function(
            "entropy_f_{}_c{}".format(num_targets, num_classes),
            [p],
            [entropy_per_target],
        )

    def expected_posterior_entropy(self, prior, likelihood_if_class0, likelihood_if_class1):
        """Expected entropy after one measurement, marginalizing both outcomes.

        Rows are targets. Prior columns are the two classes. Each likelihood
        matrix contains P(measurement | true class).  The number of outcomes
        is read from the matrices (the deployed model uses nothing/raw/ripe).
        """
        if likelihood_if_class0.size2() != likelihood_if_class1.size2():
            raise ValueError("class likelihoods must have equal outcome counts")
        eps = 1e-9
        expected_entropy = ca.MX.zeros(prior.size1(), 1)
        for observation in range(likelihood_if_class0.size2()):
            joint0 = prior[:, 0] * likelihood_if_class0[:, observation]
            joint1 = prior[:, 1] * likelihood_if_class1[:, observation]
            observation_probability = joint0 + joint1
            safe_probability = ca.fmax(eps, observation_probability)
            posterior = ca.horzcat(
                joint0 / safe_probability,
                joint1 / safe_probability,
            )
            expected_entropy += observation_probability * self.entropy_target(posterior)
        return expected_entropy

    def expected_entropy_horizon(self, prior, likelihoods_if_class0, likelihoods_if_class1):
        """Expected posterior entropy after each sequential horizon measurement.

        The belief tree branches over both possible observations.  This is
        exact for the binary model and, unlike summing one-step gains computed
        from the same prior, does not count the same uncertainty repeatedly.
        """
        if len(likelihoods_if_class0) != len(likelihoods_if_class1):
            raise ValueError("class likelihood horizons must have equal length")

        eps = 1e-9
        branches = [(ca.MX.ones(prior.size1(), 1), prior)]
        stage_entropies = []
        for class0, class1 in zip(likelihoods_if_class0, likelihoods_if_class1):
            next_branches = []
            if class0.size2() != class1.size2():
                raise ValueError("class likelihoods must have equal outcome counts")
            for branch_probability, branch_prior in branches:
                for observation in range(class0.size2()):
                    joint0 = branch_prior[:, 0] * class0[:, observation]
                    joint1 = branch_prior[:, 1] * class1[:, observation]
                    observation_probability = joint0 + joint1
                    safe_probability = ca.fmax(eps, observation_probability)
                    posterior = ca.horzcat(
                        joint0 / safe_probability,
                        joint1 / safe_probability,
                    )
                    next_branches.append(
                        (branch_probability * observation_probability, posterior)
                    )
            branches = next_branches
            expected_entropy = ca.MX.zeros(prior.size1(), 1)
            for branch_probability, posterior in branches:
                expected_entropy += branch_probability * self.entropy_target(posterior)
            stage_entropies.append(expected_entropy)
        return stage_entropies

    @staticmethod
    def observation_likelihoods(structured_output):
        """Read a flattened 2x3 conditional observation matrix.

        Model rows are true ``[raw, ripe]`` and columns are observations
        ``[nothing, raw, ripe]``. Belief columns are ordered ``[ripe, raw]``.
        """
        if structured_output.size2() != 6:
            raise ValueError("conditional observation model must have six columns")
        likelihood_if_raw = structured_output[:, 0:3]
        likelihood_if_ripe = structured_output[:, 3:6]
        return likelihood_if_ripe, likelihood_if_raw

    @staticmethod
    def observed_categories(class_scores, decision_margin=0.05):
        """Map detector scores [ripe, raw] to nothing/raw/ripe indices."""
        scores = np.asarray(class_scores, dtype=float)
        if scores.ndim != 2 or scores.shape[1] != 2:
            raise ValueError("class_scores must have shape N x 2 in [ripe, raw] order")
        if not np.all(np.isfinite(scores)):
            raise ValueError("class_scores must contain only finite values")
        categories = np.zeros(len(scores), dtype=int)
        decisive = np.abs(scores[:, 0] - scores[:, 1]) > float(decision_margin)
        categories[decisive & (scores[:, 1] > scores[:, 0])] = 1
        categories[decisive & (scores[:, 0] > scores[:, 1])] = 2
        return categories

    @staticmethod
    def realized_likelihoods(structured_output, categories):
        """Select P(observed category | ripe/raw) in belief-column order."""
        output = np.asarray(structured_output, dtype=float)
        categories = np.asarray(categories, dtype=int).reshape(-1)
        if output.ndim != 2 or output.shape[1] != 6 or len(output) != len(categories):
            raise ValueError("structured_output must be N x 6 and align with categories")
        if np.any((categories < 0) | (categories > 2)):
            raise ValueError("observation categories must be in {0, 1, 2}")
        rows = np.arange(len(categories))
        return np.column_stack(
            (output[rows, 3 + categories], output[rows, categories])
        )

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
        exploration_weight = float(self.params["exploration_weight"])
        exploration_sigma = max(float(self.params["exploration_sigma"]), 1e-6)
        attraction_weight = float(self.params["attraction_weight"])
        camera_facing_weight = float(self.params["camera_facing_weight"])
        camera_yaw_offset = float(self.params["camera_yaw_offset"])
        camera_activation_sigma = max(
            float(self.params["camera_activation_sigma"]), 1e-6
        )
        observation_standoff = max(float(self.params["observation_standoff"]), 1e-6)
        observation_standoff_weight = float(
            self.params["observation_standoff_weight"]
        )
        obj = 0
        exploration_reward = 0
        terminal_min_dist_sq = None
        terminal_camera_cost = 0.0
        terminal_standoff_cost = 0.0

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
                nn_batch.append(
                    self.perception_features(X[:3, i + 1], target_param[:, j])
                )
            ca_batch.append(ca.vcat([*nn_batch]))

            min_dist_sq = distances_sq[0]
            for j in range(1, self.num_target_trees):
                min_dist_sq = self.smooth_min(min_dist_sq, distances_sq[j])
            terminal_min_dist_sq = min_dist_sq
            # Continuous equivalent of sampling a Gaussian covariance grid.
            # Bernoulli covariance is maximal at p=0.5 and vanishes as a tree
            # becomes classified.  The target mask removes padded slots.
            target_covariance = 4.0 * L0[:, 0] * L0[:, 1]
            gaussian_covariance = ca.MX.zeros(self.num_target_trees, 1)
            for j in range(self.num_target_trees):
                gaussian_covariance[j] = ca.exp(
                    -distances_sq[j] / (2.0 * exploration_sigma ** 2)
                )
            exploration_reward += information_discount ** i * ca.dot(
                target_mask_param * target_covariance,
                gaussian_covariance,
            ) / ca.fmax(1.0, ca.sum1(target_mask_param))
            if i == steps - 1:
                camera_weights = ca.MX.zeros(self.num_target_trees, 1)
                camera_errors = ca.MX.zeros(self.num_target_trees, 1)
                for j in range(self.num_target_trees):
                    camera_weights[j] = (
                        target_mask_param[j]
                        * target_covariance[j]
                        * ca.exp(
                            -distances_sq[j] / (2.0 * camera_activation_sigma ** 2)
                        )
                    )
                    camera_errors[j] = self.camera_facing_error(
                        X[:3, i + 1], target_param[:, j], camera_yaw_offset
                    )
                terminal_camera_cost = ca.dot(camera_weights, camera_errors) / ca.fmax(
                    1e-8, ca.sum1(camera_weights)
                )
                standoff_errors = ca.MX.zeros(self.num_target_trees, 1)
                for j in range(self.num_target_trees):
                    distance = ca.sqrt(distances_sq[j])
                    standoff_errors[j] = (
                        (distance - observation_standoff) / observation_standoff
                    ) ** 2
                terminal_standoff_cost = ca.dot(
                    camera_weights, standoff_errors
                ) / ca.fmax(1e-8, ca.sum1(camera_weights))
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
        surrogate_output = self.l4c_nn(nn_full_batch_input)
        surrogate_output_ripe, surrogate_output_raw = self.observation_likelihoods(
            surrogate_output
        )

        likelihoods_ripe = []
        likelihoods_raw = []
        for i in range(steps):
            start = i * self.num_target_trees
            stop = (i + 1) * self.num_target_trees
            likelihoods_ripe.append(surrogate_output_ripe[start:stop, :])
            likelihoods_raw.append(surrogate_output_raw[start:stop, :])

        # Exact open-loop belief-space objective.  Every possible observation
        # sequence is marginalized and Bayes' rule is applied recursively.
        # Discounted entropy reductions reward information obtained earlier
        # without counting the same uncertainty more than once.
        prior_entropy = self.entropy_target(L0)
        horizon_entropies = self.expected_entropy_horizon(
            L0, likelihoods_ripe, likelihoods_raw
        )
        discounted_information_gain = ca.MX.zeros(self.num_target_trees, 1)
        previous_entropy = prior_entropy
        for i, expected_entropy in enumerate(horizon_entropies):
            discounted_information_gain += information_discount ** i * (
                previous_entropy - expected_entropy
            )
            previous_entropy = expected_entropy
        information_gain_objective = ca.dot(
            target_mask_param, discounted_information_gain
        ) / ca.fmax(
            1.0,
            ca.sum1(target_mask_param),
        )

        terminal_distance_excess = self.smooth_positive(
            ca.sqrt(self.smooth_positive(terminal_min_dist_sq)) - observation_range
        )

        opti.minimize(
            obj
            - information_gain_weight * information_gain_objective
            - exploration_weight * exploration_reward
            + attraction_weight * terminal_distance_excess
            + camera_facing_weight * terminal_camera_cost
            + observation_standoff_weight * terminal_standoff_cost
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

        # A zero control sequence is a common stationary point of radial
        # neural/grid potentials.  Give the cold solve a feasible-dynamics
        # directional seed toward the nearest active uncertain target.
        x0_np = np.asarray(x0, dtype=float).reshape(-1)
        target_np = np.asarray(target_trees, dtype=float).reshape(self.num_target_trees, 2)
        mask_np = np.asarray(target_mask, dtype=float).reshape(-1) > 0.5
        u_seed = np.zeros((self.n_control, steps), dtype=float)
        x_seed = np.zeros((self.n_state, steps + 1), dtype=float)
        x_seed[:, 0] = x0_np
        if np.any(mask_np):
            active_targets = target_np[mask_np]
            nearest = active_targets[
                np.argmin(np.linalg.norm(active_targets - x0_np[:2], axis=1))
            ]
            direction = nearest - x0_np[:2]
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 1e-9:
                u_seed[:2, :] = (
                    0.5 * float(self.params["max_accel_xy"]) * direction / direction_norm
                )[:, None]
        for i in range(steps):
            acceleration = u_seed[:, i]
            x_seed[:3, i + 1] = (
                x_seed[:3, i]
                + self.dt * x_seed[3:, i]
                + 0.5 * self.dt ** 2 * acceleration
            )
            x_seed[3:, i + 1] = x_seed[3:, i] + self.dt * acceleration
        opti.set_initial(U, u_seed)
        opti.set_initial(X, x_seed)

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
