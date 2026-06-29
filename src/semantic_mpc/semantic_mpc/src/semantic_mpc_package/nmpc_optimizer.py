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
    def bayes(prior, likelihood):
        unnorm = prior * likelihood
        norm = ca.repmat(ca.sum2(unnorm), 1, 2)
        return unnorm / norm

    @staticmethod
    def entropy_f(num_targets):
        p = ca.MX.sym("input_entropy_f{}_dim".format(num_targets), num_targets, 2)
        eps = 1e-6
        p_clipped = ca.fmax(eps, ca.fmin(1 - eps, p))
        entropy_per_target = -ca.sum2(p_clipped * (ca.log(p_clipped) / ca.log(2)))
        return ca.Function("entropy_f_{}_dim".format(num_targets), [p], [entropy_per_target])

    def mpc_opt(self, target_trees, target_lambdas, obstacle_trees, lb, ub, x0, steps=None):
        steps = self.horizon if steps is None else steps
        opti = ca.Opti()
        F_ = self.kin_model(self.n_state, self.n_control, self.dt)

        X = opti.variable(self.n_state, steps + 1)
        U = opti.variable(self.n_control, steps)

        param_size = self.n_state + self.num_target_trees * 4 + self.num_obstacle_trees * 2
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
        obstacle_param = P0[p_idx: p_idx + self.num_obstacle_trees * 2].reshape(
            (self.num_obstacle_trees, 2)
        ).T
        lambda_evol = [L0]

        q_dist = float(self.params["q_dist"])
        r_xy = float(self.params["r_xy"])
        r_theta = float(self.params["r_theta"])
        safe_distance = float(self.params["safe_distance"])
        entropy_weight = float(self.params["entropy_weight"])
        obj = 0
        attraction = 0

        opti.subject_to(X[:, 0] == X0)
        ca_batch = []

        for i in range(steps):
            opti.subject_to(opti.bounded(lb[0] - self.params["field_margin"], X[0, i], ub[0] + self.params["field_margin"]))
            opti.subject_to(opti.bounded(lb[1] - self.params["field_margin"], X[1, i], ub[1] + self.params["field_margin"]))
            opti.subject_to(opti.bounded(-self.params["max_heading_abs"], X[2, i], self.params["max_heading_abs"]))
            opti.subject_to(opti.bounded(-self.params["max_velocity"], X[3:5, i], self.params["max_velocity"]))
            opti.subject_to(opti.bounded(-self.params["max_yaw_velocity"], X[5, i], self.params["max_yaw_velocity"]))
            opti.subject_to(opti.bounded(-self.params["max_accel_xy"], U[0:2, i], self.params["max_accel_xy"]))
            opti.subject_to(opti.bounded(-self.params["max_accel_yaw"], U[2, i], self.params["max_accel_yaw"]))
            opti.subject_to(X[:, i + 1] == F_(X[:, i], U[:, i]))

            for j in range(self.num_obstacle_trees):
                dist_sq_obs = ca.sumsqr(X[:2, i + 1] - obstacle_param[:, j])
                opti.subject_to(dist_sq_obs >= safe_distance ** 2)

            distances_sq = []
            nn_batch = []
            for j in range(self.num_target_trees):
                diff = X[:2, i + 1] - target_param[:, j]
                distances_sq.append(ca.sumsqr(diff) + 1e-6)
                nn_batch.append(ca.horzcat(diff.T, X[2, i + 1]))
            ca_batch.append(ca.vcat([*nn_batch]))

            min_dist_sq = distances_sq[0]
            for j in range(1, self.num_target_trees):
                min_dist_sq = ca.fmin(min_dist_sq, distances_sq[j])
            attraction += min_dist_sq * q_dist
            obj += r_xy * ca.sumsqr(U[:2, i]) + r_theta * ca.sumsqr(U[2, i])

        nn_full_batch_input = ca.vcat(ca_batch)
        surrogate_output_ripe = self.l4c_nn[0](nn_full_batch_input)
        surrogate_output_raw = self.l4c_nn[1](nn_full_batch_input)

        L0_ext = ca.vcat([L0 for _ in range(steps)])
        sel = L0_ext[:, 0] >= L0_ext[:, 1]
        col1 = ca.if_else(sel, surrogate_output_ripe[:, 0], surrogate_output_raw[:, 0])
        col2 = ca.if_else(sel, surrogate_output_ripe[:, 1], surrogate_output_raw[:, 1])
        z_k_bin = ca.horzcat(col1, col2)

        for i in range(steps):
            lambda_next = self.bayes(
                lambda_evol[-1],
                z_k_bin[i * self.num_target_trees: (i + 1) * self.num_target_trees, :],
            )
            lambda_evol.append(lambda_next)

        entropy_obj = 0
        for i in range(1, steps + 1):
            entropy_future = self.entropy_target(lambda_evol[i])
            entropy_obj += ca.exp(-2 * i) * ca.logsumexp(-entropy_weight * entropy_future)

        min_sq_dist = ca.mmin(ca.sum1((X0[:2] - target_param) ** 2))
        sigmoid_factor = 1.0 / (
            1.0
            + ca.exp(
                -float(self.params["attraction_sigmoid_steepness"])
                * (min_sq_dist - float(self.params["attraction_threshold_sq_dist"]))
            )
        )

        opti.minimize(
            obj
            - float(self.params["entropy_objective_scale"]) * entropy_obj
            + attraction * sigmoid_factor
        )
        options = {"ipopt": dict(self.params["ipopt"])}
        opti.solver("ipopt", options)
        inputs = [P0, opti.x, opti.lam_g]
        outputs = [U[:, 0], X, opti.x, opti.lam_g]

        p0_val = ca.vertcat(
            x0,
            ca.reshape(target_trees, 2 * self.num_target_trees, 1),
            ca.reshape(target_lambdas, 2 * self.num_target_trees, 1),
            ca.reshape(obstacle_trees, 2 * self.num_obstacle_trees, 1),
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
