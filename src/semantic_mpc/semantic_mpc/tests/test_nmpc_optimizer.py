import casadi as ca
import numpy as np
import unittest

from semantic_mpc_package.nmpc_optimizer import NmpcOptimizer


class NmpcOptimizerTest(unittest.TestCase):
    def test_bayes_numpy_is_normalized_and_respects_mask(self):
        prior = np.array([[0.5, 0.5], [0.8, 0.2]])
        likelihood = np.array([[0.9, 0.1], [0.0, 0.0]])
        posterior = NmpcOptimizer.bayes_numpy(prior, likelihood, update_mask=[True, False])

        np.testing.assert_allclose(posterior[0], [0.9, 0.1])
        np.testing.assert_allclose(posterior[1], prior[1])
        np.testing.assert_allclose(np.sum(posterior, axis=1), 1.0)
        self.assertTrue(np.all(np.isfinite(posterior)))
        with self.assertRaises(ValueError):
            NmpcOptimizer.bayes_numpy([[0.5, 0.5]], [[np.nan, 0.5]])

    def test_expected_entropy_marginalizes_measurement_outcomes(self):
        optimizer = object.__new__(NmpcOptimizer)
        optimizer.entropy_target = NmpcOptimizer.entropy_f(1)
        prior = ca.MX.sym("prior", 1, 2)
        class0 = ca.MX.sym("class0", 1, 2)
        class1 = ca.MX.sym("class1", 1, 2)
        expected = optimizer.expected_posterior_entropy(prior, class0, class1)
        function = ca.Function("expected_entropy_test", [prior, class0, class1], [expected])

        uninformative = float(
            function(ca.DM([[0.5, 0.5]]), ca.DM([[0.5, 0.5]]), ca.DM([[0.5, 0.5]]))
        )
        perfect = float(
            function(ca.DM([[0.5, 0.5]]), ca.DM([[1.0, 0.0]]), ca.DM([[0.0, 1.0]]))
        )

        self.assertLess(abs(uninformative - 1.0), 1e-5)
        self.assertLess(perfect, 3e-5)

    def test_target_selection_matches_rl_and_masks_padding(self):
        trees = np.array([[5.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        beliefs = np.array(
            [
                [0.5, 0.5],
                [0.99, 0.01],
                [0.6, 0.4],
                [0.96, 0.04],
            ]
        )
        indices, mask = NmpcOptimizer.select_nearest_untracked(
            trees,
            beliefs,
            robot_position=[0.0, 0.0],
            count=5,
            confidence_threshold=0.95,
        )

        np.testing.assert_array_equal(indices[:2], [2, 0])
        np.testing.assert_array_equal(mask, [1.0, 1.0, 0.0, 0.0, 0.0])
        self.assertEqual(len(indices), 5)

    def test_optimizer_builds_and_attracts_only_outside_observation_range(self):
        batch_size = 10  # horizon 2 x five active targets
        model_input = ca.MX.sym("model_input", batch_size, 3)
        uninformative = ca.Function(
            "uninformative_test_model",
            [model_input],
            [ca.repmat(ca.DM([[0.5, 0.5]]), batch_size, 1)],
        )
        params = {
            "state_dim": 3,
            "control_dim": 3,
            "optimizer_state_dim": 6,
            "num_target_trees": 5,
            "num_obstacle_trees": 5,
            "dt": 0.25,
            "mpc_horizon": 2,
            "safe_distance": 1.5,
            "observation_range": 5.0,
            "movement_weight": 0.01,
            "yaw_movement_weight": 0.01,
            "acceleration_regularization_weight": 0.0001,
            "information_gain_weight": 1.0,
            "information_discount": 0.99,
            "attraction_weight": 0.1,
            "field_margin": 3.0,
            "max_heading_abs": 3.0 * np.pi,
            "max_velocity": 1.75,
            "max_yaw_velocity": np.pi / 4.0,
            "max_accel_xy": 1.0,
            "max_accel_yaw": np.pi / 2.0,
            "ipopt": {"print_level": 0, "sb": "yes", "max_iter": 100},
        }
        optimizer = NmpcOptimizer(params, [uninformative, uninformative])
        targets = np.array([[8.0, 0.0], [9.0, 1.0], [9.0, -1.0], [10.0, 0.0], [11.0, 0.0]])
        obstacles = np.full((5, 2), 20.0)
        result = optimizer.mpc_opt(
            targets,
            np.full((5, 2), 0.5),
            np.ones(5),
            obstacles,
            np.array([-10.0, -10.0]),
            np.array([10.0, 10.0]),
            np.zeros(6),
            steps=2,
        )
        command = np.asarray(result[1]).reshape(-1)

        self.assertTrue(np.all(np.isfinite(command)))
        self.assertGreater(command[0], 0.5)
        self.assertLessEqual(abs(command[0]), params["max_accel_xy"] + 1e-6)


if __name__ == "__main__":
    unittest.main()
