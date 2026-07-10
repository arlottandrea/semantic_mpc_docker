import unittest

import casadi as ca
import numpy as np

from semantic_mpc_package.nmpc import NeuralMPC


class ShapeCheckingModel:
    def __init__(self, expected_shape):
        self.expected_shape = expected_shape
        self.calls = 0

    def __call__(self, features):
        shape = (features.size1(), features.size2())
        if shape != self.expected_shape:
            raise RuntimeError("unexpected model input shape {}".format(shape))
        self.calls += 1
        return ca.DM.ones(self.expected_shape[0], 6)


class NeuralMPCMeasurementTest(unittest.TestCase):
    def test_measurement_model_uses_fixed_size_batches(self):
        mpc = object.__new__(NeuralMPC)
        mpc.params = {"nn_input_dim": 3, "nn_output_dim": 6}
        mpc.N = 5
        mpc.num_target_trees = 1
        mpc.nn_batch_size = 5
        mpc.l4c_nn = ShapeCheckingModel((5, 3))

        features = np.arange(75, dtype=float).reshape(25, 3)
        result = mpc._evaluate_l4c_nn_fixed_batches(features)

        self.assertEqual(mpc.l4c_nn.calls, 5)
        self.assertEqual(result.shape, (25, 6))

    def test_measurement_model_pads_final_batch(self):
        mpc = object.__new__(NeuralMPC)
        mpc.params = {"nn_input_dim": 3, "nn_output_dim": 6}
        mpc.N = 5
        mpc.num_target_trees = 1
        mpc.nn_batch_size = 5
        mpc.l4c_nn = ShapeCheckingModel((5, 3))

        features = np.arange(21, dtype=float).reshape(7, 3)
        result = mpc._evaluate_l4c_nn_fixed_batches(features)

        self.assertEqual(mpc.l4c_nn.calls, 2)
        self.assertEqual(result.shape, (7, 6))

    def test_command_reference_advances_when_solve_exceeds_dt(self):
        mpc = object.__new__(NeuralMPC)
        mpc.dt = 0.25
        mpc.nx = 3
        x_traj = ca.DM(
            [
                [0.0, 1.0, 2.0, 3.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.1, 0.2, 0.3],
            ]
        )

        fast = np.asarray(mpc._command_reference_from_trajectory(x_traj, 0.10)).reshape(-1)
        slow = np.asarray(mpc._command_reference_from_trajectory(x_traj, 0.34)).reshape(-1)

        np.testing.assert_allclose(fast, [1.0, 0.0, 0.1])
        np.testing.assert_allclose(slow, [2.0, 0.0, 0.2])

    def test_command_reference_clamps_to_horizon(self):
        mpc = object.__new__(NeuralMPC)
        mpc.dt = 0.25
        mpc.nx = 3
        x_traj = ca.DM(
            [
                [0.0, 1.0, 2.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.1, 0.2],
            ]
        )

        result = np.asarray(mpc._command_reference_from_trajectory(x_traj, 5.0)).reshape(-1)

        np.testing.assert_allclose(result, [2.0, 0.0, 0.2])


if __name__ == "__main__":
    unittest.main()
