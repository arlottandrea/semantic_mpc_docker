import unittest
import sys
import types

import casadi as ca
import numpy as np


class _Dummy:
    def __init__(self, *args, **kwargs):
        pass


def _install_ros_import_stubs():
    def module(name):
        created = sys.modules.setdefault(name, types.ModuleType(name))
        return created

    rospy = module("rospy")
    rospy.INFO = 20
    for name in [
        "init_node",
        "loginfo",
        "logwarn",
        "logerr",
        "loginfo_throttle",
        "logwarn_throttle",
        "logerr_throttle",
        "sleep",
        "wait_for_service",
    ]:
        setattr(rospy, name, getattr(rospy, name, lambda *args, **kwargs: None))
    rospy.get_param = getattr(rospy, "get_param", lambda name, default=None: default)
    rospy.has_param = getattr(rospy, "has_param", lambda name: False)
    rospy.is_shutdown = getattr(rospy, "is_shutdown", lambda: False)
    rospy.Publisher = getattr(rospy, "Publisher", _Dummy)
    rospy.Subscriber = getattr(rospy, "Subscriber", _Dummy)
    rospy.ServiceProxy = getattr(rospy, "ServiceProxy", lambda *args, **kwargs: _Dummy)
    rospy.Rate = getattr(rospy, "Rate", _Dummy)
    rospy.Time = getattr(rospy, "Time", _Dummy)
    rospy.Duration = getattr(rospy, "Duration", _Dummy)

    for package in [
        "sensor_msgs",
        "std_msgs",
        "geometry_msgs",
        "nav_msgs",
        "visualization_msgs",
        "semantic_mpc",
    ]:
        module(package)
        msg_module = module(package + ".msg")
        setattr(sys.modules[package], "msg", msg_module)

    for name in ["Point", "Pose", "Quaternion", "PoseArray", "PoseStamped"]:
        setattr(sys.modules["geometry_msgs.msg"], name, _Dummy)
    setattr(sys.modules["nav_msgs.msg"], "Path", _Dummy)
    for name in ["MarkerArray", "Marker"]:
        setattr(sys.modules["visualization_msgs.msg"], name, _Dummy)
    setattr(sys.modules["std_msgs.msg"], "Float32MultiArray", _Dummy)
    srv_module = module("semantic_mpc.srv")
    setattr(sys.modules["semantic_mpc"], "srv", srv_module)
    setattr(srv_module, "GetTreesPoses", _Dummy)

    tf = module("tf")
    tf_transformations = module("tf.transformations")
    tf_transformations.quaternion_from_euler = lambda *args, **kwargs: (0.0, 0.0, 0.0, 1.0)
    tf.transformations = tf_transformations
    tf2_ros = module("tf2_ros")
    tf2_ros.Buffer = getattr(tf2_ros, "Buffer", _Dummy)
    tf2_ros.TransformListener = getattr(tf2_ros, "TransformListener", _Dummy)


_install_ros_import_stubs()

from semantic_mpc_package.nmpc import NeuralMPC
from semantic_mpc_package.nmpc_optimizer import NmpcOptimizer


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


class RecordingModel:
    def __init__(self, output_shape):
        self.output_shape = output_shape
        self.features = []

    def __call__(self, features):
        self.features.append(np.asarray(features, dtype=float))
        return ca.DM.ones(*self.output_shape)


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

    def test_measurement_model_uses_relative_mlp_yaw(self):
        mpc = object.__new__(NeuralMPC)
        mpc.params = {
            "nn_input_dim": 3,
            "nn_output_dim": 6,
            "camera_yaw_offset": 0.25,
        }
        mpc.N = 2
        mpc.num_target_trees = 2
        mpc.num_total_trees = 2
        mpc.nn_batch_size = 2
        mpc.trees_pos = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        mpc.l4c_nn = RecordingModel((2, 6))
        mpc.optimizer = NmpcOptimizer
        mpc.observation_decision_margin = 0.05

        result = mpc.measurement_likelihoods(
            [0.0, 0.0, 0.0],
            np.asarray([[0.9, 0.1], [0.1, 0.9]]),
        )

        self.assertEqual(result.shape, (2, 2))
        recorded = mpc.l4c_nn.features[0]
        np.testing.assert_allclose(recorded[:, :2], [[-1.0, 0.0], [0.0, -1.0]])
        np.testing.assert_allclose(recorded[:, 2], [-0.25, np.pi / 2.0 - 0.25])


if __name__ == "__main__":
    unittest.main()
