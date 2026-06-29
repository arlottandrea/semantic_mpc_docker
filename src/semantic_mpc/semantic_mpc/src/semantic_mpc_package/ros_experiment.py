import math

import numpy as np
import rospy
import tf
import tf2_ros
from geometry_msgs.msg import Point, Pose, Quaternion
from std_msgs.msg import Float32MultiArray

from semantic_mpc.srv import GetTreesPoses
from semantic_mpc_package.experiment_sampling import (
    corner_initial_pose,
    get_domain,
    random_initial_pose,
    seeded_corner_initial_pose,
)


class RosExperimentContext:
    def __init__(self, params, subscribe_scores=True, publish_cmd_pose=True):
        self.params = params
        self.latest_tree_scores = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.cmd_pose_pub = None
        if publish_cmd_pose:
            self.cmd_pose_pub = rospy.Publisher(
                params.get("cmd_pose_topic", "cmd/pose"),
                Pose,
                queue_size=int(params.get("cmd_pose_queue_size", 1)),
            )

        self.tree_scores_sub = None
        if subscribe_scores:
            self.tree_scores_sub = rospy.Subscriber(
                params.get("tree_scores_topic", "tree_scores"),
                Float32MultiArray,
                self.tree_scores_callback,
                queue_size=int(params.get("tree_scores_queue_size", 1)),
            )

    def tree_scores_callback(self, msg):
        data = np.asarray(msg.data, dtype=float)
        if data.size == 0:
            return
        if msg.layout.dim and len(msg.layout.dim) >= 2:
            rows = msg.layout.dim[0].size
            cols = msg.layout.dim[1].size
            if rows * cols != data.size:
                rospy.logwarn_throttle(5.0, "tree_scores layout mismatch: %dx%d != %d", rows, cols, data.size)
                return
            shape = (rows, cols)
        elif data.size % 2 == 0:
            shape = (data.size // 2, 2)
        else:
            rospy.logwarn_throttle(5.0, "tree_scores has odd length: %d", data.size)
            return

        try:
            self.latest_tree_scores = data.reshape(shape).copy()
        except ValueError as exc:
            rospy.logwarn_throttle(5.0, "tree_scores reshape failed: %s", exc)

    def get_tree_scores(self, column=None):
        if self.latest_tree_scores is None:
            return None
        if column is None:
            return self.latest_tree_scores.copy()
        return self.latest_tree_scores[:, column].copy()

    def wait_for_tree_scores(self, rate_s=0.05):
        while self.latest_tree_scores is None and not rospy.is_shutdown():
            rospy.sleep(rate_s)
        return self.get_tree_scores()

    def get_trees_poses_and_types(self):
        service_name = self.params.get("tree_service", "/obj_pose_srv")
        timeout = float(self.params.get("tree_service_timeout", 0.0))
        if timeout > 0.0:
            rospy.wait_for_service(service_name, timeout=timeout)
        else:
            rospy.loginfo("Waiting for tree pose service: %s", service_name)
            rospy.wait_for_service(service_name)
        response = rospy.ServiceProxy(service_name, GetTreesPoses)()
        tree_positions = np.array(
            [[pose.position.x, pose.position.y] for pose in response.trees_poses.poses],
            dtype=float,
        )
        if isinstance(response.tree_types, bytes):
            tree_types = np.array(list(response.tree_types), dtype=np.uint8)
        else:
            tree_types = np.array(response.tree_types, dtype=np.uint8)
        if len(tree_positions) != len(tree_types):
            rospy.logwarn(
                "Tree pose/type count mismatch: %d poses, %d types.",
                len(tree_positions),
                len(tree_types),
            )
        return tree_positions, tree_types

    def robot_pose(self, default=None, timeout=1.0):
        try:
            trans = self.tf_buffer.lookup_transform(
                self.params.get("map_frame", "map"),
                self.params.get("base_frame", "drone_base_link"),
                rospy.Time(),
                rospy.Duration(timeout),
            )
            _, _, yaw = tf.transformations.euler_from_quaternion(
                [
                    trans.transform.rotation.x,
                    trans.transform.rotation.y,
                    trans.transform.rotation.z,
                    trans.transform.rotation.w,
                ]
            )
            return np.array(
                [trans.transform.translation.x, trans.transform.translation.y, yaw],
                dtype=float,
            )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as exc:
            rospy.logwarn_throttle(5.0, "Failed to get robot transform: %s", exc)
            return default

    def wait_for_robot_pose(self, rate_s=0.05):
        pose = self.robot_pose(default=None)
        while pose is None and not rospy.is_shutdown():
            rospy.sleep(rate_s)
            pose = self.robot_pose(default=None)
        return pose

    def publish_pose(self, pose):
        if self.cmd_pose_pub is None:
            return
        pose = np.asarray(pose, dtype=float).flatten()
        quat = tf.transformations.quaternion_from_euler(0.0, 0.0, float(pose[2]))
        msg = Pose()
        msg.position = Point(x=float(pose[0]), y=float(pose[1]), z=0.0)
        msg.orientation = Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3])
        self.cmd_pose_pub.publish(msg)


class BeliefState:
    @staticmethod
    def bayes_binary(prior, likelihood):
        prior = np.asarray(prior, dtype=float)
        likelihood = np.asarray(likelihood, dtype=float)
        numerator = prior * likelihood
        denominator = numerator + (1.0 - prior) * (1.0 - likelihood)
        return np.divide(numerator, denominator, out=prior.copy(), where=denominator > 1e-9)

    @staticmethod
    def binary_entropy(values):
        values = np.clip(np.asarray(values, dtype=float), 1e-6, 1.0 - 1e-6)
        return float(-np.sum(values * np.log2(values) + (1.0 - values) * np.log2(1.0 - values)))

    @staticmethod
    def categorical_entropy(beliefs):
        beliefs = np.clip(np.asarray(beliefs, dtype=float), 1e-6, 1.0)
        return -np.sum(beliefs * np.log2(beliefs), axis=1)

    @staticmethod
    def categorical_entropy_sum(beliefs):
        return float(np.sum(BeliefState.categorical_entropy(beliefs)))


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))
