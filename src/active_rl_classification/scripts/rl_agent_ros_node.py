#!/usr/bin/env python3
import math
import os
import io
import zipfile

import gymnasium as gym
import numpy as np
import rospy
import scipy.stats as distrib
import tf.transformations
import tf2_ros
import torch as th
from geometry_msgs.msg import Point, Pose, Quaternion
from gymnasium import spaces
from semantic_mpc.srv import GetTreesPoses
from semantic_mpc_package.experiment_metrics import RosWandbLogger
from stable_baselines3 import PPO
from std_msgs.msg import Bool, Float32, Float32MultiArray, MultiArrayDimension

from active_rl_classification.env import BELIEF_THRESHOLD, OBS_BOUNDS
from active_rl_classification.model import TreeClassFeatureExtractor


def package_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_float32_multiarray(data):
    msg = Float32MultiArray()
    arr = np.asarray(data, dtype=np.float32)
    msg.data = arr.flatten().tolist()
    return msg


def make_vector_msg(data, labels=None):
    msg = make_float32_multiarray(data)
    arr = np.asarray(data, dtype=np.float32)
    if labels is not None and arr.ndim > 0:
        stride = int(arr.size)
        dims = []
        for axis, size in enumerate(arr.shape):
            dims.append(
                MultiArrayDimension(
                    label=labels[axis] if axis < len(labels) else "",
                    size=int(size),
                    stride=stride,
                )
            )
            stride = max(1, stride // int(size))
        msg.layout.dim = dims
    return msg


class RLAgentNode:
    """Run a trained active-classification policy against live ROS data."""

    def __init__(self):
        rospy.init_node("rl_agent_ros_node", anonymous=True)

        self.k_obs = int(rospy.get_param("~k_obs", 5))
        self.nclasses = 2
        self.obs_range = float(rospy.get_param("~obs_range", 5))
        self.step_frequency = float(rospy.get_param("~step_frequency", 4.0))
        self.entropy_target = float(rospy.get_param("~entropy_target", 0.99))
        self.deterministic = bool(rospy.get_param("~deterministic", True))
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.drone_frame = rospy.get_param("~drone_frame", "drone_base_link")
        self.tree_pose_service = rospy.get_param("~tree_pose_service", "/obj_pose_srv")
        self.measurement_topic = rospy.get_param("~measurement_topic", "tree_scores")
        self.action_topic = rospy.get_param("~action_topic", "rl/action")
        self.cmd_pose_topic = rospy.get_param("~cmd_pose_topic", "cmd/pose")
        self.done_topic = rospy.get_param("~done_topic", "rl/done")
        self.entropy_topic = rospy.get_param("~entropy_topic", "rl/entropy")
        self.obs_location_topic = rospy.get_param("~obs_location_topic", "rl/obs/location")
        self.obs_belief_topic = rospy.get_param("~obs_belief_topic", "rl/obs/belief")
        self.obs_measurement_topic = rospy.get_param(
            "~obs_measurement_topic", "rl/obs/measurement"
        )
        self.obs_tracked_topic = rospy.get_param("~obs_tracked_topic", "rl/obs/tracked")
        self.obs_mask_topic = rospy.get_param("~obs_mask_topic", "rl/obs/mask")
        self.publish_observations = bool(rospy.get_param("~publish_observations", True))
        self.action_scale = np.array([2.0, 2.0, 1.0], dtype=np.float32)
        self.delta_t = float(rospy.get_param("~delta_t", 0.25))
        self.run_index = int(rospy.get_param("~run_index", 0))

        pkg_root = package_root()
        default_model = os.path.join(pkg_root, "artifacts", "gym", "models", "final_model.zip")
        model_path = rospy.get_param("~model_path", rospy.get_param("~policy_path", default_model))
        device = rospy.get_param("~device", "auto")
        self.model = self._load_ppo_policy(model_path, device)

        self.max_entropy = distrib.entropy(
            [1.0 / self.nclasses] * self.nclasses, base=2
        )
        self.uniform_proba = (
            np.ones(self.nclasses, dtype=np.float32) / float(self.nclasses)
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.action_pub = rospy.Publisher(self.action_topic, Float32MultiArray, queue_size=1)
        self.cmd_pose_pub = rospy.Publisher(self.cmd_pose_topic, Pose, queue_size=1)
        self.done_pub = rospy.Publisher(self.done_topic, Bool, queue_size=1)
        self.entropy_pub = rospy.Publisher(self.entropy_topic, Float32, queue_size=1)
        if self.publish_observations:
            self.obs_location_pub = rospy.Publisher(
                self.obs_location_topic, Float32MultiArray, queue_size=1
            )
            self.obs_belief_pub = rospy.Publisher(
                self.obs_belief_topic, Float32MultiArray, queue_size=1
            )
            self.obs_measurement_pub = rospy.Publisher(
                self.obs_measurement_topic, Float32MultiArray, queue_size=1
            )
            self.obs_tracked_pub = rospy.Publisher(
                self.obs_tracked_topic, Float32MultiArray, queue_size=1
            )
            self.obs_mask_pub = rospy.Publisher(
                self.obs_mask_topic, Float32MultiArray, queue_size=1
            )

        self.measurement_sub = rospy.Subscriber(
            self.measurement_topic, Float32MultiArray, self._on_measurements, queue_size=1
        )

        self.trees = None
        self.tree_classes = None
        self.drone = None
        self.beliefs = None
        self.tracked = None
        self.ntargets = 0
        self._observations = None
        self._latest_measurements = None
        self._prev_entropy = 0.0
        self._initial_entropy = 0.0
        self._prev_position = None
        self.total_distance = 0.0
        self.steps = 0
        self.metrics = RosWandbLogger(
            "rl_agent",
            self.run_index,
            self._wandb_params(model_path, device),
            default_project="active_rl_classification",
        )

    def _wandb_params(self, model_path, device):
        return {
            "wandb_project": rospy.get_param("~wandb_project", "active_rl_classification"),
            "wandb_entity": rospy.get_param("~wandb_entity", ""),
            "wandb_mode": rospy.get_param("~wandb_mode", "offline"),
            "wandb_name": rospy.get_param("~wandb_name", "rl_agent_run_{:03d}".format(self.run_index)),
            "wandb_log_period": float(rospy.get_param("~wandb_log_period", 1.0)),
            "run_dir": rospy.get_param(
                "~run_dir",
                os.path.join(package_root(), "artifacts", "ros", "wandb"),
            ),
            "k_obs": self.k_obs,
            "obs_range": self.obs_range,
            "step_frequency": self.step_frequency,
            "entropy_target": self.entropy_target,
            "deterministic": self.deterministic,
            "model_path": model_path,
            "device": device,
            "map_frame": self.map_frame,
            "drone_frame": self.drone_frame,
            "tree_pose_service": self.tree_pose_service,
            "measurement_topic": self.measurement_topic,
            "action_topic": self.action_topic,
            "cmd_pose_topic": self.cmd_pose_topic,
        }

    def _load_ppo_policy(self, model_path, device):
        """Load policy weights without unpickling SB3 metadata.

        Some saved SB3 zip metadata references numpy._core, which is not
        available with the NumPy version pinned by ROS1. The raw PyTorch policy
        state dict does not need that metadata, so build the matching policy
        architecture here and load policy.pth directly.
        """
        obs_space = spaces.Dict(
            {
                "location": spaces.Box(-np.inf, np.inf, shape=(self.k_obs, 3), dtype=np.float32),
                "belief": spaces.Box(0.0, 1.0 + 1e-10, shape=(self.k_obs, 1), dtype=np.float32),
                "measurement": spaces.Box(0.0, 1.0 + 1e-10, shape=(self.k_obs, 1), dtype=np.float32),
                "tracked": spaces.Box(0.0, 1.0, shape=(self.k_obs, 1), dtype=np.float32),
                "mask": spaces.Box(0.0, 1.0, shape=(self.k_obs,), dtype=np.float32),
            }
        )
        action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

        class DummyEnv(gym.Env):
            metadata = {}

            def __init__(self):
                super().__init__()
                self.observation_space = obs_space
                self.action_space = action_space

            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                return self.observation_space.sample(), {}

            def step(self, action):
                return self.observation_space.sample(), 0.0, False, False, {}

        policy_kwargs = dict(
            features_extractor_class=TreeClassFeatureExtractor,
            features_extractor_kwargs=dict(features_dim=128),
            share_features_extractor=False,
            net_arch=dict(pi=[], vf=[]),
        )

        model = PPO(
            "MultiInputPolicy",
            DummyEnv(),
            policy_kwargs=policy_kwargs,
            verbose=0,
            device=device,
        )

        map_location = "cpu" if device == "auto" else device
        with zipfile.ZipFile(model_path, "r") as archive:
            policy_bytes = archive.read("policy.pth")
        state_dict = th.load(io.BytesIO(policy_bytes), map_location=map_location)
        model.policy.load_state_dict(state_dict)
        model.policy.eval()
        rospy.loginfo("Loaded PPO policy weights from %s", model_path)
        return model

    def _on_measurements(self, msg):
        data = np.asarray(msg.data, dtype=np.float32)
        if data.size == 0:
            return

        cols = 2
        if msg.layout.dim and len(msg.layout.dim) >= 2:
            cols = int(msg.layout.dim[1].size)
        if cols != self.nclasses or data.size % self.nclasses != 0:
            rospy.logwarn_throttle(
                2.0,
                "Ignoring measurement with %d values; expected N x %d",
                data.size,
                self.nclasses,
            )
            return

        measurements = data.reshape(-1, self.nclasses)
        measurements = np.clip(measurements, 1e-6, 1.0)
        measurements /= np.sum(measurements, axis=1, keepdims=True)
        self._latest_measurements = measurements.astype(np.float32)

    def _get_tree_poses_and_types(self):
        rospy.wait_for_service(self.tree_pose_service)
        get_trees = rospy.ServiceProxy(self.tree_pose_service, GetTreesPoses)
        response = get_trees()
        trees = np.array(
            [[pose.position.x, pose.position.y] for pose in response.trees_poses.poses],
            dtype=np.float32,
        )

        if isinstance(response.tree_types, bytes):
            tree_classes = np.frombuffer(response.tree_types, dtype=np.uint8).astype(np.int64)
        else:
            tree_classes = np.asarray(list(response.tree_types), dtype=np.int64)

        if tree_classes.size != trees.shape[0]:
            rospy.logwarn(
                "Tree type count (%d) does not match tree pose count (%d); using zeros.",
                tree_classes.size,
                trees.shape[0],
            )
            tree_classes = np.zeros(trees.shape[0], dtype=np.int64)

        return trees, tree_classes

    def _read_drone_pose(self):
        trans = self.tf_buffer.lookup_transform(
            self.map_frame, self.drone_frame, rospy.Time(), rospy.Duration(1.0)
        )
        quat = [
            trans.transform.rotation.x,
            trans.transform.rotation.y,
            trans.transform.rotation.z,
            trans.transform.rotation.w,
        ]
        _, _, yaw = tf.transformations.euler_from_quaternion(quat)
        self.drone = np.array(
            [
                trans.transform.translation.x,
                trans.transform.translation.y,
                math.degrees(yaw) % 360.0,
            ],
            dtype=np.float32,
        )
        return self.drone

    def reset(self):
        self.trees, self.tree_classes = self._get_tree_poses_and_types()
        self.ntargets = int(self.trees.shape[0])
        if self.ntargets == 0:
            raise RuntimeError("No trees returned by {}".format(self.tree_pose_service))

        self._read_drone_pose()
        self.beliefs = self.uniform_proba[None, :].repeat(self.ntargets, axis=0)
        self.tracked = np.max(self.beliefs, axis=1) >= BELIEF_THRESHOLD
        self._observations = self.uniform_proba[None, :].repeat(self.ntargets, axis=0)
        self._prev_entropy = self.total_entropy(include_tracked=False)
        self._initial_entropy = self.ntargets * self.max_entropy
        self._prev_position = self.drone[:2].copy()
        self.total_distance = 0.0
        self.steps = 0
        self.metrics.start()
        obs = self._build_obs()
        self._publish_obs(obs)
        self.metrics.log(
            {
                "time_execution_s": 0.0,
                "entropy": self._prev_entropy,
                "entropy_initial": self._initial_entropy,
                "entropy_stop_value": self.entropy_stop_value,
                "num_tracked": int(np.sum(self.tracked)),
                "total_targets": self.ntargets,
                "distance_m": self.total_distance,
                "pose/x": float(self.drone[0]),
                "pose/y": float(self.drone[1]),
                "pose/theta": float(self.drone[2]),
            }
        )
        rospy.loginfo(
            "RL agent reset with %d trees. Initial entropy %.3f; stop at %.3f.",
            self.ntargets,
            self._initial_entropy,
            self.entropy_stop_value,
        )
        return obs

    @property
    def entropy_stop_value(self):
        return max(0.0, (1.0 - self.entropy_target) * self._initial_entropy)

    def total_entropy(self, include_tracked=True):
        if self.beliefs is None:
            return float("inf")
        indices = range(self.ntargets) if include_tracked else [
            t for t in range(self.ntargets) if not self.tracked[t]
        ]
        return float(sum(distrib.entropy(self.beliefs[t], base=2) for t in indices))

    def _relative_pose(self, tree_idx):
        c = np.cos(np.radians(self.drone[2]))
        s = np.sin(np.radians(self.drone[2]))
        rot_t = np.array([[c, s], [-s, c]], dtype=np.float32)
        rel_xy = rot_t @ (self.trees[tree_idx] - self.drone[:2])
        rel_bearing = np.arctan2(rel_xy[1], rel_xy[0])
        return np.array([rel_xy[0], rel_xy[1], rel_bearing], dtype=np.float32)

    def _build_obs(self):
        untracked = [t for t in range(self.ntargets) if not self.tracked[t]]
        if untracked:
            distances = np.array(
                [np.linalg.norm(self.drone[:2] - self.trees[t]) for t in untracked]
            )
            sorted_untracked = [untracked[i] for i in np.argsort(distances)][: self.k_obs]
        else:
            sorted_untracked = []

        location = np.zeros((self.k_obs, 3), dtype=np.float32)
        belief = np.zeros((self.k_obs, 1), dtype=np.float32)
        measurement = np.zeros((self.k_obs, 1), dtype=np.float32)
        tracked_obs = np.zeros((self.k_obs, 1), dtype=np.float32)
        mask = np.zeros(self.k_obs, dtype=np.float32)

        for i, t in enumerate(sorted_untracked):
            rel = self._relative_pose(t)
            rel[:2] = np.clip(rel[:2], -OBS_BOUNDS, OBS_BOUNDS)
            location[i] = rel
            belief[i, 0] = np.float32(
                1.0 - distrib.entropy(self.beliefs[t], base=2) / self.max_entropy + 1e-10
            )
            measurement[i, 0] = np.float32(
                1.0
                - distrib.entropy(self._observations[t], base=2) / self.max_entropy
                + 1e-10
            )
            tracked_obs[i, 0] = 0.0
            mask[i] = 1.0

        return {
            "location": location,
            "belief": belief,
            "measurement": measurement,
            "tracked": tracked_obs,
            "mask": mask,
        }

    def _publish_obs(self, obs):
        if not self.publish_observations:
            return
        self.obs_location_pub.publish(make_vector_msg(obs["location"], ["target", "pose"]))
        self.obs_belief_pub.publish(make_vector_msg(obs["belief"], ["target", "belief"]))
        self.obs_measurement_pub.publish(make_vector_msg(obs["measurement"], ["target", "measurement"]))
        self.obs_tracked_pub.publish(make_vector_msg(obs["tracked"], ["target", "tracked"]))
        self.obs_mask_pub.publish(make_vector_msg(obs["mask"], ["target"]))

    def _predict_action(self, obs):
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        return np.asarray(action, dtype=np.float32).reshape(3)

    def _publish_step(self, action):
        self.action_pub.publish(make_float32_multiarray(action))

        applied = action * self.action_scale * self.delta_t
        heading = self.drone[2]
        c = np.cos(np.radians(heading))
        s = np.sin(np.radians(heading))
        world_delta = np.array([[c, -s], [s, c]], dtype=np.float32) @ applied[:2]
        target_xy = self.drone[:2] + world_delta
        target_heading_deg = (heading + 60.0 * applied[2]) % 360.0
        q = tf.transformations.quaternion_from_euler(0.0, 0.0, math.radians(target_heading_deg))

        pose = Pose()
        pose.position = Point(x=float(target_xy[0]), y=float(target_xy[1]), z=0.0)
        pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self.cmd_pose_pub.publish(pose)

    def _update_beliefs_from_measurements(self):
        if self._latest_measurements is None:
            return 0

        n = min(self.ntargets, self._latest_measurements.shape[0])
        new_targets_tracked = 0
        for t in range(n):
            dist = np.linalg.norm(self.drone[:2] - self.trees[t])
            if dist > self.obs_range:
                continue

            self._observations[t] = self._latest_measurements[t]
            if self.tracked[t]:
                continue

            new_belief = self.beliefs[t] * self._observations[t]
            new_belief /= np.sum(new_belief) + 1e-32
            self.beliefs[t] = new_belief
            if np.max(new_belief) >= BELIEF_THRESHOLD:
                self.tracked[t] = True
                new_targets_tracked += 1

        return new_targets_tracked

    def step(self):
        self.steps += 1
        self._read_drone_pose()
        if self._prev_position is not None:
            self.total_distance += float(np.linalg.norm(self.drone[:2] - self._prev_position))
        self._prev_position = self.drone[:2].copy()
        new_targets_tracked = self._update_beliefs_from_measurements()
        obs = self._build_obs()
        action = self._predict_action(obs)
        self._publish_step(action)
        self._publish_obs(obs)

        curr_entropy = self.total_entropy(include_tracked=False)
        self._prev_entropy = curr_entropy
        self.entropy_pub.publish(Float32(data=curr_entropy))

        info = {
            "entropy": curr_entropy,
            "num_tracked": int(np.sum(self.tracked)),
            "new_targets_tracked": int(new_targets_tracked),
            "action": action,
        }
        self._log_step(info)
        return obs, action, info

    def _log_step(self, info):
        if not self.metrics.should_log():
            return

        action = np.asarray(info["action"], dtype=float).flatten()
        self.metrics.log(
            {
                "time_execution_s": self.metrics.elapsed(),
                "step": self.steps,
                "distance_m": self.total_distance,
                "entropy": float(info["entropy"]),
                "entropy_reduction": float(self._initial_entropy - info["entropy"]),
                "num_tracked": int(info["num_tracked"]),
                "new_targets_tracked": int(info["new_targets_tracked"]),
                "total_targets": self.ntargets,
                "pose/x": float(self.drone[0]),
                "pose/y": float(self.drone[1]),
                "pose/theta": float(self.drone[2]),
                "action/forward": float(action[0]),
                "action/lateral": float(action[1]),
                "action/yaw_rate": float(action[2]),
                "belief_mean": float(np.mean(self.beliefs)) if self.beliefs is not None else 0.0,
            }
        )

    def run(self):
        self.reset()
        rate = rospy.Rate(self.step_frequency)
        while (
            not rospy.is_shutdown()
            and self.total_entropy(include_tracked=False) > self.entropy_stop_value
        ):
            try:
                _, _, info = self.step()
                rospy.loginfo_throttle(
                    2.0,
                    "RL step %d | entropy %.3f/%.3f | tracked %d/%d",
                    self.steps,
                    info["entropy"],
                    self._initial_entropy,
                    info["num_tracked"],
                    self.ntargets,
                )
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ) as exc:
                rospy.logwarn_throttle(2.0, "Waiting for drone TF: %s", exc)
            rate.sleep()

        self.done_pub.publish(Bool(data=True))
        final_entropy = self.total_entropy(include_tracked=False)
        summary = self.metrics.finish(
            {
                "total_time_execution_s": self.metrics.elapsed(),
                "total_distance_m": self.total_distance,
                "total_steps": self.steps,
                "initial_entropy": self._initial_entropy,
                "final_entropy": final_entropy,
                "entropy_reduction": self._initial_entropy - final_entropy,
                "num_tracked_final": int(np.sum(self.tracked)) if self.tracked is not None else 0,
                "total_targets": self.ntargets,
            }
        )
        rospy.loginfo(
            "RL run complete after %d steps. Final entropy %.3f/%.3f.",
            self.steps,
            summary["final_entropy"],
            self._initial_entropy,
        )


if __name__ == "__main__":
    try:
        RLAgentNode().run()
    except rospy.ROSInterruptException:
        pass
