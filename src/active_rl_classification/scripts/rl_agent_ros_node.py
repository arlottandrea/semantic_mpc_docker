#!/usr/bin/env python3
import math
import os
import io
import time
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
from semantic_mpc_package.experiment_metrics import (
    PerTreeEntropyMetrics,
    RosWandbLogger,
    VelocityTreeMetrics,
)
from semantic_mpc_package.ros_experiment import normalize_angle, seeded_corner_initial_pose
from semantic_mpc_package.baselines import resolve_mower_heading
from stable_baselines3 import PPO
from std_msgs.msg import Bool, Float32, Float32MultiArray, MultiArrayDimension

from active_rl_classification.env import OBS_BOUNDS
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

        self.k_obs = int(rospy.get_param("~active_target_count", rospy.get_param("~k_obs", 5)))
        self.nclasses = 2
        self.obs_range = float(
            rospy.get_param("~observation_range", rospy.get_param("~obs_range", 5.0))
        )
        self.belief_tracking_threshold = float(
            rospy.get_param("~belief_tracking_threshold", 0.9975245006578829)
        )
        self.max_experiment_steps = int(rospy.get_param("~max_experiment_steps", 1200))
        self.active_obstacle_count = max(1, int(rospy.get_param("~active_obstacle_count", 5)))
        self.delta_t = float(rospy.get_param("~dt", rospy.get_param("~delta_t", 0.25)))
        self.step_frequency = float(rospy.get_param("~step_frequency", 1.0 / max(self.delta_t, 1e-6)))
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
        self.max_velocity = float(rospy.get_param("~max_velocity", 1.75))
        self.max_yaw_velocity = float(rospy.get_param("~max_yaw_velocity", math.pi / 4.0))
        self.max_lin_accel = float(rospy.get_param("~max_lin_accel", 1.0))
        self.max_yaw_accel = float(rospy.get_param("~max_yaw_accel", math.pi / 2.0))
        self.safe_distance = float(rospy.get_param("~safe_distance", 1.5))
        self.command_filter_enabled = bool(rospy.get_param("~rl_command_filter_enabled", True))
        self.measurement_period = float(rospy.get_param("~measurement_period", self.delta_t))
        self.tree_velocity_radius = float(rospy.get_param("~tree_velocity_radius", 5.0))
        self.tree_entropy_threshold = float(rospy.get_param("~tree_entropy_threshold", 0.025))
        self.tree_entropy_start_epsilon = float(
            rospy.get_param("~tree_entropy_start_epsilon", 1e-4)
        )
        self.num_runs = int(rospy.get_param("~num_runs", 1))
        self.run_index_offset = int(
            rospy.get_param("~run_index_offset", rospy.get_param("~run_index", 0))
        )
        self.base_seed = int(rospy.get_param("~seed", 1))
        self.random_initial_state = bool(rospy.get_param("~random_initial_state", True))
        self.start_pose = rospy.get_param("~start_pose", None)
        if self.start_pose in ("", [], None):
            self.start_pose = None
        self.initial_corner_margin = float(rospy.get_param("~initial_corner_margin", 1.5))
        self.mower_heading = rospy.get_param("~mower_heading", "N")
        self.mower_heading_random = bool(rospy.get_param("~mower_heading_random", False))
        self.initial_pose_timeout = float(rospy.get_param("~initial_pose_timeout", 15.0))
        self.initial_pose_tolerance = float(rospy.get_param("~initial_pose_tolerance", 0.15))
        self.initial_heading_tolerance = float(rospy.get_param("~initial_heading_tolerance", 0.2))
        self.initial_pose_publish_period = float(rospy.get_param("~initial_pose_publish_period", 0.1))
        self.run_index = self.run_index_offset
        self.trial_seed = self.base_seed + self.run_index

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

        self._last_measurement_time = None
        self._measurement_sequence = 0
        self._last_used_measurement_sequence = 0
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
        self._prev_world_velocity = np.zeros(2, dtype=np.float32)
        self._prev_yaw_velocity = 0.0
        self._last_command = np.zeros(3, dtype=np.float32)
        self.velocity_metrics = None
        self.entropy_metrics = None
        self._last_velocity_metrics = {}
        self.policy_inference_times_ms = []
        self.controller_compute_times_ms = []
        self.model_path = model_path
        self.model_device = device
        self.metrics = None

    def _wandb_params(self, model_path, device):
        run_name = rospy.get_param("~wandb_name", "") or "rl_agent_run_{:03d}".format(self.run_index)
        run_root = rospy.get_param("~run_dir", "/runs/rl")
        return {
            "wandb_enabled": bool(rospy.get_param("~wandb_enabled", False)),
            "wandb_project": rospy.get_param("~wandb_project", "active_rl_classification"),
            "wandb_entity": rospy.get_param("~wandb_entity", ""),
            "wandb_mode": rospy.get_param("~wandb_mode", "offline"),
            "wandb_name": run_name,
            "wandb_log_every_steps": int(rospy.get_param("~wandb_log_every_steps", 4)),
            "run_dir": os.path.join(run_root, run_name),
            "run_index": self.run_index,
            "trial_seed": self.trial_seed,
            "base_seed": self.base_seed,
            "mower_heading": self.mower_heading,
            "mower_heading_random": self.mower_heading_random,
            "k_obs": self.k_obs,
            "obs_range": self.obs_range,
            "observation_range": self.obs_range,
            "belief_tracking_threshold": self.belief_tracking_threshold,
            "active_target_count": self.k_obs,
            "max_experiment_steps": self.max_experiment_steps,
            "active_obstacle_count": self.active_obstacle_count,
            "step_frequency": self.step_frequency,
            "control_period": self.delta_t,
            "measurement_period": self.measurement_period,
            "max_velocity": self.max_velocity,
            "max_yaw_velocity": self.max_yaw_velocity,
            "max_lin_accel": self.max_lin_accel,
            "max_yaw_accel": self.max_yaw_accel,
            "safe_distance": self.safe_distance,
            "rl_command_filter_enabled": self.command_filter_enabled,
            "tree_velocity_radius": self.tree_velocity_radius,
            "tree_entropy_threshold": self.tree_entropy_threshold,
            "tree_entropy_start_epsilon": self.tree_entropy_start_epsilon,
            "deterministic": self.deterministic,
            "termination_criterion": "all_trees_belief_confidence_threshold",
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
        if not np.all(np.isfinite(measurements)):
            rospy.logwarn_throttle(2.0, "Ignoring non-finite tree-score measurement.")
            return
        measurements = np.clip(measurements, 1e-6, 1.0)
        measurements /= np.sum(measurements, axis=1, keepdims=True)
        self._latest_measurements = measurements.astype(np.float32)
        self._last_measurement_time = time.monotonic()
        self._measurement_sequence += 1

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

    def _resolve_initial_pose(self):
        if self.start_pose is not None:
            return np.asarray(self.start_pose, dtype=float).flatten()[:3]
        if not self.random_initial_state:
            return None
        heading_direction = None if self.mower_heading_random else self.mower_heading
        initial_heading = resolve_mower_heading(heading_direction, seed=self.trial_seed)
        return seeded_corner_initial_pose(
            self.trees,
            self.trial_seed,
            margin=self.initial_corner_margin,
            heading=initial_heading,
        )

    def _publish_pose(self, pose_values):
        pose_values = np.asarray(pose_values, dtype=float).flatten()[:3]
        q = tf.transformations.quaternion_from_euler(0.0, 0.0, float(pose_values[2]))
        pose = Pose()
        pose.position = Point(x=float(pose_values[0]), y=float(pose_values[1]), z=0.0)
        pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self.cmd_pose_pub.publish(pose)

    def _apply_initial_pose(self, initial_pose):
        initial_pose = np.asarray(initial_pose, dtype=float).flatten()[:3]
        timeout = max(0.1, self.initial_pose_timeout)
        publish_period = max(0.02, self.initial_pose_publish_period)
        deadline = time.time() + timeout
        rospy.loginfo(
            "RL trial %d seed %d: resetting to [%.3f, %.3f, %.3f].",
            self.run_index,
            self.trial_seed,
            initial_pose[0],
            initial_pose[1],
            initial_pose[2],
        )

        while not rospy.is_shutdown() and time.time() < deadline:
            self._publish_pose(initial_pose)
            try:
                current = self._read_drone_pose().copy()
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ):
                rospy.sleep(publish_period)
                continue

            position_error = np.linalg.norm(current[:2] - initial_pose[:2])
            heading_error = abs(
                normalize_angle(math.radians(float(current[2])) - float(initial_pose[2]))
            )
            if (
                position_error <= self.initial_pose_tolerance
                and heading_error <= self.initial_heading_tolerance
            ):
                rospy.loginfo(
                    "RL initial pose confirmed (position error %.3fm, heading error %.3frad).",
                    position_error,
                    heading_error,
                )
                return
            rospy.sleep(publish_period)

        raise RuntimeError(
            "RL trial {} seed {} did not reach its initial pose within {:.1f}s.".format(
                self.run_index, self.trial_seed, timeout
            )
        )

    def reset(self):
        self.trees, self.tree_classes = self._get_tree_poses_and_types()
        self.ntargets = int(self.trees.shape[0])
        if self.ntargets == 0:
            raise RuntimeError("No trees returned by {}".format(self.tree_pose_service))

        initial_pose = self._resolve_initial_pose()
        if initial_pose is not None:
            self._apply_initial_pose(initial_pose)
        else:
            self._read_drone_pose()
        self.beliefs = self.uniform_proba[None, :].repeat(self.ntargets, axis=0)
        self.tracked = np.max(self.beliefs, axis=1) >= self.belief_tracking_threshold
        self._observations = self.uniform_proba[None, :].repeat(self.ntargets, axis=0)
        self._latest_measurements = None
        self._last_measurement_time = None
        self._last_used_measurement_sequence = self._measurement_sequence
        self._prev_entropy = self.total_entropy(include_tracked=False)
        self._initial_entropy = self.ntargets * self.max_entropy
        self._prev_position = self.drone[:2].copy()
        self.total_distance = 0.0
        self.steps = 0
        self._prev_world_velocity = np.zeros(2, dtype=np.float32)
        self._prev_yaw_velocity = 0.0
        self._last_command = np.zeros(3, dtype=np.float32)
        self.velocity_metrics = VelocityTreeMetrics(self.max_velocity, self.tree_velocity_radius)
        self._last_velocity_metrics = self.velocity_metrics.update(0.0, self.drone, self.trees)
        self.entropy_metrics = PerTreeEntropyMetrics(
            self.tree_entropy_threshold,
            self.tree_entropy_start_epsilon,
        )
        self.entropy_metrics.update(0.0, self.beliefs)
        self.policy_inference_times_ms = []
        self.controller_compute_times_ms = []
        self.done_pub.publish(Bool(data=False))
        self.metrics.start()
        obs = self._build_obs()
        self._publish_obs(obs)
        self.metrics.log(
            {
                "time_execution_s": 0.0,
                "entropy": self._prev_entropy,
                "entropy_initial": self._initial_entropy,
                "num_tracked": int(np.sum(self.tracked)),
                "total_targets": self.ntargets,
                "distance_m": self.total_distance,
                "pose/x": float(self.drone[0]),
                "pose/y": float(self.drone[1]),
                "pose/theta": float(self.drone[2]),
            }
        )
        rospy.loginfo(
            "RL agent reset with %d trees. Initial entropy %.3f; belief threshold %.3f.",
            self.ntargets,
            self._initial_entropy,
            self.belief_tracking_threshold,
        )
        return obs

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
        inference_start = time.perf_counter()
        action, _ = self.model.predict(obs, deterministic=self.deterministic)
        self.policy_inference_times_ms.append((time.perf_counter() - inference_start) * 1000.0)
        return np.asarray(action, dtype=np.float32).reshape(3)

    def _measurement_age_s(self):
        if self._last_measurement_time is None:
            return float("inf")
        return max(0.0, time.monotonic() - self._last_measurement_time)

    def _publish_step(self, action):
        self.action_pub.publish(make_float32_multiarray(action))
        command_compute_start = time.perf_counter()

        local_velocity = np.asarray(action[:2], dtype=np.float32) * self.max_velocity
        local_speed = np.linalg.norm(local_velocity)
        if local_speed > self.max_velocity:
            local_velocity *= self.max_velocity / local_speed
        heading = self.drone[2]
        c = np.cos(np.radians(heading))
        s = np.sin(np.radians(heading))
        world_velocity = np.array([[c, -s], [s, c]], dtype=np.float32) @ local_velocity
        if self.command_filter_enabled:
            max_delta_velocity = self.max_lin_accel * self.delta_t
            world_velocity = self._prev_world_velocity + np.clip(
                world_velocity - self._prev_world_velocity,
                -max_delta_velocity,
                max_delta_velocity,
            )
            world_velocity = self._apply_obstacle_safety_filter(world_velocity)

        yaw_velocity = float(np.clip(action[2], -1.0, 1.0)) * self.max_yaw_velocity
        if self.command_filter_enabled:
            max_delta_yaw = self.max_yaw_accel * self.delta_t
            yaw_velocity = self._prev_yaw_velocity + float(
                np.clip(yaw_velocity - self._prev_yaw_velocity, -max_delta_yaw, max_delta_yaw)
            )
        target_xy = self.drone[:2] + world_velocity * self.delta_t
        target_heading_deg = (heading + math.degrees(yaw_velocity * self.delta_t)) % 360.0
        q = tf.transformations.quaternion_from_euler(0.0, 0.0, math.radians(target_heading_deg))

        pose = Pose()
        pose.position = Point(x=float(target_xy[0]), y=float(target_xy[1]), z=0.0)
        pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self._prev_world_velocity = world_velocity.astype(np.float32)
        self._prev_yaw_velocity = yaw_velocity
        self._last_command = np.array([world_velocity[0], world_velocity[1], yaw_velocity], dtype=np.float32)
        command_compute_time_ms = (time.perf_counter() - command_compute_start) * 1000.0
        self.cmd_pose_pub.publish(pose)
        return command_compute_time_ms

    def _apply_obstacle_safety_filter(self, world_velocity):
        """Project the RL direction onto safe, tangential tree-bypass directions."""
        velocity = np.asarray(world_velocity, dtype=np.float32).copy()
        lookahead = self.safe_distance + self.max_velocity * self.delta_t
        obstacle_indices = np.argsort(
            np.linalg.norm(self.trees - self.drone[:2], axis=1)
        )[: self.active_obstacle_count]
        for tree_id in obstacle_indices:
            tree = self.trees[tree_id]
            radial = self.drone[:2] - tree
            distance = float(np.linalg.norm(radial))
            if distance < 1e-6 or distance > lookahead:
                continue
            radial /= distance
            desired_speed = float(np.linalg.norm(velocity))
            outward_speed = float(np.dot(velocity, radial))
            minimum_outward_speed = (self.safe_distance - distance) / max(self.delta_t, 1e-6)
            if outward_speed >= minimum_outward_speed:
                continue

            # Keep the feasible radial component and use the closest tangent to
            # continue around the obstacle instead of stopping on a head-on path.
            radial_speed = float(
                np.clip(max(outward_speed, minimum_outward_speed), -self.max_velocity, self.max_velocity)
            )
            tangent = np.array([-radial[1], radial[0]], dtype=np.float32)
            tangent_alignment = float(np.dot(world_velocity, tangent))
            if tangent_alignment < -1e-9 or (
                abs(tangent_alignment) <= 1e-9 and (self.trial_seed + tree_id) % 2
            ):
                tangent = -tangent
            proximity = np.clip(
                (lookahead - distance) / max(lookahead - self.safe_distance, 1e-6),
                0.0,
                1.0,
            )
            current_tangent = float(np.dot(velocity, tangent))
            target_tangent = max(abs(current_tangent), desired_speed * float(proximity))
            max_tangent = math.sqrt(max(0.0, self.max_velocity ** 2 - radial_speed ** 2))
            velocity = radial_speed * radial + min(target_tangent, max_tangent) * tangent

        # The bypass itself must respect the same acceleration envelope as the
        # policy command. Safety takes priority only if the current state is
        # already too close for a feasible acceleration-limited correction.
        max_delta_velocity = self.max_lin_accel * self.delta_t
        limited = self._prev_world_velocity + np.clip(
            velocity - self._prev_world_velocity,
            -max_delta_velocity,
            max_delta_velocity,
        )
        speed = float(np.linalg.norm(limited))
        if speed > self.max_velocity:
            limited *= self.max_velocity / speed
        emergency_correction = False
        for tree_id in obstacle_indices:
            tree = self.trees[tree_id]
            radial = self.drone[:2] - tree
            distance = float(np.linalg.norm(radial))
            if distance < 1e-6 or distance > lookahead:
                continue
            radial /= distance
            minimum_outward_speed = (self.safe_distance - distance) / max(self.delta_t, 1e-6)
            shortfall = minimum_outward_speed - float(np.dot(limited, radial))
            if shortfall > 1e-6:
                limited += shortfall * radial
                emergency_correction = True
        if emergency_correction:
            rospy.logwarn_throttle(
                2.0,
                "RL safety filter required an emergency correction beyond the nominal acceleration envelope.",
            )
        return limited.astype(np.float32)

    def _update_beliefs_from_measurements(self):
        if (
            self._latest_measurements is None
            or self._measurement_sequence <= self._last_used_measurement_sequence
        ):
            return 0
        self._last_used_measurement_sequence = self._measurement_sequence

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
            if np.max(new_belief) >= self.belief_tracking_threshold:
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
        policy_action = self._predict_action(obs)
        action = policy_action
        command_compute_time_ms = self._publish_step(action)
        self.controller_compute_times_ms.append(
            self.policy_inference_times_ms[-1] + command_compute_time_ms
        )
        self._last_velocity_metrics = self.velocity_metrics.update(
            self.metrics.elapsed(), self.drone, self.trees
        )
        self.entropy_metrics.update(self.metrics.elapsed(), self.beliefs)
        self._publish_obs(obs)

        curr_entropy = self.total_entropy(include_tracked=True)
        self._prev_entropy = curr_entropy
        self.entropy_pub.publish(Float32(data=curr_entropy))

        info = {
            "entropy": curr_entropy,
            "num_tracked": int(np.sum(self.tracked)),
            "new_targets_tracked": int(new_targets_tracked),
            "action": action,
            "policy_action": policy_action,
        }
        self._log_step(info)
        return obs, action, info

    def _log_step(self, info):
        action = np.asarray(info["action"], dtype=float).flatten()
        policy_action = np.asarray(info["policy_action"], dtype=float).flatten()
        self.metrics.log_control(
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
                "policy_action/forward": float(policy_action[0]),
                "policy_action/lateral": float(policy_action[1]),
                "policy_action/yaw_rate": float(policy_action[2]),
                "command/vx_mps": float(self._last_command[0]),
                "command/vy_mps": float(self._last_command[1]),
                "command/yaw_rate_radps": float(self._last_command[2]),
                "command/speed_mps": float(np.linalg.norm(self._last_command[:2])),
                **self._last_velocity_metrics,
                "belief_mean": float(np.mean(self.beliefs)) if self.beliefs is not None else 0.0,
                "measurement_age_s": self._measurement_age_s(),
                "policy_inference_time_ms": self.policy_inference_times_ms[-1],
                "controller_compute_time_ms": self.controller_compute_times_ms[-1],
                "selected_target_id": -1,
                "active_target_count_current": min(int(np.sum(~self.tracked)), self.k_obs) if self.tracked is not None else 0,
                "observation_episode_active": bool(self._measurement_age_s() <= self.measurement_period),
                "worst_tree_entropy": float(np.max(self.entropy_metrics.entropies(self.beliefs))),
                "unresolved_tree_count": int(self.ntargets - info["num_tracked"]),
            }
        )

    def _run_trial(self):
        self.reset()
        rate = rospy.Rate(self.step_frequency)
        while (
            not rospy.is_shutdown()
            and not bool(np.all(self.tracked))
            and self.steps < self.max_experiment_steps
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
        final_entropy = self.total_entropy(include_tracked=True)
        all_tracked = bool(np.all(self.tracked))
        termination_reason = (
            "all_trees_tracked"
            if all_tracked
            else "ros_shutdown"
            if rospy.is_shutdown()
            else "max_experiment_steps"
        )
        inference_times = np.asarray(self.policy_inference_times_ms, dtype=float)
        compute_times = np.asarray(self.controller_compute_times_ms, dtype=float)
        entropy_reduction = self._initial_entropy - final_entropy
        final_tree_entropy = self.entropy_metrics.entropies(self.beliefs)
        total_compute_ms = float(np.sum(compute_times)) if compute_times.size else 0.0
        summary = self.metrics.finish(
            {
                "total_time_execution_s": self.metrics.elapsed(),
                "total_distance_m": self.total_distance,
                "total_steps": self.steps,
                "initial_entropy": self._initial_entropy,
                "final_entropy": final_entropy,
                "entropy_reduction": entropy_reduction,
                "entropy_reduction_per_meter": entropy_reduction / self.total_distance if self.total_distance > 0.0 else np.nan,
                "entropy_reduction_per_second": entropy_reduction / self.metrics.elapsed() if self.metrics.elapsed() > 0.0 else np.nan,
                "entropy_reduction_per_compute_ms": entropy_reduction / total_compute_ms if total_compute_ms > 0.0 else np.nan,
                "worst_tree_entropy_final": float(np.max(final_tree_entropy)),
                "p90_tree_entropy_final": float(np.percentile(final_tree_entropy, 90)),
                "p95_tree_entropy_final": float(np.percentile(final_tree_entropy, 95)),
                "num_tracked_final": int(np.sum(self.tracked)) if self.tracked is not None else 0,
                "total_targets": self.ntargets,
                "termination_reason": termination_reason,
                "success": all_tracked,
                **self.velocity_metrics.summary(),
                **self.entropy_metrics.summary(final_time=self.metrics.elapsed()),
                "mean_policy_inference_time_ms": float(np.mean(inference_times)) if inference_times.size else np.nan,
                "median_policy_inference_time_ms": float(np.median(inference_times)) if inference_times.size else np.nan,
                "p95_policy_inference_time_ms": float(np.percentile(inference_times, 95)) if inference_times.size else np.nan,
                "mean_controller_compute_time_ms": float(np.mean(compute_times)) if compute_times.size else np.nan,
                "median_controller_compute_time_ms": float(np.median(compute_times)) if compute_times.size else np.nan,
                "p95_controller_compute_time_ms": float(np.percentile(compute_times, 95)) if compute_times.size else np.nan,
                "total_controller_compute_time_ms": total_compute_ms,
            }
        )
        rospy.loginfo(
            "RL trial %d seed %d complete after %d steps. Final entropy %.3f/%.3f.",
            self.run_index,
            self.trial_seed,
            self.steps,
            summary["final_entropy"],
            self._initial_entropy,
        )

    def run(self):
        last_run_index = self.run_index_offset + self.num_runs
        for run_index in range(self.run_index_offset, last_run_index):
            if rospy.is_shutdown():
                return
            self.run_index = run_index
            self.trial_seed = self.base_seed + run_index
            self.metrics = RosWandbLogger(
                "rl_agent",
                self.run_index,
                self._wandb_params(self.model_path, self.model_device),
                default_project="active_rl_classification",
            )
            self._run_trial()


if __name__ == "__main__":
    try:
        RLAgentNode().run()
    except rospy.ROSInterruptException:
        pass
