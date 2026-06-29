"""
Tree Classification Gymnasium environment for training active inspection policies.

A single drone navigates a 2D arena to classify trees as ripe (red) or not-ripe (green)
by approaching them from informative viewpoints. The drone receives noisy classification
probabilities based on its relative position/orientation to each tree, and maintains a
Bayesian belief over each tree's class.

Adapted from env/SceneEnv_RLlibMA_reviewers.py in the original active classification codebase.
"""

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import scipy.stats as distrib

from active_rl_classification.perception import TreePerception

# Environment constants
MAX_TARGETS = 100
BELIEF_THRESHOLD = 0.95
DIM_TARGET = 0.6
MINDIST = 2.3 * DIM_TARGET
OBS_BOUNDS = 16.0  # relative location clamp (matches original codebase)


class TreeClassificationEnv(gym.Env):
    """Single-drone tree classification environment.

    Config keys:
        ntargets (int or list[int]): Number of trees. If list [min, max], sampled per episode.
        horizon (int): Max steps per episode (default 100).
        side (float): Half-side of the square arena (default 25).
        layout (str): "random", "grid", or "mixed" (random-or-grid per episode).
        mixed_grid_prob (float): When layout="mixed", probability of using grid
            on a given reset (default 0.5).
        grid_n_rows (int): Number of rows for grid layout.
        grid_n_cols (int): Number of columns for grid layout.
        grid_row_spacing (float): Distance between rows.
        grid_col_spacing (float): Distance between columns.
        grid_jitter_std (float): Std dev of position noise on grid (default 0.3).
        perception_csvs (list[str]): Two CSV paths [not_ripe_csv, ripe_csv].
        use_oracle (bool): Apply the original codebase's oracle correction to
            perception outputs (default True). Set False to pass raw CNN
            probabilities through, allowing confidently-wrong observations.
        min_separation (float): Minimum distance between trees in random layout.
    """

    metadata = {"render_modes": []}

    def __init__(self, config=None):
        super().__init__()
        config = config or {}

        # Target count configuration
        ntargets_cfg = config.get("ntargets", None)
        if ntargets_cfg is None:
            # Default: for grid layout, use all grid positions; for random, default 4
            if config.get("layout", "random") == "grid":
                n_grid = min(config.get("grid_n_rows", 3) * config.get("grid_n_cols", 3), MAX_TARGETS)
                self.ntargets_range = [n_grid, n_grid]
            else:
                self.ntargets_range = [4, 4]
        elif isinstance(ntargets_cfg, (list, tuple)):
            assert len(ntargets_cfg) in (1, 2)
            self.ntargets_range = ntargets_cfg if len(ntargets_cfg) == 2 else [ntargets_cfg[0], ntargets_cfg[0]]
        else:
            self.ntargets_range = [ntargets_cfg, ntargets_cfg]
        self.max_targets = MAX_TARGETS
        self.k_obs = config.get("k_obs", 5)
        self.obs_range = config.get("obs_range", 5.0)

        self.nclasses = 2
        self.max_steps = config.get("horizon", 100)
        self.side = config.get("side", 25.0)
        self.delta_t = 0.25
        self.action_scale = np.array([2.0, 2.0, 1.0])  # matches original action_space_mapping

        # Layout configuration — "random", "grid", or "mixed"
        # (mixed picks one of the two per episode, with probability mixed_grid_prob).
        self.layout = config.get("layout", "random")
        assert self.layout in ("random", "grid", "mixed"), self.layout
        self.mixed_grid_prob = config.get("mixed_grid_prob", 0.5)
        self.grid_n_rows = config.get("grid_n_rows", 4)
        self.grid_n_cols = config.get("grid_n_cols", 4)
        self.grid_row_spacing = config.get("grid_row_spacing", 5.0)
        self.grid_col_spacing = config.get("grid_col_spacing", 5.0)
        self.grid_jitter_std = config.get("grid_jitter_std", 0.3)
        self.min_separation = config.get("min_separation", MINDIST)

        print(f"Initialized TreeClassificationEnv {self.layout} layout with ntargets_range={self.ntargets_range}, side={self.side}")
        # Perception model — one CSV per class: [not_ripe_csv, ripe_csv]
        self.perception = TreePerception(
            use_oracle=config.get("use_oracle", True)
        )
        csv_paths = config["perception_csvs"]
        self.perception.load_probas(csv_paths)
        
        self.uniform_proba = 1/self.nclasses * np.ones(self.nclasses, dtype=np.float32)
        # Max entropy for normalization (entropy of uniform distribution)
        self.max_entropy = distrib.entropy(
            [1.0 / self.nclasses] * self.nclasses, base=2
        )

        # Action space: [forward, lateral, yaw_rate] in [-1, 1] — body-frame velocity.
        # Applied as: rotate(action[:2] * [2,2] * 0.25, heading) → max ±0.5 m/step.
        # Heading: 60 * action[2] * 0.25 → ±15°/step.
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Observation space: k nearest untracked trees, padded to k_obs
        inf = np.finfo(np.float32).max
        self.observation_space = spaces.Dict({
            "location": spaces.Box(-inf, inf, shape=(self.k_obs, 3), dtype=np.float32),
            "belief": spaces.Box(0.0, 1.0 + 1e-10, shape=(self.k_obs, 1), dtype=np.float32),
            "measurement": spaces.Box(0.0, 1.0 + 1e-10, shape=(self.k_obs, 1), dtype=np.float32),
            "tracked": spaces.Box(0.0, 1.0, shape=(self.k_obs, 1), dtype=np.float32),
            "mask": spaces.Box(0.0, 1.0, shape=(self.k_obs,), dtype=np.float32),
        })

        # State variables (initialized in reset)
        self.trees = None
        self.tree_classes = None
        self.drone = None
        self.beliefs = None
        self.tracked = None
        self.ntargets = None
        self.steps = 0
        self._observations = None
        self._prev_entropy = 0.0

    # ------------------------------------------------------------------ #
    # Tree layout generation
    # ------------------------------------------------------------------ #

    def _generate_grid_positions(self, n_rows, n_cols):
        """Place trees on a regular grid centered in the arena, with optional jitter."""
        positions = []
        # Center the grid
        offset_x = -(n_rows - 1) * self.grid_row_spacing / 2.0
        offset_y = -(n_cols - 1) * self.grid_col_spacing / 2.0
        for r in range(n_rows):
            for c in range(n_cols):
                x = offset_x + r * self.grid_row_spacing
                y = offset_y + c * self.grid_col_spacing
                if self.grid_jitter_std > 0:
                    x += self.np_random.normal(0, self.grid_jitter_std)
                    y += self.np_random.normal(0, self.grid_jitter_std)
                # Clamp to arena
                x = np.clip(x, -(self.side - 2), self.side - 2)
                y = np.clip(y, -(self.side - 2), self.side - 2)
                positions.append(np.array([x, y]))
        return positions

    def _generate_random_positions(self, n):
        """Place trees randomly with minimum separation (rejection sampling)."""
        positions = []
        max_attempts = 10000
        attempts = 0
        while len(positions) < n and attempts < max_attempts:
            x = self.np_random.uniform(-(self.side - 2), self.side - 2)
            y = self.np_random.uniform(-(self.side - 2), self.side - 2)
            candidate = np.array([x, y])
            too_close = False
            for p in positions:
                if np.linalg.norm(candidate - p) < self.min_separation:
                    too_close = True
                    break
            if not too_close:
                positions.append(candidate)
            attempts += 1
        if len(positions) < n:
            raise RuntimeError(
                f"Could not place {n} trees with min_separation={self.min_separation} "
                f"in arena side={self.side} after {max_attempts} attempts."
            )
        return positions

    # ------------------------------------------------------------------ #
    # Drone movement
    # ------------------------------------------------------------------ #

    def _rotate_action(self, action_xy, heading_deg):
        """Rotate body-frame action vector into world frame."""
        c = np.cos(np.radians(heading_deg))
        s = np.sin(np.radians(heading_deg))
        return np.array([[c, -s], [s, c]]) @ action_xy

    # ------------------------------------------------------------------ #
    # Observation helpers
    # ------------------------------------------------------------------ #

    def _relative_pose(self, tree_idx):
        """Compute relative pose of tree in drone's frame.

        Returns [rel_x, rel_y, rel_bearing] where rel_bearing = atan2(rel_y, rel_x).
        """
        c = np.cos(np.radians(self.drone[2]))
        s = np.sin(np.radians(self.drone[2]))
        rot_t = np.array([[c, s], [-s, c]])
        rel_xy = rot_t @ (self.trees[tree_idx] - self.drone[:2])
        rel_bearing = np.arctan2(rel_xy[1], rel_xy[0])
        return np.array([rel_xy[0], rel_xy[1], rel_bearing])

    def _drone_pose_in_tree_frame(self, tree_idx):
        """Compute drone pose in tree-centric world-aligned frame.

        Returns [dx, dy, heading_rad] where dx, dy = drone_pos - tree_pos
        and heading_rad = drone heading in radians. This matches the CSV
        convention: (x, y, yaw) are the drone's pose relative to the tree.
        """
        rel_xy = self.drone[:2] - self.trees[tree_idx]
        heading_rad = np.radians(self.drone[2])
        return rel_xy[0], rel_xy[1], heading_rad

    def _get_measurement(self, tree_idx):
        """Get classification probability for a tree from current drone pose."""
        dx, dy, heading = self._drone_pose_in_tree_frame(tree_idx)
        prob = self.perception.get_prob(
            dx, dy, heading,
            self.tree_classes[tree_idx]
        )
        return prob

    def _build_obs(self):
        """Build observation dict with k nearest untracked trees (regardless of distance), padded to k_obs."""
        untracked = [t for t in range(self.ntargets) if not self.tracked[t]]
        if untracked:
            distances = np.array([np.linalg.norm(self.drone[:2] - self.trees[t]) for t in untracked])
            sorted_untracked = [untracked[i] for i in np.argsort(distances)][:self.k_obs]
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
            location[i] = rel.astype(np.float32)
            belief[i, 0] = np.float32(1.0 - distrib.entropy(self.beliefs[t], base=2) / self.max_entropy + 1e-10)
            measurement[i, 0] = np.float32(1.0 - distrib.entropy(self._observations[t], base=2) / self.max_entropy + 1e-10)
            tracked_obs[i, 0] = np.float32(0.0)  # always untracked by construction
            mask[i] = 1.0

        return {
            "location": location,
            "belief": belief,
            "measurement": measurement,
            "tracked": tracked_obs,
            "mask": mask,
        }

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Sample number of trees
        self.ntargets = self.np_random.integers(
            self.ntargets_range[0], self.ntargets_range[1] + 1
        )

        # Resolve effective layout for this episode
        if self.layout == "mixed":
            effective_layout = (
                "grid" if self.np_random.random() < self.mixed_grid_prob else "random"
            )
        else:
            effective_layout = self.layout

        # Place trees
        if effective_layout == "grid":
            all_positions = self._generate_grid_positions(
                self.grid_n_rows, self.grid_n_cols
            )
            # If more grid positions than ntargets, subsample
            if len(all_positions) > self.ntargets:
                indices = self.np_random.choice(
                    len(all_positions), self.ntargets, replace=False
                )
                all_positions = [all_positions[i] for i in indices]
            self.ntargets = len(all_positions)
            self.trees = all_positions
        else:
            self.trees = self._generate_random_positions(self.ntargets)

        # Assign random classes
        self.tree_classes = self.np_random.integers(0, self.nclasses, size=self.ntargets)

        self.uniform_proba = 1/self.nclasses * np.ones(self.nclasses, dtype=np.float32)

        # Place drone at random position
        drone_positions = self._generate_random_positions(1)
        # Make sure drone is not too close to trees
        drone_xy = drone_positions[0]
        heading = self.np_random.uniform(0, 360)
        self.drone = np.array([drone_xy[0], drone_xy[1], heading])

        # Initialize beliefs as uninformative (uniform over classes)
        self.beliefs = self.uniform_proba[None, :].repeat(self.ntargets, axis=0)

        # Tracked status (always False on reset given uniform beliefs)
        self.tracked = np.max(self.beliefs, axis=1) >= BELIEF_THRESHOLD

        # Initial measurements
        self._observations = self.uniform_proba[None, :].repeat(self.ntargets, axis=0)
        for t in range(self.ntargets):
            self._observations[t] = self._get_measurement(t)

        # Compute initial entropy
        self._prev_entropy = sum(
            distrib.entropy(self.beliefs[t], base=2)
            for t in range(self.ntargets) if not self.tracked[t]
        )

        self.steps = 0
        return self._build_obs(), {}

    def step(self, action):
        self.steps += 1

        # Nearest untracked distance before move (for approach reward)
        untracked_now = [t for t in range(self.ntargets) if not self.tracked[t]]
        if untracked_now:
            prev_nearest = min(np.linalg.norm(self.drone[:2] - self.trees[t]) for t in untracked_now)
        else:
            prev_nearest = 0.0

        # Body-frame velocity: rotate into world frame, scale, integrate
        applied = action * self.action_scale * self.delta_t
        new_pos = np.copy(self.drone)
        new_pos[:2] += self._rotate_action(applied[:2], self.drone[2])
        new_pos[:2] = np.clip(new_pos[:2], -self.side, self.side)
        new_pos[2] = (self.drone[2] + 60.0 * applied[2]) % 360.0
        self.drone = new_pos

        # Get measurements and update beliefs for trees within obs_range
        new_targets_tracked = 0
        for t in range(self.ntargets):
            dist = np.linalg.norm(self.drone[:2] - self.trees[t])
            if dist > self.obs_range:
                continue

            self._observations[t] = self._get_measurement(t)

            if not self.tracked[t]:
                prob = self._observations[t].copy()
                new_belief = self.beliefs[t] * prob
                new_belief /= (np.sum(new_belief) + 1e-32)
                self.beliefs[t] = new_belief

                if np.max(new_belief) >= BELIEF_THRESHOLD:
                    self.tracked[t] = True
                    new_targets_tracked += 1

        num_tracked = np.sum(self.tracked)

        # Compute current entropy over untracked targets
        curr_entropy = sum(
            distrib.entropy(self.beliefs[t], base=2)
            for t in range(self.ntargets) if not self.tracked[t]
        )

        # Reward computation (from SceneEnv_RLlibMA_reviewers.py:1480-1524)
        reward = 0.0

        # Movement penalty (from original)
        reward += -0.01 * np.linalg.norm(action[:2])
        reward += -0.01 * abs(action[2])

        # Approach reward: positive only while outside obs_range, zero once inside
        if untracked_now:
            curr_nearest = min(np.linalg.norm(self.drone[:2] - self.trees[t]) for t in untracked_now)
            prev_outer = max(0.0, prev_nearest - self.obs_range)
            curr_outer = max(0.0, curr_nearest - self.obs_range)
            reward += 0.1 * (prev_outer - curr_outer)

        # Classification bonus
        reward += new_targets_tracked * 5.0

        # Termination signals
        all_tracked = num_tracked == self.ntargets
        timed_out = self.steps >= self.max_steps
        terminated = all_tracked
        truncated = timed_out and not all_tracked

        if all_tracked:
            reward += 100.0
        elif not timed_out:
            # Step penalty + entropy decrease reward
            reward += -0.3
            if not math.isnan(curr_entropy) and self._prev_entropy > 0:
                reward += self._prev_entropy - curr_entropy

        self._prev_entropy = curr_entropy

        obs = self._build_obs()
        info = {
            "success": all_tracked,
            "num_tracked": int(num_tracked),
            "episode_length": self.steps,
        }

        return obs, reward, terminated, truncated, info


def _visualize_env(env, save_path="env_visualization.png"):
    """Render arena, trees, drone, and a perception-confidence heatmap around each tree."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle, FancyArrow

    fig, ax = plt.subplots(figsize=(10, 10))

    # Arena boundary
    ax.add_patch(Rectangle(
        (-env.side, -env.side), 2 * env.side, 2 * env.side,
        fill=False, edgecolor="black", linewidth=2,
    ))

    class_colors = {0: "green", 1: "red"}  # 0 = not-ripe, 1 = ripe
    class_labels = {0: "not-ripe", 1: "ripe"}

    # Confidence heatmap samples around each tree (drone poses facing the tree)
    grid_radius = 6.0
    n_grid = 41
    xs = np.linspace(-grid_radius, grid_radius, n_grid)
    ys = np.linspace(-grid_radius, grid_radius, n_grid)

    all_scatter_x, all_scatter_y, all_conf = [], [], []
    for t in range(env.ntargets):
        tree_xy = env.trees[t]
        true_cls = int(env.tree_classes[t])
        for gx in xs:
            for gy in ys:
                drone_x = tree_xy[0] + gx
                drone_y = tree_xy[1] + gy
                dist = np.hypot(gx, gy)
                if dist < 0.5 or dist > grid_radius:
                    continue
                # Drone heading points toward tree
                heading_rad = np.arctan2(-gy, -gx)
                prob = env.perception.get_prob(gx, gy, heading_rad, true_cls)
                all_scatter_x.append(drone_x)
                all_scatter_y.append(drone_y)
                all_conf.append(prob[true_cls])

    sc = ax.scatter(
        all_scatter_x, all_scatter_y, c=all_conf,
        cmap="viridis", s=6, alpha=0.5, vmin=0.5, vmax=1.0,
    )
    plt.colorbar(sc, ax=ax, label="P(true class) when drone faces tree")

    # Trees
    for t in range(env.ntargets):
        tree_xy = env.trees[t]
        cls = int(env.tree_classes[t])
        ax.add_patch(Circle(
            (tree_xy[0], tree_xy[1]), DIM_TARGET,
            color=class_colors[cls], ec="black", zorder=3,
        ))
        ax.text(tree_xy[0], tree_xy[1] + DIM_TARGET + 0.2, f"T{t}",
                ha="center", fontsize=9, zorder=4)

    # Drone
    drone_xy = env.drone[:2]
    heading_rad = np.radians(env.drone[2])
    ax.add_patch(Circle(
        (drone_xy[0], drone_xy[1]), 0.4,
        color="blue", zorder=5,
    ))
    arrow_len = 1.5
    ax.add_patch(FancyArrow(
        drone_xy[0], drone_xy[1],
        arrow_len * np.cos(heading_rad), arrow_len * np.sin(heading_rad),
        width=0.15, color="blue", zorder=5,
    ))

    # Legend
    from matplotlib.lines import Line2D
    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=class_colors[c],
               markersize=12, label=f"Tree: {class_labels[c]}")
        for c in sorted(class_colors)
    ]
    legend_items.append(Line2D([0], [0], marker="o", color="w",
                               markerfacecolor="blue", markersize=12, label="Drone"))
    ax.legend(handles=legend_items, loc="upper right")

    ax.set_xlim(-env.side - 1, env.side + 1)
    ax.set_ylim(-env.side - 1, env.side + 1)
    ax.set_aspect("equal")
    ax.set_title("Tree Classification Env — perception confidence around trees")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"Saved visualization to {save_path}")


if __name__ == "__main__":
    import os
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    csvs = [
        os.path.join(data_dir, "RawData.csv"),
        os.path.join(data_dir, "RipeData.csv"),
    ]

    env = TreeClassificationEnv(config={
        "layout": "grid",
        "grid_n_rows": 2,
        "grid_n_cols": 3,
        "grid_row_spacing": 10.0,
        "grid_col_spacing": 10.0,
        "side": 20.0,
        "perception_csvs": csvs,
    })
    obs, _ = env.reset(seed=0)
    print(f"ntargets={env.ntargets}, classes={env.tree_classes.tolist()}")
    print(f"drone={env.drone}")

    # Sanity: take a few random steps and print reward
    for i in range(3):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        print(f"step {i}: reward={r:.3f} tracked={info['num_tracked']}")

    _visualize_env(env)
