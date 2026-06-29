import math

import numpy as np


def get_domain(tree_positions):
    return [
        float(np.min(tree_positions[:, 0])),
        float(np.min(tree_positions[:, 1])),
    ], [
        float(np.max(tree_positions[:, 0])),
        float(np.max(tree_positions[:, 1])),
    ]


def corner_initial_pose(tree_positions):
    lb, ub = get_domain(tree_positions)
    return np.array([ub[0] + 1.5, lb[1] - 1.5, np.pi / 2.0], dtype=float)


def seeded_corner_initial_pose(tree_positions, seed, margin=1.5, heading=None):
    """Return one of the four field corners deterministically for a trial seed."""
    lb, ub = get_domain(tree_positions)
    margin = float(margin)
    corners = np.array(
        [
            [lb[0] - margin, lb[1] - margin],
            [lb[0] - margin, ub[1] + margin],
            [ub[0] + margin, lb[1] - margin],
            [ub[0] + margin, ub[1] + margin],
        ],
        dtype=float,
    )
    rng = np.random.default_rng(int(seed))
    position = corners[int(rng.integers(0, len(corners)))]
    center = (np.asarray(lb, dtype=float) + np.asarray(ub, dtype=float)) / 2.0
    if heading is None:
        heading = math.atan2(center[1] - position[1], center[0] - position[0])
    return np.array([position[0], position[1], float(heading)], dtype=float)


def random_initial_pose(tree_positions, lb, ub, margin=1.5, rng=None):
    rng = rng or np.random.default_rng()
    while True:
        x = rng.uniform(lb[0], ub[0])
        y = rng.uniform(lb[1], ub[1])
        if np.all(np.linalg.norm(tree_positions - np.array([x, y]), axis=1) >= margin):
            theta = rng.uniform(-np.pi, np.pi)
            return np.array([x, y, theta], dtype=float)
