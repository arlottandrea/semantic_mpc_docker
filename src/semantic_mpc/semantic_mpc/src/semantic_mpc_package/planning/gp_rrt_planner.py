import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class GridCell:
    x: float
    y: float
    observed: bool = False
    value: float = 0.0


class GaussianProcessCoverage:
    """A light-weight GP-style occupancy/coverage model for the planning layer.

    The model keeps a coarse grid of visited/observed cells and smooths nearby
    cells with an exponential kernel so the planner can prefer regions that are
    still informative rather than only the exact cells already observed.
    """

    def __init__(self, bounds, resolution=1.0, kernel_scale=3.0):
        self.bounds = np.asarray(bounds, dtype=float).reshape(2, 2)
        self.resolution = float(resolution)
        self.kernel_scale = float(kernel_scale)
        self.grid = None
        self._build_grid()

    def _build_grid(self):
        x_min, x_max = self.bounds[0]
        y_min, y_max = self.bounds[1]
        x_steps = int(math.ceil((x_max - x_min) / self.resolution)) + 1
        y_steps = int(math.ceil((y_max - y_min) / self.resolution)) + 1
        self.grid = np.zeros((x_steps, y_steps), dtype=float)
        self.cells = []
        for i in range(x_steps):
            for j in range(y_steps):
                self.cells.append(
                    GridCell(
                        x=x_min + i * self.resolution,
                        y=y_min + j * self.resolution,
                    )
                )

    def _index(self, x, y):
        x_min, x_max = self.bounds[0]
        y_min, y_max = self.bounds[1]
        x = float(np.clip(x, x_min, x_max))
        y = float(np.clip(y, y_min, y_max))
        i = int(round((x - x_min) / self.resolution))
        j = int(round((y - y_min) / self.resolution))
        i = max(0, min(i, self.grid.shape[0] - 1))
        j = max(0, min(j, self.grid.shape[1] - 1))
        return i, j

    def observe(self, x, y, value=1.0):
        i, j = self._index(x, y)
        self.grid[i, j] = max(self.grid[i, j], value)
        self._propagate_local(i, j, value)

    def _propagate_local(self, i, j, value):
        for ii in range(max(0, i - 2), min(self.grid.shape[0], i + 3)):
            for jj in range(max(0, j - 2), min(self.grid.shape[1], j + 3)):
                dx = (ii - i) * self.resolution
                dy = (jj - j) * self.resolution
                dist = math.hypot(dx, dy)
                if dist <= 0.0:
                    continue
                weight = math.exp(-(dist ** 2) / (2.0 * self.kernel_scale ** 2))
                self.grid[ii, jj] = max(self.grid[ii, jj], value * weight)

    def uncertainty_map(self):
        return self.grid.copy()

    def informative_score(self, x, y):
        i, j = self._index(x, y)
        return float(self.grid[i, j])


class RRTWaypointPlanner:
    """A simple RRT-style waypoint generator over an information map."""

    def __init__(self, bounds, resolution=1.0, max_nodes=200):
        self.bounds = np.asarray(bounds, dtype=float).reshape(2, 2)
        self.resolution = float(resolution)
        self.max_nodes = int(max_nodes)
        self.nodes = []

    def _sample(self, rng, current_pose):
        if rng.random() < 0.25:
            return np.asarray(current_pose[:2], dtype=float)
        x_min, x_max = self.bounds[0]
        y_min, y_max = self.bounds[1]
        return np.array([rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)])

    def _nearest(self, point):
        if not self.nodes:
            return None
        return min(self.nodes, key=lambda node: np.linalg.norm(node[0] - point))

    def plan(self, start_pose, coverage, target_count=3):
        rng = np.random.default_rng(0)
        self.nodes = [np.asarray(start_pose[:2], dtype=float)]
        for _ in range(self.max_nodes):
            sample = self._sample(rng, start_pose)
            nearest = self._nearest(sample)
            if nearest is None:
                continue
            direction = sample - nearest
            if np.linalg.norm(direction) < 1e-8:
                continue
            direction = direction / np.linalg.norm(direction)
            step = direction * self.resolution
            candidate = nearest + step
            if self._in_bounds(candidate):
                self.nodes.append(candidate)
                if len(self.nodes) >= target_count:
                    break
        scores = []
        for node in self.nodes[1:]:
            scores.append((node, coverage.informative_score(node[0], node[1])))
        if not scores:
            return []
        scores.sort(key=lambda item: item[1], reverse=True)
        selected = [np.asarray(node, dtype=float) for node, _ in scores[:target_count]]
        while len(selected) < target_count:
            selected.append(np.asarray(self.nodes[-1], dtype=float))
        return selected

    def _in_bounds(self, point):
        x_min, x_max = self.bounds[0]
        y_min, y_max = self.bounds[1]
        return bool(x_min <= point[0] <= x_max and y_min <= point[1] <= y_max)
