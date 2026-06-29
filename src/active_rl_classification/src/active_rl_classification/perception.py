"""
Tree perception model for binary ripe/not-ripe classification confidence
based on relative drone-tree pose.

Each class (0 = not-ripe, 1 = ripe) has its own pre-computed probability table
loaded from a CSV, mirroring the original codebase where each pedestrian class
had its own Pedestrian instance with its own KDTree+CSV (see approx_perception.py).

CSV format (from the data collection pipeline):
    x, y, yaw, image_path, Ripe_scores, Raw_scores, tree_score

Where tree_score is P(ripe) at that relative pose. The KDTree indexes by
[x, y, cos(yaw), sin(yaw)] for robust angular matching.
"""

import csv
import numpy as np
from scipy import spatial


NCLASSES = 2


class TreePerceptionClass:
    """Perception model for a single tree class.

    Holds the KDTree and probability table for one class of tree.

    Args:
        class_id: The class index this model represents (0=not-ripe, 1=ripe).
        use_oracle: If True, apply the original codebase's oracle correction
            at query time (replace confidently-wrong predictions with uniform).
            If False, return raw CNN output — matches what the drone would
            actually observe at deployment.
    """

    def __init__(self, class_id, use_oracle=True):
        self.class_id = class_id
        self.use_oracle = use_oracle
        self.tree = None  # KDTree
        self.probas = []  # List of [P(not-ripe), P(ripe)] per stored pose

    def load_probas(self, csv_path):
        """Load pre-computed classification probabilities from CSV.

        Expected CSV columns: x, y, yaw, image_path, Ripe_scores, Raw_scores, tree_score
        tree_score = P(ripe); probability vector is [1 - tree_score, tree_score].
        A KDTree is built over [x, y, cos(yaw), sin(yaw)] for nearest-neighbor lookup.
        """
        poses = []
        self.probas = []
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                x = float(row['x'])
                y = float(row['y'])
                yaw = float(row['yaw'])
                tree_score = float(row['tree_score'])

                poses.append([x, y, np.cos(yaw), np.sin(yaw)])
                self.probas.append(np.array([1.0 - tree_score, tree_score]))

        self.tree = spatial.KDTree(poses)

    def get_prob(self, rel_x, rel_y, rel_orient):
        """Lookup probability from pre-computed hash table via KDTree.

        If `use_oracle` is True, applies the original codebase's correction
        (mirrors approx_perception.py:179-184): if the stored argmax matches
        this class_id and exceeds uniform, keep it (spread the remainder
        over the other class); otherwise return uniform [0.5, 0.5].

        If `use_oracle` is False, returns the raw stored probability — the
        agent may receive confidently-wrong observations, matching reality.
        """
        query = np.array([
            rel_x, rel_y,
            np.cos(rel_orient), np.sin(rel_orient)
        ])
        _, idx = self.tree.query(query)
        probabilities = self.probas[idx].copy()

        if not self.use_oracle:
            return probabilities

        maxprob = np.max(probabilities)
        if maxprob > 1.0 / NCLASSES and probabilities[self.class_id] == maxprob:
            probabilities[probabilities != maxprob] = (1 - maxprob) / (NCLASSES - 1)
        else:
            probabilities = np.ones(NCLASSES) / NCLASSES

        return probabilities


class TreePerception:
    """Manages perception models for both tree classes (not-ripe, ripe).

    Mirrors the original codebase pattern where each pedestrian class had its own
    Pedestrian instance with its own KDTree+CSV.

    Args:
        use_oracle: If True (default, matches original codebase), replace
            confidently-wrong predictions with uniform at query time.
            If False, pass raw CNN output through.
    """

    def __init__(self, use_oracle=True):
        self.nclasses = NCLASSES
        self.use_oracle = use_oracle
        self.class_models = [
            TreePerceptionClass(class_id=c, use_oracle=use_oracle)
            for c in range(NCLASSES)
        ]

    def load_probas(self, csv_paths):
        """Load pre-computed probabilities for both classes.

        Args:
            csv_paths: List of 2 CSV paths, indexed by class_id.
                       [not_ripe_csv, ripe_csv]
        """
        assert len(csv_paths) == NCLASSES, (
            f"Expected {NCLASSES} CSV paths, got {len(csv_paths)}"
        )
        for class_id, path in enumerate(csv_paths):
            self.class_models[class_id].load_probas(path)

    def get_prob(self, rel_x, rel_y, rel_orient, true_class):
        """Get classification probability vector for a tree.

        Queries the perception model for the tree's true class, matching the
        original pattern: self.unrealPedestrianData[targets_ID[target]].getProb(rel_pose)

        Args:
            rel_x, rel_y: Drone position relative to tree (world-aligned frame).
            rel_orient: Drone heading in radians.
            true_class: The tree's true class index (0 or 1).

        Returns:
            np.array of shape (2,) with classification probabilities.
        """
        return self.class_models[true_class].get_prob(rel_x, rel_y, rel_orient)


# ---------------------------------------------------------------------- #
# Tests / visualization
# ---------------------------------------------------------------------- #

def _test_csv_loading(csv_paths):
    """Load each CSV and print summary stats."""
    print("\n=== CSV loading test ===")
    for class_id, path in enumerate(csv_paths):
        p = TreePerception()
        p.class_models[class_id].load_probas(path)
        model = p.class_models[class_id]
        probs = np.stack(model.probas)
        print(f"  {path}")
        print(f"    rows={len(model.probas)}  "
              f"P(ripe) range=[{probs[:,1].min():.3f}, {probs[:,1].max():.3f}]  "
              f"mean={probs[:,1].mean():.3f}")
        # A query at the first stored pose must return a valid vector
        data = model.tree.data[0]
        x, y = data[0], data[1]
        yaw = np.arctan2(data[3], data[2])
        q = model.get_prob(x, y, yaw)
        assert q.shape == (2,), f"bad shape {q.shape}"
        print(f"    query at row0 pose -> {q}")


def _test_oracle_correction():
    """get_prob must return argmax == class_id when stored max aligns, else uniform."""
    print("\n=== Oracle correction test (use_oracle=True) ===")
    m = TreePerceptionClass(class_id=1, use_oracle=True)
    m.tree = spatial.KDTree([[0, 0, 1, 0], [1, 0, 1, 0]])
    m.probas = [np.array([0.2, 0.8]), np.array([0.7, 0.3])]
    # Query first pose: maxprob=0.8 at class 1 (matches) → keep
    out = m.get_prob(0.0, 0.0, 0.0)
    assert np.isclose(out[1], 0.8), out
    print(f"  aligned case -> {out}")
    # Query second pose: maxprob=0.7 at class 0 (mismatch) → uniform
    out = m.get_prob(1.0, 0.0, 0.0)
    assert np.allclose(out, [0.5, 0.5]), out
    print(f"  misaligned case -> {out}")

    print("\n=== Raw lookup test (use_oracle=False) ===")
    m = TreePerceptionClass(class_id=1, use_oracle=False)
    m.tree = spatial.KDTree([[0, 0, 1, 0], [1, 0, 1, 0]])
    m.probas = [np.array([0.2, 0.8]), np.array([0.7, 0.3])]
    out = m.get_prob(0.0, 0.0, 0.0)
    assert np.allclose(out, [0.2, 0.8]), out
    print(f"  aligned pose (raw) -> {out}")
    # Oracle off: confidently-wrong prediction is NOT masked
    out = m.get_prob(1.0, 0.0, 0.0)
    assert np.allclose(out, [0.7, 0.3]), out
    print(f"  misaligned pose (raw, confidently wrong) -> {out}")


def _visualize_perception(csv_path, class_id, save_path):
    """Plot P(true class) over a 2D grid of (rel_x, rel_y) at yaw=0."""
    import matplotlib.pyplot as plt

    p = TreePerception()
    p.class_models[class_id].load_probas(csv_path)

    lim = 5.0
    n = 80
    xs = np.linspace(-lim, lim, n)
    ys = np.linspace(-lim, lim, n)
    grid = np.zeros((n, n))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            prob = p.class_models[class_id].get_prob(x, y, 0.0)
            grid[j, i] = prob[class_id]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(
        grid, origin="lower", extent=[-lim, lim, -lim, lim],
        cmap="viridis", vmin=0.5, vmax=1.0,
    )
    ax.scatter([0], [0], c="red", marker="*", s=200, label="tree")
    ax.set_title(f"P(class={class_id}) — {csv_path}")
    ax.set_xlabel("rel_x"); ax.set_ylabel("rel_y")
    ax.legend()
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    print(f"  saved {save_path}")


if __name__ == "__main__":
    import os
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    notripe = os.path.join(data_dir, "TreeDatasetCNN.csv")
    ripe = os.path.join(data_dir, "RipeData.csv")

    _test_oracle_correction()
    _test_csv_loading([notripe, ripe])
    _visualize_perception(notripe, class_id=0, save_path="perception_notripe.png")
    _visualize_perception(ripe, class_id=1, save_path="perception_ripe.png")

    print("\nAll perception tests passed.")
