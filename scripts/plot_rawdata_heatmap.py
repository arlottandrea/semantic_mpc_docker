#!/usr/bin/env python3
"""Plot a heatmap of tree_score from a RawData CSV for poses looking at the tree."""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

try:
    from mlp_pipeline.visualize import looking_at_tree_mask
except Exception:
    # fallback: local implementation
    def wrap_angle(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def looking_at_tree_mask(x, y, yaw, tolerance_rad, camera_yaw_offset=0.0):
        direction_to_tree = np.arctan2(-y, -x)
        error = wrap_angle(direction_to_tree - yaw - camera_yaw_offset)
        return np.abs(error) <= tolerance_rad, error


def load_rows(path):
    rows = []
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--tolerance-deg", type=float, default=10.0)
    parser.add_argument("--camera-yaw-offset-deg", type=float, default=180.0)
    parser.add_argument("--output", default="runs/rawdata_heatmap.png")
    parser.add_argument("--bins", type=int, default=80)
    args = parser.parse_args()

    path = Path(args.csv)
    if not path.exists():
        raise FileNotFoundError(path)

    rows = load_rows(path)
    xs = np.array([float(r["x"]) for r in rows])
    ys = np.array([float(r["y"]) for r in rows])
    yaws = np.array([float(r["yaw"]) for r in rows])
    # Prefer tree_score if present, otherwise compute neutral 0.5
    tree_scores = np.array([float(r.get("tree_score", 0.5) or 0.5) for r in rows], dtype=float)

    mask, err = looking_at_tree_mask(xs, ys, yaws, np.deg2rad(args.tolerance_deg), np.deg2rad(args.camera_yaw_offset_deg))
    xs_m = xs[mask]
    ys_m = ys[mask]
    scores_m = tree_scores[mask]

    if len(xs_m) == 0:
        raise RuntimeError("No poses found that are looking at the tree with the given tolerance/offset")

    # 2D binned average of score
    bins = args.bins
    xedges = np.linspace(xs_m.min() - 1e-6, xs_m.max() + 1e-6, bins + 1)
    yedges = np.linspace(ys_m.min() - 1e-6, ys_m.max() + 1e-6, bins + 1)
    sum_hist, _, _ = np.histogram2d(xs_m, ys_m, bins=[xedges, yedges], weights=scores_m)
    count_hist, _, _ = np.histogram2d(xs_m, ys_m, bins=[xedges, yedges])
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = sum_hist / count_hist

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    im = ax.imshow(np.flipud(avg.T), extent=extent, vmin=0.0, vmax=1.0, cmap="viridis", aspect="equal")
    ax.scatter(xs_m, ys_m, c=scores_m, s=12, cmap="viridis", vmin=0.0, vmax=1.0, edgecolors="none")
    ax.scatter([0.0], [0.0], marker="*", s=140, c="red", label="tree")
    ax.set_title(f"RawData heatmap — looking at tree (±{args.tolerance_deg}°)")
    ax.set_xlabel("drone x relative to tree [m]")
    ax.set_ylabel("drone y relative to tree [m]")
    ax.grid(alpha=0.2)
    fig.colorbar(im, ax=ax, label="tree_score")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=180)
    print("Wrote", out)


if __name__ == "__main__":
    main()
