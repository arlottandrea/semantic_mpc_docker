#!/usr/bin/env python3
"""Compare the structured perception MLP with raw/ripe CSV targets."""

import argparse
import ast
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from .common import load_config, weighted_detection_score
except ImportError:
    from common import load_config, weighted_detection_score


ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_SRC = ROOT / "src" / "semantic_mpc" / "semantic_mpc" / "src"
if str(SEMANTIC_SRC) not in sys.path:
    sys.path.insert(0, str(SEMANTIC_SRC))

from semantic_mpc_package.perception_model import MultiLayerPerceptron


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def looking_at_tree_mask(x, y, yaw, tolerance_rad, camera_yaw_offset=0.0):
    """Select poses whose camera forward axis points towards the tree at (0, 0)."""
    direction_to_tree = np.arctan2(-y, -x)
    error = wrap_angle(direction_to_tree - yaw - camera_yaw_offset)
    return np.abs(error) <= tolerance_rad, error


def latest_checkpoint(directory):
    checkpoints = list(Path(directory).glob("best_model_epoch_*.pth"))
    if not checkpoints:
        raise FileNotFoundError("no best_model_epoch_*.pth found in {}".format(directory))

    def epoch(path):
        return int(path.stem.rsplit("_", 1)[-1])

    return max(checkpoints, key=epoch)


def read_dataset(path, label, root, minimum):
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("empty dataset: {}".format(path))

    features, visibility, accuracy = [], [], []
    for row in rows:
        ripe_scores = [float(value) for value in ast.literal_eval(row["Ripe_scores"])]
        raw_scores = [float(value) for value in ast.literal_eval(row["Raw_scores"])]
        ripe_evidence = weighted_detection_score(
            ripe_scores, root["dataset"]["score_midpoint"], root["dataset"]["score_steepness"]
        )
        raw_evidence = weighted_detection_score(
            raw_scores, root["dataset"]["score_midpoint"], root["dataset"]["score_steepness"]
        )
        p_ripe = float(np.clip(ripe_evidence - raw_evidence + 0.5, 0.0, 1.0))
        features.append([float(row[key]) for key in ("x", "y", "yaw")])
        visibility.append(float(len(ripe_scores) + len(raw_scores) >= minimum))
        accuracy.append(p_ripe if label == "ripe" else 1.0 - p_ripe)
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(visibility, dtype=np.float32),
        np.asarray(accuracy, dtype=np.float32),
    )


def infer(model, features, batch_size):
    outputs = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.tensor(features[start:start + batch_size], dtype=torch.float32)
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def plot_map(ax, features, values, title):
    scatter = ax.scatter(
        features[:, 0], features[:, 1], c=values, s=22,
        cmap="viridis", vmin=0.0, vmax=1.0, edgecolors="none",
    )
    ax.scatter([0.0], [0.0], marker="*", s=130, c="red", label="tree")
    ax.set_title(title)
    ax.set_xlabel("drone x relative to tree [m]")
    ax.set_ylabel("drone y relative to tree [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    return scatter


def main(args):
    root = load_config(args.config)
    cfg = root["training"]
    checkpoint = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(args.model_dir)
    model = MultiLayerPerceptron(
        input_dim=int(cfg["input_dim"]), hidden_size=int(cfg["hidden_size"]),
        hidden_layers=int(cfg["hidden_layers"]), output_dim=int(cfg["output_dim"]),
        threshold=float(cfg["threshold"]), gate_slope=float(cfg["gate_slope"]),
    )
    model.load_state_dict(torch.load(str(checkpoint), map_location="cpu"))
    model.eval()

    tolerance = np.deg2rad(args.look_at_tolerance_deg)
    minimum = int(cfg.get("min_detections_for_visibility", 5))
    datasets = (("raw", args.raw_csv, 1), ("ripe", args.ripe_csv, 2))
    results = []
    for label, path, accuracy_column in datasets:
        features, target_v, target_k = read_dataset(path, label, root, minimum)
        mask, _ = looking_at_tree_mask(
            features[:, 0], features[:, 1], features[:, 2], tolerance,
            np.deg2rad(args.camera_yaw_offset_deg),
        )
        if not np.any(mask):
            raise ValueError(
                "no {} poses look at the tree; verify yaw convention or camera_yaw_offset_deg".format(label)
            )
        features, target_v, target_k = features[mask], target_v[mask], target_k[mask]
        prediction = infer(model, features, args.batch_size)
        pred_v, pred_k = prediction[:, 0], prediction[:, accuracy_column]
        visible = target_v > 0.5
        visibility_mae = float(np.mean(np.abs(pred_v - target_v)))
        accuracy_mae = float(np.mean(np.abs(pred_k[visible] - target_k[visible]))) if np.any(visible) else float("nan")
        print(
            "{}: kept {}/{} poses ({:.1f}%), visibility MAE={:.4f}, visible accuracy MAE={:.4f}".format(
                label, int(mask.sum()), len(mask), 100.0 * mask.mean(), visibility_mae, accuracy_mae
            )
        )
        results.append((label, features, target_v, pred_v, target_k, pred_k))

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    last_scatter = None
    for row, (label, features, target_v, pred_v, target_k, pred_k) in enumerate(results):
        values = (target_v, pred_v, target_k, pred_k)
        titles = (
            "{} dataset visibility".format(label),
            "{} MLP visibility".format(label),
            "{} dataset accuracy".format(label),
            "{} MLP accuracy".format(label),
        )
        for column, (value, title) in enumerate(zip(values, titles)):
            last_scatter = plot_map(axes[row, column], features, value, title)
    fig.colorbar(last_scatter, ax=axes, label="probability", shrink=0.85)
    fig.suptitle(
        "Structured perception inference — poses looking at tree (±{:.1f}°)\n{}".format(
            args.look_at_tolerance_deg, checkpoint
        )
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=180)
    print("Wrote {}".format(output))
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--raw-csv", default=str(ROOT / "datasets" / "raw" / "no_fog" / "TreeDatasetCNN.csv"))
    parser.add_argument("--ripe-csv", default=str(ROOT / "datasets" / "ripe" / "no_fog" / "TreeDatasetCNN.csv"))
    parser.add_argument("--model-dir", default=str(ROOT / "models" / "nmpc"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", default=str(ROOT / "runs" / "mlp_inference_looking_at_tree.png"))
    parser.add_argument("--look-at-tolerance-deg", type=float, default=10.0)
    # Dataset/Unity camera forward axis is opposite to the recorded yaw axis.
    parser.add_argument("--camera-yaw-offset-deg", type=float, default=180.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--show", action="store_true")
    main(parser.parse_args())
