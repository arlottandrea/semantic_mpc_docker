#!/usr/bin/env python3
"""Compare the structured perception MLP with raw/ripe CSV targets."""

import argparse
import ast
import csv
import json
import os
import sys
from pathlib import Path

# The Windows pixi environment contains both Intel and LLVM OpenMP runtimes
# (PyTorch and Matplotlib dependencies respectively).  In this inference-only
# process, allow them to coexist so plotting does not abort at import time.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import matplotlib.pyplot as plt
import numpy as np

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


def prepare_inference_targets(targets, output_dim):
    values = np.asarray(targets, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if output_dim <= 1:
        return values[:, :1]
    if output_dim == 2:
        return values[:, :2]
    return values[:, :3]


def read_dataset(path, label, root, minimum, output_dim):
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("empty dataset: {}".format(path))

    features, target_values = [], []
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
        visibility = float(len(ripe_scores) + len(raw_scores) >= minimum)
        raw_accuracy = 1.0 - p_ripe
        ripe_accuracy = p_ripe
        if output_dim <= 1:
            target = np.asarray([raw_accuracy if label == "raw" else ripe_accuracy], dtype=np.float32)
        elif output_dim == 2:
            target = np.asarray([raw_accuracy, ripe_accuracy], dtype=np.float32)
        else:
            target = np.asarray([visibility, raw_accuracy, ripe_accuracy], dtype=np.float32)
        target_values.append(target)
    return (
        np.asarray(features, dtype=np.float32),
        prepare_inference_targets(np.asarray(target_values, dtype=np.float32), output_dim),
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
    output_dim = int(cfg.get("output_dim", 3))
    datasets = (("raw", args.raw_csv), ("ripe", args.ripe_csv))
    results = []
    metrics = {"checkpoint": str(checkpoint), "datasets": {}}
    for label, path in datasets:
        features, target_values = read_dataset(path, label, root, minimum, output_dim)
        mask, _ = looking_at_tree_mask(
            features[:, 0], features[:, 1], features[:, 2], tolerance,
            np.deg2rad(args.camera_yaw_offset_deg),
        )
        if not np.any(mask):
            raise ValueError(
                "no {} poses look at the tree; verify yaw convention or camera_yaw_offset_deg".format(label)
            )
        features = features[mask]
        target_values = target_values[mask]
        prediction = infer(model, features, args.batch_size)[:, :output_dim]
        if output_dim <= 1:
            head_names = [label]
        elif output_dim == 2:
            head_names = ["raw", "ripe"]
        else:
            head_names = ["visibility", "raw", "ripe"]
        visible = target_values[:, 0] > 0.5 if target_values.shape[1] > 0 else np.zeros(len(features), dtype=bool)
        print(
            "{}: kept {}/{} poses ({:.1f}%)".format(
                label, int(mask.sum()), len(mask), 100.0 * mask.mean()
            )
        )
        for head_index, head_name in enumerate(head_names):
            mae = float(np.mean(np.abs(prediction[:, head_index] - target_values[:, head_index])))
            print("  {} MAE={:.4f}".format(head_name, mae))
        metrics["datasets"][label] = {
            "total_poses": int(len(mask)),
            "looking_at_tree_poses": int(mask.sum()),
            "looking_at_tree_fraction": float(mask.mean()),
            "visible_looking_poses": int(visible.sum()),
            "output_heads": head_names,
            "target_shape": list(target_values.shape),
        }
        results.append((label, features, target_values, prediction, head_names))

    fig, axes = plt.subplots(
        len(results),
        max(1, 2 * output_dim),
        figsize=(4.2 * max(1, 2 * output_dim), 3.2 * len(results)),
        squeeze=False,
        constrained_layout=True,
    )
    last_scatter = None
    for row, (label, features, target_values, prediction, head_names) in enumerate(results):
        for head_index, head_name in enumerate(head_names):
            target_ax = axes[row, 2 * head_index]
            pred_ax = axes[row, 2 * head_index + 1]
            last_scatter = plot_map(target_ax, features, target_values[:, head_index], "{} {} target".format(label, head_name))
            last_scatter = plot_map(pred_ax, features, prediction[:, head_index], "{} {} prediction".format(label, head_name))
    fig.colorbar(last_scatter, ax=axes.ravel().tolist(), label="probability", shrink=0.85)
    fig.suptitle(
        "Structured perception inference — poses looking at tree (±{:.1f}°)\n{}".format(
            args.look_at_tolerance_deg, checkpoint
        )
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output), dpi=180)
    print("Wrote {}".format(output))
    stats_output = Path(args.stats_output) if args.stats_output else output.with_suffix(".json")
    stats_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Wrote {}".format(stats_output))
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
    parser.add_argument("--stats-output")
    parser.add_argument("--look-at-tolerance-deg", type=float, default=10.0)
    # Dataset/Unity camera forward axis is opposite to the recorded yaw axis.
    parser.add_argument("--camera-yaw-offset-deg", type=float, default=180.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--show", action="store_true")
    main(parser.parse_args())
