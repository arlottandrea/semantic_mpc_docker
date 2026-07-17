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
    from .common import load_config
except ImportError:
    from common import load_config


ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_SRC = ROOT / "src" / "semantic_mpc" / "semantic_mpc" / "src"
if str(SEMANTIC_SRC) not in sys.path:
    sys.path.insert(0, str(SEMANTIC_SRC))

from semantic_mpc_package.perception_model import MultiLayerPerceptron
from semantic_mpc_package.perception_protocol import tree_observation_scores


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
    return values[:, :output_dim]


def read_dataset(path, label, root, minimum, output_dim):
    with Path(path).open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("empty dataset: {}".format(path))

    features, target_values = [], []
    for row in rows:
        ripe_scores = [float(value) for value in ast.literal_eval(row["Ripe_scores"])]
        raw_scores = [float(value) for value in ast.literal_eval(row["Raw_scores"])]
        observation = tree_observation_scores(
            ripe_scores,
            raw_scores,
            minimum_score=float(root["training"].get("minimum_detection_score", 0.0)),
            minimum_tree_detections=minimum,
        )
        visible = float(np.all(np.isfinite(observation)))
        p_ripe = float(observation[0]) if visible else 0.5
        features.append([float(row[key]) for key in ("x", "y", "yaw")])
        raw_accuracy = max(0.5, 1.0 - p_ripe)
        ripe_accuracy = max(0.5, p_ripe)
        if output_dim == 4:
            raw_row = [0.5, 0.5] if visibility < 0.5 else [
                raw_accuracy, 1.0 - raw_accuracy
            ]
            ripe_row = [0.5, 0.5] if visibility < 0.5 else [
                1.0 - ripe_accuracy, ripe_accuracy
            ]
            target = np.asarray([*raw_row, *ripe_row], dtype=np.float32)
        elif output_dim <= 1:
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


def maximum_over_yaw(features, values):
    """Collapse repeated (x, y, yaw) samples to max(value) at each (x, y)."""
    features = np.asarray(features, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if len(features) != len(values):
        raise ValueError("features and values must contain the same number of poses")
    xy, inverse = np.unique(features[:, :2], axis=0, return_inverse=True)
    maxima = np.full((len(xy), values.shape[1]), -np.inf, dtype=np.float32)
    np.maximum.at(maxima, inverse, values)
    # Keep the three-column feature contract used by plotting code. Yaw is no
    # longer meaningful after maximization.
    return np.column_stack([xy, np.zeros(len(xy), dtype=np.float32)]), maxima


def plot_map(ax, features, values, title, continuous=False):
    """Plot measured samples, or a continuous interpolation of inference samples."""
    values = np.asarray(values).reshape(-1)
    color_min, color_max = 0.5, 0.99
    mappable = None
    if continuous and len(features) >= 3:
        try:
            mappable = ax.tricontourf(
                features[:, 0], features[:, 1], values,
                levels=np.linspace(color_min, color_max, 100), cmap="viridis",
                vmin=color_min, vmax=color_max, extend="neither",
            )
        except (RuntimeError, ValueError):
            # Degenerate/collinear pose sets cannot be triangulated.
            mappable = None
    scatter = ax.scatter(
        features[:, 0], features[:, 1], c=values,
        s=12 if continuous else 28, cmap="viridis",
        vmin=color_min, vmax=color_max,
        edgecolors="none" if continuous else "black",
        linewidths=0.0 if continuous else 0.25,
        alpha=0.35 if continuous else 1.0,
    )
    ax.scatter([0.0], [0.0], marker="*", s=130, c="red", label="tree")
    ax.set_title(title)
    ax.set_xlabel("drone x relative to tree [m]")
    ax.set_ylabel("drone y relative to tree [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.2)
    return mappable if mappable is not None else scatter


def value_statistics(values):
    values = np.asarray(values, dtype=float).reshape(-1)
    return {
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "standard_deviation": float(values.std()),
    }


def load_model(cfg, checkpoint):
    model = MultiLayerPerceptron(
        input_dim=int(cfg["input_dim"]), hidden_size=int(cfg["hidden_size"]),
        hidden_layers=int(cfg["hidden_layers"]), output_dim=int(cfg["output_dim"]),
        threshold=float(cfg["threshold"]), gate_slope=float(cfg["gate_slope"]),
        yaw_harmonics=int(cfg.get("yaw_harmonics", 1)),
        include_alignment_features=bool(cfg.get("include_alignment_features", False)),
        output_temperature=float(cfg.get("output_temperature", 1.0)),
        yaw_threshold_deg=float(cfg.get("yaw_threshold_deg", 30.0)),
        yaw_gate_slope=float(cfg.get("yaw_gate_slope", 50.0)),
    )
    model.load_state_dict(torch.load(str(checkpoint), map_location="cpu"))
    return model.eval()


def main(args):
    root = load_config(args.config)
    cfg = root["training"]
    checkpoint = Path(args.checkpoint) if args.checkpoint else latest_checkpoint(args.model_dir)
    checkpoints = {
        "raw": Path(args.raw_checkpoint) if args.raw_checkpoint else checkpoint,
        "ripe": Path(args.ripe_checkpoint) if args.ripe_checkpoint else checkpoint,
    }

    minimum = int(cfg.get("min_detections_for_visibility", 5))
    output_dim = int(cfg.get("output_dim", 3))
    datasets = (("raw", args.raw_csv), ("ripe", args.ripe_csv))
    results = []
    metrics = {
        "checkpoints": {label: str(path) for label, path in checkpoints.items()},
        "datasets": {},
    }
    for label, path in datasets:
        model = load_model(cfg, checkpoints[label])
        features, target_values = read_dataset(path, label, root, minimum, output_dim)
        prediction = infer(model, features, args.batch_size)[:, :output_dim]
        total_poses = len(features)
        pose_features = features
        features, target_values = maximum_over_yaw(features, target_values)
        _, prediction = maximum_over_yaw(pose_features, prediction)
        if output_dim == 4:
            head_names = [
                "raw_to_raw", "raw_to_ripe",
                "ripe_to_raw", "ripe_to_ripe",
            ]
        elif output_dim <= 1:
            head_names = [label]
        elif output_dim == 2:
            head_names = ["raw", "ripe"]
        else:
            head_names = ["visibility", "raw", "ripe"]
        visible = None
        if output_dim >= 3:
            visible = target_values[:, 0] > 0.5
        print(
            "{}: collapsed {} poses to {} (x, y) points using max over yaw".format(
                label, total_poses, len(features)
            )
        )
        for head_index, head_name in enumerate(head_names):
            mae = float(np.mean(np.abs(prediction[:, head_index] - target_values[:, head_index])))
            print("  {} MAE={:.4f}".format(head_name, mae))
        metrics["datasets"][label] = {
            "total_poses": int(total_poses),
            "xy_points": int(len(features)),
            "yaw_reduction": "maximum",
            "output_heads": head_names,
            "target_shape": list(target_values.shape),
            "heads": {
                head_name: {
                    "dataset": value_statistics(target_values[:, head_index]),
                    "inference": value_statistics(prediction[:, head_index]),
                    "mae": float(np.mean(np.abs(
                        prediction[:, head_index] - target_values[:, head_index]
                    ))),
                }
                for head_index, head_name in enumerate(head_names)
            },
        }
        if visible is not None:
            metrics["datasets"][label]["visible_xy_points"] = int(visible.sum())
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
            plot_map(
                target_ax, features, target_values[:, head_index],
                "{} {} dataset samples".format(label, head_name),
            )
            last_scatter = plot_map(
                pred_ax, features, prediction[:, head_index],
                "{} {} inference surface".format(label, head_name), continuous=True,
            )
    fig.colorbar(last_scatter, ax=axes.ravel().tolist(), label="probability", shrink=0.85)
    fig.suptitle(
        "Structured perception inference — maximum over yaw at each (x, y)\n"
        "raw: {} | ripe: {}".format(checkpoints["raw"], checkpoints["ripe"])
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
    parser.add_argument("--raw-checkpoint")
    parser.add_argument("--ripe-checkpoint")
    parser.add_argument("--output", default=str(ROOT / "runs" / "mlp_inference_looking_at_tree.png"))
    parser.add_argument("--stats-output")
    parser.add_argument("--look-at-tolerance-deg", type=float, default=10.0)
    # Dataset/Unity camera forward axis is opposite to the recorded yaw axis.
    parser.add_argument("--camera-yaw-offset-deg", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--show", action="store_true")
    main(parser.parse_args())
