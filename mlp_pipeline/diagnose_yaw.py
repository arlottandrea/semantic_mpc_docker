#!/usr/bin/env python3
"""Compare learned and empirical information gain as yaw varies."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_SRC = ROOT / "src" / "semantic_mpc" / "semantic_mpc" / "src"
sys.path.insert(0, str(SEMANTIC_SRC))
sys.path.insert(0, str(ROOT))

from mlp_pipeline.common import load_config
from mlp_pipeline.train import load_training_rows, select_model_targets
from semantic_mpc_package.perception_model import MultiLayerPerceptron


def mutual_information(likelihood):
    """Binary-class mutual information for a uniform prior."""
    likelihood = np.asarray(likelihood, dtype=float).reshape(-1, 2, 2)
    joint = 0.5 * likelihood
    observation = joint.sum(axis=1)
    posterior = joint / np.maximum(1e-12, observation[:, None, :])
    posterior_entropy = -np.sum(
        np.where(posterior > 0.0, posterior * np.log2(np.maximum(1e-12, posterior)), 0.0),
        axis=1,
    )
    return 1.0 - np.sum(observation * posterior_entropy, axis=1)


def assemble_empirical_rows(features, targets):
    by_pose = {}
    for feature, target in zip(features, targets):
        key = tuple(float(value) for value in feature)
        item = by_pose.setdefault(key, np.full((2, 2), np.nan))
        class_index = 0 if target[3] > 0.5 else 1
        selected = select_model_targets(target[None, :], {"output_dim": 4})[0]
        item[class_index] = selected.reshape(2, 2)[class_index]
    keys, matrices = [], []
    for key, matrix in by_pose.items():
        if np.all(np.isfinite(matrix)):
            keys.append(key)
            matrices.append(matrix)
    return np.asarray(keys, dtype=np.float32), np.asarray(matrices, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--plot", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()

    root = load_config(args.config)
    cfg = root["training"]
    features, targets, _ = load_training_rows(cfg, root)
    features, empirical_likelihood = assemble_empirical_rows(features, targets)
    model = MultiLayerPerceptron(
        input_dim=int(cfg["input_dim"]), hidden_size=int(cfg["hidden_size"]),
        hidden_layers=int(cfg["hidden_layers"]), output_dim=4,
        threshold=float(cfg["threshold"]), gate_slope=float(cfg["gate_slope"]),
        yaw_harmonics=int(cfg.get("yaw_harmonics", 1)),
        include_alignment_features=bool(cfg.get("include_alignment_features", False)),
        output_temperature=float(cfg.get("output_temperature", 1.0)),
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()

    tensor = torch.tensor(features, requires_grad=True)
    prediction_tensor = model(tensor)
    prediction = prediction_tensor.detach().numpy().reshape(-1, 2, 2)
    predicted_information = mutual_information(prediction)
    empirical_information = mutual_information(empirical_likelihood)

    # Differentiable torch MI for the yaw-gradient diagnostic.
    rows = prediction_tensor.reshape(-1, 2, 2)
    joint = 0.5 * rows
    observation = joint.sum(dim=1)
    posterior = joint / observation[:, None, :].clamp_min(1e-8)
    conditional_entropy = -(
        posterior * posterior.clamp_min(1e-8).log2()
    ).sum(dim=1)
    information = 1.0 - (observation * conditional_entropy).sum(dim=1)
    yaw_gradient = torch.autograd.grad(information.sum(), tensor)[0][:, 2].detach().numpy()

    xy, inverse = np.unique(features[:, :2], axis=0, return_inverse=True)
    predicted_std = np.asarray([
        predicted_information[inverse == index].std() for index in range(len(xy))
    ])
    empirical_std = np.asarray([
        empirical_information[inverse == index].std() for index in range(len(xy))
    ])
    curve_index = int(np.argmax(empirical_std))
    mask = inverse == curve_index
    order = np.argsort(features[mask, 2])

    correlation = float(np.corrcoef(predicted_information, empirical_information)[0, 1])
    report = {
        "poses": int(len(features)),
        "xy_locations": int(len(xy)),
        "information_mae_bits": float(np.mean(np.abs(predicted_information - empirical_information))),
        "information_correlation": correlation,
        "mean_predicted_yaw_std_bits": float(predicted_std.mean()),
        "mean_empirical_yaw_std_bits": float(empirical_std.mean()),
        "flat_predicted_xy_fraction": float(np.mean(predicted_std < 0.01)),
        "mean_abs_information_yaw_gradient": float(np.mean(np.abs(yaw_gradient))),
        "max_abs_information_yaw_gradient": float(np.max(np.abs(yaw_gradient))),
        "example_xy": xy[curve_index].tolist(),
    }
    Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].scatter(empirical_information, predicted_information, s=8, alpha=0.35)
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set(xlabel="empirical MI [bits]", ylabel="MLP MI [bits]", title="All poses")
    axes[0].grid(alpha=0.25)
    yaw = features[mask, 2][order]
    axes[1].plot(yaw, empirical_information[mask][order], "o-", label="empirical")
    axes[1].plot(yaw, predicted_information[mask][order], "o-", label="MLP")
    axes[1].set(xlabel="yaw [rad]", ylabel="MI [bits]", title="Yaw response at fixed (x,y)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.savefig(args.plot, dpi=180)
    plt.close(figure)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
