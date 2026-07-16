#!/usr/bin/env python3
"""Create a deterministic 2x2 checkpoint for offline NMPC diagnostics."""

import argparse
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_SRC = ROOT / "src" / "semantic_mpc" / "semantic_mpc" / "src"
sys.path.insert(0, str(SEMANTIC_SRC))

from semantic_mpc_package.perception_model import MultiLayerPerceptron


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "models" / "nmpc" / "diagnostic_binary_2x2.pth"))
    parser.add_argument("--steps", type=int, default=1200)
    args = parser.parse_args()

    torch.manual_seed(17)
    rng = np.random.default_rng(17)
    count = 12000
    radius = rng.uniform(0.8, 6.5, count)
    bearing = rng.uniform(-np.pi, np.pi, count)
    yaw = rng.uniform(-np.pi, np.pi, count)
    x = radius * np.cos(bearing)
    y = radius * np.sin(bearing)
    features = torch.tensor(np.column_stack((x, y, yaw)), dtype=torch.float32)

    # The camera is informative around 2.5 m when its heading points to the tree.
    direction = np.arctan2(-y, -x)
    alignment = np.cos(yaw - direction)
    spatial = np.exp(-0.5 * ((radius - 2.5) / 1.15) ** 2)
    quality = np.clip(spatial * np.clip((alignment + 0.2) / 1.2, 0.0, 1.0), 0.0, 1.0)
    correct = 0.52 + 0.46 * quality
    targets = torch.tensor(
        np.column_stack((correct, 1.0 - correct, 1.0 - correct, correct)),
        dtype=torch.float32,
    ).reshape(-1, 2, 2)

    model = MultiLayerPerceptron(
        hidden_size=64,
        hidden_layers=3,
        output_dim=4,
        threshold=5.0,
        gate_slope=10.0,
        yaw_harmonics=4,
        include_alignment_features=True,
        output_temperature=0.4,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for step in range(args.steps):
        index = torch.randint(0, count, (512,))
        prediction = model(features[index]).reshape(-1, 2, 2)
        loss = -(targets[index] * torch.log(prediction.clamp_min(1e-8))).sum(dim=-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    with torch.no_grad():
        mae = (model(features).reshape(-1, 2, 2) - targets).abs().mean().item()
    print("saved={} mae={:.6f}".format(output, mae))


if __name__ == "__main__":
    main()
