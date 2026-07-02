#!/usr/bin/env python3
"""Train NMPC-compatible ripe/raw perception surrogates from the generated CSV."""

import argparse
import ast
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter

try:
    from .common import load_config, resolve_device, seed_everything, weighted_detection_score
except ImportError:  # Direct script execution inside the Docker image.
    from common import load_config, resolve_device, seed_everything, weighted_detection_score


def augment(x, y, fraction, margin, rng):
    count = int(len(x) * fraction)
    if count == 0:
        return x, y
    distances = np.linalg.norm(x[:, :2], axis=1)
    radii = np.concatenate([
        rng.uniform(0.0, max(float(distances.min()), 1e-6), count // 2),
        rng.uniform(float(distances.max()), float(distances.max()) + margin, count - count // 2),
    ])
    angles = rng.uniform(-np.pi, np.pi, count)
    yaw = rng.uniform(-np.pi, np.pi, count)
    extra_x = np.column_stack([radii * np.cos(angles), radii * np.sin(angles), yaw])
    extra_y = np.full(count, 0.5)
    return np.concatenate([x, extra_x]), np.concatenate([y, extra_y])


def train_label(label, x, score, cfg, root, device, seed):
    # Preserve the class ordering expected by the existing NMPC notebooks/runtime.
    target = np.column_stack([score, 1.0 - score]) if label == "ripe" else np.column_stack([1.0 - score, score])
    dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(target, dtype=torch.float32))
    validation_count = max(1, int(round(len(dataset) * float(cfg["validation_split"]))))
    if validation_count >= len(dataset):
        raise ValueError("at least two samples are required for training")
    generator = torch.Generator().manual_seed(seed)
    train_set, validation_set = random_split(
        dataset, [len(dataset) - validation_count, validation_count], generator=generator
    )
    train_loader = DataLoader(train_set, batch_size=int(cfg["batch_size"]), shuffle=True,
                              generator=generator, num_workers=int(cfg["num_workers"]))
    validation_loader = DataLoader(validation_set, batch_size=int(cfg["batch_size"]),
                                   num_workers=int(cfg["num_workers"]))

    from semantic_mpc_package.perception_model import MultiLayerPerceptron
    model = MultiLayerPerceptron(
        input_dim=int(cfg["input_dim"]), hidden_size=int(cfg["hidden_size"]),
        hidden_layers=int(cfg["hidden_layers"]), output_dim=int(cfg["output_dim"]),
        threshold=float(cfg["threshold"]), gate_slope=float(cfg["gate_slope"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    criterion = torch.nn.MSELoss()
    output_dir = Path(cfg["output_dir"]) / label
    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.get("replace_existing_checkpoints", True):
        for old_checkpoint in output_dir.glob("best_model_epoch_*.pth"):
            old_checkpoint.unlink()
    writer = SummaryWriter(str(output_dir / "tensorboard"))
    best_loss = float("inf")
    best_path = None
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        train_total = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            train_total += loss.item() * len(features)
        model.eval()
        validation_total = 0.0
        with torch.no_grad():
            for features, targets in validation_loader:
                features, targets = features.to(device), targets.to(device)
                validation_total += criterion(model(features), targets).item() * len(features)
        train_loss = train_total / len(train_set)
        validation_loss = validation_total / len(validation_set)
        writer.add_scalars("loss", {"train": train_loss, "validation": validation_loss}, epoch)
        print("{} epoch {}/{} train={:.6f} validation={:.6f}".format(
            label, epoch, cfg["epochs"], train_loss, validation_loss))
        if validation_loss < best_loss:
            if best_path and best_path.exists():
                best_path.unlink()
            best_loss = validation_loss
            best_path = output_dir / "best_model_epoch_{}.pth".format(epoch)
            torch.save(model.state_dict(), str(best_path))
    writer.close()
    return str(best_path), best_loss


def main(config_path):
    root = load_config(config_path)
    cfg = root["training"]
    seed = int(root.get("seed", 42))
    seed_everything(seed)
    device = resolve_device(str(root.get("device", "auto")))
    source = Path(cfg["input_csv"])
    with source.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"x", "y", "yaw", "Ripe_scores", "Raw_scores"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("training CSV must contain: {}".format(", ".join(sorted(required))))
    x = np.asarray([[float(row[k]) for k in ("x", "y", "yaw")] for row in rows], dtype=np.float32)
    rng = np.random.default_rng(seed)
    results = {}
    for label in cfg["labels"]:
        if label not in ("ripe", "raw"):
            raise ValueError("training.labels supports only ripe and raw")
        detection_column = "Ripe_scores" if label == "ripe" else "Raw_scores"
        score = np.asarray([
            np.clip(
                weighted_detection_score(
                    [float(value) for value in ast.literal_eval(row[detection_column])],
                    root["dataset"]["score_midpoint"],
                    root["dataset"]["score_steepness"],
                ) + 0.5,
                0.0,
                1.0,
            )
            for row in rows
        ], dtype=np.float32)
        label_x, label_score = augment(x, score, float(cfg["augment_fraction"]),
                                       float(cfg["augment_distance_margin"]), rng)
        checkpoint, loss = train_label(label, label_x, label_score, cfg, root, device, seed)
        results[label] = {"checkpoint": checkpoint, "best_validation_loss": loss}
    metadata = {"created_at": datetime.now(timezone.utc).isoformat(), "source": str(source),
                "device": str(device), "seed": seed, "results": results}
    metadata_path = Path(cfg["output_dir"]) / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Wrote training metadata to {}".format(metadata_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    main(parser.parse_args().config)
