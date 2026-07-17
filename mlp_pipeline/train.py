#!/usr/bin/env python3
"""Train one structured NMPC observation model with [v, k_raw, k_ripe]."""

import argparse
import ast
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_SRC = ROOT / "src" / "semantic_mpc" / "semantic_mpc" / "src"
if str(SEMANTIC_SRC) not in sys.path:
    sys.path.insert(0, str(SEMANTIC_SRC))

try:
    from .common import load_config, resolve_device, seed_everything
except ImportError:
    from common import load_config, resolve_device, seed_everything

from semantic_mpc_package.perception_protocol import tree_observation_scores


def augment(x, y, fraction, margin, rng):
    """Add out-of-distribution poses whose target is no useful observation."""
    count = int(len(x) * fraction)
    if count == 0:
        return x, y
    distances = np.linalg.norm(x[:, :2], axis=1)
    radii = rng.uniform(float(distances.max()), float(distances.max()) + margin, count)
    angles = rng.uniform(-np.pi, np.pi, count)
    yaw = rng.uniform(-np.pi, np.pi, count)
    extra_x = np.column_stack([radii * np.cos(angles), radii * np.sin(angles), yaw])
    # kappa is irrelevant when v=0; 0.5 is explicitly neutral.
    extra_y = np.tile(np.asarray([0.0, 0.5, 0.5], dtype=np.float32), (count, 1))
    return np.concatenate([x, extra_x]), np.concatenate([y, extra_y])


def structured_loss(prediction, target, visibility_weight, semantic_weight):
    output_dim = int(prediction.shape[1])
    if output_dim == 4:
        likelihood = prediction.reshape(-1, 2, 2)
        target_likelihood = target[:, :4].reshape(-1, 2, 2)
        class_mask = target[:, 4:6]
        row_weight = target[:, 6:8] if target.shape[1] >= 8 else torch.ones_like(class_mask)
        cross_entropy = -(
            target_likelihood * likelihood.clamp_min(1e-8).log()
        ).sum(dim=-1)
        weights = class_mask * row_weight
        return (cross_entropy * weights).sum() / weights.sum().clamp_min(1.0)
    if output_dim >= 3:
        visibility = torch.nn.functional.binary_cross_entropy(
            prediction[:, 0], target[:, 0]
        )
        semantic_mask = target[:, 3:5] * target[:, 0:1]
        semantic_elementwise = torch.nn.functional.binary_cross_entropy(
            prediction[:, 1:3], target[:, 1:3], reduction="none"
        )
        semantic = (semantic_elementwise * semantic_mask).sum() / semantic_mask.sum().clamp_min(1.0)
        return visibility_weight * visibility + semantic_weight * semantic
    if output_dim == 2:
        target_semantic = target[:, 1:3]
        semantic_mask = target[:, 0:1]
        semantic_elementwise = torch.nn.functional.binary_cross_entropy(
            prediction[:, :2], target_semantic, reduction="none"
        )
        semantic = (semantic_elementwise * semantic_mask).sum() / semantic_mask.sum().clamp_min(1.0)
        return semantic_weight * semantic
    target_value = target[:, 1:2]
    target_mask = target[:, 0:1]
    elementwise = torch.nn.functional.binary_cross_entropy(
        prediction[:, :1], target_value, reduction="none"
    )
    return semantic_weight * ((elementwise * target_mask).sum() / target_mask.sum().clamp_min(1.0))


def mse_loss(prediction, target, visibility_weight, semantic_weight):
    output_dim = int(prediction.shape[1])
    if output_dim >= 3:
        target_value = target[:, :3]
    elif output_dim == 2:
        target_value = target[:, 1:3]
    else:
        target_value = target[:, 1:2]
    elementwise = torch.nn.functional.mse_loss(
        prediction[:, :output_dim], target_value, reduction="none"
    )
    if output_dim == 1:
        mask = target[:, 0:1]
        active = mask > 0.5
        informative = active & (target_value > 0.500001)
        neutral = active & ~informative
        group_losses = []
        if torch.any(neutral):
            group_losses.append(elementwise[neutral].mean())
        if torch.any(informative):
            group_losses.append(elementwise[informative].mean())
        if not group_losses:
            return elementwise.sum() * 0.0
        return torch.stack(group_losses).mean()
    return elementwise.mean()


def headwise_bce_loss(prediction, target, visibility_weight, semantic_weight):
    output_dim = int(prediction.shape[1])
    if output_dim >= 3:
        return structured_loss(prediction, target, visibility_weight, semantic_weight)
    if output_dim == 2:
        target_values = target[:, 1:3]
        mask = target[:, 0:1]
        loss = F.binary_cross_entropy(prediction[:, :2], target_values, reduction="none")
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)
    target_value = target[:, 1:2]
    mask = target[:, 0:1]
    loss = F.binary_cross_entropy(prediction[:, :1], target_value, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def resolve_loss_function(cfg):
    if isinstance(cfg, dict):
        loss_name = str(cfg.get("loss_function", cfg.get("loss", "structured"))).lower()
    else:
        loss_name = str(cfg).lower()
    if loss_name in {"structured", "structured_loss"}:
        return structured_loss
    if loss_name in {"bce", "binary_cross_entropy", "binary_crossentropy", "headwise_bce", "headwise"}:
        return headwise_bce_loss
    if loss_name in {"mse", "mean_squared_error", "mean_squared"}:
        return mse_loss
    raise ValueError("unsupported loss function '{}'".format(loss_name))


def train_model(x, target, cfg, device, seed):
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)
    )
    validation_count = max(1, int(round(len(dataset) * float(cfg["validation_split"]))))
    if validation_count >= len(dataset):
        raise ValueError("at least two samples are required for training")
    generator = torch.Generator().manual_seed(seed)
    train_set, validation_set = random_split(
        dataset, [len(dataset) - validation_count, validation_count], generator=generator
    )
    train_loader = DataLoader(
        train_set, batch_size=int(cfg["batch_size"]), shuffle=True,
        generator=generator, num_workers=int(cfg["num_workers"]),
    )
    validation_loader = DataLoader(
        validation_set, batch_size=int(cfg["batch_size"]), num_workers=int(cfg["num_workers"])
    )

    from semantic_mpc_package.perception_model import ClassConditionedMLP, MultiLayerPerceptron
    common_kwargs = dict(
        hidden_size=int(cfg["hidden_size"]), hidden_layers=int(cfg["hidden_layers"]),
        threshold=float(cfg["threshold"]), gate_slope=float(cfg["gate_slope"]),
        yaw_harmonics=int(cfg.get("yaw_harmonics", 1)),
        include_alignment_features=bool(cfg.get("include_alignment_features", False)),
        output_temperature=float(cfg.get("output_temperature", 1.0)),
        yaw_threshold_deg=float(cfg.get("yaw_threshold_deg", 30.0)),
        yaw_gate_slope=float(cfg.get("yaw_gate_slope", 50.0)),
        architecture=str(cfg.get("architecture", "resnet")),
        ode_steps=int(cfg.get("ode_steps", 3)),
        ode_dt=float(cfg.get("ode_dt", 0.25)),
    )
    if cfg.get("model_type") == "class_conditioned":
        model = ClassConditionedMLP(**common_kwargs).to(device)
    else:
        model = MultiLayerPerceptron(
            input_dim=int(cfg["input_dim"]), output_dim=int(cfg["output_dim"]),
            **common_kwargs
        ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["learning_rate"]))
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.get("replace_existing_checkpoints", True):
        for old_checkpoint in output_dir.glob("best_model_epoch_*.pth"):
            old_checkpoint.unlink()
    writer = SummaryWriter(str(output_dir / "tensorboard"))
    best_loss, best_path = float("inf"), None
    loss_fn = resolve_loss_function(cfg)
    weights = (float(cfg.get("visibility_loss_weight", 1.0)),
               float(cfg.get("semantic_loss_weight", 1.0)))
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        train_total = 0.0
        for features, targets in train_loader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(features), targets, *weights)
            loss.backward()
            optimizer.step()
            train_total += loss.item() * len(features)
        model.eval()
        validation_total = 0.0
        with torch.no_grad():
            for features, targets in validation_loader:
                features, targets = features.to(device), targets.to(device)
                validation_total += loss_fn(model(features), targets, *weights).item() * len(features)
        train_loss = train_total / len(train_set)
        validation_loss = validation_total / len(validation_set)
        writer.add_scalars("loss", {"train": train_loss, "validation": validation_loss}, epoch)
        print("epoch {}/{} train={:.6f} validation={:.6f}".format(
            epoch, cfg["epochs"], train_loss, validation_loss))
        if validation_loss < best_loss:
            if best_path and best_path.exists():
                best_path.unlink()
            best_loss = validation_loss
            best_path = output_dir / "best_model_epoch_{}.pth".format(epoch)
            torch.save(model.state_dict(), str(best_path))
    writer.close()
    return str(best_path), best_loss, train_loss, validation_loss


def load_training_rows(cfg, root):
    required = {"x", "y", "yaw", "Ripe_scores", "Raw_scores"}
    features, targets, sources = [], [], []
    minimum = int(cfg.get("min_detections_for_visibility", 5))
    if minimum < 1:
        raise ValueError("min_detections_for_visibility must be positive")
    for class_id, label in enumerate(("raw", "ripe")):
        source = Path(cfg["input_csvs"][label])
        with source.open("r", newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows or not required.issubset(rows[0]):
            raise ValueError("{} must contain: {}".format(source, ", ".join(sorted(required))))
        sources.append(str(source))
        for row in rows:
            ripe_scores = [float(value) for value in ast.literal_eval(row["Ripe_scores"])]
            raw_scores = [float(value) for value in ast.literal_eval(row["Raw_scores"])]
            observation = tree_observation_scores(
                ripe_scores,
                raw_scores,
                minimum_score=float(cfg.get("minimum_detection_score", 0.0)),
                minimum_tree_detections=minimum,
                evidence_count_midpoint=cfg.get("evidence_count_midpoint"),
                evidence_count_steepness=float(cfg.get("evidence_count_steepness", 1.0)),
            )
            visible = float(np.all(np.isfinite(observation)))
            p_ripe = float(observation[0]) if visible else 0.5
            features.append([float(row[k]) for k in ("x", "y", "yaw")])
            # Last two values are training-only masks for the physical class.
            targets.append([visible, max(0.5, 1.0 - p_ripe), max(0.5, p_ripe),
                            float(class_id == 0), float(class_id == 1)])
    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32), sources


def select_model_targets(target, cfg):
    """Keep the loss target layout consistent for every model width.

    Scalar conditional models use their physical-class mask in column 0 and
    selected accuracy in column 1. Wider models retain the shared structured
    target layout.
    """
    output_dim = int(cfg.get("output_dim", 3))
    if cfg.get("model_type") == "class_conditioned":
        rows = []
        for visible, accuracy_raw, accuracy_ripe, is_raw, is_ripe in target:
            observation_row = ([accuracy_raw, 1.0 - accuracy_raw] if is_raw > 0.5
                               else [1.0 - accuracy_ripe, accuracy_ripe])
            rows.append([visible, *observation_row])
        return np.asarray(rows, dtype=np.float32)
    if output_dim == 4:
        rows = []
        for visible, accuracy_raw, accuracy_ripe, is_raw, is_ripe in target:
            # Invisible samples are deliberately uninformative without a
            # dedicated no-detection class.
            raw_row = [0.5, 0.5] if visible < 0.5 else [
                accuracy_raw, 1.0 - accuracy_raw
            ]
            ripe_row = [0.5, 0.5] if visible < 0.5 else [
                1.0 - accuracy_ripe, accuracy_ripe
            ]
            rows.append([*raw_row, *ripe_row])
        return np.asarray(rows, dtype=np.float32)
    if output_dim == 1:
        label = str(cfg.get("single_output_label", "accuracy_raw"))
        target_columns = {
            "visibility": 0,
            "accuracy_raw": 1,
            "accuracy_ripe": 2,
        }
        if label not in target_columns:
            raise ValueError("unsupported scalar output label '{}'".format(label))
        if label == "visibility":
            class_mask = np.ones(len(target), dtype=target.dtype)
        else:
            class_mask = target[:, 3 if label == "accuracy_raw" else 4]
        return np.column_stack([class_mask, target[:, target_columns[label]]])
    return target[:, :3]


def main(config_path):
    root = load_config(config_path)
    cfg = root["training"]
    seed = int(root.get("seed", 42))
    seed_everything(seed)
    device = resolve_device(str(root.get("device", "auto")))
    x, target, sources = load_training_rows(cfg, root)
    if cfg.get("model_type") == "class_conditioned":
        # Physical class is a model input, not a separate output head.
        x = np.column_stack([x, target[:, 4]]).astype(np.float32)
    original_count = len(x)
    output_dim = int(cfg.get("output_dim", 3))
    model_target = select_model_targets(target, cfg)
    #x, model_target = augment(x, model_target, float(cfg["augment_fraction"]),
    #                          float(cfg["augment_distance_margin"]), np.random.default_rng(seed))
    masks = target[:, 3:5]
    if output_dim == 4:
        # A binary model has no no-detection outcome.  Invisible samples must
        # therefore be excluded from the semantic cross-entropy instead of
        # overwhelming the visible views with neutral [0.5, 0.5] targets.
        masks = masks * target[:, 0:1]
    masks = np.concatenate([masks, np.zeros((len(x) - original_count, 2), dtype=np.float32)])
    target = np.column_stack([model_target, masks])
    if output_dim == 4:
        informative_weight = float(cfg.get("informative_loss_weight", 1.0))
        likelihood = model_target.reshape(-1, 2, 2)
        semantic_strength = np.max(likelihood, axis=2)
        row_weights = 1.0 + (informative_weight - 1.0) * semantic_strength
        target = np.column_stack([target, row_weights.astype(np.float32)])
    checkpoint, loss, final_train_loss, final_validation_loss = train_model(
        x, target, cfg, device, seed
    )
    output_labels = ["visibility", "accuracy_raw", "accuracy_ripe"]
    if cfg.get("model_type") == "class_conditioned":
        output_labels = ["observed_raw", "observed_ripe"]
    if output_dim == 4:
        output_labels = [
            "raw_raw", "raw_ripe", "ripe_raw", "ripe_ripe",
        ]
    elif output_dim == 2:
        output_labels = ["accuracy_raw", "accuracy_ripe"]
    elif output_dim == 1:
        output_labels = [cfg.get("single_output_label", "accuracy_raw")]
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(), "sources": sources,
        "device": str(device), "seed": seed,
        "min_detections_for_visibility": int(cfg.get("min_detections_for_visibility", 5)),
        "hidden_size": int(cfg["hidden_size"]),
        "hidden_layers": int(cfg["hidden_layers"]),
        "yaw_harmonics": int(cfg.get("yaw_harmonics", 1)),
        "include_alignment_features": bool(cfg.get("include_alignment_features", False)),
        "output_temperature": float(cfg.get("output_temperature", 1.0)),
        "yaw_threshold_deg": float(cfg.get("yaw_threshold_deg", 30.0)),
        "yaw_gate_slope": float(cfg.get("yaw_gate_slope", 50.0)),
        "informative_loss_weight": float(cfg.get("informative_loss_weight", 1.0)),
        "output_labels": output_labels,
        "model_type": str(cfg.get("model_type", "legacy")),
        "class_encoding": {"raw": 0, "ripe": 1} if cfg.get("model_type") == "class_conditioned" else None,
        "architecture": str(cfg.get("architecture", "resnet")),
        "ode_steps": int(cfg.get("ode_steps", 3)),
        "ode_dt": float(cfg.get("ode_dt", 0.25)),
        "checkpoint": checkpoint, "best_validation_loss": loss,
        "final_train_loss": final_train_loss,
        "final_validation_loss": final_validation_loss,
    }
    metadata_path = Path(cfg["output_dir"]) / "training_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Wrote training metadata to {}".format(metadata_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    main(parser.parse_args().config)
