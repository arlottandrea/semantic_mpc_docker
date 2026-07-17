#!/usr/bin/env python3
"""Plot dataset/model diagonal fits with one architecture per row."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from mlp_pipeline.visualize import infer, load_model, maximum_over_yaw, plot_map, read_dataset


DEFAULT_VARIANTS = ("simple_small", "simple_large", "resnet", "neural_ode", "enhanced")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-dir", default="outputs/class_conditioned_ablation")
    parser.add_argument("--output", default=None)
    parser.add_argument("--variants", nargs="*", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()
    root = Path(args.ablation_dir)
    output = Path(args.output) if args.output else root / "dataset_model_fit_comparison.png"

    first_cfg = yaml.safe_load((root / args.variants[0] / "train.yaml").read_text(encoding="utf-8"))
    training = first_cfg["training"]
    minimum = int(training.get("min_detections_for_visibility", 5))
    raw_features, raw_target = read_dataset(training["input_csvs"]["raw"], "raw", first_cfg, minimum, 2)
    ripe_features, ripe_target = read_dataset(training["input_csvs"]["ripe"], "ripe", first_cfg, minimum, 2)
    raw_target = np.column_stack([raw_target[:, 0], 1.0 - raw_target[:, 0]])
    ripe_target = np.column_stack([1.0 - ripe_target[:, 1], ripe_target[:, 1]])
    raw_xy, raw_target = maximum_over_yaw(raw_features, raw_target)
    ripe_xy, ripe_target = maximum_over_yaw(ripe_features, ripe_target)

    fig, axes = plt.subplots(len(args.variants), 4, figsize=(16, 3.55 * len(args.variants)), constrained_layout=True)
    mappable = None
    for row, variant in enumerate(args.variants):
        cfg_path = root / variant / "train.yaml"
        cfg_root = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        cfg = cfg_root["training"]
        checkpoints = list((root / variant / "model").glob("best_model_epoch_*.pth"))
        if len(checkpoints) != 1:
            raise RuntimeError("expected one checkpoint for {}".format(variant))
        model = load_model(cfg, checkpoints[0])
        raw_input = np.column_stack([raw_features, np.zeros(len(raw_features), dtype=np.float32)])
        ripe_input = np.column_stack([ripe_features, np.ones(len(ripe_features), dtype=np.float32)])
        raw_prediction = infer(model, raw_input, args.batch_size)
        ripe_prediction = infer(model, ripe_input, args.batch_size)
        _, raw_prediction = maximum_over_yaw(raw_features, raw_prediction)
        _, ripe_prediction = maximum_over_yaw(ripe_features, ripe_prediction)

        panels = (
            (raw_xy, raw_target[:, 0], "raw dataset", False),
            (raw_xy, raw_prediction[:, 0], "raw inference", True),
            (ripe_xy, ripe_target[:, 1], "ripe dataset", False),
            (ripe_xy, ripe_prediction[:, 1], "ripe inference", True),
        )
        for col, (features, values, title, continuous) in enumerate(panels):
            mappable = plot_map(axes[row, col], features, values, title, continuous=continuous)
            if col == 0:
                axes[row, col].set_ylabel("{}\ny relative to tree [m]".format(variant))
    fig.colorbar(mappable, ax=axes, label="P(correct observation | true class)", shrink=0.75)
    fig.suptitle("Class-conditioned surrogate ablation — dataset versus inference", fontsize=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print("Wrote {}".format(output))


if __name__ == "__main__":
    main()
