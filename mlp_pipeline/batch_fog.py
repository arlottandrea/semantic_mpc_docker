#!/usr/bin/env python3
"""Generate YOLO labels, train and visualize all ordered fog captures."""

import argparse
import ast
import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Windows Pixi combines Intel OpenMP (NumPy/PyTorch) and LLVM OpenMP
# (Matplotlib). This is limited to the offline reporting process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import yaml

try:
    from .common import load_config
except ImportError:
    from common import load_config


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ("2m", "3m", "5m",)


def run(command):
    print("RUN", " ".join(str(value) for value in command), flush=True)
    subprocess.run([str(value) for value in command], check=True, cwd=str(ROOT))


def write_config(base, path, manifest, output_csv, raw_csv, ripe_csv, model_dir, variant,
                 scalar_label=None):
    config = json.loads(json.dumps(base))
    if variant in {"3outputs-mlp", "structured3"}:
        config["training"]["output_dim"] = 3
        config["training"]["loss_function"] = "structured"
        config["training"]["single_output_label"] = "accuracy_raw"
    elif variant in {"2outputs-mlp", "two_head"}:
        config["training"]["output_dim"] = 2
        config["training"]["loss_function"] = "headwise_bce"
        config["training"]["single_output_label"] = "accuracy_raw"
    elif variant in {"reformat-mlp-pipeline", "scalar"}:
        config["training"]["output_dim"] = 1
        config["training"]["loss_function"] = "mse"
        config["training"]["single_output_label"] = scalar_label or "accuracy_raw"
    else:
        config["training"]["output_dim"] = int(config["training"].get("output_dim", 3))
    config["dataset"].update({
        "input_manifest": str(manifest),
        "output_csv": str(output_csv),
        "annotations_dir": str(output_csv.parent / "detections"),
        "weights": str(ROOT / "models" / "yolo" / "apples.pt"),
        "yolo_source": str(ROOT / "src" / "yolov7-ros" / "src"),
        "save_annotations": False,
    })
    config["training"]["input_csvs"] = {
        "raw": str(raw_csv),
        "ripe": str(ripe_csv),
    }
    config["training"]["output_dir"] = str(model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def dataset_stats(csv_path, minimum):
    with Path(csv_path).open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    counts, tree_scores = [], []
    for row in rows:
        ripe = ast.literal_eval(row["Ripe_scores"])
        raw = ast.literal_eval(row["Raw_scores"])
        counts.append(len(ripe) + len(raw))
        tree_scores.append(float(row["tree_score"]))
    counts = np.asarray(counts, dtype=float)
    tree_scores = np.asarray(tree_scores, dtype=float)
    visible = counts >= minimum
    return {
        "samples": len(rows),
        "visible_samples": int(visible.sum()),
        "visible_fraction": float(visible.mean()),
        "detection_count_mean": float(counts.mean()),
        "detection_count_median": float(np.median(counts)),
        "detection_count_max": int(counts.max()),
        "tree_score_mean": float(tree_scores.mean()),
        "tree_score_std": float(tree_scores.std()),
        "tree_score_min": float(tree_scores.min()),
        "tree_score_max": float(tree_scores.max()),
    }


def write_summary_plot(report, output_path):
    import matplotlib.pyplot as plt

    names = list(report["scenarios"])
    labels = [name.replace("_", ".") for name in names]
    values = [report["scenarios"][name] for name in names]
    x = np.arange(len(names))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    axes[0].plot(x, [item["best_validation_loss"] for item in values],
                 "o-", color="black", linewidth=2, label="best validation")
    axes[0].plot(x, [item["final_validation_loss"] for item in values],
                 "s--", color="tab:blue", linewidth=2, label="final validation")
    axes[0].set_title("MLP loss by fog capture range")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    axes[1].bar(x - width / 2,
                [100.0 * item["raw_dataset"]["visible_fraction"] for item in values],
                width, color="green", label="raw")
    axes[1].bar(x + width / 2,
                [100.0 * item["ripe_dataset"]["visible_fraction"] for item in values],
                width, color="red", label="ripe")
    axes[1].set_title("Useful observations (at least 5 YOLO detections)")
    axes[1].set_ylabel("dataset [%]")
    axes[1].legend()

    for ax in axes:
        ax.set_xticks(x, labels)
        ax.set_xlabel("capture range")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Fog MLP batch — {}".format(report["variant"]), fontweight="bold")
    fig.savefig(str(output_path), dpi=160)
    plt.close(fig)


def main(args):
    base = load_config(args.config)
    fog_root = Path(args.fog_root)
    run_root = Path(args.run_root or ROOT / "runs" / "fog_mlp_batch" / args.variant)
    model_root = Path(args.model_root or ROOT / "models" / "nmpc" / "fog" / args.variant)
    run_root.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "variant": args.variant,
        "class_layout": "fog/<level>/{raw,ripe}",
        "scenarios": {},
    }
    for scenario in SCENARIOS:
        print("\n=== {} ===".format(scenario), flush=True)
        scenario_dir = fog_root / scenario
        raw_manifest = scenario_dir / "raw" / "dataset.csv"
        ripe_manifest = scenario_dir / "ripe" / "dataset.csv"
        raw_csv = scenario_dir / "raw" / "TreeDatasetCNN.csv"
        ripe_csv = scenario_dir / "ripe" / "TreeDatasetCNN.csv"
        raw_config = run_root / "configs" / "{}_raw.yaml".format(scenario)
        ripe_config = run_root / "configs" / "{}_ripe.yaml".format(scenario)
        model_dir = model_root / scenario
        scalar_variant = args.variant in {"reformat-mlp-pipeline", "scalar"}
        raw_model_dir = model_dir / "raw" if scalar_variant else model_dir
        ripe_model_dir = model_dir / "ripe" if scalar_variant else model_dir
        image_path = run_root / "{}_inference.png".format(scenario)
        visualization_stats_path = run_root / "{}_inference.json".format(scenario)
        write_config(base, raw_config, raw_manifest, raw_csv, raw_csv, ripe_csv,
                     raw_model_dir, args.variant, "accuracy_raw")
        write_config(base, ripe_config, ripe_manifest, ripe_csv, raw_csv, ripe_csv,
                     ripe_model_dir, args.variant, "accuracy_ripe")

        if args.force_generate or not raw_csv.exists():
            run([sys.executable, ROOT / "mlp_pipeline" / "generate.py", "--config", raw_config])
        if args.force_generate or not ripe_csv.exists():
            run([sys.executable, ROOT / "mlp_pipeline" / "generate.py", "--config", ripe_config])
        metadata_path = raw_model_dir / "training_metadata.json"
        if args.force_train or not metadata_path.exists():
            run([sys.executable, args.train_script, "--config", raw_config])
        raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if scalar_variant:
            ripe_metadata_path = ripe_model_dir / "training_metadata.json"
            if args.force_train or not ripe_metadata_path.exists():
                run([sys.executable, args.train_script, "--config", ripe_config])
            ripe_metadata = json.loads(ripe_metadata_path.read_text(encoding="utf-8"))
        else:
            ripe_metadata = raw_metadata
        metadata = raw_metadata
        checkpoint = Path(metadata["checkpoint"])
        if args.force_visualize or not image_path.exists():
            run([
                sys.executable, args.visualize_script,
                "--config", ripe_config,
                "--raw-csv", raw_csv,
                "--ripe-csv", ripe_csv,
                "--checkpoint", checkpoint,
                "--output", image_path,
                "--stats-output", visualization_stats_path,
            ])
        report["scenarios"][scenario] = {
            "raw_dataset": dataset_stats(raw_csv, int(base["training"].get("min_detections_for_visibility", 2))),
            "ripe_dataset": dataset_stats(ripe_csv, int(base["training"].get("min_detections_for_visibility", 2))),
            "best_validation_loss": float(metadata["best_validation_loss"]),
            "final_train_loss": metadata.get("final_train_loss"),
            "final_validation_loss": metadata.get("final_validation_loss"),
            "checkpoint": str(checkpoint),
            "checkpoints": {
                "raw": raw_metadata["checkpoint"],
                "ripe": ripe_metadata["checkpoint"],
            },
            "visualization": str(image_path),
            "visualization_stats": json.loads(
                visualization_stats_path.read_text(encoding="utf-8")
            ),
        }
        report_path = run_root / "report.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_summary_plot(report, run_root / "summary.png")
    print("\nWrote {}".format(run_root / "report.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "mlp_pipeline" / "config.yaml"))
    parser.add_argument("--fog-root", default=str(ROOT / "datasets" / "fog"))
    parser.add_argument("--variant", default="structured3",
                        help="Artifact namespace, e.g. structured3, two_head, scalar")
    parser.add_argument("--run-root")
    parser.add_argument("--model-root")
    parser.add_argument("--train-script", default=str(ROOT / "mlp_pipeline" / "train.py"),
                        help="Branch-specific trainer implementing --config")
    parser.add_argument("--visualize-script", default=str(ROOT / "mlp_pipeline" / "visualize.py"),
                        help="Branch-specific visualizer implementing the common CLI")
    parser.add_argument("--force-generate", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--force-visualize", action="store_true")
    main(parser.parse_args())
