#!/usr/bin/env python3
"""Run YOLOv7 over a pose/image manifest and create the NMPC training CSV."""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_SRC = ROOT / "src" / "semantic_mpc" / "semantic_mpc" / "src"
if str(SEMANTIC_SRC) not in sys.path:
    sys.path.insert(0, str(SEMANTIC_SRC))

try:
    from .common import load_config, resolve_device, seed_everything
except ImportError:  # Direct script execution inside the Docker image.
    from common import load_config, resolve_device, seed_everything

from semantic_mpc_package.perception_protocol import tree_observation_scores


def manifest_path(value):
    path = Path(value)
    if path.name != "latest":
        return path
    candidates = list(path.parent.glob("*.csv"))
    if not candidates:
        candidates = list(path.parent.glob("*/*.csv"))
    if not candidates:
        raise FileNotFoundError("no CSV manifest under {}".format(path.parent))
    return max(candidates, key=lambda item: item.stat().st_mtime)


def main(config_path):
    root = load_config(config_path)
    cfg = root["dataset"]
    seed_everything(int(root.get("seed", 42)))
    device = resolve_device(str(root.get("device", "auto")))

    yolo_source = str(Path(cfg["yolo_source"]).resolve())
    if yolo_source not in sys.path:
        sys.path.insert(0, yolo_source)
    from models.experimental import attempt_load
    from utils.datasets import letterbox
    from utils.general import non_max_suppression, scale_coords
    from utils.plots import plot_one_box

    model = attempt_load(str(cfg["weights"]), map_location=device).eval().to(device)
    names = model.module.names if hasattr(model, "module") else model.names
    source = manifest_path(cfg["input_manifest"])
    with source.open("r", newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    required = {"x", "y", "yaw", "image_path"}
    if not records or not required.issubset(records[0]):
        raise ValueError("manifest must contain columns: {}".format(", ".join(sorted(required))))

    output = Path(cfg["output_csv"])
    annotations = Path(cfg["annotations_dir"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if cfg.get("save_annotations", True):
        annotations.mkdir(parents=True, exist_ok=True)

    rows = []
    batch_size = int(cfg["batch_size"])
    started_at = time.monotonic()
    for start in range(0, len(records), batch_size):
        batch_records = records[start:start + batch_size]
        tensors, images, ratio_pads, valid_records = [], [], [], []
        for record in batch_records:
            image_path = Path(record["image_path"])
            if not image_path.is_absolute():
                image_path = source.parent / image_path
            image = cv2.imread(str(image_path))
            if image is None:
                raise FileNotFoundError("cannot read image: {}".format(image_path))
            resized, ratio, pad = letterbox(image, int(cfg["image_size"]), auto=False)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
            tensors.append(torch.from_numpy(np.ascontiguousarray(rgb)).float() / 255.0)
            images.append(image)
            ratio_pads.append((ratio, pad))
            valid_records.append((record, image_path))
        with torch.no_grad():
            prediction = model(torch.stack(tensors).to(device))[0]
        detections = non_max_suppression(
            prediction, float(cfg["confidence_threshold"]), float(cfg["iou_threshold"])
        )
        for record_info, image, tensor, ratio_pad, detection in zip(
                valid_records, images, tensors, ratio_pads, detections):
            record, image_path = record_info
            ripe_scores, raw_scores = [], []
            if detection is not None and len(detection):
                detection[:, :4] = scale_coords(tensor.shape[1:], detection[:, :4], image.shape, ratio_pad)
                for *box, confidence, class_id in detection:
                    label = names[int(class_id)]
                    score = float(confidence.item())
                    if label == cfg["ripe_class"]:
                        ripe_scores.append(score)
                    elif label == cfg["raw_class"]:
                        raw_scores.append(score)
                    if cfg.get("save_annotations", True):
                        plot_one_box(box, image, label="{} {:.2f}".format(label, score), line_thickness=2)
            observation = tree_observation_scores(
                ripe_scores,
                raw_scores,
                minimum_score=float(cfg.get("minimum_detection_score", 0.0)),
                minimum_tree_detections=int(cfg.get("minimum_tree_detections", 5)),
                evidence_count_midpoint=cfg.get("evidence_count_midpoint", cfg.get("score_midpoint")),
                evidence_count_steepness=float(cfg.get("evidence_count_steepness", cfg.get("score_steepness", 1.0))),
            )
            completed = dict(record)
            completed.update({
                "Ripe_scores": repr(ripe_scores),
                "Raw_scores": repr(raw_scores),
                "tree_score": float(observation[0]) if np.all(np.isfinite(observation)) else float("nan"),
            })
            rows.append(completed)
            if cfg.get("save_annotations", True):
                cv2.imwrite(str(annotations / image_path.name), image)
        completed_count = min(start + len(batch_records), len(records))
        if completed_count == len(records) or completed_count % (batch_size * 25) == 0:
            elapsed = max(time.monotonic() - started_at, 1e-6)
            print(
                "YOLO {}/{} images ({:.1f} img/s)".format(
                    completed_count, len(records), completed_count / elapsed
                ),
                flush=True,
            )
    fields = list(records[0].keys())
    for field in ("Ripe_scores", "Raw_scores", "tree_score"):
        if field not in fields:
            fields.append(field)
    temporary_output = output.with_suffix(output.suffix + ".tmp")
    with temporary_output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary_output.replace(output)
    print("Wrote {} samples to {}".format(len(rows), output))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    main(parser.parse_args().config)
