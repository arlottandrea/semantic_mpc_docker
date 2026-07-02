# YOLO dataset and NMPC model pipeline

This directory replaces the executable parts of the old notebooks and
`tools/dataset/generate_dataset.py`. The single configuration contract is
[`config.yaml`](config.yaml).

## Input contract

The generator reads an initial CSV with these columns:

```csv
x,y,yaw,image_path
-1.2,3.4,0.5,images/frame_0001.png
```

Relative image paths are resolved from the CSV directory. The generator runs
YOLO for every image and adds or replaces these columns:

```csv
Ripe_scores,Raw_scores,tree_score
"[0.91, 0.83]","[0.62]",0.73
```

`Ripe_scores` and `Raw_scores` retain the YOLO confidence lists for auditing.
`tree_score` is computed as `ripe_weight - raw_weight + 0.5` and clipped to
`[0, 1]`. The input and output paths may be identical; the completed CSV is
written to a temporary file and then replaced atomically.

Training consumes this same completed CSV and trains one shared surrogate with
two independent outputs in `[ripe, raw]` order.

## Run with Docker

Edit `config.yaml` if required. Container paths map as follows:

- host `datasets/` -> `/data`
- host `models/` -> `/models`
- host `runs/` -> `/runs`

Then run:

```bash
docker compose --profile pipeline run --rm dataset-generate
docker compose --profile pipeline run --rm model-train
```

For NVIDIA GPU execution, add `-f compose.gpu.yaml`; set `device: cpu` in the
YAML for a CPU-only run. Training writes a runtime-compatible checkpoint to
`models/nmpc/best_model_epoch_<N>.pth` and records provenance in
`models/nmpc/training_metadata.json`.

By default, training removes older `best_model_epoch_*.pth` files in the model
directory. This prevents the runtime's highest-epoch lookup from selecting a
stale checkpoint. Set `training.replace_existing_checkpoints: false` to retain
them, but then manage runtime model selection explicitly.

To use a separate configuration without changing the tracked default, replace
the bind-mounted config path in `compose.yaml` or invoke the image with
`--config /path/to/config.yaml`.

The notebooks remain available as exploratory records, but they are no longer
part of the reproducible generation/training path.
