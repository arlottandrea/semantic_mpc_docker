# Runtime models

Git LFS stores the YOLO and RL model binaries. After cloning, run `git lfs pull` or `./scripts/setup.sh`.

Expected layout:

```text
models/
├── yolo/apples.pt
├── rl/final_model.zip
└── nmpc/
    ├── ripe/best_model_epoch_<N>.pth
    └── raw/best_model_epoch_<N>.pth
```

The current workspace contains valid YOLO and RL files. Valid NMPC `ripe` and `raw` checkpoints were not present when this superproject was created, so they must be added before the NMPC profile can start. Add them through Git LFS and extend `checksums.sha256`.
