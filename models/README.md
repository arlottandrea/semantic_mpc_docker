# Runtime models

Git LFS stores the YOLO and RL model binaries. After cloning, run `git lfs pull` or `./scripts/setup.sh`.

Expected layout:

```text
models/
├── yolo/apples.pt
├── rl/final_model.zip
└── nmpc/best_model_epoch_<N>.pth
```

The current workspace contains valid YOLO and RL files. A valid structured NMPC
checkpoint was not present when this superproject was created, so it must be
trained before the NMPC profile can start. Add it through Git LFS and extend
`checksums.sha256` when it is ready for distribution.
