#!/usr/bin/env python3
"""Compatibility entry point for the YAML-driven ML pipeline."""

import argparse
import sys
from pathlib import Path


PIPELINE_DIR = Path(__file__).resolve().parents[4] / "mlp_pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

from generate import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PIPELINE_DIR / "config.yaml"))
    main(parser.parse_args().config)
