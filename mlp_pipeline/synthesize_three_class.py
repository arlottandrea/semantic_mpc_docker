#!/usr/bin/env python3
"""Align ripe/raw pose tables and write synthesized three-class targets."""

import argparse
import ast
import csv

import numpy as np

from semantic_mpc_package.active_sensing import synthesize_three_class


POSE_COLUMNS = ("x", "y", "yaw")


def keyed_rows(path):
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return {tuple(float(row[column]) for column in POSE_COLUMNS): row for row in rows}


def independent_detection_probability(value):
    """Probability of at least one detection from a confidence list."""
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        parsed = float(value)
    if isinstance(parsed, (list, tuple)):
        scores = np.clip(np.asarray(parsed, dtype=float), 0.0, 1.0)
        return float(1.0 - np.prod(1.0 - scores)) if len(scores) else 0.0
    return float(np.clip(parsed, 0.0, 1.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ripe-csv", required=True)
    parser.add_argument("--raw-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ripe-column", default="Ripe_scores")
    parser.add_argument("--raw-column", default="Raw_scores")
    parser.add_argument(
        "--raw-column-is-p-ripe",
        action="store_true",
        help="Convert the raw dataset's P(ripe) column to P(raw)=1-P(ripe).",
    )
    args = parser.parse_args()
    ripe_rows = keyed_rows(args.ripe_csv)
    raw_rows = keyed_rows(args.raw_csv)
    keys = sorted(set(ripe_rows).intersection(raw_rows))
    if not keys:
        raise ValueError("ripe and raw datasets have no aligned poses")
    with open(args.output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([*POSE_COLUMNS, "p_ripe", "p_raw", "p_nothing"])
        for key in keys:
            ripe = independent_detection_probability(ripe_rows[key][args.ripe_column])
            raw = independent_detection_probability(raw_rows[key][args.raw_column])
            if args.raw_column_is_p_ripe:
                raw = 1.0 - raw
            distribution = synthesize_three_class(ripe, raw)
            writer.writerow([*key, *distribution.tolist()])
    print("Wrote {} aligned poses to {}".format(len(keys), args.output))


if __name__ == "__main__":
    main()
