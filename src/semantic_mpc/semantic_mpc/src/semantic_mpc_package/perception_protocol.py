"""Pure helpers for the binary ripe/raw perception boundary.

The wire format is one row per tree in ``[ripe, raw]`` order.  ``[nan, nan]``
means that no associated fruit evidence was available.  It is an update mask,
not a third observation class.
"""

import numpy as np


INVALID_SCORE_ROW = np.asarray([np.nan, np.nan], dtype=float)


def categorical_tree_scores(associated_fruits, tree_count, minimum_score=0.0):
    """Aggregate associated detections into categorical ripe/raw score rows."""
    scores = np.full((int(tree_count), 2), np.nan, dtype=float)
    for tree_index, fruits in associated_fruits.items():
        if tree_index < 0 or tree_index >= int(tree_count):
            continue
        ripe = np.asarray(fruits.get("ripe", []), dtype=float)
        raw = np.asarray(fruits.get("raw", []), dtype=float)
        ripe = ripe[np.isfinite(ripe) & (ripe >= float(minimum_score))]
        raw = raw[np.isfinite(raw) & (raw >= float(minimum_score))]
        mass = np.asarray([ripe.sum(), raw.sum()], dtype=float)
        total = float(mass.sum())
        if total > 0.0:
            scores[tree_index] = mass / total
    return scores
