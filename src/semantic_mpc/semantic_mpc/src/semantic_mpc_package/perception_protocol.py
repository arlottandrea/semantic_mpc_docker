"""Pure helpers for the binary ripe/raw perception boundary.

The wire format is one row per tree in ``[ripe, raw]`` order.  ``[nan, nan]``
means that no associated fruit evidence was available.  It is an update mask,
not a third observation class.
"""

import numpy as np


INVALID_SCORE_ROW = np.asarray([np.nan, np.nan], dtype=float)


def tree_observation_scores(
    ripe_scores,
    raw_scores,
    minimum_score=0.0,
    minimum_tree_detections=5,
    evidence_count_midpoint=None,
    evidence_count_steepness=1.0,
):
    """Return tree-level evidence ``[ripe, raw]`` from fruit detections.

    The fruit detector is only treated as a tree-level observation when at
    least ``minimum_tree_detections`` valid fruits are associated with the
    tree.  Otherwise ``[nan, nan]`` explicitly masks the Bayesian update and
    the sample's semantic training loss.
    """
    ripe = np.asarray(ripe_scores, dtype=float)
    raw = np.asarray(raw_scores, dtype=float)
    ripe = ripe[np.isfinite(ripe) & (ripe >= float(minimum_score))]
    raw = raw[np.isfinite(raw) & (raw >= float(minimum_score))]
    if len(ripe) + len(raw) < int(minimum_tree_detections):
        return INVALID_SCORE_ROW.copy()
    # Treat detector confidence as soft class evidence rather than discarding
    # its complement.  Otherwise a view containing only detections labelled
    # ripe would produce P(ripe)=1 regardless of whether confidences are 0.51
    # or 0.99 (and symmetrically for raw).
    mass = np.asarray(
        [ripe.sum() + (1.0 - raw).sum(), raw.sum() + (1.0 - ripe).sum()],
        dtype=float,
    )
    total = float(mass.sum())
    if total <= 0.0:
        return INVALID_SCORE_ROW.copy()
    observation = mass / total
    if evidence_count_midpoint is not None:
        # A barely valid handful of fruits is weak tree-level evidence. Blend
        # it toward neutral and approach the detector distribution smoothly as
        # the associated fruit count grows. This is shared by CSV generation,
        # training and the realtime ROS node.
        count = len(ripe) + len(raw)
        strength = 0.5 + 0.5 * np.tanh(
            float(evidence_count_steepness) * (count - float(evidence_count_midpoint))
        )
        observation = 0.5 + strength * (observation - 0.5)
    return observation


def categorical_tree_scores(
    associated_fruits,
    tree_count,
    minimum_score=0.0,
    minimum_tree_detections=5,
    evidence_count_midpoint=None,
    evidence_count_steepness=1.0,
):
    """Aggregate associated detections into categorical ripe/raw score rows."""
    scores = np.full((int(tree_count), 2), np.nan, dtype=float)
    for tree_index, fruits in associated_fruits.items():
        if tree_index < 0 or tree_index >= int(tree_count):
            continue
        scores[tree_index] = tree_observation_scores(
            fruits.get("ripe", []),
            fruits.get("raw", []),
            minimum_score=minimum_score,
            minimum_tree_detections=minimum_tree_detections,
            evidence_count_midpoint=evidence_count_midpoint,
            evidence_count_steepness=evidence_count_steepness,
        )
    return scores
