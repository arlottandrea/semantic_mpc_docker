import numpy as np

from semantic_mpc_package.perception_protocol import categorical_tree_scores


def test_scores_are_ripe_raw_categorical_and_missing_is_nan_mask():
    scores = categorical_tree_scores(
        {0: {"ripe": [0.9, 0.6], "raw": [0.5]}, 2: {"ripe": [], "raw": [0.8]}},
        4,
    )
    np.testing.assert_allclose(scores[0], [0.75, 0.25])
    np.testing.assert_allclose(scores[2], [0.0, 1.0])
    assert np.all(np.isnan(scores[1]))
    assert np.all(np.isnan(scores[3]))


def test_low_and_nonfinite_detections_do_not_create_observations():
    scores = categorical_tree_scores(
        {0: {"ripe": [np.nan, 0.2], "raw": []}}, 1, minimum_score=0.3
    )
    assert np.all(np.isnan(scores[0]))
