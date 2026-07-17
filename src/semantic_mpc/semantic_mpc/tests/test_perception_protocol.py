import numpy as np

from semantic_mpc_package.perception_protocol import categorical_tree_scores


def test_scores_are_ripe_raw_categorical_and_missing_is_nan_mask():
    scores = categorical_tree_scores(
        {
            0: {"ripe": [0.9, 0.6, 0.8], "raw": [0.5, 0.7]},
            2: {"ripe": [], "raw": [0.8] * 5},
        },
        4,
    )
    # Confidence complements retain uncertainty about the opposite class.
    np.testing.assert_allclose(scores[0], [2.3 + 0.8, 1.2 + 0.7] / np.asarray(5.0))
    np.testing.assert_allclose(scores[2], [0.2, 0.8])
    assert np.all(np.isnan(scores[1]))
    assert np.all(np.isnan(scores[3]))


def test_low_and_nonfinite_detections_do_not_create_observations():
    scores = categorical_tree_scores(
        {0: {"ripe": [np.nan, 0.2], "raw": []}}, 1, minimum_score=0.3
    )
    assert np.all(np.isnan(scores[0]))


def test_too_few_high_confidence_fruits_do_not_define_a_tree_observation():
    scores = categorical_tree_scores(
        {0: {"ripe": [0.99, 0.98, 0.97], "raw": []}},
        1,
        minimum_tree_detections=5,
    )
    assert np.all(np.isnan(scores[0]))


def test_single_label_views_do_not_collapse_reliability_to_one():
    scores = categorical_tree_scores(
        {0: {"ripe": [0.8] * 5, "raw": []}},
        1,
        minimum_tree_detections=5,
    )
    np.testing.assert_allclose(scores[0], [0.8, 0.2])
