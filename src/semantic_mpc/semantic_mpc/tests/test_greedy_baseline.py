import numpy as np

from semantic_mpc_package.baselines.greedy import (
    expected_information_gain,
    select_greedy_ig_target,
)


def test_eig_is_highest_for_most_uncertain_belief():
    scores = expected_information_gain(np.array([0.5, 0.7, 0.9]))
    assert scores[0] > scores[1] > scores[2]


def test_selection_scores_only_nearest_active_untracked_targets():
    positions = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    beliefs = np.array([0.8, 0.6, 0.5])
    selected = select_greedy_ig_target(positions, [0.0, 0.0, 0.0], beliefs, 2, 0.95)
    assert selected == 1


def test_selection_replans_from_updated_belief_and_pose():
    positions = np.array([[1.0, 0.0], [2.0, 0.0], [10.0, 0.0]])
    beliefs = np.array([0.5, 0.6, 0.5])
    first = select_greedy_ig_target(positions, [0.0, 0.0, 0.0], beliefs, 2, 0.95)
    beliefs[first] = 0.99
    second = select_greedy_ig_target(positions, positions[first].tolist() + [0.0], beliefs, 2, 0.95)
    assert first == 0
    assert second == 2
