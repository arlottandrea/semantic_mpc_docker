import numpy as np


def _binary_entropy(values):
    values = np.clip(np.asarray(values, dtype=float), 1e-12, 1.0 - 1e-12)
    return -(values * np.log2(values) + (1.0 - values) * np.log2(1.0 - values))


def expected_information_gain(priors, observation_accuracy=0.9):
    """Return one-step EIG for binary beliefs under a symmetric sensor model.

    ``observation_accuracy`` is P(z=true class). Both possible observations are
    marginalized, using the same binary Bayes convention as the runtime.
    """
    priors = np.asarray(priors, dtype=float)
    accuracy = float(observation_accuracy)
    if not 0.5 < accuracy < 1.0:
        raise ValueError("observation_accuracy must be between 0.5 and 1.0")

    expected_entropy = np.zeros_like(priors, dtype=float)
    for observation_is_class1 in (False, True):
        likelihood_class1 = accuracy if observation_is_class1 else 1.0 - accuracy
        likelihood_class0 = 1.0 - likelihood_class1
        probability = priors * likelihood_class1 + (1.0 - priors) * likelihood_class0
        posterior = np.divide(
            priors * likelihood_class1,
            probability,
            out=priors.copy(),
            where=probability > 1e-12,
        )
        expected_entropy += probability * _binary_entropy(posterior)
    return np.maximum(0.0, _binary_entropy(priors) - expected_entropy)


def select_greedy_ig_target(
    tree_positions,
    current_pose,
    lambda_values,
    active_target_count,
    belief_tracking_threshold,
    observation_accuracy=0.9,
):
    """Select the highest-EIG tree from the nearest active untracked set."""
    positions = np.asarray(tree_positions, dtype=float)
    beliefs = np.asarray(lambda_values, dtype=float)
    if len(positions) != len(beliefs):
        raise ValueError("tree_positions and lambda_values must have equal length")
    if len(positions) == 0 or active_target_count <= 0:
        return None

    current_position = np.asarray(current_pose[:2], dtype=float)
    confidence = np.maximum(beliefs, 1.0 - beliefs)
    untracked = np.flatnonzero(confidence < float(belief_tracking_threshold))
    distances = np.linalg.norm(positions - current_position, axis=1)
    nearest = sorted(untracked, key=lambda idx: (float(distances[idx]), int(idx)))[
        : int(active_target_count)
    ]
    if not nearest:
        return None

    scores = expected_information_gain(beliefs[nearest], observation_accuracy)
    return min(
        zip(nearest, scores),
        key=lambda item: (-float(item[1]), float(distances[item[0]]), int(item[0])),
    )[0]


def generate_greedy_order(tree_positions, current_pose, lambda_values):
    if len(tree_positions) == 0:
        return []

    current_pos = np.asarray(current_pose[:2], dtype=float)
    candidates = list(range(len(tree_positions)))
    order = []

    while candidates:
        next_idx = min(
            candidates,
            key=lambda idx: (
                float(lambda_values[idx]),
                np.linalg.norm(tree_positions[idx] - current_pos),
            ),
        )
        order.append(next_idx)
        current_pos = tree_positions[next_idx]
        candidates.remove(next_idx)

    return order
