import numpy as np


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
