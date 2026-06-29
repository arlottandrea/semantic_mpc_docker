import numpy as np


def generate_linear_order(tree_positions, current_pose, same_row_tol=1e-2):
    if len(tree_positions) == 0:
        return []

    remaining = list(range(len(tree_positions)))
    current_pos = np.asarray(current_pose[:2], dtype=float)
    order = []

    while remaining:
        same_row = [
            idx for idx in remaining
            if abs(tree_positions[idx][1] - current_pos[1]) < same_row_tol
        ]
        if same_row:
            next_idx = min(same_row, key=lambda idx: abs(tree_positions[idx][0] - current_pos[0]))
        else:
            next_idx = min(
                remaining,
                key=lambda idx: np.linalg.norm(current_pos - tree_positions[idx]),
            )
        order.append(next_idx)
        current_pos = tree_positions[next_idx]
        remaining.remove(next_idx)

    return order
