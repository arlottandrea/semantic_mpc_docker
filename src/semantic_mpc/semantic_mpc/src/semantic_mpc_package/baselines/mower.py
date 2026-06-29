import numpy as np


def _inclusive_coords(lo, hi, step):
    n = max(1, int(np.floor((hi - lo) / step + 1e-9)))
    values = lo + step * np.arange(n + 1)
    if values[-1] < hi - 1e-9:
        values = np.append(values, hi)
    else:
        values[-1] = hi
    return values


def generate_mower_path(
    tree_positions,
    current_pose,
    offset=2.0,
    spacing=4.0,
    heading_direction="N",
    axis=None,
    seed=None,
):
    if tree_positions.size == 0:
        return []

    rng = np.random.default_rng(seed)
    direction_map = {"N": np.pi / 2.0, "E": 0.0, "S": -np.pi / 2.0, "W": np.pi, "O": np.pi}
    if heading_direction is None:
        heading = direction_map[rng.choice(list(direction_map.keys()))]
    elif isinstance(heading_direction, str):
        heading = direction_map.get(heading_direction.upper(), 0.0)
    else:
        heading = float(heading_direction)

    x_min = float(np.min(tree_positions[:, 0]) - offset)
    x_max = float(np.max(tree_positions[:, 0]) + offset)
    y_min = float(np.min(tree_positions[:, 1]) - offset)
    y_max = float(np.max(tree_positions[:, 1]) + offset)

    xs_inc = _inclusive_coords(x_min, x_max, spacing)
    ys_inc = _inclusive_coords(y_min, y_max, spacing)
    if axis not in ("x", "y"):
        axis = rng.choice(["x", "y"])

    drone_x, drone_y = current_pose[:2]
    corners = [(x_min, y_min), (x_min, y_max), (x_max, y_min), (x_max, y_max)]
    x_start, y_start = min(corners, key=lambda c: (drone_x - c[0]) ** 2 + (drone_y - c[1]) ** 2)

    xs = xs_inc if x_start == x_min else xs_inc[::-1]
    ys = ys_inc if y_start == y_min else ys_inc[::-1]

    waypoints = []
    if axis == "x":
        for i, y in enumerate(ys):
            row = xs if i % 2 == 0 else xs[::-1]
            waypoints.extend((float(x), float(y), float(heading)) for x in row)
    else:
        for i, x in enumerate(xs):
            col = ys if i % 2 == 0 else ys[::-1]
            waypoints.extend((float(x), float(y), float(heading)) for y in col)

    return waypoints
