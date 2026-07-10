"""Native CasADi implementation of the trained structured perception MLP."""

from pathlib import Path
import subprocess
import sys

import casadi as ca
import numpy as np


def export_checkpoint_cache(checkpoint, cache_path):
    """Export a PyTorch state dict to NumPy in a subprocess.

    Keeping Torch out of the parent process avoids Windows OpenMP runtime
    conflicts when IPOPT is loaded by CasADi.
    """
    if cache_path.is_file() and cache_path.stat().st_mtime >= checkpoint.stat().st_mtime:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    script = r"""
import sys
import numpy as np
import torch

checkpoint = sys.argv[1]
cache_path = sys.argv[2]
state = torch.load(checkpoint, map_location="cpu")
arrays = {
    key: value.detach().cpu().numpy().astype(np.float64)
    for key, value in state.items()
    if hasattr(value, "detach")
}
np.savez_compressed(cache_path, **arrays)
"""
    subprocess.run(
        [sys.executable, "-c", script, str(checkpoint), str(cache_path)],
        check=True,
    )


def load_trained_weights(path, cache_path=None):
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise FileNotFoundError("MLP checkpoint not found: {}".format(checkpoint))
    if cache_path is None:
        cache_path = checkpoint.with_suffix(checkpoint.suffix + ".npz")
    cache_path = Path(cache_path)
    export_checkpoint_cache(checkpoint, cache_path)
    with np.load(cache_path) as state:
        return {key: state[key].astype(float) for key in state.files}


def ca_linear(x, weight, bias):
    return ca.mtimes(x, ca.DM(weight).T) + ca.repmat(ca.DM(bias).T, x.size1(), 1)


def ca_gelu(x):
    return 0.5 * x * (1.0 + ca.erf(x / np.sqrt(2.0)))


def ca_layer_norm(x, weight, bias, eps=1e-5):
    mean = ca.sum2(x) / x.size2()
    centered = x - ca.repmat(mean, 1, x.size2())
    variance = ca.sum2(centered ** 2) / x.size2()
    normalized = centered / ca.sqrt(ca.repmat(variance + eps, 1, x.size2()))
    return (
        normalized * ca.repmat(ca.DM(weight).T, x.size1(), 1)
        + ca.repmat(ca.DM(bias).T, x.size1(), 1)
    )


def ca_softmax_rows(x):
    exp_x = ca.exp(x)
    return exp_x / ca.repmat(ca.sum2(exp_x), 1, x.size2())


def encode_pose_casadi(features, yaw_harmonics=4, include_alignment_features=True):
    xy = features[:, 0:2]
    yaw = features[:, 2]
    encoded = [xy]
    for harmonic in range(1, int(yaw_harmonics) + 1):
        angle = harmonic * yaw
        encoded.extend([ca.sin(angle), ca.cos(angle)])
    if include_alignment_features:
        distance = ca.sqrt(features[:, 0] ** 2 + features[:, 1] ** 2 + 1e-6)
        direction_x = -features[:, 0] / distance
        direction_y = -features[:, 1] / distance
        heading_x = ca.cos(yaw)
        heading_y = ca.sin(yaw)
        facing = heading_x * direction_x + heading_y * direction_y
        lateral = heading_x * direction_y - heading_y * direction_x
        encoded.extend([distance, direction_x, direction_y, facing, lateral])
    return ca.horzcat(*encoded)


def trained_mlp_sensor(
    batch_size,
    checkpoint_path,
    weights_cache_path=None,
    output_temperature=0.4,
    yaw_harmonics=4,
    include_alignment_features=True,
):
    state = load_trained_weights(checkpoint_path, weights_cache_path)
    features = ca.MX.sym("trained_mlp_features", int(batch_size), 3)
    pose_xy = features[:, 0:2]
    x = encode_pose_casadi(
        features,
        yaw_harmonics=yaw_harmonics,
        include_alignment_features=include_alignment_features,
    )
    h = ca_linear(x, state["input_layer.weight"], state["input_layer.bias"])
    block_count = len(
        {
            key.split(".")[1]
            for key in state
            if key.startswith("blocks.") and key.endswith(".fc1.weight")
        }
    )
    for block_idx in range(block_count):
        prefix = "blocks.{}.".format(block_idx)
        residual = ca_linear(h, state[prefix + "fc1.weight"], state[prefix + "fc1.bias"])
        residual = ca_gelu(residual)
        residual = ca_linear(
            residual,
            state[prefix + "fc2.weight"],
            state[prefix + "fc2.bias"],
        )
        h = ca_layer_norm(
            h + residual,
            state[prefix + "norm.weight"],
            state[prefix + "norm.bias"],
        )
    h = ca_layer_norm(h, state["norm.weight"], state["norm.bias"])
    logits = ca_linear(h, state["out_layer.weight"], state["out_layer.bias"])

    temperature = max(float(output_temperature), 1e-3)
    learned_raw = ca_softmax_rows(logits[:, 0:3] / temperature)
    learned_ripe = ca_softmax_rows(logits[:, 3:6] / temperature)
    distance = ca.sqrt(pose_xy[:, 0] ** 2 + pose_xy[:, 1] ** 2 + 1e-6)
    threshold = float(np.asarray(state.get("threshold", 5.0)).reshape(-1)[0])
    gate_slope = float(np.asarray(state.get("gate_slope", 10.0)).reshape(-1)[0])
    gate = 1.0 / (1.0 + ca.exp(-gate_slope * (threshold - distance)))
    gate_col = ca.reshape(gate, int(batch_size), 1)
    raw = ca.horzcat(
        (1.0 - gate_col) + gate_col * learned_raw[:, 0],
        gate_col * learned_raw[:, 1],
        gate_col * learned_raw[:, 2],
    )
    ripe = ca.horzcat(
        (1.0 - gate_col) + gate_col * learned_ripe[:, 0],
        gate_col * learned_ripe[:, 1],
        gate_col * learned_ripe[:, 2],
    )
    return ca.Function(
        "trained_mlp_sensor_{}".format(int(batch_size)),
        [features],
        [ca.horzcat(raw, ripe)],
    )
