import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import casadi as ca
import numpy as np

def _repo_root():
    path = Path(__file__).resolve()
    for parent in [path] + list(path.parents):
        if (parent / "pixi.toml").is_file():
            return parent
    return None


ROOT = _repo_root()
if ROOT is not None:
    semantic_src = ROOT / "src" / "semantic_mpc" / "semantic_mpc" / "src"
    if str(semantic_src) not in sys.path:
        sys.path.insert(0, str(semantic_src))


from semantic_mpc_package.nmpc_config import default_nmpc_params
from semantic_mpc_package.nmpc_optimizer import NmpcOptimizer
from semantic_mpc_package.casadi_mlp_sensor import trained_mlp_sensor


def _yaw_enriched_checkpoint():
    root = ROOT
    if root is None:
        return None
    checkpoint = root / "models" / "nmpc" / "fog" / "yaw_enriched" / "5m" / "best_model_epoch_39.pth"
    return checkpoint if checkpoint.is_file() else None


def _diagnostic_params(horizon=6):
    params = default_nmpc_params()
    params.update(
        {
            "model_device": "cpu",
            "num_target_trees": 1,
            "num_obstacle_trees": 1,
            "active_target_count": 1,
            "active_obstacle_count": 1,
            "dt": 0.5,
            "mpc_horizon": horizon,
            "safe_distance": 1.0,
            "movement_weight": 1e-3,
            "yaw_movement_weight": 1e-3,
            "acceleration_regularization_weight": 1e-5,
            "information_gain_weight": 3.0,
            "exploration_weight": 0.2,
            "attraction_weight": 1.0,
            "camera_facing_weight": 0.5,
            "observation_standoff_weight": 0.2,
            "running_camera_facing_weight": 0.1,
            "running_observation_standoff_weight": 0.0,
            "orbit_velocity_weight": 0.0,
            "max_velocity": 2.0,
            "max_yaw_velocity": np.pi / 2.0,
            "max_accel_xy": 2.0,
            "max_accel_yaw": np.pi,
            "ipopt": {
                "print_level": 0,
                "sb": "yes",
                "max_iter": 100,
                "tol": 1e-5,
                "max_cpu_time": 5.0,
            },
        }
    )
    return params


def _load_yaw_enriched_model(checkpoint):
    import torch

    from semantic_mpc_package.perception_model import MultiLayerPerceptron

    params = _diagnostic_params()
    model = MultiLayerPerceptron(
        input_dim=int(params["nn_input_dim"]),
        hidden_size=128,
        hidden_layers=4,
        output_dim=int(params["nn_output_dim"]),
        threshold=float(params["nn_threshold"]),
        gate_slope=float(params["nn_gate_slope"]),
        yaw_harmonics=int(params["nn_yaw_harmonics"]),
        include_alignment_features=bool(params["nn_include_alignment_features"]),
        output_temperature=float(params["nn_output_temperature"]),
    )
    model.load_state_dict(torch.load(str(checkpoint), map_location="cpu"))
    model.eval()
    return model


def _expected_entropy_from_structured_output(row):
    row = np.asarray(row, dtype=float).reshape(2, 3)
    raw = row[0]
    ripe = row[1]
    entropy = 0.0
    for observation in range(3):
        ripe_joint = 0.5 * ripe[observation]
        raw_joint = 0.5 * raw[observation]
        probability = ripe_joint + raw_joint
        if probability <= 0.0:
            continue
        posterior = np.asarray([ripe_joint, raw_joint]) / probability
        entropy -= probability * np.sum(
            posterior * np.log2(np.clip(posterior, 1e-9, 1.0))
        )
    return float(entropy)


class NmpcMlpCasadiIntegrationTest(unittest.TestCase):
    def test_00_one_tree_nmpc_uses_trained_mlp_to_move_to_entropy_reducing_pose(self):
        checkpoint = _yaw_enriched_checkpoint()
        if checkpoint is None:
            self.skipTest("yaw-enriched NMPC checkpoint is not present")
        params = _diagnostic_params(horizon=6)
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_cache = Path(tmpdir) / "trained_mlp_weights.npz"
            sensor = trained_mlp_sensor(
                params["mpc_horizon"],
                checkpoint,
                weights_cache_path=weights_cache,
                output_temperature=params["nn_output_temperature"],
                yaw_harmonics=params["nn_yaw_harmonics"],
                include_alignment_features=params["nn_include_alignment_features"],
            )
            update_sensor = trained_mlp_sensor(
                1,
                checkpoint,
                weights_cache_path=weights_cache,
                output_temperature=params["nn_output_temperature"],
                yaw_harmonics=params["nn_yaw_harmonics"],
                include_alignment_features=params["nn_include_alignment_features"],
            )
            optimizer = NmpcOptimizer(params, sensor)
        target = np.asarray([[8.0, 0.0]])
        try:
            _, command, trajectory, _, _ = optimizer.mpc_opt(
                target,
                np.asarray([[0.5, 0.5]]),
                np.ones(1),
                np.asarray([[50.0, 50.0]]),
                np.asarray([-10.0, -10.0]),
                np.asarray([10.0, 10.0]),
                np.zeros(6),
                steps=params["mpc_horizon"],
            )
        except RuntimeError as exc:
            if "Plugin 'ipopt' is not found" in str(exc):
                self.skipTest("CasADi IPOPT plugin is unavailable")
            raise

        command = np.asarray(command, dtype=float).reshape(-1)
        trajectory = np.asarray(trajectory, dtype=float)
        distances = np.linalg.norm(trajectory[:2, :].T - target[0], axis=1)

        self.assertGreater(command[0], 0.0)
        self.assertLess(distances[-1], distances[0])
        self.assertLess(distances[-1], params["observation_range"])

        sensor_output = update_sensor(
            ca.DM(
                [
                    [
                        trajectory[0, -1] - target[0, 0],
                        trajectory[1, -1] - target[0, 1],
                        trajectory[2, -1],
                    ]
                ]
            )
        )
        terminal_entropy = _expected_entropy_from_structured_output(sensor_output)
        self.assertLess(terminal_entropy, 0.8)

    def test_current_checkpoint_metadata_matches_runtime_architecture(self):
        from semantic_mpc_package.nmpc_model import _is_compatible_checkpoint

        checkpoint = _yaw_enriched_checkpoint()
        if checkpoint is None:
            self.skipTest("yaw-enriched NMPC checkpoint is not present")

        self.assertTrue(_is_compatible_checkpoint(str(checkpoint), _diagnostic_params()))

    def test_packaged_legacy_checkpoint_is_not_silently_accepted_by_current_defaults(self):
        from semantic_mpc_package.nmpc_model import _is_compatible_checkpoint

        checkpoint = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "semantic_mpc_package"
            / "models"
            / "best_model_epoch_8.pth"
        )
        if not checkpoint.is_file():
            self.skipTest("packaged legacy checkpoint is not present")

        self.assertFalse(_is_compatible_checkpoint(str(checkpoint), _diagnostic_params()))

    def test_trained_mlp_reduces_expected_entropy_only_for_near_facing_pose(self):
        checkpoint = _yaw_enriched_checkpoint()
        if checkpoint is None:
            self.skipTest("yaw-enriched NMPC checkpoint is not present")
        script = r"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

root = Path(os.environ["NMPC_DIAG_ROOT"])
sys.path.insert(0, str(root / "src" / "semantic_mpc" / "semantic_mpc" / "src"))

from semantic_mpc_package.perception_model import MultiLayerPerceptron

checkpoint = Path(os.environ["NMPC_DIAG_CHECKPOINT"])
model = MultiLayerPerceptron(
    input_dim=3,
    hidden_size=128,
    hidden_layers=4,
    output_dim=6,
    threshold=5.0,
    gate_slope=10.0,
    yaw_harmonics=4,
    include_alignment_features=True,
    output_temperature=0.4,
)
model.load_state_dict(torch.load(str(checkpoint), map_location="cpu"))
model.eval()
features = torch.tensor(
    [
        [8.0, 0.0, 0.0],
        [-4.0, 0.0, 0.0],
        [-4.0, 0.0, np.pi],
    ],
    dtype=torch.float32,
)
with torch.no_grad():
    output = model(features).detach().numpy()

def expected_entropy(row):
    row = np.asarray(row, dtype=float).reshape(2, 3)
    raw = row[0]
    ripe = row[1]
    entropy = 0.0
    for observation in range(3):
        ripe_joint = 0.5 * ripe[observation]
        raw_joint = 0.5 * raw[observation]
        probability = ripe_joint + raw_joint
        if probability <= 0.0:
            continue
        posterior = np.asarray([ripe_joint, raw_joint]) / probability
        entropy -= probability * np.sum(
            posterior * np.log2(np.clip(posterior, 1e-9, 1.0))
        )
    return float(entropy)

far_entropy = expected_entropy(output[0])
facing_entropy = expected_entropy(output[1])
away_entropy = expected_entropy(output[2])
assert far_entropy > 0.95, far_entropy
assert facing_entropy < 0.2, facing_entropy
assert away_entropy > 0.95, away_entropy
"""
        env = os.environ.copy()
        env["NMPC_DIAG_ROOT"] = str(ROOT)
        env["NMPC_DIAG_CHECKPOINT"] = str(checkpoint)
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(ROOT),
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
        )
        self.assertEqual(
            result.returncode,
            0,
            "stdout:\n{}\nstderr:\n{}".format(result.stdout, result.stderr),
        )

    def test_l4casadi_matches_pytorch_mlp_outputs(self):
        checkpoint = _yaw_enriched_checkpoint()
        if checkpoint is None:
            self.skipTest("yaw-enriched NMPC checkpoint is not present")
        script = r"""
import os
import sys
from pathlib import Path

import casadi as ca
import l4casadi as l4c
import numpy as np
import torch

root = Path(os.environ["NMPC_DIAG_ROOT"])
sys.path.insert(0, str(root / "src" / "semantic_mpc" / "semantic_mpc" / "src"))

from semantic_mpc_package.perception_model import MultiLayerPerceptron

checkpoint = Path(os.environ["NMPC_DIAG_CHECKPOINT"])
model = MultiLayerPerceptron(
    input_dim=3,
    hidden_size=128,
    hidden_layers=4,
    output_dim=6,
    threshold=5.0,
    gate_slope=10.0,
    yaw_harmonics=4,
    include_alignment_features=True,
    output_temperature=0.4,
)
model.load_state_dict(torch.load(str(checkpoint), map_location="cpu"))
model.eval()
features = np.asarray(
    [[8.0, 0.0, 0.0], [-4.0, 0.0, 0.0], [-4.0, 0.0, np.pi]],
    dtype=np.float32,
)
with torch.no_grad():
    torch_output = model(torch.tensor(features)).detach().numpy()
casadi_model = l4c.L4CasADi(
    model,
    batched=True,
    device="cpu",
    name="perception_diagnostic_{}".format(os.getpid()),
)
casadi_output = np.asarray(casadi_model(ca.DM(features)), dtype=float)
np.testing.assert_allclose(casadi_output, torch_output, rtol=1e-6, atol=1e-8)
"""
        env = os.environ.copy()
        env["NMPC_DIAG_ROOT"] = str(ROOT)
        env["NMPC_DIAG_CHECKPOINT"] = str(checkpoint)
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(ROOT),
            env=env,
            text=True,
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0 and "No module named 'l4casadi'" in result.stderr:
            self.skipTest("l4casadi is unavailable")
        self.assertEqual(
            result.returncode,
            0,
            "stdout:\n{}\nstderr:\n{}".format(result.stdout, result.stderr),
        )


if __name__ == "__main__":
    unittest.main()
