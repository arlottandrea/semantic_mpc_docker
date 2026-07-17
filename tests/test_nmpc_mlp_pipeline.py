import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "semantic_mpc" / "semantic_mpc" / "src"))
sys.path.insert(0, str(ROOT))

from mlp_pipeline.common import weighted_detection_score
from mlp_pipeline.train import augment, resolve_loss_function
from mlp_pipeline.visualize import prepare_inference_targets
from semantic_mpc_package.perception_model import MultiLayerPerceptron


def test_shared_model_returns_structured_probabilities():
    output = MultiLayerPerceptron()(torch.zeros(4, 3))
    assert output.shape == (4, 4)
    assert torch.all((output >= 0.0) & (output <= 1.0))


def test_visibility_gate_rejects_out_of_range_pose():
    output = MultiLayerPerceptron(threshold=5.0, gate_slope=10.0)(
        torch.tensor([[100.0, 0.0, 0.0]])
    )
    rows = output.reshape(1, 2, 2)
    assert torch.allclose(rows, torch.full((1, 2, 2), 0.5), atol=1e-6)


def test_relative_yaw_gate_is_uninformative_at_ninety_degrees():
    output = MultiLayerPerceptron(threshold=5.0, gate_slope=10.0)(
        torch.tensor([[2.0, 0.0, np.pi / 2.0]])
    )
    rows = output.reshape(1, 2, 2)
    assert torch.allclose(rows, torch.full((1, 2, 2), 0.5), atol=1e-5)


def test_empty_detection_score_is_neutral_after_generator_offset():
    centered_score = weighted_detection_score([], midpoint=5, steepness=10.0)
    assert centered_score + 0.5 == 0.5


def test_augmentation_is_deterministic_and_neutral():
    x = np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.5]], dtype=np.float32)
    y = np.asarray([[1.0, 0.8, 0.7], [1.0, 0.7, 0.9]], dtype=np.float32)
    augmented_x, augmented_y = augment(x, y, 1.0, 5.0, np.random.default_rng(4))
    assert augmented_x.shape == (4, 3)
    np.testing.assert_allclose(augmented_y[-2:], [[0.0, 0.5, 0.5]] * 2)


def test_loss_function_can_be_selected_from_config():
    loss_fn = resolve_loss_function({"loss_function": "mse"})
    assert loss_fn is not None


def test_inference_targets_support_multiple_output_heads():
    targets = prepare_inference_targets(np.asarray([[0.0, 0.4, 0.6]], dtype=np.float32), output_dim=3)
    assert targets.shape == (1, 3)
    assert targets[0, 0] == 0.0
