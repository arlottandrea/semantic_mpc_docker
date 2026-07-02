import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "semantic_mpc" / "semantic_mpc" / "src"))
sys.path.insert(0, str(ROOT))

from mlp_pipeline.common import weighted_detection_score
from mlp_pipeline.train import augment
from semantic_mpc_package.perception_model import MultiLayerPerceptron


def test_shared_model_returns_two_independent_probabilities():
    output = MultiLayerPerceptron()(torch.zeros(4, 3))
    assert output.shape == (4, 2)
    assert torch.all((output >= 0.0) & (output <= 1.0))


def test_empty_detection_score_is_neutral_after_generator_offset():
    centered_score = weighted_detection_score([], midpoint=5, steepness=10.0)
    assert centered_score + 0.5 == 0.5


def test_augmentation_is_deterministic_and_neutral():
    x = np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.5]], dtype=np.float32)
    y = np.asarray([[0.8, 0.6], [0.7, 0.9]], dtype=np.float32)
    augmented_x, augmented_y = augment(x, y, 1.0, 5.0, np.random.default_rng(4))
    assert augmented_x.shape == (4, 3)
    np.testing.assert_allclose(augmented_y[-2:], 0.5)
