import unittest

import numpy as np
import torch

from mlp_pipeline.train import mse_loss, select_model_targets
from semantic_mpc_package.perception_model import MultiLayerPerceptron


class SelectModelTargetsTest(unittest.TestCase):
    def setUp(self):
        # visibility, accuracy_raw, accuracy_ripe, is_raw, is_ripe
        self.targets = np.asarray(
            [[1.0, 0.8, 0.2, 1.0, 0.0], [0.0, 0.3, 0.7, 0.0, 1.0]],
            dtype=np.float32,
        )

    def test_scalar_raw_keeps_class_mask_and_raw_accuracy(self):
        selected = select_model_targets(
            self.targets, {"output_dim": 1, "single_output_label": "accuracy_raw"}
        )
        np.testing.assert_allclose(selected, [[1.0, 0.8], [0.0, 0.3]])

    def test_scalar_ripe_keeps_class_mask_and_ripe_accuracy(self):
        selected = select_model_targets(
            self.targets, {"output_dim": 1, "single_output_label": "accuracy_ripe"}
        )
        np.testing.assert_allclose(selected, [[0.0, 0.2], [1.0, 0.7]])

    def test_scalar_mse_uses_accuracy_and_class_mask(self):
        prediction = torch.tensor([[0.8], [0.9]])
        target = torch.tensor([[1.0, 0.8], [0.0, 0.3]])
        self.assertAlmostEqual(float(mse_loss(prediction, target, 1.0, 1.0)), 0.0)

    def test_scalar_mse_balances_neutral_and_informative_samples(self):
        prediction = torch.tensor([[0.6], [0.5], [0.5]])
        target = torch.tensor([[1.0, 0.8], [1.0, 0.5], [1.0, 0.5]])
        # Informative MSE=0.04 and neutral MSE=0, averaged by group, not sample.
        self.assertAlmostEqual(float(mse_loss(prediction, target, 1.0, 1.0)), 0.02)

    def test_accuracy_model_is_neutral_outside_range_and_never_below_half(self):
        model = MultiLayerPerceptron(
            input_dim=3, hidden_size=8, hidden_layers=1, output_dim=1,
            threshold=5.0, gate_slope=10.0,
        ).eval()
        inputs = torch.tensor([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        output = model(inputs).detach().numpy().reshape(-1)

        self.assertTrue(np.all(output >= 0.5))
        self.assertTrue(np.all(output <= 1.0))
        self.assertAlmostEqual(float(output[1]), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
