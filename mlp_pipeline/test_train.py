import unittest

import numpy as np
import torch

from mlp_pipeline.train import mse_loss, select_model_targets, structured_loss
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

    def test_conditional_targets_form_two_observation_rows(self):
        selected = select_model_targets(self.targets, {"output_dim": 6})
        self.assertEqual(selected.shape, (2, 6))
        np.testing.assert_allclose(selected.reshape(-1, 2, 3).sum(axis=2), 1.0)
        np.testing.assert_allclose(selected[1], [1.0, 0.0, 0.0] * 2)

    def test_structured_model_is_nothing_and_neutral_outside_range(self):
        model = MultiLayerPerceptron(
            input_dim=3, hidden_size=8, hidden_layers=1, output_dim=6,
            threshold=5.0, gate_slope=10.0,
        ).eval()
        inputs = torch.tensor([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        output = model(inputs).detach().numpy()

        self.assertTrue(np.all(output >= 0.0))
        self.assertTrue(np.all(output <= 1.0))
        np.testing.assert_allclose(output.reshape(-1, 2, 3).sum(axis=2), 1.0, atol=1e-6)
        np.testing.assert_allclose(
            output[1].reshape(2, 3), [[1.0, 0.0, 0.0]] * 2, atol=1e-6
        )

    def test_structured_loss_uses_optional_informative_row_weights(self):
        prediction = torch.tensor([[0.1, 0.8, 0.1, 0.1, 0.1, 0.8]])
        target = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 4.0, 1.0]])
        weighted = structured_loss(prediction, target, 1.0, 1.0)
        unweighted = structured_loss(prediction, target[:, :8], 1.0, 1.0)
        self.assertAlmostEqual(float(weighted), float(unweighted), places=6)

    def test_yaw_enriched_model_keeps_probability_contract(self):
        model = MultiLayerPerceptron(
            input_dim=3, hidden_size=8, hidden_layers=1, output_dim=6,
            threshold=5.0, gate_slope=10.0, yaw_harmonics=4,
            include_alignment_features=True,
        ).eval()
        output = model(torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, np.pi]]))
        self.assertEqual(output.shape, (2, 6))
        np.testing.assert_allclose(
            output.detach().numpy().reshape(-1, 2, 3).sum(axis=2),
            1.0,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
