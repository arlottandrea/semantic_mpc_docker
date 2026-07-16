import unittest

import numpy as np
import torch

from semantic_mpc_package.active_sensing import (
    binary_channel_capacity,
    conditional_fruit_entropy,
    fuse_binary_belief,
    reliability_weighted_eig,
    synthesize_three_class,
    torch_reliability_weighted_eig,
)
from semantic_mpc_package.perception_model import ReliabilityMLP, ThreeClassFieldMLP
from mlp_pipeline.synthesize_three_class import independent_detection_probability


class ActiveSensingTest(unittest.TestCase):
    def test_synthesis_is_normalized_when_scores_overlap(self):
        distribution = synthesize_three_class(0.8, 0.7)
        np.testing.assert_allclose(distribution.sum(), 1.0)
        self.assertEqual(distribution[2], 0.0)

    def test_detection_lists_preserve_empty_space(self):
        self.assertEqual(independent_detection_probability("[]"), 0.0)
        self.assertAlmostEqual(
            independent_detection_probability("[0.8, 0.5]"), 0.9
        )

    def test_nothing_only_has_zero_conditional_entropy(self):
        entropy = conditional_fruit_entropy([[0.0, 0.0, 1.0], [0.25, 0.25, 0.5]])
        np.testing.assert_allclose(entropy, [0.0, 1.0])

    def test_chance_reliability_has_zero_capacity(self):
        np.testing.assert_allclose(binary_channel_capacity([0.5, 1.0]), [0.0, 1.0])
        np.testing.assert_allclose(
            reliability_weighted_eig([[0.5, 0.5, 0.0]], [0.5]), [0.0]
        )

    def test_repeated_bayes_fusion_reduces_entropy(self):
        belief = np.asarray([0.5, 0.5])
        initial = conditional_fruit_entropy([*belief, 0.0])
        for _ in range(3):
            belief = fuse_binary_belief(belief, [0.9, 0.1])
        final = conditional_fruit_entropy([*belief, 0.0])
        self.assertLess(final, initial)

    def test_torch_eig_is_finite_and_differentiable(self):
        probabilities = torch.tensor([[0.4, 0.4, 0.2]], requires_grad=True)
        reliability = torch.tensor([0.9], requires_grad=True)
        value = torch_reliability_weighted_eig(probabilities, reliability).sum()
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.all(torch.isfinite(probabilities.grad)))
        self.assertTrue(torch.all(torch.isfinite(reliability.grad)))

    def test_surrogate_output_contracts(self):
        poses = torch.zeros(4, 3)
        field = ThreeClassFieldMLP()(poses)
        reliability = ReliabilityMLP()(poses)
        np.testing.assert_allclose(field.detach().sum(dim=1), 1.0, atol=1e-6)
        self.assertTrue(torch.all((reliability >= 0.5) & (reliability <= 1.0)))


if __name__ == "__main__":
    unittest.main()
