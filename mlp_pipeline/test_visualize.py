import unittest

import numpy as np

from mlp_pipeline.visualize import maximum_over_yaw


class MaximumOverYawTest(unittest.TestCase):
    def test_selects_independent_maximum_for_each_head_at_each_xy(self):
        features = np.asarray([
            [1.0, 2.0, 0.0],
            [1.0, 2.0, 1.0],
            [3.0, 4.0, 0.0],
        ])
        values = np.asarray([
            [0.6, 0.9],
            [0.8, 0.7],
            [0.55, 0.65],
        ])

        collapsed_features, maxima = maximum_over_yaw(features, values)

        np.testing.assert_allclose(collapsed_features[:, :2], [[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(maxima, [[0.8, 0.9], [0.55, 0.65]])


if __name__ == "__main__":
    unittest.main()
