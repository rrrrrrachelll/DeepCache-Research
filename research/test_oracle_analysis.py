import unittest

import numpy as np

from research.analyze_oracle import correlation, within_step_ranks, summarize


class AnalysisTests(unittest.TestCase):
    def test_constant_predictor_is_undefined(self):
        self.assertIsNone(correlation([1, 1, 1], [1, 2, 3]))
        self.assertIsNone(within_step_ranks(np.array([1, 1, 2, 2]),
                                          np.array([1, 2, 3, 4]), np.array([0, 0, 1, 1])))

    def test_stage_trend_can_mask_negative_content_relation(self):
        steps = np.repeat(np.arange(5), 4)
        variation = np.tile(np.arange(4), 5)
        x = steps*10+variation
        y = steps*10-variation
        self.assertGreater(correlation(x, y), .8)
        self.assertAlmostEqual(within_step_ranks(x, y, steps), -1)

    def test_refresh_rows_excluded(self):
        rows = []
        for prompt in ["a", "b"]:
            for i in range(4):
                rows.append(dict(prompt_id=prompt, index=i, refresh=i == 0, age=i,
                                 conv_previous=i, conv_cache=i, delta_lambda=i,
                                 lambda_distance=i, boundary_error=i,
                                 mid_error=i, guided_noise_error=i))
        stats = summarize(rows, bootstrap=10)
        self.assertEqual(len(stats), 15)
        self.assertTrue(all(s["n"] == 6 for s in stats))
        self.assertTrue(all(abs(s["spearman"]-1) < 1e-10 for s in stats))


if __name__ == "__main__":
    unittest.main()
