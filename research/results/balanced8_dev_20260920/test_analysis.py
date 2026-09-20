import unittest

from research.results.balanced8_dev_20260920.analyze_balanced import (
    grouped_difference, quality_difference,
)


class BalancedAnalysisTests(unittest.TestCase):
    def test_quality_direction(self):
        self.assertAlmostEqual(quality_difference(.8, .7, "ssim"), .1)
        self.assertAlmostEqual(quality_difference(20, 19, "psnr_db"), 1)
        self.assertAlmostEqual(quality_difference(.2, .3, "mean_absolute_rgb_error"), .1)

    def test_prompt_grouping(self):
        rows = []
        for prompt in range(10):
            for seed in (101, 202):
                case = f"{prompt}_{seed}"
                rows += [dict(case=case, prompt_id=str(prompt), mode="a", ssim=.81),
                         dict(case=case, prompt_id=str(prompt), mode="b", ssim=.80)]
        result = grouped_difference(rows, "a", "b", "ssim", draws=100)
        self.assertAlmostEqual(result["mean_difference"], .01)
        self.assertEqual(result["prompts_better"], 10)


if __name__ == "__main__":
    unittest.main()
