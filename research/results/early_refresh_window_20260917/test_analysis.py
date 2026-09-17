import unittest

from research.results.early_refresh_window_20260917.analyze_results import (
    grouped_difference, quality_difference,
)


class EarlyWindowAnalysisTests(unittest.TestCase):
    def test_quality_direction(self):
        self.assertAlmostEqual(quality_difference(.8, .7, "ssim"), .1)
        self.assertAlmostEqual(quality_difference(20, 19, "psnr_db"), 1)
        self.assertAlmostEqual(quality_difference(.2, .3, "mean_absolute_rgb_error"), .1)

    def test_prompt_grouped_difference(self):
        candidate, reference = [], []
        for prompt in range(10):
            for seed in (101, 202):
                case = f"{prompt}_{seed}"
                candidate.append(dict(case=case, prompt_id=str(prompt), mode="early1", ssim=.8))
                reference.append(dict(case=case, prompt_id=str(prompt), ssim=.7))
        result = grouped_difference(candidate, reference, "early1", "ssim", draws=100)
        self.assertAlmostEqual(result["mean_difference"], .1)
        self.assertEqual(result["prompts_better"], 10)
        self.assertAlmostEqual(result["prompt_bootstrap_95ci"][0], .1)
        self.assertAlmostEqual(result["prompt_bootstrap_95ci"][1], .1)


if __name__ == "__main__":
    unittest.main()
