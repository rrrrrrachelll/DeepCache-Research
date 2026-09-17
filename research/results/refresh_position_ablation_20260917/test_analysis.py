import unittest

from research.results.refresh_position_ablation_20260917.analyze_results import (
    grouped_benefit,
)


class MarginalBenefitTests(unittest.TestCase):
    def test_grouped_ssim_benefit(self):
        rows, uniform = [], []
        for prompt in range(10):
            for seed in (101, 202):
                case=f"{prompt}_{seed}"
                rows.append(dict(case=case, prompt_id=str(prompt), mode="drop", ssim=.7))
                uniform.append(dict(case=case, prompt_id=str(prompt), ssim=.8))
        result = grouped_benefit(rows, uniform, "drop", "ssim", draws=100)
        self.assertAlmostEqual(result["mean_benefit"], .1)
        self.assertEqual(result["prompts_positive"], 10)
        self.assertAlmostEqual(result["prompt_bootstrap_95ci"][0], .1)
        self.assertAlmostEqual(result["prompt_bootstrap_95ci"][1], .1)

    def test_rgb_error_direction(self):
        rows = [dict(case=f"{p}_{s}", prompt_id=str(p), mode="drop", mean_absolute_rgb_error=.3)
                for p in range(10) for s in range(2)]
        uniform = [dict(case=f"{p}_{s}", prompt_id=str(p), mean_absolute_rgb_error=.2)
                   for p in range(10) for s in range(2)]
        result = grouped_benefit(
            rows, uniform, "drop", "mean_absolute_rgb_error", draws=100)
        self.assertAlmostEqual(result["mean_benefit"], .1)
        self.assertEqual(result["prompts_positive"], 10)


if __name__ == "__main__":
    unittest.main()
