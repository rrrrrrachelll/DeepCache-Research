import unittest

from research.results.stage_aware_v2_dev_20260915.analyze_results import (
    paired_prompt_bootstrap,
)


class AnalysisTests(unittest.TestCase):
    def test_prompt_grouped_difference(self):
        rows = []
        for prompt in range(10):
            for seed in (1, 2):
                rows.extend([
                    dict(prompt_id=str(prompt), mode="stage", ssim=0.8 + prompt/100),
                    dict(prompt_id=str(prompt), mode="uniform", ssim=0.7 + prompt/100),
                ])
        result = paired_prompt_bootstrap(rows, "stage", "uniform", "ssim", draws=100)
        self.assertAlmostEqual(result["mean_difference"], 0.1)
        self.assertEqual(result["prompts_better"], 10)
        self.assertEqual(result["prompts_worse"], 0)
        self.assertAlmostEqual(result["prompt_bootstrap_95ci"][0], 0.1)
        self.assertAlmostEqual(result["prompt_bootstrap_95ci"][1], 0.1)


if __name__ == "__main__":
    unittest.main()
