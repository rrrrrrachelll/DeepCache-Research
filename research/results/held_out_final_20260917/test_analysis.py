import unittest

from research.results.held_out_final_20260917.analyze_final import (
    evaluate_decision, paired_prompt_difference, quality_difference,
)


class FinalAnalysisTests(unittest.TestCase):
    def test_quality_directions(self):
        self.assertAlmostEqual(quality_difference(.8, .7, "ssim"), .1)
        self.assertAlmostEqual(quality_difference(20, 19, "psnr_db"), 1)
        self.assertAlmostEqual(quality_difference(.2, .3, "mean_absolute_rgb_error"), .1)

    def test_prompt_grouping(self):
        rows = []
        for prompt in range(5):
            for seed in (101, 202):
                case = f"{prompt}_{seed}"
                rows += [dict(case=case, prompt_id=str(prompt), mode="a", ssim=.79),
                         dict(case=case, prompt_id=str(prompt), mode="b", ssim=.80)]
        result = paired_prompt_difference(rows, "a", "b", "ssim", draws=100)
        self.assertAlmostEqual(result["mean_difference"], -.01)
        self.assertEqual(result["prompts_worse"], 5)

    def test_frozen_decision(self):
        primary = {"mean_difference": -.005,
                   "by_prompt": {"a": -.01, "b": -.02, "c": -.04, "d": 0, "e": .01}}
        summary = {"drop_middle8": {"mean_full_calls": 8, "mean_seconds": 30},
                   "uniform9": {"mean_seconds": 32}}
        result = evaluate_decision(primary, summary)
        self.assertTrue(result["accepted"])
        primary["by_prompt"]["b"] = -.05
        self.assertFalse(evaluate_decision(primary, summary)["accepted"])


if __name__ == "__main__":
    unittest.main()
