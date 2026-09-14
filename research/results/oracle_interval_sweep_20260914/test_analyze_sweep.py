import unittest

from research.results.oracle_interval_sweep_20260914.analyze_sweep import (
    age_step_stats, leave_one_prompt_out, paired_model_comparisons, stage_age_stats,
)


class SweepAnalysisTests(unittest.TestCase):
    def test_lopo_keeps_prompt_group_and_recovers_combined_signal(self):
        rows = []
        for prompt in range(10):
            for interval in (2, 3, 4, 5):
                for age in range(1, interval):
                    distance = age * (0.1 + prompt * 0.01 + interval * 0.005)
                    rows.append(dict(prompt_id=f"p{prompt}", interval=interval,
                                     index=age, refresh=False, age=age,
                                     lambda_distance=distance,
                                     guided_noise_error=0.2 + age * 0.3 + distance * 2,
                                     boundary_error=0.1 + age * 0.2 + distance))
        results, predictions = leave_one_prompt_out(rows)
        combined = next(r for r in results if r["target"] == "guided_noise_error"
                        and r["model"] == "age+lambda_distance")
        age_only = next(r for r in results if r["target"] == "guided_noise_error"
                        and r["model"] == "age")
        self.assertLess(combined["rmse"], 1e-12)
        self.assertGreater(age_only["rmse"], 0.01)
        self.assertEqual(set(combined["folds"]), {f"p{i}" for i in range(10)})
        self.assertEqual(len(predictions), 2 * 4 * len(rows))
        comparisons = paired_model_comparisons(results, draws=100)
        combined_vs_age = next(c for c in comparisons
                               if c["target"] == "guided_noise_error"
                               and c["simple"] == "age"
                               and c["richer"] == "age+lambda_distance")
        self.assertLess(combined_vs_age["delta_rmse"], 0)
        self.assertLess(combined_vs_age["prompt_bootstrap_95ci"][1], 0)

    def test_refresh_zeros_do_not_enter_age_step_means(self):
        rows = [dict(refresh=True, age=0, index=1, guided_noise_error=0),
                dict(refresh=False, age=1, index=1, guided_noise_error=2),
                dict(refresh=False, age=1, index=1, guided_noise_error=4)]
        self.assertEqual(age_step_stats(rows),
                         [dict(age=1, step=1, n=2, mean=3.0, median=3.0)])
        rows[1]["boundary_error"] = 5
        rows[2]["boundary_error"] = 7
        stage = stage_age_stats(rows)
        self.assertEqual(next(r for r in stage if r["target"] == "boundary_error"),
                         dict(stage="early", age=1, target="boundary_error",
                              n=2, mean=6.0, median=6.0))


if __name__ == "__main__":
    unittest.main()
