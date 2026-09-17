import unittest

from research.results.early_refresh_window_20260917.early_schedules import (
    MEASURED_SCHEDULES, REFERENCE_MODE, SCHEDULES, validate_schedules,
)
from research.results.stage_aware_v2_dev_20260915.stage_aware_cache import UNIFORM9_REFRESH


class EarlyWindowScheduleTests(unittest.TestCase):
    def test_equal_budget_and_uniform_reference(self):
        self.assertTrue(validate_schedules())
        self.assertEqual(SCHEDULES[REFERENCE_MODE], tuple(UNIFORM9_REFRESH))
        self.assertEqual(set(MEASURED_SCHEDULES), {"early1", "early3", "early4"})
        self.assertTrue(all(len(schedule) == 9 for schedule in SCHEDULES.values()))

    def test_only_second_refresh_changes(self):
        tails = {schedule[2:] for schedule in SCHEDULES.values()}
        self.assertEqual(tails, {(5, 7, 10, 12, 14, 17, 19)})
        self.assertEqual({schedule[1] for schedule in SCHEDULES.values()}, {1, 2, 3, 4})


if __name__ == "__main__":
    unittest.main()
