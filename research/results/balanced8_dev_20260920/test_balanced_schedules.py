import unittest

from research.results.balanced8_dev_20260920.balanced_schedules import (
    DROP_MIDDLE8, SCHEDULES, gaps, validate_schedules,
)


class BalancedScheduleTests(unittest.TestCase):
    def test_strict_equal_budget(self):
        self.assertTrue(validate_schedules())
        self.assertEqual(set(SCHEDULES), {"balanced8_a", "balanced8_b"})
        self.assertTrue(all(len(schedule) == 8 for schedule in SCHEDULES.values()))

    def test_balancing_removes_five_step_gap(self):
        self.assertEqual(max(gaps(DROP_MIDDLE8)), 5)
        self.assertTrue(all(max(gaps(schedule)) == 3 for schedule in SCHEDULES.values()))
        self.assertTrue(all(schedule[:3] == (0, 2, 5) for schedule in SCHEDULES.values()))


if __name__ == "__main__":
    unittest.main()
