import unittest

from research.results.refresh_position_ablation_20260917.position_schedules import (
    DROPPED_REFRESH, SCHEDULES, validate_ablation,
)
from research.results.stage_aware_v2_dev_20260915.stage_aware_cache import (
    UNIFORM9_REFRESH,
)


class PositionScheduleTests(unittest.TestCase):
    def test_equal_budget_single_refresh_dropout(self):
        self.assertTrue(validate_ablation())
        for name, schedule in SCHEDULES.items():
            self.assertEqual(len(schedule), 8)
            self.assertEqual(set(UNIFORM9_REFRESH) - set(schedule),
                             {DROPPED_REFRESH[name]})

    def test_each_dropout_creates_same_span(self):
        expected_anchors = {
            "drop_early8": (0, 5),
            "drop_middle8": (7, 12),
            "drop_late8": (14, 19),
        }
        for name, schedule in SCHEDULES.items():
            dropped = DROPPED_REFRESH[name]
            left = max(i for i in schedule if i < dropped)
            right = min(i for i in schedule if i > dropped)
            self.assertEqual((left, right), expected_anchors[name])
            self.assertEqual(right-left, 5)


if __name__ == "__main__":
    unittest.main()
