import json
from pathlib import Path
import unittest

from research.results.held_out_final_20260917.run_final import (
    MODES, SCHEDULES, build_cases,
)


class FinalProtocolTests(unittest.TestCase):
    def test_frozen_modes_and_schedules(self):
        self.assertEqual(MODES, ("dpm20", "fixed3", "uniform9",
                                 "stage_aware9", "drop_middle8"))
        self.assertEqual(SCHEDULES["drop_middle8"],
                         (0, 2, 5, 7, 12, 14, 17, 19))
        self.assertEqual(len(SCHEDULES["fixed3"]), 7)
        self.assertEqual(len(SCHEDULES["uniform9"]), 9)
        self.assertEqual(len(SCHEDULES["stage_aware9"]), 9)

    def test_only_held_out_cases_are_selected(self):
        prompts_path = (Path(__file__).resolve().parent.parent /
                        "oracle_20260911" / "oracle_prompts.json")
        prompts = json.loads(prompts_path.read_text())
        cases = build_cases(prompts)
        held_out_ids = {item["id"] for item in prompts["held_out_not_run"]}
        analysis_ids = {item["id"] for item in prompts["analysis"]}
        self.assertEqual(len(cases), 10)
        self.assertEqual({case[1] for case in cases}, held_out_ids)
        self.assertTrue(held_out_ids.isdisjoint(analysis_ids))


if __name__ == "__main__":
    unittest.main()
