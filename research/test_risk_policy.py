"""Focused tests distinguishing risk triggers from guard/spike triggers."""
import unittest
from research.adaptive_cache import CachePolicy, RefreshController


class RiskTests(unittest.TestCase):
    def test_risk_trigger_below_feature_spike_and_age_limit(self):
        c = RefreshController(CachePolicy(risk_threshold=0.5, initial_full_steps=1, max_age=5), 6)
        c.decide(0, 0, 0)
        self.assertFalse(c.decide(1, 0.2, 0)["refresh"])
        decision = c.decide(2, 0.4, 0.4)
        self.assertEqual(decision["reason"], "risk")
        self.assertEqual(decision["risk_after"], 0)

    def test_feature_changes_decision_at_same_noise_schedule(self):
        decisions = []
        for weight in (0, 2):
            c = RefreshController(CachePolicy(risk_threshold=0.5, feature_weight=weight,
                                             initial_full_steps=1, max_age=5), 5)
            c.decide(0, 0, 0)
            c.decide(1, 0.2, 0.1)
            decisions.append(c.decide(2, 0.4, 0.2)["refresh"])
        self.assertEqual(decisions, [False, True])


if __name__ == "__main__":
    unittest.main()
