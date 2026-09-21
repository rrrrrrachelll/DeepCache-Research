import unittest
from .mechanism_oracle import UpBlock1MechanismOracle


class ProbeActionTests(unittest.TestCase):
    def test_probe_actions(self):
        self.assertEqual(UpBlock1MechanismOracle.probe_action("activation_only"), (True, False))
        self.assertEqual(UpBlock1MechanismOracle.probe_action("suffix_only"), (False, True))
        self.assertEqual(UpBlock1MechanismOracle.probe_action("activation_plus_suffix"), (True, True))

    def test_unknown_probe_rejected(self):
        with self.assertRaises(ValueError):
            UpBlock1MechanismOracle.probe_action("unknown")


if __name__ == "__main__":
    unittest.main()
